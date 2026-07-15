#!/usr/bin/env python
import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class SegmentViewSample:
    view_index: int
    group_index: int
    batch_indices: torch.Tensor
    importance_ratio: float
    clipped_ratio: float


def _kl_divergence(distribution, reference):
    valid = distribution > 0
    return float(
        (
            distribution[valid]
            * (distribution[valid].log() - reference[valid].clamp_min(1e-20).log())
        ).sum()
    )


def _renyi2_divergence(target, behavior):
    valid = target > 0
    second_moment = (
        target[valid].square() / behavior[valid].clamp_min(1e-20)
    ).sum()
    return float(second_moment.clamp_min(1.0).log())


def _mix_to_kl_limit(reference, candidate, max_kl):
    if max_kl <= 0 or _kl_divergence(candidate, reference) <= max_kl:
        return candidate
    low = 0.0
    high = 1.0
    for _ in range(32):
        amount = 0.5 * (low + high)
        mixed = (1.0 - amount) * reference + amount * candidate
        if _kl_divergence(mixed, reference) <= max_kl:
            low = amount
        else:
            high = amount
    return (1.0 - low) * reference + low * candidate


@torch.no_grad()
def _spherical_kmeans(
    features,
    num_groups,
    sample_count,
    iterations,
    assignment_chunk_size,
    device,
    seed,
):
    if features.shape[0] == 0:
        raise ValueError("Cannot cluster an empty Gaussian feature table")
    num_groups = min(int(num_groups), int(features.shape[0]))
    generator = torch.Generator().manual_seed(seed)
    sample_count = min(int(sample_count), int(features.shape[0]))
    sample_indices = torch.randperm(features.shape[0], generator=generator)[:sample_count]
    samples = F.normalize(features[sample_indices].float().to(device), dim=-1)

    first_index = int(torch.randint(sample_count, (1,), generator=generator))
    centers = [samples[first_index]]
    closest_distance = 1.0 - samples @ centers[0]
    for _ in range(1, num_groups):
        next_index = int(closest_distance.argmax())
        center = samples[next_index]
        centers.append(center)
        closest_distance = torch.minimum(closest_distance, 1.0 - samples @ center)
    centers = torch.stack(centers, dim=0)

    for _ in range(iterations):
        assignments = (samples @ centers.T).argmax(dim=1)
        sums = torch.zeros_like(centers)
        sums.index_add_(0, assignments, samples)
        counts = torch.bincount(assignments, minlength=num_groups)
        nonempty = counts > 0
        centers[nonempty] = F.normalize(sums[nonempty], dim=-1)

    all_assignments = []
    for start in range(0, features.shape[0], assignment_chunk_size):
        chunk = F.normalize(
            features[start : start + assignment_chunk_size].float().to(device),
            dim=-1,
        )
        all_assignments.append((chunk @ centers.T).argmax(dim=1).cpu())
    return centers.cpu(), torch.cat(all_assignments, dim=0)


class SegmentWiseViewSampler:
    """Importance sampler over Gaussian semantic groups and their contributing views."""

    def __init__(
        self,
        caches,
        gaussian_features,
        support_weights,
        num_groups=16,
        temperature=1.0,
        uniform_mix=0.25,
        max_step_kl=0.02,
        max_base_kl=0.5,
        update_interval=100,
        ema_decay=0.95,
        ratio_clip=5.0,
        rarity_weight=0.1,
        kmeans_samples=65536,
        kmeans_iterations=8,
        assignment_chunk_size=65536,
        seed=0,
    ):
        if num_groups <= 1:
            raise ValueError("num_groups must be greater than one")
        if not 0 <= uniform_mix <= 1:
            raise ValueError("uniform_mix must be in [0, 1]")
        if not 0 <= ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if update_interval <= 0 or ratio_clip <= 0:
            raise ValueError("update_interval and ratio_clip must be positive")

        self.caches = caches
        self.temperature = float(temperature)
        self.uniform_mix = float(uniform_mix)
        self.max_step_kl = float(max_step_kl)
        self.max_base_kl = float(max_base_kl)
        self.update_interval = int(update_interval)
        self.ema_decay = float(ema_decay)
        self.ratio_clip = float(ratio_clip)
        self.rarity_weight = float(rarity_weight)
        self.generator = torch.Generator().manual_seed(seed)
        self.device = gaussian_features.device

        dominant_ids = [cache["point_ids"][:, 0].long() for cache in caches]
        observed_ids = torch.unique(torch.cat(dominant_ids, dim=0), sorted=True)
        observed_features = gaussian_features[observed_ids.to(self.device)]
        centers, observed_groups = _spherical_kmeans(
            observed_features,
            num_groups=num_groups,
            sample_count=kmeans_samples,
            iterations=kmeans_iterations,
            assignment_chunk_size=assignment_chunk_size,
            device=self.device,
            seed=seed,
        )
        self.centers = centers
        self.num_groups = int(centers.shape[0])
        self.num_views = len(caches)
        self.gaussian_groups = torch.full(
            (gaussian_features.shape[0],),
            -1,
            dtype=torch.int16,
        )
        self.gaussian_groups[observed_ids] = observed_groups.to(torch.int16)

        self.pixel_indices = []
        counts = torch.zeros((self.num_groups, self.num_views), dtype=torch.float64)
        static_scores = torch.zeros_like(counts)
        support_weights = support_weights.to(self.device)
        for view_index, (cache, view_dominant_ids) in enumerate(zip(caches, dominant_ids)):
            pixel_groups = self.gaussian_groups[view_dominant_ids].long()
            view_buckets = []
            for group_index in range(self.num_groups):
                indices = torch.nonzero(pixel_groups == group_index, as_tuple=False).squeeze(1)
                view_buckets.append(indices)
                counts[group_index, view_index] = indices.numel()
            self.pixel_indices.append(view_buckets)

            view_dominant_ids_device = view_dominant_ids.to(self.device)
            targets = cache["feature_latents"][cache["segment_ids"].long()].float().to(self.device)
            predictions = F.normalize(gaussian_features[view_dominant_ids_device].float(), dim=-1)
            disagreement = 1.0 - F.cosine_similarity(predictions, targets, dim=-1)
            rarity = support_weights[view_dominant_ids_device].clamp_min(1.0).rsqrt()
            rarity = rarity / rarity.mean().clamp_min(1e-8)
            scores = disagreement + self.rarity_weight * rarity
            groups_device = pixel_groups.to(self.device)
            score_sums = torch.zeros(self.num_groups, device=self.device)
            score_sums.index_add_(0, groups_device, scores)
            group_counts = torch.bincount(groups_device, minlength=self.num_groups)
            valid_groups = group_counts > 0
            static_scores[valid_groups.cpu(), view_index] = (
                score_sums[valid_groups] / group_counts[valid_groups]
            ).cpu().double()

        view_sizes = torch.tensor(
            [cache["point_ids"].shape[0] for cache in caches],
            dtype=torch.float64,
        )
        base_joint = counts / view_sizes.unsqueeze(0).clamp_min(1.0)
        base_joint /= float(self.num_views)
        group_mass = base_joint.sum(dim=1)
        active_groups = group_mass > 0
        self.group_probabilities = group_mass / group_mass.sum()
        self.base_conditional = torch.zeros_like(base_joint)
        self.base_conditional[active_groups] = (
            base_joint[active_groups]
            / group_mass[active_groups].unsqueeze(1)
        )
        self.sampling_conditional = self.base_conditional.clone()
        self.scores = static_scores
        self.sample_counts = torch.zeros_like(counts, dtype=torch.long)
        self.update_count = 0
        self.ratio_count = 0
        self.ratio_sum = 0.0
        self.ratio_square_sum = 0.0
        self.ratio_min = float("inf")
        self.ratio_max = 0.0
        self.clipped_count = 0
        self._update_distributions()

    def _score_candidate(self, group_index):
        base = self.base_conditional[group_index]
        valid = base > 0
        weights = base[valid]
        scores = self.scores[group_index, valid]
        mean = (weights * scores).sum()
        variance = (weights * (scores - mean).square()).sum()
        standardized = (scores - mean) / variance.sqrt().clamp_min(1e-6)
        logits = (self.temperature * standardized).clamp(-12.0, 12.0)
        candidate_values = weights * logits.exp()
        candidate_values /= candidate_values.sum()
        candidate_values = (
            (1.0 - self.uniform_mix) * candidate_values
            + self.uniform_mix * weights
        )
        candidate = torch.zeros_like(base)
        candidate[valid] = candidate_values
        return candidate

    def _update_distributions(self):
        for group_index in range(self.num_groups):
            base = self.base_conditional[group_index]
            if not bool((base > 0).any()):
                continue
            old = self.sampling_conditional[group_index]
            candidate = self._score_candidate(group_index)
            candidate = _mix_to_kl_limit(base, candidate, self.max_base_kl)
            self.sampling_conditional[group_index] = _mix_to_kl_limit(
                old,
                candidate,
                self.max_step_kl,
            )
        self.update_count += 1

    def sample(self, batch_pixels):
        group_index = int(
            torch.multinomial(
                self.group_probabilities,
                1,
                generator=self.generator,
            )
        )
        view_distribution = self.sampling_conditional[group_index]
        view_index = int(
            torch.multinomial(view_distribution, 1, generator=self.generator)
        )
        candidates = self.pixel_indices[view_index][group_index]
        if candidates.numel() == 0:
            raise RuntimeError("Sampled an empty group-view pair")
        selected = torch.randint(
            candidates.numel(),
            (int(batch_pixels),),
            generator=self.generator,
        )
        batch_indices = candidates[selected]
        ratio = float(
            self.base_conditional[group_index, view_index]
            / view_distribution[view_index]
        )
        clipped_ratio = min(ratio, self.ratio_clip)
        self.sample_counts[group_index, view_index] += 1
        self.ratio_count += 1
        self.ratio_sum += ratio
        self.ratio_square_sum += ratio * ratio
        self.ratio_min = min(self.ratio_min, ratio)
        self.ratio_max = max(self.ratio_max, ratio)
        self.clipped_count += int(ratio > self.ratio_clip)
        return SegmentViewSample(
            view_index=view_index,
            group_index=group_index,
            batch_indices=batch_indices,
            importance_ratio=ratio,
            clipped_ratio=clipped_ratio,
        )

    def observe(self, group_index, view_index, priority):
        old = float(self.scores[group_index, view_index])
        self.scores[group_index, view_index] = (
            self.ema_decay * old + (1.0 - self.ema_decay) * float(priority)
        )

    def maybe_update(self, iteration):
        if iteration % self.update_interval == 0:
            self._update_distributions()

    def diagnostics(self):
        group_kl = []
        group_renyi2 = []
        for group_index in range(self.num_groups):
            base = self.base_conditional[group_index]
            if not bool((base > 0).any()):
                continue
            behavior = self.sampling_conditional[group_index]
            group_kl.append(_kl_divergence(behavior, base))
            group_renyi2.append(_renyi2_divergence(base, behavior))
        probabilities = self.group_probabilities[self.group_probabilities > 0]
        weights = probabilities / probabilities.sum()
        weighted_kl = sum(weight * value for weight, value in zip(weights, group_kl))
        weighted_renyi2 = sum(
            weight * value for weight, value in zip(weights, group_renyi2)
        )
        ratio_mean = self.ratio_sum / max(1, self.ratio_count)
        ratio_variance = (
            self.ratio_square_sum / max(1, self.ratio_count) - ratio_mean * ratio_mean
        )
        return {
            "num_groups": self.num_groups,
            "num_views": self.num_views,
            "num_active_pairs": int((self.base_conditional > 0).sum()),
            "update_count": self.update_count,
            "weighted_kl_behavior_to_base": float(weighted_kl),
            "max_group_kl_behavior_to_base": max(group_kl, default=0.0),
            "weighted_renyi2_target_to_behavior": float(weighted_renyi2),
            "min_group_ess_fraction": math.exp(-max(group_renyi2, default=0.0)),
            "importance_ratio_mean": ratio_mean,
            "importance_ratio_std": math.sqrt(max(0.0, ratio_variance)),
            "importance_ratio_min": self.ratio_min if self.ratio_count else 0.0,
            "importance_ratio_max": self.ratio_max,
            "clipped_fraction": self.clipped_count / max(1, self.ratio_count),
            "sampled_pair_fraction": float(
                ((self.sample_counts > 0) & (self.base_conditional > 0)).sum()
                / (self.base_conditional > 0).sum().clamp_min(1)
            ),
        }

    def state_dict(self):
        return {
            "cluster_centers": self.centers,
            "gaussian_groups": self.gaussian_groups,
            "group_probabilities": self.group_probabilities,
            "base_conditional": self.base_conditional,
            "sampling_conditional": self.sampling_conditional,
            "scores": self.scores,
            "sample_counts": self.sample_counts,
            "diagnostics": self.diagnostics(),
        }
