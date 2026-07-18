#!/usr/bin/env python
import copy
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from scipy import ndimage
from torch.nn import functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import count_render
from scene import GaussianModel, Scene
from semantic_field_utils import (
    IdentitySemanticCodec,
    SemanticAutoencoder,
    collect_mask_features,
    inspect_mask_features,
    l2_normalize,
    load_geometry_checkpoint,
    sample_segment_pixels,
    save_json,
    save_semantic_codec,
)
from utils.general_utils import safe_state


def train_codec(features, semantic_dim, hidden_dims, epochs, batch_size, learning_rate, device, seed):
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    validation_size = max(1, int(0.05 * features.shape[0]))
    validation = features[permutation[:validation_size]]
    training = features[permutation[validation_size:]]
    model = SemanticAutoencoder(semantic_dim=semantic_dim, hidden_dims=hidden_dims).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    best_loss = float("inf")
    best_state = None
    history = []

    for epoch in range(epochs):
        model.train()
        epoch_permutation = torch.randperm(training.shape[0], generator=generator)
        train_loss = 0.0
        train_count = 0
        for start in range(0, training.shape[0], batch_size):
            batch_indices = epoch_permutation[start : start + batch_size]
            batch = training[batch_indices].to(device, non_blocking=True)
            reconstruction = model(batch)
            cosine = 1.0 - F.cosine_similarity(reconstruction, batch, dim=-1).mean()
            mse = F.mse_loss(reconstruction, batch)
            loss = cosine + 0.1 * mse
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.detach()) * batch.shape[0]
            train_count += batch.shape[0]

        model.eval()
        validation_loss = 0.0
        validation_cosine = 0.0
        validation_count = 0
        with torch.no_grad():
            for start in range(0, validation.shape[0], batch_size):
                batch = validation[start : start + batch_size].to(device, non_blocking=True)
                reconstruction = model(batch)
                cosine_values = F.cosine_similarity(reconstruction, batch, dim=-1)
                cosine_loss = 1.0 - cosine_values.mean()
                loss = cosine_loss + 0.1 * F.mse_loss(reconstruction, batch)
                validation_loss += float(loss) * batch.shape[0]
                validation_cosine += float(cosine_values.sum())
                validation_count += batch.shape[0]
        validation_loss /= max(1, validation_count)
        validation_cosine /= max(1, validation_count)
        row = {
            "epoch": epoch + 1,
            "train_loss": train_loss / max(1, train_count),
            "validation_loss": validation_loss,
            "validation_cosine": validation_cosine,
        }
        history.append(row)
        print(json.dumps(row))
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("Semantic codec training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    return model, history


def encode_feature_table(codec, feature_path, device, batch_size=4096):
    features = torch.from_numpy(np.load(feature_path).astype(np.float32, copy=False))
    features = l2_normalize(features)
    latents = []
    with torch.no_grad():
        for start in range(0, features.shape[0], batch_size):
            latents.append(codec.encode(features[start : start + batch_size].to(device)).cpu())
    return torch.cat(latents, dim=0)


def aggregate_view_observations(
    point_ids,
    point_weights,
    segment_ids,
    feature_latents,
    return_pixel_indices=True,
):
    flat_ids = point_ids.reshape(-1)
    flat_weights = point_weights.reshape(-1)
    valid = (flat_ids >= 0) & (flat_weights > 0)
    valid_ids = flat_ids[valid].long()
    unique_ids, inverse = torch.unique(valid_ids, sorted=True, return_inverse=True)
    pixel_aggregate_indices = None
    if return_pixel_indices:
        pixel_aggregate_indices = torch.full_like(flat_ids, -1, dtype=torch.long)
        pixel_aggregate_indices[valid] = inverse
        pixel_aggregate_indices = pixel_aggregate_indices.reshape_as(point_ids)

    repeated_targets = feature_latents[segment_ids.long()].unsqueeze(1).expand(
        -1, point_ids.shape[1], -1
    ).reshape(-1, feature_latents.shape[1])[valid]
    valid_weights = flat_weights[valid].float()
    aggregate_weights = torch.zeros(unique_ids.shape[0], dtype=torch.float32, device=point_ids.device)
    aggregate_sums = torch.zeros(
        (unique_ids.shape[0], feature_latents.shape[1]),
        dtype=torch.float32,
        device=point_ids.device,
    )
    aggregate_weights.index_add_(0, inverse, valid_weights)
    aggregate_sums.index_add_(0, inverse, repeated_targets * valid_weights.unsqueeze(-1))
    return unique_ids, aggregate_weights, aggregate_sums, pixel_aggregate_indices


def aggregate_owned_view_observations(
    point_ids,
    point_weights,
    dominant_segment,
    feature_latents,
    return_pixel_indices=False,
):
    """Aggregate a signed-ownership view without expanding a PxKxD tensor."""
    if point_ids.shape != point_weights.shape or point_ids.ndim != 2:
        raise ValueError("Point IDs and weights must have matching [P, K] shapes")
    flat_ids = point_ids.reshape(-1)
    flat_weights = point_weights.reshape(-1).float()
    valid = (flat_ids >= 0) & (flat_weights > 0.0)
    valid_ids = flat_ids[valid].long()
    unique_ids, inverse = torch.unique(valid_ids, sorted=True, return_inverse=True)
    aggregate_weights = torch.zeros(
        unique_ids.shape[0], dtype=torch.float32, device=point_ids.device
    )
    aggregate_weights.index_add_(0, inverse, flat_weights[valid])
    winning_segments = dominant_segment[unique_ids]
    if (winning_segments < 0).any() or (
        winning_segments >= feature_latents.shape[0]
    ).any():
        raise ValueError("Owned Gaussian references an invalid segment feature")
    aggregate_sums = (
        feature_latents[winning_segments.long()].float()
        * aggregate_weights.unsqueeze(-1)
    )
    pixel_aggregate_indices = None
    if return_pixel_indices:
        pixel_aggregate_indices = torch.full_like(flat_ids, -1, dtype=torch.long)
        pixel_aggregate_indices[valid] = inverse
        pixel_aggregate_indices = pixel_aggregate_indices.reshape_as(point_ids)
    return unique_ids, aggregate_weights, aggregate_sums, pixel_aggregate_indices


def accumulate_consensus_chunk(
    total_sums,
    total_weights,
    point_ids,
    point_weights,
    segment_ids,
    feature_latents,
):
    """Accumulate one pixel chunk without materializing a full-view PxKxD tensor."""
    valid = (point_ids >= 0) & (point_weights > 0)
    if not valid.any():
        return
    pixels, topk = point_ids.shape
    segment_features = feature_latents[segment_ids.long()].float()
    flat_valid = valid.reshape(-1)
    flat_ids = point_ids.reshape(-1)[flat_valid].long()
    flat_weights = point_weights.reshape(-1)[flat_valid].float()
    repeated_features = segment_features.repeat_interleave(topk, dim=0)[flat_valid]
    total_weights.index_add_(0, flat_ids, flat_weights)
    total_sums.index_add_(0, flat_ids, repeated_features * flat_weights.unsqueeze(-1))


def visibility_truncate_weights(
    point_ids,
    point_weights,
    mass_fraction=1.0,
    relative_floor=0.0,
    min_contributors=1,
):
    """Drop low-contribution ray tails while preserving raw rendering evidence."""
    if not 0.0 < mass_fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0, 1]")
    if not 0.0 <= relative_floor < 1.0:
        raise ValueError("relative_floor must be in [0, 1)")
    if min_contributors <= 0:
        raise ValueError("min_contributors must be positive")

    valid = (point_ids >= 0) & (point_weights > 0.0)
    weights = torch.where(
        valid,
        point_weights.float().clamp_min(0.0),
        torch.zeros_like(point_weights.float()),
    )
    sorted_weights, order = weights.sort(dim=1, descending=True)
    sorted_valid = torch.gather(valid, 1, order)
    totals = sorted_weights.sum(dim=1, keepdim=True)
    cumulative_before = sorted_weights.cumsum(dim=1) - sorted_weights
    mass_keep = cumulative_before < mass_fraction * totals
    relative_keep = sorted_weights >= relative_floor * sorted_weights[:, :1]
    ranks = torch.arange(weights.shape[1], device=weights.device).unsqueeze(0)
    minimum_keep = ranks < min_contributors
    keep_sorted = sorted_valid & ((mass_keep & relative_keep) | minimum_keep)

    keep = torch.zeros_like(keep_sorted)
    keep.scatter_(1, order, keep_sorted)
    truncated = torch.where(keep, weights, torch.zeros_like(weights))
    retained_mass = truncated.sum(dim=1) / totals.squeeze(1).clamp_min(1e-8)
    return truncated, retained_mass, keep.sum(dim=1)


def signed_segment_ownership(
    point_ids,
    point_weights,
    segment_ids,
    num_gaussians,
):
    """Estimate per-view Gaussian ownership by competition between 2D segments."""
    if point_ids.ndim != 2 or point_weights.shape != point_ids.shape:
        raise ValueError("Point IDs and weights must have matching [P, K] shapes")
    if segment_ids.shape != (point_ids.shape[0],):
        raise ValueError("Segment IDs must have one entry per pixel")
    if num_gaussians <= 0:
        raise ValueError("num_gaussians must be positive")

    device = point_weights.device
    total_mass = torch.zeros(num_gaussians, dtype=torch.float32, device=device)
    dominant_mass = torch.zeros_like(total_mass)
    dominant_segment = torch.full(
        (num_gaussians,), -1, dtype=torch.long, device=device
    )
    valid_pixels = segment_ids >= 0
    if not valid_pixels.any():
        return dominant_segment, dominant_mass, dominant_mass, total_mass

    sorted_pixels = torch.argsort(segment_ids)
    sorted_segments = segment_ids[sorted_pixels]
    valid_start = int(torch.searchsorted(sorted_segments, 0).item())
    sorted_pixels = sorted_pixels[valid_start:]
    sorted_segments = sorted_segments[valid_start:]
    unique_segments, counts = torch.unique_consecutive(
        sorted_segments, return_counts=True
    )
    offsets = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=device),
            counts.cumsum(dim=0),
        )
    )

    for segment_index, segment in enumerate(unique_segments):
        start = int(offsets[segment_index].item())
        end = int(offsets[segment_index + 1].item())
        rows = sorted_pixels[start:end]
        ids = point_ids[rows].reshape(-1)
        weights = point_weights[rows].reshape(-1).float()
        valid = (ids >= 0) & (ids < num_gaussians) & (weights > 0.0)
        if not valid.any():
            continue
        ids = ids[valid].long()
        weights = weights[valid]
        unique_ids, inverse = torch.unique(ids, sorted=True, return_inverse=True)
        segment_mass = torch.zeros(
            unique_ids.shape[0], dtype=torch.float32, device=device
        )
        segment_mass.index_add_(0, inverse, weights)
        total_mass.index_add_(0, unique_ids, segment_mass)
        better = segment_mass > dominant_mass[unique_ids]
        if better.any():
            winning_ids = unique_ids[better]
            dominant_mass[winning_ids] = segment_mass[better]
            dominant_segment[winning_ids] = segment.long()

    signed_confidence = (
        (2.0 * dominant_mass - total_mass) / total_mass.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    signed_confidence[total_mass <= 0.0] = 0.0
    return dominant_segment, signed_confidence, dominant_mass, total_mass


def _distribution_kl(distribution, reference):
    valid = distribution > 0.0
    return (
        distribution[valid]
        * (
            distribution[valid].log()
            - reference[valid].clamp_min(1e-20).log()
        )
    ).sum()


def kl_constrained_importance_ratios(
    behavior,
    utility,
    temperature=1.0,
    max_kl=0.02,
    ratio_clip=5.0,
):
    """Exponentially tilt a distribution inside a KL trust region."""
    if behavior.ndim != 1 or utility.shape != behavior.shape:
        raise ValueError("Behavior and utility must have matching vector shapes")
    if temperature <= 0.0 or max_kl < 0.0 or ratio_clip < 1.0:
        raise ValueError("Temperature/KL/ratio constraints are invalid")
    valid = behavior > 0.0
    if not valid.any():
        return torch.ones_like(behavior), torch.zeros((), device=behavior.device)
    behavior = behavior.float().clamp_min(0.0)
    behavior = behavior / behavior.sum().clamp_min(1e-20)
    logits = utility.float() / temperature
    logits = logits - logits[valid].max()
    target = behavior * torch.exp(logits.clamp(-30.0, 30.0))
    target = target / target.sum().clamp_min(1e-20)

    target_kl = _distribution_kl(target, behavior)
    if max_kl == 0.0:
        target = behavior
    elif float(target_kl) > max_kl:
        low, high = 0.0, 1.0
        for _ in range(32):
            amount = 0.5 * (low + high)
            candidate = (1.0 - amount) * behavior + amount * target
            if float(_distribution_kl(candidate, behavior)) <= max_kl:
                low = amount
            else:
                high = amount
        target = (1.0 - low) * behavior + low * target

    ratios = torch.ones_like(behavior)
    ratios[valid] = target[valid] / behavior[valid].clamp_min(1e-20)
    maximum_ratio = float(ratios[valid].max())
    if maximum_ratio > ratio_clip:
        amount = (ratio_clip - 1.0) / max(maximum_ratio - 1.0, 1e-20)
        target = (1.0 - amount) * behavior + amount * target
        ratios[valid] = target[valid] / behavior[valid].clamp_min(1e-20)
    return ratios, _distribution_kl(target, behavior)


def segment_view_importance(
    gaussian_segment_ids,
    gaussian_masses,
    segment_features,
    reference_features,
    reference_split_weights,
    reference_total_weights,
    temperature=1.0,
    max_kl=0.02,
    ratio_clip=5.0,
    information_weight=0.0,
):
    """Score one view's segments using opposite-split agreement and support gain."""
    count = gaussian_segment_ids.numel()
    expected_vectors = (count, segment_features.shape[1])
    if gaussian_masses.shape != (count,):
        raise ValueError("Gaussian masses must match segment assignments")
    if reference_features.shape != expected_vectors:
        raise ValueError("Reference features must match Gaussian observations")
    if reference_split_weights.shape != (count,) or reference_total_weights.shape != (
        count,
    ):
        raise ValueError("Reference weights must match Gaussian observations")
    if information_weight < 0.0:
        raise ValueError("Information weight must be non-negative")

    num_segments = int(segment_features.shape[0])
    ratios = torch.ones(num_segments, dtype=torch.float32, device=segment_features.device)
    valid = (
        (gaussian_segment_ids >= 0)
        & (gaussian_segment_ids < num_segments)
        & (gaussian_masses > 0.0)
    )
    if not valid.any():
        return ratios, {
            "kl": 0.0,
            "effective_segments": 0.0,
            "max_ratio": 1.0,
            "behavior_entropy": 0.0,
            "target_entropy": 0.0,
            "mean_agreement": 1.0,
            "mean_split_reliability": 0.0,
            "mean_information_gain": 0.0,
            "total_mass": 0.0,
        }

    ids = gaussian_segment_ids[valid].long()
    masses = gaussian_masses[valid].float()
    references = F.normalize(reference_features[valid].float(), dim=-1)
    split_weights = reference_split_weights[valid].float().clamp_min(0.0)
    total_weights = reference_total_weights[valid].float().clamp_min(0.0)
    segment_mass = torch.zeros(num_segments, dtype=torch.float32, device=ids.device)
    segment_mass.index_add_(0, ids, masses)
    active = segment_mass > 0.0
    behavior = segment_mass / segment_mass.sum().clamp_min(1e-20)

    observations = F.normalize(segment_features[ids].float(), dim=-1)
    reference_valid = (split_weights > 0.0) & (references.norm(dim=-1) > 0.0)
    agreement_values = F.cosine_similarity(observations, references, dim=-1).clamp(
        0.0, 1.0
    )
    reliable_mass = masses * reference_valid.float()
    segment_reliable_mass = torch.zeros_like(segment_mass)
    segment_agreement_sum = torch.zeros_like(segment_mass)
    segment_reliable_mass.index_add_(0, ids, reliable_mass)
    segment_agreement_sum.index_add_(0, ids, reliable_mass * agreement_values)
    split_reliability = segment_reliable_mass / segment_mass.clamp_min(1e-20)
    observed_agreement = segment_agreement_sum / segment_reliable_mass.clamp_min(1e-20)
    agreement = (
        split_reliability * observed_agreement + (1.0 - split_reliability)
    ).clamp(1e-4, 1.0)

    prior_without_view = (total_weights - masses).clamp_min(1e-6)
    information_values = torch.log1p(masses / prior_without_view)
    segment_information_sum = torch.zeros_like(segment_mass)
    segment_information_sum.index_add_(0, ids, masses * information_values)
    information = segment_information_sum / segment_mass.clamp_min(1e-20)
    if active.any():
        information = information / information[active].max().clamp_min(1e-8)
    utility = agreement.log() + (
        information_weight * split_reliability * agreement * information
    )
    ratios, kl = kl_constrained_importance_ratios(
        behavior,
        utility,
        temperature=temperature,
        max_kl=max_kl,
        ratio_clip=ratio_clip,
    )
    target = behavior * ratios
    target = target / target.sum().clamp_min(1e-20)
    behavior_entropy = -(behavior[active] * behavior[active].log()).sum()
    target_entropy = -(target[active] * target[active].clamp_min(1e-20).log()).sum()
    return ratios, {
        "kl": float(kl),
        "effective_segments": float(1.0 / target.square().sum().clamp_min(1e-20)),
        "max_ratio": float(ratios[active].max()),
        "behavior_entropy": float(behavior_entropy),
        "target_entropy": float(target_entropy),
        "mean_agreement": float((behavior * agreement).sum()),
        "mean_split_reliability": float((behavior * split_reliability).sum()),
        "mean_information_gain": float((behavior * information).sum()),
        "total_mass": float(segment_mass.sum()),
    }


def apply_signed_segment_ownership(
    point_ids,
    point_weights,
    segment_ids,
    dominant_segment,
    signed_confidence,
):
    """Keep only a Gaussian's majority segment and scale by its signed margin."""
    safe_ids = point_ids.clamp_min(0).long()
    valid = (point_ids >= 0) & (safe_ids < dominant_segment.shape[0])
    safe_ids = safe_ids.clamp_max(dominant_segment.shape[0] - 1)
    selected = valid & (
        dominant_segment[safe_ids] == segment_ids.long().unsqueeze(1)
    )
    gates = torch.where(
        selected,
        signed_confidence[safe_ids],
        torch.zeros_like(point_weights, dtype=torch.float32),
    )
    return point_weights.float() * gates


def mask_interior_confidence(segmentation, distance_pixels, boundary_floor):
    """Return a soft mask-interior confidence without using semantic labels."""
    boundary = np.zeros(segmentation.shape, dtype=bool)
    vertical = segmentation[1:] != segmentation[:-1]
    horizontal = segmentation[:, 1:] != segmentation[:, :-1]
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    distance = ndimage.distance_transform_edt(~boundary)
    confidence = np.clip(distance / distance_pixels, 0.0, 1.0)
    return (boundary_floor + (1.0 - boundary_floor) * confidence).astype(np.float32)


def surface_responsibility_weights(
    point_ids,
    point_weights,
    gaussian_depths,
    pixel_confidence,
    front_weight_ratio=0.05,
    depth_relative_scale=0.01,
    max_kl=0.02,
    ratio_clip=5.0,
    projection_steps=10,
):
    """Reweight ray contributors toward the first supported surface under a KL bound."""
    valid = (point_ids >= 0) & (point_weights > 0.0)
    weights = torch.where(valid, point_weights.float(), torch.zeros_like(point_weights.float()))
    totals = weights.sum(dim=1, keepdim=True)
    supported = totals.squeeze(1) > 1e-8
    if not supported.any():
        zeros = torch.zeros(point_ids.shape[0], dtype=torch.float32, device=point_ids.device)
        return weights, zeros, zeros

    safe_ids = point_ids.clamp_min(0).long()
    depths = gaussian_depths[safe_ids]
    behavior = weights / totals.clamp_min(1e-8)
    maximum = weights.max(dim=1, keepdim=True).values
    front_candidates = valid & (weights >= front_weight_ratio * maximum)
    positive_infinity = torch.full_like(depths, float("inf"))
    front_depth = torch.where(front_candidates, depths, positive_infinity).min(
        dim=1, keepdim=True
    ).values
    depth_scale = (front_depth.abs() * depth_relative_scale).clamp_min(1e-4)
    behind = torch.relu(depths - front_depth)
    surface_gate = torch.exp(-behind / depth_scale)
    surface_gate = torch.where(valid, surface_gate, torch.zeros_like(surface_gate))
    proposal = behavior * surface_gate
    proposal = proposal / proposal.sum(dim=1, keepdim=True).clamp_min(1e-8)
    proposal_ratio = proposal / behavior.clamp_min(1e-12)

    def row_kl(distribution):
        terms = distribution * (
            distribution.clamp_min(1e-12).log()
            - behavior.clamp_min(1e-12).log()
        )
        return torch.where(valid, terms, torch.zeros_like(terms)).sum(dim=1)

    proposal_kl = row_kl(proposal)
    low = torch.zeros_like(proposal_kl)
    maximum_proposal_ratio = torch.where(
        valid, proposal_ratio, torch.zeros_like(proposal_ratio)
    ).max(dim=1).values
    ratio_alpha = torch.where(
        maximum_proposal_ratio > ratio_clip,
        (ratio_clip - 1.0) / (maximum_proposal_ratio - 1.0).clamp_min(1e-8),
        torch.ones_like(maximum_proposal_ratio),
    ).clamp(0.0, 1.0)
    high = ratio_alpha
    ratio_limited = behavior + high[:, None] * (proposal - behavior)
    constrained = row_kl(ratio_limited) > max_kl
    for _ in range(projection_steps):
        middle = 0.5 * (low + high)
        mixed = behavior + middle[:, None] * (proposal - behavior)
        middle_kl = row_kl(mixed)
        accepted = middle_kl <= max_kl
        low = torch.where(accepted, middle, low)
        high = torch.where(accepted, high, middle)
    alpha = torch.where(constrained, low, ratio_alpha)
    target = behavior + alpha[:, None] * (proposal - behavior)
    final_kl = row_kl(target)
    final_ratio = target / behavior.clamp_min(1e-12)
    final_ratio = torch.where(valid, final_ratio, torch.zeros_like(final_ratio))
    adjusted = target * totals * pixel_confidence[:, None].clamp(0.0, 1.0)
    return adjusted, final_kl, final_ratio.max(dim=1).values


def main():
    parser = ArgumentParser(description="Train a semantic codec and cache visibility-weighted 2D observations.")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--feature_dir", default=None)
    parser.add_argument("--semantic_dim", type=int, default=16)
    parser.add_argument(
        "--identity_codec",
        action="store_true",
        help="Keep the original normalized 512D feature space; no learned bottleneck.",
    )
    parser.add_argument("--codec_hidden_dims", nargs="+", type=int, default=[256, 64])
    parser.add_argument("--codec_epochs", type=int, default=20)
    parser.add_argument("--codec_batch_size", type=int, default=1024)
    parser.add_argument("--codec_lr", type=float, default=3e-4)
    parser.add_argument("--min_codec_validation_cosine", type=float, default=0.0)
    parser.add_argument("--codec_only", action="store_true")
    parser.add_argument("--max_codec_features", type=int, default=200000)
    parser.add_argument("--max_pixels_per_view", type=int, default=32768)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--view_stride", type=int, default=1)
    parser.add_argument("--view_offset", type=int, default=0)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument(
        "--raw_contribution_weights",
        action="store_true",
        help="Preserve absolute render confidence when aggregating observations across views.",
    )
    parser.add_argument(
        "--consensus_only",
        action="store_true",
        help="Accumulate consensus without saving per-view pixel caches; suitable for full-view voting initialization.",
    )
    parser.add_argument(
        "--compact_view_cache",
        action="store_true",
        help="Save only per-view Gaussian aggregates, omitting pixel-level cache fields.",
    )
    parser.add_argument(
        "--view_cache_reference",
        default=None,
        help="Optional training-only consensus used to retain discordant view aggregates.",
    )
    parser.add_argument(
        "--view_cache_deviation_cosine_max",
        type=float,
        default=1.0,
    )
    parser.add_argument("--consensus_chunk_pixels", type=int, default=1024)
    parser.add_argument(
        "--compact_consensus",
        action="store_true",
        help="Omit dense raw sums and retain the mean norm needed to merge view shards.",
    )
    parser.add_argument(
        "--consensus_splits",
        type=int,
        default=1,
        help="Optionally retain interleaved view-split consensus features for reliability estimation.",
    )
    parser.add_argument("--surface_responsibility", action="store_true")
    parser.add_argument("--surface_front_weight_ratio", type=float, default=0.05)
    parser.add_argument("--surface_depth_relative_scale", type=float, default=0.01)
    parser.add_argument("--surface_boundary_distance", type=float, default=8.0)
    parser.add_argument("--surface_boundary_floor", type=float, default=0.5)
    parser.add_argument("--surface_max_kl", type=float, default=0.02)
    parser.add_argument("--surface_ratio_clip", type=float, default=5.0)
    parser.add_argument("--visibility_mass_fraction", type=float, default=1.0)
    parser.add_argument("--visibility_relative_floor", type=float, default=0.0)
    parser.add_argument("--visibility_min_contributors", type=int, default=1)
    parser.add_argument(
        "--signed_segment_ownership",
        action="store_true",
        help=(
            "Within each view, let 2D segments compete for every Gaussian and retain "
            "only positive foreground-vs-background contribution margin."
        ),
    )
    parser.add_argument(
        "--segment_view_importance_reference",
        default=None,
        help=(
            "Optional signed-ownership split consensus used for opposite-split "
            "segment-view reliability and KL-constrained aggregation."
        ),
    )
    parser.add_argument("--segment_importance_temperature", type=float, default=1.0)
    parser.add_argument("--segment_importance_max_kl", type=float, default=0.02)
    parser.add_argument("--segment_importance_ratio_clip", type=float, default=5.0)
    parser.add_argument("--segment_information_weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.semantic_dim <= 0:
        raise ValueError("--semantic_dim must be positive")
    if args.identity_codec and args.semantic_dim != 512:
        raise ValueError("--identity_codec requires --semantic_dim 512")
    if args.topk <= 0 or args.topk > 100:
        raise ValueError("--topk must be in [1, 100]")
    if (
        args.max_pixels_per_view < 0
        or args.consensus_chunk_pixels <= 0
        or args.consensus_splits <= 0
    ):
        raise ValueError("Pixel budgets and consensus chunk size must be non-negative/positive")
    if args.view_stride <= 0 or not 0 <= args.view_offset < args.view_stride:
        raise ValueError("view offset must be in [0, view_stride)")
    if args.compact_consensus and not args.consensus_only:
        raise ValueError("--compact_consensus requires --consensus_only")
    if args.view_cache_reference and args.consensus_only:
        raise ValueError("View-cache prefiltering requires per-view cache mode")
    if args.view_cache_reference and not args.compact_view_cache:
        raise ValueError("View-cache prefiltering requires --compact_view_cache")
    if not -1.0 <= args.view_cache_deviation_cosine_max <= 1.0:
        raise ValueError("View-cache deviation cosine must be in [-1, 1]")
    if args.consensus_splits > 1 and not args.consensus_only:
        raise ValueError("--consensus_splits > 1 requires --consensus_only")
    if args.surface_responsibility and not args.consensus_only:
        raise ValueError("--surface_responsibility requires --consensus_only")
    if not 0.0 < args.surface_front_weight_ratio <= 1.0:
        raise ValueError("surface front weight ratio must be in (0, 1]")
    if args.surface_depth_relative_scale <= 0.0 or args.surface_boundary_distance <= 0.0:
        raise ValueError("surface depth scale and boundary distance must be positive")
    if not 0.0 <= args.surface_boundary_floor <= 1.0:
        raise ValueError("surface boundary floor must be in [0, 1]")
    if args.surface_max_kl < 0.0 or args.surface_ratio_clip < 1.0:
        raise ValueError("surface KL must be non-negative and ratio clip at least one")
    if not 0.0 < args.visibility_mass_fraction <= 1.0:
        raise ValueError("visibility mass fraction must be in (0, 1]")
    if not 0.0 <= args.visibility_relative_floor < 1.0:
        raise ValueError("visibility relative floor must be in [0, 1)")
    if args.visibility_min_contributors <= 0:
        raise ValueError("visibility minimum contributors must be positive")
    visibility_truncation = (
        args.visibility_mass_fraction < 1.0
        or args.visibility_relative_floor > 0.0
        or args.visibility_min_contributors > 1
    )
    if args.surface_responsibility and visibility_truncation:
        raise ValueError("Surface responsibility and visibility truncation are separate probes")
    if args.signed_segment_ownership and (
        args.surface_responsibility or visibility_truncation
    ):
        raise ValueError(
            "Signed ownership, surface responsibility, and visibility truncation are separate probes"
        )
    if args.segment_view_importance_reference:
        if not args.signed_segment_ownership or args.consensus_splits != 2:
            raise ValueError(
                "Segment-view importance requires signed ownership and two consensus splits"
            )
        if not args.raw_contribution_weights:
            raise ValueError("Segment-view importance requires raw contribution weights")
        if args.segment_importance_temperature <= 0.0:
            raise ValueError("Segment importance temperature must be positive")
        if args.segment_importance_max_kl < 0.0:
            raise ValueError("Segment importance KL must be non-negative")
        if args.segment_importance_ratio_clip < 1.0:
            raise ValueError("Segment importance ratio clip must be at least one")
        if args.segment_information_weight < 0.0:
            raise ValueError("Segment information weight must be non-negative")
    safe_state(args.quiet)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    output_dir = os.path.abspath(dataset.model_path)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        print(f"Reuse existing semantic observation cache: {manifest_path}")
        return
    os.makedirs(os.path.join(output_dir, "views"), exist_ok=True)

    feature_dir = os.path.abspath(args.feature_dir or os.path.join(dataset.source_path, "language_features"))
    codec_features = None
    if args.identity_codec:
        codec_feature_count, codec_feature_paths = inspect_mask_features(feature_dir)
        codec = IdentitySemanticCodec().cuda()
        codec_history = [
            {
                "epoch": 0,
                "train_loss": 0.0,
                "validation_loss": 0.0,
                "validation_cosine": 1.0,
            }
        ]
    else:
        codec_features, codec_feature_paths = collect_mask_features(
            feature_dir,
            max_features=args.max_codec_features,
            seed=args.seed,
        )
        codec_feature_count = int(codec_features.shape[0])
        codec, codec_history = train_codec(
            codec_features,
            semantic_dim=args.semantic_dim,
            hidden_dims=args.codec_hidden_dims,
            epochs=args.codec_epochs,
            batch_size=args.codec_batch_size,
            learning_rate=args.codec_lr,
            device="cuda",
            seed=args.seed,
        )
    codec_path = os.path.join(output_dir, "semantic_codec.pt")
    save_semantic_codec(
        codec_path,
        codec,
        metadata={
            "feature_dir": feature_dir,
            "num_features": codec_feature_count,
            "history": codec_history,
        },
    )
    best_codec_cosine = max(row["validation_cosine"] for row in codec_history)
    if best_codec_cosine < args.min_codec_validation_cosine:
        raise RuntimeError(
            f"Codec validation cosine {best_codec_cosine:.6f} is below the required "
            f"minimum {args.min_codec_validation_cosine:.6f}"
        )
    if args.codec_only:
        print(
            json.dumps(
                {
                    "codec": codec_path,
                    "semantic_dim": args.semantic_dim,
                    "best_validation_cosine": best_codec_cosine,
                    "num_features": codec_feature_count,
                },
                indent=2,
            )
        )
        return
    if codec_features is not None:
        del codec_features

    importance_reference_features = None
    importance_reference_split_weights = None
    importance_reference_total_weights = None
    importance_reference_path = None
    if args.segment_view_importance_reference:
        importance_reference_path = os.path.abspath(
            args.segment_view_importance_reference
        )
        reference_payload = torch.load(importance_reference_path, map_location="cpu")
        required_reference = {
            "split_initial_features",
            "split_weights",
            "total_weights",
        }
        missing_reference = required_reference.difference(reference_payload)
        if missing_reference:
            raise ValueError(
                f"Segment-view reference is missing fields: {sorted(missing_reference)}"
            )
        importance_reference_features = reference_payload[
            "split_initial_features"
        ].detach().cpu().contiguous()
        importance_reference_split_weights = reference_payload[
            "split_weights"
        ].detach().cpu().float().contiguous()
        importance_reference_total_weights = reference_payload[
            "total_weights"
        ].detach().cpu().float().contiguous()
        del reference_payload

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(scene.gaussians, args.geometry_checkpoint)
    cameras = scene.getTrainCameras() + scene.getTestCameras()
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    indexed_cameras = [
        (index, camera)
        for index, camera in enumerate(cameras)
        if index % args.view_stride == args.view_offset
    ]
    if not indexed_cameras:
        raise ValueError("No cameras found for semantic observation caching")

    num_gaussians = int(scene.gaussians.get_xyz.shape[0])
    view_cache_reference_path = None
    view_cache_reference_features = None
    if args.view_cache_reference:
        view_cache_reference_path = os.path.abspath(args.view_cache_reference)
        reference_payload = torch.load(view_cache_reference_path, map_location="cpu")
        view_cache_reference_features = reference_payload.get("initial_features")
        if view_cache_reference_features is None or view_cache_reference_features.shape != (
            num_gaussians,
            args.semantic_dim,
        ):
            raise ValueError("View-cache reference features do not match the scene")
        view_cache_reference_features = (
            view_cache_reference_features.detach().cpu().contiguous()
        )
        del reference_payload
    if importance_reference_features is not None:
        if importance_reference_features.shape != (
            2,
            num_gaussians,
            args.semantic_dim,
        ):
            raise ValueError("Segment-view reference feature shape does not match the scene")
        if importance_reference_split_weights.shape != (2, num_gaussians):
            raise ValueError("Segment-view split weights do not match the scene")
        if importance_reference_total_weights.shape != (num_gaussians,):
            raise ValueError("Segment-view total weights do not match the scene")
    if args.consensus_splits == 1:
        split_sums = None
        split_weights = None
        total_sums = torch.zeros(
            (num_gaussians, args.semantic_dim), dtype=torch.float32, device="cuda"
        )
        total_weights = torch.zeros(num_gaussians, dtype=torch.float32, device="cuda")
    else:
        split_sums = torch.zeros(
            (args.consensus_splits, num_gaussians, args.semantic_dim),
            dtype=torch.float32,
            device="cuda",
        )
        split_weights = torch.zeros(
            (args.consensus_splits, num_gaussians), dtype=torch.float32, device="cuda"
        )
        total_sums = None
        total_weights = None
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    view_entries = []
    surface_kl_sum = 0.0
    surface_ratio_max = 0.0
    surface_pixel_count = 0
    visibility_retained_mass_sum = 0.0
    visibility_retained_contributors = 0
    visibility_input_contributors = 0
    visibility_pixel_count = 0
    ownership_gaussian_observations = 0
    ownership_positive_observations = 0
    ownership_confidence_sum = 0.0
    ownership_input_mass = 0.0
    ownership_retained_mass = 0.0
    importance_view_count = 0
    importance_mass_sum = 0.0
    importance_kl_sum = 0.0
    importance_effective_segments_sum = 0.0
    importance_ratio_max = 1.0
    importance_behavior_entropy_sum = 0.0
    importance_target_entropy_sum = 0.0
    importance_agreement_sum = 0.0
    importance_split_reliability_sum = 0.0
    importance_information_sum = 0.0
    view_cache_input_aggregates = 0
    view_cache_retained_aggregates = 0

    for view_index, camera in tqdm(indexed_cameras, desc="Caching semantic observations"):
        feature_stem = os.path.join(feature_dir, camera.image_name)
        feature_path = feature_stem + "_f.npy"
        segmentation_path = feature_stem + "_s.npy"
        if not os.path.isfile(feature_path) or not os.path.isfile(segmentation_path):
            raise ValueError(f"Missing semantic feature files for camera {camera.image_name}")
        segmentations = np.load(segmentation_path, mmap_mode="r")
        if not (0 <= dataset.feature_level < segmentations.shape[0]):
            raise ValueError(
                f"feature_level={dataset.feature_level} is invalid for {segmentation_path} with shape {segmentations.shape}"
            )
        segmentation = np.asarray(segmentations[dataset.feature_level])
        if segmentation.shape != (camera.image_height, camera.image_width):
            raise ValueError(
                f"Segmentation shape {segmentation.shape} does not match camera shape "
                f"{(camera.image_height, camera.image_width)} for {camera.image_name}"
            )
        sampled_flat = sample_segment_pixels(
            segmentation,
            max_pixels=args.max_pixels_per_view,
            seed=args.seed + view_index,
        )
        if sampled_flat.size == 0:
            continue
        sampled_y = torch.from_numpy(sampled_flat // camera.image_width).long().cuda()
        sampled_x = torch.from_numpy(sampled_flat % camera.image_width).long().cuda()
        segment_ids = torch.from_numpy(segmentation.reshape(-1)[sampled_flat].astype(np.int64)).cuda()
        if args.surface_responsibility:
            interior_confidence = mask_interior_confidence(
                segmentation,
                args.surface_boundary_distance,
                args.surface_boundary_floor,
            )
            sampled_confidence = torch.from_numpy(
                interior_confidence.reshape(-1)[sampled_flat]
            ).cuda()
            gaussian_depths = (
                scene.gaussians.get_xyz @ camera.world_view_transform[:3, 2]
                + camera.world_view_transform[3, 2]
            )
        else:
            sampled_confidence = None
            gaussian_depths = None
        feature_latents = encode_feature_table(codec, feature_path, "cuda").cuda()
        if int(segment_ids.max()) >= feature_latents.shape[0]:
            raise ValueError(f"Segmentation ids exceed feature table for {camera.image_name}")

        render_package = count_render(camera, scene.gaussians, pipe, background)
        if args.consensus_only:
            if split_sums is None:
                target_sums = total_sums
                target_weights = total_weights
            else:
                split_index = view_index % args.consensus_splits
                target_sums = split_sums[split_index]
                target_weights = split_weights[split_index]
            flat_ids = render_package["per_pixel_gaussian_ids"].reshape(-1, 100)
            flat_weights = render_package["per_pixel_gaussian_contributions"].reshape(-1, 100)
            ownership_top_ids = None
            ownership_top_weights = None
            dominant_segment = None
            signed_confidence = None
            dominant_mass = None
            segment_importance_ratios = None
            if args.signed_segment_ownership:
                sampled_indices = torch.from_numpy(sampled_flat).long().cuda()
                sampled_weights = flat_weights[sampled_indices]
                ownership_top_weights, ownership_top_indices = torch.topk(
                    sampled_weights, k=args.topk, dim=1
                )
                ownership_top_ids = torch.gather(
                    flat_ids[sampled_indices], 1, ownership_top_indices
                ).long()
                ownership_valid = ownership_top_ids >= 0
                ownership_top_weights = torch.where(
                    ownership_valid,
                    ownership_top_weights.float().clamp_min(0.0),
                    torch.zeros_like(ownership_top_weights.float()),
                )
                (
                    dominant_segment,
                    signed_confidence,
                    dominant_mass,
                    ownership_total_mass,
                ) = (
                    signed_segment_ownership(
                        ownership_top_ids,
                        ownership_top_weights,
                        segment_ids,
                        num_gaussians,
                    )
                )
                ownership_supported = ownership_total_mass > 0.0
                ownership_gaussian_observations += int(ownership_supported.sum().item())
                ownership_positive_observations += int(
                    (signed_confidence > 0.0).sum().item()
                )
                ownership_confidence_sum += float(
                    signed_confidence[ownership_supported].sum().item()
                )
                if importance_reference_features is not None:
                    importance_ids = torch.nonzero(
                        ownership_supported, as_tuple=False
                    ).squeeze(1)
                    importance_ids_cpu = importance_ids.detach().cpu()
                    opposite_split = 1 - (view_index % 2)
                    reference_features = importance_reference_features[
                        opposite_split, importance_ids_cpu
                    ].to("cuda", non_blocking=True)
                    reference_split_weights = importance_reference_split_weights[
                        opposite_split, importance_ids_cpu
                    ].to("cuda", non_blocking=True)
                    reference_total_weights = importance_reference_total_weights[
                        importance_ids_cpu
                    ].to("cuda", non_blocking=True)
                    segment_importance_ratios, importance_diagnostics = (
                        segment_view_importance(
                            dominant_segment[importance_ids],
                            dominant_mass[importance_ids]
                            * signed_confidence[importance_ids],
                            feature_latents,
                            reference_features,
                            reference_split_weights,
                            reference_total_weights,
                            temperature=args.segment_importance_temperature,
                            max_kl=args.segment_importance_max_kl,
                            ratio_clip=args.segment_importance_ratio_clip,
                            information_weight=args.segment_information_weight,
                        )
                    )
                    diagnostic_mass = importance_diagnostics["total_mass"]
                    importance_view_count += 1
                    importance_mass_sum += diagnostic_mass
                    importance_kl_sum += importance_diagnostics["kl"] * diagnostic_mass
                    importance_effective_segments_sum += (
                        importance_diagnostics["effective_segments"] * diagnostic_mass
                    )
                    importance_ratio_max = max(
                        importance_ratio_max,
                        importance_diagnostics["max_ratio"],
                    )
                    importance_behavior_entropy_sum += (
                        importance_diagnostics["behavior_entropy"] * diagnostic_mass
                    )
                    importance_target_entropy_sum += (
                        importance_diagnostics["target_entropy"] * diagnostic_mass
                    )
                    importance_agreement_sum += (
                        importance_diagnostics["mean_agreement"] * diagnostic_mass
                    )
                    importance_split_reliability_sum += (
                        importance_diagnostics["mean_split_reliability"]
                        * diagnostic_mass
                    )
                    importance_information_sum += (
                        importance_diagnostics["mean_information_gain"]
                        * diagnostic_mass
                    )
                    del (
                        importance_ids,
                        importance_ids_cpu,
                        reference_features,
                        reference_split_weights,
                        reference_total_weights,
                    )
                del sampled_indices, sampled_weights, ownership_top_indices
            for start in range(0, sampled_flat.size, args.consensus_chunk_pixels):
                stop = min(start + args.consensus_chunk_pixels, sampled_flat.size)
                if ownership_top_ids is not None:
                    top_ids = ownership_top_ids[start:stop]
                    top_weights = ownership_top_weights[start:stop]
                else:
                    flat_chunk = torch.from_numpy(sampled_flat[start:stop]).long().cuda()
                    chunk_ids = flat_ids[flat_chunk]
                    chunk_weights = flat_weights[flat_chunk]
                    top_weights, top_indices = torch.topk(
                        chunk_weights, k=args.topk, dim=1
                    )
                    top_ids = torch.gather(chunk_ids, 1, top_indices).long()
                    valid = top_ids >= 0
                    top_weights = torch.where(
                        valid,
                        top_weights.float().clamp_min(0.0),
                        torch.zeros_like(top_weights.float()),
                    )
                if args.signed_segment_ownership:
                    ownership_input_mass += float(top_weights.sum().item())
                    top_weights = apply_signed_segment_ownership(
                        top_ids,
                        top_weights,
                        segment_ids[start:stop],
                        dominant_segment,
                        signed_confidence,
                    )
                    ownership_retained_mass += float(top_weights.sum().item())
                if segment_importance_ratios is not None:
                    top_weights = top_weights * segment_importance_ratios[
                        segment_ids[start:stop].long()
                    ].unsqueeze(1)
                valid_pixels = top_weights.sum(dim=1) > 1e-8
                if not valid_pixels.any():
                    continue
                top_ids = top_ids[valid_pixels]
                top_weights = top_weights[valid_pixels]
                if visibility_truncation:
                    visibility_input_contributors += int((top_weights > 0.0).sum().item())
                    top_weights, retained_mass, retained_count = visibility_truncate_weights(
                        top_ids,
                        top_weights,
                        args.visibility_mass_fraction,
                        args.visibility_relative_floor,
                        args.visibility_min_contributors,
                    )
                    visibility_retained_mass_sum += float(retained_mass.sum().item())
                    visibility_retained_contributors += int(retained_count.sum().item())
                    visibility_pixel_count += int(retained_mass.numel())
                if args.surface_responsibility:
                    top_weights, chunk_kl, chunk_ratio = surface_responsibility_weights(
                        top_ids,
                        top_weights,
                        gaussian_depths,
                        sampled_confidence[start:stop][valid_pixels],
                        args.surface_front_weight_ratio,
                        args.surface_depth_relative_scale,
                        args.surface_max_kl,
                        args.surface_ratio_clip,
                    )
                    surface_kl_sum += float(chunk_kl.sum().item())
                    surface_ratio_max = max(
                        surface_ratio_max, float(chunk_ratio.max().item())
                    )
                    surface_pixel_count += int(chunk_kl.numel())
                if not args.raw_contribution_weights:
                    top_weights = top_weights / top_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
                chunk_segments = segment_ids[start:stop][valid_pixels]
                accumulate_consensus_chunk(
                    target_sums,
                    target_weights,
                    top_ids,
                    top_weights,
                    chunk_segments,
                    feature_latents,
                )
            del ownership_top_ids, ownership_top_weights
            if dominant_segment is not None:
                del (
                    dominant_segment,
                    signed_confidence,
                    dominant_mass,
                    ownership_total_mass,
                )
            if segment_importance_ratios is not None:
                del segment_importance_ratios
            del render_package
            torch.cuda.empty_cache()
            continue

        all_ids = render_package["per_pixel_gaussian_ids"][sampled_y, sampled_x]
        all_weights = render_package["per_pixel_gaussian_contributions"][sampled_y, sampled_x]
        top_weights, top_indices = torch.topk(all_weights, k=args.topk, dim=1)
        top_ids = torch.gather(all_ids, 1, top_indices).long()
        valid = top_ids >= 0
        top_weights = torch.where(valid, top_weights.float().clamp_min(0.0), torch.zeros_like(top_weights.float()))
        if args.signed_segment_ownership:
            (
                dominant_segment,
                signed_confidence,
                dominant_mass,
                ownership_total_mass,
            ) = signed_segment_ownership(
                top_ids,
                top_weights,
                segment_ids,
                num_gaussians,
            )
            ownership_supported = ownership_total_mass > 0.0
            ownership_gaussian_observations += int(ownership_supported.sum().item())
            ownership_positive_observations += int(
                (signed_confidence > 0.0).sum().item()
            )
            ownership_confidence_sum += float(
                signed_confidence[ownership_supported].sum().item()
            )
            ownership_input_mass += float(top_weights.sum().item())
            top_weights = apply_signed_segment_ownership(
                top_ids,
                top_weights,
                segment_ids,
                dominant_segment,
                signed_confidence,
            )
            ownership_retained_mass += float(top_weights.sum().item())
        weight_sums = top_weights.sum(dim=1, keepdim=True)
        valid_pixels = weight_sums.squeeze(1) > 1e-8
        if not valid_pixels.any():
            del render_package, all_ids, all_weights
            torch.cuda.empty_cache()
            continue
        top_ids = top_ids[valid_pixels]
        top_weights = top_weights[valid_pixels]
        if visibility_truncation:
            visibility_input_contributors += int((top_weights > 0.0).sum().item())
            top_weights, retained_mass, retained_count = visibility_truncate_weights(
                top_ids,
                top_weights,
                args.visibility_mass_fraction,
                args.visibility_relative_floor,
                args.visibility_min_contributors,
            )
            visibility_retained_mass_sum += float(retained_mass.sum().item())
            visibility_retained_contributors += int(retained_count.sum().item())
            visibility_pixel_count += int(retained_mass.numel())
        if not args.raw_contribution_weights:
            top_weights = top_weights / top_weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        segment_ids = segment_ids[valid_pixels]
        sampled_flat = sampled_flat[valid_pixels.detach().cpu().numpy()]

        if args.signed_segment_ownership:
            (
                aggregate_ids,
                aggregate_weights,
                aggregate_sums,
                pixel_aggregate_indices,
            ) = (
                aggregate_owned_view_observations(
                    top_ids,
                    top_weights,
                    dominant_segment,
                    feature_latents,
                    return_pixel_indices=not args.compact_view_cache,
                )
            )
        else:
            (
                aggregate_ids,
                aggregate_weights,
                aggregate_sums,
                pixel_aggregate_indices,
            ) = aggregate_view_observations(
                top_ids,
                top_weights,
                segment_ids,
                feature_latents,
                return_pixel_indices=not args.compact_view_cache,
            )
        total_weights.index_add_(0, aggregate_ids, aggregate_weights)
        total_sums.index_add_(0, aggregate_ids, aggregate_sums)

        view_cache_input_aggregates += int(aggregate_ids.numel())
        if view_cache_reference_features is not None:
            aggregate_ids_cpu = aggregate_ids.detach().cpu()
            reference_features = view_cache_reference_features[
                aggregate_ids_cpu
            ].to("cuda", dtype=torch.float32, non_blocking=True)
            observations = F.normalize(aggregate_sums.float(), dim=-1)
            reference_valid = reference_features.norm(dim=-1) > 0.0
            deviation_cosine = F.cosine_similarity(
                observations,
                F.normalize(reference_features, dim=-1),
                dim=-1,
            )
            retained = reference_valid & (
                deviation_cosine <= args.view_cache_deviation_cosine_max
            )
            aggregate_ids = aggregate_ids[retained]
            aggregate_weights = aggregate_weights[retained]
            aggregate_sums = aggregate_sums[retained]
            del (
                aggregate_ids_cpu,
                reference_features,
                observations,
                reference_valid,
                deviation_cosine,
                retained,
            )
        view_cache_retained_aggregates += int(aggregate_ids.numel())

        view_cache_path = os.path.join(output_dir, "views", f"{view_index:04d}_{camera.image_name}.pt")
        view_payload = {
            "view_index": view_index,
            "image_name": camera.image_name,
            "aggregate_ids": aggregate_ids.detach().cpu().to(torch.int32),
            "aggregate_weights": aggregate_weights.detach().cpu(),
            "aggregate_sums": aggregate_sums.detach().cpu().to(torch.float16),
            "image_height": int(camera.image_height),
            "image_width": int(camera.image_width),
        }
        if not args.compact_view_cache:
            view_payload.update(
                {
                    "point_ids": top_ids.detach().cpu().to(torch.int32),
                    "point_weights": top_weights.detach().cpu().to(torch.float16),
                    "segment_ids": segment_ids.detach().cpu().to(torch.int32),
                    "sampled_flat_indices": torch.from_numpy(sampled_flat).to(torch.int64),
                    "feature_latents": feature_latents.detach().cpu().to(torch.float16),
                    "pixel_aggregate_indices": pixel_aggregate_indices.detach()
                    .cpu()
                    .to(torch.int32),
                }
            )
        torch.save(view_payload, view_cache_path)
        view_entries.append(
            {
                "view_index": view_index,
                "image_name": camera.image_name,
                "cache": os.path.relpath(view_cache_path, output_dir),
                "num_pixels": int(top_ids.shape[0]),
                "num_observed_gaussians": int(aggregate_ids.shape[0]),
            }
        )
        del render_package, all_ids, all_weights, top_ids, top_weights
        torch.cuda.empty_cache()

    consensus_path = os.path.join(output_dir, "consensus.pt")
    if split_sums is None:
        support = total_weights > 0
        if args.compact_consensus:
            normalization_chunk = 8192
            initial_features_cpu = torch.empty(
                (num_gaussians, args.semantic_dim), dtype=torch.float16, device="cpu"
            )
            mean_feature_norm_cpu = torch.zeros(
                num_gaussians, dtype=torch.float16, device="cpu"
            )
            for start in tqdm(
                range(0, num_gaussians, normalization_chunk),
                desc="Normalizing compact consensus",
            ):
                end = min(start + normalization_chunk, num_gaussians)
                weight_chunk = total_weights[start:end]
                mean_chunk = total_sums[start:end] / weight_chunk.clamp_min(1e-8).unsqueeze(-1)
                norm_chunk = mean_chunk.norm(dim=-1)
                feature_chunk = l2_normalize(mean_chunk)
                feature_chunk[weight_chunk <= 0] = 0.0
                initial_features_cpu[start:end].copy_(feature_chunk.to(torch.float16).cpu())
                mean_feature_norm_cpu[start:end].copy_(norm_chunk.to(torch.float16).cpu())
            consensus_payload = {
                "total_weights": total_weights.detach().cpu(),
                "initial_features": initial_features_cpu,
                "mean_feature_norm": mean_feature_norm_cpu,
            }
        else:
            initial_features = torch.zeros_like(total_sums)
            initial_features[support] = l2_normalize(
                total_sums[support] / total_weights[support, None]
            )
            consensus_payload = {
                "total_sums": total_sums.detach().cpu(),
                "total_weights": total_weights.detach().cpu(),
                "initial_features": initial_features.detach().cpu().to(torch.float16),
            }
    else:
        # Split consensuses are diagnostic artifacts. Normalize in bounded chunks and
        # move each chunk to CPU immediately instead of materializing several full
        # [N, D] CUDA tensors during finalization.
        normalization_chunk = 8192
        total_weights_cuda = split_weights.sum(dim=0)
        support = total_weights_cuda > 0
        initial_features_cpu = torch.empty(
            (num_gaussians, args.semantic_dim), dtype=torch.float16, device="cpu"
        )
        split_initial_features_cpu = torch.empty(
            (args.consensus_splits, num_gaussians, args.semantic_dim),
            dtype=torch.float16,
            device="cpu",
        )
        for start in tqdm(
            range(0, num_gaussians, normalization_chunk),
            desc="Normalizing split consensuses",
        ):
            end = min(start + normalization_chunk, num_gaussians)
            weight_chunk = total_weights_cuda[start:end]
            summed_chunk = split_sums[:, start:end].sum(dim=0)
            feature_chunk = l2_normalize(
                summed_chunk / weight_chunk.clamp_min(1e-8).unsqueeze(-1)
            )
            initial_features_cpu[start:end].copy_(feature_chunk.to(torch.float16).cpu())
            for split_index in range(args.consensus_splits):
                split_weight_chunk = split_weights[split_index, start:end]
                split_feature_chunk = l2_normalize(
                    split_sums[split_index, start:end]
                    / split_weight_chunk.clamp_min(1e-8).unsqueeze(-1)
                )
                split_initial_features_cpu[split_index, start:end].copy_(
                    split_feature_chunk.to(torch.float16).cpu()
                )
        consensus_payload = {
            "total_weights": total_weights_cuda.detach().cpu(),
            "initial_features": initial_features_cpu,
            "split_initial_features": split_initial_features_cpu,
            "split_weights": split_weights.detach().cpu(),
        }
    torch.save(consensus_payload, consensus_path)

    manifest = {
        "format_version": 1,
        "source_path": dataset.source_path,
        "feature_dir": feature_dir,
        "feature_level": int(dataset.feature_level),
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "num_gaussians": num_gaussians,
        "semantic_dim": int(args.semantic_dim),
        "codec_type": "identity" if args.identity_codec else "autoencoder",
        "topk": int(args.topk),
        "raw_contribution_weights": bool(args.raw_contribution_weights),
        "consensus_only": bool(args.consensus_only),
        "consensus_chunk_pixels": int(args.consensus_chunk_pixels),
        "consensus_splits": int(args.consensus_splits),
        "consensus_has_total_sums": bool(split_sums is None and not args.compact_consensus),
        "surface_responsibility": bool(args.surface_responsibility),
        "surface_front_weight_ratio": float(args.surface_front_weight_ratio),
        "surface_depth_relative_scale": float(args.surface_depth_relative_scale),
        "surface_boundary_distance": float(args.surface_boundary_distance),
        "surface_boundary_floor": float(args.surface_boundary_floor),
        "surface_max_kl": float(args.surface_max_kl),
        "surface_ratio_clip": float(args.surface_ratio_clip),
        "surface_mean_kl": float(surface_kl_sum / max(1, surface_pixel_count)),
        "surface_observed_max_ratio": float(surface_ratio_max),
        "visibility_truncation": bool(visibility_truncation),
        "visibility_mass_fraction": float(args.visibility_mass_fraction),
        "visibility_relative_floor": float(args.visibility_relative_floor),
        "visibility_min_contributors": int(args.visibility_min_contributors),
        "visibility_mean_retained_mass": float(
            visibility_retained_mass_sum / max(1, visibility_pixel_count)
        ),
        "visibility_retained_contributor_fraction": float(
            visibility_retained_contributors / max(1, visibility_input_contributors)
        ),
        "signed_segment_ownership": bool(args.signed_segment_ownership),
        "ownership_positive_observation_fraction": float(
            ownership_positive_observations
            / max(1, ownership_gaussian_observations)
        ),
        "ownership_mean_signed_confidence": float(
            ownership_confidence_sum / max(1, ownership_gaussian_observations)
        ),
        "ownership_retained_mass_fraction": float(
            ownership_retained_mass / max(ownership_input_mass, 1e-12)
        ),
        "segment_view_importance_reference": importance_reference_path,
        "segment_importance_temperature": float(
            args.segment_importance_temperature
        ),
        "segment_importance_max_kl": float(args.segment_importance_max_kl),
        "segment_importance_ratio_clip": float(
            args.segment_importance_ratio_clip
        ),
        "segment_information_weight": float(args.segment_information_weight),
        "segment_importance_num_views": int(importance_view_count),
        "segment_importance_weighted_kl": float(
            importance_kl_sum / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_weighted_effective_segments": float(
            importance_effective_segments_sum / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_max_ratio": float(importance_ratio_max),
        "segment_importance_weighted_behavior_entropy": float(
            importance_behavior_entropy_sum / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_weighted_target_entropy": float(
            importance_target_entropy_sum / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_weighted_agreement": float(
            importance_agreement_sum / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_weighted_split_reliability": float(
            importance_split_reliability_sum
            / max(importance_mass_sum, 1e-12)
        ),
        "segment_importance_weighted_information_gain": float(
            importance_information_sum / max(importance_mass_sum, 1e-12)
        ),
        "max_pixels_per_view": int(args.max_pixels_per_view),
        "max_views": int(args.max_views),
        "view_stride": int(args.view_stride),
        "view_offset": int(args.view_offset),
        "num_selected_views": int(len(indexed_cameras)),
        "compact_consensus": bool(args.compact_consensus),
        "compact_view_cache": bool(args.compact_view_cache),
        "view_cache_reference": view_cache_reference_path,
        "view_cache_deviation_cosine_max": float(
            args.view_cache_deviation_cosine_max
        ),
        "view_cache_input_aggregates": int(view_cache_input_aggregates),
        "view_cache_retained_aggregates": int(view_cache_retained_aggregates),
        "view_cache_retained_fraction": float(
            view_cache_retained_aggregates / max(1, view_cache_input_aggregates)
        ),
        "codec": os.path.relpath(codec_path, output_dir),
        "consensus": os.path.relpath(consensus_path, output_dir),
        "num_supported_gaussians": int(support.sum().item()),
        "supported_fraction": float(support.float().mean().item()),
        "views": view_entries,
        "codec_feature_files": len(codec_feature_paths),
    }
    save_json(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    print(f"Saved semantic observation cache to {output_dir}")


if __name__ == "__main__":
    main()
