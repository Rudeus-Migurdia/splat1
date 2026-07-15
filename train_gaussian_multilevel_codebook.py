#!/usr/bin/env python
"""Train shared semantic codebooks directly from cached 2D observations."""

import json
import os
import random
import shutil
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from segment_view_sampler import SegmentWiseViewSampler
from semantic_field_utils import l2_normalize, load_json, save_json
from train_semantic_field import ViewNuisance, leave_one_view_out_target, load_view_caches
from utils.general_utils import safe_state


class MultilevelGaussianCodebook(nn.Module):
    def __init__(self, artifact_dir, device="cuda"):
        super().__init__()
        self.source_dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.source_dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        representation = self.manifest.get("representation")
        if representation not in {
            "gaussian_multilevel_residual_codebook",
            "gaussian_adaptive_shared_codebook",
        }:
            raise ValueError("Unsupported codebook initialization artifact")
        self.shared_codebook = representation == "gaussian_adaptive_shared_codebook"
        loaded_ids = np.load(
            os.path.join(self.source_dir, self.manifest["point_code_ids"])
        ).astype(np.int64)
        sparse_overflow = (
            self.shared_codebook
            and self.manifest.get("storage_layout") == "base_plus_sparse_overflow"
        )
        if sparse_overflow:
            num_gaussians = int(self.manifest["num_gaussians"])
            id_slots = int(self.manifest["id_slots"])
            if loaded_ids.shape != (num_gaussians,):
                raise ValueError("Sparse base IDs do not match the Gaussian count")
            packed_ids = np.full(
                (num_gaussians, id_slots),
                int(self.manifest["invalid_id"]),
                dtype=np.int64,
            )
            packed_weights = np.zeros((num_gaussians, id_slots), dtype=np.float32)
            packed_ids[:, 0] = loaded_ids
            packed_weights[loaded_ids != int(self.manifest["invalid_id"]), 0] = 1.0
            overflow_points = np.load(
                os.path.join(self.source_dir, self.manifest["overflow_point_ids"])
            ).astype(np.int64)
            overflow_slots = np.load(
                os.path.join(self.source_dir, self.manifest["overflow_slots"])
            ).astype(np.int64)
            overflow_ids = np.load(
                os.path.join(self.source_dir, self.manifest["overflow_code_ids"])
            ).astype(np.int64)
            overflow_weights = np.load(
                os.path.join(self.source_dir, self.manifest["overflow_weights"])
            ).astype(np.float32) / 255.0
            if not (
                overflow_points.shape
                == overflow_slots.shape
                == overflow_ids.shape
                == overflow_weights.shape
            ):
                raise ValueError("Sparse overflow arrays must have matching shapes")
            packed_ids[overflow_points, overflow_slots] = overflow_ids
            packed_weights[overflow_points, overflow_slots] = overflow_weights
        else:
            packed_ids = loaded_ids
        valid_mask = np.load(
            os.path.join(self.source_dir, self.manifest["valid_mask"])
        ).astype(bool)
        invalid_id = int(self.manifest["invalid_id"])
        packed_ids[packed_ids == invalid_id] = -1
        self.register_buffer(
            "point_code_ids",
            torch.from_numpy(packed_ids.astype(np.int32)).to(device),
        )
        self.register_buffer("valid_mask", torch.from_numpy(valid_mask).to(device))
        if self.shared_codebook:
            if not sparse_overflow:
                weights_name = self.manifest.get("point_code_weights")
                if weights_name:
                    packed_weights = np.load(
                        os.path.join(self.source_dir, weights_name)
                    ).astype(np.float32) / 255.0
                elif self.manifest.get("weight_dtype") == "implicit_unit":
                    packed_weights = (packed_ids >= 0).astype(np.float32)
                else:
                    raise ValueError(
                        "Shared codebooks require point weights or weight_dtype=implicit_unit"
                    )
            if packed_weights.shape != packed_ids.shape:
                raise ValueError("Adaptive code IDs and weights must have matching shapes")
            packed_weights[packed_ids < 0] = 0.0
            self.register_buffer(
                "point_code_weights", torch.from_numpy(packed_weights).to(device)
            )
        self.codebooks = nn.ParameterList()
        for name in self.manifest["codebook_files"]:
            values = np.load(os.path.join(self.source_dir, name)).astype(np.float32)
            self.codebooks.append(nn.Parameter(torch.from_numpy(values).to(device)))
        self.feature_dim = int(self.manifest["feature_dim"])
        self.num_gaussians = int(self.manifest["num_gaussians"])
        expected_slots = (
            int(self.manifest["id_slots"])
            if self.shared_codebook
            else len(self.codebooks)
        )
        if self.point_code_ids.shape != (self.num_gaussians, expected_slots):
            raise ValueError("Point code IDs do not match the artifact layout")

    def forward(self, gaussian_ids):
        valid_ids = gaussian_ids >= 0
        safe_gaussian_ids = gaussian_ids.clamp_min(0)
        point_valid = valid_ids & self.valid_mask[safe_gaussian_ids]
        code_ids = self.point_code_ids[safe_gaussian_ids].long()
        reconstruction = torch.zeros(
            (*gaussian_ids.shape, self.feature_dim),
            dtype=self.codebooks[0].dtype,
            device=gaussian_ids.device,
        )
        if self.shared_codebook:
            slot_weights = self.point_code_weights[safe_gaussian_ids]
            for slot in range(code_ids.shape[-1]):
                slot_ids = code_ids[..., slot]
                slot_valid = slot_ids >= 0
                reconstruction = reconstruction + (
                    self.codebooks[0][slot_ids.clamp_min(0)]
                    * slot_weights[..., slot].unsqueeze(-1)
                    * slot_valid.unsqueeze(-1)
                )
        else:
            for level, codebook in enumerate(self.codebooks):
                level_ids = code_ids[..., level].clamp_min(0)
                reconstruction = reconstruction + codebook[level_ids]
        reconstruction = torch.where(
            point_valid.unsqueeze(-1),
            reconstruction,
            torch.zeros_like(reconstruction),
        )
        return l2_normalize(reconstruction)

    def save_deployment_artifact(self, output_dir, training_metadata):
        os.makedirs(output_dir, exist_ok=True)
        ids_name = self.manifest["point_code_ids"]
        mask_name = self.manifest["valid_mask"]
        shutil.copy2(os.path.join(self.source_dir, ids_name), os.path.join(output_dir, ids_name))
        shutil.copy2(os.path.join(self.source_dir, mask_name), os.path.join(output_dir, mask_name))
        weight_bytes = 0
        overflow_point_bytes = 0
        overflow_slot_bytes = 0
        overflow_code_bytes = 0
        if self.shared_codebook:
            if self.manifest.get("storage_layout") == "base_plus_sparse_overflow":
                for key in (
                    "overflow_point_ids",
                    "overflow_code_ids",
                    "overflow_slots",
                    "overflow_weights",
                ):
                    name = self.manifest[key]
                    shutil.copy2(
                        os.path.join(self.source_dir, name), os.path.join(output_dir, name)
                    )
                overflow_point_bytes = os.path.getsize(
                    os.path.join(output_dir, self.manifest["overflow_point_ids"])
                )
                overflow_code_bytes = os.path.getsize(
                    os.path.join(output_dir, self.manifest["overflow_code_ids"])
                )
                overflow_slot_bytes = os.path.getsize(
                    os.path.join(output_dir, self.manifest["overflow_slots"])
                )
                weight_bytes = os.path.getsize(
                    os.path.join(output_dir, self.manifest["overflow_weights"])
                )
            elif self.manifest.get("weight_dtype") != "implicit_unit":
                weights_name = self.manifest["point_code_weights"]
                shutil.copy2(
                    os.path.join(self.source_dir, weights_name),
                    os.path.join(output_dir, weights_name),
                )
                weight_bytes = os.path.getsize(os.path.join(output_dir, weights_name))
        codebook_files = []
        codebook_bytes = 0
        for level, codebook in enumerate(self.codebooks):
            name = f"codebook_level_{level}.npy"
            values = codebook.detach().cpu().numpy().astype(np.float16)
            np.save(os.path.join(output_dir, name), values)
            codebook_files.append(name)
            codebook_bytes += int(values.nbytes)
        point_id_bytes = os.path.getsize(os.path.join(output_dir, ids_name)) + overflow_code_bytes
        valid_mask_bytes = os.path.getsize(os.path.join(output_dir, mask_name))
        total_semantic_bytes = (
            codebook_bytes
            + point_id_bytes
            + overflow_point_bytes
            + overflow_slot_bytes
            + weight_bytes
            + valid_mask_bytes
        )
        manifest = dict(self.manifest)
        manifest["codebook_files"] = codebook_files
        manifest["source_initialization"] = self.manifest.get("source")
        manifest["source"] = {
            "type": "direct_2d_semantic_training",
            "initial_artifact": self.source_dir,
        }
        manifest["training"] = training_metadata
        manifest["storage"] = {
            "codebook_bytes_fp16": codebook_bytes,
            "point_id_bytes": point_id_bytes,
            "overflow_point_bytes": overflow_point_bytes,
            "overflow_slot_bytes": overflow_slot_bytes,
            "point_weight_bytes": weight_bytes,
            "valid_mask_bytes": valid_mask_bytes,
            "total_semantic_bytes": total_semantic_bytes,
            "full_per_gaussian_fp16_bytes": self.num_gaussians * self.feature_dim * 2,
            "compression_ratio_vs_512d_fp16": (
                self.num_gaussians
                * self.feature_dim
                * 2
                / max(1, total_semantic_bytes)
            ),
            "bytes_per_gaussian_amortized": (
                total_semantic_bytes / self.num_gaussians
            ),
        }
        with open(os.path.join(output_dir, "manifest.json"), "w") as output:
            json.dump(manifest, output, indent=2)
        return manifest


def render_sampled_codebook(codebook, point_ids, point_weights):
    valid = point_ids >= 0
    features = codebook(point_ids)
    weights = torch.where(valid, point_weights, torch.zeros_like(point_weights))
    return l2_normalize((features * weights.unsqueeze(-1)).sum(dim=1))


def load_query_bank(path, semantic_dim, device):
    if not path:
        return None
    values = np.load(path).astype(np.float32)
    if values.ndim != 2 or values.shape[1] != semantic_dim:
        raise ValueError(
            f"Query bank must have shape [Q, {semantic_dim}], got {values.shape}"
        )
    return l2_normalize(torch.from_numpy(values).to(device))


def query_distribution_kl(
    prediction,
    target,
    query_bank,
    temperature,
    confidence_power=0.0,
):
    if query_bank is None:
        return prediction.new_zeros(())
    prediction_logits = prediction @ query_bank.T / temperature
    with torch.no_grad():
        target_probabilities = torch.softmax(target @ query_bank.T / temperature, dim=-1)
    per_sample = F.kl_div(
        torch.log_softmax(prediction_logits, dim=-1),
        target_probabilities,
        reduction="none",
    ).sum(dim=-1)
    if confidence_power <= 0.0:
        return per_sample.mean()
    target_top2 = target_probabilities.topk(k=2, dim=-1).values
    confidence = (target_top2[:, 0] - target_top2[:, 1]).pow(confidence_power)
    return (per_sample * confidence).sum() / confidence.sum().clamp_min(1e-8)


def segment_contrastive_loss(prediction, segment_ids, segment_features, temperature):
    """Distinguish the observed SAM segment from same-view visual hard negatives."""
    if segment_features.ndim != 2 or segment_features.shape[1] != prediction.shape[1]:
        raise ValueError("Segment feature table must match prediction dimensionality")
    if segment_ids.numel() and int(segment_ids.max()) >= segment_features.shape[0]:
        raise ValueError("Segment IDs exceed the current view feature table")
    logits = prediction @ l2_normalize(segment_features).T / temperature
    return F.cross_entropy(logits, segment_ids)


@torch.no_grad()
def evaluate_consistency(
    codebook,
    nuisance,
    caches,
    total_sums,
    total_weights,
    max_pixels_per_view,
    lovo_topk,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    observation_cosine_sum = 0.0
    adjusted_cosine_sum = 0.0
    lovo_cosine_sum = 0.0
    observation_count = 0
    lovo_count = 0
    device = codebook.codebooks[0].device
    for view_index, cache in enumerate(caches):
        num_pixels = cache["point_ids"].shape[0]
        count = min(max_pixels_per_view, num_pixels)
        indices = torch.randperm(num_pixels, generator=generator)[:count]
        point_ids = cache["point_ids"][indices].long().to(device)
        point_weights = cache["point_weights"][indices].float().to(device)
        segment_ids = cache["segment_ids"][indices].long()
        targets = cache["feature_latents"][segment_ids].float().to(device)
        canonical = render_sampled_codebook(codebook, point_ids, point_weights)
        observation_cosine_sum += float(
            F.cosine_similarity(canonical, targets, dim=-1).sum()
        )
        adjusted = canonical
        if nuisance is not None:
            adjusted = l2_normalize(canonical + nuisance(view_index).unsqueeze(0))
        adjusted_cosine_sum += float(
            F.cosine_similarity(adjusted, targets, dim=-1).sum()
        )
        observation_count += count
        loo_target, valid_loo = leave_one_view_out_target(
            cache,
            indices,
            point_ids[:, :lovo_topk],
            point_weights[:, :lovo_topk],
            total_sums,
            total_weights,
        )
        if valid_loo.any():
            lovo_cosine_sum += float(
                F.cosine_similarity(
                    canonical[valid_loo],
                    loo_target[valid_loo],
                    dim=-1,
                ).sum()
            )
            lovo_count += int(valid_loo.sum())
    return {
        "canonical_to_observation_cosine": observation_cosine_sum
        / max(1, observation_count),
        "nuisance_adjusted_to_observation_cosine": adjusted_cosine_sum
        / max(1, observation_count),
        "canonical_to_lovo_cosine": lovo_cosine_sum / max(1, lovo_count),
        "num_evaluated_pixels": observation_count,
        "num_lovo_pixels": lovo_count,
    }


def main():
    parser = ArgumentParser(
        description="Train large shared codebooks without per-Gaussian continuous semantic vectors."
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--initial_codebook_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch_pixels", type=int, default=4096)
    parser.add_argument("--codebook_lr", type=float, default=1e-3)
    parser.add_argument("--direct_weight", type=float, default=1.0)
    parser.add_argument("--lovo_weight", type=float, default=0.5)
    parser.add_argument("--lovo_topk", type=int, default=4)
    parser.add_argument("--query_bank", default=None)
    parser.add_argument("--query_kl_weight", type=float, default=0.0)
    parser.add_argument("--lovo_query_kl_weight", type=float, default=0.0)
    parser.add_argument("--query_temperature", type=float, default=0.07)
    parser.add_argument(
        "--query_confidence_power",
        type=float,
        default=0.0,
        help="Gate query KL by the frozen target's top-1/top-2 anchor margin; zero preserves legacy KL.",
    )
    parser.add_argument("--segment_contrastive_weight", type=float, default=0.0)
    parser.add_argument("--segment_contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--nuisance_rank", type=int, default=4)
    parser.add_argument("--nuisance_lr", type=float, default=5e-3)
    parser.add_argument("--nuisance_regularization", type=float, default=1e-3)
    parser.add_argument(
        "--view_sampling",
        choices=["uniform", "segment_importance"],
        default="uniform",
    )
    parser.add_argument("--importance_groups", type=int, default=16)
    parser.add_argument("--importance_temperature", type=float, default=1.0)
    parser.add_argument("--importance_uniform_mix", type=float, default=0.25)
    parser.add_argument("--importance_max_step_kl", type=float, default=0.02)
    parser.add_argument("--importance_max_base_kl", type=float, default=0.25)
    parser.add_argument("--importance_update_interval", type=int, default=100)
    parser.add_argument("--importance_ema_decay", type=float, default=0.95)
    parser.add_argument("--importance_ratio_clip", type=float, default=5.0)
    parser.add_argument("--importance_rarity_weight", type=float, default=0.1)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_pixels_per_view", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.iterations <= 0 or args.batch_pixels <= 0:
        raise ValueError("Iterations and batch size must be positive")
    if args.lovo_topk <= 0 or args.query_temperature <= 0.0:
        raise ValueError("LOVO top-k and query temperature must be positive")
    if args.query_confidence_power < 0.0:
        raise ValueError("query confidence power must be non-negative")
    if args.segment_contrastive_weight < 0.0 or args.segment_contrastive_temperature <= 0.0:
        raise ValueError("Segment contrastive weight must be non-negative and temperature positive")
    if args.nuisance_rank < 0:
        raise ValueError("Nuisance rank must be non-negative")
    if args.view_sampling == "segment_importance" and args.importance_groups <= 1:
        raise ValueError("Importance sampling requires at least two groups")

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output)
    deployment_dir = os.path.join(output_dir, "artifact")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if os.path.isfile(os.path.join(deployment_dir, "manifest.json")) and not args.force:
        print(f"Reuse trained Gaussian codebook: {deployment_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    cache_manifest = load_json(os.path.join(cache_dir, "manifest.json"))
    if cache_manifest.get("codec_type") != "identity":
        raise ValueError(
            "Large semantic codebooks require an identity 512D observation cache; "
            "the 64D autoencoder cache is intentionally rejected."
        )
    consensus = torch.load(
        os.path.join(cache_dir, cache_manifest["consensus"]),
        map_location="cpu",
    )
    caches = load_view_caches(cache_dir, cache_manifest)
    codebook = MultilevelGaussianCodebook(args.initial_codebook_dir).cuda()
    semantic_dim = int(cache_manifest["semantic_dim"])
    if semantic_dim != 512 or codebook.feature_dim != semantic_dim:
        raise ValueError("Codebook and observation cache must both use 512D semantics")
    if codebook.num_gaussians != int(cache_manifest["num_gaussians"]):
        raise ValueError("Codebook IDs do not match the observation cache")

    query_bank = load_query_bank(args.query_bank, semantic_dim, "cuda")
    if (args.query_kl_weight > 0.0 or args.lovo_query_kl_weight > 0.0) and query_bank is None:
        raise ValueError("Positive query KL weights require --query_bank")
    optimizer = torch.optim.AdamW(
        codebook.codebooks.parameters(),
        lr=args.codebook_lr,
        weight_decay=1e-5,
    )
    nuisance = None
    nuisance_optimizer = None
    if args.nuisance_rank > 0:
        nuisance = ViewNuisance(len(caches), args.nuisance_rank, semantic_dim).cuda()
        nuisance_optimizer = torch.optim.Adam(nuisance.parameters(), lr=args.nuisance_lr)

    total_sums = consensus["total_sums"].float().cuda()
    total_weights = consensus["total_weights"].float().cuda()
    initial_features = consensus["initial_features"].float()
    view_sampler = None
    if args.view_sampling == "segment_importance":
        view_sampler = SegmentWiseViewSampler(
            caches,
            initial_features.cuda(),
            consensus["total_weights"].float().cuda(),
            num_groups=args.importance_groups,
            temperature=args.importance_temperature,
            uniform_mix=args.importance_uniform_mix,
            max_step_kl=args.importance_max_step_kl,
            max_base_kl=args.importance_max_base_kl,
            update_interval=args.importance_update_interval,
            ema_decay=args.importance_ema_decay,
            ratio_clip=args.importance_ratio_clip,
            rarity_weight=args.importance_rarity_weight,
            seed=args.seed,
        )

    generator = torch.Generator().manual_seed(args.seed)
    view_order = torch.randperm(len(caches), generator=generator).tolist()
    view_cursor = 0
    history = []
    running = {
        "loss": 0.0,
        "direct": 0.0,
        "lovo": 0.0,
        "query_kl": 0.0,
        "lovo_query_kl": 0.0,
        "segment_contrastive": 0.0,
        "nuisance": 0.0,
        "importance_ratio": 0.0,
    }

    for iteration in range(1, args.iterations + 1):
        sampled_group = None
        importance_ratio = 1.0
        clipped_ratio = 1.0
        if view_sampler is None:
            if view_cursor >= len(view_order):
                view_order = torch.randperm(len(caches), generator=generator).tolist()
                view_cursor = 0
            view_index = view_order[view_cursor]
            view_cursor += 1
            cache = caches[view_index]
            batch_indices = torch.randint(
                cache["point_ids"].shape[0],
                (min(args.batch_pixels, cache["point_ids"].shape[0]),),
                generator=generator,
            )
        else:
            sample = view_sampler.sample(args.batch_pixels)
            view_index = sample.view_index
            sampled_group = sample.group_index
            batch_indices = sample.batch_indices
            importance_ratio = sample.importance_ratio
            clipped_ratio = sample.clipped_ratio
            cache = caches[view_index]

        point_ids = cache["point_ids"][batch_indices].long().cuda(non_blocking=True)
        point_weights = cache["point_weights"][batch_indices].float().cuda(non_blocking=True)
        segment_ids_cpu = cache["segment_ids"][batch_indices].long()
        targets = cache["feature_latents"][segment_ids_cpu].float().cuda(non_blocking=True)
        segment_ids = segment_ids_cpu.cuda(non_blocking=True)
        segment_features = cache["feature_latents"].float().cuda(non_blocking=True)
        canonical = render_sampled_codebook(codebook, point_ids, point_weights)
        adjusted = canonical
        if nuisance is not None:
            nuisance_vector = nuisance(view_index)
            adjusted = l2_normalize(canonical + nuisance_vector.unsqueeze(0))
            nuisance_loss = (
                nuisance_vector.square().mean()
                + 0.01 * nuisance.basis.square().mean()
            )
        else:
            nuisance_loss = canonical.new_zeros(())

        direct_loss = 1.0 - F.cosine_similarity(adjusted, targets, dim=-1).mean()
        query_kl_loss = query_distribution_kl(
            canonical,
            targets,
            query_bank,
            args.query_temperature,
            args.query_confidence_power,
        )
        contrastive_loss = segment_contrastive_loss(
            canonical,
            segment_ids,
            segment_features,
            args.segment_contrastive_temperature,
        )
        loo_target, valid_loo = leave_one_view_out_target(
            cache,
            batch_indices,
            point_ids[:, : args.lovo_topk],
            point_weights[:, : args.lovo_topk],
            total_sums,
            total_weights,
        )
        if valid_loo.any():
            lovo_loss = 1.0 - F.cosine_similarity(
                canonical[valid_loo],
                loo_target[valid_loo],
                dim=-1,
            ).mean()
            lovo_query_kl_loss = query_distribution_kl(
                canonical[valid_loo],
                loo_target[valid_loo],
                query_bank,
                args.query_temperature,
                args.query_confidence_power,
            )
        else:
            lovo_loss = canonical.new_zeros(())
            lovo_query_kl_loss = canonical.new_zeros(())

        data_loss = (
            args.direct_weight * direct_loss
            + args.lovo_weight * lovo_loss
            + args.query_kl_weight * query_kl_loss
            + args.lovo_query_kl_weight * lovo_query_kl_loss
            + args.segment_contrastive_weight * contrastive_loss
        )
        loss = clipped_ratio * data_loss + args.nuisance_regularization * nuisance_loss
        optimizer.zero_grad(set_to_none=True)
        if nuisance_optimizer is not None:
            nuisance_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if nuisance_optimizer is not None:
            nuisance_optimizer.step()
        if view_sampler is not None:
            view_sampler.observe(sampled_group, view_index, float(data_loss.detach()))
            view_sampler.maybe_update(iteration)

        values = {
            "loss": float(loss.detach()),
            "direct": float(direct_loss.detach()),
            "lovo": float(lovo_loss.detach()),
            "query_kl": float(query_kl_loss.detach()),
            "lovo_query_kl": float(lovo_query_kl_loss.detach()),
            "segment_contrastive": float(contrastive_loss.detach()),
            "nuisance": float(nuisance_loss.detach()),
            "importance_ratio": importance_ratio,
        }
        for name, value in values.items():
            running[name] += value
        if iteration % args.log_interval == 0 or iteration == args.iterations:
            divisor = args.log_interval
            if iteration % args.log_interval:
                divisor = iteration % args.log_interval
            row = {"iteration": iteration}
            row.update({name: value / max(1, divisor) for name, value in running.items()})
            history.append(row)
            print(json.dumps(row))
            running = {name: 0.0 for name in running}

    consistency = evaluate_consistency(
        codebook,
        nuisance,
        caches,
        total_sums,
        total_weights,
        args.eval_pixels_per_view,
        args.lovo_topk,
        args.seed + 1000,
    )
    sampler_metrics = view_sampler.diagnostics() if view_sampler is not None else None
    deployment_manifest = codebook.save_deployment_artifact(
        deployment_dir,
        {
            "cache_dir": cache_dir,
            "config": vars(args),
            "consistency": consistency,
            "importance_sampling": sampler_metrics,
        },
    )
    metrics = {
        "cache_dir": cache_dir,
        "initial_codebook_dir": os.path.abspath(args.initial_codebook_dir),
        "deployment_artifact": deployment_dir,
        "semantic_dim": semantic_dim,
        "num_gaussians": codebook.num_gaussians,
        "num_codebook_levels": len(codebook.codebooks),
        "code_counts": [int(codebook_level.shape[0]) for codebook_level in codebook.codebooks],
        "storage": deployment_manifest["storage"],
        "history": history,
        "consistency": consistency,
        "importance_sampling": sampler_metrics,
        "config": vars(args),
    }
    save_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved trained Gaussian codebook to {deployment_dir}")


if __name__ == "__main__":
    main()
