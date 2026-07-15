#!/usr/bin/env python
import json
import os
import random
import sys
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from segment_view_sampler import SegmentWiseViewSampler
from semantic_field_utils import l2_normalize, load_json, load_semantic_codec, save_json
from utils.general_utils import safe_state


def render_sampled_latents(embedding, point_ids, point_weights):
    valid = point_ids >= 0
    safe_ids = torch.where(valid, point_ids, torch.zeros_like(point_ids))
    point_features = l2_normalize(embedding(safe_ids))
    weights = torch.where(valid, point_weights, torch.zeros_like(point_weights))
    rendered = (point_features * weights.unsqueeze(-1)).sum(dim=1)
    return l2_normalize(rendered)


def leave_one_view_out_target(cache, batch_indices, point_ids, point_weights, total_sums, total_weights):
    aggregate_indices_cpu = cache["pixel_aggregate_indices"][
        batch_indices, : point_ids.shape[1]
    ].long()
    aggregate_indices = aggregate_indices_cpu.to(total_sums.device)
    valid_points = (point_ids >= 0) & (aggregate_indices >= 0)
    safe_ids = torch.where(valid_points, point_ids, torch.zeros_like(point_ids))

    safe_aggregate_indices_cpu = torch.where(
        aggregate_indices_cpu >= 0,
        aggregate_indices_cpu,
        torch.zeros_like(aggregate_indices_cpu),
    )
    current_sums = cache["aggregate_sums"][safe_aggregate_indices_cpu].to(total_sums.device)
    current_weights = cache["aggregate_weights"][safe_aggregate_indices_cpu].to(total_weights.device)
    loo_sums = total_sums[safe_ids] - current_sums
    loo_weights = total_weights[safe_ids] - current_weights
    valid_loo = valid_points & (loo_weights > 1e-6)
    safe_loo_weights = loo_weights.clamp_min(1e-6)
    point_targets = l2_normalize(loo_sums / safe_loo_weights.unsqueeze(-1))
    effective_weights = torch.where(valid_loo, point_weights, torch.zeros_like(point_weights))
    rendered = (point_targets * effective_weights.unsqueeze(-1)).sum(dim=1)
    valid_pixels = effective_weights.sum(dim=1) > 1e-6
    return l2_normalize(rendered), valid_pixels


class ViewNuisance(nn.Module):
    def __init__(self, num_views, rank, semantic_dim):
        super().__init__()
        self.codes = nn.Embedding(num_views, rank)
        self.basis = nn.Parameter(torch.empty(rank, semantic_dim))
        nn.init.zeros_(self.codes.weight)
        nn.init.normal_(self.basis, mean=0.0, std=0.02)

    def forward(self, view_index):
        index = torch.tensor(view_index, dtype=torch.long, device=self.basis.device)
        return self.codes(index) @ self.basis


def load_view_caches(cache_dir, manifest):
    caches = []
    for entry in tqdm(manifest["views"], desc="Loading observation caches"):
        cache = torch.load(os.path.join(cache_dir, entry["cache"]), map_location="cpu")
        if cache["point_ids"].shape[0] == 0:
            continue
        caches.append(cache)
    if not caches:
        raise ValueError("Semantic observation manifest contains no views")
    return caches


@torch.no_grad()
def evaluate_consistency(
    embedding,
    nuisance,
    caches,
    total_sums,
    total_weights,
    max_pixels_per_view,
    lovo_topk,
    seed,
):
    generator = torch.Generator().manual_seed(seed)
    direct_sum = 0.0
    nuisance_sum = 0.0
    lovo_sum = 0.0
    direct_count = 0
    lovo_count = 0
    device = embedding.weight.device
    for view_index, cache in enumerate(caches):
        num_pixels = cache["point_ids"].shape[0]
        count = min(max_pixels_per_view, num_pixels)
        indices = torch.randperm(num_pixels, generator=generator)[:count]
        point_ids = cache["point_ids"][indices].long().to(device)
        point_weights = cache["point_weights"][indices].float().to(device)
        segment_ids = cache["segment_ids"][indices].long()
        targets = cache["feature_latents"][segment_ids].float().to(device)
        canonical = render_sampled_latents(embedding, point_ids, point_weights)
        direct_sum += float(F.cosine_similarity(canonical, targets, dim=-1).sum())
        if nuisance is not None:
            adjusted = l2_normalize(canonical + nuisance(view_index).unsqueeze(0))
        else:
            adjusted = canonical
        nuisance_sum += float(F.cosine_similarity(adjusted, targets, dim=-1).sum())
        direct_count += count
        loo_target, valid_loo = leave_one_view_out_target(
            cache,
            indices,
            point_ids[:, :lovo_topk],
            point_weights[:, :lovo_topk],
            total_sums,
            total_weights,
        )
        if valid_loo.any():
            lovo_sum += float(
                F.cosine_similarity(canonical[valid_loo], loo_target[valid_loo], dim=-1).sum()
            )
            lovo_count += int(valid_loo.sum())
    return {
        "canonical_to_observation_cosine": direct_sum / max(1, direct_count),
        "nuisance_adjusted_to_observation_cosine": nuisance_sum / max(1, direct_count),
        "canonical_to_lovo_cosine": lovo_sum / max(1, lovo_count),
        "num_evaluated_pixels": direct_count,
        "num_lovo_pixels": lovo_count,
    }


def main():
    parser = ArgumentParser(description="Train a low-dimensional Gaussian semantic field from cached 2D observations.")
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--batch_pixels", type=int, default=4096)
    parser.add_argument("--semantic_lr", type=float, default=0.02)
    parser.add_argument("--nuisance_lr", type=float, default=0.005)
    parser.add_argument("--direct_weight", type=float, default=1.0)
    parser.add_argument("--decoded_weight", type=float, default=0.25)
    parser.add_argument("--lovo_weight", type=float, default=0.0)
    parser.add_argument("--lovo_topk", type=int, default=4)
    parser.add_argument("--nuisance_rank", type=int, default=0)
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
    parser.add_argument("--importance_max_base_kl", type=float, default=0.5)
    parser.add_argument("--importance_update_interval", type=int, default=100)
    parser.add_argument("--importance_ema_decay", type=float, default=0.95)
    parser.add_argument("--importance_ratio_clip", type=float, default=5.0)
    parser.add_argument("--importance_rarity_weight", type=float, default=0.1)
    parser.add_argument("--importance_kmeans_samples", type=int, default=65536)
    parser.add_argument("--importance_kmeans_iterations", type=int, default=8)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--eval_pixels_per_view", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.iterations <= 0:
        raise ValueError("--iterations must be positive")
    if args.batch_pixels <= 0:
        raise ValueError("--batch_pixels must be positive")
    if args.nuisance_rank < 0:
        raise ValueError("--nuisance_rank must be non-negative")
    if args.lovo_topk <= 0:
        raise ValueError("--lovo_topk must be positive")
    if args.view_sampling == "segment_importance" and args.importance_groups <= 1:
        raise ValueError("--importance_groups must be greater than one")
    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    cache_dir = os.path.abspath(args.cache_dir)
    output_dir = os.path.abspath(args.output)
    artifact_path = os.path.join(output_dir, "semantic_field.pt")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if os.path.exists(artifact_path) and not args.force:
        print(f"Reuse existing semantic field: {artifact_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    manifest = load_json(os.path.join(cache_dir, "manifest.json"))
    consensus = torch.load(os.path.join(cache_dir, manifest["consensus"]), map_location="cpu")
    caches = load_view_caches(cache_dir, manifest)
    codec, codec_payload = load_semantic_codec(os.path.join(cache_dir, manifest["codec"]), device="cuda")
    semantic_dim = int(manifest["semantic_dim"])
    num_gaussians = int(manifest["num_gaussians"])
    if codec.semantic_dim != semantic_dim:
        raise ValueError("Codec dimension does not match semantic observation cache")

    initial_features = consensus["initial_features"].float()
    embedding = nn.Embedding(num_gaussians, semantic_dim, sparse=True).cuda()
    with torch.no_grad():
        embedding.weight.copy_(initial_features.cuda())
    semantic_optimizer = torch.optim.SparseAdam(embedding.parameters(), lr=args.semantic_lr)

    nuisance = None
    nuisance_optimizer = None
    if args.nuisance_rank > 0:
        nuisance = ViewNuisance(len(caches), args.nuisance_rank, semantic_dim).cuda()
        nuisance_optimizer = torch.optim.Adam(nuisance.parameters(), lr=args.nuisance_lr)

    total_sums = consensus["total_sums"].float().cuda()
    total_weights = consensus["total_weights"].float().cuda()
    view_sampler = None
    if args.view_sampling == "segment_importance":
        view_sampler = SegmentWiseViewSampler(
            caches,
            embedding.weight.detach(),
            total_weights,
            num_groups=args.importance_groups,
            temperature=args.importance_temperature,
            uniform_mix=args.importance_uniform_mix,
            max_step_kl=args.importance_max_step_kl,
            max_base_kl=args.importance_max_base_kl,
            update_interval=args.importance_update_interval,
            ema_decay=args.importance_ema_decay,
            ratio_clip=args.importance_ratio_clip,
            rarity_weight=args.importance_rarity_weight,
            kmeans_samples=args.importance_kmeans_samples,
            kmeans_iterations=args.importance_kmeans_iterations,
            seed=args.seed,
        )
        print(json.dumps({"initial_importance_sampler": view_sampler.diagnostics()}))
    generator = torch.Generator().manual_seed(args.seed)
    view_order = torch.randperm(len(caches), generator=generator).tolist()
    view_cursor = 0
    history = []
    running = {
        "loss": 0.0,
        "data": 0.0,
        "direct": 0.0,
        "decoded": 0.0,
        "lovo": 0.0,
        "nuisance": 0.0,
        "importance_ratio": 0.0,
        "clipped_importance_ratio": 0.0,
    }

    for iteration in range(1, args.iterations + 1):
        sampled_group = None
        importance_ratio = 1.0
        clipped_importance_ratio = 1.0
        if view_sampler is None:
            if view_cursor >= len(view_order):
                view_order = torch.randperm(len(caches), generator=generator).tolist()
                view_cursor = 0
            view_index = view_order[view_cursor]
            view_cursor += 1
            cache = caches[view_index]
            num_pixels = cache["point_ids"].shape[0]
            batch_count = min(args.batch_pixels, num_pixels)
            batch_indices = torch.randint(num_pixels, (batch_count,), generator=generator)
        else:
            sample = view_sampler.sample(args.batch_pixels)
            view_index = sample.view_index
            sampled_group = sample.group_index
            batch_indices = sample.batch_indices
            importance_ratio = sample.importance_ratio
            clipped_importance_ratio = sample.clipped_ratio
        cache = caches[view_index]
        point_ids = cache["point_ids"][batch_indices].long().cuda(non_blocking=True)
        point_weights = cache["point_weights"][batch_indices].float().cuda(non_blocking=True)
        segment_ids = cache["segment_ids"][batch_indices].long()
        targets = cache["feature_latents"][segment_ids].float().cuda(non_blocking=True)

        canonical = render_sampled_latents(embedding, point_ids, point_weights)
        if nuisance is not None:
            nuisance_vector = nuisance(view_index)
            direct_prediction = l2_normalize(canonical + nuisance_vector.unsqueeze(0))
            nuisance_loss = nuisance_vector.square().mean() + 0.01 * nuisance.basis.square().mean()
        else:
            direct_prediction = canonical
            nuisance_loss = canonical.new_zeros(())

        direct_loss = 1.0 - F.cosine_similarity(direct_prediction, targets, dim=-1).mean()
        if args.decoded_weight > 0:
            decoded_prediction = codec.decode(direct_prediction)
            with torch.no_grad():
                decoded_target = codec.decode(targets)
            decoded_loss = 1.0 - F.cosine_similarity(decoded_prediction, decoded_target, dim=-1).mean()
        else:
            decoded_loss = canonical.new_zeros(())

        if args.lovo_weight > 0:
            lovo_topk = min(args.lovo_topk, point_ids.shape[1])
            loo_target, valid_loo = leave_one_view_out_target(
                cache,
                batch_indices,
                point_ids[:, :lovo_topk],
                point_weights[:, :lovo_topk],
                total_sums,
                total_weights,
            )
            if valid_loo.any():
                lovo_loss = 1.0 - F.cosine_similarity(
                    canonical[valid_loo],
                    loo_target[valid_loo],
                    dim=-1,
                ).mean()
            else:
                lovo_loss = canonical.new_zeros(())
        else:
            lovo_loss = canonical.new_zeros(())

        data_loss = (
            args.direct_weight * direct_loss
            + args.decoded_weight * decoded_loss
            + args.lovo_weight * lovo_loss
        )
        loss = (
            clipped_importance_ratio * data_loss
            + args.nuisance_regularization * nuisance_loss
        )
        semantic_optimizer.zero_grad(set_to_none=True)
        if nuisance_optimizer is not None:
            nuisance_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        semantic_optimizer.step()
        if nuisance_optimizer is not None:
            nuisance_optimizer.step()
        if view_sampler is not None:
            view_sampler.observe(sampled_group, view_index, float(data_loss.detach()))
            view_sampler.maybe_update(iteration)

        values = {
            "loss": float(loss.detach()),
            "data": float(data_loss.detach()),
            "direct": float(direct_loss.detach()),
            "decoded": float(decoded_loss.detach()),
            "lovo": float(lovo_loss.detach()),
            "nuisance": float(nuisance_loss.detach()),
            "importance_ratio": importance_ratio,
            "clipped_importance_ratio": clipped_importance_ratio,
        }
        for name, value in values.items():
            running[name] += value
        if iteration % args.log_interval == 0 or iteration == args.iterations:
            divisor = args.log_interval if iteration % args.log_interval == 0 else args.iterations % args.log_interval
            divisor = max(1, divisor)
            row = {"iteration": iteration}
            row.update({name: value / divisor for name, value in running.items()})
            history.append(row)
            print(json.dumps(row))
            running = {name: 0.0 for name in running}

    consistency = evaluate_consistency(
        embedding,
        nuisance,
        caches,
        total_sums,
        total_weights,
        max_pixels_per_view=args.eval_pixels_per_view,
        lovo_topk=min(args.lovo_topk, caches[0]["point_ids"].shape[1]),
        seed=args.seed + 1000,
    )
    with torch.no_grad():
        semantic_features = l2_normalize(embedding.weight).detach().cpu().to(torch.float16)
    support_weights = consensus["total_weights"].detach().cpu()
    sampler_state_path = None
    sampler_metrics = None
    if view_sampler is not None:
        sampler_state_path = os.path.join(output_dir, "segment_view_sampler.pt")
        sampler_metrics = view_sampler.diagnostics()
        torch.save(view_sampler.state_dict(), sampler_state_path)
    artifact = {
        "format_version": 1,
        "semantic_features": semantic_features,
        "support_weights": support_weights,
        "semantic_dim": semantic_dim,
        "num_gaussians": num_gaussians,
        "cache_manifest": os.path.join(cache_dir, "manifest.json"),
        "codec": os.path.join(cache_dir, manifest["codec"]),
        "config": vars(args),
        "nuisance_state": nuisance.state_dict() if nuisance is not None else None,
        "sampler_state": sampler_state_path,
    }
    torch.save(artifact, artifact_path)
    semantic_bytes = int(semantic_features.numel() * semantic_features.element_size())
    metrics = {
        "cache_dir": cache_dir,
        "output": output_dir,
        "artifact": artifact_path,
        "semantic_dim": semantic_dim,
        "num_gaussians": num_gaussians,
        "supported_fraction": float((support_weights > 0).float().mean()),
        "semantic_storage_bytes_fp16": semantic_bytes,
        "semantic_storage_megabytes_fp16": semantic_bytes / (1024.0 ** 2),
        "codec_storage_bytes": os.path.getsize(os.path.join(cache_dir, manifest["codec"])),
        "history": history,
        "consistency": consistency,
        "importance_sampling": sampler_metrics,
        "config": vars(args),
    }
    save_json(metrics_path, metrics)
    print(json.dumps(metrics, indent=2))
    print(f"Saved semantic field to {artifact_path}")


if __name__ == "__main__":
    main()
