#!/usr/bin/env python
"""Extract a sparse second semantic mode from reproducible multiview evidence."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm


def load_torch(path, mmap=False):
    try:
        return torch.load(path, map_location="cpu", mmap=mmap)
    except TypeError:
        return torch.load(path, map_location="cpu")


def select_reproducible_modes(
    base_features,
    split_sums,
    split_weights,
    split_support,
    min_views_per_split,
    min_compactness,
    min_cross_split_cosine,
    max_base_cosine,
    support_saturation,
):
    """Select modes independently recovered from both interleaved view splits."""
    if split_sums.ndim != 3 or split_sums.shape[0] != 2:
        raise ValueError("split_sums must have shape [2, N, D]")
    if base_features.shape != split_sums.shape[1:]:
        raise ValueError("base_features and split_sums must match")
    if split_weights.shape != split_sums.shape[:2]:
        raise ValueError("split_weights must have shape [2, N]")
    if split_support.shape != split_weights.shape:
        raise ValueError("split_support must match split_weights")
    if min_views_per_split <= 0 or support_saturation < min_views_per_split:
        raise ValueError("view support thresholds are inconsistent")
    if not 0.0 <= min_compactness <= 1.0:
        raise ValueError("min_compactness must be in [0, 1]")
    if not -1.0 <= min_cross_split_cosine <= 1.0:
        raise ValueError("min_cross_split_cosine must be in [-1, 1]")
    if not -1.0 <= max_base_cosine <= 1.0:
        raise ValueError("max_base_cosine must be in [-1, 1]")

    norms = split_sums.float().norm(dim=-1)
    centers = split_sums.float() / norms.clamp_min(1e-8).unsqueeze(-1)
    supported = split_weights > 0.0
    centers = torch.where(supported.unsqueeze(-1), centers, torch.zeros_like(centers))
    compactness = norms / split_weights.float().clamp_min(1e-8)
    compactness = compactness.clamp(0.0, 1.0)

    cross_cosine = (centers[0] * centers[1]).sum(dim=-1).clamp(-1.0, 1.0)
    base = F.normalize(base_features.float(), dim=-1)
    candidate = F.normalize(centers[0] + centers[1], dim=-1)
    base_cosine = (candidate * base).sum(dim=-1).clamp(-1.0, 1.0)
    base_valid = base_features.float().norm(dim=-1) > 0.0

    selected = (
        base_valid
        & supported.all(dim=0)
        & (split_support >= min_views_per_split).all(dim=0)
        & (compactness >= min_compactness).all(dim=0)
        & (cross_cosine >= min_cross_split_cosine)
        & (base_cosine <= max_base_cosine)
    )

    support_reliability = (
        split_support.min(dim=0).values.float() / float(support_saturation)
    ).clamp(0.0, 1.0)
    weight_balance = (
        2.0
        * split_weights.min(dim=0).values
        / split_weights.sum(dim=0).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    reproducibility = (
        (cross_cosine - min_cross_split_cosine)
        / max(1e-8, 1.0 - min_cross_split_cosine)
    ).clamp(0.0, 1.0)
    compact_reliability = compactness.prod(dim=0).sqrt()
    reliability = (
        reproducibility
        * compact_reliability
        * weight_balance.sqrt()
        * support_reliability.sqrt()
    ).clamp(0.0, 1.0)
    reliability[~selected] = 0.0
    return {
        "candidate": candidate,
        "selected": selected,
        "reliability": reliability,
        "compactness": compactness,
        "cross_cosine": cross_cosine,
        "base_cosine": base_cosine,
        "weight_balance": weight_balance,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--view_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--deviation_cosine_max", type=float, default=0.75)
    parser.add_argument("--observation_weight_power", type=float, default=0.5)
    parser.add_argument("--min_observation_weight", type=float, default=0.0)
    parser.add_argument("--min_views_per_split", type=int, default=3)
    parser.add_argument("--support_saturation", type=int, default=6)
    parser.add_argument("--min_compactness", type=float, default=0.88)
    parser.add_argument("--min_cross_split_cosine", type=float, default=0.90)
    parser.add_argument("--max_base_cosine", type=float, default=0.90)
    parser.add_argument("--final_chunk", type=int, default=8192)
    parser.add_argument("--max_views", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    if not -1.0 <= args.deviation_cosine_max <= 1.0:
        raise ValueError("deviation cosine must be in [-1, 1]")
    if args.observation_weight_power <= 0.0 or args.min_observation_weight < 0.0:
        raise ValueError("observation weights must be non-negative with positive power")
    if args.final_chunk <= 0 or args.max_views < 0:
        raise ValueError("chunk size must be positive and max views non-negative")

    base_path = os.path.abspath(args.base_consensus)
    cache_dir = os.path.abspath(args.view_cache_dir)
    base_payload = load_torch(base_path, mmap=True)
    if "initial_features" not in base_payload:
        raise ValueError("Base consensus does not contain initial_features")
    base_cpu = base_payload["initial_features"].detach().cpu()
    num_gaussians, feature_dim = base_cpu.shape
    del base_payload

    with open(os.path.join(cache_dir, "manifest.json")) as source:
        cache_manifest = json.load(source)
    views = cache_manifest.get("views", [])
    if args.max_views:
        views = views[: args.max_views]
    if not views:
        raise ValueError("View cache manifest contains no per-view caches")

    device = torch.device(args.device)
    base = base_cpu.to(device=device, dtype=torch.float16)
    split_sums = torch.zeros(
        (2, num_gaussians, feature_dim), dtype=torch.float32, device=device
    )
    split_weights = torch.zeros((2, num_gaussians), dtype=torch.float32, device=device)
    split_support = torch.zeros((2, num_gaussians), dtype=torch.int32, device=device)
    selected_observations = [0, 0]
    total_observations = [0, 0]

    for view in tqdm(views, desc="Collecting discordant view modes"):
        cache_path = os.path.join(cache_dir, view["cache"])
        payload = load_torch(cache_path)
        required = {"aggregate_ids", "aggregate_weights", "aggregate_sums"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"View cache is missing fields: {sorted(missing)}")
        ids = payload["aggregate_ids"].long().to(device)
        if ids.numel() and int(ids.max()) >= num_gaussians:
            raise ValueError("View cache Gaussian IDs exceed the base consensus")
        raw_weights = payload["aggregate_weights"].float().to(device)
        observations = F.normalize(
            payload["aggregate_sums"].to(device=device, dtype=torch.float32), dim=-1
        )
        del payload

        split_index = int(view["view_index"]) % 2
        total_observations[split_index] += int(ids.numel())
        cosine = (observations * base[ids].float()).sum(dim=-1).clamp(-1.0, 1.0)
        valid = (
            (raw_weights >= args.min_observation_weight)
            & (cosine <= args.deviation_cosine_max)
            & (base[ids].float().norm(dim=-1) > 0.0)
        )
        if valid.any():
            ids = ids[valid]
            observations = observations[valid]
            evidence = raw_weights[valid].pow(args.observation_weight_power)
            split_sums[split_index].index_add_(
                0, ids, observations * evidence.unsqueeze(-1)
            )
            split_weights[split_index].index_add_(0, ids, evidence)
            split_support[split_index].index_add_(
                0, ids, torch.ones_like(ids, dtype=torch.int32)
            )
            selected_observations[split_index] += int(ids.numel())
        del ids, raw_weights, observations, cosine, valid

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    selected_ids_parts = []
    feature_parts = []
    reliability_parts = []
    diagnostic_sums = {
        "compactness": 0.0,
        "cross_cosine": 0.0,
        "base_cosine": 0.0,
        "weight_balance": 0.0,
    }
    selected_count = 0

    for start in tqdm(range(0, num_gaussians, args.final_chunk), desc="Validating modes"):
        end = min(start + args.final_chunk, num_gaussians)
        result = select_reproducible_modes(
            base[start:end],
            split_sums[:, start:end],
            split_weights[:, start:end],
            split_support[:, start:end],
            args.min_views_per_split,
            args.min_compactness,
            args.min_cross_split_cosine,
            args.max_base_cosine,
            args.support_saturation,
        )
        local = torch.nonzero(result["selected"], as_tuple=False).squeeze(1)
        if local.numel():
            selected_ids_parts.append((local + start).cpu())
            feature_parts.append(result["candidate"][local].to(torch.float16).cpu())
            reliability_parts.append(result["reliability"][local].cpu())
            count = int(local.numel())
            selected_count += count
            for name in diagnostic_sums:
                values = result[name]
                if values.ndim == 2:
                    values = values.mean(dim=0)
                diagnostic_sums[name] += float(values[local].sum())

    point_ids = torch.cat(selected_ids_parts) if selected_ids_parts else torch.empty(0, dtype=torch.long)
    features = (
        torch.cat(feature_parts)
        if feature_parts
        else torch.empty((0, feature_dim), dtype=torch.float16)
    )
    reliability = (
        torch.cat(reliability_parts)
        if reliability_parts
        else torch.empty(0, dtype=torch.float32)
    )
    packed_ids = point_ids.numpy().astype(np.uint32)
    packed_features = features.numpy().astype(np.float16)
    packed_reliability = np.rint(reliability.numpy() * 255.0).astype(np.uint8)
    np.save(os.path.join(output_dir, "point_ids.npy"), packed_ids)
    np.save(os.path.join(output_dir, "features.npy"), packed_features)
    np.save(os.path.join(output_dir, "reliability.npy"), packed_reliability)

    storage_bytes = int(packed_ids.nbytes + packed_features.nbytes + packed_reliability.nbytes)
    manifest = {
        "format_version": 1,
        "representation": "sparse_continuous_semantic_hypothesis",
        "method": "split_reproducible_view_semantic_mode",
        "num_gaussians": int(num_gaussians),
        "num_hypotheses": int(selected_count),
        "selected_fraction": float(selected_count / max(1, num_gaussians)),
        "feature_dim": int(feature_dim),
        "point_ids": "point_ids.npy",
        "features": "features.npy",
        "reliability": "reliability.npy",
        "feature_dtype": "float16",
        "id_dtype": "uint32",
        "reliability_dtype": "uint8",
        "mean_reliability": float(reliability.mean()) if reliability.numel() else 0.0,
        "selected_diagnostics": {
            name: float(value / max(1, selected_count))
            for name, value in diagnostic_sums.items()
        },
        "observation_statistics": {
            "num_views": int(len(views)),
            "total_by_split": total_observations,
            "discordant_by_split": selected_observations,
            "discordant_fraction_by_split": [
                float(selected_observations[index] / max(1, total_observations[index]))
                for index in range(2)
            ],
        },
        "storage": {
            "point_id_bytes": int(packed_ids.nbytes),
            "feature_bytes_fp16": int(packed_features.nbytes),
            "reliability_bytes": int(packed_reliability.nbytes),
            "total_semantic_bytes": storage_bytes,
        },
        "source": {
            "base_consensus": base_path,
            "view_cache_dir": cache_dir,
            "view_feature_source": cache_manifest.get("feature_dir"),
            "paper_inspiration": [
                "LaGa adaptive multi-view descriptors",
                "CCL-LGS contrastive semantic conflict resolution",
                "OpenGaFF sparse codebook attention",
            ],
            "leakage_control": "training-view observations and split reliability only",
        },
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
