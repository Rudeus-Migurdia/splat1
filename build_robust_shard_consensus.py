#!/usr/bin/env python
"""Merge interleaved view shards with a mean or robust geometric median."""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from tqdm import tqdm


def weighted_geometric_median(features, weights, iterations=8, eps=1e-4):
    """Batched Weiszfeld updates for features shaped [shards, points, dim]."""
    if features.ndim != 3 or weights.shape != features.shape[:2]:
        raise ValueError("Expected features [S, N, D] and weights [S, N]")
    if iterations <= 0 or eps <= 0.0:
        raise ValueError("iterations and eps must be positive")
    weights = weights.float().clamp_min(0.0)
    total = weights.sum(dim=0)
    center = (features.float() * weights.unsqueeze(-1)).sum(dim=0)
    center = center / total.clamp_min(eps).unsqueeze(-1)
    center = F.normalize(center, dim=-1)
    center[total <= 0.0] = 0.0
    for _ in range(iterations):
        distances = (features.float() - center.unsqueeze(0)).norm(dim=-1)
        adjusted = weights / distances.clamp_min(eps)
        adjusted_total = adjusted.sum(dim=0)
        next_center = (features.float() * adjusted.unsqueeze(-1)).sum(dim=0)
        next_center = next_center / adjusted_total.clamp_min(eps).unsqueeze(-1)
        next_center = F.normalize(next_center, dim=-1)
        next_center[total <= 0.0] = 0.0
        center = next_center
    return center


def load_payload(path):
    try:
        payload = torch.load(path, map_location="cpu", mmap=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    required = {"initial_features", "total_weights", "mean_feature_norm"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Shard {path} is missing compact fields: {sorted(missing)}")
    return payload


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--method", choices=("weighted_mean", "geometric_median"), required=True)
    parser.add_argument("--weight_power", type=float, default=0.5)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=8192)
    args = parser.parse_args(sys.argv[1:])
    if len(args.shards) < 3:
        raise ValueError("Robust view aggregation requires at least three shards")
    if args.weight_power <= 0.0 or args.chunk_size <= 0 or args.iterations <= 0:
        raise ValueError("weight power, chunk size, and iterations must be positive")

    shard_paths = [os.path.abspath(path) for path in args.shards]
    payloads = [load_payload(path) for path in shard_paths]
    shape = payloads[0]["initial_features"].shape
    if any(payload["initial_features"].shape != shape for payload in payloads):
        raise ValueError("All view shards must have the same feature shape")
    num_gaussians, feature_dim = shape
    for payload in payloads:
        if payload["total_weights"].shape != (num_gaussians,):
            raise ValueError("Shard total weights do not match the Gaussian count")
        if payload["mean_feature_norm"].shape != (num_gaussians,):
            raise ValueError("Shard mean norms do not match the Gaussian count")

    output_features = torch.empty((num_gaussians, feature_dim), dtype=torch.float16)
    total_weights = torch.zeros(num_gaussians, dtype=torch.float32)
    split_features = torch.zeros((2, num_gaussians, feature_dim), dtype=torch.float16)
    split_weights = torch.zeros((2, num_gaussians), dtype=torch.float32)
    dispersion_sum = 0.0
    dispersion_weight = 0.0
    multi_shard_points = 0

    for start in tqdm(range(0, num_gaussians, args.chunk_size), desc=args.method):
        end = min(start + args.chunk_size, num_gaussians)
        features = torch.stack(
            [payload["initial_features"][start:end].float() for payload in payloads], dim=0
        )
        raw_weights = torch.stack(
            [payload["total_weights"][start:end].float() for payload in payloads], dim=0
        )
        mean_norms = torch.stack(
            [payload["mean_feature_norm"][start:end].float() for payload in payloads], dim=0
        )
        evidence = raw_weights * mean_norms.clamp_min(0.0)
        supported = evidence > 0.0
        total_weights[start:end] = raw_weights.sum(dim=0)

        exact_sum = (features * evidence.unsqueeze(-1)).sum(dim=0)
        mean_center = F.normalize(exact_sum, dim=-1)
        mean_center[evidence.sum(dim=0) <= 0.0] = 0.0
        if args.method == "weighted_mean":
            center = mean_center
        else:
            robust_weights = evidence.pow(args.weight_power)
            center = weighted_geometric_median(
                features, robust_weights, iterations=args.iterations
            )
        output_features[start:end].copy_(center.to(torch.float16))

        for shard_index in range(len(payloads)):
            split_index = shard_index % 2
            split_weights[split_index, start:end] += raw_weights[shard_index]
        for split_index in range(2):
            selected = list(range(split_index, len(payloads), 2))
            split_sum = (
                features[selected] * evidence[selected].unsqueeze(-1)
            ).sum(dim=0)
            split_center = F.normalize(split_sum, dim=-1)
            split_center[evidence[selected].sum(dim=0) <= 0.0] = 0.0
            split_features[split_index, start:end].copy_(split_center.to(torch.float16))

        cosine = (features * center.unsqueeze(0)).sum(dim=-1).clamp(-1.0, 1.0)
        dispersion_sum += float(((1.0 - cosine) * supported).sum())
        dispersion_weight += float(supported.sum())
        multi_shard_points += int((supported.sum(dim=0) >= 3).sum())

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "initial_features": output_features,
            "total_weights": total_weights,
            "split_initial_features": split_features,
            "split_weights": split_weights,
        },
        output,
    )
    valid = total_weights > 0.0
    summary = {
        "method": args.method,
        "shards": shard_paths,
        "output": output,
        "num_shards": len(payloads),
        "num_gaussians": num_gaussians,
        "feature_dim": feature_dim,
        "valid_fraction": float(valid.float().mean()),
        "three_shard_support_fraction": float(multi_shard_points / max(1, num_gaussians)),
        "mean_angular_dispersion": float(dispersion_sum / max(1.0, dispersion_weight)),
        "weight_power": float(args.weight_power),
        "iterations": int(args.iterations),
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
