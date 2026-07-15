#!/usr/bin/env python
"""Write a fixed normalized interpolation of two continuous semantic consensuses."""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F


@torch.no_grad()
def blend_consensus_features(base_features, candidate_features, weight, chunk_size=65536):
    if base_features.shape != candidate_features.shape or base_features.ndim != 2:
        raise ValueError("Base and candidate features must have matching [N, D] shapes")
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must be in [0, 1]")
    output = torch.empty_like(base_features, dtype=torch.float16)
    for start in range(0, base_features.shape[0], chunk_size):
        end = min(start + chunk_size, base_features.shape[0])
        base = F.normalize(base_features[start:end].float(), dim=-1)
        candidate = F.normalize(candidate_features[start:end].float(), dim=-1)
        blended = F.normalize((1.0 - weight) * base + weight * candidate, dim=-1)
        output[start:end].copy_(blended.half())
    return output


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--candidate_weight", type=float, required=True)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output_path = os.path.abspath(args.output)
    if os.path.exists(output_path) and not args.force:
        print(f"Reuse existing blended consensus: {output_path}")
        return

    base_path = os.path.abspath(args.base_consensus)
    candidate_path = os.path.abspath(args.candidate_consensus)
    base = torch.load(base_path, map_location="cpu")
    candidate = torch.load(candidate_path, map_location="cpu")
    base_features = base["initial_features"]
    candidate_features = candidate["initial_features"]
    if base["total_weights"].shape != candidate["total_weights"].shape:
        raise ValueError("Base and candidate support tables must match")
    features = blend_consensus_features(
        base_features,
        candidate_features,
        args.candidate_weight,
        args.chunk_size,
    )
    support = base["total_weights"].float().cpu()
    features[support <= 0.0] = 0.0
    payload = {
        "initial_features": features,
        "total_weights": support,
        "metadata": {
            "representation": "normalized_semantic_consensus_blend",
            "base_consensus": base_path,
            "candidate_consensus": candidate_path,
            "candidate_weight": float(args.candidate_weight),
            "semantic_opacity": "discarded_after_training",
        },
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = output_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, output_path)
    print(
        json.dumps(
            {
                "output": output_path,
                "shape": list(features.shape),
                "candidate_weight": args.candidate_weight,
                "num_valid_gaussians": int((support > 0.0).sum()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
