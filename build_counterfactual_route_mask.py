#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


def split_counterfactual_advantage(base, candidate, split_targets):
    base = F.normalize(base.float(), dim=-1)
    candidate = F.normalize(candidate.float(), dim=-1)
    split_targets = F.normalize(split_targets.float(), dim=-1)
    base_similarity = (split_targets * base.unsqueeze(0)).sum(dim=-1)
    candidate_similarity = (split_targets * candidate.unsqueeze(0)).sum(dim=-1)
    return (candidate_similarity - base_similarity).amin(dim=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--split_consensus", required=True)
    parser.add_argument("--min_advantage", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    base = torch.load(os.path.abspath(args.base_consensus), map_location="cpu")
    candidate = torch.load(os.path.abspath(args.candidate_consensus), map_location="cpu")
    split = torch.load(os.path.abspath(args.split_consensus), map_location="cpu")
    base_features = base["initial_features"]
    candidate_features = candidate["initial_features"]
    split_features = split["split_initial_features"]
    if base_features.shape != candidate_features.shape:
        raise ValueError("Base and candidate consensus shapes must match")
    if split_features.shape != (2,) + base_features.shape:
        raise ValueError("Split consensus must contain two feature tables")

    num_gaussians = base_features.shape[0]
    mask = torch.zeros(num_gaussians, dtype=torch.bool)
    valid_advantages = []
    for start in range(0, num_gaussians, args.chunk_size):
        end = min(start + args.chunk_size, num_gaussians)
        valid = (
            (base["total_weights"][start:end] > 0)
            & (candidate["total_weights"][start:end] > 0)
            & (split["split_weights"][:, start:end] > 0).all(dim=0)
        )
        advantage = split_counterfactual_advantage(
            base_features[start:end],
            candidate_features[start:end],
            split_features[:, start:end],
        )
        mask[start:end] = valid & (advantage > args.min_advantage)
        valid_advantages.append(advantage[valid])

    advantages = torch.cat(valid_advantages)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, mask.numpy())
    diagnostics = {
        "base_consensus": os.path.abspath(args.base_consensus),
        "candidate_consensus": os.path.abspath(args.candidate_consensus),
        "split_consensus": os.path.abspath(args.split_consensus),
        "min_advantage": args.min_advantage,
        "num_gaussians": num_gaussians,
        "num_both_split_valid": int(advantages.numel()),
        "num_candidate_points": int(mask.sum()),
        "candidate_fraction": float(mask.float().mean()),
        "advantage_quantiles": {
            str(q): float(torch.quantile(advantages, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
    }
    with open(os.path.splitext(output)[0] + ".json", "w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
