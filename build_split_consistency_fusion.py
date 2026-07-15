#!/usr/bin/env python
"""Fuse two semantic consensuses using unlabeled cross-view split reliability."""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F


def split_reliability(split_features, split_weights, stability_floor=0.0):
    if split_features.ndim != 3 or split_features.shape[0] != 2:
        raise ValueError("Expected split features with shape [2, N, D]")
    if split_weights.shape != split_features.shape[:2]:
        raise ValueError("Split weights must have shape [2, N]")
    if not -1.0 <= stability_floor < 1.0:
        raise ValueError("stability_floor must be in [-1, 1)")
    supported = (split_weights[0] > 0) & (split_weights[1] > 0)
    stability = torch.zeros(split_features.shape[1], dtype=torch.float32)
    if supported.any():
        cosine = F.cosine_similarity(
            split_features[0, supported].float(),
            split_features[1, supported].float(),
            dim=-1,
        )
        stability[supported] = (
            (cosine - stability_floor) / (1.0 - stability_floor)
        ).clamp(0.0, 1.0)
    weight_sum = split_weights.sum(dim=0).clamp_min(1e-8)
    balance = (2.0 * split_weights.min(dim=0).values / weight_sum).clamp(0.0, 1.0)
    return stability * balance.sqrt(), supported


def load_consensus(path):
    payload = torch.load(path, map_location="cpu")
    required = {"initial_features", "total_weights", "split_initial_features", "split_weights"}
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Consensus is missing split fields: {sorted(missing)}")
    result = {name: payload[name].detach().cpu() for name in required}
    del payload
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--aux_consensus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max_aux_weight", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--stability_floor", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    if args.max_aux_weight < 0.0 or args.temperature <= 0.0:
        raise ValueError("Fusion weight must be non-negative and temperature positive")

    base = load_consensus(args.base_consensus)
    aux = load_consensus(args.aux_consensus)
    if base["initial_features"].shape != aux["initial_features"].shape:
        raise ValueError("Consensus feature tables must have identical shapes")

    base_reliability, base_split_valid = split_reliability(
        base["split_initial_features"], base["split_weights"], args.stability_floor
    )
    aux_reliability, aux_split_valid = split_reliability(
        aux["split_initial_features"], aux["split_weights"], args.stability_floor
    )
    base_valid = base["total_weights"] > 0
    aux_valid = aux["total_weights"] > 0
    gate = torch.sigmoid((aux_reliability - base_reliability) / args.temperature)
    gate = torch.where(aux_split_valid, gate, torch.zeros_like(gate))
    gate = torch.where(~base_split_valid & aux_split_valid, torch.ones_like(gate), gate)
    aux_weight = args.max_aux_weight * gate

    base_features = base["initial_features"].float()
    aux_features = aux["initial_features"].float()
    fused = base_features + aux_weight[:, None] * aux_features
    fused_valid = base_valid | aux_valid
    fused[~base_valid & aux_valid] = aux_features[~base_valid & aux_valid]
    fused[~fused_valid] = 0.0
    fused[fused_valid] = F.normalize(fused[fused_valid], dim=-1)
    total_weights = fused_valid.float()

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "initial_features": fused.to(torch.float16),
            "total_weights": total_weights,
            "fusion_gate": gate.to(torch.float16),
        },
        output,
    )
    summary = {
        "base_consensus": os.path.abspath(args.base_consensus),
        "aux_consensus": os.path.abspath(args.aux_consensus),
        "output": output,
        "num_gaussians": int(fused.shape[0]),
        "valid_fraction": float(fused_valid.float().mean()),
        "max_aux_weight": float(args.max_aux_weight),
        "temperature": float(args.temperature),
        "stability_floor": float(args.stability_floor),
        "mean_gate": float(gate[fused_valid].mean()),
        "mean_base_reliability": float(base_reliability[base_split_valid].mean()),
        "mean_aux_reliability": float(aux_reliability[aux_split_valid].mean()),
        "aux_preferred_fraction": float((gate[fused_valid] > 0.5).float().mean()),
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
