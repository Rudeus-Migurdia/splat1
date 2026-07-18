#!/usr/bin/env python
"""Blend a robust semantic candidate with its baseline using split reliability."""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from build_split_consistency_fusion import load_consensus, split_reliability


def blend_with_split_reliability(
    baseline_features,
    candidate_features,
    baseline_valid,
    candidate_valid,
    old_reliability,
    old_valid,
    l2_reliability,
    l2_valid,
    scale_gate,
):
    if baseline_features.shape != candidate_features.shape:
        raise ValueError("Baseline and candidate feature tables must match")
    num_gaussians = baseline_features.shape[0]
    vectors = (
        baseline_valid,
        candidate_valid,
        old_reliability,
        old_valid,
        l2_reliability,
        l2_valid,
        scale_gate,
    )
    if any(value.shape != (num_gaussians,) for value in vectors):
        raise ValueError("Reliability vectors must match the Gaussian count")

    scale_gate = scale_gate.float().clamp(0.0, 1.0)
    reliability = (
        (1.0 - scale_gate) * old_reliability.float()
        + scale_gate * l2_reliability.float()
    )
    reliability = torch.where(old_valid & ~l2_valid, old_reliability, reliability)
    reliability = torch.where(l2_valid & ~old_valid, l2_reliability, reliability)
    reliability = torch.where(old_valid | l2_valid, reliability, torch.zeros_like(reliability))
    reliability = reliability.clamp(0.0, 1.0)

    baseline = baseline_features.float()
    candidate = candidate_features.float()
    output = F.normalize(
        (1.0 - reliability[:, None]) * baseline
        + reliability[:, None] * candidate,
        dim=-1,
    )
    output_valid = baseline_valid | candidate_valid
    output[~baseline_valid & candidate_valid] = F.normalize(
        candidate[~baseline_valid & candidate_valid], dim=-1
    )
    output[~output_valid] = 0.0
    return output, output_valid, reliability


def load_feature_payload(path):
    payload = torch.load(os.path.abspath(path), map_location="cpu")
    features = payload.get("initial_features")
    weights = payload.get("total_weights")
    if features is None or weights is None:
        raise ValueError(f"Consensus is missing features or weights: {path}")
    return payload, features.detach().cpu(), weights.detach().cpu().reshape(-1) > 0.0


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--old_split_consensus", required=True)
    parser.add_argument("--l2_split_consensus", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:])

    baseline_payload, baseline_features, baseline_valid = load_feature_payload(
        args.baseline
    )
    _, candidate_features, candidate_valid = load_feature_payload(args.candidate)
    old = load_consensus(args.old_split_consensus)
    l2 = load_consensus(args.l2_split_consensus)
    old_reliability, old_split_valid = split_reliability(
        old["split_initial_features"], old["split_weights"]
    )
    l2_reliability, l2_split_valid = split_reliability(
        l2["split_initial_features"], l2["split_weights"]
    )
    scale_gate = baseline_payload.get("fusion_gate")
    if scale_gate is None:
        raise ValueError("Baseline fusion must contain its old/L2 scale gate")
    output_features, output_valid, fallback_gate = blend_with_split_reliability(
        baseline_features,
        candidate_features,
        baseline_valid,
        candidate_valid,
        old_reliability,
        old_split_valid,
        l2_reliability,
        l2_split_valid,
        scale_gate.detach().cpu().reshape(-1),
    )

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "initial_features": output_features.to(torch.float16),
            "total_weights": output_valid.float(),
            "fallback_gate": fallback_gate.to(torch.float16),
        },
        output,
    )
    supported = output_valid & (old_split_valid | l2_split_valid)
    summary = {
        "baseline": os.path.abspath(args.baseline),
        "candidate": os.path.abspath(args.candidate),
        "old_split_consensus": os.path.abspath(args.old_split_consensus),
        "l2_split_consensus": os.path.abspath(args.l2_split_consensus),
        "output": output,
        "valid_fraction": float(output_valid.float().mean()),
        "mean_candidate_gate": float(fallback_gate[supported].mean()),
        "candidate_gate_quantiles": {
            str(value): float(torch.quantile(fallback_gate[supported], value))
            for value in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
