#!/usr/bin/env python
"""Export post-hoc A28 readout ablations without retraining the semantic experts."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F


MODES = ("raw_convex", "raw_top2", "old_anchored")


def normalize(value):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def mix_raw_experts(raw, weights, valid, mode, auxiliary_scale=1.0):
    if raw.ndim != 3 or raw.shape[1] != 3:
        raise ValueError("Raw experts must have shape [B, 3, D]")
    if weights.shape != valid.shape or weights.shape != raw.shape[:2]:
        raise ValueError("Weights and expert validity must have shape [B, 3]")
    effective = weights * valid.float()
    any_valid = valid.any(dim=1)
    convex = effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)

    if mode == "raw_convex":
        mixed = (convex[:, :, None] * raw).sum(dim=1)
    elif mode == "raw_top2":
        top_count = min(2, weights.shape[1])
        top_ids = torch.topk(
            torch.where(valid, weights, torch.full_like(weights, -1.0)),
            k=top_count,
            dim=1,
        ).indices
        keep = torch.zeros_like(valid)
        keep.scatter_(1, top_ids, True)
        effective = effective * keep.float()
        effective = effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
        mixed = (effective[:, :, None] * raw).sum(dim=1)
    elif mode == "old_anchored":
        auxiliary = (
            effective[:, 1, None] * raw[:, 1]
            + effective[:, 2, None] * raw[:, 2]
        )
        anchored = raw[:, 0] + auxiliary_scale * auxiliary
        fallback = (convex[:, :, None] * raw).sum(dim=1)
        mixed = torch.where(valid[:, 0, None], anchored, fallback)
    else:
        raise ValueError(f"Unknown readout ablation: {mode}")
    return torch.where(any_valid[:, None], normalize(mixed), torch.zeros_like(mixed))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old_consensus", required=True)
    parser.add_argument("--l2_consensus", required=True)
    parser.add_argument("--l3_consensus", required=True)
    parser.add_argument("--expert_weights", required=True)
    parser.add_argument("--expert_valid", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--auxiliary_scale", type=float, default=1.0)
    args = parser.parse_args(sys.argv[1:])
    if args.chunk_size <= 0 or args.auxiliary_scale < 0.0:
        raise ValueError("Chunk size must be positive and auxiliary scale non-negative")

    paths = (args.old_consensus, args.l2_consensus, args.l3_consensus)
    payloads = [torch.load(os.path.abspath(path), map_location="cpu") for path in paths]
    shapes = [tuple(payload["initial_features"].shape) for payload in payloads]
    if len(set(shapes)) != 1:
        raise ValueError(f"Expert shapes differ: {shapes}")
    count, feature_dim = shapes[0]
    weights = np.load(args.expert_weights, mmap_mode="r")
    valid = np.load(args.expert_valid, mmap_mode="r")
    if weights.shape != (count, 3) or valid.shape != (count, 3):
        raise ValueError("A28 gate tables do not match the expert consensuses")

    output_root = os.path.abspath(args.output_dir)
    os.makedirs(output_root, exist_ok=True)
    device = torch.device(args.device)
    outputs = {
        mode: torch.zeros((count, feature_dim), dtype=torch.float16) for mode in MODES
    }
    cosine_sums = {
        mode: torch.zeros(3, dtype=torch.float64) for mode in MODES
    }
    expert_counts = torch.zeros(3, dtype=torch.int64)
    valid_count = 0
    for start in range(0, count, args.chunk_size):
        end = min(start + args.chunk_size, count)
        raw = torch.stack(
            [payload["initial_features"][start:end].float() for payload in payloads],
            dim=1,
        ).to(device)
        raw = normalize(raw)
        chunk_weights = torch.from_numpy(
            np.asarray(weights[start:end], dtype=np.float32)
        ).to(device)
        chunk_valid = torch.from_numpy(
            np.asarray(valid[start:end], dtype=np.bool_)
        ).to(device)
        any_valid = chunk_valid.any(dim=1)
        valid_count += int(any_valid.sum())
        expert_counts += chunk_valid.sum(dim=0).cpu()
        for mode in MODES:
            mixed = mix_raw_experts(
                raw,
                chunk_weights,
                chunk_valid,
                mode,
                args.auxiliary_scale,
            )
            outputs[mode][start:end] = mixed.cpu().to(torch.float16)
            for expert_id in range(3):
                supported = any_valid & chunk_valid[:, expert_id]
                if supported.any():
                    cosine_sums[mode][expert_id] += F.cosine_similarity(
                        mixed[supported], raw[supported, expert_id], dim=-1
                    ).double().sum().cpu()

    summary = {
        "purpose": "posthoc A28 failure attribution only",
        "must_not_be_used_for_training_or_model_selection": True,
        "sources": [os.path.abspath(path) for path in paths],
        "expert_weights": os.path.abspath(args.expert_weights),
        "expert_valid": os.path.abspath(args.expert_valid),
        "num_gaussians": count,
        "feature_dim": feature_dim,
        "valid_fraction": valid_count / max(1, count),
        "modes": {},
        "args": vars(args),
    }
    total_weights = torch.from_numpy(np.asarray(valid.any(axis=1), dtype=np.float32))
    for mode, features in outputs.items():
        mode_dir = os.path.join(output_root, mode)
        os.makedirs(mode_dir, exist_ok=True)
        torch.save(
            {"initial_features": features, "total_weights": total_weights},
            os.path.join(mode_dir, "consensus.pt"),
        )
        mode_manifest = {
            "representation": "continuous_semantic_moe_readout_ablation",
            "mode": mode,
            "num_gaussians": count,
            "feature_dim": feature_dim,
            "valid_fraction": valid_count / max(1, count),
            "mean_cosine_to_old_l2_l3": [
                float(value / max(1, int(expert_counts[expert_id])))
                for expert_id, value in enumerate(cosine_sums[mode])
            ],
            "auxiliary_scale": args.auxiliary_scale,
            "posthoc_diagnostic": True,
        }
        with open(os.path.join(mode_dir, "manifest.json"), "w") as output:
            json.dump(mode_manifest, output, indent=2)
        summary["modes"][mode] = mode_manifest
    with open(os.path.join(output_root, "summary.json"), "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
