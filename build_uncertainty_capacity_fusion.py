#!/usr/bin/env python
"""Allocate per-Gaussian semantic capacity from unlabeled multiview evidence."""

import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from build_split_consistency_fusion import load_consensus, split_reliability


def normalized_disagreement(first, second):
    return (1.0 - F.cosine_similarity(first.float(), second.float(), dim=-1)).clamp(0.0, 2.0)


def select_sparse_fine_points(score, eligible, fraction, population_count=None):
    selected = torch.zeros_like(eligible)
    eligible_ids = torch.nonzero(eligible, as_tuple=False).flatten()
    if eligible_ids.numel() == 0 or fraction <= 0.0:
        return selected, None
    if population_count is None:
        population_count = eligible_ids.numel()
    count = min(eligible_ids.numel(), max(1, int(round(float(fraction) * population_count))))
    values = score[eligible_ids]
    chosen_local = torch.topk(values, count, sorted=False).indices
    selected[eligible_ids[chosen_local]] = True
    threshold = float(values.topk(count).values.min())
    return selected, threshold


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--aux_consensus", required=True)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--variant", choices=("a7.0", "a7.1", "a7.2", "a7.3"), required=True
    )
    parser.add_argument("--max_aux_weight", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--fallback_reliability", type=float, default=0.50)
    parser.add_argument("--fallback_margin", type=float, default=0.03)
    parser.add_argument("--fallback_ambiguous_ceiling", type=float, default=0.75)
    parser.add_argument("--fine_min_reliability", type=float, default=0.60)
    parser.add_argument("--fine_min_disagreement", type=float, default=0.10)
    parser.add_argument("--fine_fraction", type=float, default=0.15)
    parser.add_argument("--fine_weight", type=float, default=0.50)
    parser.add_argument(
        "--fine_score_mode",
        choices=("stability", "margin", "support_rarity", "margin_rarity"),
        default="stability",
    )
    parser.add_argument("--stability_floor", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    if args.max_aux_weight < 0.0 or args.temperature <= 0.0 or args.fine_weight < 0.0:
        raise ValueError("Fusion weights must be non-negative and temperature positive")
    if not 0.0 <= args.fine_fraction <= 1.0:
        raise ValueError("fine_fraction must be in [0, 1]")

    base = load_consensus(args.base_consensus)
    aux = load_consensus(args.aux_consensus)
    fine = load_consensus(args.fine_consensus)
    shape = base["initial_features"].shape
    if aux["initial_features"].shape != shape or fine["initial_features"].shape != shape:
        raise ValueError("All consensus feature tables must have identical shapes")

    base_rel, base_split = split_reliability(
        base["split_initial_features"], base["split_weights"], args.stability_floor
    )
    aux_rel, aux_split = split_reliability(
        aux["split_initial_features"], aux["split_weights"], args.stability_floor
    )
    fine_rel, fine_split = split_reliability(
        fine["split_initial_features"], fine["split_weights"], args.stability_floor
    )
    base_valid = base["total_weights"] > 0
    aux_valid = aux["total_weights"] > 0
    fine_valid = fine["total_weights"] > 0
    valid = base_valid | aux_valid

    gate = torch.sigmoid((aux_rel - base_rel) / args.temperature)
    gate = torch.where(aux_split, gate, torch.zeros_like(gate))
    gate = torch.where(~base_split & aux_split, torch.ones_like(gate), gate)
    reliability_peak = torch.maximum(base_rel, aux_rel)
    reliability_margin = (aux_rel - base_rel).abs()
    fallback = base_valid & (
        (~aux_split)
        | (reliability_peak < args.fallback_reliability)
        | (
            (reliability_margin < args.fallback_margin)
            & (reliability_peak < args.fallback_ambiguous_ceiling)
        )
    )
    if args.variant not in {"a7.1", "a7.3"}:
        fallback.zero_()
    gate = torch.where(fallback, torch.zeros_like(gate), gate)

    base_features = base["initial_features"].float()
    aux_features = aux["initial_features"].float()
    fine_features = fine["initial_features"].float()
    fused = base_features + (args.max_aux_weight * gate)[:, None] * aux_features
    fused[~base_valid & aux_valid] = aux_features[~base_valid & aux_valid]

    disagreement = normalized_disagreement(fine_features, aux_features)
    fine_score = fine_rel * disagreement
    fine_eligible = valid & fine_valid & fine_split & aux_valid
    fine_eligible &= fine_rel >= args.fine_min_reliability
    fine_eligible &= disagreement >= args.fine_min_disagreement
    margin_factor = torch.sigmoid((fine_rel - aux_rel) / args.temperature)
    support_log = torch.log1p(fine["total_weights"].float())
    eligible_support = support_log[fine_eligible]
    support_center = eligible_support.median() if eligible_support.numel() else 0.0
    rarity_factor = torch.sigmoid(support_center - support_log)
    if args.fine_score_mode in {"margin", "margin_rarity"}:
        fine_score *= margin_factor
    if args.fine_score_mode in {"support_rarity", "margin_rarity"}:
        fine_score *= rarity_factor
    fine_selected = torch.zeros_like(valid)
    fine_threshold = None
    if args.variant in {"a7.2", "a7.3"}:
        fine_selected, fine_threshold = select_sparse_fine_points(
            fine_score, fine_eligible, args.fine_fraction, int(valid.sum())
        )
        fused[fine_selected] += args.fine_weight * fine_features[fine_selected]

    fused[~valid] = 0.0
    fused[valid] = F.normalize(fused[valid], dim=-1)
    capacity = torch.zeros(fused.shape[0], dtype=torch.uint8)
    capacity[valid] = 2
    capacity[fine_selected] = 3

    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    torch.save(
        {
            "initial_features": fused.to(torch.float16),
            "total_weights": valid.float(),
            "semantic_capacity": capacity,
            "fusion_gate": gate.to(torch.float16),
            "fallback_mask": fallback,
            "fine_mask": fine_selected,
        },
        output,
    )
    valid_count = max(1, int(valid.sum()))
    summary = {
        "variant": args.variant,
        "base_consensus": os.path.abspath(args.base_consensus),
        "aux_consensus": os.path.abspath(args.aux_consensus),
        "fine_consensus": os.path.abspath(args.fine_consensus),
        "output": output,
        "num_gaussians": int(fused.shape[0]),
        "valid_fraction": float(valid.float().mean()),
        "fallback_fraction_of_valid": float(fallback.sum() / valid_count),
        "fine_eligible_fraction_of_valid": float(fine_eligible.sum() / valid_count),
        "fine_selected_fraction_of_valid": float(fine_selected.sum() / valid_count),
        "requested_average_ids": float(capacity[valid].float().mean()),
        "fine_score_threshold": fine_threshold,
        "mean_gate": float(gate[valid].mean()),
        "mean_base_reliability": float(base_rel[base_split].mean()),
        "mean_aux_reliability": float(aux_rel[aux_split].mean()),
        "mean_fine_reliability": float(fine_rel[fine_split].mean()),
        "mean_l1_l2_disagreement": float(disagreement[fine_eligible].mean())
        if fine_eligible.any()
        else 0.0,
        "args": vars(args),
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
