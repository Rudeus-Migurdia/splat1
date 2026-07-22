#!/usr/bin/env python
"""Audit whether multiscale SAM evidence needs set-valued 3D relations."""

import hashlib
import json
import os
import time
from argparse import ArgumentParser

import numpy as np


LEVEL_COUNT = 4


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return {str(q): 0.0 for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def dominant_gaussian_segments_from_labels(
    point_ids,
    point_weights,
    segment_ids,
    num_gaussians,
    minimum_fraction,
):
    """Assign each Gaussian its contribution-weighted dominant segment."""
    point_ids = np.asarray(point_ids, dtype=np.int64)
    point_weights = np.asarray(point_weights, dtype=np.float32)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    if point_ids.ndim != 2 or point_weights.shape != point_ids.shape:
        raise ValueError("Point IDs and weights must have matching [P, K] shapes")
    if segment_ids.shape != (point_ids.shape[0],):
        raise ValueError("Segment IDs must have one value per sampled pixel")

    repeated_segments = np.repeat(segment_ids, point_ids.shape[1])
    points = point_ids.reshape(-1)
    weights = point_weights.reshape(-1)
    valid = (
        (points >= 0)
        & (points < num_gaussians)
        & (repeated_segments >= 0)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    segments = np.full(num_gaussians, -1, dtype=np.int32)
    confidence = np.zeros(num_gaussians, dtype=np.float32)
    visibility = np.zeros(num_gaussians, dtype=np.float32)
    if not valid.any():
        return segments, confidence, visibility

    points = points[valid]
    weights = weights[valid]
    repeated_segments = repeated_segments[valid]
    segment_count = int(repeated_segments.max()) + 1
    pair_keys = points * segment_count + repeated_segments
    unique_pairs, inverse = np.unique(pair_keys, return_inverse=True)
    pair_mass = np.bincount(inverse, weights=weights).astype(np.float32)
    pair_points = unique_pairs // segment_count
    pair_segments = unique_pairs % segment_count

    total_mass = np.bincount(
        pair_points, weights=pair_mass, minlength=num_gaussians
    ).astype(np.float32)
    order = np.lexsort((-pair_mass, pair_points))
    ordered_points = pair_points[order]
    first = np.r_[True, ordered_points[1:] != ordered_points[:-1]]
    winners = order[first]
    winner_points = pair_points[winners]
    winner_mass = pair_mass[winners]
    winner_fraction = winner_mass / np.maximum(total_mass[winner_points], 1e-12)
    accepted = winner_fraction >= minimum_fraction
    winner_points = winner_points[accepted]
    winner_mass = winner_mass[accepted]
    winner_fraction = winner_fraction[accepted]
    if not winner_points.size:
        return segments, confidence, visibility

    segments[winner_points] = pair_segments[winners][accepted].astype(np.int32)
    confidence[winner_points] = winner_fraction.astype(np.float32)
    view_scale = float(np.median(winner_mass))
    visibility[winner_points] = winner_mass / np.maximum(
        winner_mass + max(view_scale, 1e-8), 1e-8
    )
    return segments, confidence, visibility


def _relation_probabilities(positive_mass, negative_mass, beta_prior):
    return (positive_mass + beta_prior) / (
        positive_mass + negative_mass + 2.0 * beta_prior
    )


def evaluate_relation_models(
    positive_mass,
    negative_mass,
    observation_count,
    minimum_split_views,
    beta_prior=1.0,
    positive_threshold=0.65,
    negative_threshold=0.35,
):
    """Compare fixed and level-conditioned Bernoulli relations by split NLL."""
    if not (
        positive_mass.shape == negative_mass.shape == observation_count.shape
        and positive_mass.ndim == 4
        and positive_mass.shape[:2] == (2, LEVEL_COUNT)
    ):
        raise ValueError("Relation evidence must have shape [2, 4, N, K]")
    if beta_prior <= 0.0:
        raise ValueError("Beta prior must be positive")

    valid = observation_count.min(axis=0) >= minimum_split_views
    multilevel_edges = valid.sum(axis=0) >= 2
    eval_mask = valid & multilevel_edges[None]
    fixed_nll = 0.0
    set_nll = 0.0
    evaluation_mass = 0.0
    split_probabilities = []

    for train_split in (0, 1):
        eval_split = 1 - train_split
        train_positive = np.where(valid, positive_mass[train_split], 0.0)
        train_negative = np.where(valid, negative_mass[train_split], 0.0)
        fixed_probability = (
            train_positive.sum(axis=0) + beta_prior
        ) / (
            train_positive.sum(axis=0)
            + train_negative.sum(axis=0)
            + 2.0 * beta_prior
        )
        set_probability = _relation_probabilities(
            positive_mass[train_split], negative_mass[train_split], beta_prior
        )
        split_probabilities.append(set_probability.astype(np.float32))

        eval_positive = np.where(eval_mask, positive_mass[eval_split], 0.0)
        eval_negative = np.where(eval_mask, negative_mass[eval_split], 0.0)
        clipped_fixed = np.clip(fixed_probability, 1e-7, 1.0 - 1e-7)
        clipped_set = np.clip(set_probability, 1e-7, 1.0 - 1e-7)
        fixed_nll -= float(
            (
                eval_positive * np.log(clipped_fixed[None])
                + eval_negative * np.log1p(-clipped_fixed[None])
            ).sum(dtype=np.float64)
        )
        set_nll -= float(
            (
                eval_positive * np.log(clipped_set)
                + eval_negative * np.log1p(-clipped_set)
            ).sum(dtype=np.float64)
        )
        evaluation_mass += float(
            (eval_positive + eval_negative).sum(dtype=np.float64)
        )

    probabilities = np.stack(split_probabilities, axis=0)
    split_same = probabilities >= 0.5
    signature_agreement = float(
        (split_same[0][eval_mask] == split_same[1][eval_mask]).mean()
    ) if eval_mask.any() else 0.0
    stable_positive = (
        (probabilities[0] >= positive_threshold)
        & (probabilities[1] >= positive_threshold)
        & valid
    )
    stable_negative = (
        (probabilities[0] <= negative_threshold)
        & (probabilities[1] <= negative_threshold)
        & valid
    )
    stable_set_ambiguous = (
        stable_positive.any(axis=0)
        & stable_negative.any(axis=0)
        & multilevel_edges
    )
    relation_signature = np.zeros(valid.shape, dtype=np.int8)
    relation_signature[stable_positive] = 1
    relation_signature[stable_negative] = -1

    mean_fixed_nll = fixed_nll / max(evaluation_mass, 1e-12)
    mean_set_nll = set_nll / max(evaluation_mass, 1e-12)
    improvement = (fixed_nll - set_nll) / max(fixed_nll, 1e-12)
    return {
        "fixed_nll": fixed_nll,
        "set_nll": set_nll,
        "evaluation_mass": evaluation_mass,
        "mean_fixed_nll": mean_fixed_nll,
        "mean_set_nll": mean_set_nll,
        "relative_nll_improvement": improvement,
        "relation_signature_agreement": signature_agreement,
        "valid_level_edge_slots": int(eval_mask.sum()),
        "multilevel_directed_edges": int(multilevel_edges.sum()),
        "stable_set_ambiguous_directed_edges": int(stable_set_ambiguous.sum()),
        "stable_set_ambiguous_fraction": float(
            stable_set_ambiguous.sum() / max(int(multilevel_edges.sum()), 1)
        ),
        "valid": valid,
        "multilevel_edges": multilevel_edges,
        "stable_set_ambiguous": stable_set_ambiguous,
        "relation_signature": relation_signature,
        "probabilities": probabilities,
    }


def evaluate_train_selected_conflicts(
    positive_mass,
    negative_mass,
    observation_count,
    minimum_split_views,
    beta_prior=1.0,
    positive_threshold=0.65,
    negative_threshold=0.35,
):
    """Score multiscale conflicts selected only from the opposite view split."""
    valid = observation_count.min(axis=0) >= minimum_split_views
    multilevel_edges = valid.sum(axis=0) >= 2
    totals = {
        "fixed_nll": 0.0,
        "set_nll": 0.0,
        "fixed_positive_nll": 0.0,
        "set_positive_nll": 0.0,
        "fixed_negative_nll": 0.0,
        "set_negative_nll": 0.0,
        "positive_mass": 0.0,
        "negative_mass": 0.0,
    }
    selected_counts = []
    selected_masks = []
    for train_split in (0, 1):
        eval_split = 1 - train_split
        train_positive = np.where(valid, positive_mass[train_split], 0.0)
        train_negative = np.where(valid, negative_mass[train_split], 0.0)
        set_probability = _relation_probabilities(
            positive_mass[train_split], negative_mass[train_split], beta_prior
        )
        fixed_probability = (
            train_positive.sum(axis=0) + beta_prior
        ) / (
            train_positive.sum(axis=0)
            + train_negative.sum(axis=0)
            + 2.0 * beta_prior
        )
        selected = (
            (set_probability >= positive_threshold).any(axis=0)
            & (set_probability <= negative_threshold).any(axis=0)
            & multilevel_edges
        )
        selected_counts.append(int(selected.sum()))
        selected_masks.append(selected)
        eval_mask = valid & selected[None]
        eval_positive = np.where(eval_mask, positive_mass[eval_split], 0.0)
        eval_negative = np.where(eval_mask, negative_mass[eval_split], 0.0)
        clipped_fixed = np.clip(fixed_probability, 1e-7, 1.0 - 1e-7)
        clipped_set = np.clip(set_probability, 1e-7, 1.0 - 1e-7)
        fixed_positive_nll = -float(
            (eval_positive * np.log(clipped_fixed[None])).sum(dtype=np.float64)
        )
        fixed_negative_nll = -float(
            (eval_negative * np.log1p(-clipped_fixed[None])).sum(dtype=np.float64)
        )
        set_positive_nll = -float(
            (eval_positive * np.log(clipped_set)).sum(dtype=np.float64)
        )
        set_negative_nll = -float(
            (eval_negative * np.log1p(-clipped_set)).sum(dtype=np.float64)
        )
        totals["fixed_positive_nll"] += fixed_positive_nll
        totals["fixed_negative_nll"] += fixed_negative_nll
        totals["set_positive_nll"] += set_positive_nll
        totals["set_negative_nll"] += set_negative_nll
        totals["fixed_nll"] += fixed_positive_nll + fixed_negative_nll
        totals["set_nll"] += set_positive_nll + set_negative_nll
        totals["positive_mass"] += float(eval_positive.sum(dtype=np.float64))
        totals["negative_mass"] += float(eval_negative.sum(dtype=np.float64))

    evaluation_mass = totals["positive_mass"] + totals["negative_mass"]
    fixed_balanced_nll = 0.5 * (
        totals["fixed_positive_nll"] / max(totals["positive_mass"], 1e-12)
        + totals["fixed_negative_nll"] / max(totals["negative_mass"], 1e-12)
    )
    set_balanced_nll = 0.5 * (
        totals["set_positive_nll"] / max(totals["positive_mass"], 1e-12)
        + totals["set_negative_nll"] / max(totals["negative_mass"], 1e-12)
    )
    intersection = selected_masks[0] & selected_masks[1]
    union = selected_masks[0] | selected_masks[1]
    return {
        **totals,
        "evaluation_mass": evaluation_mass,
        "mean_fixed_nll": totals["fixed_nll"] / max(evaluation_mass, 1e-12),
        "mean_set_nll": totals["set_nll"] / max(evaluation_mass, 1e-12),
        "relative_nll_improvement": (
            (totals["fixed_nll"] - totals["set_nll"])
            / max(totals["fixed_nll"], 1e-12)
        ),
        "fixed_balanced_nll": fixed_balanced_nll,
        "set_balanced_nll": set_balanced_nll,
        "relative_balanced_nll_improvement": (
            (fixed_balanced_nll - set_balanced_nll)
            / max(fixed_balanced_nll, 1e-12)
        ),
        "selected_directed_edges_by_train_split": selected_counts,
        "selected_intersection_directed_edges": int(intersection.sum()),
        "selected_union_directed_edges": int(union.sum()),
        "selection_jaccard": float(
            intersection.sum() / max(int(union.sum()), 1)
        ),
    }


def make_gate_decision(metrics, minimum_nll_improvement, minimum_stability):
    checks = {
        "heldout_nll_improvement": (
            metrics["relative_nll_improvement"] >= minimum_nll_improvement
        ),
        "split_relation_stability": (
            metrics["relation_signature_agreement"] >= minimum_stability
        ),
        "has_multilevel_evidence": metrics["multilevel_directed_edges"] > 0,
    }
    return {
        "pass": bool(all(checks.values())),
        "decision": "PROCEED_TO_A46_1" if all(checks.values()) else "STOP_BEFORE_CODEBOOK_TRAINING",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": {
            "minimum_relative_nll_improvement": minimum_nll_improvement,
            "minimum_split_relation_stability": minimum_stability,
        },
    }


def _load_json(path):
    with open(path) as source:
        return json.load(source)


def main():
    import torch

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--memory_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--relation_graph_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--minimum_dominant_fraction", type=float, default=0.55)
    parser.add_argument("--minimum_split_views", type=int, default=3)
    parser.add_argument("--beta_prior", type=float, default=1.0)
    parser.add_argument("--positive_threshold", type=float, default=0.65)
    parser.add_argument("--negative_threshold", type=float, default=0.35)
    parser.add_argument("--minimum_nll_improvement", type=float, default=0.10)
    parser.add_argument("--minimum_split_stability", type=float, default=0.80)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--expected_memory_seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.minimum_dominant_fraction <= 1.0:
        raise ValueError("Dominant fraction must be in (0, 1]")
    if args.minimum_split_views <= 0 or args.chunk_size <= 0:
        raise ValueError("Minimum views and chunk size must be positive")
    if not 0.5 < args.positive_threshold < 1.0:
        raise ValueError("Positive threshold must be in (0.5, 1)")
    if not 0.0 < args.negative_threshold < 0.5:
        raise ValueError("Negative threshold must be in (0, 0.5)")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse multiscale set relation diagnostic: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    memory_dir = os.path.abspath(args.memory_dir)
    memory_manifest_path = os.path.join(memory_dir, "manifest.json")
    memory_manifest = _load_json(memory_manifest_path)
    if memory_manifest.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError("A46 requires an independent four-token memory")
    if int(memory_manifest.get("resident_slots_required", 0)) != LEVEL_COUNT:
        raise ValueError("A46 requires exactly four resident token slots")
    if int(memory_manifest["reproducibility"]["seed"]) != args.expected_memory_seed:
        raise ValueError("Memory seed does not match the fixed A46 seed")
    num_gaussians = int(memory_manifest["num_gaussians"])

    cache_dir = os.path.abspath(args.cache_dir)
    cache_manifest_path = os.path.join(cache_dir, "manifest.json")
    cache_manifest = _load_json(cache_manifest_path)
    if int(cache_manifest["num_gaussians"]) != num_gaussians:
        raise ValueError("View cache and resident memory Gaussian counts differ")
    if not cache_manifest.get("raw_contribution_weights"):
        raise ValueError("A46 requires raw T*alpha contribution weights")
    if int(cache_manifest.get("topk", 0)) < 45:
        raise ValueError("A46 requires at least top-45 contributors")
    entries = cache_manifest.get("views", [])
    if len(entries) < 2 * args.minimum_split_views:
        raise ValueError("Not enough cached views for odd/even validation")

    relation_graph_dir = os.path.abspath(args.relation_graph_dir)
    relation_manifest_path = os.path.join(relation_graph_dir, "manifest.json")
    relation_manifest = _load_json(relation_manifest_path)
    if int(relation_manifest["num_gaussians"]) != num_gaussians:
        raise ValueError("Relation graph and memory Gaussian counts differ")
    neighbors = np.load(
        os.path.join(relation_graph_dir, relation_manifest["neighbor_ids"]),
        mmap_mode="r",
    )
    if neighbors.ndim != 2 or neighbors.shape[0] != num_gaussians:
        raise ValueError("Neighbor graph must have shape [N, K]")
    neighbor_count = int(neighbors.shape[1])
    a39_relations = np.load(
        os.path.join(
            relation_graph_dir, relation_manifest["signed_relation_weights"]
        ),
        mmap_mode="r",
    )
    if a39_relations.shape != neighbors.shape:
        raise ValueError("A39 relation and neighbor arrays must match")

    feature_dir = os.path.abspath(args.feature_dir)
    feature_paths = []
    for entry in entries:
        feature_path = os.path.join(feature_dir, f"{entry['image_name']}_s.npy")
        if not os.path.isfile(feature_path):
            raise FileNotFoundError(feature_path)
        feature_paths.append(feature_path)

    shape = (2, LEVEL_COUNT, num_gaussians, neighbor_count)
    positive_mass = np.zeros(shape, dtype=np.float32)
    negative_mass = np.zeros(shape, dtype=np.float32)
    observation_count = np.zeros(shape, dtype=np.uint8)
    accepted_per_view_level = []

    for entry_index, (entry, feature_path) in enumerate(zip(entries, feature_paths)):
        payload = torch.load(
            os.path.join(cache_dir, entry["cache"]),
            map_location="cpu",
            weights_only=False,
        )
        point_ids = payload["point_ids"].numpy().astype(np.int64, copy=False)
        point_weights = payload["point_weights"].numpy().astype(np.float32, copy=False)
        sampled = payload["sampled_flat_indices"].numpy().astype(np.int64, copy=False)
        image_height = int(payload["image_height"])
        image_width = int(payload["image_width"])
        segment_maps = np.load(feature_path, mmap_mode="r")
        if segment_maps.shape != (LEVEL_COUNT, image_height, image_width):
            raise ValueError(
                f"Unexpected multiscale map shape for {entry['image_name']}: "
                f"{segment_maps.shape}"
            )
        split = entry_index % 2
        accepted_levels = []
        for level in range(LEVEL_COUNT):
            sampled_segments = np.asarray(segment_maps[level]).reshape(-1)[sampled]
            segments, confidence, visibility = dominant_gaussian_segments_from_labels(
                point_ids,
                point_weights,
                sampled_segments,
                num_gaussians,
                args.minimum_dominant_fraction,
            )
            accepted_levels.append(int((segments >= 0).sum()))
            for start in range(0, num_gaussians, args.chunk_size):
                end = min(start + args.chunk_size, num_gaussians)
                adjacent = np.asarray(neighbors[start:end], dtype=np.int64)
                row_segments = segments[start:end, None]
                neighbor_segments = segments[adjacent]
                observed = (row_segments >= 0) & (neighbor_segments >= 0)
                edge_weight = np.sqrt(
                    np.clip(
                        visibility[start:end, None]
                        * visibility[adjacent]
                        * confidence[start:end, None]
                        * confidence[adjacent],
                        0.0,
                        1.0,
                    )
                )
                same = observed & (row_segments == neighbor_segments)
                different = observed & ~same
                positive_mass[split, level, start:end] += np.where(
                    same, edge_weight, 0.0
                )
                negative_mass[split, level, start:end] += np.where(
                    different, edge_weight, 0.0
                )
                observation_count[split, level, start:end] += observed.astype(
                    np.uint8
                )
        accepted_per_view_level.append(accepted_levels)
        del payload, segment_maps, point_ids, point_weights
        print(
            f"[{entry_index + 1}/{len(entries)}] {entry['image_name']} "
            f"accepted={accepted_levels}",
            flush=True,
        )

    metrics = evaluate_relation_models(
        positive_mass,
        negative_mass,
        observation_count,
        args.minimum_split_views,
        args.beta_prior,
        args.positive_threshold,
        args.negative_threshold,
    )
    conflict_metrics = evaluate_train_selected_conflicts(
        positive_mass,
        negative_mass,
        observation_count,
        args.minimum_split_views,
        args.beta_prior,
        args.positive_threshold,
        args.negative_threshold,
    )
    gate = make_gate_decision(
        metrics,
        args.minimum_nll_improvement,
        args.minimum_split_stability,
    )

    signature_path = os.path.join(output_dir, "multiscale_relation_signature.npy")
    ambiguous_path = os.path.join(output_dir, "stable_set_ambiguous_edges.npy")
    np.save(signature_path, np.moveaxis(metrics["relation_signature"], 0, -1))
    np.save(ambiguous_path, metrics["stable_set_ambiguous"])

    a39_positive = np.asarray(a39_relations) > 0
    a39_active = np.asarray(a39_relations) != 0
    multilevel = metrics["multilevel_edges"]
    stable_ambiguous = metrics["stable_set_ambiguous"]
    per_level = []
    for level in range(LEVEL_COUNT):
        valid = metrics["valid"][level]
        combined_positive = positive_mass[:, level].sum(axis=0)
        combined_negative = negative_mass[:, level].sum(axis=0)
        total = combined_positive + combined_negative
        per_level.append(
            {
                "level": level,
                "valid_directed_edges": int(valid.sum()),
                "same_relation_mass_fraction": float(
                    combined_positive[valid].sum(dtype=np.float64)
                    / max(total[valid].sum(dtype=np.float64), 1e-12)
                ),
                "minimum_split_view_quantiles": quantiles(
                    observation_count[:, level].min(axis=0)[valid]
                ),
            }
        )

    public_metrics = {
        key: value
        for key, value in metrics.items()
        if key
        not in {
            "valid",
            "multilevel_edges",
            "stable_set_ambiguous",
            "relation_signature",
            "probabilities",
        }
    }
    manifest = {
        "format_version": 1,
        "experiment": "A46.0b_boundary_stratified_multiscale_set_relation_audit",
        "representation": "heldout_multiscale_set_relation_diagnostic",
        "scene": "ramen",
        "num_gaussians": num_gaussians,
        "num_views": len(entries),
        "levels": list(range(LEVEL_COUNT)),
        "neighbors": neighbor_count,
        "metrics": public_metrics,
        "train_selected_conflict_metrics": conflict_metrics,
        "per_level": per_level,
        "a39_comparison": {
            "a39_active_directed_edges": int(a39_active.sum()),
            "a39_positive_directed_edges": int(a39_positive.sum()),
            "a39_positive_with_multilevel_evidence": int(
                (a39_positive & multilevel).sum()
            ),
            "a39_positive_recovered_as_stable_set_ambiguous": int(
                (a39_positive & stable_ambiguous).sum()
            ),
            "recovery_fraction_among_supported_a39_positive": float(
                (a39_positive & stable_ambiguous).sum()
                / max(int((a39_positive & multilevel).sum()), 1)
            ),
        },
        "gate": gate,
        "artifacts": {
            "multiscale_relation_signature": os.path.basename(signature_path),
            "stable_set_ambiguous_edges": os.path.basename(ambiguous_path),
        },
        "inputs": {
            "memory_dir": memory_dir,
            "memory_manifest_sha256": file_sha256(memory_manifest_path),
            "cache_dir": cache_dir,
            "cache_manifest_sha256": file_sha256(cache_manifest_path),
            "feature_dir": feature_dir,
            "relation_graph_dir": relation_graph_dir,
            "relation_manifest_sha256": file_sha256(relation_manifest_path),
        },
        "source_contract": {
            "training_views_only": True,
            "all_four_post_nms_sam_proposal_maps": True,
            "cross_level_overlap_preserved": True,
            "a39_fixed_3d_knn_reused": True,
            "raw_top45_talpha_used": True,
            "odd_even_heldout_evaluation": True,
            "evaluation_queries_or_labels_used": False,
            "codebooks_trained": False,
            "fixed_seed": args.expected_memory_seed,
        },
        "args": vars(args),
        "elapsed_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    with open(os.path.join(output_dir, "gate.json"), "w") as output:
        json.dump(gate, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
