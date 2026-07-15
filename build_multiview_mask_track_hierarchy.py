#!/usr/bin/env python
"""Build self-trained object tracks from large SAM masks and Gaussian co-visibility."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


class UnionFind:
    def __init__(self):
        self.parent = []
        self.rank = []

    def add(self, count):
        start = len(self.parent)
        self.parent.extend(range(start, start + count))
        self.rank.extend([0] * count)
        return start

    def find(self, value):
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return
        if self.rank[first] < self.rank[second]:
            first, second = second, first
        self.parent[second] = first
        if self.rank[first] == self.rank[second]:
            self.rank[first] += 1


def normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def dominant_gaussian_segments(cache):
    """Return each visible Gaussian's dominant SAM segment and accumulated support."""
    point_ids = cache["point_ids"][:, 0].numpy().astype(np.int64, copy=False)
    segment_ids = cache["segment_ids"].numpy().astype(np.int64, copy=False)
    weights = cache["point_weights"][:, 0].numpy().astype(np.float32, copy=False)
    valid = (point_ids >= 0) & (segment_ids >= 0) & (weights > 0)
    if not valid.any():
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
        )
    point_ids = point_ids[valid]
    segment_ids = segment_ids[valid]
    weights = weights[valid]
    segment_count = int(cache["feature_latents"].shape[0])
    pairs = point_ids * segment_count + segment_ids
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
    pair_weights = np.bincount(inverse, weights=weights).astype(np.float32)
    gaussians = unique_pairs // segment_count
    segments = unique_pairs % segment_count
    order = np.lexsort((-pair_weights, gaussians))
    ordered_gaussians = gaussians[order]
    chosen = order[np.r_[True, ordered_gaussians[1:] != ordered_gaussians[:-1]]]
    return gaussians[chosen], segments[chosen], pair_weights[chosen]


def load_cache(cache_dir, entry):
    import torch

    return torch.load(
        os.path.join(cache_dir, entry["cache"]), map_location="cpu", weights_only=False
    )


def gather_node_features(node_ids, view_offsets, feature_tables):
    """Fetch features by node ID without materializing a growing global table."""
    offsets = np.asarray(view_offsets, dtype=np.int64)
    source_views = np.searchsorted(offsets, node_ids, side="right") - 1
    gathered = np.empty((node_ids.size, feature_tables[0].shape[1]), dtype=np.float32)
    for source_view in np.unique(source_views):
        selected = source_views == source_view
        local_ids = node_ids[selected] - offsets[source_view]
        gathered[selected] = feature_tables[source_view][local_ids]
    return gathered


def _normalize_distribution(values):
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    total = values.sum()
    if total <= 0.0:
        return np.full(values.shape, 1.0 / max(1, values.size), dtype=np.float64)
    return values / total


def _kl_divergence(target, behavior):
    valid = (target > 0.0) & (behavior > 0.0)
    return float(np.sum(target[valid] * np.log(target[valid] / behavior[valid])))


def _kl_trust_region(target, behavior, max_kl):
    if max_kl <= 0.0 or _kl_divergence(target, behavior) <= max_kl:
        return target
    low, high = 0.0, 1.0
    for _ in range(32):
        mix = 0.5 * (low + high)
        candidate = (1.0 - mix) * behavior + mix * target
        if _kl_divergence(candidate, behavior) <= max_kl:
            low = mix
        else:
            high = mix
    return (1.0 - low) * behavior + low * target


def compute_node_importance(
    features,
    supports,
    track_ids,
    mode,
    temperature,
    max_kl,
    ratio_clip,
    agreement_power,
    information_weight,
):
    """Return normalized per-track view weights and SPIRE-style diagnostics."""
    weights = np.zeros(supports.shape[0], dtype=np.float32)
    diagnostics = []
    for track_id in range(int(track_ids.max()) + 1):
        indices = np.flatnonzero(track_ids == track_id)
        if not indices.size:
            continue
        if mode == "uniform":
            behavior = _normalize_distribution(np.ones(indices.size))
        else:
            behavior = _normalize_distribution(supports[indices])
        target = behavior
        if mode == "information_kl":
            track_features = features[indices].astype(np.float64)
            consensus = normalize(
                np.sum(behavior[:, None] * track_features, axis=0, keepdims=True)
            )[0]
            agreement = np.clip(track_features @ consensus, 0.0, 1.0) ** agreement_power
            node_information = supports[indices, None].astype(np.float64) * np.square(
                track_features
            )
            total_information = 1e-4 + node_information.sum(axis=0)
            without_node = np.maximum(1e-4, total_information[None, :] - node_information)
            information_gain = np.log(total_information[None, :] / without_node).mean(axis=1)
            information_gain /= max(float(information_gain.max()), 1e-8)
            utility = information_weight * information_gain + np.log(
                np.maximum(agreement, 1e-4)
            )
            logits = np.clip(utility / temperature, -30.0, 30.0)
            target = _normalize_distribution(behavior * np.exp(logits - logits.max()))
            target = _kl_trust_region(target, behavior, max_kl)
            ratios = target / np.maximum(behavior, 1e-20)
            ratios = np.minimum(ratios, ratio_clip)
            target = _normalize_distribution(behavior * ratios)
        ratios = target / np.maximum(behavior, 1e-20)
        weights[indices] = target.astype(np.float32)
        diagnostics.append(
            {
                "support": float(supports[indices].sum()),
                "kl": _kl_divergence(target, behavior),
                "ess": float(1.0 / np.maximum(np.square(target).sum(), 1e-20)),
                "ratio_max": float(ratios.max()),
            }
        )
    total_support = sum(item["support"] for item in diagnostics)
    metric_weights = np.asarray(
        [item["support"] for item in diagnostics], dtype=np.float64
    )
    metric_weights = _normalize_distribution(metric_weights)
    summary = {
        "mode": mode,
        "weighted_kl_target_to_behavior": float(
            sum(weight * item["kl"] for weight, item in zip(metric_weights, diagnostics))
        ),
        "weighted_effective_nodes": float(
            sum(weight * item["ess"] for weight, item in zip(metric_weights, diagnostics))
        ),
        "max_importance_ratio": max(
            (item["ratio_max"] for item in diagnostics), default=1.0
        ),
        "total_track_support": float(total_support),
    }
    return weights, summary


def aggregate_point_tracks(num_gaussians, num_tracks, points, tracks, scores, top_m):
    pairs = points.astype(np.int64) * num_tracks + tracks.astype(np.int64)
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
    pair_scores = np.bincount(inverse, weights=scores).astype(np.float32)
    pair_points = unique_pairs // num_tracks
    pair_tracks = unique_pairs % num_tracks
    order = np.lexsort((-pair_scores, pair_points))
    sorted_points = pair_points[order]
    starts = np.r_[True, sorted_points[1:] != sorted_points[:-1]]
    group_starts = np.maximum.accumulate(
        np.where(starts, np.arange(sorted_points.size), 0)
    )
    ranks = np.arange(sorted_points.size) - group_starts
    selected = order[ranks < top_m]
    selected_points = pair_points[selected]
    selected_tracks = pair_tracks[selected]
    selected_scores = pair_scores[selected]
    selected_order = np.lexsort((-selected_scores, selected_points))
    selected_points = selected_points[selected_order]
    selected_tracks = selected_tracks[selected_order]
    selected_scores = selected_scores[selected_order]
    starts = np.r_[True, selected_points[1:] != selected_points[:-1]]
    group_starts = np.maximum.accumulate(
        np.where(starts, np.arange(selected_points.size), 0)
    )
    slots = np.arange(selected_points.size) - group_starts
    output_ids = np.full((num_gaussians, top_m), -1, dtype=np.int64)
    output_scores = np.zeros((num_gaussians, top_m), dtype=np.float32)
    output_ids[selected_points, slots] = selected_tracks
    output_scores[selected_points, slots] = selected_scores
    output_scores /= np.maximum(output_scores.sum(axis=1, keepdims=True), 1e-8)
    return output_ids, output_scores


def main():
    parser = ArgumentParser(
        description="Construct compact object-track IDs from multiview SAM masks without teacher semantics."
    )
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.82)
    parser.add_argument("--min_track_support", type=float, default=32.0)
    parser.add_argument(
        "--view_weighting",
        choices=["legacy", "uniform", "contribution", "information_kl"],
        default="legacy",
    )
    parser.add_argument("--top_m", type=int, default=1)
    parser.add_argument("--importance_temperature", type=float, default=1.0)
    parser.add_argument("--max_view_kl", type=float, default=0.1)
    parser.add_argument("--importance_ratio_clip", type=float, default=5.0)
    parser.add_argument("--agreement_power", type=float, default=1.0)
    parser.add_argument("--information_weight", type=float, default=1.0)
    parser.add_argument("--max_views", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    if not 0.0 < args.similarity_threshold <= 1.0:
        raise ValueError("similarity_threshold must be in (0, 1]")
    if args.min_track_support <= 0:
        raise ValueError("min_track_support must be positive")
    if args.top_m <= 0 or args.importance_temperature <= 0.0:
        raise ValueError("top-M and importance temperature must be positive")
    if args.max_view_kl < 0.0 or args.importance_ratio_clip < 1.0:
        raise ValueError("KL must be non-negative and ratio clip at least one")
    if args.agreement_power <= 0.0 or args.information_weight < 0.0:
        raise ValueError("Agreement power must be positive and information weight non-negative")

    cache_dir = os.path.abspath(args.cache_dir)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("codec_type") != "identity" or int(manifest.get("semantic_dim", 0)) != 512:
        raise ValueError("Mask tracks require an identity 512D semantic cache")
    entries = manifest["views"]
    if args.max_views > 0:
        entries = entries[: args.max_views]
    num_gaussians = int(manifest["num_gaussians"])
    last_node = np.full(num_gaussians, -1, dtype=np.int32)
    union_find = UnionFind()
    node_features = []
    node_supports = []
    view_offsets = []

    for view_index, entry in enumerate(entries):
        cache = load_cache(cache_dir, entry)
        features = normalize(cache["feature_latents"].numpy())
        node_offset = union_find.add(features.shape[0])
        view_offsets.append(node_offset)
        node_features.append(features)
        segment_weights = np.bincount(
            cache["segment_ids"].numpy().astype(np.int64, copy=False),
            weights=cache["point_weights"][:, 0].numpy().astype(np.float32, copy=False),
            minlength=features.shape[0],
        ).astype(np.float32)
        node_supports.append(segment_weights)

        gaussians, segments, supports = dominant_gaussian_segments(cache)
        if gaussians.size:
            current_nodes = node_offset + segments
            old_nodes = last_node[gaussians]
            valid_old = old_nodes >= 0
            if valid_old.any():
                previous_features = gather_node_features(
                    old_nodes[valid_old], view_offsets, node_features
                )
                similarities = np.einsum(
                    "ij,ij->i", features[segments[valid_old]], previous_features
                )
                for current, previous in zip(
                    current_nodes[valid_old][similarities >= args.similarity_threshold],
                    old_nodes[valid_old][similarities >= args.similarity_threshold],
                ):
                    union_find.union(int(current), int(previous))
            last_node[gaussians] = current_nodes.astype(np.int32)
        print(
            json.dumps(
                {
                    "view": view_index,
                    "nodes": len(union_find.parent),
                    "dominant_gaussians": int(gaussians.size),
                }
            )
        )

    all_features = np.concatenate(node_features, axis=0)
    all_supports = np.concatenate(node_supports, axis=0)
    roots = np.asarray([union_find.find(index) for index in range(len(union_find.parent))])
    unique_roots, inverse = np.unique(roots, return_inverse=True)
    track_supports = np.bincount(inverse, weights=all_supports).astype(np.float32)
    if args.view_weighting == "legacy":
        node_importance = all_supports.astype(np.float32)
        importance_summary = {"mode": "legacy"}
    else:
        node_importance, importance_summary = compute_node_importance(
            all_features,
            all_supports,
            inverse,
            args.view_weighting,
            args.importance_temperature,
            args.max_view_kl,
            args.importance_ratio_clip,
            args.agreement_power,
            args.information_weight,
        )
    track_sums = np.zeros((unique_roots.size, all_features.shape[1]), dtype=np.float64)
    np.add.at(track_sums, inverse, all_features.astype(np.float64) * node_importance[:, None])
    track_features = normalize(track_sums.astype(np.float32))
    supported_tracks = track_supports >= args.min_track_support
    remap = np.full(unique_roots.size, -1, dtype=np.int64)
    remap[supported_tracks] = np.arange(supported_tracks.sum(), dtype=np.int64)

    point_track_ids = np.full((num_gaussians, args.top_m), -1, dtype=np.int64)
    point_track_scores = np.zeros((num_gaussians, args.top_m), dtype=np.float32)
    point_observations = []
    track_observations = []
    score_observations = []
    for entry, node_offset in zip(entries, view_offsets):
        cache = load_cache(cache_dir, entry)
        gaussians, segments, supports = dominant_gaussian_segments(cache)
        if not gaussians.size:
            continue
        node_tracks = remap[inverse[node_offset + segments]]
        valid = node_tracks >= 0
        if args.view_weighting == "legacy":
            better = valid & (supports > point_track_scores[gaussians, 0])
            point_track_ids[gaussians[better], 0] = node_tracks[better]
            point_track_scores[gaussians[better], 0] = supports[better]
        elif valid.any():
            nodes = node_offset + segments[valid]
            within_node = supports[valid] / np.maximum(all_supports[nodes], 1e-8)
            scores = node_importance[nodes] * within_node
            point_observations.append(gaussians[valid])
            track_observations.append(node_tracks[valid])
            score_observations.append(scores)

    if args.view_weighting != "legacy" and point_observations:
        point_track_ids, point_track_scores = aggregate_point_tracks(
            num_gaussians,
            int(supported_tracks.sum()),
            np.concatenate(point_observations),
            np.concatenate(track_observations),
            np.concatenate(score_observations),
            args.top_m,
        )

    track_features = track_features[supported_tracks].astype(np.float16)
    if track_features.shape[0] <= np.iinfo(np.uint16).max:
        id_dtype = np.uint16
    else:
        id_dtype = np.uint32
    invalid_id = int(np.iinfo(id_dtype).max)
    packed_ids = np.full(point_track_ids.shape, invalid_id, dtype=id_dtype)
    valid_assignments = point_track_ids >= 0
    packed_ids[valid_assignments] = point_track_ids[valid_assignments].astype(id_dtype)
    packed_weights = np.rint(np.clip(point_track_scores, 0.0, 1.0) * 255.0).astype(np.uint8)
    packed_weights[~valid_assignments] = 0
    valid_points = valid_assignments.any(axis=1)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "group_codebook.npy"), track_features)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), packed_weights)
    storage_bytes = int(track_features.nbytes + packed_ids.nbytes + packed_weights.nbytes)
    result = {
        "format_version": 1,
        # Keep the established compact artifact contract so existing evaluation
        # code can consume tracks without a special-case loader.
        "representation": "compact_group_hierarchy",
        "hierarchy_type": "multiview_sam_track_hierarchy",
        "num_gaussians": num_gaussians,
        "num_group_codes": int(track_features.shape[0]),
        "feature_dim": int(track_features.shape[1]),
        "top_m": int(args.top_m),
        "group_codebook": "group_codebook.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "id_dtype": str(packed_ids.dtype),
        "invalid_id": invalid_id,
        "weight_dtype": "uint8_normalized",
        "covered_fraction": float(valid_points.mean()),
        "mean_track_support": float(track_supports[supported_tracks].mean()),
        "view_importance": importance_summary,
        "storage": {
            "group_codebook_bytes_fp16": int(track_features.nbytes),
            "point_group_id_bytes": int(packed_ids.nbytes),
            "point_group_weight_bytes": int(packed_weights.nbytes),
            "total_semantic_bytes": storage_bytes,
            "bytes_per_gaussian_amortized": float(storage_bytes / num_gaussians),
        },
        "source": {
            "cache_dir": cache_dir,
            "views": len(entries),
            "similarity_threshold": args.similarity_threshold,
            "min_track_support": args.min_track_support,
            "semantic_source": "multiview_large_sam_masks_and_frozen_2d_clip_features",
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
