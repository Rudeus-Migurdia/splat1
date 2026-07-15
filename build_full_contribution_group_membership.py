#!/usr/bin/env python
"""Build MuSplat-style soft group membership from all cached ray contributions."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
from scipy import sparse

from build_multiview_mask_track_hierarchy import (
    UnionFind,
    compute_node_importance,
    dominant_gaussian_segments,
    gather_node_features,
    load_cache,
    normalize,
)
from semantic_gaussian_association import (
    ResidualCodebookSemanticScorer,
    ViewClassifierSemanticScorer,
    aggregate_segment_signatures,
    normalize_sparse_rows,
    prune_sparse_rows,
    semantic_geometry_union,
)


def reduce_sparse_pairs(pairs, values):
    pairs = np.asarray(pairs, dtype=np.int64)
    values = np.asarray(values, dtype=np.float32)
    if not pairs.size:
        return pairs, values
    order = np.argsort(pairs, kind="stable")
    pairs = pairs[order]
    values = values[order]
    starts = np.r_[0, np.flatnonzero(pairs[1:] != pairs[:-1]) + 1]
    return pairs[starts], np.add.reduceat(values, starts).astype(np.float32)


def merge_sparse_pairs(first_pairs, first_values, second_pairs, second_values):
    if not first_pairs.size:
        return second_pairs, second_values
    if not second_pairs.size:
        return first_pairs, first_values
    return reduce_sparse_pairs(
        np.concatenate((first_pairs, second_pairs)),
        np.concatenate((first_values, second_values)),
    )


def foreground_pairs_from_signatures(signatures, segment_tracks, num_tracks):
    signatures = signatures.tocoo()
    tracks = segment_tracks[signatures.row]
    valid = tracks >= 0
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pairs = signatures.col[valid].astype(np.int64) * num_tracks + tracks[valid]
    return reduce_sparse_pairs(pairs, signatures.data[valid])


def foreground_pairs(cache, segment_tracks, num_tracks):
    point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
    point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
    segment_ids = cache["segment_ids"].numpy().astype(np.int64, copy=False)
    pixel_tracks = np.full(segment_ids.shape, -1, dtype=np.int64)
    valid_segments = (segment_ids >= 0) & (segment_ids < segment_tracks.size)
    pixel_tracks[valid_segments] = segment_tracks[segment_ids[valid_segments]]
    tracks = np.broadcast_to(pixel_tracks[:, None], point_ids.shape)
    valid = (point_ids >= 0) & (point_weights > 0.0) & (tracks >= 0)
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
    pairs = point_ids[valid] * num_tracks + tracks[valid]
    return reduce_sparse_pairs(pairs, point_weights[valid])


def candidate_offsets(candidate_points, num_gaussians):
    counts = np.bincount(candidate_points, minlength=num_gaussians)
    offsets = np.empty(num_gaussians + 1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(counts, out=offsets[1:])
    return offsets


def candidate_positions_for_points(points, offsets):
    starts = offsets[points]
    counts = offsets[points + 1] - starts
    valid = counts > 0
    if not valid.any():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    starts = starts[valid]
    counts = counts[valid]
    source_rows = np.flatnonzero(valid)
    repeated_starts = np.repeat(starts, counts)
    repeated_bases = np.repeat(np.cumsum(counts) - counts, counts)
    within = np.arange(int(counts.sum()), dtype=np.int64) - repeated_bases
    return repeated_starts + within, np.repeat(source_rows, counts)


def pack_top_memberships(
    num_gaussians,
    candidate_points,
    candidate_tracks,
    memberships,
    top_m,
    membership_threshold,
    min_foreground,
    foreground,
    excluded=None,
):
    eligible = (
        (memberships > membership_threshold)
        & (foreground >= min_foreground)
        & np.isfinite(memberships)
    )
    if excluded is not None:
        eligible &= ~excluded
    points = candidate_points[eligible]
    tracks = candidate_tracks[eligible]
    scores = memberships[eligible]
    output_ids = np.full((num_gaussians, top_m), -1, dtype=np.int64)
    output_scores = np.zeros((num_gaussians, top_m), dtype=np.float32)
    if not points.size:
        return output_ids, output_scores
    order = np.lexsort((-scores, points))
    points = points[order]
    tracks = tracks[order]
    scores = scores[order]
    starts = np.r_[True, points[1:] != points[:-1]]
    group_starts = np.maximum.accumulate(np.where(starts, np.arange(points.size), 0))
    ranks = np.arange(points.size) - group_starts
    keep = ranks < top_m
    output_ids[points[keep], ranks[keep]] = tracks[keep]
    output_scores[points[keep], ranks[keep]] = scores[keep]
    return output_ids, output_scores


def selected_candidate_values(
    point_ids,
    candidate_points,
    candidate_tracks,
    candidate_values,
    num_tracks,
):
    output = np.zeros(point_ids.shape, dtype=np.float32)
    rows, slots = np.nonzero(point_ids >= 0)
    if not rows.size:
        return output
    candidate_pairs = candidate_points * num_tracks + candidate_tracks
    selected_pairs = rows * num_tracks + point_ids[rows, slots]
    positions = np.searchsorted(candidate_pairs, selected_pairs)
    found = (
        (positions < candidate_pairs.size)
        & (candidate_pairs[np.minimum(positions, candidate_pairs.size - 1)] == selected_pairs)
    )
    output[rows[found], slots[found]] = candidate_values[positions[found]]
    return output


def rofa_inlier_mask(features, track_ids, tau):
    """Filter low-agreement view observations within each semantic track."""
    features = normalize(features)
    track_ids = np.asarray(track_ids, dtype=np.int64)
    if features.shape[0] != track_ids.size:
        raise ValueError("ROFA features and track IDs must have matching rows")
    if tau <= 0.0:
        return np.ones(track_ids.shape, dtype=bool), {
            "enabled": False,
            "tau": float(tau),
            "num_observations": int(track_ids.size),
            "num_removed": 0,
            "removed_fraction": 0.0,
            "num_affected_tracks": 0,
        }

    keep = np.ones(track_ids.shape, dtype=bool)
    affected_tracks = 0
    for track_id in range(int(track_ids.max()) + 1):
        indices = np.flatnonzero(track_ids == track_id)
        if indices.size <= 1:
            continue
        track_features = features[indices].astype(np.float64, copy=False)
        feature_sum = track_features.sum(axis=0)
        mean_similarity = (
            np.einsum("ij,j->i", track_features, feature_sum) - 1.0
        ) / (indices.size - 1)
        threshold = mean_similarity.mean() - tau * mean_similarity.std()
        track_keep = mean_similarity >= threshold
        if not track_keep.any():
            track_keep[mean_similarity.argmax()] = True
        keep[indices] = track_keep
        affected_tracks += int((~track_keep).any())

    removed = int((~keep).sum())
    return keep, {
        "enabled": True,
        "tau": float(tau),
        "num_observations": int(track_ids.size),
        "num_removed": removed,
        "removed_fraction": float(removed / max(1, track_ids.size)),
        "num_affected_tracks": affected_tracks,
    }


def segment_gaussian_signatures(cache, num_gaussians):
    """Build L2-normalized segment signatures from every cached ray contribution."""
    return normalize_sparse_rows(aggregate_segment_signatures(cache, num_gaussians))


class SegmentSignatureProvider:
    def __init__(self, output_dir, num_gaussians, args, semantic_scorer=None):
        self.num_gaussians = int(num_gaussians)
        self.args = args
        self.semantic_scorer = semantic_scorer
        self.cache_dir = os.path.abspath(
            args.association_cache_dir
            or os.path.join(output_dir, "association_signatures")
        )
        self.summaries = {}
        if args.membership_mode == "saga_union":
            os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, cache, entry, view_index):
        if self.args.membership_mode != "saga_union":
            return aggregate_segment_signatures(cache, self.num_gaussians)
        cache_path = os.path.join(self.cache_dir, f"{view_index:04d}.npz")
        summary_path = os.path.join(self.cache_dir, f"{view_index:04d}.json")
        if os.path.isfile(cache_path) and os.path.isfile(summary_path):
            signatures = sparse.load_npz(cache_path).tocsr()
            with open(summary_path) as source:
                summary = json.load(source)
        else:
            raw = aggregate_segment_signatures(cache, self.num_gaussians)
            if hasattr(self.semantic_scorer, "set_view"):
                self.semantic_scorer.set_view(view_index)
            signatures, summary = semantic_geometry_union(
                raw,
                normalize(cache["feature_latents"].numpy()),
                self.semantic_scorer,
                self.args.association_fraction,
                self.args.association_max_candidates,
            )
            sparse.save_npz(cache_path, signatures, compressed=True)
            with open(summary_path, "w") as output:
                json.dump(summary, output, indent=2)
        self.summaries[int(view_index)] = summary
        return signatures

    def summary(self):
        if not self.summaries:
            return {"mode": self.args.membership_mode}
        items = list(self.summaries.values())
        keys = items[0]
        totals = {key: int(sum(item[key] for item in items)) for key in keys}
        totals.update(
            {
                "mode": self.args.membership_mode,
                "keep_fraction": float(self.args.association_fraction),
                "max_candidates": int(self.args.association_max_candidates),
                "cache_dir": self.cache_dir,
                "semantic_rescued_fraction": float(
                    totals["semantic_rescued_pairs"]
                    / max(1, totals["selected_pairs"])
                ),
            }
        )
        return totals


def segment_spatial_statistics(cache, gaussian_xyz):
    """Estimate each 2D segment's 3D center and scale from all ray contributions."""
    point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
    point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
    segment_ids = cache["segment_ids"].numpy().astype(np.int64, copy=False)
    num_segments = int(cache["feature_latents"].shape[0])
    rows = np.broadcast_to(segment_ids[:, None], point_ids.shape)
    valid = (
        (rows >= 0)
        & (rows < num_segments)
        & (point_ids >= 0)
        & (point_ids < gaussian_xyz.shape[0])
        & (point_weights > 0.0)
    )
    rows = rows[valid]
    ids = point_ids[valid]
    weights = point_weights[valid].astype(np.float64, copy=False)
    totals = np.bincount(rows, weights=weights, minlength=num_segments)
    centers = np.zeros((num_segments, 3), dtype=np.float64)
    second_moment = np.zeros(num_segments, dtype=np.float64)
    if rows.size:
        points = gaussian_xyz[ids].astype(np.float64, copy=False)
        for axis in range(3):
            centers[:, axis] = np.bincount(
                rows,
                weights=weights * points[:, axis],
                minlength=num_segments,
            )
        second_moment = np.bincount(
            rows,
            weights=weights * np.einsum("ij,ij->i", points, points),
            minlength=num_segments,
        )
    nonzero = totals > 0.0
    centers[nonzero] /= totals[nonzero, None]
    mean_square = np.zeros(num_segments, dtype=np.float64)
    mean_square[nonzero] = second_moment[nonzero] / totals[nonzero]
    radii = np.sqrt(
        np.maximum(mean_square - np.einsum("ij,ij->i", centers, centers), 0.0)
    )
    return centers.astype(np.float32), radii.astype(np.float32)


def mutual_soft_overlap_matches(
    current_signatures,
    current_features,
    previous_views,
    min_overlap,
    min_semantic_similarity,
    current_centers=None,
    current_radii=None,
    max_spatial_distance_ratio=None,
    spatial_score_power=1.0,
):
    """Match each current segment to at most one mutual-best previous segment."""
    if not previous_views or current_signatures.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    previous_signatures = sparse.vstack(
        [item["signatures"] for item in previous_views], format="csr"
    )
    previous_features = np.concatenate(
        [item["features"] for item in previous_views], axis=0
    )
    previous_nodes = np.concatenate([item["nodes"] for item in previous_views])
    overlap = (current_signatures @ previous_signatures.T).toarray()
    semantic = current_features @ previous_features.T
    semantic_factor = np.clip(
        (semantic - min_semantic_similarity) / max(1e-8, 1.0 - min_semantic_similarity),
        0.0,
        1.0,
    )
    score = overlap * semantic_factor
    score[(overlap < min_overlap) | (semantic < min_semantic_similarity)] = 0.0
    if current_centers is not None:
        if current_radii is None or max_spatial_distance_ratio is None:
            raise ValueError("Spatial matching requires radii and a distance ratio")
        previous_centers = np.concatenate(
            [item["centers"] for item in previous_views], axis=0
        )
        previous_radii = np.concatenate(
            [item["radii"] for item in previous_views], axis=0
        )
        current_square = np.einsum("ij,ij->i", current_centers, current_centers)
        previous_square = np.einsum("ij,ij->i", previous_centers, previous_centers)
        distance_square = np.maximum(
            current_square[:, None]
            + previous_square[None, :]
            - 2.0 * current_centers @ previous_centers.T,
            0.0,
        )
        distance = np.sqrt(distance_square)
        spatial_scale = np.maximum(
            current_radii[:, None] + previous_radii[None, :], 1e-6
        )
        distance_ratio = distance / spatial_scale
        spatial_factor = np.exp(-0.5 * distance_ratio ** 2)
        score *= spatial_factor ** spatial_score_power
        score[distance_ratio > max_spatial_distance_ratio] = 0.0
    if not np.any(score > 0.0):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    current_best_previous = score.argmax(axis=1)
    previous_best_current = score.argmax(axis=0)
    current_rows = np.arange(score.shape[0], dtype=np.int64)
    accepted = (
        (score[current_rows, current_best_previous] > 0.0)
        & (previous_best_current[current_best_previous] == current_rows)
    )
    return current_rows[accepted], previous_nodes[current_best_previous[accepted]]


def mutual_memory_matches(
    current_signatures,
    current_features,
    memory_signatures,
    memory_features,
    min_overlap,
    min_semantic_similarity,
):
    """Reciprocal matching against global track evidence rather than recent views."""
    if current_signatures.shape[0] == 0 or memory_signatures.shape[0] == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    overlap = (current_signatures @ memory_signatures.T).toarray()
    semantic = current_features @ memory_features.T
    semantic_factor = np.clip(
        (semantic - min_semantic_similarity)
        / max(1e-8, 1.0 - min_semantic_similarity),
        0.0,
        1.0,
    )
    score = overlap * semantic_factor
    score[(overlap < min_overlap) | (semantic < min_semantic_similarity)] = 0.0
    if not np.any(score > 0.0):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    current_best = score.argmax(axis=1)
    memory_best = score.argmax(axis=0)
    rows = np.arange(score.shape[0], dtype=np.int64)
    accepted = (
        (score[rows, current_best] > 0.0)
        & (memory_best[current_best] == rows)
    )
    return rows[accepted], current_best[accepted]


def update_memory_signature(previous, current, count, max_nonzero):
    merged = previous.multiply(float(count)) + current
    merged = prune_sparse_rows(merged, max_nonzero)
    return normalize_sparse_rows(merged)


def build_tracks(
    cache_dir,
    entries,
    num_gaussians,
    args,
    signature_provider,
    gaussian_xyz=None,
):
    last_node = (
        np.full(num_gaussians, -1, dtype=np.int32)
        if args.track_linking == "legacy_last_node"
        else None
    )
    previous_views = []
    union_find = UnionFind()
    node_features = []
    node_supports = []
    view_offsets = []
    memory_signatures = []
    memory_feature_sums = []
    memory_representatives = []
    memory_counts = []
    memory_links = 0

    for view_index, entry in enumerate(entries):
        cache = load_cache(cache_dir, entry)
        features = normalize(cache["feature_latents"].numpy())
        node_offset = union_find.add(features.shape[0])
        view_offsets.append(node_offset)
        node_features.append(features)
        point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
        support_weights = (
            point_weights[:, 0]
            if args.track_linking == "legacy_last_node"
            else point_weights.sum(axis=1)
        )
        segment_weights = np.bincount(
            cache["segment_ids"].numpy().astype(np.int64, copy=False),
            weights=support_weights,
            minlength=features.shape[0],
        ).astype(np.float32)
        node_supports.append(segment_weights)

        if args.track_linking == "legacy_last_node":
            gaussians, segments, _ = dominant_gaussian_segments(cache)
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
                    accepted = similarities >= args.similarity_threshold
                    for current, previous in zip(
                        current_nodes[valid_old][accepted], old_nodes[valid_old][accepted]
                    ):
                        union_find.union(int(current), int(previous))
                last_node[gaussians] = current_nodes.astype(np.int32)
        else:
            raw_signatures = signature_provider.get(cache, entry, view_index)
            signatures = normalize_sparse_rows(raw_signatures)
            centers = None
            radii = None
            if args.track_linking == "trace_memory_mutual":
                if memory_signatures:
                    memory_matrix = sparse.vstack(memory_signatures, format="csr")
                    memory_features = normalize(np.stack(memory_feature_sums))
                    current_segments, memory_ids = mutual_memory_matches(
                        signatures,
                        features,
                        memory_matrix,
                        memory_features,
                        args.min_soft_overlap,
                        args.similarity_threshold,
                    )
                else:
                    current_segments = np.empty(0, dtype=np.int64)
                    memory_ids = np.empty(0, dtype=np.int64)
                matched = dict(zip(current_segments.tolist(), memory_ids.tolist()))
                for segment in range(features.shape[0]):
                    current_node = node_offset + segment
                    if segment in matched:
                        memory_id = matched[segment]
                        union_find.union(
                            int(current_node), int(memory_representatives[memory_id])
                        )
                        memory_signatures[memory_id] = update_memory_signature(
                            memory_signatures[memory_id],
                            signatures.getrow(segment),
                            memory_counts[memory_id],
                            args.memory_signature_points,
                        )
                        memory_feature_sums[memory_id] += (
                            features[segment] * max(float(segment_weights[segment]), 1e-6)
                        )
                        memory_counts[memory_id] += 1
                        memory_links += 1
                    else:
                        memory_signatures.append(
                            normalize_sparse_rows(
                                prune_sparse_rows(
                                    signatures.getrow(segment),
                                    args.memory_signature_points,
                                )
                            )
                        )
                        memory_feature_sums.append(
                            features[segment]
                            * max(float(segment_weights[segment]), 1e-6)
                        )
                        memory_representatives.append(current_node)
                        memory_counts.append(1)
                print(
                    json.dumps(
                        {
                            "memory_view": view_index,
                            "memory_tracks": len(memory_signatures),
                            "accepted_links": int(current_segments.size),
                        }
                    )
                )
                print(json.dumps({"track_view": view_index, "nodes": len(union_find.parent)}))
                continue
            if args.track_linking == "semantic_spatial_mutual":
                centers, radii = segment_spatial_statistics(cache, gaussian_xyz)
            current_segments, previous_nodes = mutual_soft_overlap_matches(
                signatures,
                features,
                previous_views,
                args.min_soft_overlap,
                args.similarity_threshold,
                centers,
                radii,
                args.max_spatial_distance_ratio
                if args.track_linking == "semantic_spatial_mutual"
                else None,
                args.spatial_score_power,
            )
            for current, previous in zip(node_offset + current_segments, previous_nodes):
                union_find.union(int(current), int(previous))
            previous_views.append(
                {
                    "signatures": signatures,
                    "features": features,
                    "nodes": node_offset + np.arange(features.shape[0], dtype=np.int64),
                    "centers": centers,
                    "radii": radii,
                }
            )
            previous_views = previous_views[-args.track_window :]
        print(json.dumps({"track_view": view_index, "nodes": len(union_find.parent)}))

    all_features = np.concatenate(node_features, axis=0)
    all_supports = np.concatenate(node_supports, axis=0)
    roots = np.asarray([union_find.find(index) for index in range(len(union_find.parent))])
    _, inverse = np.unique(roots, return_inverse=True)
    rofa_keep, rofa_summary = rofa_inlier_mask(
        all_features,
        inverse,
        args.rofa_tau,
    )
    filtered_supports = all_supports.copy()
    filtered_supports[~rofa_keep] = 0.0
    track_supports = np.bincount(inverse, weights=filtered_supports).astype(np.float32)
    node_importance, importance_summary = compute_node_importance(
        all_features,
        filtered_supports,
        inverse,
        args.view_weighting,
        args.importance_temperature,
        args.max_view_kl,
        args.importance_ratio_clip,
        args.agreement_power,
        args.information_weight,
    )
    importance_summary["rofa"] = rofa_summary
    importance_summary["tracking"] = {
        "mode": args.track_linking,
        "global_memory_tracks": int(len(memory_signatures)),
        "global_memory_links": int(memory_links),
    }
    track_sums = np.zeros((track_supports.size, all_features.shape[1]), dtype=np.float64)
    np.add.at(track_sums, inverse, all_features.astype(np.float64) * node_importance[:, None])
    track_features = normalize(track_sums.astype(np.float32))
    node_views = np.concatenate(
        [np.full(features.shape[0], index, dtype=np.int64) for index, features in enumerate(node_features)]
    )
    retained_nodes = node_importance > 0.0
    track_view_pairs, track_view_weights = reduce_sparse_pairs(
        inverse[retained_nodes].astype(np.int64) * len(entries)
        + node_views[retained_nodes],
        node_importance[retained_nodes],
    )
    pair_tracks = track_view_pairs // len(entries)
    view_counts = np.bincount(pair_tracks, minlength=track_supports.size).astype(np.float32)
    view_weight_sum = np.bincount(
        pair_tracks, weights=track_view_weights, minlength=track_supports.size
    ).astype(np.float32)
    view_weight_square_sum = np.bincount(
        pair_tracks, weights=track_view_weights ** 2, minlength=track_supports.size
    ).astype(np.float32)
    effective_views = view_weight_sum ** 2 / np.maximum(view_weight_square_sum, 1e-12)
    node_cosine = np.einsum("ij,ij->i", all_features, track_features[inverse])
    concentration_sum = np.bincount(
        inverse,
        weights=node_importance * node_cosine,
        minlength=track_supports.size,
    ).astype(np.float32)
    concentration_weight = np.bincount(
        inverse,
        weights=node_importance,
        minlength=track_supports.size,
    ).astype(np.float32)
    concentration = concentration_sum / np.maximum(concentration_weight, 1e-12)
    track_reliability = np.clip(concentration, 0.0, 1.0) * (
        effective_views / (effective_views + 1.0)
    )
    supported = (
        (track_supports >= args.min_track_support)
        & (view_counts >= args.min_track_views)
    )
    remap = np.full(track_supports.size, -1, dtype=np.int64)
    remap[supported] = np.arange(int(supported.sum()), dtype=np.int64)
    segment_tracks = []
    for offset, features in zip(view_offsets, node_features):
        stop = offset + features.shape[0]
        mapped = remap[inverse[offset:stop]].copy()
        mapped[~rofa_keep[offset:stop]] = -1
        segment_tracks.append(mapped)
    return (
        track_features[supported],
        track_supports[supported],
        segment_tracks,
        importance_summary,
        view_counts[supported],
        effective_views[supported],
        concentration[supported],
        track_reliability[supported],
    )


def binary_entropy(foreground_votes, total_votes):
    probability = foreground_votes.astype(np.float64) / np.maximum(total_votes, 1)
    probability = np.clip(probability, 1e-12, 1.0 - 1e-12)
    entropy = -(
        probability * np.log2(probability)
        + (1.0 - probability) * np.log2(1.0 - probability)
    )
    entropy[total_votes < 2] = 0.0
    return entropy.astype(np.float32)


def load_gaussian_opacity(checkpoint_path, num_gaussians):
    import torch

    model_params, _ = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) not in (12, 13):
        raise ValueError("Unsupported geometry checkpoint tuple")
    opacity = torch.sigmoid(model_params[6].detach().float()).reshape(-1).numpy()
    if opacity.shape != (num_gaussians,):
        raise ValueError("Geometry opacity does not match the cache Gaussian count")
    return opacity


def load_gaussian_xyz(checkpoint_path, num_gaussians):
    import torch

    model_params, _ = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) not in (12, 13):
        raise ValueError("Unsupported geometry checkpoint tuple")
    xyz = model_params[1].detach().float().numpy()
    if xyz.shape != (num_gaussians, 3):
        raise ValueError("Geometry coordinates do not match the cache Gaussian count")
    return xyz


def build_membership(
    cache_dir,
    entries,
    segment_tracks,
    num_gaussians,
    num_tracks,
    args,
    signature_provider,
):
    foreground_pair_ids = np.empty(0, dtype=np.int64)
    foreground_values = np.empty(0, dtype=np.float32)
    buffered_pairs = []
    buffered_values = []
    buffered_count = 0

    for view_index, (entry, tracks) in enumerate(zip(entries, segment_tracks)):
        cache = load_cache(cache_dir, entry)
        if args.membership_mode == "saga_union":
            signatures = signature_provider.get(cache, entry, view_index)
            pairs, values = foreground_pairs_from_signatures(
                signatures, tracks, num_tracks
            )
        else:
            pairs, values = foreground_pairs(cache, tracks, num_tracks)
        buffered_pairs.append(pairs)
        buffered_values.append(values)
        buffered_count += pairs.size
        if buffered_count >= args.reduce_buffer_pairs or view_index + 1 == len(entries):
            batch_pairs, batch_values = reduce_sparse_pairs(
                np.concatenate(buffered_pairs), np.concatenate(buffered_values)
            )
            foreground_pair_ids, foreground_values = merge_sparse_pairs(
                foreground_pair_ids,
                foreground_values,
                batch_pairs,
                batch_values,
            )
            buffered_pairs.clear()
            buffered_values.clear()
            buffered_count = 0
        print(json.dumps({"foreground_view": view_index, "pairs": int(foreground_pair_ids.size)}))

    candidate_points = foreground_pair_ids // num_tracks
    candidate_tracks = foreground_pair_ids % num_tracks
    offsets = candidate_offsets(candidate_points, num_gaussians)
    total_visible_contribution = np.zeros(foreground_values.shape, dtype=np.float32)
    foreground_votes = np.zeros(foreground_values.shape, dtype=np.uint16)
    total_votes = np.zeros(foreground_values.shape, dtype=np.uint16)

    for view_index, (entry, tracks) in enumerate(zip(entries, segment_tracks)):
        cache = load_cache(cache_dir, entry)
        signatures = (
            signature_provider.get(cache, entry, view_index)
            if args.membership_mode == "saga_union"
            else None
        )
        visible = np.zeros(num_tracks, dtype=bool)
        visible[tracks[tracks >= 0]] = True
        points = cache["aggregate_ids"].numpy().astype(np.int64, copy=False)
        totals = cache["aggregate_weights"].numpy().astype(np.float32, copy=False)
        valid = (points >= 0) & (points < num_gaussians) & (totals > 0.0)
        points = points[valid]
        totals = totals[valid]
        if points.size > 1 and np.any(points[1:] < points[:-1]):
            order = np.argsort(points, kind="stable")
            points = points[order]
            totals = totals[order]
        positions, source_rows = candidate_positions_for_points(points, offsets)
        if positions.size:
            accepted = visible[candidate_tracks[positions]]
            positions = positions[accepted]
            view_totals = totals[source_rows[accepted]]
            total_visible_contribution[positions] += view_totals
            vote_valid = view_totals >= args.min_view_contribution
            if vote_valid.any():
                vote_positions = positions[vote_valid]
                vote_totals = view_totals[vote_valid]
                if signatures is not None:
                    view_pairs, view_foreground = foreground_pairs_from_signatures(
                        signatures, tracks, num_tracks
                    )
                else:
                    view_pairs, view_foreground = foreground_pairs(
                        cache, tracks, num_tracks
                    )
                foreground_positions = np.searchsorted(
                    foreground_pair_ids, view_pairs
                )
                found = (
                    (foreground_positions < foreground_pair_ids.size)
                    & (foreground_pair_ids[np.minimum(
                        foreground_positions, foreground_pair_ids.size - 1
                    )] == view_pairs)
                )
                foreground_positions = foreground_positions[found]
                view_foreground = view_foreground[found]
                local = np.searchsorted(vote_positions, foreground_positions)
                matched = (
                    (local < vote_positions.size)
                    & (vote_positions[np.minimum(local, vote_positions.size - 1)] == foreground_positions)
                )
                foreground_by_vote = np.zeros(vote_positions.size, dtype=np.float32)
                foreground_by_vote[local[matched]] = view_foreground[matched]
                total_votes[vote_positions] += 1
                foreground_votes[vote_positions] += (
                    foreground_by_vote > args.view_foreground_ratio * vote_totals
                ).astype(np.uint16)
        print(json.dumps({"background_view": view_index, "candidate_updates": int(positions.size)}))

    memberships = foreground_values / np.maximum(total_visible_contribution, 1e-12)
    memberships = np.clip(memberships, 0.0, 1.0)
    entropy = binary_entropy(foreground_votes, total_votes)
    neutral = np.zeros(entropy.shape, dtype=bool)
    if args.neutral_filter:
        opacity = load_gaussian_opacity(args.geometry_checkpoint, num_gaussians)
        neutral = (
            (entropy > args.entropy_threshold)
            & (opacity[candidate_points] < args.opacity_threshold)
        )
    point_ids, point_scores = pack_top_memberships(
        num_gaussians,
        candidate_points,
        candidate_tracks,
        memberships,
        args.top_m,
        args.membership_threshold,
        args.min_foreground,
        foreground_values,
        neutral,
    )
    point_entropy = selected_candidate_values(
        point_ids,
        candidate_points,
        candidate_tracks,
        entropy,
        num_tracks,
    )
    return (
        point_ids,
        point_scores,
        foreground_values,
        memberships,
        entropy,
        neutral,
        total_votes,
        point_entropy,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--similarity_threshold", type=float, default=0.82)
    parser.add_argument("--min_track_support", type=float, default=32.0)
    parser.add_argument(
        "--track_linking",
        choices=[
            "legacy_last_node",
            "mutual_soft_overlap",
            "semantic_spatial_mutual",
            "trace_memory_mutual",
        ],
        default="legacy_last_node",
    )
    parser.add_argument("--track_window", type=int, default=3)
    parser.add_argument("--min_soft_overlap", type=float, default=0.05)
    parser.add_argument("--memory_signature_points", type=int, default=512)
    parser.add_argument("--max_spatial_distance_ratio", type=float, default=1.5)
    parser.add_argument("--spatial_score_power", type=float, default=1.0)
    parser.add_argument("--min_track_views", type=int, default=1)
    parser.add_argument(
        "--membership_mode",
        choices=["contribution", "saga_union"],
        default="contribution",
    )
    parser.add_argument("--semantic_codebook_dir", default=None)
    parser.add_argument("--semantic_classifier_dir", default=None)
    parser.add_argument("--association_cache_dir", default=None)
    parser.add_argument("--association_fraction", type=float, default=0.2)
    parser.add_argument("--association_max_candidates", type=int, default=2048)
    parser.add_argument("--association_device", default="cuda")
    parser.add_argument("--association_chunk_size", type=int, default=65536)
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--membership_threshold", type=float, default=0.5)
    parser.add_argument("--min_foreground", type=float, default=1e-4)
    parser.add_argument("--min_view_contribution", type=float, default=1e-4)
    parser.add_argument("--view_foreground_ratio", type=float, default=0.5)
    parser.add_argument("--neutral_filter", action="store_true")
    parser.add_argument("--geometry_checkpoint", default=None)
    parser.add_argument("--entropy_threshold", type=float, default=0.9)
    parser.add_argument("--opacity_threshold", type=float, default=0.1)
    parser.add_argument("--view_weighting", choices=["contribution", "information_kl"], default="information_kl")
    parser.add_argument(
        "--rofa_tau",
        type=float,
        default=0.0,
        help="ReLaGS-style per-track Z-score outlier threshold; zero disables it.",
    )
    parser.add_argument("--importance_temperature", type=float, default=1.0)
    parser.add_argument("--max_view_kl", type=float, default=0.02)
    parser.add_argument("--importance_ratio_clip", type=float, default=5.0)
    parser.add_argument("--agreement_power", type=float, default=1.0)
    parser.add_argument("--information_weight", type=float, default=1.0)
    parser.add_argument("--reduce_buffer_pairs", type=int, default=5_000_000)
    parser.add_argument("--max_views", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])

    if not 0.0 <= args.membership_threshold < 1.0:
        raise ValueError("membership_threshold must be in [0, 1)")
    if (
        args.top_m <= 0
        or args.reduce_buffer_pairs <= 0
        or args.track_window <= 0
        or args.memory_signature_points <= 0
        or args.min_track_views <= 0
        or args.association_max_candidates <= 0
        or args.association_chunk_size <= 0
        or args.max_spatial_distance_ratio <= 0.0
        or args.spatial_score_power < 0.0
    ):
        raise ValueError("top_m, buffers, track window, and track views must be positive")
    if not 0.0 <= args.min_soft_overlap <= 1.0:
        raise ValueError("min_soft_overlap must be in [0, 1]")
    if not 0.0 < args.association_fraction <= 1.0:
        raise ValueError("association_fraction must be in (0, 1]")
    if args.semantic_codebook_dir and args.semantic_classifier_dir:
        raise ValueError("Choose one semantic association source")
    if args.membership_mode == "saga_union" and not (
        args.semantic_codebook_dir or args.semantic_classifier_dir
    ):
        raise ValueError("saga_union membership requires a semantic association source")
    if args.neutral_filter and not args.geometry_checkpoint:
        raise ValueError("--neutral_filter requires --geometry_checkpoint")
    if args.track_linking == "semantic_spatial_mutual" and not args.geometry_checkpoint:
        raise ValueError("semantic_spatial_mutual requires --geometry_checkpoint")
    if not 0.0 <= args.view_foreground_ratio <= 1.0:
        raise ValueError("view_foreground_ratio must be in [0, 1]")
    if not 0.0 <= args.entropy_threshold <= 1.0:
        raise ValueError("entropy_threshold must be in [0, 1]")
    if not 0.0 <= args.opacity_threshold <= 1.0:
        raise ValueError("opacity_threshold must be in [0, 1]")
    if args.rofa_tau < 0.0:
        raise ValueError("ROFA tau must be non-negative")

    cache_dir = os.path.abspath(args.cache_dir)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("codec_type") != "identity" or int(manifest.get("semantic_dim", 0)) != 512:
        raise ValueError("Full-contribution groups require an identity 512D cache")
    if not manifest.get("raw_contribution_weights", False):
        raise ValueError("Full-contribution groups require raw T*alpha weights")
    entries = manifest["views"]
    if args.max_views > 0:
        entries = entries[: args.max_views]
    if not entries:
        raise ValueError("Cache has no per-view observations")
    num_gaussians = int(manifest["num_gaussians"])
    gaussian_xyz = (
        load_gaussian_xyz(args.geometry_checkpoint, num_gaussians)
        if args.track_linking == "semantic_spatial_mutual"
        else None
    )
    semantic_scorer = (
        ViewClassifierSemanticScorer(
            args.semantic_classifier_dir,
            device=args.association_device,
            chunk_size=args.association_chunk_size,
        )
        if args.semantic_classifier_dir
        else
        ResidualCodebookSemanticScorer(
            args.semantic_codebook_dir,
            device=args.association_device,
            chunk_size=args.association_chunk_size,
        )
        if args.membership_mode == "saga_union"
        else None
    )
    signature_provider = SegmentSignatureProvider(
        args.output_dir,
        num_gaussians,
        args,
        semantic_scorer,
    )

    (
        track_features,
        track_supports,
        segment_tracks,
        importance_summary,
        track_view_counts,
        track_effective_views,
        track_concentration,
        track_reliability,
    ) = build_tracks(
        cache_dir,
        entries,
        num_gaussians,
        args,
        signature_provider,
        gaussian_xyz,
    )
    if not track_features.shape[0]:
        raise ValueError("No supported tracks remain after filtering")
    (
        point_ids,
        point_scores,
        foreground,
        memberships,
        entropy,
        neutral,
        total_votes,
        point_entropy,
    ) = build_membership(
        cache_dir,
        entries,
        segment_tracks,
        num_gaussians,
        int(track_features.shape[0]),
        args,
        signature_provider,
    )

    id_dtype = np.uint16 if track_features.shape[0] <= np.iinfo(np.uint16).max else np.uint32
    invalid_id = int(np.iinfo(id_dtype).max)
    packed_ids = np.full(point_ids.shape, invalid_id, dtype=id_dtype)
    valid = point_ids >= 0
    packed_ids[valid] = point_ids[valid].astype(id_dtype)
    packed_weights = np.rint(np.clip(point_scores, 0.0, 1.0) * 255.0).astype(np.uint8)
    packed_weights[~valid] = 0
    packed_entropy = np.rint(np.clip(point_entropy, 0.0, 1.0) * 255.0).astype(np.uint8)
    packed_entropy[~valid] = 0
    valid_points = valid.any(axis=1)
    track_features = track_features.astype(np.float16)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "group_codebook.npy"), track_features)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), packed_weights)
    np.save(os.path.join(output_dir, "point_group_entropy.npy"), packed_entropy)
    np.save(os.path.join(output_dir, "group_reliability.npy"), track_reliability.astype(np.float16))
    np.save(os.path.join(output_dir, "group_view_counts.npy"), track_view_counts.astype(np.uint16))
    np.save(os.path.join(output_dir, "group_effective_views.npy"), track_effective_views.astype(np.float16))
    np.save(os.path.join(output_dir, "group_concentration.npy"), track_concentration.astype(np.float16))
    metadata_bytes = int(
        packed_entropy.nbytes
        + track_reliability.astype(np.float16).nbytes
        + track_view_counts.astype(np.uint16).nbytes
        + track_effective_views.astype(np.float16).nbytes
        + track_concentration.astype(np.float16).nbytes
    )
    storage_bytes = int(
        track_features.nbytes + packed_ids.nbytes + packed_weights.nbytes + metadata_bytes
    )
    selected_scores = point_scores[valid]
    result = {
        "format_version": 1,
        "representation": "compact_group_hierarchy",
        "hierarchy_type": "full_contribution_soft_group_membership",
        "num_gaussians": num_gaussians,
        "num_group_codes": int(track_features.shape[0]),
        "feature_dim": int(track_features.shape[1]),
        "top_m": int(args.top_m),
        "group_codebook": "group_codebook.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_entropy": "point_group_entropy.npy",
        "group_reliability": "group_reliability.npy",
        "group_view_counts": "group_view_counts.npy",
        "group_effective_views": "group_effective_views.npy",
        "group_concentration": "group_concentration.npy",
        "id_dtype": str(packed_ids.dtype),
        "invalid_id": invalid_id,
        "weight_dtype": "uint8_membership_probability",
        "covered_fraction": float(valid_points.mean()),
        "mean_active_groups_per_covered_point": float(valid[valid_points].sum(axis=1).mean()) if valid_points.any() else 0.0,
        "mean_selected_membership": float(selected_scores.mean()) if selected_scores.size else 0.0,
        "num_foreground_pairs": int(foreground.size),
        "mean_foreground_support": float(foreground.mean()) if foreground.size else 0.0,
        "mean_candidate_membership": float(memberships.mean()) if memberships.size else 0.0,
        "mean_candidate_entropy": float(entropy.mean()) if entropy.size else 0.0,
        "mean_candidate_view_votes": float(total_votes.mean()) if total_votes.size else 0.0,
        "neutral_pair_fraction": float(neutral.mean()) if neutral.size else 0.0,
        "num_neutral_pairs": int(neutral.sum()),
        "mean_track_support": float(track_supports.mean()),
        "mean_track_view_count": float(track_view_counts.mean()),
        "mean_track_effective_views": float(track_effective_views.mean()),
        "mean_track_concentration": float(track_concentration.mean()),
        "mean_track_reliability": float(track_reliability.mean()),
        "view_importance": importance_summary,
        "association": signature_provider.summary(),
        "storage": {
            "group_codebook_bytes_fp16": int(track_features.nbytes),
            "point_group_id_bytes": int(packed_ids.nbytes),
            "point_group_weight_bytes": int(packed_weights.nbytes),
            "reliability_metadata_bytes": metadata_bytes,
            "total_semantic_bytes": storage_bytes,
            "bytes_per_gaussian_amortized": float(storage_bytes / num_gaussians),
        },
        "source": {
            "cache_dir": cache_dir,
            "views": len(entries),
            "topk": int(manifest["topk"]),
            "raw_contribution_weights": True,
            "similarity_threshold": args.similarity_threshold,
            "min_track_support": args.min_track_support,
            "track_linking": args.track_linking,
            "track_window": args.track_window,
            "memory_signature_points": args.memory_signature_points,
            "min_soft_overlap": args.min_soft_overlap,
            "max_spatial_distance_ratio": args.max_spatial_distance_ratio,
            "spatial_score_power": args.spatial_score_power,
            "min_track_views": args.min_track_views,
            "membership_mode": args.membership_mode,
            "semantic_codebook_dir": os.path.abspath(args.semantic_codebook_dir)
            if args.semantic_codebook_dir
            else None,
            "semantic_classifier_dir": os.path.abspath(args.semantic_classifier_dir)
            if args.semantic_classifier_dir
            else None,
            "association_cache_dir": signature_provider.cache_dir
            if args.membership_mode == "saga_union"
            else None,
            "association_fraction": args.association_fraction,
            "association_max_candidates": args.association_max_candidates,
            "membership_threshold": args.membership_threshold,
            "min_foreground": args.min_foreground,
            "min_view_contribution": args.min_view_contribution,
            "view_foreground_ratio": args.view_foreground_ratio,
            "neutral_filter": bool(args.neutral_filter),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint)
            if args.geometry_checkpoint
            else None,
            "entropy_threshold": args.entropy_threshold,
            "opacity_threshold": args.opacity_threshold,
            "rofa_tau": args.rofa_tau,
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
