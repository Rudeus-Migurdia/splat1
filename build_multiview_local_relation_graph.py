#!/usr/bin/env python
"""Build a 3D-first signed Gaussian graph from multiview SAM agreement."""

import hashlib
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch

from build_gaussian_superpoint_support import build_knn, load_geometry


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dominant_gaussian_segments(cache, num_gaussians, minimum_fraction):
    """Assign each Gaussian its contribution-weighted dominant segment in a view."""
    point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
    point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
    segment_ids = cache["segment_ids"].numpy().astype(np.int64, copy=False)
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
    pairs = points * segment_count + repeated_segments
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
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


def finalize_signed_relations(
    positive_mass,
    negative_mass,
    observation_count,
    minimum_split_views,
    minimum_absolute_relation,
):
    """Convert odd/even-view evidence into a conservative signed edge weight."""
    if not (
        positive_mass.shape == negative_mass.shape == observation_count.shape
        and positive_mass.shape[0] == 2
    ):
        raise ValueError("Split relation tensors must have shape [2, N, K]")
    total = positive_mass + negative_mass
    signed = np.divide(
        positive_mass - negative_mass,
        total,
        out=np.zeros_like(total, dtype=np.float32),
        where=total > 0.0,
    )
    minimum_views = observation_count.min(axis=0).astype(np.float32)
    view_balance = 2.0 * observation_count.min(axis=0) / np.maximum(
        observation_count.sum(axis=0), 1
    )
    mass_balance = 2.0 * total.min(axis=0) / np.maximum(total.sum(axis=0), 1e-12)
    split_agreement = np.clip(1.0 - 0.5 * np.abs(signed[0] - signed[1]), 0.0, 1.0)
    support_gate = np.clip(minimum_views / float(minimum_split_views), 0.0, 1.0)
    consistent_sign = (signed[0] * signed[1]) > 0.0
    magnitude = np.minimum(np.abs(signed[0]), np.abs(signed[1]))
    reliability = (
        np.sqrt(np.clip(view_balance * mass_balance, 0.0, 1.0))
        * split_agreement
        * support_gate
    )
    relation = (
        np.sign(signed[0] + signed[1])
        * magnitude
        * reliability
    ).astype(np.float32)
    relation[
        (~consistent_sign)
        | (minimum_views < minimum_split_views)
        | (np.abs(relation) < minimum_absolute_relation)
    ] = 0.0
    return relation, {
        "split_signed": signed,
        "view_balance": view_balance.astype(np.float32),
        "mass_balance": mass_balance.astype(np.float32),
        "split_agreement": split_agreement.astype(np.float32),
        "minimum_split_views": minimum_views,
    }


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return {str(q): 0.0 for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--memory_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--spatial_radius_factor", type=float, default=1.5)
    parser.add_argument("--minimum_dominant_fraction", type=float, default=0.55)
    parser.add_argument("--minimum_split_views", type=int, default=3)
    parser.add_argument("--minimum_absolute_relation", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--expected_memory_seed", type=int, required=True)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.neighbors <= 1 or args.chunk_size <= 0 or args.knn_workers <= 0:
        raise ValueError("Neighbor, chunk, and worker counts must be positive")
    if args.spatial_radius_factor <= 0.0:
        raise ValueError("Spatial radius factor must be positive")
    if not 0.0 < args.minimum_dominant_fraction <= 1.0:
        raise ValueError("Dominant fraction must be in (0, 1]")
    if args.minimum_split_views <= 0:
        raise ValueError("Minimum split views must be positive")
    if not 0.0 <= args.minimum_absolute_relation <= 1.0:
        raise ValueError("Minimum absolute relation must be in [0, 1]")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse multiview relation graph: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    memory_dir = os.path.abspath(args.memory_dir)
    memory_manifest_path = os.path.join(memory_dir, "manifest.json")
    with open(memory_manifest_path) as source:
        memory_manifest = json.load(source)
    if memory_manifest.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError("A39 requires an independent four-token memory")
    if int(memory_manifest.get("resident_slots_required", 0)) != 4:
        raise ValueError("A39 requires exactly four resident token slots")
    if int(memory_manifest["reproducibility"]["seed"]) != args.expected_memory_seed:
        raise ValueError("Memory seed does not match the fixed A39 seed")
    num_gaussians = int(memory_manifest["num_gaussians"])

    cache_dir = os.path.abspath(args.cache_dir)
    cache_manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(cache_manifest_path) as source:
        cache_manifest = json.load(source)
    if int(cache_manifest["num_gaussians"]) != num_gaussians:
        raise ValueError("View cache and resident memory Gaussian counts differ")
    if not cache_manifest.get("raw_contribution_weights"):
        raise ValueError("A39 requires raw T*alpha contribution weights")
    if int(cache_manifest.get("topk", 0)) < 45:
        raise ValueError("A39 requires at least top-45 contributors")
    entries = cache_manifest.get("views", [])
    if len(entries) < 2 * args.minimum_split_views:
        raise ValueError("Not enough cached views for split-consistent relations")

    xyz, _, _, checkpoint_iteration = load_geometry(
        os.path.abspath(args.geometry_checkpoint), num_gaussians
    )
    neighbors, distances, resources, knn_backend = build_knn(
        xyz,
        args.neighbors,
        args.chunk_size,
        args.faiss_gpu,
        args.knn_workers,
    )
    del resources
    radius = distances[:, -1]
    spatial_valid = distances <= (
        args.spatial_radius_factor ** 2
        * np.minimum(radius[:, None], radius[neighbors])
    )

    shape = (2, num_gaussians, args.neighbors)
    positive_mass = np.zeros(shape, dtype=np.float32)
    negative_mass = np.zeros(shape, dtype=np.float32)
    observation_count = np.zeros(shape, dtype=np.uint8)
    accepted_gaussians = []

    for entry_index, entry in enumerate(entries):
        payload = torch.load(
            os.path.join(cache_dir, entry["cache"]),
            map_location="cpu",
            weights_only=False,
        )
        segments, confidence, visibility = dominant_gaussian_segments(
            payload,
            num_gaussians,
            args.minimum_dominant_fraction,
        )
        del payload
        accepted_gaussians.append(int((segments >= 0).sum()))
        split = entry_index % 2
        for start in range(0, num_gaussians, args.chunk_size):
            end = min(start + args.chunk_size, num_gaussians)
            adjacent = neighbors[start:end]
            row_segments = segments[start:end, None]
            neighbor_segments = segments[adjacent]
            observed = (
                spatial_valid[start:end]
                & (row_segments >= 0)
                & (neighbor_segments >= 0)
            )
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
            positive_mass[split, start:end] += np.where(same, edge_weight, 0.0)
            negative_mass[split, start:end] += np.where(different, edge_weight, 0.0)
            observation_count[split, start:end] += observed.astype(np.uint8)

    relation, diagnostics = finalize_signed_relations(
        positive_mass,
        negative_mass,
        observation_count,
        args.minimum_split_views,
        args.minimum_absolute_relation,
    )
    packed_relation = np.rint(np.clip(relation, -1.0, 1.0) * 127.0).astype(np.int8)
    np.save(os.path.join(output_dir, "neighbor_ids.npy"), neighbors.astype(np.int32))
    np.save(os.path.join(output_dir, "signed_relation_weights.npy"), packed_relation)

    active = packed_relation != 0
    positive = packed_relation > 0
    negative = packed_relation < 0
    storage_bytes = int(neighbors.astype(np.int32).nbytes + packed_relation.nbytes)
    manifest = {
        "format_version": 1,
        "representation": "multiview_local_signed_gaussian_relation_graph",
        "num_gaussians": num_gaussians,
        "neighbors": args.neighbors,
        "neighbor_ids": "neighbor_ids.npy",
        "signed_relation_weights": "signed_relation_weights.npy",
        "relation_dtype": "int8",
        "relation_scale": 1.0 / 127.0,
        "directed_edge_slots": int(active.size),
        "active_directed_edges": int(active.sum()),
        "positive_directed_edges": int(positive.sum()),
        "negative_directed_edges": int(negative.sum()),
        "active_edge_fraction": float(active.mean()),
        "positive_edge_fraction_active": float(positive.sum() / max(int(active.sum()), 1)),
        "absolute_relation_quantiles": quantiles(np.abs(relation[active])),
        "minimum_split_view_quantiles_active": quantiles(
            diagnostics["minimum_split_views"][active]
        ),
        "mean_accepted_gaussians_per_view": float(np.mean(accepted_gaussians)),
        "accepted_gaussian_fraction_per_view": float(
            np.mean(accepted_gaussians) / num_gaussians
        ),
        "knn_backend": knn_backend,
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "memory_dir": memory_dir,
        "memory_manifest_sha256": file_sha256(memory_manifest_path),
        "cache_dir": cache_dir,
        "cache_manifest_sha256": file_sha256(cache_manifest_path),
        "source_contract": {
            "four_peer_tokens_unchanged": True,
            "cross_view_segment_ids_never_matched": True,
            "edge_candidates_are_3d_knn_first": True,
            "same_view_same_segment_is_positive": True,
            "same_view_different_segment_is_negative": True,
            "view_weight_is_gaussian_contribution_based": True,
            "odd_even_split_consistency_required": True,
            "evaluation_queries_or_labels_used": False,
        },
        "storage": {
            "neighbor_id_bytes": int(neighbors.astype(np.int32).nbytes),
            "relation_weight_bytes": int(packed_relation.nbytes),
            "total_bytes": storage_bytes,
            "bytes_per_gaussian": float(storage_bytes / num_gaussians),
        },
        "args": vars(args),
        "elapsed_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
