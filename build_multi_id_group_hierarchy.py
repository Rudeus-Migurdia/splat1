#!/usr/bin/env python
"""Build a 3D-first part/object hierarchy from resident multi-ID semantics."""

import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

from build_gaussian_superpoint_support import (
    BoundedUnionFind,
    build_knn,
    compact_components,
    load_geometry,
    project_codebook,
    reconstruct_projected_semantics,
)
from train_joint_query_preserving_vocabulary import FixedSharedAssignment


def multi_mode_set_similarity(first_base, first_candidate, second_base, second_candidate):
    """Symmetric Chamfer cosine between two unordered two-mode signatures."""
    bb = np.sum(first_base * second_base, axis=-1)
    bc = np.sum(first_base * second_candidate, axis=-1)
    cb = np.sum(first_candidate * second_base, axis=-1)
    cc = np.sum(first_candidate * second_candidate, axis=-1)
    return 0.25 * (
        np.maximum(bb, bc)
        + np.maximum(cb, cc)
        + np.maximum(bb, cb)
        + np.maximum(bc, cc)
    )


def geometry_edges(
    neighbors,
    distances,
    rgb,
    log_scale,
    spatial_radius_factor,
    rgb_threshold,
    log_scale_threshold,
    start,
    end,
):
    rows = np.arange(start, end, dtype=np.int32)[:, None]
    adjacent = neighbors[start:end]
    edge_distance = distances[start:end]
    radius = distances[:, -1]
    valid = (adjacent >= 0) & (rows < adjacent)
    radius_limit = spatial_radius_factor * np.minimum(
        radius[start:end, None], radius[adjacent]
    )
    valid &= edge_distance <= radius_limit
    valid &= (
        np.linalg.norm(rgb[start:end, None, :] - rgb[adjacent], axis=-1)
        <= rgb_threshold
    )
    valid &= (
        np.abs(log_scale[start:end, None] - log_scale[adjacent])
        <= log_scale_threshold
    )
    local_rows, slots = np.nonzero(valid)
    first = local_rows.astype(np.int64) + start
    second = adjacent[local_rows, slots].astype(np.int64)
    return first, second


def build_multi_id_hierarchy(
    neighbors,
    distances,
    rgb,
    log_scale,
    base_semantics,
    candidate_semantics,
    spatial_radius_factor,
    rgb_threshold,
    log_scale_threshold,
    part_base_threshold,
    part_set_threshold,
    object_base_threshold,
    object_set_threshold,
    maximum_part_size,
    maximum_object_size,
    chunk_size,
):
    count = neighbors.shape[0]
    part_union = BoundedUnionFind(count)
    geometry_edge_count = 0
    part_edge_count = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        first, second = geometry_edges(
            neighbors,
            distances,
            rgb,
            log_scale,
            spatial_radius_factor,
            rgb_threshold,
            log_scale_threshold,
            start,
            end,
        )
        geometry_edge_count += int(first.size)
        if not first.size:
            continue
        base_similarity = np.sum(
            base_semantics[first].astype(np.float32)
            * base_semantics[second].astype(np.float32),
            axis=-1,
        )
        set_similarity = multi_mode_set_similarity(
            base_semantics[first].astype(np.float32),
            candidate_semantics[first].astype(np.float32),
            base_semantics[second].astype(np.float32),
            candidate_semantics[second].astype(np.float32),
        )
        accepted = (base_similarity >= part_base_threshold) & (
            set_similarity >= part_set_threshold
        )
        part_edge_count += int(accepted.sum())
        for first_id, second_id in zip(first[accepted], second[accepted]):
            part_union.union(first_id, second_id, maximum_part_size)

    part_labels = compact_components(part_union)
    part_count = int(part_labels.max()) + 1 if part_labels.size else 0
    part_sizes = np.bincount(part_labels, minlength=part_count).astype(np.int32)
    object_union = BoundedUnionFind(part_count)
    object_union.size = part_sizes.copy()
    object_edge_count = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        first, second = geometry_edges(
            neighbors,
            distances,
            rgb,
            log_scale,
            spatial_radius_factor,
            rgb_threshold,
            log_scale_threshold,
            start,
            end,
        )
        if not first.size:
            continue
        first_parts = part_labels[first]
        second_parts = part_labels[second]
        cross_part = first_parts != second_parts
        if not cross_part.any():
            continue
        first = first[cross_part]
        second = second[cross_part]
        first_parts = first_parts[cross_part]
        second_parts = second_parts[cross_part]
        base_similarity = np.sum(
            base_semantics[first].astype(np.float32)
            * base_semantics[second].astype(np.float32),
            axis=-1,
        )
        set_similarity = multi_mode_set_similarity(
            base_semantics[first].astype(np.float32),
            candidate_semantics[first].astype(np.float32),
            base_semantics[second].astype(np.float32),
            candidate_semantics[second].astype(np.float32),
        )
        accepted = (base_similarity >= object_base_threshold) & (
            set_similarity >= object_set_threshold
        )
        object_edge_count += int(accepted.sum())
        for first_part, second_part in zip(
            first_parts[accepted], second_parts[accepted]
        ):
            object_union.union(first_part, second_part, maximum_object_size)

    object_part_labels = compact_components(object_union)
    object_labels = object_part_labels[part_labels]
    return {
        "part_labels": part_labels,
        "object_labels": object_labels.astype(np.int32),
        "geometry_edges": geometry_edge_count,
        "part_edges": part_edge_count,
        "object_edges": object_edge_count,
    }


def component_density(labels, positives):
    labels = np.asarray(labels, dtype=np.int64)
    positives = np.asarray(positives, dtype=bool)
    count = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, minlength=count).astype(np.int32)
    positive_counts = np.bincount(
        labels,
        weights=positives.astype(np.float32),
        minlength=count,
    ).astype(np.float32)
    density = positive_counts[labels] / np.maximum(sizes[labels], 1)
    other_size = sizes[labels] - 1
    other_positive = positive_counts[labels] - positives.astype(np.float32)
    leave_one_out = np.divide(
        other_positive,
        np.maximum(other_size, 1),
        out=np.zeros_like(other_positive),
        where=other_size > 0,
    )
    return density.astype(np.float32), leave_one_out.astype(np.float32), sizes


def hierarchy_route_reliability(
    part_labels,
    object_labels,
    candidate_mask,
    min_part_size=3,
    min_object_size=8,
    min_part_density=0.5,
    min_object_density=0.25,
):
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    part_density, part_loo, part_sizes = component_density(
        part_labels, candidate_mask
    )
    object_density, object_loo, object_sizes = component_density(
        object_labels, candidate_mask
    )
    support = np.sqrt(part_density * object_density).astype(np.float32)
    expandable = (
        (part_sizes[part_labels] >= min_part_size)
        & (object_sizes[object_labels] >= min_object_size)
        & (part_density >= min_part_density)
        & (object_density >= min_object_density)
    )
    expansion = np.where(expandable, support, 0.0).astype(np.float32)
    expansion[candidate_mask] = 1.0

    consensus_support = np.sqrt(part_loo * object_loo).astype(np.float32)
    consensus = np.where(candidate_mask, consensus_support, 0.0).astype(np.float32)
    return {
        "expansion": expansion,
        "consensus": consensus,
        "part_density": part_density,
        "object_density": object_density,
        "part_sizes": part_sizes,
        "object_sizes": object_sizes,
    }


def component_summary(labels):
    count = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, minlength=count)
    return {
        "count": count,
        "singleton_fraction": float((sizes == 1).mean()) if sizes.size else 0.0,
        "size_quantiles": {
            str(q): float(np.quantile(sizes, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--base_artifact_dir", required=True)
    parser.add_argument("--candidate_artifact_dir", required=True)
    parser.add_argument("--candidate_mask", required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--spatial_radius_factor", type=float, default=1.5)
    parser.add_argument("--rgb_threshold", type=float, default=0.15)
    parser.add_argument("--log_scale_threshold", type=float, default=0.7)
    parser.add_argument("--semantic_dim", type=int, default=64)
    parser.add_argument("--part_base_threshold", type=float, default=0.88)
    parser.add_argument("--part_set_threshold", type=float, default=0.90)
    parser.add_argument("--object_base_threshold", type=float, default=0.80)
    parser.add_argument("--object_set_threshold", type=float, default=0.85)
    parser.add_argument("--maximum_part_size", type=int, default=128)
    parser.add_argument("--maximum_object_size", type=int, default=2048)
    parser.add_argument("--min_part_size", type=int, default=3)
    parser.add_argument("--min_object_size", type=int, default=8)
    parser.add_argument("--min_part_density", type=float, default=0.5)
    parser.add_argument("--min_object_density", type=float, default=0.25)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.neighbors <= 1 or args.semantic_dim <= 1 or args.chunk_size <= 0:
        raise ValueError("Neighbor, semantic, and chunk sizes must be positive")
    if args.maximum_part_size <= 1 or args.maximum_object_size < args.maximum_part_size:
        raise ValueError("Object capacity must be at least the part capacity")
    for name in (
        "part_base_threshold",
        "part_set_threshold",
        "object_base_threshold",
        "object_set_threshold",
    ):
        if not -1.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name} must be in [-1, 1]")
    for name in ("min_part_density", "min_object_density"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name} must be in [0, 1]")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse multi-ID group hierarchy: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    base = FixedSharedAssignment(args.base_artifact_dir)
    candidate = FixedSharedAssignment(args.candidate_artifact_dir)
    if base.num_gaussians != candidate.num_gaussians:
        raise ValueError("Base and candidate assignments must match")
    valid_mask = base.valid_mask & candidate.valid_mask
    valid_global = np.flatnonzero(valid_mask)
    resident_counts = (base.ids[valid_global] >= 0).sum(axis=1) + (
        candidate.ids[valid_global] >= 0
    ).sum(axis=1)
    novelty = np.load(os.path.abspath(args.candidate_mask)).astype(bool)
    if novelty.shape != (base.num_gaussians,):
        raise ValueError("Candidate mask does not match the assignments")
    novelty_valid = novelty[valid_global]

    xyz, rgb, log_scale, checkpoint_iteration = load_geometry(
        args.geometry_checkpoint, base.num_gaussians
    )
    xyz = xyz[valid_global]
    rgb = rgb[valid_global]
    log_scale = log_scale[valid_global]

    codebook_path = os.path.join(
        base.artifact_dir, base.manifest["codebook_files"][0]
    )
    codebook = np.load(codebook_path).astype(np.float32)
    projection_device = (
        "cuda" if args.faiss_gpu and torch.cuda.is_available() else "cpu"
    )
    projected_codebook = project_codebook(
        codebook, args.semantic_dim, args.seed, projection_device
    )
    base_semantics = reconstruct_projected_semantics(
        base, projected_codebook, valid_global, args.chunk_size
    )
    candidate_semantics = reconstruct_projected_semantics(
        candidate, projected_codebook, valid_global, args.chunk_size
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    neighbors, distances, resources, knn_backend = build_knn(
        xyz,
        args.neighbors,
        args.chunk_size,
        args.faiss_gpu,
        args.knn_workers,
    )
    hierarchy = build_multi_id_hierarchy(
        neighbors,
        distances,
        rgb,
        log_scale,
        base_semantics,
        candidate_semantics,
        args.spatial_radius_factor,
        args.rgb_threshold,
        args.log_scale_threshold,
        args.part_base_threshold,
        args.part_set_threshold,
        args.object_base_threshold,
        args.object_set_threshold,
        args.maximum_part_size,
        args.maximum_object_size,
        args.chunk_size,
    )
    del resources
    routes = hierarchy_route_reliability(
        hierarchy["part_labels"],
        hierarchy["object_labels"],
        novelty_valid,
        args.min_part_size,
        args.min_object_size,
        args.min_part_density,
        args.min_object_density,
    )

    part_global = np.full(base.num_gaussians, -1, dtype=np.int32)
    object_global = np.full(base.num_gaussians, -1, dtype=np.int32)
    expansion_global = np.zeros(base.num_gaussians, dtype=np.float32)
    consensus_global = np.zeros(base.num_gaussians, dtype=np.float32)
    part_global[valid_global] = hierarchy["part_labels"]
    object_global[valid_global] = hierarchy["object_labels"]
    expansion_global[valid_global] = routes["expansion"]
    consensus_global[valid_global] = routes["consensus"]
    np.save(os.path.join(output_dir, "part_group_ids.npy"), part_global)
    np.save(os.path.join(output_dir, "object_group_ids.npy"), object_global)
    np.save(os.path.join(output_dir, "route_expand.npy"), expansion_global)
    np.save(os.path.join(output_dir, "route_consensus.npy"), consensus_global)

    expanded = (routes["expansion"] > 0.0) & ~novelty_valid
    retained = routes["consensus"] > 0.0
    manifest = {
        "format_version": 1,
        "representation": "resident_multi_id_part_object_hierarchy",
        "source": "training geometry/RGB, four resident shared-codebook IDs, and A14 novelty only",
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "base_artifact_dir": base.artifact_dir,
        "candidate_artifact_dir": candidate.artifact_dir,
        "candidate_mask": os.path.abspath(args.candidate_mask),
        "num_gaussians": base.num_gaussians,
        "num_valid_gaussians": int(valid_global.size),
        "resident_id_slots_per_valid_gaussian": int(
            base.ids.shape[1] + candidate.ids.shape[1]
        ),
        "mean_resident_ids_per_valid_gaussian": float(resident_counts.mean()),
        "full_four_id_fraction_valid": float(
            (resident_counts == base.ids.shape[1] + candidate.ids.shape[1]).mean()
        ),
        "original_candidate_fraction_valid": float(novelty_valid.mean()),
        "expanded_fraction_valid": float(expanded.mean()),
        "consensus_retained_fraction_of_candidates": float(
            retained[novelty_valid].mean() if novelty_valid.any() else 0.0
        ),
        "mean_expand_reliability": float(routes["expansion"].mean()),
        "mean_consensus_reliability_on_candidates": float(
            routes["consensus"][novelty_valid].mean()
            if novelty_valid.any()
            else 0.0
        ),
        "knn_backend": knn_backend,
        "edges": {
            "geometry": hierarchy["geometry_edges"],
            "part": hierarchy["part_edges"],
            "object": hierarchy["object_edges"],
        },
        "part": component_summary(hierarchy["part_labels"]),
        "object": component_summary(hierarchy["object_labels"]),
        "files": {
            "part_group_ids": "part_group_ids.npy",
            "object_group_ids": "object_group_ids.npy",
            "route_expand": "route_expand.npy",
            "route_consensus": "route_consensus.npy",
        },
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
