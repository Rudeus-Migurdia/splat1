#!/usr/bin/env python
"""Build a four-level SAM semantic memory with split-consistent source selection.

The artifact deliberately keeps the L0--L3 codebooks independent.  A Gaussian
stores one resident token per level, while the evaluator performs query-time
selection and fusion instead of reducing the hierarchy to one static feature.
"""

import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np


LEVEL_NAMES = ("sam_l0", "sam_l1", "sam_l2", "sam_l3")
LEVEL_ROLES = ("coarse", "object", "part", "micro")
OLD_SOURCE = np.uint8(0)
AUXILIARY_SOURCE = np.uint8(1)
INVALID_SOURCE = np.uint8(255)


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return {}
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def resolve_group_source(
    old_reliability,
    old_supported,
    auxiliary_reliability,
    auxiliary_supported,
    margin=0.0,
):
    """Select exactly one source per group from split-consistency reliability."""
    old_reliability = np.asarray(old_reliability, dtype=np.float32)
    auxiliary_reliability = np.asarray(auxiliary_reliability, dtype=np.float32)
    old_supported = np.asarray(old_supported, dtype=bool)
    auxiliary_supported = np.asarray(auxiliary_supported, dtype=bool)
    if not (
        old_reliability.shape
        == auxiliary_reliability.shape
        == old_supported.shape
        == auxiliary_supported.shape
    ):
        raise ValueError("Source reliability and support arrays must have matching shapes")
    if margin < 0.0:
        raise ValueError("Source-selection margin must be non-negative")

    selected = np.full(old_reliability.shape, INVALID_SOURCE, dtype=np.uint8)
    selected[old_supported] = OLD_SOURCE
    take_auxiliary = auxiliary_supported & (
        ~old_supported | (auxiliary_reliability > old_reliability + margin)
    )
    selected[take_auxiliary] = AUXILIARY_SOURCE
    selected[~old_supported & auxiliary_supported] = AUXILIARY_SOURCE
    return selected


def parent_group_lookup(parent_labels, child_labels):
    """Assign every child component its modal parent component deterministically."""
    parent_labels = np.asarray(parent_labels, dtype=np.int64)
    child_labels = np.asarray(child_labels, dtype=np.int64)
    if parent_labels.shape != child_labels.shape:
        raise ValueError("Parent and child labels must have matching shapes")
    if not child_labels.size:
        return np.empty(0, dtype=np.int32)
    child_count = int(child_labels.max()) + 1
    result = np.full(child_count, -1, dtype=np.int32)
    order = np.lexsort((parent_labels, child_labels))
    sorted_children = child_labels[order]
    starts = np.r_[0, np.flatnonzero(sorted_children[1:] != sorted_children[:-1]) + 1]
    ends = np.r_[starts[1:], sorted_children.size]
    for start, end in zip(starts, ends):
        parents, counts = np.unique(parent_labels[order[start:end]], return_counts=True)
        result[int(sorted_children[start])] = int(parents[np.argmax(counts)])
    return result


def validate_level_configuration(thresholds, maximum_sizes, minimum_sizes):
    if not (
        len(thresholds) == len(maximum_sizes) == len(minimum_sizes) == len(LEVEL_NAMES)
    ):
        raise ValueError("All level configurations must provide exactly L0--L3 values")
    if any(not -1.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("Semantic thresholds must be in [-1, 1]")
    if any(size <= 1 for size in maximum_sizes) or any(size <= 0 for size in minimum_sizes):
        raise ValueError("Group sizes must be positive and maximum sizes must exceed one")
    if any(minimum > maximum for minimum, maximum in zip(minimum_sizes, maximum_sizes)):
        raise ValueError("Every minimum group size must fit within its maximum size")
    if any(first > second for first, second in zip(thresholds, thresholds[1:])):
        raise ValueError("L0--L3 thresholds must become at least as selective toward L3")
    if any(first < second for first, second in zip(maximum_sizes, maximum_sizes[1:])):
        raise ValueError("L0--L3 maximum group sizes must not grow toward L3")


def build_level_groups(
    neighbors,
    distances,
    rgb,
    log_scale,
    semantics,
    parent_labels,
    spatial_radius_factor,
    rgb_threshold,
    log_scale_threshold,
    semantic_threshold,
    maximum_size,
    chunk_size,
):
    """Build geometry-aware semantic components, constrained to a parent level."""
    from build_gaussian_superpoint_support import BoundedUnionFind, compact_components

    count = int(neighbors.shape[0])
    union = BoundedUnionFind(count)
    radius = distances[:, -1]
    geometry_edges = 0
    semantic_edges = 0
    parent_edges = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        rows = np.arange(start, end, dtype=np.int32)[:, None]
        adjacent = neighbors[start:end]
        edge_distance = distances[start:end]
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
        row_ids, slots = np.nonzero(valid)
        first = row_ids.astype(np.int64) + start
        second = adjacent[row_ids, slots].astype(np.int64)
        geometry_edges += int(first.size)
        if not first.size:
            continue
        if parent_labels is not None:
            same_parent = parent_labels[first] == parent_labels[second]
            parent_edges += int(same_parent.sum())
            first = first[same_parent]
            second = second[same_parent]
        else:
            parent_edges += int(first.size)
        if not first.size:
            continue
        cosine = np.sum(semantics[first] * semantics[second], axis=-1)
        accepted = cosine >= semantic_threshold
        semantic_edges += int(accepted.sum())
        for first_id, second_id in zip(first[accepted], second[accepted]):
            union.union(int(first_id), int(second_id), maximum_size)
    labels = compact_components(union)
    sizes = np.bincount(labels, minlength=int(labels.max()) + 1).astype(np.int32)
    return labels, {
        "num_groups": int(sizes.size),
        "group_size_quantiles": quantiles(sizes),
        "singleton_fraction": float((sizes == 1).mean()) if sizes.size else 0.0,
        "geometry_edges": geometry_edges,
        "parent_compatible_edges": parent_edges,
        "semantic_edges": semantic_edges,
    }


def combined_split_features(payload):
    import torch
    from torch.nn import functional as F

    features = payload["split_initial_features"].detach().cpu().float()
    weights = payload["split_weights"].detach().cpu().float()
    if features.ndim != 3 or features.shape[0] != 2 or weights.shape != features.shape[:2]:
        raise ValueError("Every source must contain two matching split feature tensors")
    combined = F.normalize(
        features[0] * weights[0].unsqueeze(-1)
        + features[1] * weights[1].unsqueeze(-1),
        dim=-1,
    )
    combined[weights.sum(dim=0) <= 0.0] = 0.0
    return combined.numpy().astype(np.float32), features.shape


def selected_group_features(old, auxiliary, source_ids):
    old_features = old["features"].float().cpu().numpy()
    auxiliary_features = auxiliary["features"].float().cpu().numpy()
    old_reliability = old["reliability"].cpu().numpy().astype(np.float32)
    auxiliary_reliability = auxiliary["reliability"].cpu().numpy().astype(np.float32)
    features = np.zeros_like(old_features, dtype=np.float32)
    reliability = np.zeros_like(old_reliability, dtype=np.float32)
    old_selected = source_ids == OLD_SOURCE
    auxiliary_selected = source_ids == AUXILIARY_SOURCE
    features[old_selected] = old_features[old_selected]
    features[auxiliary_selected] = auxiliary_features[auxiliary_selected]
    reliability[old_selected] = old_reliability[old_selected]
    reliability[auxiliary_selected] = auxiliary_reliability[auxiliary_selected]
    return normalize(features), reliability


def main():
    import torch

    from build_gaussian_superpoint_support import build_knn, load_geometry
    from build_hierarchical_group_semantic_codebook import aggregate_source

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--old_consensus", required=True)
    for level in range(4):
        parser.add_argument(f"--sam_l{level}_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--spatial_radius_factor", type=float, default=1.5)
    parser.add_argument("--rgb_threshold", type=float, default=0.15)
    parser.add_argument("--log_scale_threshold", type=float, default=0.7)
    parser.add_argument(
        "--semantic_thresholds", nargs=4, type=float, default=[0.76, 0.82, 0.87, 0.91]
    )
    parser.add_argument(
        "--maximum_group_sizes", nargs=4, type=int, default=[2048, 512, 128, 32]
    )
    parser.add_argument(
        "--minimum_group_sizes", nargs=4, type=int, default=[16, 8, 4, 2]
    )
    parser.add_argument("--stability_floor", type=float, default=0.5)
    parser.add_argument("--minimum_reliability", type=float, default=0.25)
    parser.add_argument("--source_margin", type=float, default=0.0)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.neighbors <= 1 or args.chunk_size <= 0 or args.knn_workers <= 0:
        raise ValueError("Neighbor, chunk, and worker counts must be positive")
    if args.spatial_radius_factor <= 0.0 or args.rgb_threshold <= 0.0 or args.log_scale_threshold <= 0.0:
        raise ValueError("Geometry thresholds must be positive")
    if not -1.0 <= args.stability_floor < 1.0:
        raise ValueError("Stability floor must be in [-1, 1)")
    if not 0.0 <= args.minimum_reliability <= 1.0:
        raise ValueError("Minimum reliability must be in [0, 1]")
    validate_level_configuration(
        args.semantic_thresholds, args.maximum_group_sizes, args.minimum_group_sizes
    )

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse hierarchical semantic memory: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    old_path = os.path.abspath(args.old_consensus)
    old_payload = torch.load(old_path, map_location="cpu")
    old_features, old_shape = combined_split_features(old_payload)
    num_gaussians = int(old_shape[1])
    feature_dim = int(old_shape[2])
    level_payloads = []
    level_features = []
    level_paths = []
    for level in range(4):
        path = os.path.abspath(getattr(args, f"sam_l{level}_consensus"))
        payload = torch.load(path, map_location="cpu")
        features, shape = combined_split_features(payload)
        if shape[1:] != old_shape[1:]:
            raise ValueError(f"SAM L{level} consensus does not match the Old source")
        level_payloads.append(payload)
        level_features.append(features)
        level_paths.append(path)

    xyz, rgb, log_scale, checkpoint_iteration = load_geometry(
        os.path.abspath(args.geometry_checkpoint), num_gaussians
    )
    neighbors, distances, resources, knn_backend = build_knn(
        xyz, args.neighbors, args.chunk_size, args.faiss_gpu, args.knn_workers
    )
    del resources

    raw_labels = []
    level_groups = []
    parent = None
    for level, name in enumerate(LEVEL_NAMES):
        labels, diagnostics = build_level_groups(
            neighbors,
            distances,
            rgb,
            log_scale,
            level_features[level],
            parent,
            args.spatial_radius_factor,
            args.rgb_threshold,
            args.log_scale_threshold,
            args.semantic_thresholds[level],
            args.maximum_group_sizes[level],
            args.chunk_size,
        )
        diagnostics.update(
            {
                "name": name,
                "semantic_role": LEVEL_ROLES[level],
                "sam_feature_level": level,
                "semantic_threshold": float(args.semantic_thresholds[level]),
                "maximum_group_size": int(args.maximum_group_sizes[level]),
                "minimum_group_size": int(args.minimum_group_sizes[level]),
            }
        )
        raw_labels.append(labels)
        level_groups.append(diagnostics)
        parent = labels

    point_ids_by_level = []
    point_weights_by_level = []
    token_semantic_ids = []
    token_levels = []
    token_reliability = []
    token_source = []
    token_parent = []
    level_codebooks = []
    level_selection = []
    previous_group_to_token = None
    previous_labels = None
    token_offset = 0
    for level, (name, labels, payload) in enumerate(
        zip(LEVEL_NAMES, raw_labels, level_payloads)
    ):
        group_count = int(labels.max()) + 1
        sizes = np.bincount(labels, minlength=group_count).astype(np.int32)
        old_group = aggregate_source(
            old_payload,
            labels,
            group_count,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        auxiliary_group = aggregate_source(
            payload,
            labels,
            group_count,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        old_supported = old_group["supported"].cpu().numpy()
        auxiliary_supported = auxiliary_group["supported"].cpu().numpy()
        source_ids = resolve_group_source(
            old_group["reliability"].cpu().numpy(),
            old_supported,
            auxiliary_group["reliability"].cpu().numpy(),
            auxiliary_supported,
            args.source_margin,
        )
        selected_features, selected_reliability = selected_group_features(
            old_group, auxiliary_group, source_ids
        )
        selected = (
            (source_ids != INVALID_SOURCE)
            & (sizes >= args.minimum_group_sizes[level])
            & (selected_reliability >= args.minimum_reliability)
        )
        selected_groups = np.flatnonzero(selected)
        if not selected_groups.size:
            raise ValueError(
                f"{name} has no valid tokens; adjust only training-derived group or gate thresholds"
            )
        group_to_token = np.full(group_count, -1, dtype=np.int64)
        group_to_token[selected_groups] = token_offset + np.arange(selected_groups.size)
        point_tokens = group_to_token[labels]
        point_ids_by_level.append(point_tokens)
        point_weights_by_level.append(np.where(point_tokens >= 0, 255, 0).astype(np.uint8))

        local_ids = np.arange(selected_groups.size, dtype=np.int64)[:, None]
        token_semantic_ids.append(local_ids)
        token_levels.append(np.full(selected_groups.size, level, dtype=np.uint8))
        token_reliability.append(selected_reliability[selected_groups].astype(np.float32))
        token_source.append(source_ids[selected_groups].astype(np.uint8))
        level_codebooks.append(selected_features[selected_groups].astype(np.float16))
        if previous_labels is None:
            token_parent.append(np.full(selected_groups.size, -1, dtype=np.int64))
        else:
            child_to_parent = parent_group_lookup(previous_labels, labels)
            parent_groups = child_to_parent[selected_groups]
            parent_tokens = previous_group_to_token[parent_groups]
            token_parent.append(parent_tokens.astype(np.int64))

        selected_old = int((source_ids[selected_groups] == OLD_SOURCE).sum())
        selected_auxiliary = int((source_ids[selected_groups] == AUXILIARY_SOURCE).sum())
        level_selection.append(
            {
                "name": name,
                "raw_groups": group_count,
                "selected_tokens": int(selected_groups.size),
                "covered_gaussian_fraction": float((point_tokens >= 0).mean()),
                "reliability_quantiles": quantiles(selected_reliability[selected_groups]),
                "selected_old_tokens": selected_old,
                "selected_sam_tokens": selected_auxiliary,
                "selected_sam_fraction": float(selected_auxiliary / selected_groups.size),
                "old_split_cosine_mean": float(
                    old_group["cross_cosine"][selected].mean().item()
                ),
                "sam_split_cosine_mean": float(
                    auxiliary_group["cross_cosine"][selected].mean().item()
                ),
            }
        )
        previous_group_to_token = group_to_token
        previous_labels = labels
        token_offset += int(selected_groups.size)

    total_tokens = token_offset
    semantic_dtype = (
        np.uint32
        if max(codebook.shape[0] for codebook in level_codebooks) > np.iinfo(np.uint16).max
        else np.uint16
    )
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    semantic_ids = np.concatenate(token_semantic_ids, axis=0).astype(semantic_dtype)
    point_dtype = np.uint32 if total_tokens > np.iinfo(np.uint16).max else np.uint16
    point_invalid = int(np.iinfo(point_dtype).max)
    point_ids = np.stack(point_ids_by_level, axis=1)
    packed_point_ids = np.full(point_ids.shape, point_invalid, dtype=point_dtype)
    point_valid = point_ids >= 0
    packed_point_ids[point_valid] = point_ids[point_valid].astype(point_dtype)
    point_weights = np.stack(point_weights_by_level, axis=1)
    group_levels = np.concatenate(token_levels, axis=0)
    group_reliability = np.concatenate(token_reliability, axis=0).astype(np.float16)
    group_source = np.concatenate(token_source, axis=0)
    group_parent_ids = np.concatenate(token_parent, axis=0)

    level_codebook_manifest = []
    start = 0
    for level, (name, codebook) in enumerate(zip(LEVEL_NAMES, level_codebooks)):
        filename = f"{name}_codebook.npy"
        np.save(os.path.join(output_dir, filename), codebook)
        end = start + int(codebook.shape[0])
        level_codebook_manifest.append(
            {
                "name": name,
                "level": level,
                "semantic_role": LEVEL_ROLES[level],
                "codebook": filename,
                "num_codes": int(codebook.shape[0]),
                "group_token_start": start,
                "group_token_end": end,
                "quantization": "exact_fp16_per_level",
            }
        )
        start = end
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_level.npy"), group_levels)
    np.save(os.path.join(output_dir, "group_reliability.npy"), group_reliability)
    np.save(os.path.join(output_dir, "group_source.npy"), group_source)
    np.save(os.path.join(output_dir, "group_parent_ids.npy"), group_parent_ids)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)

    semantic_bytes = int(
        sum(codebook.nbytes for codebook in level_codebooks)
        + semantic_ids.nbytes
        + group_levels.nbytes
        + group_reliability.nbytes
        + group_source.nbytes
        + group_parent_ids.nbytes
        + packed_point_ids.nbytes
        + point_weights.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "hierarchical_independent_group_codebooks",
        "method": "sam_l0_l3_hierarchical_semantic_memory",
        "num_gaussians": num_gaussians,
        "num_group_codes": total_tokens,
        "feature_dim": feature_dim,
        "top_m": 4,
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "group_source": "group_source.npy",
        "group_source_labels": {"0": "Old", "1": "SAM_level", "255": "invalid"},
        "group_parent_ids": "group_parent_ids.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(packed_point_ids.dtype),
        "weight_dtype": "uint8_full_resident_membership",
        "level_codebooks": level_codebook_manifest,
        "vocabulary_modalities": ["base", *LEVEL_NAMES],
        "modality_token_counts": {
            "base": 0,
            **{name: int(codebook.shape[0]) for name, codebook in zip(LEVEL_NAMES, level_codebooks)},
        },
        "covered_fraction": float(point_valid.any(axis=1).mean()),
        "mean_ids_per_covered_gaussian": float(
            point_valid.sum() / max(1, point_valid.any(axis=1).sum())
        ),
        "hierarchy": {
            "levels": level_groups,
            "semantic_roles": dict(zip(LEVEL_NAMES, LEVEL_ROLES)),
            "nesting": "L1, L2, and L3 edges are permitted only inside their selected parent-level group",
            "source_selection": "Per-group hard choice: highest split-consistency reliability wins; ties keep Old",
            "level_selection": level_selection,
        },
        "codebook": {
            "layout": "four independent exact FP16 codebooks; group semantic IDs are local to their level",
            "query_readout": "hierarchical_memory softmaxes query similarity across resident L0--L3 tokens and reliability-weightedly interpolates from the base score",
        },
        "storage": {
            "total_semantic_bytes": semantic_bytes,
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
            "shared_vocabulary_bytes_unique": 0,
        },
        "source": {
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
            "geometry_checkpoint_iteration": checkpoint_iteration,
            "old_consensus": old_path,
            "sam_l0_l3_consensus": level_paths,
            "sam_feature_levels": [0, 1, 2, 3],
            "knn_backend": knn_backend,
            "leakage_control": "training cameras, SAM masks, geometry, and training-view split consensuses only",
        },
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
