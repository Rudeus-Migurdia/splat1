#!/usr/bin/env python
"""Train seeded L0--L3 resident codebooks with one token slot per Gaussian.

Unlike A26's exact per-group tables, this builder learns a reusable spherical
codebook at each SAM level.  Every Gaussian receives one ID from every level;
the accompanying per-point reliability gate determines whether a slot should
influence query-time fusion.
"""

import json
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np

from build_hierarchical_semantic_memory import (
    AUXILIARY_SOURCE,
    INVALID_SOURCE,
    LEVEL_NAMES,
    LEVEL_ROLES,
    OLD_SOURCE,
    build_level_groups,
    combined_split_features,
    normalize,
    quantiles,
    validate_level_configuration,
)


def set_deterministic_seed(seed):
    """Seed every stochastic backend used by the builder before CUDA work starts."""
    import torch

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_group_feature_source(old, auxiliary, agreement_floor, source_margin):
    """Prefer SAM only when it is stable and agrees with the Old group feature.

    A disagreement deliberately falls back to Old with a reduced reliability;
    the codebook still receives a resident ID, but the query reader can retain
    the canonical score instead of trusting an unstable modality switch.
    """
    if not 0.0 <= agreement_floor <= 1.0:
        raise ValueError("agreement_floor must be in [0, 1]")
    if source_margin < 0.0:
        raise ValueError("source_margin must be non-negative")

    old_features = old["features"].float().cpu().numpy().astype(np.float32)
    sam_features = auxiliary["features"].float().cpu().numpy().astype(np.float32)
    old_reliability = old["reliability"].cpu().numpy().astype(np.float32)
    sam_reliability = auxiliary["reliability"].cpu().numpy().astype(np.float32)
    old_supported = old["supported"].cpu().numpy().astype(bool)
    sam_supported = auxiliary["supported"].cpu().numpy().astype(bool)
    if not (
        old_features.shape
        == sam_features.shape
        and old_reliability.shape == sam_reliability.shape == old_supported.shape
        and old_supported.shape == sam_supported.shape
    ):
        raise ValueError("Old and SAM group statistics must have matching shapes")

    agreement = np.sum(old_features * sam_features, axis=1).clip(-1.0, 1.0)
    both = old_supported & sam_supported
    source = np.full(old_supported.shape, INVALID_SOURCE, dtype=np.uint8)
    source[old_supported] = OLD_SOURCE
    source[~old_supported & sam_supported] = AUXILIARY_SOURCE

    sam_wins = both & (agreement >= agreement_floor) & (
        sam_reliability > old_reliability + source_margin
    )
    source[sam_wins] = AUXILIARY_SOURCE
    valid = source != INVALID_SOURCE

    features = np.zeros_like(old_features, dtype=np.float32)
    reliability = np.zeros_like(old_reliability, dtype=np.float32)
    use_old = source == OLD_SOURCE
    use_sam = source == AUXILIARY_SOURCE
    features[use_old] = old_features[use_old]
    features[use_sam] = sam_features[use_sam]
    reliability[use_old] = old_reliability[use_old]
    reliability[use_sam] = sam_reliability[use_sam]

    # Both sources can be individually split-stable but semantically disagree.
    # Keep Old as a conservative fallback, with its confidence scaled by agreement.
    conflict = both & (agreement < agreement_floor)
    agreement_scale = np.clip(agreement / max(agreement_floor, 1e-8), 0.0, 1.0)
    reliability[conflict] *= agreement_scale[conflict]
    features[~valid] = 0.0
    reliability[~valid] = 0.0
    return normalize(features), reliability, source, agreement, conflict


def complete_resident_sources(
    features,
    reliability,
    source,
    old_full,
    sam_full,
    fallback_reliability,
):
    """Fill split-unsupported slots from full training-view group consensuses."""
    if not 0.0 < fallback_reliability <= 1.0:
        raise ValueError("fallback_reliability must be in (0, 1]")
    features = np.asarray(features, dtype=np.float32).copy()
    reliability = np.asarray(reliability, dtype=np.float32).copy()
    source = np.asarray(source, dtype=np.uint8).copy()
    old_features, _, old_compactness, old_valid = old_full
    sam_features, _, sam_compactness, sam_valid = sam_full
    old_features = old_features.float().cpu().numpy().astype(np.float32)
    sam_features = sam_features.float().cpu().numpy().astype(np.float32)
    old_compactness = old_compactness.cpu().numpy().astype(np.float32)
    sam_compactness = sam_compactness.cpu().numpy().astype(np.float32)
    old_valid = old_valid.cpu().numpy().astype(bool)
    sam_valid = sam_valid.cpu().numpy().astype(bool)

    missing = source == INVALID_SOURCE
    fallback_sam = missing & sam_valid
    fallback_old = missing & ~sam_valid & old_valid
    features[fallback_sam] = sam_features[fallback_sam]
    features[fallback_old] = old_features[fallback_old]
    reliability[fallback_sam] = fallback_reliability * np.clip(
        sam_compactness[fallback_sam], 0.25, 1.0
    )
    reliability[fallback_old] = fallback_reliability * np.clip(
        old_compactness[fallback_old], 0.25, 1.0
    )
    source[fallback_sam] = AUXILIARY_SOURCE
    source[fallback_old] = OLD_SOURCE
    unresolved = source == INVALID_SOURCE
    features[unresolved] = 0.0
    reliability[unresolved] = 0.0
    return (
        normalize(features),
        reliability,
        source,
        fallback_old,
        fallback_sam,
        unresolved,
    )


def deterministic_weighted_indices(indices, weights, maximum, seed):
    """Sample unique training groups deterministically, weighted by support."""
    indices = np.asarray(indices, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float64)
    if indices.ndim != 1 or weights.shape != indices.shape:
        raise ValueError("Weighted sampling inputs must be one-dimensional and aligned")
    if not indices.size:
        raise ValueError("Cannot sample an empty group set")
    count = min(int(maximum), int(indices.size))
    if count <= 0:
        raise ValueError("maximum must be positive")
    if count == indices.size:
        return indices.copy()
    weights = np.maximum(weights, 0.0)
    if not np.isfinite(weights).all() or weights.sum() <= 0.0:
        weights = np.ones_like(weights)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=count, replace=False, p=probabilities))


def assign_codebook_in_chunks(index, features, assignable, chunk_size):
    """Assign group features without materializing a full groups-by-codes matrix."""
    features = np.asarray(features, dtype=np.float32)
    assignable = np.asarray(assignable, dtype=np.int64)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    output = np.zeros(features.shape[0], dtype=np.int32)
    for start in range(0, assignable.size, chunk_size):
        group_ids = assignable[start : start + chunk_size]
        output[group_ids] = index.search(features[group_ids])
    return output


def train_level_codebook(
    features,
    reliability,
    group_sizes,
    minimum_size,
    minimum_reliability,
    num_codes,
    train_samples,
    iterations,
    seed,
    faiss_gpu,
    assignment_chunk_size,
):
    """Train a reusable spherical vocabulary and assign every raw group to it."""
    from build_gaussian_multilevel_codebook import faiss_kmeans

    features = normalize(features)
    reliability = np.asarray(reliability, dtype=np.float32)
    group_sizes = np.asarray(group_sizes, dtype=np.int32)
    nonzero = np.linalg.norm(features, axis=1) > 0.0
    training = np.flatnonzero(
        nonzero
        & (group_sizes >= int(minimum_size))
        & (reliability >= float(minimum_reliability))
    )
    if not training.size:
        raise ValueError("No stable groups remain for codebook training")
    weights = np.sqrt(group_sizes[training].astype(np.float64)) * reliability[training]
    sample_ids = deterministic_weighted_indices(training, weights, train_samples, seed)
    actual_codes = min(int(num_codes), int(sample_ids.size))
    if actual_codes <= 0:
        raise ValueError("Each level must train at least one code")
    codebook, index = faiss_kmeans(
        features[sample_ids], actual_codes, iterations, seed, spherical=True, use_gpu=faiss_gpu
    )
    codebook = normalize(codebook)
    assignable = np.flatnonzero(nonzero)
    group_code_ids = assign_codebook_in_chunks(
        index, features, assignable, assignment_chunk_size
    )
    reconstruction = codebook[group_code_ids[assignable]] if assignable.size else np.empty((0, features.shape[1]))
    assignment_cosine = (
        np.sum(features[assignable] * reconstruction, axis=1)
        if assignable.size
        else np.empty(0, dtype=np.float32)
    )
    used = np.bincount(group_code_ids[assignable], minlength=actual_codes) if assignable.size else np.zeros(actual_codes)
    return codebook.astype(np.float16), group_code_ids, {
        "num_training_groups": int(training.size),
        "num_training_samples": int(sample_ids.size),
        "num_codes": int(actual_codes),
        "occupied_codes": int((used > 0).sum()),
        "assignment_cosine_quantiles": quantiles(assignment_cosine),
    }


def main():
    import torch

    from build_gaussian_superpoint_support import build_knn, load_geometry
    from build_hierarchical_group_semantic_codebook import (
        aggregate_source,
        aggregate_split_groups,
    )

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--old_consensus", required=True)
    for level in range(4):
        parser.add_argument(f"--sam_l{level}_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
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
    parser.add_argument(
        "--codes_per_level", nargs=4, type=int, default=[2048, 4096, 8192, 16384]
    )
    parser.add_argument("--train_samples", type=int, default=200000)
    parser.add_argument("--kmeans_iterations", type=int, default=25)
    parser.add_argument("--assignment_chunk_size", type=int, default=8192)
    parser.add_argument("--stability_floor", type=float, default=0.5)
    parser.add_argument("--minimum_reliability", type=float, default=0.25)
    parser.add_argument("--source_agreement_floor", type=float, default=0.80)
    parser.add_argument("--source_margin", type=float, default=0.0)
    parser.add_argument("--fallback_reliability", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.seed < 0:
        raise ValueError("seed must be non-negative")
    if args.neighbors <= 1 or args.chunk_size <= 0 or args.knn_workers <= 0:
        raise ValueError("Neighbor, chunk, and worker counts must be positive")
    if (
        args.train_samples <= 0
        or args.kmeans_iterations <= 0
        or args.assignment_chunk_size <= 0
    ):
        raise ValueError("Codebook sample count and iterations must be positive")
    if any(value <= 0 for value in args.codes_per_level):
        raise ValueError("Every level must request at least one code")
    if args.spatial_radius_factor <= 0.0 or args.rgb_threshold <= 0.0 or args.log_scale_threshold <= 0.0:
        raise ValueError("Geometry thresholds must be positive")
    if not -1.0 <= args.stability_floor < 1.0:
        raise ValueError("stability_floor must be in [-1, 1)")
    if not 0.0 <= args.minimum_reliability <= 1.0:
        raise ValueError("minimum_reliability must be in [0, 1]")
    if not 0.0 < args.fallback_reliability <= 1.0:
        raise ValueError("fallback_reliability must be in (0, 1]")
    validate_level_configuration(
        args.semantic_thresholds, args.maximum_group_sizes, args.minimum_group_sizes
    )
    set_deterministic_seed(args.seed)

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse seeded hierarchical resident memory: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    old_path = os.path.abspath(args.old_consensus)
    old_payload = torch.load(old_path, map_location="cpu")
    _, old_shape = combined_split_features(old_payload)
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
                "minimum_group_size_for_codebook": int(args.minimum_group_sizes[level]),
            }
        )
        raw_labels.append(labels)
        level_groups.append(diagnostics)
        parent = labels

    point_ids_by_level = []
    point_reliability_by_level = []
    point_source_by_level = []
    level_codebooks = []
    level_training = []
    raw_source_summary = []
    code_offsets = []
    offset = 0
    for level, (name, labels, payload) in enumerate(
        zip(LEVEL_NAMES, raw_labels, level_payloads)
    ):
        group_count = int(labels.max()) + 1
        group_sizes = np.bincount(labels, minlength=group_count).astype(np.int32)
        old_group = aggregate_source(
            old_payload,
            labels,
            group_count,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        sam_group = aggregate_source(
            payload,
            labels,
            group_count,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        features, reliability, source, agreement, conflict = select_group_feature_source(
            old_group, sam_group, args.source_agreement_floor, args.source_margin
        )
        old_full = aggregate_split_groups(
            old_payload["initial_features"].detach().cpu(),
            old_payload["total_weights"].detach().cpu(),
            labels,
            group_count,
            args.device,
            args.chunk_size,
        )
        sam_full = aggregate_split_groups(
            payload["initial_features"].detach().cpu(),
            payload["total_weights"].detach().cpu(),
            labels,
            group_count,
            args.device,
            args.chunk_size,
        )
        features, reliability, source, fallback_old, fallback_sam, unresolved = (
            complete_resident_sources(
                features,
                reliability,
                source,
                old_full,
                sam_full,
                args.fallback_reliability,
            )
        )
        codebook, group_codes, training = train_level_codebook(
            features,
            reliability,
            group_sizes,
            args.minimum_group_sizes[level],
            args.minimum_reliability,
            args.codes_per_level[level],
            args.train_samples,
            args.kmeans_iterations,
            args.seed + 1009 * (level + 1),
            args.faiss_gpu,
            args.assignment_chunk_size,
        )
        code_offsets.append(offset)
        offset += int(codebook.shape[0])
        level_codebooks.append(codebook)
        point_ids_by_level.append(group_codes[labels])
        point_reliability_by_level.append(reliability[labels])
        point_source_by_level.append(source[labels])
        training.update(
            {
                "name": name,
                "raw_groups": group_count,
                "source_agreement_quantiles": quantiles(agreement),
                "source_conflict_fraction": float(conflict.mean()),
                "selected_old_fraction": float((source == OLD_SOURCE).mean()),
                "selected_sam_fraction": float((source == AUXILIARY_SOURCE).mean()),
                "supported_group_fraction": float((source != INVALID_SOURCE).mean()),
                "full_consensus_fallback_old_fraction": float(fallback_old.mean()),
                "full_consensus_fallback_sam_fraction": float(fallback_sam.mean()),
                "unresolved_group_fraction": float(unresolved.mean()),
                "resident_reliability_quantiles": quantiles(reliability[labels]),
                "resident_usable_fraction": float((reliability[labels] > 0.0).mean()),
            }
        )
        level_training.append(training)
        raw_source_summary.append(
            {
                "name": name,
                "old_split_cosine_mean": float(
                    old_group["cross_cosine"].mean().item()
                ),
                "sam_split_cosine_mean": float(
                    sam_group["cross_cosine"].mean().item()
                ),
            }
        )

    total_codes = offset
    semantic_dtype = np.uint16 if max(codebook.shape[0] for codebook in level_codebooks) <= np.iinfo(np.uint16).max else np.uint32
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    point_dtype = np.uint16 if total_codes <= np.iinfo(np.uint16).max else np.uint32
    point_invalid = int(np.iinfo(point_dtype).max)
    global_point_ids = np.stack(
        [ids + code_offsets[level] for level, ids in enumerate(point_ids_by_level)], axis=1
    ).astype(point_dtype)
    point_reliability = np.stack(point_reliability_by_level, axis=1).astype(np.float16)
    point_sources = np.stack(point_source_by_level, axis=1).astype(np.uint8)
    point_weights = np.full(global_point_ids.shape, 255, dtype=np.uint8)
    semantic_ids = np.concatenate(
        [np.arange(codebook.shape[0], dtype=np.int64)[:, None] for codebook in level_codebooks], axis=0
    ).astype(semantic_dtype)
    group_levels = np.concatenate(
        [np.full(codebook.shape[0], level, dtype=np.uint8) for level, codebook in enumerate(level_codebooks)]
    )
    group_reliability = np.ones(total_codes, dtype=np.float16)
    group_source = np.full(total_codes, INVALID_SOURCE, dtype=np.uint8)
    group_parent_ids = np.full(total_codes, -1, dtype=np.int64)

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
                "quantization": "seeded_spherical_kmeans_per_level",
            }
        )
        start = end
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_level.npy"), group_levels)
    np.save(os.path.join(output_dir, "group_reliability.npy"), group_reliability)
    np.save(os.path.join(output_dir, "group_source.npy"), group_source)
    np.save(os.path.join(output_dir, "group_parent_ids.npy"), group_parent_ids)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), global_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    np.save(os.path.join(output_dir, "point_group_reliability.npy"), point_reliability)
    np.save(os.path.join(output_dir, "point_group_source.npy"), point_sources)

    semantic_bytes = int(
        sum(codebook.nbytes for codebook in level_codebooks)
        + semantic_ids.nbytes
        + group_levels.nbytes
        + group_reliability.nbytes
        + group_source.nbytes
        + group_parent_ids.nbytes
        + global_point_ids.nbytes
        + point_weights.nbytes
        + point_reliability.nbytes
        + point_sources.nbytes
    )
    usable = point_reliability > 0.0
    manifest = {
        "format_version": 2,
        "representation": "hierarchical_independent_group_codebooks",
        "method": "seeded_four_slot_sam_hierarchical_resident_memory",
        "num_gaussians": num_gaussians,
        "num_group_codes": total_codes,
        "feature_dim": feature_dim,
        "top_m": 4,
        "resident_slots_required": 4,
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "group_source": "group_source.npy",
        "group_source_labels": {"0": "Old", "1": "SAM_level", "255": "codebook"},
        "group_parent_ids": "group_parent_ids.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_reliability": "point_group_reliability.npy",
        "point_group_source": "point_group_source.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(global_point_ids.dtype),
        "weight_dtype": "uint8_full_resident_membership",
        "level_codebooks": level_codebook_manifest,
        "vocabulary_modalities": ["base", *LEVEL_NAMES],
        "modality_token_counts": {
            "base": 0,
            **{name: int(codebook.shape[0]) for name, codebook in zip(LEVEL_NAMES, level_codebooks)},
        },
        "covered_fraction": 1.0,
        "mean_ids_per_covered_gaussian": 4.0,
        "usable_slot_fraction": float(usable.mean()),
        "usable_covered_fraction": float(usable.any(axis=1).mean()),
        "hierarchy": {
            "levels": level_groups,
            "semantic_roles": dict(zip(LEVEL_NAMES, LEVEL_ROLES)),
            "nesting": "L1, L2, and L3 groups are constructed inside their raw parent group; resident code slots are peer candidates at query time",
            "source_selection": "SAM replaces Old only when both agree semantically and SAM split reliability wins; split-unsupported groups receive a low-confidence full-consensus SAM/Old fallback",
            "level_codebook_training": level_training,
            "split_source_diagnostics": raw_source_summary,
        },
        "codebook": {
            "layout": "four independent seeded spherical K-means codebooks; every Gaussian stores one resident ID per level",
            "query_readout": "reader-defined peer-token query fusion; resident slots encode no parent preference or fixed level priority",
        },
        "storage": {
            "total_semantic_bytes": semantic_bytes,
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
            "shared_vocabulary_bytes_unique": int(sum(codebook.nbytes for codebook in level_codebooks)),
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
        "reproducibility": {
            "seed": int(args.seed),
            "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "torch_deterministic_algorithms": True,
        },
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
