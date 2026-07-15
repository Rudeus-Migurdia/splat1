#!/usr/bin/env python
"""Build one large shared semantic codebook with variable IDs per Gaussian."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from build_gaussian_multilevel_codebook import (
    ConsensusFeatureSource,
    SearchIndex,
    TorchSearchIndex,
    faiss_kmeans,
    l2_normalize,
)


def compact_ids(code_ids, num_codes):
    if num_codes <= np.iinfo(np.uint16).max:
        dtype = np.uint16
    else:
        dtype = np.uint32
    invalid_id = int(np.iinfo(dtype).max)
    packed = np.full(code_ids.shape, invalid_id, dtype=dtype)
    valid = code_ids >= 0
    packed[valid] = code_ids[valid].astype(dtype)
    return packed, invalid_id


def assign_sparse_codes(
    targets,
    codebook,
    index,
    max_ids,
    min_gain,
    target_cosine,
    min_ids=1,
    max_ids_per_item=None,
    required_ids_per_item=None,
):
    count = targets.shape[0]
    code_ids = np.full((count, max_ids), -1, dtype=np.int32)
    coefficients = np.zeros((count, max_ids), dtype=np.float32)
    reconstruction = np.zeros_like(targets, dtype=np.float32)
    cosine = np.zeros(count, dtype=np.float32)
    if not 1 <= min_ids <= max_ids:
        raise ValueError("min_ids must be between one and max_ids")
    if max_ids_per_item is None:
        max_ids_per_item = np.full(count, max_ids, dtype=np.int16)
    else:
        max_ids_per_item = np.asarray(max_ids_per_item, dtype=np.int16)
        if max_ids_per_item.shape != (count,):
            raise ValueError("Per-item capacities must match the target count")
        if np.any(max_ids_per_item < min_ids) or np.any(max_ids_per_item > max_ids):
            raise ValueError("Per-item capacities must be within [min_ids, max_ids]")
    if required_ids_per_item is None:
        required_ids_per_item = np.full(count, min_ids, dtype=np.int16)
    else:
        required_ids_per_item = np.asarray(required_ids_per_item, dtype=np.int16)
        if required_ids_per_item.shape != (count,):
            raise ValueError("Required ID counts must match the target count")
        if np.any(required_ids_per_item < min_ids) or np.any(
            required_ids_per_item > max_ids_per_item
        ):
            raise ValueError("Required ID counts must be within [min_ids, capacity]")
    active = max_ids_per_item > 0

    for slot in range(max_ids):
        active &= max_ids_per_item > slot
        if not active.any():
            break
        active_indices = np.flatnonzero(active)
        if slot == 0:
            query = targets[active_indices]
        else:
            query = l2_normalize(
                targets[active_indices] - l2_normalize(reconstruction[active_indices])
            )
        selected_ids = index.search(query)
        selected_codes = codebook[selected_ids]
        if slot == 0:
            selected_coefficients = np.ones(active_indices.size, dtype=np.float32)
        else:
            residual = targets[active_indices] - l2_normalize(reconstruction[active_indices])
            selected_coefficients = np.clip(
                np.sum(residual * selected_codes, axis=1), 0.0, 1.0
            ).astype(np.float32)
        candidate = reconstruction[active_indices] + selected_coefficients[:, None] * selected_codes
        candidate_cosine = np.sum(targets[active_indices] * l2_normalize(candidate), axis=1)
        gain = candidate_cosine - cosine[active_indices]
        accepted = selected_coefficients > 1e-6
        mandatory = slot < required_ids_per_item[active_indices]
        optional = ~mandatory
        accepted[optional] &= gain[optional] >= min_gain
        accepted[optional] &= cosine[active_indices[optional]] < target_cosine
        accepted_indices = active_indices[accepted]
        code_ids[accepted_indices, slot] = selected_ids[accepted]
        coefficients[accepted_indices, slot] = selected_coefficients[accepted]
        reconstruction[accepted_indices] = candidate[accepted]
        cosine[accepted_indices] = candidate_cosine[accepted]
        active[active_indices[~accepted]] = False
        required_filled = slot + 1 >= required_ids_per_item[accepted_indices]
        target_reached = cosine[accepted_indices] >= target_cosine
        active[accepted_indices[required_filled & target_reached]] = False

    row_scale = coefficients.max(axis=1, keepdims=True)
    normalized_weights = coefficients / np.maximum(row_scale, 1e-8)
    packed_weights = np.rint(normalized_weights * 255.0).astype(np.uint8)
    packed_weights[code_ids < 0] = 0
    return code_ids, packed_weights, cosine


def make_search_index(codebook, use_gpu):
    try:
        import faiss
    except ImportError:
        return TorchSearchIndex(codebook, spherical=True) if use_gpu else SearchIndex(
            codebook, spherical=True
        )
    if use_gpu and not hasattr(faiss, "StandardGpuResources"):
        return TorchSearchIndex(codebook, spherical=True)
    index = faiss.IndexFlatIP(codebook.shape[1])
    index.add(np.ascontiguousarray(codebook, dtype=np.float32))
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    return SearchIndex(codebook, spherical=True, faiss_index=index)


def main():
    parser = ArgumentParser(
        description="Build a unified shared codebook with rate-distortion adaptive Gaussian IDs."
    )
    parser.add_argument("--consensus", required=True)
    parser.add_argument(
        "--codebook",
        default=None,
        help="Reuse an existing shared codebook and only recompute sparse Gaussian IDs.",
    )
    parser.add_argument("--num_codes", type=int, default=16384)
    parser.add_argument("--max_ids", type=int, default=4)
    parser.add_argument("--min_ids", type=int, default=1)
    parser.add_argument(
        "--use_consensus_capacity",
        action="store_true",
        help="Cap each Gaussian's IDs using semantic_capacity stored in the consensus.",
    )
    parser.add_argument(
        "--fill_consensus_capacity",
        action="store_true",
        help="Require every valid Gaussian to fill its requested semantic capacity.",
    )
    parser.add_argument("--min_cosine_gain", type=float, default=0.002)
    parser.add_argument("--target_cosine", type=float, default=0.995)
    parser.add_argument("--train_samples", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--assignment_chunk", type=int, default=4096)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if not 1 <= args.num_codes <= np.iinfo(np.uint32).max:
        raise ValueError("--num_codes must be positive")
    if args.max_ids <= 0 or args.train_samples <= 0 or args.iterations <= 0:
        raise ValueError("ID, sample, and iteration counts must be positive")
    if not 1 <= args.min_ids <= args.max_ids:
        raise ValueError("--min_ids must be between one and --max_ids")
    if args.assignment_chunk <= 0 or args.min_cosine_gain < 0.0:
        raise ValueError("Assignment chunk must be positive and gain non-negative")
    if not -1.0 <= args.target_cosine <= 1.0:
        raise ValueError("--target_cosine must be in [-1, 1]")

    source = ConsensusFeatureSource(args.consensus)
    if args.use_consensus_capacity and source.id_capacity is None:
        raise ValueError("Consensus does not contain semantic_capacity")
    if args.fill_consensus_capacity and not args.use_consensus_capacity:
        raise ValueError("--fill_consensus_capacity requires --use_consensus_capacity")
    valid_indices = np.flatnonzero(source.valid_mask)
    if valid_indices.size == 0:
        raise ValueError("Consensus contains no valid Gaussians")
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.train_samples, valid_indices.size)
    sample_indices = rng.choice(valid_indices, sample_count, replace=False)
    if args.codebook:
        codebook = np.load(args.codebook).astype(np.float32)
        if codebook.ndim != 2 or codebook.shape[1] != source.feature_dim:
            raise ValueError("Reused codebook dimension does not match the consensus")
        codebook = l2_normalize(codebook)
        index = make_search_index(codebook, args.faiss_gpu)
    else:
        training_features = source.read(sample_indices)
        codebook, index = faiss_kmeans(
            training_features,
            min(args.num_codes, sample_count),
            args.iterations,
            args.seed,
            spherical=True,
            use_gpu=args.faiss_gpu,
        )
        codebook = l2_normalize(codebook)

    point_ids = np.full((source.num_items, args.max_ids), -1, dtype=np.int32)
    point_weights = np.zeros((source.num_items, args.max_ids), dtype=np.uint8)
    point_cosine = np.zeros(source.num_items, dtype=np.float32)
    capacity = np.full(source.num_items, args.max_ids, dtype=np.int16)
    if args.use_consensus_capacity:
        capacity = np.clip(source.id_capacity, args.min_ids, args.max_ids)
    capacity[~source.valid_mask] = 0
    for start in range(0, valid_indices.size, args.assignment_chunk):
        indices = valid_indices[start : start + args.assignment_chunk]
        ids, weights, cosine = assign_sparse_codes(
            source.read(indices),
            codebook,
            index,
            args.max_ids,
            args.min_cosine_gain,
            args.target_cosine,
            min_ids=args.min_ids,
            max_ids_per_item=capacity[indices],
            required_ids_per_item=capacity[indices]
            if args.fill_consensus_capacity
            else None,
        )
        point_ids[indices] = ids
        point_weights[indices] = weights
        point_cosine[indices] = cosine

    packed_ids, invalid_id = compact_ids(point_ids, codebook.shape[0])
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / "codebook_shared.npy", codebook.astype(np.float16))
    base_ids = packed_ids[:, 0]
    overflow_points, overflow_slots_zero = np.nonzero(point_ids[:, 1:] >= 0)
    overflow_slots = (overflow_slots_zero + 1).astype(np.uint8)
    overflow_ids = packed_ids[overflow_points, overflow_slots]
    overflow_weights = point_weights[overflow_points, overflow_slots]
    np.save(output_dir / "point_code_ids.npy", base_ids)
    np.save(output_dir / "overflow_point_ids.npy", overflow_points.astype(np.uint32))
    np.save(output_dir / "overflow_code_ids.npy", overflow_ids)
    np.save(output_dir / "overflow_slots.npy", overflow_slots)
    np.save(output_dir / "overflow_weights.npy", overflow_weights)
    np.save(output_dir / "valid_mask.npy", source.valid_mask.astype(np.bool_))

    counts = (point_ids[valid_indices] >= 0).sum(axis=1)
    histogram = {
        str(value): int((counts == value).sum())
        for value in range(1, args.max_ids + 1)
    }
    codebook_bytes = int(codebook.size * np.dtype(np.float16).itemsize)
    id_bytes = int(base_ids.nbytes + overflow_ids.nbytes)
    overflow_point_bytes = int(overflow_points.astype(np.uint32).nbytes)
    overflow_slot_bytes = int(overflow_slots.nbytes)
    weight_bytes = int(overflow_weights.nbytes)
    mask_bytes = int(source.valid_mask.nbytes)
    total_bytes = (
        codebook_bytes
        + id_bytes
        + overflow_point_bytes
        + overflow_slot_bytes
        + weight_bytes
        + mask_bytes
    )
    full_bytes = int(source.num_items * source.feature_dim * 2)
    manifest = {
        "format_version": 1,
        "representation": "gaussian_adaptive_shared_codebook",
        "feature_dim": int(source.feature_dim),
        "num_gaussians": int(source.num_items),
        "num_valid_gaussians": int(valid_indices.size),
        "valid_fraction": float(valid_indices.size / source.num_items),
        "num_codes": int(codebook.shape[0]),
        "id_slots": int(args.max_ids),
        "minimum_ids_per_valid_gaussian": int(args.min_ids),
        "uses_consensus_capacity": bool(args.use_consensus_capacity),
        "fills_consensus_capacity": bool(args.fill_consensus_capacity),
        "requested_average_capacity": float(capacity[valid_indices].mean()),
        "storage_layout": "base_plus_sparse_overflow",
        "codebook_files": ["codebook_shared.npy"],
        "point_code_ids": "point_code_ids.npy",
        "overflow_point_ids": "overflow_point_ids.npy",
        "overflow_code_ids": "overflow_code_ids.npy",
        "overflow_slots": "overflow_slots.npy",
        "overflow_weights": "overflow_weights.npy",
        "valid_mask": "valid_mask.npy",
        "id_dtype": str(packed_ids.dtype),
        "invalid_id": invalid_id,
        "weight_dtype": "uint8_relative",
        "average_ids_per_valid_gaussian": float(counts.mean()),
        "id_count_histogram": histogram,
        "mean_reconstruction_cosine": float(point_cosine[valid_indices].mean()),
        "std_reconstruction_cosine": float(point_cosine[valid_indices].std()),
        "min_reconstruction_cosine": float(point_cosine[valid_indices].min()),
        "storage": {
            "codebook_bytes_fp16": codebook_bytes,
            "point_id_bytes": id_bytes,
            "overflow_point_bytes": overflow_point_bytes,
            "overflow_slot_bytes": overflow_slot_bytes,
            "point_weight_bytes": weight_bytes,
            "valid_mask_bytes": mask_bytes,
            "total_semantic_bytes": total_bytes,
            "full_per_gaussian_fp16_bytes": full_bytes,
            "compression_ratio_vs_512d_fp16": float(full_bytes / max(1, total_bytes)),
            "bytes_per_gaussian_amortized": float(total_bytes / source.num_items),
        },
        "source": source.metadata(),
        "args": vars(args),
    }
    with open(output_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
