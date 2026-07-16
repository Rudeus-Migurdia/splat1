#!/usr/bin/env python
"""Append residual codewords and a sparse third ID for hard Gaussians."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from build_gaussian_multilevel_codebook import (
    ConsensusFeatureSource,
    faiss_kmeans,
    l2_normalize,
)
from train_joint_query_preserving_vocabulary import FixedSharedAssignment


def reconstruct_numpy(codebook, point_ids, point_weights):
    valid = point_ids >= 0
    safe_ids = np.maximum(point_ids, 0)
    values = codebook[safe_ids]
    reconstruction = (
        values * point_weights[..., None] * valid[..., None]
    ).sum(axis=1)
    return l2_normalize(reconstruction)


def assign_residual_codes(targets, current, residual_codebook, index, min_gain):
    residual = l2_normalize(targets - current)
    extension_ids = index.search(residual)
    selected = residual_codebook[extension_ids]
    coefficients = np.clip(
        np.sum((targets - current) * selected, axis=1), 0.0, 1.0
    ).astype(np.float32)
    candidate = l2_normalize(current + coefficients[:, None] * selected)
    old_cosine = np.sum(targets * current, axis=1)
    new_cosine = np.sum(targets * candidate, axis=1)
    gain = new_cosine - old_cosine
    accepted = (coefficients > 1e-6) & (gain >= min_gain)
    return extension_ids, coefficients, accepted, np.where(accepted, new_cosine, old_cosine)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--consensus", required=True)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--initial_codebook", required=True)
    parser.add_argument("--num_extension_codes", type=int, default=4096)
    parser.add_argument("--train_samples", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--assignment_chunk", type=int, default=4096)
    parser.add_argument("--min_cosine_gain", type=float, default=1e-4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.num_extension_codes <= 0 or args.train_samples <= 0:
        raise ValueError("Code and sample counts must be positive")
    if args.iterations <= 0 or args.assignment_chunk <= 0:
        raise ValueError("Iteration and chunk counts must be positive")
    if args.min_cosine_gain < 0.0:
        raise ValueError("--min_cosine_gain must be non-negative")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse residual vocabulary extension: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    source = ConsensusFeatureSource(args.consensus)
    assignment = FixedSharedAssignment(args.artifact_dir)
    if source.num_items != assignment.num_gaussians:
        raise ValueError("Consensus and fixed-ID artifact sizes differ")
    old_codebook = l2_normalize(
        np.load(os.path.abspath(args.initial_codebook)).astype(np.float32)
    )
    if old_codebook.shape != (assignment.num_codes, assignment.feature_dim):
        raise ValueError("Initial codebook shape does not match the artifact")
    invalid_id = int(assignment.manifest["invalid_id"])
    if old_codebook.shape[0] + args.num_extension_codes >= invalid_id:
        raise ValueError("Expanded vocabulary does not fit the artifact ID dtype")

    valid = np.asarray(source.valid_mask, dtype=bool) & assignment.valid_mask
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError("No hard Gaussians are valid in both inputs")
    rng = np.random.default_rng(args.seed)
    sample = rng.choice(
        valid_indices,
        min(args.train_samples, valid_indices.size),
        replace=False,
    )
    sample_targets = source.read(sample)
    sample_current = reconstruct_numpy(
        old_codebook, assignment.ids[sample], assignment.weights[sample]
    )
    residual_training = l2_normalize(sample_targets - sample_current)
    extension, extension_index = faiss_kmeans(
        residual_training,
        min(args.num_extension_codes, residual_training.shape[0]),
        args.iterations,
        args.seed,
        spherical=True,
        use_gpu=args.faiss_gpu,
    )
    extension = l2_normalize(extension)
    expanded_codebook = np.concatenate((old_codebook, extension), axis=0)

    accepted_points = []
    accepted_ids = []
    accepted_weights = []
    cosine_values = []
    old_code_count = old_codebook.shape[0]
    for start in range(0, valid_indices.size, args.assignment_chunk):
        indices = valid_indices[start : start + args.assignment_chunk]
        targets = source.read(indices)
        current = reconstruct_numpy(
            old_codebook, assignment.ids[indices], assignment.weights[indices]
        )
        ids, coefficients, accepted, cosine = assign_residual_codes(
            targets,
            current,
            extension,
            extension_index,
            args.min_cosine_gain,
        )
        accepted_points.append(indices[accepted])
        accepted_ids.append(ids[accepted] + old_code_count)
        accepted_weights.append(
            np.rint(coefficients[accepted] * 255.0).astype(np.uint8)
        )
        cosine_values.append(cosine)

    new_points = np.concatenate(accepted_points).astype(np.uint32)
    new_ids = np.concatenate(accepted_ids)
    new_weights = np.concatenate(accepted_weights)
    id_dtype = np.load(
        os.path.join(
            assignment.artifact_dir, assignment.manifest["point_code_ids"]
        )
    ).dtype
    new_ids = new_ids.astype(id_dtype)

    def load(key):
        return np.load(
            os.path.join(assignment.artifact_dir, assignment.manifest[key])
        )

    base_ids = load("point_code_ids")
    valid_mask = load("valid_mask")
    overflow_points = np.concatenate(
        (load("overflow_point_ids").astype(np.uint32), new_points)
    )
    overflow_ids = np.concatenate(
        (load("overflow_code_ids").astype(id_dtype), new_ids)
    )
    overflow_slots = np.concatenate(
        (
            load("overflow_slots").astype(np.uint8),
            np.full(new_points.shape, 2, dtype=np.uint8),
        )
    )
    overflow_weights = np.concatenate(
        (load("overflow_weights").astype(np.uint8), new_weights)
    )

    np.save(os.path.join(output_dir, "codebook_shared.npy"), expanded_codebook.astype(np.float16))
    np.save(os.path.join(output_dir, "point_code_ids.npy"), base_ids)
    np.save(os.path.join(output_dir, "valid_mask.npy"), valid_mask)
    np.save(os.path.join(output_dir, "overflow_point_ids.npy"), overflow_points)
    np.save(os.path.join(output_dir, "overflow_code_ids.npy"), overflow_ids)
    np.save(os.path.join(output_dir, "overflow_slots.npy"), overflow_slots)
    np.save(os.path.join(output_dir, "overflow_weights.npy"), overflow_weights)

    num_valid = int(valid_mask.sum())
    previous_counts = (assignment.ids[valid] >= 0).sum(axis=1)
    added = np.zeros(assignment.num_gaussians, dtype=np.int8)
    added[new_points.astype(np.int64)] = 1
    counts = previous_counts + added[valid]
    histogram = {
        str(value): int((counts == value).sum()) for value in range(1, 4)
    }
    codebook_bytes = int(expanded_codebook.size * np.dtype(np.float16).itemsize)
    point_id_bytes = int(base_ids.nbytes + overflow_ids.nbytes)
    storage = {
        "codebook_bytes_fp16": codebook_bytes,
        "point_id_bytes": point_id_bytes,
        "overflow_point_bytes": int(overflow_points.nbytes),
        "overflow_slot_bytes": int(overflow_slots.nbytes),
        "point_weight_bytes": int(overflow_weights.nbytes),
        "valid_mask_bytes": int(valid_mask.nbytes),
    }
    storage["total_semantic_bytes"] = int(sum(storage.values()))
    storage["full_per_gaussian_fp16_bytes"] = int(
        assignment.num_gaussians * assignment.feature_dim * 2
    )
    storage["compression_ratio_vs_512d_fp16"] = (
        storage["full_per_gaussian_fp16_bytes"] / storage["total_semantic_bytes"]
    )
    storage["bytes_per_gaussian_amortized"] = (
        storage["total_semantic_bytes"] / assignment.num_gaussians
    )

    manifest = dict(assignment.manifest)
    manifest.update(
        {
            "num_codes": int(expanded_codebook.shape[0]),
            "id_slots": 3,
            "codebook_files": ["codebook_shared.npy"],
            "point_code_ids": "point_code_ids.npy",
            "valid_mask": "valid_mask.npy",
            "overflow_point_ids": "overflow_point_ids.npy",
            "overflow_code_ids": "overflow_code_ids.npy",
            "overflow_slots": "overflow_slots.npy",
            "overflow_weights": "overflow_weights.npy",
            "average_ids_per_valid_gaussian": float(counts.mean()),
            "id_count_histogram": histogram,
            "mean_reconstruction_cosine": float(np.concatenate(cosine_values).mean()),
            "storage": storage,
            "source_before_extension": assignment.manifest.get("source"),
            "source": {
                "type": "hard_gaussian_residual_vocabulary_extension",
                "consensus": os.path.abspath(args.consensus),
                "artifact_dir": assignment.artifact_dir,
                "initial_codebook": os.path.abspath(args.initial_codebook),
                "num_extension_codes": int(extension.shape[0]),
                "num_third_ids": int(new_points.size),
                "third_id_fraction_of_valid": float(new_points.size / max(1, num_valid)),
                "args": vars(args),
            },
        }
    )
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({
        "num_codes": manifest["num_codes"],
        "num_valid_gaussians": num_valid,
        "num_third_ids": int(new_points.size),
        "third_id_fraction_of_valid": manifest["source"]["third_id_fraction_of_valid"],
        "average_ids_per_valid_gaussian": manifest["average_ids_per_valid_gaussian"],
        "mean_reconstruction_cosine": manifest["mean_reconstruction_cosine"],
        "storage_megabytes": storage["total_semantic_bytes"] / 2**20,
    }, indent=2))


if __name__ == "__main__":
    main()
