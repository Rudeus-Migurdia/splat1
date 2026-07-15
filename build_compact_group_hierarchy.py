#!/usr/bin/env python
"""Pack shared group semantics into a small codebook plus per-Gaussian IDs."""

import json
import os
from argparse import ArgumentParser

import numpy as np


def l2_normalize(value, eps=1e-8):
    value = np.asarray(value, dtype=np.float32)
    return value / np.maximum(np.linalg.norm(value, axis=-1, keepdims=True), eps)


def reestimate_group_features(codebook_dir, group_ids, group_scores, chunk_size):
    manifest_path = os.path.join(codebook_dir, "manifest.json")
    with open(manifest_path) as source:
        manifest = json.load(source)
    codebooks = [
        np.load(os.path.join(codebook_dir, filename)).astype(np.float32)
        for filename in manifest["codebook_files"]
    ]
    point_code_ids = np.load(os.path.join(codebook_dir, manifest["point_code_ids"]), mmap_mode="r")
    valid_mask = np.load(os.path.join(codebook_dir, manifest["valid_mask"]), mmap_mode="r")
    if point_code_ids.shape[0] != group_ids.shape[0]:
        raise ValueError("Point code IDs and group assignments must contain the same Gaussians")
    num_groups = int(group_ids[group_ids >= 0].max()) + 1 if np.any(group_ids >= 0) else 0
    feature_dim = int(codebooks[0].shape[1])
    group_sums = np.zeros((num_groups, feature_dim), dtype=np.float64)
    group_weights = np.zeros(num_groups, dtype=np.float64)

    for start in range(0, point_code_ids.shape[0], chunk_size):
        end = min(start + chunk_size, point_code_ids.shape[0])
        point_valid = np.asarray(valid_mask[start:end], dtype=bool)
        if not point_valid.any():
            continue
        ids_chunk = np.asarray(point_code_ids[start:end], dtype=np.int64)
        features = np.zeros((end - start, feature_dim), dtype=np.float32)
        for level, codebook in enumerate(codebooks):
            level_ids = ids_chunk[:, level]
            level_valid = point_valid & (level_ids >= 0) & (level_ids < codebook.shape[0])
            features[level_valid] += codebook[level_ids[level_valid]]
        features = l2_normalize(features)
        chunk_group_ids = group_ids[start:end]
        chunk_scores = np.maximum(group_scores[start:end], 0.0)
        for slot in range(chunk_group_ids.shape[1]):
            slot_ids = chunk_group_ids[:, slot]
            slot_weights = chunk_scores[:, slot]
            slot_valid = point_valid & (slot_ids >= 0) & (slot_weights > 0)
            if not slot_valid.any():
                continue
            np.add.at(
                group_sums,
                slot_ids[slot_valid],
                features[slot_valid].astype(np.float64) * slot_weights[slot_valid, None],
            )
            np.add.at(group_weights, slot_ids[slot_valid], slot_weights[slot_valid])

    supported = group_weights > 0
    group_features = np.zeros_like(group_sums, dtype=np.float32)
    group_features[supported] = (
        group_sums[supported] / group_weights[supported, None]
    ).astype(np.float32)
    return l2_normalize(group_features), supported, manifest_path


def main():
    parser = ArgumentParser(
        description="Compact multi-group attachments into uint16 IDs and uint8 weights."
    )
    parser.add_argument("--group_features", default=None)
    parser.add_argument(
        "--point_codebook_dir",
        default=None,
        help="Re-estimate group tokens from a Gaussian codebook instead of teacher group features.",
    )
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--top_m", type=int, default=3)
    parser.add_argument("--reestimate_top_m", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.top_m <= 0:
        raise ValueError("--top_m must be positive")
    if bool(args.group_features) == bool(args.point_codebook_dir):
        raise ValueError("Provide exactly one of --group_features or --point_codebook_dir")
    if args.chunk_size <= 0:
        raise ValueError("--chunk_size must be positive")

    assignments_path = os.path.abspath(args.assignments)
    assignments = np.load(assignments_path)
    all_group_ids = assignments["top_group_ids"].astype(np.int64)
    all_group_scores = assignments["top_group_scores"].astype(np.float32)
    source = {}
    supported_groups = None
    if args.point_codebook_dir:
        reestimate_top_m = args.reestimate_top_m or all_group_ids.shape[1]
        point_codebook_dir = os.path.abspath(args.point_codebook_dir)
        group_features, supported_groups, codebook_manifest = reestimate_group_features(
            point_codebook_dir,
            all_group_ids[:, :reestimate_top_m],
            all_group_scores[:, :reestimate_top_m],
            args.chunk_size,
        )
        source.update(
            point_codebook_dir=point_codebook_dir,
            point_codebook_manifest=codebook_manifest,
            group_token_estimator="weighted_mean_of_discrete_gaussian_reconstructions",
            reestimate_top_m=int(reestimate_top_m),
        )
    else:
        group_features_path = os.path.abspath(args.group_features)
        group_features = l2_normalize(np.load(group_features_path).astype(np.float32))
        source["group_features"] = group_features_path
    group_ids = all_group_ids[:, : args.top_m]
    group_scores = all_group_scores[:, : args.top_m]
    if group_ids.shape != group_scores.shape:
        raise ValueError("Group IDs and scores must have matching shapes")
    valid = group_ids >= 0
    if supported_groups is not None:
        valid &= np.where(group_ids >= 0, supported_groups[np.maximum(group_ids, 0)], False)
    if valid.any() and int(group_ids[valid].max()) >= group_features.shape[0]:
        raise ValueError("Assignments reference IDs outside group_features")

    if group_features.shape[0] <= np.iinfo(np.uint16).max:
        id_dtype = np.uint16
    else:
        id_dtype = np.uint32
    invalid_id = int(np.iinfo(id_dtype).max)
    packed_ids = np.full(group_ids.shape, invalid_id, dtype=id_dtype)
    packed_ids[valid] = group_ids[valid].astype(id_dtype)

    scores = np.where(valid, np.maximum(group_scores, 0.0), 0.0)
    score_sums = scores.sum(axis=1, keepdims=True)
    normalized_scores = np.divide(
        scores,
        np.maximum(score_sums, 1e-8),
        out=np.zeros_like(scores),
        where=score_sums > 0,
    )
    packed_weights = np.rint(normalized_scores * 255.0).clip(0, 255).astype(np.uint8)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    codebook_values = group_features.astype(np.float16)
    np.save(os.path.join(output_dir, "group_codebook.npy"), codebook_values)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), packed_weights)
    storage_bytes = int(codebook_values.nbytes + packed_ids.nbytes + packed_weights.nbytes)
    coverage = valid.any(axis=1)
    manifest = {
        "format_version": 1,
        "representation": "compact_group_hierarchy",
        "num_gaussians": int(group_ids.shape[0]),
        "num_group_codes": int(group_features.shape[0]),
        "feature_dim": int(group_features.shape[1]),
        "top_m": int(group_ids.shape[1]),
        "group_codebook": "group_codebook.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "id_dtype": str(packed_ids.dtype),
        "invalid_id": invalid_id,
        "weight_dtype": "uint8_normalized",
        "covered_fraction": float(coverage.mean()),
        "mean_ids_per_covered_gaussian": float(valid[coverage].sum(axis=1).mean())
        if coverage.any()
        else 0.0,
        "storage": {
            "group_codebook_bytes_fp16": int(codebook_values.nbytes),
            "point_group_id_bytes": int(packed_ids.nbytes),
            "point_group_weight_bytes": int(packed_weights.nbytes),
            "total_semantic_bytes": storage_bytes,
            "bytes_per_gaussian_amortized": float(storage_bytes / group_ids.shape[0]),
        },
        "source": {**source, "assignments": assignments_path},
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
