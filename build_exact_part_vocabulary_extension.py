#!/usr/bin/env python
"""Append reliable part tokens as exact rows in the scene shared vocabulary."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--group_codebook_dir", required=True)
    parser.add_argument("--base_vocabulary", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(sys.argv[1:])

    source_dir = os.path.abspath(args.group_codebook_dir)
    diagnostic_dir = os.path.join(source_dir, "continuous_diagnostic")
    with open(os.path.join(source_dir, "manifest.json")) as source:
        source_manifest = json.load(source)
    features = np.load(os.path.join(diagnostic_dir, "group_codebook.npy")).astype(
        np.float16
    )
    levels = np.load(os.path.join(source_dir, source_manifest["group_level"]))
    reliability = np.load(
        os.path.join(source_dir, source_manifest["group_reliability"])
    )
    point_ids = np.load(
        os.path.join(source_dir, source_manifest["point_group_ids"])
    ).astype(np.int64)
    point_weights = np.load(
        os.path.join(source_dir, source_manifest["point_group_weights"])
    )
    source_invalid = int(source_manifest["invalid_id"])
    point_ids[point_ids == source_invalid] = -1
    if features.shape[0] != levels.size or reliability.shape != levels.shape:
        raise ValueError("Group token tables do not match")
    part_tokens = np.flatnonzero(levels == 0)
    if not part_tokens.size:
        raise ValueError("No part tokens are available for vocabulary extension")

    base_path = os.path.abspath(args.base_vocabulary)
    base = np.load(base_path).astype(np.float16)
    extended = np.concatenate((base, features[part_tokens]), axis=0)
    semantic_dtype = np.uint32 if extended.shape[0] > np.iinfo(np.uint16).max else np.uint16
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    semantic_ids = np.full((features.shape[0], 2), semantic_invalid, dtype=semantic_dtype)
    semantic_ids[part_tokens, 0] = (
        base.shape[0] + np.arange(part_tokens.size, dtype=np.int64)
    ).astype(semantic_dtype)

    point_dtype = np.uint32 if features.shape[0] > np.iinfo(np.uint16).max else np.uint16
    point_invalid = int(np.iinfo(point_dtype).max)
    packed_point_ids = np.full(point_ids.shape, point_invalid, dtype=point_dtype)
    part_point_valid = point_ids[:, 0] >= 0
    packed_point_ids[part_point_valid, 0] = point_ids[part_point_valid, 0].astype(
        point_dtype
    )
    packed_point_weights = np.zeros_like(point_weights, dtype=np.uint8)
    packed_point_weights[part_point_valid, 0] = 255

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), extended)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_reliability.npy"), reliability.astype(np.float16))
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), packed_point_weights)
    extension_bytes = int(features[part_tokens].nbytes)
    hierarchy_bytes = int(
        semantic_ids.nbytes
        + reliability.astype(np.float16).nbytes
        + packed_point_ids.nbytes
        + packed_point_weights.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "shared_codebook_group_hierarchy",
        "num_gaussians": int(point_ids.shape[0]),
        "num_group_codes": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "top_m": 1,
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "group_reliability": "group_reliability.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(packed_point_ids.dtype),
        "weight_dtype": "uint8_unit_membership",
        "covered_fraction": float(part_point_valid.mean()),
        "mean_ids_per_covered_gaussian": 1.0,
        "vocabulary": {
            "base_codes": int(base.shape[0]),
            "exact_part_codes": int(part_tokens.size),
            "total_codes": int(extended.shape[0]),
            "construction": "A14 joint vocabulary plus one exact codeword per reliable part token",
        },
        "storage": {
            "base_vocabulary_bytes_already_owned": int(base.nbytes),
            "part_vocabulary_extension_bytes_fp16": extension_bytes,
            "hierarchy_semantic_bytes": hierarchy_bytes,
            "total_semantic_bytes": extension_bytes + hierarchy_bytes,
            "bytes_per_gaussian_amortized": float(
                (extension_bytes + hierarchy_bytes) / point_ids.shape[0]
            ),
        },
        "source": {
            "group_codebook_dir": source_dir,
            "base_vocabulary": base_path,
            "leakage_control": "Exact rows copy training-derived reliable part tokens only",
        },
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
