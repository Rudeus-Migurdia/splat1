#!/usr/bin/env python
"""Compose exact part and sparse boundary semantics in one shared vocabulary."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from validate_semantic_vocabulary_contract import validate_contract


def l2_normalize(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--part_artifact_dir", required=True)
    parser.add_argument("--boundary_hypothesis_dir", required=True)
    parser.add_argument("--interior_support", required=True)
    parser.add_argument("--maximum_interior_support", type=float, default=0.75)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(sys.argv[1:])
    if not 0.0 < args.maximum_interior_support < 1.0:
        raise ValueError("Boundary interior cutoff must be in (0, 1)")

    part_dir = os.path.abspath(args.part_artifact_dir)
    with open(os.path.join(part_dir, "manifest.json")) as source:
        part_manifest = json.load(source)
    part_vocabulary = np.load(
        os.path.join(part_dir, part_manifest["group_codebook"])
    ).astype(np.float16)
    part_semantic_ids = np.load(
        os.path.join(part_dir, part_manifest["group_semantic_code_ids"])
    ).astype(np.int64)
    part_semantic_invalid = int(part_manifest["semantic_invalid_id"])
    part_semantic_ids[part_semantic_ids == part_semantic_invalid] = -1
    part_reliability = np.load(
        os.path.join(part_dir, part_manifest["group_reliability"])
    ).astype(np.float32)
    part_point_ids = np.load(
        os.path.join(part_dir, part_manifest["point_group_ids"])
    ).astype(np.int64)
    part_point_weights = np.load(
        os.path.join(part_dir, part_manifest["point_group_weights"])
    ).astype(np.uint8)
    part_point_invalid = int(part_manifest["invalid_id"])
    part_point_ids[part_point_ids == part_point_invalid] = -1

    boundary_dir = os.path.abspath(args.boundary_hypothesis_dir)
    with open(os.path.join(boundary_dir, "manifest.json")) as source:
        boundary_manifest = json.load(source)
    boundary_points = np.load(
        os.path.join(boundary_dir, boundary_manifest["point_ids"])
    ).astype(np.int64)
    boundary_features = l2_normalize(
        np.load(os.path.join(boundary_dir, boundary_manifest["features"]))
    ).astype(np.float16)
    boundary_reliability = np.load(
        os.path.join(boundary_dir, boundary_manifest["reliability"])
    ).astype(np.float32) / 255.0
    if not (
        boundary_points.shape
        == boundary_reliability.shape
        == (boundary_features.shape[0],)
    ):
        raise ValueError("Boundary hypothesis arrays must match")
    support = np.load(os.path.abspath(args.interior_support)).astype(np.float32)
    if support.shape != (part_point_ids.shape[0],):
        raise ValueError("Interior support does not match the part artifact")
    if boundary_points.size and int(boundary_points.max()) >= support.size:
        raise ValueError("Boundary point IDs exceed the part artifact")

    has_part = part_point_ids[:, 0] >= 0
    selected = has_part[boundary_points] & (
        support[boundary_points] < args.maximum_interior_support
    )
    boundary_points = boundary_points[selected]
    boundary_features = boundary_features[selected]
    boundary_reliability = boundary_reliability[selected]
    if not boundary_points.size:
        raise ValueError("No reproducible boundary modes satisfy the part boundary rule")
    if np.unique(boundary_points).size != boundary_points.size:
        raise ValueError("Boundary hypothesis contains duplicate Gaussian IDs")

    extended_vocabulary = np.concatenate(
        (part_vocabulary, boundary_features), axis=0
    )
    semantic_dtype = (
        np.uint32
        if extended_vocabulary.shape[0] > np.iinfo(np.uint16).max
        else np.uint16
    )
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    token_count = part_semantic_ids.shape[0] + boundary_points.size
    semantic_ids = np.full((token_count, 2), semantic_invalid, dtype=semantic_dtype)
    part_valid = part_semantic_ids >= 0
    semantic_ids[: part_semantic_ids.shape[0]][part_valid] = part_semantic_ids[
        part_valid
    ].astype(semantic_dtype)
    boundary_offset = part_semantic_ids.shape[0]
    semantic_ids[boundary_offset:, 0] = (
        part_vocabulary.shape[0] + np.arange(boundary_points.size)
    ).astype(semantic_dtype)
    reliability = np.concatenate((part_reliability, boundary_reliability)).astype(
        np.float16
    )

    point_dtype = np.uint32 if token_count > np.iinfo(np.uint16).max else np.uint16
    point_invalid = int(np.iinfo(point_dtype).max)
    point_ids = np.full((support.size, 2), point_invalid, dtype=point_dtype)
    point_weights = np.zeros((support.size, 2), dtype=np.uint8)
    part_valid_points = part_point_ids[:, 0] >= 0
    point_ids[part_valid_points, 0] = part_point_ids[part_valid_points, 0].astype(
        point_dtype
    )
    point_weights[part_valid_points, 0] = part_point_weights[part_valid_points, 0]
    point_ids[boundary_points, 1] = (
        boundary_offset + np.arange(boundary_points.size)
    ).astype(point_dtype)
    boundary_membership = (1.0 - support[boundary_points]).clip(0.0, 1.0)
    point_weights[boundary_points, 1] = np.rint(boundary_membership * 255.0).astype(
        np.uint8
    )

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), extended_vocabulary)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_reliability.npy"), reliability)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    extension_bytes = int(boundary_features.nbytes)
    part_extension_bytes = int(
        part_manifest["storage"]["part_vocabulary_extension_bytes_fp16"]
    )
    table_bytes = int(
        semantic_ids.nbytes
        + reliability.nbytes
        + point_ids.nbytes
        + point_weights.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "shared_codebook_group_hierarchy",
        "num_gaussians": int(support.size),
        "num_group_codes": int(token_count),
        "feature_dim": int(extended_vocabulary.shape[1]),
        "top_m": 2,
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "group_reliability": "group_reliability.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(point_ids.dtype),
        "weight_dtype": "uint8_part_interior_and_boundary_complement",
        "covered_fraction": float((point_weights > 0).any(axis=1).mean()),
        "mean_ids_per_covered_gaussian": float(
            (point_weights[(point_weights > 0).any(axis=1)] > 0).sum(axis=1).mean()
        ),
        "vocabulary_modalities": ["base", "part", "boundary"],
        "modality_token_counts": {
            "base": int(part_manifest["vocabulary"]["base_codes"]),
            "part": int(part_manifest["vocabulary"]["exact_part_codes"]),
            "boundary": int(boundary_points.size),
        },
        "boundary": {
            "num_input_hypotheses": int(selected.size),
            "num_selected_tokens": int(boundary_points.size),
            "selected_fraction_of_input": float(boundary_points.size / selected.size),
            "covered_fraction": float((point_weights[:, 1] > 0).mean()),
            "mean_membership": float(boundary_membership.mean()),
            "mean_reliability": float(boundary_reliability.mean()),
        },
        "storage": {
            "part_artifact_bytes_already_owned": int(
                part_manifest["storage"]["total_semantic_bytes"]
            ),
            "part_vocabulary_extension_bytes_fp16": part_extension_bytes,
            "boundary_vocabulary_extension_bytes_fp16": extension_bytes,
            "composite_table_bytes": table_bytes,
            "total_additional_bytes": extension_bytes + table_bytes,
            "total_semantic_bytes": (
                part_extension_bytes + extension_bytes + table_bytes
            ),
            "bytes_per_gaussian_amortized": float(
                (extension_bytes + table_bytes) / support.size
            ),
        },
        "source": {
            "part_artifact_dir": part_dir,
            "boundary_hypothesis_dir": boundary_dir,
            "interior_support": os.path.abspath(args.interior_support),
            "leakage_control": "Training-only split-reproducible signed-ownership modes and 3D part interior support",
        },
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    contract = validate_contract(output_dir, ["base", "part", "boundary"])
    print(json.dumps({"manifest": manifest, "contract": contract}, indent=2))


if __name__ == "__main__":
    main()
