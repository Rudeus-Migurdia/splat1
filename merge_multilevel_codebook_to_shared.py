#!/usr/bin/env python
"""Convert a residual multi-level codebook into one exact unit-sum shared table."""

import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def reconstruct_multilevel(codebooks, point_ids, invalid_id):
    """Return unnormalized residual reconstruction for a small batch of points."""
    output = np.zeros(
        (point_ids.shape[0], codebooks[0].shape[1]), dtype=np.float32
    )
    valid = np.all(point_ids != invalid_id, axis=1)
    for level, codebook in enumerate(codebooks):
        level_ids = point_ids[:, level]
        output[valid] += codebook[level_ids[valid]].astype(np.float32)
    return output, valid


def reconstruct_unit_sum(shared_codebook, point_ids, invalid_id):
    """Return unnormalized reconstruction from a shared table with unit weights."""
    output = np.zeros(
        (point_ids.shape[0], shared_codebook.shape[1]), dtype=np.float32
    )
    valid = np.all(point_ids != invalid_id, axis=1)
    for slot in range(point_ids.shape[1]):
        output[valid] += shared_codebook[point_ids[valid, slot]].astype(np.float32)
    return output, valid


def merge_multilevel_artifact(input_dir, output_dir, force=False, verification_chunk=8192):
    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    with open(input_dir / "manifest.json") as source:
        source_manifest = json.load(source)
    if source_manifest.get("representation") != "gaussian_multilevel_residual_codebook":
        raise ValueError("Input must be a gaussian_multilevel_residual_codebook artifact")

    levels = int(source_manifest["levels"])
    code_counts = [int(value) for value in source_manifest["code_counts"]]
    if levels != len(code_counts) or levels != len(source_manifest["codebook_files"]):
        raise ValueError("Inconsistent multi-level codebook manifest")
    invalid_id = int(source_manifest["invalid_id"])
    point_ids = np.load(input_dir / source_manifest["point_code_ids"])
    valid_mask = np.load(input_dir / source_manifest["valid_mask"]).astype(bool)
    if point_ids.shape != (int(source_manifest["num_gaussians"]), levels):
        raise ValueError("Point code IDs do not match the source manifest")
    if valid_mask.shape != (point_ids.shape[0],):
        raise ValueError("Valid mask does not match point IDs")

    codebooks = [
        np.load(input_dir / name)
        for name in source_manifest["codebook_files"]
    ]
    if any(codebook.ndim != 2 for codebook in codebooks):
        raise ValueError("Codebooks must be rank-2 arrays")
    feature_dim = int(source_manifest["feature_dim"])
    if any(codebook.shape[1] != feature_dim for codebook in codebooks):
        raise ValueError("Codebook feature dimensions do not match the manifest")

    total_codes = sum(code_counts)
    output_dtype = np.uint16 if total_codes <= np.iinfo(np.uint16).max else np.uint32
    if invalid_id > np.iinfo(output_dtype).max:
        raise ValueError("Output ID dtype cannot represent the source invalid ID")
    shared_ids = point_ids.astype(output_dtype, copy=True)
    offset = 0
    for level, code_count in enumerate(code_counts):
        level_ids = shared_ids[:, level]
        valid = level_ids != invalid_id
        if valid.any() and int(level_ids[valid].max()) >= code_count:
            raise ValueError(f"Level {level} contains IDs outside its codebook")
        level_ids[valid] += offset
        offset += code_count
    shared_codebook = np.concatenate(codebooks, axis=0)

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    np.save(output_dir / "codebook_shared.npy", shared_codebook)
    np.save(output_dir / "point_code_ids.npy", shared_ids)
    np.save(output_dir / "valid_mask.npy", valid_mask)

    max_abs_error = 0.0
    checked_points = 0
    for start in range(0, point_ids.shape[0], verification_chunk):
        end = min(start + verification_chunk, point_ids.shape[0])
        original, original_valid = reconstruct_multilevel(
            codebooks, point_ids[start:end], invalid_id
        )
        converted, converted_valid = reconstruct_unit_sum(
            shared_codebook, shared_ids[start:end], invalid_id
        )
        if not np.array_equal(original_valid, converted_valid):
            raise AssertionError("Converted valid-point mask differs from the source")
        if original_valid.any():
            max_abs_error = max(
                max_abs_error,
                float(np.abs(original[original_valid] - converted[original_valid]).max()),
            )
            checked_points += int(original_valid.sum())
    if max_abs_error != 0.0:
        raise AssertionError(f"Conversion is not exact: max abs error={max_abs_error}")

    codebook_bytes = int(shared_codebook.nbytes)
    point_id_bytes = int(shared_ids.nbytes)
    valid_mask_bytes = int(valid_mask.nbytes)
    total_semantic_bytes = codebook_bytes + point_id_bytes + valid_mask_bytes
    manifest = {
        "format_version": 1,
        "representation": "gaussian_adaptive_shared_codebook",
        "feature_dim": feature_dim,
        "num_gaussians": int(point_ids.shape[0]),
        "num_valid_gaussians": int(valid_mask.sum()),
        "valid_fraction": float(valid_mask.mean()),
        "num_codes": int(total_codes),
        "id_slots": levels,
        "codebook_files": ["codebook_shared.npy"],
        "point_code_ids": "point_code_ids.npy",
        "valid_mask": "valid_mask.npy",
        "id_dtype": np.dtype(output_dtype).name,
        "invalid_id": invalid_id,
        "weight_dtype": "implicit_unit",
        "codebook_composition": "unit_sum",
        "source": {
            "type": "exact_multilevel_to_shared_conversion",
            "source_artifact": str(input_dir),
            "source_representation": source_manifest["representation"],
        },
        "conversion": {
            "level_offsets": np.cumsum([0] + code_counts[:-1]).tolist(),
            "source_code_counts": code_counts,
            "verified_points": checked_points,
            "max_abs_reconstruction_error": max_abs_error,
        },
        "storage": {
            "codebook_bytes_fp16": codebook_bytes,
            "point_id_bytes": point_id_bytes,
            "point_weight_bytes": 0,
            "valid_mask_bytes": valid_mask_bytes,
            "total_semantic_bytes": total_semantic_bytes,
            "full_per_gaussian_fp16_bytes": point_ids.shape[0] * feature_dim * 2,
            "compression_ratio_vs_512d_fp16": (
                point_ids.shape[0] * feature_dim * 2 / max(1, total_semantic_bytes)
            ),
            "bytes_per_gaussian_amortized": total_semantic_bytes / point_ids.shape[0],
        },
    }
    with open(output_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def main():
    parser = ArgumentParser(
        description="Merge a residual multi-level codebook into an exact shared-table control."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--verification_chunk", type=int, default=8192)
    args = parser.parse_args()
    if args.verification_chunk <= 0:
        raise ValueError("--verification_chunk must be positive")
    manifest = merge_multilevel_artifact(
        args.input_dir,
        args.output_dir,
        force=args.force,
        verification_chunk=args.verification_chunk,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
