#!/usr/bin/env python
"""Complete a discrete codebook with agreement-gated local source mixtures."""

import json
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from propagate_gaussian_codebook_coverage import load_checkpoint_xyz, select_nearest_fill


def normalized_cosine(first, second, eps=1e-8):
    first = np.asarray(first, dtype=np.float32)
    second = np.asarray(second, dtype=np.float32)
    numerator = np.einsum("ij,ij->i", first, second)
    denominator = np.maximum(
        np.linalg.norm(first, axis=1) * np.linalg.norm(second, axis=1), eps
    )
    return numerator / denominator


def inverse_distance_weights(first_distance, second_distance):
    """Return quantized two-neighbor weights that sum to one before rounding."""
    total = np.maximum(first_distance + second_distance, 1e-8)
    first = second_distance / total
    second = first_distance / total
    return np.rint(np.stack([first, second], axis=1) * 255.0).clip(0, 255).astype(
        np.uint8
    )


def _shared_ids(point_ids, code_counts, invalid_id, dtype):
    output = point_ids.astype(dtype, copy=True)
    offset = 0
    for level, code_count in enumerate(code_counts):
        valid = output[:, level] != invalid_id
        if valid.any() and int(output[valid, level].max()) >= code_count:
            raise ValueError(f"Level {level} contains IDs outside its codebook")
        output[valid, level] += offset
        offset += code_count
    return output


def build_consensus_propagated_codebook(
    input_dir,
    geometry_checkpoint,
    output_dir,
    target_coverage,
    min_agreement=0.8,
    max_neighbor_distance_ratio=1.5,
    force=False,
):
    """Fill nearby missing points with two-source mixtures only in coherent regions."""
    from scipy.spatial import cKDTree

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    source_manifest = json.load(open(input_dir / "manifest.json"))
    if source_manifest.get("representation") != "gaussian_multilevel_residual_codebook":
        raise ValueError("Input must be a gaussian_multilevel_residual_codebook artifact")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("target_coverage must be in (0, 1]")
    if not -1.0 <= min_agreement <= 1.0:
        raise ValueError("min_agreement must be in [-1, 1]")
    if max_neighbor_distance_ratio < 1.0:
        raise ValueError("max_neighbor_distance_ratio must be at least one")

    levels = int(source_manifest["levels"])
    if levels != 2:
        raise ValueError("This probe currently expects a two-level residual codebook")
    code_counts = [int(value) for value in source_manifest["code_counts"]]
    invalid_id = int(source_manifest["invalid_id"])
    point_ids = np.load(input_dir / source_manifest["point_code_ids"])
    valid_mask = np.load(input_dir / source_manifest["valid_mask"]).astype(bool)
    valid_mask &= np.all(point_ids != invalid_id, axis=1)
    total_count = int(valid_mask.size)
    valid_count = int(valid_mask.sum())
    if target_coverage < valid_count / total_count:
        raise ValueError("Target coverage cannot be below the input artifact coverage")

    codebooks = [np.load(input_dir / name) for name in source_manifest["codebook_files"]]
    feature_dim = int(source_manifest["feature_dim"])
    if any(table.shape[1] != feature_dim for table in codebooks):
        raise ValueError("Codebook feature dimensions do not match the manifest")
    shared_codebook = np.concatenate(codebooks, axis=0)
    output_dtype = np.uint16 if shared_codebook.shape[0] <= np.iinfo(np.uint16).max else np.uint32
    if invalid_id > np.iinfo(output_dtype).max:
        raise ValueError("Output ID dtype cannot represent the invalid ID")
    shared_ids = _shared_ids(point_ids, code_counts, invalid_id, output_dtype)

    xyz = load_checkpoint_xyz(geometry_checkpoint)
    if xyz.shape != (total_count, 3):
        raise ValueError("Geometry checkpoint and codebook artifact have different point counts")
    source_indices = np.flatnonzero(valid_mask)
    missing_indices = np.flatnonzero(~valid_mask)
    tree = cKDTree(xyz[source_indices])
    distances, local_neighbors = tree.query(xyz[missing_indices], k=2, workers=-1)
    selected, radius = select_nearest_fill(
        distances[:, 0], valid_count, total_count, target_coverage
    )
    propagated_indices = missing_indices[selected]
    propagated_sources = source_indices[local_neighbors[selected]]

    output_ids = np.full(
        (total_count, levels * 2), invalid_id, dtype=output_dtype
    )
    output_weights = np.zeros(output_ids.shape, dtype=np.uint8)
    output_ids[valid_mask, :levels] = shared_ids[valid_mask]
    output_weights[valid_mask, :levels] = 255
    output_ids[propagated_indices, :levels] = shared_ids[propagated_sources[:, 0]]
    output_weights[propagated_indices, :levels] = 255

    reconstructed_sources = (
        codebooks[0][point_ids[propagated_sources, 0]].astype(np.float32)
        + codebooks[1][point_ids[propagated_sources, 1]].astype(np.float32)
    )
    agreement = normalized_cosine(
        reconstructed_sources[:, 0], reconstructed_sources[:, 1]
    )
    selected_distances = distances[selected]
    distance_ratio = selected_distances[:, 1] / np.maximum(
        selected_distances[:, 0], 1e-8
    )
    blend = (agreement >= min_agreement) & (
        distance_ratio <= max_neighbor_distance_ratio
    )
    if blend.any():
        blended_indices = propagated_indices[blend]
        blended_sources = propagated_sources[blend]
        weights = inverse_distance_weights(
            selected_distances[blend, 0], selected_distances[blend, 1]
        )
        output_ids[blended_indices, :levels] = shared_ids[blended_sources[:, 0]]
        output_ids[blended_indices, levels:] = shared_ids[blended_sources[:, 1]]
        output_weights[blended_indices, :levels] = weights[:, :1]
        output_weights[blended_indices, levels:] = weights[:, 1:]

    output_valid = valid_mask.copy()
    output_valid[propagated_indices] = True
    propagated_mask = np.zeros(total_count, dtype=bool)
    propagated_mask[propagated_indices] = True
    blend_mask = np.zeros(total_count, dtype=bool)
    blend_mask[propagated_indices[blend]] = True

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    np.save(output_dir / "codebook_shared.npy", shared_codebook)
    np.save(output_dir / "point_code_ids.npy", output_ids)
    np.save(output_dir / "point_code_weights.npy", output_weights)
    np.save(output_dir / "valid_mask.npy", output_valid)
    np.save(output_dir / "propagated_mask.npy", propagated_mask)
    np.save(output_dir / "blend_mask.npy", blend_mask)

    codebook_bytes = int(shared_codebook.nbytes)
    point_id_bytes = int(output_ids.nbytes)
    point_weight_bytes = int(output_weights.nbytes)
    valid_mask_bytes = int(output_valid.nbytes)
    manifest = {
        "format_version": 1,
        "representation": "gaussian_adaptive_shared_codebook",
        "feature_dim": feature_dim,
        "num_gaussians": total_count,
        "num_valid_gaussians": int(output_valid.sum()),
        "valid_fraction": float(output_valid.mean()),
        "num_codes": int(shared_codebook.shape[0]),
        "id_slots": levels * 2,
        "codebook_files": ["codebook_shared.npy"],
        "point_code_ids": "point_code_ids.npy",
        "point_code_weights": "point_code_weights.npy",
        "valid_mask": "valid_mask.npy",
        "id_dtype": np.dtype(output_dtype).name,
        "invalid_id": invalid_id,
        "weight_dtype": "uint8_normalized",
        "codebook_composition": "agreement_gated_two_source_mixture",
        "source": {
            "type": "local_consensus_discrete_codebook_completion",
            "source_artifact": str(input_dir),
            "geometry_checkpoint": str(Path(geometry_checkpoint).resolve()),
        },
        "coverage_completion": {
            "method": "agreement_gated_two_source_mixture",
            "target_coverage": float(target_coverage),
            "initial_coverage": float(valid_count / total_count),
            "final_coverage": float(output_valid.mean()),
            "propagated_count": int(propagated_mask.sum()),
            "max_propagation_distance": radius,
            "min_agreement": float(min_agreement),
            "max_neighbor_distance_ratio": float(max_neighbor_distance_ratio),
            "blended_count": int(blend_mask.sum()),
            "blended_fraction_of_propagated": float(
                blend_mask.sum() / max(1, propagated_mask.sum())
            ),
            "mean_blend_agreement": float(agreement[blend].mean()) if blend.any() else None,
        },
        "storage": {
            "codebook_bytes_fp16": codebook_bytes,
            "point_id_bytes": point_id_bytes,
            "point_weight_bytes": point_weight_bytes,
            "valid_mask_bytes": valid_mask_bytes,
            "total_semantic_bytes": (
                codebook_bytes + point_id_bytes + point_weight_bytes + valid_mask_bytes
            ),
            "full_per_gaussian_fp16_bytes": total_count * feature_dim * 2,
            "compression_ratio_vs_512d_fp16": (
                total_count * feature_dim * 2
                / max(1, codebook_bytes + point_id_bytes + point_weight_bytes + valid_mask_bytes)
            ),
            "bytes_per_gaussian_amortized": (
                codebook_bytes + point_id_bytes + point_weight_bytes + valid_mask_bytes
            )
            / total_count,
        },
    }
    with open(output_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def main():
    parser = ArgumentParser(
        description="Complete a discrete codebook with agreement-gated two-source mixtures."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_coverage", type=float, required=True)
    parser.add_argument("--min_agreement", type=float, default=0.8)
    parser.add_argument("--max_neighbor_distance_ratio", type=float, default=1.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            build_consensus_propagated_codebook(
                args.input_dir,
                args.geometry_checkpoint,
                args.output_dir,
                args.target_coverage,
                args.min_agreement,
                args.max_neighbor_distance_ratio,
                force=args.force,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
