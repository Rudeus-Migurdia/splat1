#!/usr/bin/env python
"""Conservatively complete missing codebook IDs from nearby supported Gaussians."""

import json
import os
import shutil
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def select_nearest_fill(distances, valid_count, total_count, target_coverage):
    """Select the nearest unsupported points needed to reach target coverage."""
    target_count = int(round(float(target_coverage) * total_count))
    needed = max(0, min(int(distances.size), target_count - int(valid_count)))
    selected = np.zeros(distances.shape[0], dtype=bool)
    if needed == 0:
        return selected, None
    if needed == distances.size:
        selected.fill(True)
    else:
        chosen = np.argpartition(distances, needed - 1)[:needed]
        selected[chosen] = True
    return selected, float(distances[selected].max())


def load_checkpoint_xyz(checkpoint_path):
    import torch

    payload = torch.load(checkpoint_path, map_location="cpu")
    if not isinstance(payload, (tuple, list)) or len(payload) < 1:
        raise ValueError("Expected a Gaussian checkpoint tuple")
    model_params = payload[0]
    if not isinstance(model_params, (tuple, list)) or len(model_params) < 2:
        raise ValueError("Checkpoint does not expose Gaussian positions")
    xyz = model_params[1]
    if not isinstance(xyz, torch.Tensor) or xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError("Checkpoint Gaussian positions must have shape [N, 3]")
    return xyz.detach().cpu().numpy().astype(np.float32, copy=False)


def load_source_mask(source_mask_path, total_count):
    """Load an optional boolean gate for codebook IDs allowed to seed propagation."""
    if source_mask_path is None:
        return None
    source_mask = np.load(source_mask_path).astype(bool, copy=False)
    if source_mask.shape != (total_count,):
        raise ValueError(
            "Source mask must have shape "
            f"({total_count},), got {source_mask.shape}"
        )
    return source_mask


def propagate_coverage(
    input_dir,
    geometry_checkpoint,
    output_dir,
    target_coverage,
    source_mask_path=None,
    force=False,
):
    from scipy.spatial import cKDTree

    input_dir = Path(input_dir).resolve()
    output_dir = Path(output_dir).resolve()
    with open(input_dir / "manifest.json") as source:
        source_manifest = json.load(source)
    point_ids_name = source_manifest["point_code_ids"]
    valid_mask_name = source_manifest["valid_mask"]
    invalid_id = int(source_manifest["invalid_id"])
    point_ids = np.load(input_dir / point_ids_name)
    valid_mask = np.load(input_dir / valid_mask_name).astype(bool)
    if point_ids.shape[0] != valid_mask.shape[0]:
        raise ValueError("Point IDs and valid mask do not match")
    if not 0.0 < target_coverage <= 1.0:
        raise ValueError("--target_coverage must be in (0, 1]")

    valid_mask &= np.all(point_ids != invalid_id, axis=1)
    initial_count = int(valid_mask.sum())
    total_count = int(valid_mask.size)
    initial_coverage = initial_count / total_count
    if target_coverage < initial_coverage:
        raise ValueError("Target coverage cannot be below the input artifact coverage")

    xyz = load_checkpoint_xyz(geometry_checkpoint)
    if xyz.shape[0] != total_count:
        raise ValueError("Geometry checkpoint and codebook artifact have different point counts")
    source_mask = load_source_mask(source_mask_path, total_count)
    source_eligible = valid_mask.copy()
    if source_mask is not None:
        source_eligible &= source_mask
    valid_indices = np.flatnonzero(source_eligible)
    if not valid_indices.size:
        raise ValueError("No valid codebook IDs remain after applying the source gate")
    missing_indices = np.flatnonzero(~valid_mask)
    tree = cKDTree(xyz[valid_indices])
    distances, nearest_local = tree.query(xyz[missing_indices], k=1, workers=-1)
    selected, radius = select_nearest_fill(
        distances,
        initial_count,
        total_count,
        target_coverage,
    )
    propagated_indices = missing_indices[selected]
    propagated_sources = valid_indices[nearest_local[selected]]
    output_ids = point_ids.copy()
    output_ids[propagated_indices] = point_ids[propagated_sources]
    output_valid = valid_mask.copy()
    output_valid[propagated_indices] = True
    propagated_mask = np.zeros(total_count, dtype=bool)
    propagated_mask[propagated_indices] = True

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output exists: {output_dir}; pass --force to replace it")
        shutil.rmtree(output_dir)
    shutil.copytree(input_dir, output_dir)
    np.save(output_dir / point_ids_name, output_ids)
    np.save(output_dir / valid_mask_name, output_valid)
    np.save(output_dir / "propagated_mask.npy", propagated_mask)

    manifest = dict(source_manifest)
    manifest["num_valid_gaussians"] = int(output_valid.sum())
    manifest["valid_fraction"] = float(output_valid.mean())
    manifest["source_initialization"] = source_manifest.get("source")
    manifest["source"] = {
        "type": "geometry_gated_codebook_id_propagation",
        "input_artifact": str(input_dir),
        "geometry_checkpoint": str(Path(geometry_checkpoint).resolve()),
    }
    manifest["coverage_completion"] = {
        "target_coverage": float(target_coverage),
        "initial_coverage": float(initial_coverage),
        "final_coverage": float(output_valid.mean()),
        "propagated_count": int(propagated_mask.sum()),
        "max_propagation_distance": radius,
        "propagated_mask": "propagated_mask.npy",
        "method": "nearest_supported_gaussian",
        "source_count": int(valid_indices.size),
        "source_fraction": float(valid_indices.size / total_count),
        "source_gate": (
            str(Path(source_mask_path).resolve()) if source_mask_path is not None else None
        ),
    }
    with open(output_dir / "manifest.json", "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def main():
    parser = ArgumentParser(
        description="Copy discrete IDs to the nearest supported Gaussians up to a coverage budget."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_coverage", type=float, required=True)
    parser.add_argument(
        "--source_mask",
        default=None,
        help="Optional boolean Nx1 .npy mask restricting valid IDs allowed as propagation sources.",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    manifest = propagate_coverage(
        args.input_dir,
        args.geometry_checkpoint,
        args.output_dir,
        args.target_coverage,
        source_mask_path=args.source_mask,
        force=args.force,
    )
    print(json.dumps(manifest["coverage_completion"], indent=2))


if __name__ == "__main__":
    main()
