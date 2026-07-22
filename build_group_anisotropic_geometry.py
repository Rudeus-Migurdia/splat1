#!/usr/bin/env python
"""Build query-independent Group shape tensors for anisotropic propagation."""

import argparse
import json
import os
import time

import numpy as np

from build_geometry_conditioned_tracklet_partition import load_atom_geometry


def weighted_group_shape(centroids, weights):
    centroids = np.asarray(centroids, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if centroids.ndim != 2 or centroids.shape[1] != 3 or weights.shape != (len(centroids),):
        raise ValueError("Group centroids and weights must have shapes [M, 3] and [M]")
    valid = np.isfinite(weights) & (weights > 0.0) & np.isfinite(centroids).all(axis=1)
    if valid.sum() < 3:
        return np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32), 0.0, 0.0
    points = centroids[valid]
    local_weights = weights[valid]
    local_weights /= local_weights.sum()
    center = (local_weights[:, None] * points).sum(axis=0)
    centered = points - center
    covariance = np.einsum("n,ni,nj->ij", local_weights, centered, centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    axes = eigenvectors[:, order].T
    if eigenvalues[0] <= 1e-12:
        return np.eye(3, dtype=np.float32), np.ones(3, dtype=np.float32), 0.0, 0.0
    ratios = np.sqrt(eigenvalues / eigenvalues[0]).clip(1e-3, 1.0)
    linearity = 1.0 - eigenvalues[1] / eigenvalues[0]
    planarity = (eigenvalues[1] - eigenvalues[2]) / eigenvalues[0]
    return (
        axes.astype(np.float32),
        ratios.astype(np.float32),
        float(np.clip(linearity, 0.0, 1.0)),
        float(np.clip(planarity, 0.0, 1.0)),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spatial_posterior_dir", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--minimum_membership", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    if not 0.0 < args.minimum_membership <= 1.0:
        raise ValueError("minimum_membership must be in (0, 1]")

    source_dir = os.path.abspath(args.spatial_posterior_dir)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse Group anisotropic geometry: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    with open(os.path.join(source_dir, "manifest.json")) as source:
        source_manifest = json.load(source)
    if source_manifest.get("representation") != "query_conditioned_top2_spatial_group_posterior":
        raise ValueError("Anisotropic geometry requires an A52-compatible spatial posterior")
    if int(source_manifest["seed"]) != args.seed:
        raise ValueError("Spatial posterior seed does not match the anisotropic build seed")

    def load_source(name):
        return np.load(os.path.join(source_dir, source_manifest[name]))

    gaussian_atom_ids = load_source("gaussian_atom_ids").astype(np.int64)
    point_group_ids = load_source("point_group_ids").astype(np.int64)
    point_memberships = load_source("point_group_memberships").astype(np.float32)
    num_gaussians = int(source_manifest["num_gaussians"])
    num_atoms = int(source_manifest["num_atoms"])
    num_groups = int(source_manifest["num_groups"])
    if gaussian_atom_ids.shape != (num_gaussians,):
        raise ValueError("Gaussian atom IDs do not match the source manifest")
    first_gaussian = np.full(num_atoms, -1, dtype=np.int64)
    first_gaussian[gaussian_atom_ids] = np.arange(num_gaussians, dtype=np.int64)
    if (first_gaussian < 0).any():
        raise ValueError("Every spatial atom must contain at least one Gaussian")
    atom_group_ids = point_group_ids[first_gaussian].reshape(num_atoms, -1)
    atom_memberships = point_memberships[first_gaussian].reshape(num_atoms, -1)

    atom_geometry = load_atom_geometry(args.geometry_checkpoint, os.path.join(source_dir, source_manifest["gaussian_atom_ids"]))
    atom_centroids = atom_geometry["centroid"].astype(np.float32)
    atom_mass = (
        atom_geometry["gaussian_count"].astype(np.float64)
        * np.maximum(atom_geometry["opacity"].astype(np.float64), 1e-4)
    )
    axes = np.repeat(np.eye(3, dtype=np.float32)[None], num_groups, axis=0)
    ratios = np.ones((num_groups, 3), dtype=np.float32)
    linearity = np.zeros(num_groups, dtype=np.float32)
    planarity = np.zeros(num_groups, dtype=np.float32)
    atom_counts = np.zeros(num_groups, dtype=np.int32)

    for group_id in range(num_groups):
        membership = np.where(atom_group_ids == group_id, atom_memberships, 0.0).max(axis=1)
        selected = membership >= args.minimum_membership
        atom_counts[group_id] = int(selected.sum())
        shape = weighted_group_shape(
            atom_centroids[selected], membership[selected] * atom_mass[selected]
        )
        axes[group_id], ratios[group_id], linearity[group_id], planarity[group_id] = shape

    arrays = {
        "atom_centroids.npy": atom_centroids,
        "group_principal_axes.npy": axes.astype(np.float16),
        "group_axis_ratios.npy": ratios.astype(np.float16),
        "group_linearity.npy": linearity.astype(np.float16),
        "group_planarity.npy": planarity.astype(np.float16),
        "group_atom_counts.npy": atom_counts,
    }
    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array)
    valid_shape = atom_counts >= 3
    manifest = {
        "format_version": 1,
        "representation": "group_anisotropic_propagation_geometry",
        "method": "weighted_3d_group_covariance_diffusion_tensor",
        "scene": source_manifest["scene"],
        "seed": args.seed,
        "num_gaussians": num_gaussians,
        "num_atoms": num_atoms,
        "num_groups": num_groups,
        "atom_centroids": "atom_centroids.npy",
        "group_principal_axes": "group_principal_axes.npy",
        "group_axis_ratios": "group_axis_ratios.npy",
        "group_linearity": "group_linearity.npy",
        "group_planarity": "group_planarity.npy",
        "group_atom_counts": "group_atom_counts.npy",
        "statistics": {
            "valid_shape_groups": int(valid_shape.sum()),
            "linearity_quantiles": np.quantile(linearity[valid_shape], [0, 0.25, 0.5, 0.75, 1]).tolist() if valid_shape.any() else [],
            "second_axis_ratio_quantiles": np.quantile(ratios[valid_shape, 1], [0, 0.25, 0.5, 0.75, 1]).tolist() if valid_shape.any() else [],
        },
        "source_contract": {
            "query_independent": True,
            "evaluation_queries_or_labels_used": False,
            "resident_tokens_modified": False,
            "codebooks_trained": False,
            "fixed_seed": args.seed,
        },
        "source": {
            "spatial_posterior_dir": source_dir,
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        },
        "args": vars(args),
        "storage_bytes": int(sum(array.nbytes for array in arrays.values())),
        "elapsed_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
