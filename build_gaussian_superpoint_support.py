#!/usr/bin/env python
"""Build bounded 3D Gaussian superpoints and candidate-support priors."""

import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch

from train_joint_query_preserving_vocabulary import FixedSharedAssignment


SH_C0 = 0.28209479177387814


class BoundedUnionFind:
    def __init__(self, count):
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, index):
        root = int(index)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[index] != index:
            parent = int(self.parent[index])
            self.parent[index] = root
            index = parent
        return root

    def union(self, first, second, maximum_size):
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.size[first_root] + self.size[second_root] > maximum_size:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        return True

    def roots(self):
        return np.asarray(
            [self.find(index) for index in range(self.parent.size)],
            dtype=np.int32,
        )


def compact_components(union_find):
    roots = union_find.roots()
    _, labels = np.unique(roots, return_inverse=True)
    return labels.astype(np.int32)


def leave_one_out_candidate_support(labels, candidate_mask):
    labels = np.asarray(labels, dtype=np.int64)
    candidate_mask = np.asarray(candidate_mask, dtype=bool)
    if labels.shape != candidate_mask.shape:
        raise ValueError("Labels and candidate mask must have matching shapes")
    component_count = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, minlength=component_count)
    candidate_counts = np.bincount(
        labels, weights=candidate_mask.astype(np.float32), minlength=component_count
    )
    denominator = sizes[labels] - 1
    numerator = candidate_counts[labels] - candidate_mask.astype(np.float32)
    support = np.zeros(labels.shape, dtype=np.float32)
    valid = candidate_mask & (denominator > 0)
    support[valid] = numerator[valid] / denominator[valid]
    return support, sizes, candidate_counts


def load_geometry(checkpoint_path, expected_count):
    model_params, iteration = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) not in (12, 13):
        raise ValueError("Unsupported geometry checkpoint tuple")
    xyz = model_params[1].detach().float().numpy()
    features_dc = model_params[2].detach().float().numpy()
    log_scaling = model_params[4].detach().float().numpy()
    if xyz.shape != (expected_count, 3):
        raise ValueError("Geometry coordinates do not match the codebook")
    rgb = np.clip(
        0.5 + SH_C0 * features_dc.reshape(expected_count, -1, 3)[:, 0],
        0.0,
        1.0,
    )
    log_scale = log_scaling.mean(axis=1)
    return (
        np.ascontiguousarray(xyz, dtype=np.float32),
        np.ascontiguousarray(rgb, dtype=np.float32),
        np.asarray(log_scale, dtype=np.float32),
        int(iteration),
    )


def project_codebook(codebook, output_dim, seed, device):
    values = torch.from_numpy(codebook).float().to(device)
    if output_dim >= values.shape[1]:
        return torch.nn.functional.normalize(values, dim=-1).cpu().numpy()
    torch.manual_seed(seed)
    _, _, basis = torch.pca_lowrank(
        values,
        q=output_dim,
        center=False,
        niter=3,
    )
    projected = torch.nn.functional.normalize(values @ basis, dim=-1)
    return projected.cpu().numpy().astype(np.float32)


def reconstruct_projected_semantics(
    assignment,
    projected_codebook,
    valid_indices,
    chunk_size,
):
    output = np.zeros((valid_indices.size, projected_codebook.shape[1]), dtype=np.float16)
    for start in range(0, valid_indices.size, chunk_size):
        indices = valid_indices[start : start + chunk_size]
        ids = assignment.ids[indices]
        weights = assignment.weights[indices]
        valid = ids >= 0
        safe = np.maximum(ids, 0)
        reconstruction = (
            projected_codebook[safe]
            * weights[..., None]
            * valid[..., None]
        ).sum(axis=1)
        reconstruction /= np.maximum(
            np.linalg.norm(reconstruction, axis=1, keepdims=True), 1e-8
        )
        output[start : start + indices.size] = reconstruction.astype(np.float16)
    return output


def build_knn(points, neighbors, chunk_size, use_gpu, workers):
    try:
        import faiss
    except ImportError:
        faiss = None
    resources = None
    if use_gpu and faiss is not None and hasattr(faiss, "StandardGpuResources"):
        index = faiss.IndexFlatL2(3)
        index.add(np.ascontiguousarray(points, dtype=np.float32))
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
        backend = "faiss_gpu_flat_l2"

        def query(values):
            return index.search(values, neighbors + 1)

    else:
        from scipy.spatial import cKDTree

        tree = cKDTree(points)
        backend = "scipy_ckdtree_exact"

        def query(values):
            distances, indices = tree.query(
                values,
                k=neighbors + 1,
                workers=workers,
            )
            # FAISS IndexFlatL2 returns squared Euclidean distance.
            return np.square(distances, dtype=np.float32), indices

    all_neighbors = np.empty((points.shape[0], neighbors), dtype=np.int32)
    all_distances = np.empty((points.shape[0], neighbors), dtype=np.float32)
    for start in range(0, points.shape[0], chunk_size):
        end = min(start + chunk_size, points.shape[0])
        distances, local_neighbors = query(
            np.ascontiguousarray(points[start:end], dtype=np.float32)
        )
        rows = np.arange(start, end, dtype=np.int64)
        is_self = local_neighbors == rows[:, None]
        distances[is_self] = np.inf
        order = np.argsort(distances, axis=1)[:, :neighbors]
        all_distances[start:end] = np.take_along_axis(distances, order, axis=1)
        all_neighbors[start:end] = np.take_along_axis(local_neighbors, order, axis=1)
    return all_neighbors, all_distances, resources, backend


def build_superpoints(
    neighbors,
    distances,
    rgb,
    log_scale,
    semantics,
    spatial_radius_factor,
    rgb_threshold,
    log_scale_threshold,
    semantic_threshold,
    maximum_size,
    chunk_size,
):
    count = neighbors.shape[0]
    geometry_union = BoundedUnionFind(count)
    semantic_union = BoundedUnionFind(count)
    radius = distances[:, -1]
    geometry_edges = 0
    semantic_edges = 0
    for start in range(0, count, chunk_size):
        end = min(start + chunk_size, count)
        rows = np.arange(start, end, dtype=np.int32)[:, None]
        adjacent = neighbors[start:end]
        edge_distance = distances[start:end]
        valid = (adjacent >= 0) & (rows < adjacent)
        radius_limit = spatial_radius_factor * np.minimum(
            radius[start:end, None], radius[adjacent]
        )
        valid &= edge_distance <= radius_limit
        rgb_distance = np.linalg.norm(
            rgb[start:end, None, :] - rgb[adjacent], axis=-1
        )
        valid &= rgb_distance <= rgb_threshold
        scale_distance = np.abs(
            log_scale[start:end, None] - log_scale[adjacent]
        )
        valid &= scale_distance <= log_scale_threshold

        row_ids, slots = np.nonzero(valid)
        first = row_ids.astype(np.int64) + start
        second = adjacent[row_ids, slots].astype(np.int64)
        if first.size == 0:
            continue
        semantic_similarity = np.sum(
            semantics[first].astype(np.float32)
            * semantics[second].astype(np.float32),
            axis=1,
        )
        semantic_valid = semantic_similarity >= semantic_threshold
        geometry_edges += int(first.size)
        semantic_edges += int(semantic_valid.sum())
        for edge_index, (first_id, second_id) in enumerate(zip(first, second)):
            geometry_union.union(first_id, second_id, maximum_size)
            if semantic_valid[edge_index]:
                semantic_union.union(first_id, second_id, maximum_size)
    return (
        compact_components(geometry_union),
        compact_components(semantic_union),
        geometry_edges,
        semantic_edges,
    )


def component_diagnostics(labels, candidate_mask, support):
    component_count = int(labels.max()) + 1 if labels.size else 0
    sizes = np.bincount(labels, minlength=component_count)
    candidate_support = support[candidate_mask]
    return {
        "num_superpoints": component_count,
        "component_size_quantiles": {
            str(q): float(np.quantile(sizes, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "singleton_fraction": float((sizes == 1).mean()),
        "candidate_support_quantiles": {
            str(q): float(np.quantile(candidate_support, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "candidate_support_above_0.25": float((candidate_support >= 0.25).mean()),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--base_artifact_dir", required=True)
    parser.add_argument("--candidate_mask", required=True)
    parser.add_argument("--neighbors", type=int, default=6)
    parser.add_argument("--spatial_radius_factor", type=float, default=1.5)
    parser.add_argument("--rgb_threshold", type=float, default=0.15)
    parser.add_argument("--log_scale_threshold", type=float, default=0.7)
    parser.add_argument("--semantic_threshold", type=float, default=0.85)
    parser.add_argument("--semantic_dim", type=int, default=64)
    parser.add_argument("--maximum_superpoint_size", type=int, default=512)
    parser.add_argument("--chunk_size", type=int, default=65536)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if (
        args.neighbors <= 1
        or args.semantic_dim <= 1
        or args.chunk_size <= 0
        or args.knn_workers <= 0
    ):
        raise ValueError("Neighbor, semantic, and chunk sizes must be positive")
    if args.maximum_superpoint_size <= 1:
        raise ValueError("Superpoints must permit at least two Gaussians")
    for name in (
        "spatial_radius_factor",
        "rgb_threshold",
        "log_scale_threshold",
    ):
        if getattr(args, name) <= 0.0:
            raise ValueError(f"--{name} must be positive")
    if not -1.0 <= args.semantic_threshold <= 1.0:
        raise ValueError("--semantic_threshold must be in [-1, 1]")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse Gaussian superpoint support: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    assignment = FixedSharedAssignment(args.base_artifact_dir)
    candidate_mask = np.load(os.path.abspath(args.candidate_mask)).astype(bool)
    if candidate_mask.shape != (assignment.num_gaussians,):
        raise ValueError("Candidate mask does not match the base artifact")
    valid_global = np.flatnonzero(assignment.valid_mask)
    candidate_valid = candidate_mask[valid_global]
    xyz, rgb, log_scale, checkpoint_iteration = load_geometry(
        args.geometry_checkpoint, assignment.num_gaussians
    )
    xyz = xyz[valid_global]
    rgb = rgb[valid_global]
    log_scale = log_scale[valid_global]

    codebook_path = os.path.join(
        assignment.artifact_dir, assignment.manifest["codebook_files"][0]
    )
    codebook = np.load(codebook_path).astype(np.float32)
    projected_codebook = project_codebook(
        codebook,
        args.semantic_dim,
        args.seed,
        "cuda" if args.faiss_gpu and torch.cuda.is_available() else "cpu",
    )
    semantics = reconstruct_projected_semantics(
        assignment,
        projected_codebook,
        valid_global,
        args.chunk_size,
    )
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    neighbors, distances, resources, knn_backend = build_knn(
        xyz,
        args.neighbors,
        args.chunk_size,
        args.faiss_gpu,
        args.knn_workers,
    )
    geometry_labels, semantic_labels, geometry_edges, semantic_edges = build_superpoints(
        neighbors,
        distances,
        rgb,
        log_scale,
        semantics,
        args.spatial_radius_factor,
        args.rgb_threshold,
        args.log_scale_threshold,
        args.semantic_threshold,
        args.maximum_superpoint_size,
        args.chunk_size,
    )
    del resources

    geometry_support, _, _ = leave_one_out_candidate_support(
        geometry_labels, candidate_valid
    )
    semantic_support, _, _ = leave_one_out_candidate_support(
        semantic_labels, candidate_valid
    )
    geometry_support_global = np.zeros(assignment.num_gaussians, dtype=np.float32)
    semantic_support_global = np.zeros_like(geometry_support_global)
    geometry_support_global[valid_global] = geometry_support
    semantic_support_global[valid_global] = semantic_support
    geometry_ids_global = np.full(assignment.num_gaussians, -1, dtype=np.int32)
    semantic_ids_global = np.full(assignment.num_gaussians, -1, dtype=np.int32)
    geometry_ids_global[valid_global] = geometry_labels
    semantic_ids_global[valid_global] = semantic_labels

    np.save(os.path.join(output_dir, "s0_geometry_rgb_support.npy"), geometry_support_global)
    np.save(os.path.join(output_dir, "s1_semantic_support.npy"), semantic_support_global)
    np.save(os.path.join(output_dir, "s0_superpoint_ids.npy"), geometry_ids_global)
    np.save(os.path.join(output_dir, "s1_superpoint_ids.npy"), semantic_ids_global)

    manifest = {
        "representation": "bounded_3d_gaussian_superpoint_candidate_support",
        "source": "training RGB geometry, A6 discrete semantics, and E8.3 capacity mask only",
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "base_artifact_dir": assignment.artifact_dir,
        "candidate_mask": os.path.abspath(args.candidate_mask),
        "num_gaussians": assignment.num_gaussians,
        "num_valid_gaussians": int(valid_global.size),
        "num_candidate_gaussians": int(candidate_valid.sum()),
        "geometry_edges": geometry_edges,
        "semantic_edges": semantic_edges,
        "knn_backend": knn_backend,
        "s0": component_diagnostics(
            geometry_labels, candidate_valid, geometry_support
        ),
        "s1": component_diagnostics(
            semantic_labels, candidate_valid, semantic_support
        ),
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
