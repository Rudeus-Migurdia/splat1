#!/usr/bin/env python
"""Filter candidate semantics by local 3D and semantic neighborhood support."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

from build_gaussian_multilevel_codebook import ConsensusFeatureSource


def semantic_neighbor_support(
    query_features,
    neighbor_features,
    squared_distances,
    neighbor_candidate_mask,
    semantic_floor,
):
    similarity = np.sum(query_features[:, None, :] * neighbor_features, axis=-1)
    semantic_valid = similarity >= semantic_floor
    distance_scale = np.maximum(squared_distances[:, -1:], 1e-12)
    spatial_weight = np.exp(-squared_distances / distance_scale)
    weights = spatial_weight * semantic_valid
    denominator = weights.sum(axis=1)
    support = (
        weights * np.asarray(neighbor_candidate_mask, dtype=np.float32)
    ).sum(axis=1) / np.maximum(denominator, 1e-8)
    return support.astype(np.float32), semantic_valid.sum(axis=1)


def semantic_neighbor_support_cuda(
    query_features,
    neighbor_features,
    squared_distances,
    neighbor_candidate_mask,
    semantic_floor,
):
    query = torch.from_numpy(query_features).float().cuda(non_blocking=True)
    neighbors = torch.from_numpy(neighbor_features).float().cuda(non_blocking=True)
    distances = torch.from_numpy(squared_distances).float().cuda(non_blocking=True)
    candidate = torch.from_numpy(
        np.asarray(neighbor_candidate_mask, dtype=np.float32)
    ).cuda(non_blocking=True)
    similarity = (query[:, None, :] * neighbors).sum(dim=-1)
    semantic_valid = similarity >= semantic_floor
    distance_scale = distances[:, -1:].clamp_min(1e-12)
    weights = torch.exp(-distances / distance_scale) * semantic_valid
    support = (weights * candidate).sum(dim=1) / weights.sum(dim=1).clamp_min(1e-8)
    return support.cpu().numpy(), semantic_valid.sum(dim=1).cpu().numpy()


def load_gaussian_xyz(checkpoint_path, expected_count):
    model_params, _ = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) not in (12, 13):
        raise ValueError("Unsupported geometry checkpoint tuple")
    xyz = model_params[1].detach().float().numpy()
    if xyz.shape != (expected_count, 3):
        raise ValueError("Geometry coordinates do not match semantic Gaussians")
    return np.ascontiguousarray(xyz, dtype=np.float32)


def make_knn_index(points, use_gpu):
    try:
        import faiss
    except ImportError as exc:
        raise RuntimeError("Spatial semantic support requires FAISS") from exc
    index = faiss.IndexFlatL2(3)
    index.add(np.ascontiguousarray(points, dtype=np.float32))
    resources = None
    if use_gpu and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    return index, resources


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--candidate_mask", required=True)
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--semantic_floor", type=float, default=0.9)
    parser.add_argument("--minimum_support", type=float, default=0.25)
    parser.add_argument("--minimum_semantic_neighbors", type=int, default=4)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--output", required=True)
    args = parser.parse_args(sys.argv[1:])
    if args.neighbors <= 1 or args.chunk_size <= 0:
        raise ValueError("Neighbor and chunk counts must be positive")
    if not -1.0 <= args.semantic_floor <= 1.0:
        raise ValueError("--semantic_floor must be in [-1, 1]")
    if not 0.0 <= args.minimum_support <= 1.0:
        raise ValueError("--minimum_support must be in [0, 1]")
    if not 1 <= args.minimum_semantic_neighbors <= args.neighbors:
        raise ValueError("Semantic neighbor count must be within the kNN size")

    output = os.path.abspath(args.output)
    diagnostics_output = os.path.splitext(output)[0] + ".json"
    support_output = os.path.splitext(output)[0] + "_scores.npy"
    if all(os.path.isfile(path) for path in (output, diagnostics_output, support_output)):
        print(f"Reuse spatial semantic support: {output}")
        return

    source = ConsensusFeatureSource(args.candidate_consensus)
    candidate_mask = np.load(os.path.abspath(args.candidate_mask)).astype(bool)
    if candidate_mask.shape != (source.num_items,):
        raise ValueError("Candidate mask does not match the consensus")
    candidate_mask &= np.asarray(source.valid_mask, dtype=bool)
    candidate_points = np.flatnonzero(candidate_mask)
    semantic_points = np.flatnonzero(source.valid_mask)
    if candidate_points.size == 0 or semantic_points.size <= args.neighbors:
        raise ValueError("Insufficient candidate or semantic points")

    xyz = load_gaussian_xyz(args.geometry_checkpoint, source.num_items)
    index, resources = make_knn_index(xyz[semantic_points], args.faiss_gpu)
    keep = np.zeros(source.num_items, dtype=bool)
    support_map = np.zeros(source.num_items, dtype=np.float32)
    support_values = []
    neighbor_counts = []
    for start in range(0, candidate_points.size, args.chunk_size):
        points = candidate_points[start : start + args.chunk_size]
        distances, local_neighbors = index.search(
            np.ascontiguousarray(xyz[points]), args.neighbors + 1
        )
        neighbors = semantic_points[local_neighbors]
        distances = np.asarray(distances, dtype=np.float32)
        is_self = neighbors == points[:, None]
        distances[is_self] = np.inf
        order = np.argsort(distances, axis=1)[:, : args.neighbors]
        distances = np.take_along_axis(distances, order, axis=1)
        neighbors = np.take_along_axis(neighbors, order, axis=1)

        query_features = source.read(points)
        flat_neighbors = neighbors.reshape(-1)
        neighbor_features = source.read(flat_neighbors).reshape(
            points.size, args.neighbors, source.feature_dim
        )
        support_function = (
            semantic_neighbor_support_cuda
            if args.faiss_gpu and torch.cuda.is_available()
            else semantic_neighbor_support
        )
        support, semantic_count = support_function(
            query_features,
            neighbor_features,
            distances,
            candidate_mask[neighbors],
            args.semantic_floor,
        )
        support_map[points] = support
        selected = (
            (support >= args.minimum_support)
            & (semantic_count >= args.minimum_semantic_neighbors)
        )
        keep[points[selected]] = True
        support_values.append(support)
        neighbor_counts.append(semantic_count)

    support_values = np.concatenate(support_values)
    neighbor_counts = np.concatenate(neighbor_counts)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, keep)
    np.save(support_output, support_map)
    diagnostics = {
        "representation": "local_3d_semantic_support_gate",
        "source": "training geometry and continuous self-trained semantics only",
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "candidate_consensus": os.path.abspath(args.candidate_consensus),
        "candidate_mask": os.path.abspath(args.candidate_mask),
        "support_scores": support_output,
        "num_gaussians": source.num_items,
        "num_input_candidates": int(candidate_points.size),
        "num_kept_candidates": int(keep.sum()),
        "kept_fraction_of_candidates": float(keep.sum() / candidate_points.size),
        "kept_fraction_of_all_gaussians": float(keep.mean()),
        "support_quantiles": {
            str(q): float(np.quantile(support_values, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
        "semantic_neighbor_count_quantiles": {
            str(q): float(np.quantile(neighbor_counts, q))
            for q in (0.0, 0.1, 0.5, 0.9, 1.0)
        },
        "args": vars(args),
    }
    with open(diagnostics_output, "w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
