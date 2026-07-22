#!/usr/bin/env python
"""Filter multiscale SAM masks with a projected 3D semantic scaffold.

The filter follows Splat Feature Solver's post-aggregation contract: cluster a
provisional lifted feature, render cluster labels to every training view, match
each SAM mask to its dominant projected cluster, and retain only masks whose
IoU exceeds a fixed threshold.  The retained L0--L3 masks are then aggregated
independently into split consensuses for fresh hierarchical codebook training.
"""

import json
import os
import random
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from build_gaussian_multilevel_codebook import faiss_kmeans
from gaussian_renderer import count_render
from prepare_semantic_field import (
    accumulate_consensus_chunk,
    apply_signed_segment_ownership,
    signed_segment_ownership,
)
from scene import GaussianModel, Scene
from semantic_field_utils import l2_normalize, load_geometry_checkpoint
from utils.general_utils import safe_state


LEVEL_NAMES = ("sam_l0", "sam_l1", "sam_l2", "sam_l3")


def set_deterministic_seed(seed):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def normalize_numpy(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def quantile_summary(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {str(q): 0.0 for q in (0.0, 0.25, 0.5, 0.75, 1.0)}
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.25, 0.5, 0.75, 1.0)
    }


def remap_hdbscan_labels(labels, noise_policy="pooled_background"):
    """Map HDBSCAN labels to dense projected labels under an explicit policy."""
    labels = np.asarray(labels, dtype=np.int64)
    if noise_policy not in ("pooled_background", "exclude"):
        raise ValueError("Unknown HDBSCAN noise policy")
    offset = 1 if noise_policy == "pooled_background" else 0
    fill = 0 if noise_policy == "pooled_background" else -1
    mapped = np.full(labels.shape, fill, dtype=np.int32)
    mapping = {}
    for projected_label, source_label in enumerate(
        np.unique(labels[labels >= 0]).tolist(), start=offset
    ):
        mapped[labels == source_label] = projected_label
        mapping[int(source_label)] = int(projected_label)
    return mapped, mapping


def scaffold_size_diagnostics(code_to_cluster, object_ids, num_clusters):
    code_labels = code_to_cluster[code_to_cluster >= 0]
    point_labels = code_to_cluster[object_ids]
    point_labels = point_labels[point_labels >= 0]
    code_sizes = np.bincount(code_labels, minlength=num_clusters)
    point_sizes = np.bincount(point_labels, minlength=num_clusters)
    return {
        "codebook_cluster_size_quantiles": quantile_summary(code_sizes),
        "gaussian_cluster_size_quantiles": quantile_summary(point_sizes),
        "empty_projected_clusters": int((point_sizes == 0).sum()),
    }


def allocate_stratified_cluster_budget(stratum_sizes, total_clusters):
    """Allocate a fixed cluster budget proportionally with one per stratum."""
    sizes = np.asarray(stratum_sizes, dtype=np.int64)
    if sizes.ndim != 1 or sizes.size == 0 or np.any(sizes <= 0):
        raise ValueError("Stratum sizes must be a non-empty positive vector")
    if total_clusters < sizes.size or total_clusters > sizes.sum():
        raise ValueError("Cluster budget must lie between strata and sample counts")
    allocations = np.ones(sizes.shape, dtype=np.int64)
    for _ in range(int(total_clusters - sizes.size)):
        eligible = allocations < sizes
        scores = np.where(eligible, sizes / allocations, -np.inf)
        allocations[int(np.argmax(scores))] += 1
    return allocations


def build_core_residual_scaffold(
    raw_labels,
    object_ids,
    object_codebook,
    total_clusters,
    iterations,
    seed,
    use_gpu,
):
    """Split HDBSCAN cores and noise with local spherical residual K-means."""
    strata = np.unique(raw_labels)
    sizes = np.asarray([(raw_labels == label).sum() for label in strata], dtype=np.int64)
    allocations = allocate_stratified_cluster_budget(sizes, int(total_clusters))
    code_to_cluster = np.full(raw_labels.shape, -1, dtype=np.int32)
    centers = []
    records = []
    cluster_offset = 0
    for index, (label, size, count) in enumerate(zip(strata, sizes, allocations)):
        members = np.flatnonzero(raw_labels == label)
        values = object_codebook[members]
        if int(count) == 1:
            local_centers = normalize_numpy(values.mean(axis=0, keepdims=True))
            local_ids = np.zeros(members.shape, dtype=np.int32)
        else:
            local_centers, local_index = faiss_kmeans(
                values,
                int(count),
                int(iterations),
                int(seed + index + 1),
                spherical=True,
                use_gpu=use_gpu,
            )
            local_centers = normalize_numpy(local_centers)
            local_ids = local_index.search(values).astype(np.int32, copy=False)
        code_to_cluster[members] = local_ids + cluster_offset
        centers.append(local_centers)
        records.append(
            {
                "hdbscan_label": int(label),
                "is_noise": bool(label < 0),
                "codebook_entries": int(size),
                "allocated_subclusters": int(count),
            }
        )
        cluster_offset += int(count)
    if np.any(code_to_cluster < 0) or cluster_offset != int(total_clusters):
        raise RuntimeError("Core-residual scaffold assignment is incomplete")
    scaffold = normalize_numpy(np.concatenate(centers, axis=0))
    return scaffold, code_to_cluster, code_to_cluster[object_ids], records


def load_object_slot(memory_dir):
    manifest_path = os.path.join(memory_dir, "manifest.json")
    with open(manifest_path) as handle:
        manifest = json.load(handle)
    if manifest.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError("The provisional memory must contain four independent codebooks")
    if len(manifest.get("level_codebooks", [])) != 4:
        raise ValueError("The provisional memory must provide exactly L0--L3")

    object_entry = next(
        entry for entry in manifest["level_codebooks"] if int(entry["level"]) == 1
    )
    start = int(object_entry["group_token_start"])
    end = int(object_entry["group_token_end"])
    point_ids = np.load(
        os.path.join(memory_dir, manifest["point_group_ids"]), mmap_mode="r"
    )
    if point_ids.shape != (int(manifest["num_gaussians"]), 4):
        raise ValueError("Provisional resident IDs do not match the memory manifest")
    object_ids = np.asarray(point_ids[:, 1], dtype=np.int64) - start
    if object_ids.min() < 0 or object_ids.max() >= end - start:
        raise ValueError("The L1 resident IDs are outside the declared object codebook")
    codebook = normalize_numpy(
        np.load(os.path.join(memory_dir, object_entry["codebook"]))
    )
    if codebook.shape[0] != end - start:
        raise ValueError("The L1 codebook size does not match its token range")
    return manifest, object_ids, codebook


def build_semantic_scaffold(
    object_ids,
    object_codebook,
    num_clusters,
    train_samples,
    iterations,
    seed,
    use_gpu,
):
    """Compress the object slot into broad labels used only for mask filtering."""
    if num_clusters <= 1:
        raise ValueError("The semantic scaffold needs at least two clusters")
    rng = np.random.default_rng(seed)
    count = min(int(train_samples), int(object_ids.size))
    sample_points = rng.choice(object_ids.size, size=count, replace=False)
    training = object_codebook[object_ids[sample_points]]
    scaffold, index = faiss_kmeans(
        training,
        min(int(num_clusters), int(training.shape[0])),
        int(iterations),
        int(seed),
        spherical=True,
        use_gpu=use_gpu,
    )
    scaffold = normalize_numpy(scaffold)
    code_to_cluster = index.search(object_codebook).astype(np.int32, copy=False)
    point_clusters = code_to_cluster[object_ids]
    diagnostics = {
        "method": "seeded_spherical_kmeans_over_provisional_object_tokens",
        "semantic_cluster_count": int(scaffold.shape[0]),
        "projected_cluster_count": int(scaffold.shape[0]),
        "codebook_noise_fraction": 0.0,
        "gaussian_noise_fraction": 0.0,
    }
    diagnostics.update(
        scaffold_size_diagnostics(code_to_cluster, object_ids, scaffold.shape[0])
    )
    return scaffold.astype(np.float16), code_to_cluster, point_clusters, diagnostics


def build_density_adaptive_scaffold(
    object_ids,
    object_codebook,
    pca_dim,
    min_cluster_size,
    min_samples,
    noise_policy,
    cluster_selection_method,
    residual_clusters,
    residual_iterations,
    seed,
    use_gpu,
):
    """Cluster unique object tokens with the SFS PCA+HDBSCAN contract."""
    try:
        import hdbscan
        from sklearn.decomposition import PCA
    except ImportError as exc:
        raise RuntimeError(
            "The hdbscan_pca scaffold needs isolated hdbscan and scikit-learn dependencies"
        ) from exc

    components = min(
        int(pca_dim), int(object_codebook.shape[0]), int(object_codebook.shape[1])
    )
    if components <= 0:
        raise ValueError("PCA component count must be positive")
    pca = PCA(
        n_components=components,
        random_state=0,
        svd_solver="full",
    )
    reduced = pca.fit_transform(object_codebook.astype(np.float32, copy=False))
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=int(min_cluster_size),
        min_samples=int(min_samples),
        metric="euclidean",
        cluster_selection_method=cluster_selection_method,
        core_dist_n_jobs=1,
    )
    raw_labels = clusterer.fit_predict(reduced).astype(np.int32, copy=False)
    residual_records = None
    if noise_policy == "residual_kmeans":
        scaffold, code_to_cluster, point_clusters, residual_records = (
            build_core_residual_scaffold(
                raw_labels,
                object_ids,
                object_codebook,
                residual_clusters,
                residual_iterations,
                seed,
                use_gpu,
            )
        )
        label_mapping = {
            int(label): int(index) for index, label in enumerate(np.unique(raw_labels))
        }
        num_clusters = int(scaffold.shape[0])
    else:
        code_to_cluster, label_mapping = remap_hdbscan_labels(raw_labels, noise_policy)
        num_clusters = len(label_mapping) + int(noise_policy == "pooled_background")
        if num_clusters <= 0:
            raise ValueError("HDBSCAN produced no non-noise semantic clusters")
        scaffold = np.zeros((num_clusters, object_codebook.shape[1]), dtype=np.float32)
        for label in range(num_clusters):
            members = code_to_cluster == label
            if members.any():
                scaffold[label] = object_codebook[members].mean(axis=0)
        scaffold = normalize_numpy(scaffold)
        point_clusters = code_to_cluster[object_ids]
    diagnostics = {
        "method": "sfs_pca_hdbscan_over_unique_object_tokens",
        "pca_components": int(components),
        "pca_explained_variance_ratio": float(
            np.asarray(pca.explained_variance_ratio_).sum()
        ),
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "noise_policy": noise_policy,
        "cluster_selection_method": cluster_selection_method,
        "fit_samples": int(object_codebook.shape[0]),
        "semantic_cluster_count": int(np.unique(raw_labels[raw_labels >= 0]).size),
        "projected_cluster_count": int(num_clusters),
        "noise_projected_label": 0 if noise_policy == "pooled_background" else None,
        "codebook_noise_fraction": float((raw_labels < 0).mean()),
        "gaussian_noise_fraction": float((raw_labels[object_ids] < 0).mean()),
        "gaussian_unassigned_fraction": float((point_clusters < 0).mean()),
        "cluster_persistence_quantiles": quantile_summary(
            getattr(clusterer, "cluster_persistence_", np.empty(0))
        ),
    }
    if residual_records is not None:
        diagnostics["residual_cluster_budget"] = int(residual_clusters)
        diagnostics["residual_kmeans_iterations"] = int(residual_iterations)
        diagnostics["residual_strata"] = residual_records
    diagnostics.update(
        scaffold_size_diagnostics(code_to_cluster, object_ids, num_clusters)
    )
    return scaffold.astype(np.float16), code_to_cluster, point_clusters, diagnostics


def project_cluster_labels(
    point_ids,
    point_weights,
    point_clusters,
    num_clusters,
    topk,
    chunk_size,
):
    """Render discrete labels by summing contributor mass per cluster."""
    if point_ids.shape != point_weights.shape or point_ids.ndim != 2:
        raise ValueError("Render IDs and contributions must have matching [P, K] shapes")
    output = torch.full(
        (point_ids.shape[0],), -1, dtype=torch.int32, device="cpu"
    )
    for start in range(0, point_ids.shape[0], chunk_size):
        end = min(start + chunk_size, point_ids.shape[0])
        weights, order = torch.topk(point_weights[start:end], k=topk, dim=1)
        ids = torch.gather(point_ids[start:end], 1, order).long()
        valid = (ids >= 0) & (weights > 0.0)
        safe_ids = ids.clamp(0, point_clusters.shape[0] - 1)
        clusters = point_clusters[safe_ids]
        valid = valid & (clusters >= 0)
        clusters = clusters.clamp_min(0)
        masses = torch.zeros(
            (end - start, num_clusters), dtype=torch.float32, device=ids.device
        )
        masses.scatter_add_(
            1,
            clusters.long(),
            torch.where(valid, weights.float(), torch.zeros_like(weights.float())),
        )
        projected = masses.argmax(dim=1).to(torch.int32)
        projected[masses.sum(dim=1) <= 0.0] = -1
        output[start:end].copy_(projected.cpu())
    return output.numpy()


def match_projected_masks(segmentation, projected_labels, iou_threshold):
    """Match every SAM segment to its modal projected cluster and filter by IoU."""
    segmentation = np.asarray(segmentation, dtype=np.int64)
    projected_labels = np.asarray(projected_labels, dtype=np.int64)
    if segmentation.shape != projected_labels.shape:
        raise ValueError("SAM and projected label images must have matching shapes")
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError("Mask IoU threshold must be in [0, 1]")

    flat_segments = segmentation.reshape(-1)
    flat_clusters = projected_labels.reshape(-1)
    segment_ids, segment_areas = np.unique(
        flat_segments[flat_segments >= 0], return_counts=True
    )
    cluster_ids, cluster_areas = np.unique(
        flat_clusters[flat_clusters >= 0], return_counts=True
    )
    cluster_area = dict(zip(cluster_ids.tolist(), cluster_areas.tolist()))
    pair_valid = (flat_segments >= 0) & (flat_clusters >= 0)
    pair_segments = flat_segments[pair_valid]
    pair_clusters = flat_clusters[pair_valid]

    dominant = {}
    if pair_segments.size:
        pairs = np.stack((pair_segments, pair_clusters), axis=1)
        unique_pairs, intersections = np.unique(pairs, axis=0, return_counts=True)
        for (segment, cluster), intersection in zip(unique_pairs, intersections):
            current = dominant.get(int(segment))
            candidate = (int(intersection), int(cluster))
            if current is None or candidate > current:
                dominant[int(segment)] = candidate

    trusted = set()
    records = []
    for segment, area in zip(segment_ids.tolist(), segment_areas.tolist()):
        intersection, cluster = dominant.get(int(segment), (0, -1))
        union = int(area) + int(cluster_area.get(cluster, 0)) - intersection
        iou = float(intersection / union) if union > 0 else 0.0
        keep = iou > iou_threshold
        if keep:
            trusted.add(int(segment))
        records.append(
            {
                "segment_id": int(segment),
                "cluster_id": int(cluster),
                "intersection": int(intersection),
                "segment_area": int(area),
                "cluster_area": int(cluster_area.get(cluster, 0)),
                "iou": iou,
                "kept": keep,
            }
        )

    if trusted:
        keep_pixels = np.isin(flat_segments, np.fromiter(trusted, dtype=np.int64))
    else:
        keep_pixels = np.zeros(flat_segments.shape, dtype=bool)
    filtered = np.where(keep_pixels, flat_segments, -1).reshape(segmentation.shape)
    return filtered.astype(np.int32), records


def aggregate_filtered_level(
    flat_ids,
    flat_weights,
    filtered_segments,
    feature_latents,
    split_sums,
    split_weights,
    split_index,
    num_gaussians,
    topk,
    chunk_size,
):
    sampled_flat = np.flatnonzero(filtered_segments.reshape(-1) >= 0)
    if not sampled_flat.size:
        return {"pixels": 0, "input_mass": 0.0, "retained_mass": 0.0}
    sampled_indices = torch.from_numpy(sampled_flat).long().cuda()
    sampled_weights = flat_weights[sampled_indices]
    top_weights, order = torch.topk(sampled_weights, k=topk, dim=1)
    top_ids = torch.gather(flat_ids[sampled_indices], 1, order).long()
    valid = top_ids >= 0
    top_weights = torch.where(
        valid,
        top_weights.float().clamp_min(0.0),
        torch.zeros_like(top_weights.float()),
    )
    segment_ids = torch.from_numpy(
        filtered_segments.reshape(-1)[sampled_flat].astype(np.int64, copy=False)
    ).cuda()
    dominant_segment, confidence, _, total_mass = signed_segment_ownership(
        top_ids, top_weights, segment_ids, num_gaussians
    )
    input_mass = float(top_weights.sum().item())
    retained_mass = 0.0
    for start in range(0, sampled_flat.size, chunk_size):
        end = min(start + chunk_size, sampled_flat.size)
        ids = top_ids[start:end]
        weights = apply_signed_segment_ownership(
            ids,
            top_weights[start:end],
            segment_ids[start:end],
            dominant_segment,
            confidence,
        )
        retained_mass += float(weights.sum().item())
        valid_pixels = weights.sum(dim=1) > 1e-8
        if not valid_pixels.any():
            continue
        accumulate_consensus_chunk(
            split_sums[split_index],
            split_weights[split_index],
            ids[valid_pixels],
            weights[valid_pixels],
            segment_ids[start:end][valid_pixels],
            feature_latents,
        )
    del sampled_indices, sampled_weights, top_weights, top_ids
    del segment_ids, dominant_segment, confidence, total_mass
    return {
        "pixels": int(sampled_flat.size),
        "input_mass": input_mass,
        "retained_mass": retained_mass,
    }


def finalize_consensus(split_sums, split_weights, output_path):
    num_splits, num_gaussians, semantic_dim = split_sums.shape
    initial = torch.empty(
        (num_gaussians, semantic_dim), dtype=torch.float16, device="cpu"
    )
    split_initial = torch.empty(
        (num_splits, num_gaussians, semantic_dim),
        dtype=torch.float16,
        device="cpu",
    )
    total_weights = split_weights.sum(dim=0)
    for start in range(0, num_gaussians, 8192):
        end = min(start + 8192, num_gaussians)
        weights = total_weights[start:end]
        features = l2_normalize(
            split_sums[:, start:end].sum(dim=0)
            / weights.clamp_min(1e-8).unsqueeze(-1)
        )
        features[weights <= 0.0] = 0.0
        initial[start:end].copy_(features.to(torch.float16).cpu())
        for split in range(num_splits):
            split_weight = split_weights[split, start:end]
            split_feature = l2_normalize(
                split_sums[split, start:end]
                / split_weight.clamp_min(1e-8).unsqueeze(-1)
            )
            split_feature[split_weight <= 0.0] = 0.0
            split_initial[split, start:end].copy_(
                split_feature.to(torch.float16).cpu()
            )
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "total_weights": total_weights.detach().cpu(),
            "initial_features": initial,
            "split_initial_features": split_initial,
            "split_weights": split_weights.detach().cpu(),
        },
        output_path,
    )


def main():
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--provisional_memory_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--topk", type=int, default=45)
    parser.add_argument("--consensus_chunk_pixels", type=int, default=1024)
    parser.add_argument("--projection_chunk_pixels", type=int, default=8192)
    parser.add_argument("--scaffold_clusters", type=int, default=256)
    parser.add_argument("--scaffold_train_samples", type=int, default=200000)
    parser.add_argument("--scaffold_iterations", type=int, default=25)
    parser.add_argument(
        "--scaffold_method",
        choices=("spherical_kmeans", "hdbscan_pca"),
        default="spherical_kmeans",
    )
    parser.add_argument("--scaffold_pca_dim", type=int, default=50)
    parser.add_argument("--hdbscan_min_cluster_size", type=int, default=500)
    parser.add_argument("--hdbscan_min_samples", type=int, default=10)
    parser.add_argument(
        "--hdbscan_noise_policy",
        choices=("pooled_background", "exclude", "residual_kmeans"),
        default="pooled_background",
    )
    parser.add_argument(
        "--hdbscan_cluster_selection_method",
        choices=("eom", "leaf"),
        default="eom",
    )
    parser.add_argument("--mask_iou_threshold", type=float, default=0.8)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument(
        "--levels",
        nargs="+",
        type=int,
        default=[0, 1, 2, 3],
        help="SAM levels to filter and aggregate; defaults to all L0--L3.",
    )
    parser.add_argument(
        "--diagnostic_only",
        action="store_true",
        help="Measure projected-mask retention without allocating or saving consensuses.",
    )
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.topk <= 0 or args.topk > 100:
        raise ValueError("--topk must be in [1, 100]")
    if args.consensus_chunk_pixels <= 0 or args.projection_chunk_pixels <= 0:
        raise ValueError("Chunk sizes must be positive")
    if args.max_views < 0:
        raise ValueError("--max_views must be non-negative")
    if not args.levels or len(set(args.levels)) != len(args.levels):
        raise ValueError("--levels must contain unique L0--L3 indices")
    if any(level < 0 or level >= len(LEVEL_NAMES) for level in args.levels):
        raise ValueError("--levels entries must lie in [0, 3]")
    if args.scaffold_pca_dim <= 0:
        raise ValueError("--scaffold_pca_dim must be positive")
    if args.hdbscan_min_cluster_size <= 1 or args.hdbscan_min_samples <= 0:
        raise ValueError("HDBSCAN cluster sizes must be positive")
    active_levels = set(args.levels)

    output_dir = os.path.abspath(args.output_dir)
    complete_name = "DIAGNOSTIC_COMPLETE" if args.diagnostic_only else "FILTER_COMPLETE"
    complete_path = os.path.join(output_dir, complete_name)
    if os.path.isfile(complete_path) and not args.force:
        print(f"Reuse post-aggregation filtered consensuses: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    safe_state(args.quiet)
    set_deterministic_seed(args.seed)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    os.makedirs(os.path.abspath(dataset.model_path), exist_ok=True)

    provisional_dir = os.path.abspath(args.provisional_memory_dir)
    provisional_manifest, object_ids, object_codebook = load_object_slot(
        provisional_dir
    )
    if args.scaffold_method == "hdbscan_pca":
        scaffold, code_to_cluster, point_clusters_np, scaffold_diagnostics = (
            build_density_adaptive_scaffold(
                object_ids,
                object_codebook,
                args.scaffold_pca_dim,
                args.hdbscan_min_cluster_size,
                args.hdbscan_min_samples,
                args.hdbscan_noise_policy,
                args.hdbscan_cluster_selection_method,
                args.scaffold_clusters,
                args.scaffold_iterations,
                args.seed,
                args.faiss_gpu,
            )
        )
    else:
        scaffold, code_to_cluster, point_clusters_np, scaffold_diagnostics = (
            build_semantic_scaffold(
                object_ids,
                object_codebook,
                args.scaffold_clusters,
                args.scaffold_train_samples,
                args.scaffold_iterations,
                args.seed,
                args.faiss_gpu,
            )
        )
    np.save(os.path.join(output_dir, "scaffold_codebook.npy"), scaffold)
    np.save(os.path.join(output_dir, "l1_code_to_scaffold.npy"), code_to_cluster)
    np.save(os.path.join(output_dir, "point_scaffold_ids.npy"), point_clusters_np)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(
        scene.gaussians, args.geometry_checkpoint
    )
    cameras = scene.getTrainCameras() + scene.getTestCameras()
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    if not cameras:
        raise ValueError("No cameras are available for post-aggregation filtering")
    num_gaussians = int(scene.gaussians.get_xyz.shape[0])
    if num_gaussians != point_clusters_np.shape[0]:
        raise ValueError("Geometry and provisional memory have different point counts")
    point_clusters = torch.from_numpy(point_clusters_np).long().cuda()

    split_sums = [None for _ in LEVEL_NAMES]
    split_weights = [None for _ in LEVEL_NAMES]
    if not args.diagnostic_only:
        for level in args.levels:
            split_sums[level] = torch.zeros(
                (2, num_gaussians, 512), dtype=torch.float32, device="cuda"
            )
            split_weights[level] = torch.zeros(
                (2, num_gaussians), dtype=torch.float32, device="cuda"
            )
    diagnostics = [
        {
            "input_masks": 0,
            "kept_masks": 0,
            "input_pixels": 0,
            "kept_pixels": 0,
            "input_mass": 0.0,
            "retained_mass": 0.0,
            "mask_ious": [],
        }
        for _ in LEVEL_NAMES
    ]
    view_records = []
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    feature_dir = os.path.abspath(args.feature_dir)

    for view_index, camera in enumerate(tqdm(cameras, desc="A40 mask filtering")):
        feature_stem = os.path.join(feature_dir, camera.image_name)
        feature_path = feature_stem + "_f.npy"
        segmentation_path = feature_stem + "_s.npy"
        if not os.path.isfile(feature_path) or not os.path.isfile(segmentation_path):
            raise ValueError(f"Missing multiscale features for {camera.image_name}")
        segmentations = np.load(segmentation_path, mmap_mode="r")
        if segmentations.shape[0] != 4:
            raise ValueError("A40 requires exactly four multiscale SAM levels")
        if tuple(segmentations.shape[1:]) != (
            camera.image_height,
            camera.image_width,
        ):
            raise ValueError(f"Mask and camera sizes differ for {camera.image_name}")
        if args.diagnostic_only:
            features = None
            feature_latents = None
        else:
            features = torch.from_numpy(
                np.load(feature_path).astype(np.float32, copy=False)
            ).cuda()
            feature_latents = F.normalize(features, dim=-1)

        render = count_render(camera, scene.gaussians, pipe, background)
        flat_ids = render["per_pixel_gaussian_ids"].reshape(-1, 100).long()
        flat_weights = render["per_pixel_gaussian_contributions"].reshape(-1, 100)
        projected = project_cluster_labels(
            flat_ids,
            flat_weights,
            point_clusters,
            scaffold.shape[0],
            args.topk,
            args.projection_chunk_pixels,
        ).reshape(camera.image_height, camera.image_width)
        view_record = {"view_index": view_index, "image_name": camera.image_name, "levels": []}

        for level in args.levels:
            name = LEVEL_NAMES[level]
            segmentation = np.asarray(segmentations[level])
            filtered, records = match_projected_masks(
                segmentation, projected, args.mask_iou_threshold
            )
            if args.diagnostic_only:
                stats = {"input_mass": 0.0, "retained_mass": 0.0}
            else:
                stats = aggregate_filtered_level(
                    flat_ids,
                    flat_weights,
                    filtered,
                    feature_latents,
                    split_sums[level],
                    split_weights[level],
                    view_index % 2,
                    num_gaussians,
                    args.topk,
                    args.consensus_chunk_pixels,
                )
            input_pixels = int((segmentation >= 0).sum())
            kept_pixels = int((filtered >= 0).sum())
            kept_masks = sum(record["kept"] for record in records)
            diagnostics[level]["input_masks"] += len(records)
            diagnostics[level]["kept_masks"] += kept_masks
            diagnostics[level]["input_pixels"] += input_pixels
            diagnostics[level]["kept_pixels"] += kept_pixels
            diagnostics[level]["input_mass"] += stats["input_mass"]
            diagnostics[level]["retained_mass"] += stats["retained_mass"]
            diagnostics[level]["mask_ious"].extend(record["iou"] for record in records)
            view_record["levels"].append(
                {
                    "name": name,
                    "input_masks": len(records),
                    "kept_masks": kept_masks,
                    "input_pixels": input_pixels,
                    "kept_pixels": kept_pixels,
                    "mean_mask_iou": float(np.mean([r["iou"] for r in records]))
                    if records
                    else 0.0,
                }
            )
        view_records.append(view_record)
        del render, flat_ids, flat_weights
        if features is not None:
            del features, feature_latents
        torch.cuda.empty_cache()

    level_manifests = []
    for level, name in enumerate(LEVEL_NAMES):
        if level not in active_levels:
            continue
        level_dir = os.path.join(output_dir, f"{name}_split2")
        if not args.diagnostic_only:
            consensus_path = os.path.join(level_dir, "consensus.pt")
            finalize_consensus(split_sums[level], split_weights[level], consensus_path)
        values = np.asarray(diagnostics[level].pop("mask_ious"), dtype=np.float32)
        diagnostics[level]["mask_iou_quantiles"] = {
            str(q): float(np.quantile(values, q)) if values.size else 0.0
            for q in (0.0, 0.25, 0.5, 0.75, 1.0)
        }
        diagnostics[level]["kept_mask_fraction"] = diagnostics[level][
            "kept_masks"
        ] / max(1, diagnostics[level]["input_masks"])
        diagnostics[level]["kept_pixel_fraction"] = diagnostics[level][
            "kept_pixels"
        ] / max(1, diagnostics[level]["input_pixels"])
        level_manifest = {
            "format_version": 1,
            "method": "sfs_post_aggregation_filtered_multiscale_consensus",
            "feature_level": level,
            "semantic_dim": 512,
            "num_gaussians": num_gaussians,
            "consensus_splits": 2,
            "filter_diagnostics": diagnostics[level],
        }
        if not args.diagnostic_only:
            level_manifest["consensus"] = "consensus.pt"
        os.makedirs(level_dir, exist_ok=True)
        with open(os.path.join(level_dir, "manifest.json"), "w") as handle:
            json.dump(level_manifest, handle, indent=2)
        level_manifests.append(level_manifest)
        if not args.diagnostic_only:
            split_sums[level] = None
            split_weights[level] = None
            torch.cuda.empty_cache()

    manifest = {
        "format_version": 1,
        "method": "a40_post_aggregation_filtered_hierarchical_memory",
        "seed": int(args.seed),
        "source_path": os.path.abspath(dataset.source_path),
        "feature_dir": feature_dir,
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": int(checkpoint_iteration),
        "provisional_memory_dir": provisional_dir,
        "provisional_memory_seed": provisional_manifest.get("seed"),
        "num_gaussians": num_gaussians,
        "num_views": len(cameras),
        "diagnostic_only": bool(args.diagnostic_only),
        "active_levels": list(args.levels),
        "topk": int(args.topk),
        "mask_iou_threshold": float(args.mask_iou_threshold),
        "mask_keep_rule": "dominant projected 3D scaffold cluster IoU > threshold",
        "scaffold": {
            "source_level": "sam_l1_object",
            "method": scaffold_diagnostics["method"],
            "num_clusters": int(scaffold.shape[0]),
            "train_samples": int(
                scaffold_diagnostics.get(
                    "fit_samples", min(args.scaffold_train_samples, object_ids.size)
                )
            ),
            "iterations": int(args.scaffold_iterations),
            "codebook": "scaffold_codebook.npy",
            "l1_code_assignment": "l1_code_to_scaffold.npy",
            "point_assignment": "point_scaffold_ids.npy",
            "diagnostics": scaffold_diagnostics,
        },
        "levels": level_manifests,
        "views": view_records,
        "leakage_control": "geometry, provisional training-view semantics, training cameras, and SAM masks only; no text query or evaluation labels",
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    with open(complete_path, "w") as handle:
        handle.write("complete\n")
    print(json.dumps({"output_dir": output_dir, "levels": diagnostics}, indent=2))


if __name__ == "__main__":
    main()
