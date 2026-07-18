#!/usr/bin/env python
"""Build a small L3 codebook and a training-only sparse score residual."""

import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_gaussian_multilevel_codebook import faiss_kmeans


def normalize_numpy(features, eps=1e-8):
    features = np.asarray(features, dtype=np.float32)
    return features / np.maximum(np.linalg.norm(features, axis=1, keepdims=True), eps)


def load_consensus(path):
    payload = torch.load(os.path.abspath(path), map_location="cpu")
    required = {
        "initial_features",
        "total_weights",
        "split_initial_features",
        "split_weights",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Consensus is missing fields: {sorted(missing)}")
    output = {name: payload[name].detach().cpu().contiguous() for name in required}
    count, feature_dim = output["initial_features"].shape
    if output["split_initial_features"].shape != (2, count, feature_dim):
        raise ValueError("Split feature table does not match the consensus")
    if output["split_weights"].shape != (2, count):
        raise ValueError("Split weights do not match split features")
    if output["total_weights"].shape != (count,):
        raise ValueError("Total weights do not match consensus features")
    return output


def split_statistics(split_features, split_weights, stability_floor):
    supported = (split_weights[0] > 0) & (split_weights[1] > 0)
    cosine = F.cosine_similarity(
        split_features[0].float(), split_features[1].float(), dim=-1
    )
    stability = ((cosine - stability_floor) / (1.0 - stability_floor)).clamp(0.0, 1.0)
    balance = (
        2.0 * split_weights.min(dim=0).values
        / split_weights.sum(dim=0).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    reliability = stability * balance.sqrt()
    reliability = torch.where(supported, reliability, torch.zeros_like(reliability))
    return reliability, supported, cosine


def deterministic_top_fraction(eligible, scores, maximum_count):
    point_ids = np.flatnonzero(eligible)
    if maximum_count <= 0 or point_ids.size == 0:
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((point_ids, -scores[point_ids]))
    return point_ids[order[: min(int(maximum_count), point_ids.size)]].astype(np.int64)


def residual_alphas(selected_scores, alpha_max):
    selected_scores = np.asarray(selected_scores, dtype=np.float32)
    if selected_scores.size == 0:
        return selected_scores
    scale = max(float(np.quantile(selected_scores, 0.95)), 1e-8)
    normalized = np.clip(selected_scores / scale, 0.0, 1.0)
    return (float(alpha_max) * (0.25 + 0.75 * normalized)).astype(np.float32)


def compute_selection(base, l3, part_ids, interior_support, args):
    count = base["initial_features"].shape[0]
    l3_reliability = np.zeros(count, dtype=np.float32)
    base_reliability = np.zeros(count, dtype=np.float32)
    l3_split_cosine = np.zeros(count, dtype=np.float32)
    residual_disagreement = np.zeros(count, dtype=np.float32)
    l3_split_supported = np.zeros(count, dtype=np.bool_)

    for start in range(0, count, args.chunk_size):
        end = min(start + args.chunk_size, count)
        base_rel, _, _ = split_statistics(
            base["split_initial_features"][:, start:end],
            base["split_weights"][:, start:end],
            args.stability_floor,
        )
        l3_rel, l3_supported, l3_cosine = split_statistics(
            l3["split_initial_features"][:, start:end],
            l3["split_weights"][:, start:end],
            args.stability_floor,
        )
        base_feature = F.normalize(base["initial_features"][start:end].float(), dim=-1)
        l3_feature = F.normalize(l3["initial_features"][start:end].float(), dim=-1)
        residual = 1.0 - F.cosine_similarity(base_feature, l3_feature, dim=-1)
        base_reliability[start:end] = base_rel.numpy()
        l3_reliability[start:end] = l3_rel.numpy()
        l3_split_cosine[start:end] = l3_cosine.numpy()
        l3_split_supported[start:end] = l3_supported.numpy()
        residual_disagreement[start:end] = residual.clamp(0.0, 2.0).numpy()

    boundary = np.clip(1.0 - interior_support.astype(np.float32), 0.0, 1.0)
    l3_valid = l3["total_weights"].numpy() > 0
    part_valid = part_ids >= 0
    relative_reliability = l3_reliability + args.relative_reliability_slack >= base_reliability
    eligible = (
        l3_valid
        & part_valid
        & l3_split_supported
        & (boundary >= args.minimum_boundary)
        & (l3_split_cosine >= args.minimum_split_cosine)
        & (l3_reliability >= args.minimum_l3_reliability)
        & relative_reliability
        & (residual_disagreement >= args.minimum_residual)
        & (residual_disagreement <= args.maximum_residual)
    )
    residual_factor = np.clip(
        (residual_disagreement - args.minimum_residual)
        / max(args.maximum_residual - args.minimum_residual, 1e-8),
        0.0,
        1.0,
    )
    relative_factor = np.clip(
        (l3_reliability - base_reliability + args.relative_reliability_slack)
        / max(2.0 * args.relative_reliability_slack, 1e-8),
        0.0,
        1.0,
    )
    scores = (
        boundary
        * l3_reliability
        * residual_factor
        * (0.5 + 0.5 * relative_factor)
    ).astype(np.float32)
    scores[~eligible] = 0.0
    maximum_count = int(np.ceil(count * args.maximum_sparse_fraction))
    selected = deterministic_top_fraction(eligible, scores, maximum_count)
    alpha = residual_alphas(scores[selected], args.alpha_max)
    arrays = {
        "boundary": boundary,
        "base_reliability": base_reliability,
        "l3_reliability": l3_reliability,
        "l3_split_cosine": l3_split_cosine,
        "residual_disagreement": residual_disagreement,
        "scores": scores,
        "eligible": eligible,
        "selected": selected,
        "alpha": alpha,
        "l3_valid": l3_valid,
    }
    return arrays


def feature_rows(payload, point_ids):
    return normalize_numpy(payload["initial_features"][point_ids].float().numpy())


def train_and_assign_codebook(l3, args):
    raw_valid = l3["total_weights"].numpy() > 0
    feature_norm = l3["initial_features"].float().norm(dim=1).numpy()
    valid_ids = np.flatnonzero(raw_valid & (feature_norm > 0)).astype(np.int64)
    if valid_ids.size < args.num_codes:
        raise ValueError(
            f"Only {valid_ids.size} valid L3 points are available for {args.num_codes} codes"
        )
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.train_samples, valid_ids.size)
    sample_ids = rng.choice(valid_ids, sample_count, replace=False)
    sample = feature_rows(l3, sample_ids)
    codebook, index = faiss_kmeans(
        sample,
        args.num_codes,
        args.iterations,
        args.seed,
        True,
        args.faiss_gpu,
    )
    codebook = normalize_numpy(codebook)
    assignments = np.full(l3["initial_features"].shape[0], -1, dtype=np.int32)
    cosine_sum = 0.0
    cosine_square_sum = 0.0
    cosine_min = 1.0
    evaluated = 0
    for start in range(0, valid_ids.size, args.assignment_chunk):
        point_ids = valid_ids[start : start + args.assignment_chunk]
        features = feature_rows(l3, point_ids)
        code_ids = index.search(features).astype(np.int32)
        assignments[point_ids] = code_ids
        decoded = codebook[code_ids]
        cosine = np.sum(features * decoded, axis=1)
        cosine_sum += float(cosine.sum())
        cosine_square_sum += float(np.square(cosine).sum())
        cosine_min = min(cosine_min, float(cosine.min()))
        evaluated += cosine.size
    mean = cosine_sum / max(1, evaluated)
    variance = cosine_square_sum / max(1, evaluated) - mean * mean
    diagnostics = {
        "num_valid": int(valid_ids.size),
        "train_samples": int(sample_count),
        "mean_reconstruction_cosine": mean,
        "std_reconstruction_cosine": float(max(0.0, variance) ** 0.5),
        "min_reconstruction_cosine": cosine_min,
    }
    return codebook, assignments, valid_ids, diagnostics


def encode_alpha(alpha):
    return np.rint(np.clip(alpha, 0.0, 1.0) * 255.0).astype(np.uint8)


def write_quantized_hypothesis(
    output_dir,
    root_dir,
    point_ids,
    code_ids,
    alpha,
    codebook,
    num_gaussians,
    method,
    source,
):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    point_ids = np.asarray(point_ids, dtype=np.int64)
    code_ids = np.asarray(code_ids, dtype=np.int64)
    alpha = np.asarray(alpha, dtype=np.float32)
    if point_ids.shape != code_ids.shape or point_ids.shape != alpha.shape:
        raise ValueError("Point IDs, code IDs, and alpha must match")
    if point_ids.size and (
        point_ids.min() < 0
        or point_ids.max() >= num_gaussians
        or np.unique(point_ids).size != point_ids.size
    ):
        raise ValueError("Sparse hypothesis point IDs are invalid or duplicated")
    if code_ids.size and (code_ids.min() < 0 or code_ids.max() >= codebook.shape[0]):
        raise ValueError("Sparse hypothesis code IDs are invalid")
    if codebook.shape[0] <= np.iinfo(np.uint16).max:
        packed_codes = code_ids.astype(np.uint16)
    else:
        packed_codes = code_ids.astype(np.uint32)
    packed_points = point_ids.astype(np.uint32)
    packed_alpha = encode_alpha(alpha)
    np.save(os.path.join(output_dir, "point_ids.npy"), packed_points)
    np.save(os.path.join(output_dir, "code_ids.npy"), packed_codes)
    np.save(os.path.join(output_dir, "reliability.npy"), packed_alpha)
    codebook_path = os.path.join(root_dir, "l3_codebook.npy")
    relative_codebook = os.path.relpath(codebook_path, output_dir)
    semantic_bytes = int(
        codebook.astype(np.float16).nbytes
        + packed_points.nbytes
        + packed_codes.nbytes
        + packed_alpha.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "sparse_quantized_semantic_hypothesis",
        "method": method,
        "num_gaussians": int(num_gaussians),
        "num_hypotheses": int(point_ids.size),
        "feature_dim": int(codebook.shape[1]),
        "num_codes": int(codebook.shape[0]),
        "point_ids": "point_ids.npy",
        "code_ids": "code_ids.npy",
        "codebook": relative_codebook,
        "reliability": "reliability.npy",
        "reliability_semantics": "query_positive_score_residual_alpha",
        "mean_alpha": float(alpha.mean()) if alpha.size else 0.0,
        "maximum_alpha": float(alpha.max()) if alpha.size else 0.0,
        "uses_evaluation_queries": False,
        "uses_ground_truth": False,
        "source": source,
        "storage": {
            "codebook_bytes_fp16": int(codebook.astype(np.float16).nbytes),
            "point_id_bytes": int(packed_points.nbytes),
            "code_id_bytes": int(packed_codes.nbytes),
            "reliability_bytes": int(packed_alpha.nbytes),
            "total_semantic_bytes": semantic_bytes,
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def quantiles(values):
    values = np.asarray(values)
    if values.size == 0:
        return [0.0] * 5
    return [float(value) for value in np.quantile(values, [0.0, 0.25, 0.5, 0.75, 1.0])]


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--l3_consensus", required=True)
    parser.add_argument("--part_group_ids", required=True)
    parser.add_argument("--part_interior_support", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--num_codes", type=int, default=2048)
    parser.add_argument("--train_samples", type=int, default=100000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--assignment_chunk", type=int, default=8192)
    parser.add_argument("--chunk_size", type=int, default=8192)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--stability_floor", type=float, default=0.50)
    parser.add_argument("--minimum_boundary", type=float, default=0.25)
    parser.add_argument("--minimum_split_cosine", type=float, default=0.85)
    parser.add_argument("--minimum_l3_reliability", type=float, default=0.65)
    parser.add_argument("--relative_reliability_slack", type=float, default=0.05)
    parser.add_argument("--minimum_residual", type=float, default=0.05)
    parser.add_argument("--maximum_residual", type=float, default=0.35)
    parser.add_argument("--maximum_sparse_fraction", type=float, default=0.10)
    parser.add_argument("--alpha_max", type=float, default=0.20)
    parser.add_argument("--global_weak_alpha", type=float, default=0.10)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed < 0 or args.num_codes <= 0 or args.train_samples <= 0:
        raise ValueError("Seed and codebook sizes must be valid")
    if args.iterations <= 0 or args.assignment_chunk <= 0 or args.chunk_size <= 0:
        raise ValueError("Iteration and chunk sizes must be positive")
    if not 0.0 <= args.stability_floor < 1.0:
        raise ValueError("Stability floor must be in [0, 1)")
    if not 0.0 <= args.minimum_boundary <= 1.0:
        raise ValueError("Minimum boundary must be in [0, 1]")
    if not 0.0 < args.maximum_sparse_fraction <= 1.0:
        raise ValueError("Maximum sparse fraction must be in (0, 1]")
    if not 0.0 <= args.alpha_max <= 1.0 or not 0.0 <= args.global_weak_alpha <= 1.0:
        raise ValueError("Residual alpha values must be in [0, 1]")
    if not 0.0 <= args.minimum_residual < args.maximum_residual <= 2.0:
        raise ValueError("Residual thresholds must be ordered within [0, 2]")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        print(f"Reuse sparse L3 residual codebook: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    base = load_consensus(args.base_consensus)
    l3 = load_consensus(args.l3_consensus)
    if base["initial_features"].shape != l3["initial_features"].shape:
        raise ValueError("Base and L3 consensus shapes differ")
    count, feature_dim = base["initial_features"].shape
    part_ids = np.load(os.path.abspath(args.part_group_ids)).astype(np.int64)
    interior_support = np.load(os.path.abspath(args.part_interior_support)).astype(np.float32)
    if part_ids.shape != (count,) or interior_support.shape != (count,):
        raise ValueError("Part hierarchy arrays do not match the consensus")

    selection = compute_selection(base, l3, part_ids, interior_support, args)
    codebook, assignments, valid_ids, codebook_diagnostics = train_and_assign_codebook(l3, args)
    np.save(os.path.join(output_dir, "l3_codebook.npy"), codebook.astype(np.float16))
    selected = selection["selected"]
    sparse_manifest = write_quantized_hypothesis(
        os.path.join(output_dir, "sparse_residual"),
        output_dir,
        selected,
        assignments[selected],
        selection["alpha"],
        codebook,
        count,
        "training_only_boundary_reliable_l3_score_residual",
        os.path.abspath(args.l3_consensus),
    )
    global_manifest = write_quantized_hypothesis(
        os.path.join(output_dir, "global_weak_residual"),
        output_dir,
        valid_ids,
        assignments[valid_ids],
        np.full(valid_ids.size, args.global_weak_alpha, dtype=np.float32),
        codebook,
        count,
        "global_weak_l3_score_residual_control",
        os.path.abspath(args.l3_consensus),
    )
    eligible = selection["eligible"]
    selected_mask = np.zeros(count, dtype=np.bool_)
    selected_mask[selected] = True
    np.save(os.path.join(output_dir, "selected_point_ids.npy"), selected.astype(np.uint32))
    np.save(os.path.join(output_dir, "selected_alpha.npy"), encode_alpha(selection["alpha"]))
    manifest = {
        "format_version": 1,
        "method": "a29_sparse_l3_score_residual",
        "representation": "small_codebook_sparse_query_positive_residual",
        "num_gaussians": int(count),
        "feature_dim": int(feature_dim),
        "base_consensus": os.path.abspath(args.base_consensus),
        "l3_consensus": os.path.abspath(args.l3_consensus),
        "part_group_ids": os.path.abspath(args.part_group_ids),
        "part_interior_support": os.path.abspath(args.part_interior_support),
        "uses_evaluation_queries": False,
        "uses_ground_truth": False,
        "selection_rule": {
            "part_assignment_required": True,
            "boundary_minimum": args.minimum_boundary,
            "l3_split_cosine_minimum": args.minimum_split_cosine,
            "l3_split_reliability_minimum": args.minimum_l3_reliability,
            "l3_relative_reliability_slack": args.relative_reliability_slack,
            "residual_disagreement_range": [args.minimum_residual, args.maximum_residual],
            "maximum_sparse_fraction": args.maximum_sparse_fraction,
            "alpha_max": args.alpha_max,
        },
        "selection_diagnostics": {
            "eligible_points": int(eligible.sum()),
            "eligible_fraction": float(eligible.mean()),
            "selected_points": int(selected.size),
            "selected_fraction": float(selected.size / count),
            "selected_boundary_quantiles": quantiles(selection["boundary"][selected]),
            "selected_l3_reliability_quantiles": quantiles(selection["l3_reliability"][selected]),
            "selected_base_reliability_quantiles": quantiles(selection["base_reliability"][selected]),
            "selected_residual_quantiles": quantiles(selection["residual_disagreement"][selected]),
            "selected_alpha_quantiles": quantiles(selection["alpha"]),
            "part_valid_fraction": float((part_ids >= 0).mean()),
            "l3_valid_fraction": float(selection["l3_valid"].mean()),
        },
        "codebook": codebook_diagnostics,
        "artifacts": {
            "codebook": "l3_codebook.npy",
            "sparse_residual": "sparse_residual",
            "global_weak_residual": "global_weak_residual",
            "selected_point_ids": "selected_point_ids.npy",
            "selected_alpha": "selected_alpha.npy",
        },
        "storage": {
            "sparse_residual": sparse_manifest["storage"],
            "global_weak_residual": global_manifest["storage"],
        },
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
