#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

import numpy as np


def l2_normalize(x, eps=1e-9):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def spherical_kmeans(features, num_codes, iterations, seed, sample_weight=None):
    rng = np.random.default_rng(seed)
    features = l2_normalize(features.astype(np.float32))
    num_items = features.shape[0]
    if num_codes >= num_items:
        code_ids = np.arange(num_items, dtype=np.int32)
        return features.copy(), code_ids

    init_ids = rng.choice(num_items, size=num_codes, replace=False)
    codebook = features[init_ids].copy()
    code_ids = np.zeros(num_items, dtype=np.int32)

    for _ in range(iterations):
        sims = features @ codebook.T
        next_ids = sims.argmax(axis=1).astype(np.int32)
        next_codebook = np.zeros_like(codebook)
        counts = np.bincount(next_ids, minlength=num_codes)
        for code in range(num_codes):
            if counts[code] == 0:
                farthest = np.argmin(sims.max(axis=1))
                next_codebook[code] = features[farthest]
            else:
                members = next_ids == code
                if sample_weight is None:
                    next_codebook[code] = features[members].mean(axis=0)
                else:
                    weights = sample_weight[members].astype(np.float32)
                    next_codebook[code] = (features[members] * weights[:, None]).sum(axis=0) / max(float(weights.sum()), 1e-9)
        codebook = l2_normalize(next_codebook)
        if np.array_equal(code_ids, next_ids):
            break
        code_ids = next_ids

    return codebook.astype(np.float32), code_ids.astype(np.int32)


def residual_spherical_quantize(features, num_codes, levels, iterations, seed, sample_weight=None):
    features = l2_normalize(features.astype(np.float32))
    residual = features.copy()
    codebooks = []
    codes = []
    reconstruction = np.zeros_like(features)
    for level in range(levels):
        codebook, code_ids = spherical_kmeans(
            residual,
            num_codes,
            iterations,
            seed + level * 9973,
            sample_weight=sample_weight,
        )
        codebooks.append(codebook)
        codes.append(code_ids)
        reconstruction += codebook[code_ids]
        reconstruction = l2_normalize(reconstruction)
        residual = l2_normalize(features - np.sum(features * reconstruction, axis=1, keepdims=True) * reconstruction)
    return np.stack(codebooks, axis=0), np.stack(codes, axis=1).astype(np.int32), reconstruction.astype(np.float32)


def group_usage_weights(assignments, num_groups):
    top_group_ids = assignments["top_group_ids"]
    top_group_scores = assignments["top_group_scores"]
    weights = np.ones(num_groups, dtype=np.float32)
    valid = top_group_ids >= 0
    for gid, score in zip(top_group_ids[valid], top_group_scores[valid]):
        weights[int(gid)] += float(score)
    weights /= max(float(weights.mean()), 1e-9)
    return weights


def remap_ids(ids, group_to_code):
    out = ids.copy()
    valid = out >= 0
    out[valid] = group_to_code[out[valid]]
    return out.astype(np.int32)


def main():
    parser = ArgumentParser(description="Quantize multi-group semantic tokens into a discrete codebook.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--num_codes", type=int, default=128)
    parser.add_argument("--levels", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usage_weighted", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    group_features_path = os.path.join(args.artifact_dir, "group_features.npy")
    assignments_path = os.path.join(args.artifact_dir, "point_group_assignments.npz")
    group_features = np.load(group_features_path).astype(np.float32)
    assignments = np.load(assignments_path)

    sample_weight = group_usage_weights(assignments, group_features.shape[0]) if args.usage_weighted else None
    if args.levels == 1:
        codebook, group_to_code = spherical_kmeans(
            group_features,
            args.num_codes,
            args.iterations,
            args.seed,
            sample_weight=sample_weight,
        )
        codebooks = codebook[None, ...]
        group_codes = group_to_code[:, None]
        reconstructed_group_features = l2_normalize(codebook[group_to_code]).astype(np.float32)
    else:
        codebooks, group_codes, reconstructed_group_features = residual_spherical_quantize(
            group_features,
            args.num_codes,
            args.levels,
            args.iterations,
            args.seed,
            sample_weight=sample_weight,
        )
        group_to_code = group_codes[:, 0]

    code_assignments = {}
    for key in assignments.files:
        value = assignments[key]
        if key in ("top_group_ids", "best_group"):
            code_assignments[key] = remap_ids(value, group_to_code)
        else:
            code_assignments[key] = value

    np.save(os.path.join(args.output_dir, "codebook.npy"), codebooks[0])
    np.save(os.path.join(args.output_dir, "codebooks.npy"), codebooks)
    np.save(os.path.join(args.output_dir, "group_to_code.npy"), group_to_code)
    np.save(os.path.join(args.output_dir, "group_codes.npy"), group_codes)
    np.save(os.path.join(args.output_dir, "group_features_quantized.npy"), reconstructed_group_features)
    np.savez_compressed(os.path.join(args.output_dir, "point_code_assignments.npz"), **code_assignments)

    sims = np.sum(group_features * reconstructed_group_features, axis=1)
    summary = {
        "num_groups": int(group_features.shape[0]),
        "feature_dim": int(group_features.shape[1]),
        "num_codes": int(codebooks.shape[1]),
        "levels": int(args.levels),
        "total_code_vectors": int(codebooks.shape[0] * codebooks.shape[1]),
        "usage_weighted": bool(args.usage_weighted),
        "compression_ratio_groups_to_codes": float(group_features.shape[0] / (codebooks.shape[0] * codebooks.shape[1])),
        "mean_reconstruction_cosine": float(np.mean(sims)),
        "min_reconstruction_cosine": float(np.min(sims)),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "codebook_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
