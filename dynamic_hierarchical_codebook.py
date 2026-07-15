#!/usr/bin/env python
"""Build a query-routable, fully discrete coarse-to-fine group codebook.

The source group features may be distilled from a PQ teacher, but this artifact
contains only code vectors and integer group-to-code attachments for inference.
"""

import json
import os
from argparse import ArgumentParser

import numpy as np

from quantize_multigroup_codebook import group_usage_weights, l2_normalize, spherical_kmeans


def make_reverse_mount(candidate_ids, num_codes):
    """Pack the many-to-many fine-code -> group attachment into CSR arrays."""
    num_groups = candidate_ids.shape[0]
    group_ids = np.repeat(np.arange(num_groups, dtype=np.int32), candidate_ids.shape[1])
    code_ids = candidate_ids.reshape(-1)
    valid = code_ids >= 0
    group_ids = group_ids[valid]
    code_ids = code_ids[valid]
    order = np.argsort(code_ids, kind="stable")
    counts = np.bincount(code_ids[order], minlength=num_codes).astype(np.int64)
    offsets = np.zeros(num_codes + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return group_ids[order], offsets


def main():
    parser = ArgumentParser(
        description="Build a dynamic hierarchical discrete codebook with top-M fine-code mounts."
    )
    parser.add_argument("--group_features", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--coarse_codes", type=int, default=32)
    parser.add_argument("--fine_codes", type=int, default=8)
    parser.add_argument("--top_m", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usage_weighted", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.coarse_codes <= 0 or args.fine_codes <= 0 or args.top_m <= 0:
        raise ValueError("--coarse_codes, --fine_codes, and --top_m must be positive.")

    os.makedirs(args.output_dir, exist_ok=True)
    features = l2_normalize(np.load(args.group_features).astype(np.float32))
    assignments = np.load(args.assignments)
    weights = group_usage_weights(assignments, features.shape[0]) if args.usage_weighted else None

    coarse_codebook, group_coarse_ids = spherical_kmeans(
        features,
        args.coarse_codes,
        args.iterations,
        args.seed,
        sample_weight=weights,
    )

    candidate_ids = np.full((features.shape[0], args.top_m), -1, dtype=np.int32)
    candidate_scores = np.zeros((features.shape[0], args.top_m), dtype=np.float32)
    primary_fine_ids = np.full(features.shape[0], -1, dtype=np.int32)
    fine_vectors = []
    fine_parent_ids = []
    next_fine_id = 0

    for coarse_id in range(coarse_codebook.shape[0]):
        member_ids = np.flatnonzero(group_coarse_ids == coarse_id)
        if member_ids.size == 0:
            continue
        member_features = features[member_ids]
        member_weights = weights[member_ids] if weights is not None else None
        local_count = min(args.fine_codes, member_ids.size)
        local_book, _ = spherical_kmeans(
            member_features,
            local_count,
            args.iterations,
            args.seed + 1009 * (coarse_id + 1),
            sample_weight=member_weights,
        )
        local_scores = member_features @ local_book.T
        local_top_m = min(args.top_m, local_count)
        local_order = np.argsort(-local_scores, axis=1)[:, :local_top_m]
        global_ids = local_order + next_fine_id
        candidate_ids[member_ids, :local_top_m] = global_ids.astype(np.int32)
        candidate_scores[member_ids, :local_top_m] = np.take_along_axis(
            local_scores, local_order, axis=1
        ).astype(np.float32)
        primary_fine_ids[member_ids] = global_ids[:, 0]
        fine_vectors.append(local_book.astype(np.float32))
        fine_parent_ids.append(np.full(local_count, coarse_id, dtype=np.int32))
        next_fine_id += local_count

    fine_codebook = np.concatenate(fine_vectors, axis=0)
    fine_parent_ids = np.concatenate(fine_parent_ids, axis=0)
    reconstructed = l2_normalize(fine_codebook[primary_fine_ids])
    coarse_reconstructed = coarse_codebook[group_coarse_ids]
    reverse_indices, reverse_offsets = make_reverse_mount(candidate_ids, fine_codebook.shape[0])
    fine_usage = np.zeros(fine_codebook.shape[0], dtype=np.float32)
    if weights is None:
        usage = np.ones(features.shape[0], dtype=np.float32)
    else:
        usage = weights
    for rank in range(candidate_ids.shape[1]):
        valid = candidate_ids[:, rank] >= 0
        np.add.at(fine_usage, candidate_ids[valid, rank], usage[valid])

    np.save(os.path.join(args.output_dir, "coarse_codebook.npy"), coarse_codebook.astype(np.float32))
    np.save(os.path.join(args.output_dir, "fine_codebook.npy"), fine_codebook.astype(np.float32))
    np.save(os.path.join(args.output_dir, "fine_parent_ids.npy"), fine_parent_ids)
    np.save(os.path.join(args.output_dir, "group_coarse_ids.npy"), group_coarse_ids.astype(np.int32))
    np.save(os.path.join(args.output_dir, "group_fine_candidate_ids.npy"), candidate_ids)
    np.save(os.path.join(args.output_dir, "group_fine_candidate_scores.npy"), candidate_scores)
    np.save(os.path.join(args.output_dir, "group_primary_fine_ids.npy"), primary_fine_ids)
    np.save(os.path.join(args.output_dir, "group_features_dynamic_hierarchical.npy"), reconstructed.astype(np.float32))
    np.save(os.path.join(args.output_dir, "group_features_coarse.npy"), coarse_reconstructed.astype(np.float32))
    np.save(os.path.join(args.output_dir, "fine_usage.npy"), fine_usage)
    np.savez_compressed(
        os.path.join(args.output_dir, "fine_code_to_groups.npz"),
        indices=reverse_indices,
        offsets=reverse_offsets,
    )

    reconstruction_cosine = np.sum(features * reconstructed, axis=1)
    coarse_cosine = np.sum(features * coarse_reconstructed, axis=1)
    active_fine = int((fine_usage > 0).sum())
    summary = {
        "num_groups": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "coarse_codes": int(coarse_codebook.shape[0]),
        "fine_codes_total": int(fine_codebook.shape[0]),
        "fine_codes_per_coarse_max": int(args.fine_codes),
        "top_m": int(args.top_m),
        "total_code_vectors": int(coarse_codebook.shape[0] + fine_codebook.shape[0]),
        "compression_ratio_groups_to_code_vectors": float(
            features.shape[0] / max(coarse_codebook.shape[0] + fine_codebook.shape[0], 1)
        ),
        "usage_weighted": bool(args.usage_weighted),
        "mean_coarse_cosine": float(coarse_cosine.mean()),
        "mean_fine_reconstruction_cosine": float(reconstruction_cosine.mean()),
        "min_fine_reconstruction_cosine": float(reconstruction_cosine.min()),
        "active_fine_codes": active_fine,
        "dead_fine_codes": int(fine_codebook.shape[0] - active_fine),
        "mean_primary_margin": float(
            (candidate_scores[:, 0] - candidate_scores[:, 1]).mean()
            if args.top_m > 1
            else 0.0
        ),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "dynamic_hierarchical_codebook_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
