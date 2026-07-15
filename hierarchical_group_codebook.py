#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

import numpy as np

from quantize_multigroup_codebook import group_usage_weights, l2_normalize, spherical_kmeans


def main():
    parser = ArgumentParser(description="Build a coarse-to-fine hierarchical codebook for group tokens.")
    parser.add_argument("--group_features", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--coarse_codes", type=int, default=32)
    parser.add_argument("--fine_codes", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--usage_weighted", action="store_true")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    features = l2_normalize(np.load(args.group_features).astype(np.float32))
    assignments = np.load(args.assignments)
    weights = group_usage_weights(assignments, features.shape[0]) if args.usage_weighted else None

    coarse_book, coarse_ids = spherical_kmeans(
        features,
        args.coarse_codes,
        args.iterations,
        args.seed,
        sample_weight=weights,
    )

    reconstructed = np.zeros_like(features)
    fine_ids = np.full(features.shape[0], -1, dtype=np.int32)
    fine_books = {}
    fine_counts = {}
    total_fine_vectors = 0

    for coarse in range(coarse_book.shape[0]):
        member_ids = np.flatnonzero(coarse_ids == coarse)
        if member_ids.size == 0:
            continue
        member_features = features[member_ids]
        member_weights = weights[member_ids] if weights is not None else None
        num_fine = min(args.fine_codes, member_ids.size)
        fine_book, local_fine_ids = spherical_kmeans(
            member_features,
            num_fine,
            args.iterations,
            args.seed + 1009 * (coarse + 1),
            sample_weight=member_weights,
        )
        fine_books[f"coarse_{coarse}"] = fine_book.astype(np.float32)
        fine_counts[f"coarse_{coarse}"] = int(num_fine)
        fine_ids[member_ids] = local_fine_ids
        reconstructed[member_ids] = fine_book[local_fine_ids]
        total_fine_vectors += int(num_fine)

    reconstructed = l2_normalize(reconstructed).astype(np.float32)
    sims = np.sum(features * reconstructed, axis=1)

    np.save(os.path.join(args.output_dir, "coarse_codebook.npy"), coarse_book.astype(np.float32))
    np.save(os.path.join(args.output_dir, "coarse_ids.npy"), coarse_ids.astype(np.int32))
    np.save(os.path.join(args.output_dir, "fine_ids.npy"), fine_ids.astype(np.int32))
    np.save(os.path.join(args.output_dir, "group_features_hierarchical.npy"), reconstructed)
    np.savez_compressed(os.path.join(args.output_dir, "fine_codebooks.npz"), **fine_books)

    summary = {
        "num_groups": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "coarse_codes": int(coarse_book.shape[0]),
        "fine_codes_per_coarse_max": int(args.fine_codes),
        "total_fine_vectors": int(total_fine_vectors),
        "total_vectors_including_coarse": int(total_fine_vectors + coarse_book.shape[0]),
        "usage_weighted": bool(args.usage_weighted),
        "mean_reconstruction_cosine": float(np.mean(sims)),
        "min_reconstruction_cosine": float(np.min(sims)),
        "fine_counts": fine_counts,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "hierarchical_codebook_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
