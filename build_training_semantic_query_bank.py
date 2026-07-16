#!/usr/bin/env python
"""Build a balanced semantic query bank from training-view mask features."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch

from build_gaussian_multilevel_codebook import faiss_kmeans, l2_normalize


def sample_view_features(features, max_features, rng):
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("View feature_latents must be a 2D table")
    if values.shape[0] > max_features:
        indices = rng.choice(values.shape[0], max_features, replace=False)
        values = values[indices]
    return l2_normalize(values)


def load_cache(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def collect_balanced_training_features(cache_dir, max_features_per_view, seed):
    cache_dir = os.path.abspath(cache_dir)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("codec_type") != "identity":
        raise ValueError("Training query banks require an identity semantic cache")
    views = manifest.get("views", [])
    if not views:
        raise ValueError("Cache manifest contains no training views")

    rng = np.random.default_rng(seed)
    tables = []
    view_counts = []
    semantic_dim = int(manifest["semantic_dim"])
    for entry in views:
        cache = load_cache(os.path.join(cache_dir, entry["cache"]))
        features = cache.get("feature_latents")
        if features is None:
            raise ValueError(f"Missing feature_latents in {entry['cache']}")
        sampled = sample_view_features(
            features.float().numpy(), max_features_per_view, rng
        )
        if sampled.shape[1] != semantic_dim:
            raise ValueError("View feature dimension does not match the manifest")
        tables.append(sampled)
        view_counts.append(int(sampled.shape[0]))
    return l2_normalize(np.concatenate(tables, axis=0)), manifest, view_counts


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=512)
    parser.add_argument("--max_features_per_view", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.num_queries <= 1:
        raise ValueError("--num_queries must be greater than one")
    if args.max_features_per_view <= 0 or args.iterations <= 0:
        raise ValueError("Sampling and iteration counts must be positive")

    output = os.path.abspath(args.output)
    metadata_path = os.path.splitext(output)[0] + ".json"
    if os.path.isfile(output) and os.path.isfile(metadata_path) and not args.force:
        print(f"Reuse training semantic query bank: {output}")
        return

    training, manifest, view_counts = collect_balanced_training_features(
        args.cache_dir,
        args.max_features_per_view,
        args.seed,
    )
    queries, _ = faiss_kmeans(
        training,
        min(args.num_queries, training.shape[0]),
        args.iterations,
        args.seed,
        spherical=True,
        use_gpu=args.faiss_gpu,
    )
    queries = l2_normalize(queries)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, queries.astype(np.float16))
    metadata = {
        "representation": "training_view_semantic_query_bank",
        "cache_dir": os.path.abspath(args.cache_dir),
        "source": "training-view feature_latents only; no evaluation text or 3D labels",
        "num_views": len(view_counts),
        "num_training_features": int(training.shape[0]),
        "features_per_view": {
            "min": min(view_counts),
            "mean": float(np.mean(view_counts)),
            "max": max(view_counts),
        },
        "num_queries": int(queries.shape[0]),
        "semantic_dim": int(queries.shape[1]),
        "cache_feature_level": manifest.get("feature_level"),
        "args": vars(args),
    }
    with open(metadata_path, "w") as handle:
        json.dump(metadata, handle, indent=2)
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
