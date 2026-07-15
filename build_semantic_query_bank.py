#!/usr/bin/env python
"""Build non-evaluation semantic anchors for query-distribution KL training."""

import json
import os
from argparse import ArgumentParser

import numpy as np

from build_gaussian_multilevel_codebook import faiss_kmeans
from semantic_field_utils import collect_mask_features


def main():
    parser = ArgumentParser(
        description="Cluster frozen 2D CLIP observations into a semantic query-anchor bank."
    )
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--num_queries", type=int, default=256)
    parser.add_argument("--max_features", type=int, default=200000)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.num_queries <= 1 or args.max_features <= 0 or args.iterations <= 0:
        raise ValueError("Query count, feature count, and iterations must be positive")

    features, feature_paths = collect_mask_features(
        args.feature_dir,
        max_features=args.max_features,
        seed=args.seed,
    )
    values = features.numpy().astype(np.float32, copy=False)
    codebook, _ = faiss_kmeans(
        values,
        args.num_queries,
        args.iterations,
        args.seed,
        spherical=True,
        use_gpu=args.faiss_gpu,
    )
    output_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, codebook.astype(np.float16))
    summary = {
        "output": output_path,
        "num_queries": int(codebook.shape[0]),
        "feature_dim": int(codebook.shape[1]),
        "num_training_features": int(values.shape[0]),
        "num_feature_files": len(feature_paths),
        "source": "Frozen 2D CLIP observations; no LeRF-OVS evaluation labels are used.",
        "args": vars(args),
    }
    with open(os.path.splitext(output_path)[0] + "_summary.json", "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
