#!/usr/bin/env python
"""Train one ample shared vocabulary on two semantic modes."""

import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np

from build_gaussian_multilevel_codebook import (
    ConsensusFeatureSource,
    faiss_kmeans,
    l2_normalize,
)


def balanced_training_features(base, candidate, samples_per_source, seed):
    if base.feature_dim != candidate.feature_dim:
        raise ValueError("Semantic modes must have the same feature dimension")
    rng = np.random.default_rng(seed)
    tables = []
    for source in (base, candidate):
        valid = np.flatnonzero(source.valid_mask)
        if valid.size == 0:
            raise ValueError("Semantic mode contains no valid features")
        count = min(int(samples_per_source), int(valid.size))
        indices = rng.choice(valid, count, replace=False)
        tables.append(source.read(indices))
    return l2_normalize(np.concatenate(tables, axis=0))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--num_codes", type=int, default=32768)
    parser.add_argument("--samples_per_source", type=int, default=262144)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.num_codes <= 0 or args.samples_per_source <= 0 or args.iterations <= 0:
        raise ValueError("Code, sample, and iteration counts must be positive")

    output_dir = Path(args.output_dir).resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists() and not args.force:
        print(f"Reuse existing joint vocabulary: {output_dir}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    base = ConsensusFeatureSource(args.base_consensus)
    candidate = ConsensusFeatureSource(args.candidate_consensus)
    if base.num_items != candidate.num_items:
        raise ValueError("Semantic modes must describe the same Gaussians")
    training = balanced_training_features(
        base,
        candidate,
        args.samples_per_source,
        args.seed,
    )
    codebook, _ = faiss_kmeans(
        training,
        min(args.num_codes, training.shape[0]),
        args.iterations,
        args.seed,
        spherical=True,
        use_gpu=args.faiss_gpu,
    )
    codebook = l2_normalize(codebook)
    np.save(output_dir / "codebook_shared.npy", codebook.astype(np.float16))
    manifest = {
        "representation": "joint_semantic_mode_vocabulary",
        "num_codes": int(codebook.shape[0]),
        "feature_dim": int(codebook.shape[1]),
        "training_samples": int(training.shape[0]),
        "samples_per_source_requested": int(args.samples_per_source),
        "base_consensus": os.path.abspath(args.base_consensus),
        "candidate_consensus": os.path.abspath(args.candidate_consensus),
        "storage_bytes_fp16": int(codebook.size * np.dtype(np.float16).itemsize),
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
