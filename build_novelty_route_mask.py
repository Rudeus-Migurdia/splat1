#!/usr/bin/env python
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from eval_lerf_ovs_gaussian_codebook_miou import GaussianCodebookArtifact


def semantic_novelty_and_quantization_noise(
    base_feature,
    candidate_feature,
    base_reconstruction,
    candidate_reconstruction,
):
    base_feature = F.normalize(base_feature.float(), dim=-1)
    candidate_feature = F.normalize(candidate_feature.float(), dim=-1)
    base_reconstruction = F.normalize(base_reconstruction.float(), dim=-1)
    candidate_reconstruction = F.normalize(candidate_reconstruction.float(), dim=-1)
    novelty = (1.0 - F.cosine_similarity(
        base_reconstruction, candidate_reconstruction, dim=-1
    )).clamp_min(0.0)
    base_noise = (1.0 - F.cosine_similarity(
        base_feature, base_reconstruction, dim=-1
    )).clamp_min(0.0)
    candidate_noise = (1.0 - F.cosine_similarity(
        candidate_feature, candidate_reconstruction, dim=-1
    )).clamp_min(0.0)
    return novelty, torch.maximum(base_noise, candidate_noise)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--base_codebook_dir", required=True)
    parser.add_argument("--candidate_codebook_dir", required=True)
    parser.add_argument("--noise_ratio", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.noise_ratio <= 0.0:
        raise ValueError("--noise_ratio must be positive")

    base = torch.load(os.path.abspath(args.base_consensus), map_location="cpu")
    candidate = torch.load(
        os.path.abspath(args.candidate_consensus), map_location="cpu"
    )
    base_feature = base["initial_features"]
    candidate_feature = candidate["initial_features"]
    if base_feature.shape != candidate_feature.shape:
        raise ValueError("Base and candidate consensus shapes must match")

    base_codebook = GaussianCodebookArtifact(args.base_codebook_dir, device="cpu")
    candidate_codebook = GaussianCodebookArtifact(
        args.candidate_codebook_dir, device="cpu"
    )
    num_gaussians = base_feature.shape[0]
    if base_codebook.num_gaussians != num_gaussians:
        raise ValueError("Base codebook does not match the consensus")
    if candidate_codebook.num_gaussians != num_gaussians:
        raise ValueError("Candidate codebook does not match the consensus")

    mask = torch.zeros(num_gaussians, dtype=torch.bool)
    valid_ratios = []
    for start in range(0, num_gaussians, args.chunk_size):
        end = min(start + args.chunk_size, num_gaussians)
        base_reconstruction = base_codebook.reconstruct_range(start, end)
        candidate_reconstruction = candidate_codebook.reconstruct_range(start, end)
        valid = (
            (base_reconstruction.norm(dim=-1) > 0.0)
            & (candidate_reconstruction.norm(dim=-1) > 0.0)
        )
        novelty, noise = semantic_novelty_and_quantization_noise(
            base_feature[start:end],
            candidate_feature[start:end],
            base_reconstruction,
            candidate_reconstruction,
        )
        mask[start:end] = valid & (novelty > args.noise_ratio * noise)
        valid_ratios.append((novelty / noise.clamp_min(1e-8))[valid])

    ratios = torch.cat(valid_ratios)
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, mask.numpy())
    diagnostics = {
        "base_consensus": os.path.abspath(args.base_consensus),
        "candidate_consensus": os.path.abspath(args.candidate_consensus),
        "base_codebook_dir": os.path.abspath(args.base_codebook_dir),
        "candidate_codebook_dir": os.path.abspath(args.candidate_codebook_dir),
        "noise_ratio": args.noise_ratio,
        "num_gaussians": num_gaussians,
        "num_both_codebook_valid": int(ratios.numel()),
        "num_candidate_points": int(mask.sum()),
        "candidate_fraction": float(mask.float().mean()),
        "novelty_noise_ratio_quantiles": {
            str(q): float(torch.quantile(ratios, q))
            for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
        },
    }
    with open(os.path.splitext(output)[0] + ".json", "w") as handle:
        json.dump(diagnostics, handle, indent=2)
    print(json.dumps(diagnostics, indent=2))


if __name__ == "__main__":
    main()
