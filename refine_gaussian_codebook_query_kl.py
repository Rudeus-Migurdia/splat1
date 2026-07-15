#!/usr/bin/env python
"""Refine fixed-ID shared codebooks with cosine and semantic-anchor KL losses."""

import json
import os
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_gaussian_multilevel_codebook import (
    ConsensusFeatureSource,
    DrSplatPqFeatureSource,
    NumpyFeatureSource,
    l2_normalize,
)


def load_artifact(artifact_dir, device):
    artifact_dir = os.path.abspath(artifact_dir)
    with open(os.path.join(artifact_dir, "manifest.json")) as source:
        manifest = json.load(source)
    point_ids = np.load(
        os.path.join(artifact_dir, manifest["point_code_ids"]),
        mmap_mode="r",
    )
    valid_mask = np.load(
        os.path.join(artifact_dir, manifest["valid_mask"]),
        mmap_mode="r",
    )
    codebooks = []
    for name in manifest["codebook_files"]:
        values = np.load(os.path.join(artifact_dir, name)).astype(np.float32)
        codebooks.append(torch.nn.Parameter(torch.from_numpy(values).to(device)))
    return artifact_dir, manifest, point_ids, valid_mask, torch.nn.ParameterList(codebooks)


def reconstruct(codebooks, code_ids):
    value = torch.zeros(
        (code_ids.shape[0], codebooks[0].shape[1]),
        dtype=codebooks[0].dtype,
        device=code_ids.device,
    )
    for level, codebook in enumerate(codebooks):
        value = value + codebook[code_ids[:, level]]
    return F.normalize(value, dim=-1)


def query_kl(prediction, target, query_bank, temperature):
    prediction_logits = prediction @ query_bank.T / temperature
    with torch.no_grad():
        target_probabilities = torch.softmax(target @ query_bank.T / temperature, dim=-1)
    return F.kl_div(
        torch.log_softmax(prediction_logits, dim=-1),
        target_probabilities,
        reduction="batchmean",
    )


@torch.no_grad()
def evaluate_sample(source, valid_indices, point_ids, invalid_id, codebooks, sample_count, seed):
    rng = np.random.default_rng(seed)
    indices = rng.choice(valid_indices, min(sample_count, valid_indices.size), replace=False)
    ids = np.asarray(point_ids[indices], dtype=np.int64)
    if np.any(ids == invalid_id):
        raise ValueError("Valid points contain invalid code IDs")
    targets = torch.from_numpy(source.read(indices)).float().cuda()
    decoded = reconstruct(codebooks, torch.from_numpy(ids).long().cuda())
    cosine = F.cosine_similarity(decoded, targets, dim=-1)
    return {
        "mean_cosine": float(cosine.mean()),
        "min_cosine": float(cosine.min()),
        "num_samples": int(cosine.numel()),
    }


def save_artifact(source_dir, output_dir, manifest, codebooks, training_summary):
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for key in ("point_code_ids", "valid_mask"):
        name = manifest[key]
        shutil.copy2(os.path.join(source_dir, name), os.path.join(output_dir, name))
    codebook_files = []
    codebook_bytes = 0
    for level, codebook in enumerate(codebooks):
        name = f"codebook_level_{level}.npy"
        values = codebook.detach().cpu().numpy().astype(np.float16)
        np.save(os.path.join(output_dir, name), values)
        codebook_files.append(name)
        codebook_bytes += int(values.nbytes)
    output_manifest = dict(manifest)
    output_manifest["codebook_files"] = codebook_files
    output_manifest["query_kl_refinement"] = training_summary
    output_manifest["source_initial_artifact"] = source_dir
    storage = dict(output_manifest["storage"])
    storage["codebook_bytes_fp16"] = codebook_bytes
    storage["total_semantic_bytes"] = (
        codebook_bytes + storage["point_id_bytes"] + storage["valid_mask_bytes"]
    )
    storage["compression_ratio_vs_512d_fp16"] = (
        storage["full_per_gaussian_fp16_bytes"] / storage["total_semantic_bytes"]
    )
    storage["bytes_per_gaussian_amortized"] = (
        storage["total_semantic_bytes"] / output_manifest["num_gaussians"]
    )
    output_manifest["storage"] = storage
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(output_manifest, output, indent=2)
    return output_manifest


def main():
    parser = ArgumentParser(description="Refine a Gaussian codebook using query-anchor KL.")
    parser.add_argument("--initial_codebook_dir", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--features")
    source.add_argument("--consensus")
    source.add_argument("--drsplat_checkpoint")
    parser.add_argument("--valid_mask", default=None)
    parser.add_argument("--pq_index", default=None)
    parser.add_argument("--query_bank", required=True)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch_gaussians", type=int, default=8192)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--query_kl_weight", type=float, default=0.1)
    parser.add_argument("--query_temperature", type=float, default=0.07)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.batch_gaussians <= 0 or args.learning_rate <= 0:
        raise ValueError("Iterations, batch size, and learning rate must be positive")
    if args.query_temperature <= 0 or args.query_kl_weight < 0 or args.cosine_weight < 0:
        raise ValueError("Loss weights must be non-negative and temperature positive")

    if args.drsplat_checkpoint:
        if not args.pq_index:
            raise ValueError("--drsplat_checkpoint requires --pq_index")
        feature_source = DrSplatPqFeatureSource(args.drsplat_checkpoint, args.pq_index)
    elif args.consensus:
        feature_source = ConsensusFeatureSource(args.consensus)
    else:
        feature_source = NumpyFeatureSource(args.features, args.valid_mask)

    source_dir, manifest, point_ids, artifact_valid, codebooks = load_artifact(
        args.initial_codebook_dir,
        "cuda",
    )
    if feature_source.num_items != int(manifest["num_gaussians"]):
        raise ValueError("Feature source and codebook contain different Gaussian counts")
    valid_mask = np.asarray(feature_source.valid_mask, dtype=bool) & np.asarray(
        artifact_valid,
        dtype=bool,
    )
    valid_indices = np.flatnonzero(valid_mask)
    invalid_id = int(manifest["invalid_id"])
    query_bank = np.load(args.query_bank).astype(np.float32)
    if query_bank.ndim != 2 or query_bank.shape[1] != int(manifest["feature_dim"]):
        raise ValueError("Query bank dimension does not match the codebook")
    query_bank = torch.from_numpy(l2_normalize(query_bank)).float().cuda()

    optimizer = torch.optim.AdamW(codebooks.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    rng = np.random.default_rng(args.seed)
    history = []
    for iteration in range(1, args.iterations + 1):
        indices = rng.choice(
            valid_indices,
            min(args.batch_gaussians, valid_indices.size),
            replace=True,
        )
        ids = np.asarray(point_ids[indices], dtype=np.int64)
        if np.any(ids == invalid_id):
            raise ValueError("Valid points contain invalid code IDs")
        targets = torch.from_numpy(feature_source.read(indices)).float().cuda()
        prediction = reconstruct(codebooks, torch.from_numpy(ids).long().cuda())
        cosine_loss = 1.0 - F.cosine_similarity(prediction, targets, dim=-1).mean()
        kl_loss = query_kl(prediction, targets, query_bank, args.query_temperature)
        loss = args.cosine_weight * cosine_loss + args.query_kl_weight * kl_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if iteration % args.log_interval == 0 or iteration == 1:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "cosine_loss": float(cosine_loss.detach()),
                "query_kl": float(kl_loss.detach()),
            }
            history.append(row)
            print(json.dumps(row))

    sample_metrics = evaluate_sample(
        feature_source,
        valid_indices,
        point_ids,
        invalid_id,
        codebooks,
        sample_count=65536,
        seed=args.seed + 1000,
    )
    training_summary = {
        "query_bank": os.path.abspath(args.query_bank),
        "source": feature_source.metadata(),
        "history": history,
        "sample_metrics": sample_metrics,
        "args": vars(args),
    }
    output_manifest = save_artifact(
        source_dir,
        args.output_dir,
        manifest,
        codebooks,
        training_summary,
    )
    print(json.dumps({"output": os.path.abspath(args.output_dir), "manifest": output_manifest}, indent=2))


if __name__ == "__main__":
    main()
