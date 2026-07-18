#!/usr/bin/env python
"""Keep an A24 micro ID only where its Gaussian-level L2 evidence prefers it."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def decode_group_features(vocabulary, semantic_ids, invalid):
    valid = semantic_ids != invalid
    safe = np.where(valid, semantic_ids, 0)
    decoded = (vocabulary[safe].astype(np.float32) * valid[..., None]).sum(axis=1)
    return normalize(decoded)


def prefer_micro(point_features, fine_features, micro_features):
    point_features = normalize(np.asarray(point_features, dtype=np.float32))
    fine_features = normalize(np.asarray(fine_features, dtype=np.float32))
    micro_features = normalize(np.asarray(micro_features, dtype=np.float32))
    point_supported = np.linalg.norm(point_features, axis=-1) > 0.0
    fine_similarity = np.einsum("ij,ij->i", point_features, fine_features)
    micro_similarity = np.einsum("ij,ij->i", point_features, micro_features)
    return point_supported & (micro_similarity > fine_similarity), (
        micro_similarity - fine_similarity
    )


def link(source, destination):
    if os.path.lexists(destination):
        os.unlink(destination)
    os.symlink(os.path.abspath(source), destination)


def quantiles(values):
    values = np.asarray(values)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    import torch

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_dir", required=True)
    parser.add_argument("--l2_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--fine_slot", type=int, default=1)
    parser.add_argument("--micro_slot", type=int, default=2)
    parser.add_argument("--chunk_size", type=int, default=65536)
    args = parser.parse_args(sys.argv[1:])
    if args.chunk_size <= 0:
        raise ValueError("Chunk size must be positive")

    source_dir = os.path.abspath(args.source_artifact_dir)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if "micro" not in manifest.get("vocabulary_modalities", []):
        raise ValueError("Point-supported capacity requires an A24 micro artifact")
    vocabulary = np.load(
        os.path.join(source_dir, manifest["group_codebook"]), mmap_mode="r"
    )
    semantic_ids = np.load(
        os.path.join(source_dir, manifest["group_semantic_code_ids"])
    )
    decoded = decode_group_features(
        vocabulary, semantic_ids, int(manifest["semantic_invalid_id"])
    )
    point_ids = np.load(os.path.join(source_dir, manifest["point_group_ids"]))
    point_weights = np.load(
        os.path.join(source_dir, manifest["point_group_weights"])
    )
    invalid = int(manifest["invalid_id"])
    if not (
        0 <= args.fine_slot < point_ids.shape[1]
        and 0 <= args.micro_slot < point_ids.shape[1]
    ):
        raise ValueError("Fine or micro slot is outside the point table")
    fine_valid = point_ids[:, args.fine_slot] != invalid
    micro_valid = (
        (point_ids[:, args.micro_slot] != invalid)
        & (point_weights[:, args.micro_slot] > 0)
        & fine_valid
    )
    if not micro_valid.any():
        raise ValueError("A24 artifact has no resident micro IDs")

    consensus_path = os.path.abspath(args.l2_consensus)
    consensus = torch.load(consensus_path, map_location="cpu")
    point_features = consensus.get("initial_features")
    total_weights = consensus.get("total_weights")
    if point_features is None or point_features.shape != (
        point_ids.shape[0],
        vocabulary.shape[1],
    ):
        raise ValueError("L2 consensus features do not match the A24 artifact")
    if total_weights is None or total_weights.shape != (point_ids.shape[0],):
        raise ValueError("L2 consensus weights do not match the A24 artifact")

    keep_micro = np.zeros(point_ids.shape[0], dtype=bool)
    margins = np.zeros(point_ids.shape[0], dtype=np.float32)
    for start in range(0, point_ids.shape[0], args.chunk_size):
        end = min(start + args.chunk_size, point_ids.shape[0])
        local_valid = micro_valid[start:end]
        if not local_valid.any():
            continue
        local_rows = np.flatnonzero(local_valid)
        global_rows = start + local_rows
        fine_tokens = point_ids[global_rows, args.fine_slot].astype(np.int64)
        micro_tokens = point_ids[global_rows, args.micro_slot].astype(np.int64)
        if fine_tokens.max() >= decoded.shape[0] or micro_tokens.max() >= decoded.shape[0]:
            raise ValueError("Point tokens exceed the semantic table")
        features = point_features[global_rows].float().numpy()
        preferred, local_margin = prefer_micro(
            features, decoded[fine_tokens], decoded[micro_tokens]
        )
        supported = total_weights[global_rows].numpy() > 0.0
        preferred &= supported
        keep_micro[global_rows] = preferred
        margins[global_rows] = local_margin

    output_ids = point_ids.copy()
    output_weights = point_weights.copy()
    removed = micro_valid & ~keep_micro
    output_ids[removed, args.micro_slot] = invalid
    output_weights[removed, args.micro_slot] = 0

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for key in ("group_codebook", "group_semantic_code_ids", "group_reliability"):
        name = manifest[key]
        link(os.path.join(source_dir, name), os.path.join(output_dir, name))
    np.save(os.path.join(output_dir, "point_group_ids.npy"), output_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), output_weights)
    np.save(os.path.join(output_dir, "micro_support_margin.npy"), margins.astype(np.float16))

    output_manifest = dict(manifest)
    output_manifest.update(
        {
            "method": "point_supported_adaptive_micro_capacity",
            "point_group_ids": "point_group_ids.npy",
            "point_group_weights": "point_group_weights.npy",
            "adaptive_capacity": {
                "fine_slot": int(args.fine_slot),
                "micro_slot": int(args.micro_slot),
                "input_micro_points": int(micro_valid.sum()),
                "retained_micro_points": int(keep_micro.sum()),
                "retained_fraction": float(keep_micro.sum() / micro_valid.sum()),
                "mean_ids_per_covered_gaussian": float(
                    (output_weights > 0).sum()
                    / max(1, (output_weights > 0).any(axis=1).sum())
                ),
                "retained_margin": quantiles(margins[keep_micro]),
                "rejected_margin": quantiles(margins[removed]),
                "selection_rule": "cos(point_L2, micro_L2) > cos(point_L2, fine_L1)",
            },
            "module_codebook_contract": {
                **manifest.get("module_codebook_contract", {}),
                "enabled_modules": [
                    "A14_base",
                    "A18_part",
                    "A20_fine_part",
                    "A24_multiscale_micro_identity",
                    "A25_point_supported_capacity",
                ],
                "codebook_reuse_reason": (
                    "A25 only removes unsupported point-to-micro attachments; all "
                    "semantic targets, token IDs, and vocabulary rows remain A24"
                ),
            },
            "storage": {
                **manifest["storage"],
                "adaptive_point_table_bytes": int(
                    output_ids.nbytes + output_weights.nbytes + margins.astype(np.float16).nbytes
                ),
            },
            "source": {
                **manifest.get("source", {}),
                "adaptive_capacity_parent": source_dir,
                "l2_consensus": consensus_path,
                "leakage_control": (
                    "training-view signed L2 point consensus and fixed A20/A24 IDs only"
                ),
            },
            "args": vars(args),
        }
    )
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(output_manifest, output, indent=2)
    print(json.dumps(output_manifest["adaptive_capacity"], indent=2))


if __name__ == "__main__":
    main()
