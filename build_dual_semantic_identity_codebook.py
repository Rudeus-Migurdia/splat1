#!/usr/bin/env python
"""Build shared semantic atoms plus exact per-identity fine codes."""

import json
import math
import os
import sys
from argparse import ArgumentParser

import faiss
import numpy as np
import torch
from torch.nn import functional as F

from build_hierarchical_group_semantic_codebook import aggregate_split_groups
from train_view_invariant_semantic_atoms import decode_group_features


def nearest(index, values):
    _, ids = index.search(np.ascontiguousarray(values.astype(np.float32)), 1)
    return ids[:, 0]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a20_artifact_dir", required=True)
    parser.add_argument("--a21_artifact_dir", required=True)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_occupancy", type=float, default=4.0)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.target_occupancy <= 1.0 or args.iterations <= 0:
        raise ValueError("Dual-code clustering parameters are invalid")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse dual semantic/identity codebook: {output_dir}")
        return

    a20_dir = os.path.abspath(args.a20_artifact_dir)
    a21_dir = os.path.abspath(args.a21_artifact_dir)
    with open(os.path.join(a20_dir, "manifest.json")) as source:
        a20 = json.load(source)
    with open(os.path.join(a21_dir, "manifest.json")) as source:
        a21 = json.load(source)
    if set(a21.get("vocabulary_modalities", [])) != {
        "base", "part", "fine", "competitor"
    }:
        raise ValueError("A22 requires the complete A21 competitor artifact")

    a20_vocabulary = np.load(
        os.path.join(a20_dir, a20["group_codebook"])
    ).astype(np.float16)
    vocabulary = np.load(
        os.path.join(a21_dir, a21["group_codebook"])
    ).astype(np.float16)
    semantic_ids = np.load(
        os.path.join(a21_dir, a21["group_semantic_code_ids"])
    ).astype(np.uint32)
    semantic_invalid = int(a21["semantic_invalid_id"])
    point_ids = np.load(
        os.path.join(a21_dir, a21["point_group_ids"])
    ).astype(np.int64)
    point_invalid = int(a21["invalid_id"])
    point_ids[point_ids == point_invalid] = -1
    point_weights = np.load(
        os.path.join(a21_dir, a21["point_group_weights"])
    )
    point_competitor_ids = np.load(
        os.path.join(a21_dir, a21["point_competitor_ids"])
    )
    reliability = np.load(
        os.path.join(a21_dir, a21["group_reliability"])
    ).astype(np.float16)

    fine_count = int(a20["modality_token_counts"]["fine"])
    competitor_count = int(a21["modality_token_counts"]["competitor"])
    positive_token_count = semantic_ids.shape[0] - competitor_count
    fine_start = positive_token_count - fine_count
    fine_tokens = np.arange(fine_start, positive_token_count, dtype=np.int64)
    fine_semantic_rows = semantic_ids[fine_tokens, 0].astype(np.int64)
    if (fine_semantic_rows == semantic_invalid).any():
        raise ValueError("A20 fine identity codes must be exact first-slot rows")
    if int(fine_semantic_rows.max()) >= a20_vocabulary.shape[0]:
        raise ValueError("A20 fine identity rows exceed its vocabulary")
    # Restore the exact A20 identity channel; A21's unique-atom edits are not used.
    vocabulary[fine_semantic_rows] = a20_vocabulary[fine_semantic_rows]

    point_fine = point_ids[:, 1]
    fine_points = point_fine >= fine_start
    fine_to_part = np.full(fine_count, -1, dtype=np.int64)
    pairs = np.unique(
        np.stack(
            (point_fine[fine_points] - fine_start, point_ids[fine_points, 0]),
            axis=1,
        ),
        axis=0,
    )
    counts = np.bincount(pairs[:, 0], minlength=fine_count)
    if pairs.shape[0] != fine_count or (counts != 1).any():
        raise ValueError("Fine identities must map one-to-one to part groups")
    fine_to_part[pairs[:, 0]] = pairs[:, 1]
    part_to_local = np.full(semantic_ids.shape[0], -1, dtype=np.int64)
    part_to_local[fine_to_part] = np.arange(fine_count)
    local_groups = np.full(point_ids.shape[0], -1, dtype=np.int64)
    valid_part = point_ids[:, 0] >= 0
    local_groups[valid_part] = part_to_local[point_ids[valid_part, 0]]

    fine_payload = torch.load(args.fine_consensus, map_location="cpu")
    split_features = fine_payload["split_initial_features"].detach().cpu()
    split_weights = fine_payload["split_weights"].detach().cpu()
    split_results = [
        aggregate_split_groups(
            split_features[index], split_weights[index], local_groups,
            fine_count, args.device, args.chunk_size
        )
        for index in range(2)
    ]
    first = F.normalize(split_results[0][0].float(), dim=-1).numpy()
    second = F.normalize(split_results[1][0].float(), dim=-1).numpy()
    paired = F.normalize(
        split_results[0][0].float() * split_results[0][1].unsqueeze(-1)
        + split_results[1][0].float() * split_results[1][1].unsqueeze(-1),
        dim=-1,
    ).numpy()
    del fine_payload, split_features, split_weights

    num_atoms = max(2, int(math.ceil(fine_count / args.target_occupancy)))
    kmeans = faiss.Kmeans(
        paired.shape[1], num_atoms, niter=args.iterations, nredo=1,
        spherical=True, seed=args.seed, gpu=args.device.startswith("cuda"),
        verbose=False,
    )
    kmeans.train(np.ascontiguousarray(paired.astype(np.float32)))
    atoms = np.asarray(kmeans.centroids, dtype=np.float32)
    atoms /= np.maximum(np.linalg.norm(atoms, axis=-1, keepdims=True), 1e-8)
    index = faiss.IndexFlatIP(atoms.shape[1])
    index.add(np.ascontiguousarray(atoms))
    assignments = nearest(index, paired)
    first_ids = nearest(index, first)
    second_ids = nearest(index, second)

    atom_offset = vocabulary.shape[0]
    vocabulary = np.concatenate((vocabulary, atoms.astype(np.float16)), axis=0)
    atom_ids = np.full(semantic_ids.shape, semantic_invalid, dtype=np.uint32)
    atom_ids[fine_tokens, 0] = (atom_offset + assignments).astype(np.uint32)
    identity_features = decode_group_features(
        vocabulary.astype(np.float32), semantic_ids.astype(np.int64), semantic_invalid
    )[fine_tokens]
    atom_features = atoms[assignments]
    identity_atom_cosine = np.sum(identity_features * atom_features, axis=-1)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), vocabulary)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_semantic_atom_code_ids.npy"), atom_ids)
    np.save(os.path.join(output_dir, "group_reliability.npy"), reliability)
    np.save(
        os.path.join(output_dir, "point_group_ids.npy"),
        np.where(point_ids >= 0, point_ids, point_invalid).astype(np.uint32),
    )
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    np.save(os.path.join(output_dir, "point_competitor_ids.npy"), point_competitor_ids)

    storage_bytes = sum(
        os.path.getsize(os.path.join(output_dir, name))
        for name in (
            "shared_vocabulary.npy", "group_semantic_code_ids.npy",
            "group_semantic_atom_code_ids.npy", "group_reliability.npy",
            "point_group_ids.npy", "point_group_weights.npy",
            "point_competitor_ids.npy",
        )
    )
    manifest = {
        **a21,
        "format_version": 5,
        "method": "shared_semantic_atom_plus_identity_code",
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "group_semantic_atom_code_ids": "group_semantic_atom_code_ids.npy",
        "semantic_atom_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_competitor_ids": "point_competitor_ids.npy",
        "group_reliability": "group_reliability.npy",
        "vocabulary_modalities": [
            "base", "part", "fine", "competitor", "semantic_atom"
        ],
        "modality_token_counts": {
            **a21["modality_token_counts"],
            "semantic_atom": num_atoms,
        },
        "vocabulary": {
            **a21.get("vocabulary", {}),
            "shared_semantic_atom_codes": num_atoms,
            "total_codes": int(vocabulary.shape[0]),
            "construction": "A20 exact identity codes plus paired shared semantic atoms and A21 exact competitors",
        },
        "dual_code_training": {
            "num_fine_identities": fine_count,
            "num_shared_semantic_atoms": num_atoms,
            "mean_identities_per_atom": float(fine_count / num_atoms),
            "physical_cross_split_same_atom_fraction": 1.0,
            "retrieval_cross_split_same_atom_fraction": float(
                (first_ids == second_ids).mean()
            ),
            "split0_assigned_atom_top1_fraction": float(
                (first_ids == assignments).mean()
            ),
            "split1_assigned_atom_top1_fraction": float(
                (second_ids == assignments).mean()
            ),
            "mean_identity_atom_cosine": float(identity_atom_cosine.mean()),
            "minimum_identity_atom_cosine": float(identity_atom_cosine.min()),
        },
        "continuous_discrete_contract": {
            "semantic_atom_target": "paired split cluster centroid",
            "identity_target": "exact A20 fine identity feature",
            "discrete_encoding": "one exact shared atom row plus one exact identity row",
            "reconstruction_gap": "FP16 roundoff only for each independent channel",
        },
        "module_codebook_contract": {
            "enabled_modules": [
                "A14_base", "A18_part", "A20_fine_identity",
                "A22_shared_semantic_atom", "A22_identity_code",
                "A21_competitor",
            ],
            "readout_slots": ["part", "fine"],
            "dual_code_slot": "fine",
        },
        "storage": {
            "total_semantic_bytes": int(storage_bytes),
            "bytes_per_gaussian_amortized": float(
                storage_bytes / point_ids.shape[0]
            ),
        },
        "source": {
            "a20_artifact_dir": a20_dir,
            "a21_artifact_dir": a21_dir,
            "fine_consensus": os.path.abspath(args.fine_consensus),
            "leakage_control": "training split pairs and training-derived identities/competitors only",
        },
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
