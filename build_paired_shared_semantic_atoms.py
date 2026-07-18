#!/usr/bin/env python
"""Cluster paired split descriptors into shared atoms with neighbor exclusion."""

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


def paired_assign(index, anchors, competitors, topk):
    _, candidates = index.search(np.ascontiguousarray(anchors), topk)
    _, competitor_ids = index.search(np.ascontiguousarray(competitors), 1)
    competitor_ids = competitor_ids[:, 0]
    assignments = candidates[:, 0].copy()
    conflicts = assignments == competitor_ids
    for rank in range(1, topk):
        replace = conflicts & (candidates[:, rank] != competitor_ids)
        assignments[replace] = candidates[replace, rank]
        conflicts[replace] = False
    return assignments, competitor_ids, conflicts


def nearest_metrics(atoms, first, second, expected, competitor):
    index = faiss.IndexFlatIP(atoms.shape[1])
    index.add(np.ascontiguousarray(atoms))
    _, first_ids = index.search(np.ascontiguousarray(first), 1)
    _, second_ids = index.search(np.ascontiguousarray(second), 1)
    _, competitor_ids = index.search(np.ascontiguousarray(competitor), 1)
    first_ids = first_ids[:, 0]
    second_ids = second_ids[:, 0]
    competitor_ids = competitor_ids[:, 0]
    return {
        "physical_cross_split_same_code_fraction": 1.0,
        "retrieval_cross_split_same_code_fraction": float(
            (first_ids == second_ids).mean()
        ),
        "split0_assigned_atom_top1_fraction": float((first_ids == expected).mean()),
        "split1_assigned_atom_top1_fraction": float((second_ids == expected).mean()),
        "neighbor_assigned_atom_collision_fraction": float(
            (competitor_ids == expected).mean()
        ),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a21_artifact_dir", required=True)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_occupancy", type=float, default=1.5)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--assignment_topk", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260716)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.target_occupancy <= 1.0 or args.iterations <= 0 or args.assignment_topk < 2:
        raise ValueError("Paired atom parameters are invalid")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse paired shared semantic atoms: {output_dir}")
        return

    source_dir = os.path.abspath(args.a21_artifact_dir)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    vocabulary = np.load(
        os.path.join(source_dir, manifest["group_codebook"])
    ).astype(np.float16)
    semantic_ids = np.load(
        os.path.join(source_dir, manifest["group_semantic_code_ids"])
    ).astype(np.uint32)
    semantic_invalid = int(manifest["semantic_invalid_id"])
    group_features = decode_group_features(
        vocabulary.astype(np.float32), semantic_ids.astype(np.int64), semantic_invalid
    )
    point_ids = np.load(
        os.path.join(source_dir, manifest["point_group_ids"])
    ).astype(np.int64)
    point_invalid = int(manifest["invalid_id"])
    point_ids[point_ids == point_invalid] = -1
    point_competitor_ids = np.load(
        os.path.join(source_dir, manifest["point_competitor_ids"])
    ).astype(np.int64)
    competitor_invalid = int(manifest["competitor_invalid_id"])
    point_competitor_ids[point_competitor_ids == competitor_invalid] = -1
    point_weights = np.load(
        os.path.join(source_dir, manifest["point_group_weights"])
    )
    reliability = np.load(
        os.path.join(source_dir, manifest["group_reliability"])
    ).astype(np.float16)

    fine_count = int(manifest["modality_token_counts"]["fine"])
    competitor_count = int(manifest["modality_token_counts"]["competitor"])
    positive_token_count = semantic_ids.shape[0] - competitor_count
    fine_start = positive_token_count - fine_count
    fine_tokens = np.arange(fine_start, positive_token_count, dtype=np.int64)
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
        raise ValueError("Fine identities are not one-to-one with part groups")
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
            split_features[index],
            split_weights[index],
            local_groups,
            fine_count,
            args.device,
            args.chunk_size,
        )
        for index in range(2)
    ]
    first = F.normalize(split_results[0][0].float(), dim=-1).numpy()
    second = F.normalize(split_results[1][0].float(), dim=-1).numpy()
    anchors = F.normalize(
        split_results[0][0].float() * split_results[0][1].unsqueeze(-1)
        + split_results[1][0].float() * split_results[1][1].unsqueeze(-1),
        dim=-1,
    ).numpy()
    del fine_payload, split_features, split_weights

    competitor_token_by_fine = np.full(fine_count, -1, dtype=np.int64)
    local_point = point_fine - fine_start
    valid_fine_point = (local_point >= 0) & (local_point < fine_count)
    competitor_pairs = np.unique(
        np.stack(
            (
                local_point[valid_fine_point],
                point_competitor_ids[valid_fine_point, 1],
            ),
            axis=1,
        ),
        axis=0,
    )
    competitor_pairs = competitor_pairs[competitor_pairs[:, 1] >= 0]
    competitor_token_by_fine[competitor_pairs[:, 0]] = competitor_pairs[:, 1]
    competitor_valid = competitor_token_by_fine >= 0
    if not competitor_valid.all():
        # A tiny number of isolated groups have no boundary competitor; their own
        # anchor is neutral and exclusion is disabled by using an impossible code.
        safe = np.where(competitor_valid, competitor_token_by_fine, fine_tokens)
    else:
        safe = competitor_token_by_fine
    competitors = group_features[safe].astype(np.float32)

    num_atoms = max(2, int(math.ceil(fine_count / args.target_occupancy)))
    kmeans = faiss.Kmeans(
        anchors.shape[1],
        num_atoms,
        niter=args.iterations,
        nredo=1,
        spherical=True,
        seed=args.seed,
        gpu=args.device.startswith("cuda"),
        verbose=False,
    )
    kmeans.train(np.ascontiguousarray(anchors.astype(np.float32)))
    atoms = np.asarray(kmeans.centroids, dtype=np.float32)
    atoms /= np.maximum(np.linalg.norm(atoms, axis=-1, keepdims=True), 1e-8)
    index = faiss.IndexFlatIP(atoms.shape[1])
    index.add(np.ascontiguousarray(atoms))
    assignments, competitor_assignments, unresolved = paired_assign(
        index,
        anchors.astype(np.float32),
        competitors,
        min(args.assignment_topk, num_atoms),
    )
    unresolved &= competitor_valid

    # Re-estimate each atom from both split descriptors under the paired assignment.
    sums = np.zeros_like(atoms, dtype=np.float64)
    np.add.at(sums, assignments, first.astype(np.float64))
    np.add.at(sums, assignments, second.astype(np.float64))
    nonempty = np.linalg.norm(sums, axis=-1) > 0.0
    atoms[nonempty] = sums[nonempty].astype(np.float32)
    atoms /= np.maximum(np.linalg.norm(atoms, axis=-1, keepdims=True), 1e-8)
    diagnostics = nearest_metrics(
        atoms, first, second, assignments, competitors
    )
    diagnostics.update(
        {
            "num_identity_pairs": fine_count,
            "num_shared_atoms": num_atoms,
            "mean_identities_per_atom": float(fine_count / num_atoms),
            "neighbor_exclusion_unresolved_fraction": float(unresolved.mean()),
            "assignment_entropy_normalized": float(
                -np.sum(
                    (np.bincount(assignments, minlength=num_atoms) / fine_count)
                    * np.log(
                        np.maximum(
                            np.bincount(assignments, minlength=num_atoms) / fine_count,
                            1e-12,
                        )
                    )
                )
                / math.log(num_atoms)
            ),
        }
    )

    atom_offset = vocabulary.shape[0]
    updated_vocabulary = np.concatenate(
        (vocabulary, atoms.astype(np.float16)), axis=0
    )
    updated_semantic = semantic_ids.copy()
    updated_semantic[fine_tokens, :] = semantic_invalid
    updated_semantic[fine_tokens, 0] = (
        atom_offset + assignments
    ).astype(np.uint32)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), updated_vocabulary)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), updated_semantic)
    np.save(os.path.join(output_dir, "group_reliability.npy"), reliability)
    packed_point_ids = np.where(point_ids >= 0, point_ids, point_invalid).astype(np.uint32)
    packed_competitor_ids = np.where(
        point_competitor_ids >= 0, point_competitor_ids, competitor_invalid
    ).astype(np.uint32)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    np.save(os.path.join(output_dir, "point_competitor_ids.npy"), packed_competitor_ids)

    storage_bytes = int(
        updated_vocabulary.nbytes
        + updated_semantic.nbytes
        + reliability.nbytes
        + packed_point_ids.nbytes
        + point_weights.nbytes
        + packed_competitor_ids.nbytes
    )
    output_manifest = {
        **manifest,
        "format_version": 4,
        "method": "paired_assignment_shared_semantic_atoms",
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_competitor_ids": "point_competitor_ids.npy",
        "group_reliability": "group_reliability.npy",
        "paired_atom_training": diagnostics,
        "vocabulary": {
            **manifest.get("vocabulary", {}),
            "paired_shared_fine_atom_codes": num_atoms,
            "total_codes": int(updated_vocabulary.shape[0]),
            "construction": "A21 vocabulary plus paired shared atoms; every identity split pair is physically assigned to one code",
        },
        "continuous_discrete_contract": {
            "continuous_target": "paired split shared semantic atom",
            "discrete_encoding": "one exact shared atom row selected jointly by both splits",
            "reconstruction_cosine": 1.0,
            "ranking_gap_source": "FP16 roundoff only",
        },
        "module_codebook_contract": {
            **manifest.get("module_codebook_contract", {}),
            "enabled_modules": [
                "A14_base",
                "A18_part",
                "A20_fine_part",
                "A21_paired_shared_atom",
                "A21_competitor",
            ],
            "paired_assignment": True,
            "neighbor_same_atom_exclusion": True,
        },
        "storage": {
            "total_semantic_bytes": storage_bytes,
            "bytes_per_gaussian_amortized": float(
                storage_bytes / point_ids.shape[0]
            ),
        },
        "source": {
            "a21_artifact_dir": source_dir,
            "fine_consensus": os.path.abspath(args.fine_consensus),
            "leakage_control": "training split pairs and training-derived adjacent competitors only",
        },
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(output_manifest, output, indent=2)
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
