#!/usr/bin/env python
"""Build top-2 Group supports for query-conditioned score-space regularization."""

import argparse
import json
import os
import time

import numpy as np

from build_full_group_addressed_memory import descriptor_consistency, file_sha256, quantiles
from build_geometry_conditioned_tracklet_partition import (
    build_atom_contact_graph,
    load_atom_geometry,
)
from build_group_addressed_spatial_memory_audit import (
    bounded_group_profiles,
    build_ring_descriptors,
    load_ring_views,
    prepare_ring_incidence,
    signed_group_profiles,
)
from build_multi_hypothesis_entity_tomography import load_prepared_views, normalize_rows
from build_persistent_entity_tomography import fit_persistent_slots, incidence_entries


def top2_group_support(profiles, levels, minimum_membership):
    profiles = np.asarray(profiles, dtype=np.float32)
    levels = np.asarray(levels, dtype=np.int64)
    atom_count = profiles.shape[1]
    ids = np.full((atom_count, 4, 2), -1, dtype=np.int32)
    memberships = np.zeros((atom_count, 4, 2), dtype=np.float32)
    entropy = np.ones((atom_count, 4), dtype=np.float32)
    for level in range(4):
        groups = np.flatnonzero(levels == level)
        if not groups.size:
            raise RuntimeError(f"No persistent Groups survived at level {level}")
        values = profiles[groups]
        count = min(2, len(groups))
        order = np.argsort(-values, axis=0, kind="stable")[:count]
        selected = np.take_along_axis(values, order, axis=0).T
        selected_ids = groups[order].T
        valid = selected >= minimum_membership
        ids[:, level, :count] = np.where(valid, selected_ids, -1)
        memberships[:, level, :count] = np.where(valid, selected, 0.0)
        pair = memberships[:, level]
        normalized = pair / pair.sum(axis=1, keepdims=True).clip(min=1e-8)
        local_entropy = -(normalized * np.log(normalized.clip(min=1e-8))).sum(axis=1)
        entropy[:, level] = np.where(
            pair[:, 1] > 0.0, local_entropy / np.log(2.0), 0.0
        )
    return ids, memberships, entropy


def packed_neighbors(contact_graph, maximum_neighbors):
    graph = contact_graph.tocsr()
    output = np.full((graph.shape[0], maximum_neighbors), -1, dtype=np.int32)
    weights = np.zeros((graph.shape[0], maximum_neighbors), dtype=np.float32)
    for atom in range(graph.shape[0]):
        start, end = graph.indptr[atom], graph.indptr[atom + 1]
        columns = graph.indices[start:end]
        values = graph.data[start:end]
        order = np.argsort(-values, kind="stable")[:maximum_neighbors]
        output[atom, : len(order)] = columns[order]
        weights[atom, : len(order)] = values[order]
    return output, weights


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a47_audit_dir", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--coverage_threshold", type=float, default=0.30)
    parser.add_argument("--minimum_spatial_jaccard", type=float, default=0.35)
    parser.add_argument("--minimum_semantic_cosine", type=float, default=0.75)
    parser.add_argument("--minimum_association", type=float, default=0.40)
    parser.add_argument("--spatial_weight", type=float, default=0.85)
    parser.add_argument("--temporal_neighbors", type=int, default=2)
    parser.add_argument("--minimum_persistence_views", type=int, default=3)
    parser.add_argument("--minimum_visibility", type=float, default=1e-4)
    parser.add_argument("--merge_jaccard", type=float, default=0.85)
    parser.add_argument("--merge_semantic_cosine", type=float, default=0.90)
    parser.add_argument("--maximum_slots", type=int, default=4096)
    parser.add_argument("--atom_neighbors", type=int, default=8)
    parser.add_argument("--minimum_atom_contact", type=float, default=0.05)
    parser.add_argument("--ring_radius_scale", type=float, default=0.02)
    parser.add_argument("--minimum_ring_radius", type=int, default=3)
    parser.add_argument("--maximum_ring_radius", type=int, default=15)
    parser.add_argument("--exterior_evidence_weight", type=float, default=1.0)
    parser.add_argument("--signed_evidence_epsilon", type=float, default=1e-6)
    parser.add_argument("--core_coverage_threshold", type=float, default=0.30)
    parser.add_argument("--boundary_coverage_threshold", type=float, default=0.05)
    parser.add_argument("--minimum_core_atoms", type=int, default=1)
    parser.add_argument("--maximum_ring_neighbors", type=int, default=8)
    parser.add_argument("--maximum_ring_neighbor_jaccard", type=float, default=0.20)
    parser.add_argument("--minimum_owner_membership", type=float, default=0.02)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse query-conditioned spatial posterior: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    a47_manifest_path = os.path.join(args.a47_audit_dir, "manifest.json")
    with open(a47_manifest_path) as source:
        a47_manifest = json.load(source)
    contract = a47_manifest["source_contract"]
    if not (
        contract["training_views_only"]
        and contract["raw_overlapping_proposals"]
        and contract["raw_top45_talpha"]
        and not contract["evaluation_queries_or_labels_used"]
        and not contract["codebooks_trained"]
        and int(contract["fixed_seed"]) == args.seed
    ):
        raise ValueError("A47 source contract is incompatible with A52")
    if file_sha256(args.geometry_checkpoint) != a47_manifest["inputs"]["geometry_checkpoint_sha256"]:
        raise ValueError("Geometry checkpoint does not match A47 atom ownership")

    entries = incidence_entries(args.a47_audit_dir)
    views = load_prepared_views(args.a47_audit_dir, entries)
    atom_ids_path = os.path.join(args.a47_audit_dir, "gaussian_atom_ids.npy")
    gaussian_atom_ids = np.load(atom_ids_path).astype(np.int64)
    atom_geometry = load_atom_geometry(args.geometry_checkpoint, atom_ids_path)
    contact_graph = build_atom_contact_graph(atom_geometry, args.atom_neighbors)
    ring_dir = os.path.join(output_dir, "ring_incidence_views")
    ring_manifest = prepare_ring_incidence(
        a47_manifest, args.a47_audit_dir, gaussian_atom_ids, ring_dir, args
    )
    ring_views = load_ring_views(ring_dir, ring_manifest)
    persistent = fit_persistent_slots(views, None, args)
    signed, signed_stats = signed_group_profiles(persistent, views, ring_views, args)
    bounded, core_masks, support_masks, adjacency, bounded_stats = bounded_group_profiles(
        signed, contact_graph, args
    )
    ring_keys, ring_valid, _, ring_stats = build_ring_descriptors(
        persistent, support_masks, adjacency, args
    )
    consistency = descriptor_consistency(persistent, views)
    atom_group_ids, atom_memberships, atom_entropy = top2_group_support(
        bounded, persistent["levels"], args.minimum_owner_membership
    )
    neighbor_ids, neighbor_weights = packed_neighbors(adjacency, args.atom_neighbors)

    arrays = {
        "group_core_keys.npy": normalize_rows(persistent["descriptors"]).astype(np.float16),
        "group_ring_keys.npy": ring_keys.astype(np.float16),
        "group_ring_valid.npy": ring_valid,
        "group_level.npy": persistent["levels"].astype(np.uint8),
        "group_reliability.npy": consistency.astype(np.float16),
        "point_group_ids.npy": atom_group_ids[gaussian_atom_ids].astype(np.int32),
        "point_group_memberships.npy": atom_memberships[gaussian_atom_ids].astype(np.float16),
        "point_group_entropy.npy": atom_entropy[gaussian_atom_ids].astype(np.float16),
        "gaussian_atom_ids.npy": gaussian_atom_ids.astype(np.int32),
        "atom_neighbor_ids.npy": neighbor_ids,
        "atom_neighbor_weights.npy": neighbor_weights.astype(np.float16),
    }
    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array)
    valid = arrays["point_group_ids.npy"] >= 0
    manifest = {
        "format_version": 1,
        "representation": "query_conditioned_top2_spatial_group_posterior",
        "method": "decoupled_semantic_retrieval_and_spatial_posterior",
        "scene": a47_manifest["scene"],
        "seed": args.seed,
        "num_gaussians": int(len(gaussian_atom_ids)),
        "num_atoms": int(bounded.shape[1]),
        "num_groups": int(len(bounded)),
        "feature_dim": int(persistent["descriptors"].shape[1]),
        "top_groups_per_level": 2,
        "group_core_keys": "group_core_keys.npy",
        "group_ring_keys": "group_ring_keys.npy",
        "group_ring_valid": "group_ring_valid.npy",
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_memberships": "point_group_memberships.npy",
        "point_group_entropy": "point_group_entropy.npy",
        "gaussian_atom_ids": "gaussian_atom_ids.npy",
        "atom_neighbor_ids": "atom_neighbor_ids.npy",
        "atom_neighbor_weights": "atom_neighbor_weights.npy",
        "coverage": {
            "any_level_fraction": float(valid.any(axis=(1, 2)).mean()),
            "per_level_fraction": [float(valid[:, level].any(axis=1).mean()) for level in range(4)],
            "mean_groups_per_level": [float(valid[:, level].sum(axis=1).mean()) for level in range(4)],
            "entropy_quantiles": quantiles(arrays["point_group_entropy.npy"]),
        },
        "statistics": {
            **persistent["statistics"],
            **signed_stats,
            **bounded_stats,
            **ring_stats,
        },
        "source_contract": {
            "semantic_tokens_not_modified": True,
            "spatial_posterior_applied_after_semantic_retrieval": True,
            "top2_overlapping_groups": True,
            "core_boundary_ring_evidence": True,
            "atom_geodesic_neighbors": True,
            "evaluation_queries_or_labels_used": False,
            "fixed_seed": args.seed,
        },
        "source": {
            "a47_audit_dir": os.path.abspath(args.a47_audit_dir),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        },
        "storage_bytes": int(sum(array.nbytes for array in arrays.values())),
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
