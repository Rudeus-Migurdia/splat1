#!/usr/bin/env python
"""Build a four-level semantic memory with explicit spatial Group addresses."""

import argparse
import hashlib
import json
import os
import time

import numpy as np

from build_group_addressed_spatial_memory_audit import (
    bounded_group_profiles,
    build_ring_descriptors,
    load_ring_views,
    prepare_ring_incidence,
    signed_group_profiles,
)
from build_multi_hypothesis_entity_tomography import load_prepared_views, normalize_rows
from build_persistent_entity_tomography import (
    fit_persistent_slots,
    incidence_entries,
)
from build_geometry_conditioned_tracklet_partition import (
    build_atom_contact_graph,
    load_atom_geometry,
)
from build_seeded_hierarchical_resident_memory import (
    quantize_chord_error_upper_bound,
    train_level_codebook,
)


LEVEL_NAMES = ("sam_l0", "sam_l1", "sam_l2", "sam_l3")
LEVEL_ROLES = ("coarse", "object", "part", "micro")


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantiles(values):
    values = np.asarray(values)
    if not values.size:
        return {str(value): 0.0 for value in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)}
    return {
        str(value): float(np.quantile(values, value))
        for value in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def descriptor_consistency(model, views):
    """Measure within-track semantic agreement without labels or text queries."""
    view_map = {int(view["view_index"]): view for view in views}
    consistency = np.zeros(len(model["profiles"]), dtype=np.float32)
    for group_id, members in enumerate(model["members"]):
        descriptor = model["descriptors"][group_id]
        scores = []
        weights = []
        for member in members:
            view = view_map[int(member["view_index"])]
            proposal_id = int(member["proposal_index"])
            scores.append(float(view["descriptors"][proposal_id] @ descriptor))
            weights.append(max(float(view["quality"][proposal_id]), 1e-4))
        consistency[group_id] = np.average(scores, weights=weights) if scores else 0.0
    return np.clip((consistency - 0.5) / 0.5, 0.0, 1.0)


def aggregate_reference_memory(reference_dir, decomposition, gaussian_atom_ids):
    """Aggregate a proven resident memory into each unique spatial Group."""
    from scipy.sparse import csr_matrix

    with open(os.path.join(reference_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError("Reference semantics must use independent hierarchical codebooks")
    point_ids = np.load(
        os.path.join(reference_dir, manifest["point_group_ids"]), mmap_mode="r"
    )
    semantic_ids = np.load(
        os.path.join(reference_dir, manifest["group_semantic_code_ids"]), mmap_mode="r"
    )[:, 0].astype(np.int64)
    group_levels = np.load(
        os.path.join(reference_dir, manifest["group_level"]), mmap_mode="r"
    ).astype(np.int64)
    spatial_point_ids = decomposition["atom_group_ids"][gaussian_atom_ids]
    spatial_weights = (
        decomposition["atom_membership"][gaussian_atom_ids]
        * decomposition["atom_reliability"][gaussian_atom_ids]
    )
    group_count = len(decomposition["levels"])
    keys = np.zeros((group_count, int(manifest["feature_dim"])), dtype=np.float32)
    consistency = np.zeros(group_count, dtype=np.float32)
    valid_groups = np.zeros(group_count, dtype=bool)
    level_specs = {int(item["level"]): item for item in manifest["level_codebooks"]}
    for level in range(4):
        codebook = np.load(
            os.path.join(reference_dir, level_specs[level]["codebook"])
        ).astype(np.float32)
        codebook = normalize_rows(codebook)
        reference_tokens = np.asarray(point_ids[:, level], dtype=np.int64)
        spatial_groups = spatial_point_ids[:, level]
        weights = spatial_weights[:, level]
        valid = (
            (spatial_groups >= 0)
            & (reference_tokens >= 0)
            & (reference_tokens < len(semantic_ids))
            & (group_levels[np.clip(reference_tokens, 0, len(group_levels) - 1)] == level)
            & (weights > 0.0)
        )
        local_codes = semantic_ids[reference_tokens[valid]]
        if np.any(local_codes < 0) or np.any(local_codes >= len(codebook)):
            raise ValueError(f"Reference semantic IDs exceed level {level} codebook")
        histogram = csr_matrix(
            (
                weights[valid],
                (spatial_groups[valid], local_codes),
            ),
            shape=(group_count, len(codebook)),
            dtype=np.float32,
        )
        sums = np.asarray(histogram @ codebook, dtype=np.float32)
        masses = np.asarray(histogram.sum(axis=1)).reshape(-1).astype(np.float32)
        groups = np.flatnonzero((decomposition["levels"] == level) & (masses > 0.0))
        norms = np.linalg.norm(sums[groups], axis=1)
        keys[groups] = sums[groups] / np.maximum(norms[:, None], 1e-8)
        consistency[groups] = norms / np.maximum(masses[groups], 1e-8)
        valid_groups[groups] = True
    return keys, np.clip(consistency, 0.0, 1.0), valid_groups, manifest


def add_split_consistency_keys(
    decomposition, reference_dir, gaussian_atom_ids, adjacency, args
):
    reference_keys, reference_consistency, reference_valid, reference_manifest = (
        aggregate_reference_memory(reference_dir, decomposition, gaussian_atom_ids)
    )
    proposal_keys = decomposition["core_keys"]
    proposal_consistency = decomposition["consistency"]
    agreement = np.sum(proposal_keys * reference_keys, axis=1)
    select_reference = (
        reference_valid
        & (reference_consistency + args.source_consistency_margin >= proposal_consistency)
    )
    gated_keys = proposal_keys.copy()
    gated_keys[select_reference] = reference_keys[select_reference]
    gated_consistency = proposal_consistency.copy()
    gated_consistency[select_reference] = reference_consistency[select_reference]
    gated_model = dict(decomposition)
    gated_model["descriptors"] = gated_keys
    gated_ring, gated_ring_valid, _, gated_ring_stats = build_ring_descriptors(
        gated_model, decomposition["support_masks"], adjacency, args
    )
    gated_residual = gated_keys.copy()
    usable = gated_ring_valid & (np.linalg.norm(gated_ring, axis=1) > 0.0)
    gated_residual[usable] = normalize_rows(
        gated_keys[usable] - args.exterior_semantic_weight * gated_ring[usable]
    )
    result = dict(decomposition)
    result.update(
        {
            "gated_core_keys": gated_keys,
            "gated_residual_keys": gated_residual,
            "gated_core_keys_reliability": gated_consistency,
            "gated_residual_keys_reliability": gated_consistency,
            "reference_keys": reference_keys,
            "reference_consistency": reference_consistency,
            "reference_valid": reference_valid,
            "source_agreement": agreement,
            "selected_reference": select_reference,
            "statistics": {
                **decomposition["statistics"],
                "split_consistency_gate": {
                    "selected_reference_groups": int(select_reference.sum()),
                    "selected_reference_fraction": float(select_reference.mean()),
                    "reference_valid_fraction": float(reference_valid.mean()),
                    "proposal_consistency_quantiles": quantiles(proposal_consistency),
                    "reference_consistency_quantiles": quantiles(
                        reference_consistency[reference_valid]
                    ),
                    "source_agreement_quantiles": quantiles(agreement[reference_valid]),
                    "reference_method": reference_manifest.get("method"),
                    "evaluation_queries_or_labels_used": False,
                    **gated_ring_stats,
                },
            },
        }
    )
    return result


def decompose_overlapping_groups(model, views, ring_views, contact_graph, args):
    """Turn overlapping persistent supports into one addressed Group per level/atom."""
    signed, signed_stats = signed_group_profiles(model, views, ring_views, args)
    bounded, core, support, adjacency, profile_stats = bounded_group_profiles(
        signed, contact_graph, args
    )
    ring_descriptors, ring_valid, ring_masks, ring_stats = build_ring_descriptors(
        model, support, adjacency, args
    )
    consistency = descriptor_consistency(model, views)
    core_keys = normalize_rows(model["descriptors"].astype(np.float32))
    residual_keys = core_keys.copy()
    usable_ring = ring_valid & (np.linalg.norm(ring_descriptors, axis=1) > 0.0)
    residual_keys[usable_ring] = normalize_rows(
        core_keys[usable_ring]
        - args.exterior_semantic_weight * ring_descriptors[usable_ring]
    )

    group_count, atom_count = bounded.shape
    atom_group_ids = np.full((atom_count, 4), -1, dtype=np.int32)
    atom_membership = np.zeros((atom_count, 4), dtype=np.float32)
    atom_reliability = np.zeros((atom_count, 4), dtype=np.float32)
    atom_entropy = np.ones((atom_count, 4), dtype=np.float32)
    boundary = np.zeros((atom_count, 4), dtype=bool)
    for level in range(4):
        groups = np.flatnonzero(model["levels"] == level)
        if not groups.size:
            raise RuntimeError(f"No persistent Groups survived at level {level}")
        values = bounded[groups]
        order = np.argsort(-values, axis=0, kind="stable")
        best_local = order[0]
        best = values[best_local, np.arange(atom_count)]
        second = (
            values[order[1], np.arange(atom_count)]
            if len(groups) > 1
            else np.zeros(atom_count, dtype=np.float32)
        )
        valid = best >= args.minimum_owner_membership
        margin = np.divide(
            best - second,
            np.maximum(best, 1e-8),
            out=np.zeros_like(best),
            where=best > 0.0,
        )
        atom_group_ids[valid, level] = groups[best_local[valid]]
        atom_membership[valid, level] = best[valid]
        atom_reliability[valid, level] = (
            consistency[groups[best_local[valid]]]
            * (args.boundary_reliability_floor
               + (1.0 - args.boundary_reliability_floor) * margin[valid])
        )
        boundary[valid, level] = margin[valid] < args.boundary_margin
        pair = np.stack([best, second], axis=1)
        pair /= pair.sum(axis=1, keepdims=True).clip(min=1e-8)
        entropy = -(pair * np.log(pair.clip(min=1e-8))).sum(axis=1) / np.log(2.0)
        atom_entropy[valid, level] = entropy[valid]

    owned_atoms = np.bincount(
        atom_group_ids[atom_group_ids >= 0], minlength=group_count
    ).astype(np.int32)
    return {
        **model,
        "profiles": bounded,
        "core_masks": core,
        "support_masks": support,
        "ring_masks": ring_masks,
        "ring_descriptors": ring_descriptors,
        "ring_valid": ring_valid,
        "core_keys": core_keys,
        "residual_keys": residual_keys,
        "consistency": consistency,
        "atom_group_ids": atom_group_ids,
        "atom_membership": atom_membership,
        "atom_reliability": atom_reliability,
        "atom_entropy": atom_entropy,
        "atom_boundary": boundary,
        "owned_atoms": owned_atoms,
        "statistics": {
            **model["statistics"],
            **signed_stats,
            **profile_stats,
            **ring_stats,
            "covered_atom_fraction_by_level": [
                float((atom_group_ids[:, level] >= 0).mean()) for level in range(4)
            ],
            "boundary_atom_fraction_by_level": [
                float(boundary[:, level].mean()) for level in range(4)
            ],
            "semantic_consistency_quantiles": quantiles(consistency),
            "groups_with_owned_atoms": int((owned_atoms > 0).sum()),
        },
    }


def write_memory(output_dir, decomposition, gaussian_atom_ids, key_name, args):
    os.makedirs(output_dir, exist_ok=True)
    group_keys = decomposition[key_name]
    levels = decomposition["levels"].astype(np.uint8)
    group_count = len(levels)
    group_sizes = np.bincount(
        decomposition["atom_group_ids"][decomposition["atom_group_ids"] >= 0],
        minlength=group_count,
    ).astype(np.int32)
    key_reliability = decomposition.get(
        f"{key_name}_reliability", decomposition["consistency"]
    )
    group_reliability = np.clip(
        key_reliability
        * np.minimum(decomposition["support_views"] / args.reliable_support_views, 1.0),
        args.minimum_group_reliability,
        1.0,
    ).astype(np.float32)

    semantic_ids = np.zeros((group_count, 1), dtype=np.int64)
    quantization_error = np.zeros(group_count, dtype=np.float32)
    level_specs = []
    codebooks = []
    training = []
    for level in range(4):
        groups = np.flatnonzero(levels == level)
        requested = min(int(args.codes_per_level[level]), len(groups))
        codebook, local_ids, errors, stats = train_level_codebook(
            group_keys[groups],
            group_reliability[groups],
            np.maximum(group_sizes[groups], 1),
            1,
            0.0,
            requested,
            len(groups),
            args.kmeans_iterations,
            args.seed + 1009 * (level + 1),
            args.faiss_gpu,
            args.assignment_chunk_size,
        )
        semantic_ids[groups, 0] = local_ids
        quantization_error[groups] = errors
        filename = f"{LEVEL_NAMES[level]}_codebook.npy"
        np.save(os.path.join(output_dir, filename), codebook)
        codebooks.append(codebook)
        stats.update(
            {
                "level": level,
                "name": LEVEL_NAMES[level],
                "num_spatial_groups": int(len(groups)),
                "owned_groups": int((group_sizes[groups] > 0).sum()),
            }
        )
        training.append(stats)
        level_specs.append(
            {
                "name": LEVEL_NAMES[level],
                "level": level,
                "semantic_role": LEVEL_ROLES[level],
                "codebook": filename,
                "num_codes": int(codebook.shape[0]),
                "quantization": "fresh_seeded_spherical_kmeans_on_unique_spatial_groups",
            }
        )

    point_ids = decomposition["atom_group_ids"][gaussian_atom_ids]
    point_membership = decomposition["atom_membership"][gaussian_atom_ids]
    point_reliability = decomposition["atom_reliability"][gaussian_atom_ids]
    point_entropy = decomposition["atom_entropy"][gaussian_atom_ids]
    valid = point_ids >= 0
    id_dtype = np.uint16 if group_count < np.iinfo(np.uint16).max else np.uint32
    invalid_id = int(np.iinfo(id_dtype).max)
    stored_ids = np.where(valid, point_ids, invalid_id).astype(id_dtype)
    point_weights = np.rint(np.clip(point_membership, 0.0, 1.0) * 255).astype(np.uint8)
    packed_entropy = np.rint(np.clip(point_entropy, 0.0, 1.0) * 255).astype(np.uint8)
    point_quantization_error = np.zeros_like(point_membership, dtype=np.float32)
    point_quantization_error[valid] = quantization_error[point_ids[valid]]
    packed_error = quantize_chord_error_upper_bound(point_quantization_error)
    semantic_dtype = np.uint16 if max(len(codebook) for codebook in codebooks) < np.iinfo(np.uint16).max else np.uint32
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    semantic_ids = semantic_ids.astype(semantic_dtype)

    arrays = {
        "group_semantic_code_ids.npy": semantic_ids,
        "group_level.npy": levels,
        "group_reliability.npy": group_reliability.astype(np.float16),
        "point_group_ids.npy": stored_ids,
        "point_group_weights.npy": point_weights,
        "point_group_reliability.npy": point_reliability.astype(np.float16),
        "point_group_entropy.npy": packed_entropy,
        "point_group_quantization_error.npy": packed_error,
        "atom_group_ids.npy": decomposition["atom_group_ids"].astype(np.int32),
        "atom_boundary.npy": decomposition["atom_boundary"],
        "group_core_keys.npy": decomposition["core_keys"].astype(np.float16),
        "group_ring_keys.npy": decomposition["ring_descriptors"].astype(np.float16),
        "group_residual_keys.npy": decomposition["residual_keys"].astype(np.float16),
    }
    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array)
    semantic_bytes = sum(array.nbytes for array in arrays.values()) + sum(
        codebook.nbytes for codebook in codebooks
    )
    covered = valid.any(axis=1)
    manifest = {
        "format_version": 1,
        "representation": "hierarchical_independent_group_codebooks",
        "method": "full_group_addressed_hierarchical_semantic_memory",
        "variant": key_name,
        "num_gaussians": int(len(gaussian_atom_ids)),
        "num_spatial_groups": int(group_count),
        "feature_dim": int(group_keys.shape[1]),
        "top_m": 4,
        "resident_slots_required": 4,
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_reliability": "point_group_reliability.npy",
        "point_group_entropy": "point_group_entropy.npy",
        "point_group_quantization_error": "point_group_quantization_error.npy",
        "point_group_quantization_error_scale": 2.0 / 255.0,
        "invalid_id": invalid_id,
        "id_dtype": str(stored_ids.dtype),
        "weight_dtype": "uint8_group_membership",
        "level_codebooks": level_specs,
        "covered_fraction": float(covered.mean()),
        "mean_ids_per_covered_gaussian": float(valid[covered].sum(axis=1).mean()) if covered.any() else 0.0,
        "usable_slot_fraction": float((valid & (point_reliability > 0.0)).mean()),
        "usable_covered_fraction": float((valid & (point_reliability > 0.0)).any(axis=1).mean()),
        "group_addressing": {
            "address": "unique persistent spatial Group ID",
            "semantic_value": "per-level freshly trained code ID",
            "overlap_resolution": "winner-take-all partition per level with top-two boundary entropy",
            "decode": "query Group score is expanded only to Gaussians owned by that Group",
            "peer_levels": True,
            "training": training,
            "statistics": decomposition["statistics"],
        },
        "storage": {
            "total_semantic_bytes": int(semantic_bytes),
            "bytes_per_gaussian_amortized": float(semantic_bytes / len(gaussian_atom_ids)),
            "shared_vocabulary_bytes_unique": int(sum(codebook.nbytes for codebook in codebooks)),
        },
        "reproducibility": {"seed": int(args.seed), "fixed": True},
        "leakage_control": {
            "evaluation_queries_or_labels_used": False,
            "training_views_only": True,
        },
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def write_composite_refinement_memory(
    output_dir,
    decomposition,
    gaussian_atom_ids,
    reference_dir,
    parent_key_name,
    group_key_weight,
    args,
):
    """Create unique (spatial Group, Gaussian token) addresses and fresh codebooks."""
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(reference_dir, "manifest.json")) as source:
        reference_manifest = json.load(source)
    reference_point_ids = np.load(
        os.path.join(reference_dir, reference_manifest["point_group_ids"]), mmap_mode="r"
    )
    reference_semantic_ids = np.load(
        os.path.join(reference_dir, reference_manifest["group_semantic_code_ids"]),
        mmap_mode="r",
    )[:, 0].astype(np.int64)
    reference_levels = np.load(
        os.path.join(reference_dir, reference_manifest["group_level"]), mmap_mode="r"
    ).astype(np.int64)
    level_specs_by_id = {
        int(item["level"]): item for item in reference_manifest["level_codebooks"]
    }
    spatial_ids = decomposition["atom_group_ids"][gaussian_atom_ids]
    membership = decomposition["atom_membership"][gaussian_atom_ids]
    point_reliability = decomposition["atom_reliability"][gaussian_atom_ids]
    point_entropy = decomposition["atom_entropy"][gaussian_atom_ids]
    parent_keys = decomposition[parent_key_name]
    parent_reliability = decomposition.get(
        f"{parent_key_name}_reliability", decomposition["consistency"]
    )
    num_gaussians = len(gaussian_atom_ids)
    point_ids = np.full((num_gaussians, 4), -1, dtype=np.int64)
    group_levels = []
    group_reliabilities = []
    semantic_ids_by_level = []
    quantization_by_level = []
    codebooks = []
    level_specs = []
    training = []
    offset = 0
    for level in range(4):
        reference_codebook = np.load(
            os.path.join(reference_dir, level_specs_by_id[level]["codebook"])
        ).astype(np.float32)
        reference_codebook = normalize_rows(reference_codebook)
        global_reference_ids = np.asarray(reference_point_ids[:, level], dtype=np.int64)
        safe_reference_ids = np.clip(
            global_reference_ids, 0, len(reference_semantic_ids) - 1
        )
        local_reference_ids = reference_semantic_ids[safe_reference_ids]
        valid = (
            (spatial_ids[:, level] >= 0)
            & (global_reference_ids >= 0)
            & (global_reference_ids < len(reference_semantic_ids))
            & (reference_levels[safe_reference_ids] == level)
            & (local_reference_ids >= 0)
            & (local_reference_ids < len(reference_codebook))
            & (membership[:, level] > 0.0)
        )
        pair_rows = np.stack(
            [spatial_ids[valid, level], local_reference_ids[valid]], axis=1
        )
        unique_pairs, inverse = np.unique(pair_rows, axis=0, return_inverse=True)
        if not len(unique_pairs):
            raise RuntimeError(f"No composite Group addresses survived at level {level}")
        composite_keys = normalize_rows(
            group_key_weight * parent_keys[unique_pairs[:, 0]]
            + (1.0 - group_key_weight) * reference_codebook[unique_pairs[:, 1]]
        )
        group_sizes = np.bincount(inverse, minlength=len(unique_pairs)).astype(np.int32)
        composite_reliability = parent_reliability[unique_pairs[:, 0]].astype(np.float32)
        codebook, local_codes, errors, stats = train_level_codebook(
            composite_keys,
            composite_reliability,
            group_sizes,
            1,
            0.0,
            min(int(args.codes_per_level[level]), len(unique_pairs)),
            min(int(args.train_samples), len(unique_pairs)),
            args.kmeans_iterations,
            args.seed + 3001 * (level + 1) + int(round(group_key_weight * 1000)),
            args.faiss_gpu,
            args.assignment_chunk_size,
        )
        point_ids[valid, level] = offset + inverse
        group_levels.append(np.full(len(unique_pairs), level, dtype=np.uint8))
        group_reliabilities.append(composite_reliability)
        semantic_ids_by_level.append(local_codes.astype(np.int64))
        quantization_by_level.append(errors)
        filename = f"{LEVEL_NAMES[level]}_codebook.npy"
        np.save(os.path.join(output_dir, filename), codebook)
        codebooks.append(codebook)
        stats.update(
            {
                "level": level,
                "name": LEVEL_NAMES[level],
                "num_composite_groups": int(len(unique_pairs)),
                "num_spatial_parents": int(np.unique(unique_pairs[:, 0]).size),
                "group_size_quantiles": quantiles(group_sizes),
            }
        )
        training.append(stats)
        level_specs.append(
            {
                "name": LEVEL_NAMES[level],
                "level": level,
                "semantic_role": LEVEL_ROLES[level],
                "codebook": filename,
                "num_codes": int(len(codebook)),
                "quantization": "fresh_seeded_spherical_kmeans_on_spatial_semantic_composites",
            }
        )
        offset += len(unique_pairs)

    group_levels = np.concatenate(group_levels)
    group_reliability = np.concatenate(group_reliabilities).astype(np.float16)
    semantic_ids = np.concatenate(semantic_ids_by_level)[:, None]
    group_quantization_error = np.concatenate(quantization_by_level)
    id_dtype = np.uint16 if offset < np.iinfo(np.uint16).max else np.uint32
    invalid_id = int(np.iinfo(id_dtype).max)
    stored_ids = np.where(point_ids >= 0, point_ids, invalid_id).astype(id_dtype)
    semantic_dtype = (
        np.uint16
        if max(len(codebook) for codebook in codebooks) < np.iinfo(np.uint16).max
        else np.uint32
    )
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    semantic_ids = semantic_ids.astype(semantic_dtype)
    valid = point_ids >= 0
    point_weights = np.rint(np.clip(membership, 0.0, 1.0) * 255).astype(np.uint8)
    packed_entropy = np.rint(np.clip(point_entropy, 0.0, 1.0) * 255).astype(np.uint8)
    point_error = np.zeros_like(membership, dtype=np.float32)
    point_error[valid] = group_quantization_error[point_ids[valid]]
    packed_error = quantize_chord_error_upper_bound(point_error)
    arrays = {
        "group_semantic_code_ids.npy": semantic_ids,
        "group_level.npy": group_levels,
        "group_reliability.npy": group_reliability,
        "point_group_ids.npy": stored_ids,
        "point_group_weights.npy": point_weights,
        "point_group_reliability.npy": point_reliability.astype(np.float16),
        "point_group_entropy.npy": packed_entropy,
        "point_group_quantization_error.npy": packed_error,
    }
    for filename, array in arrays.items():
        np.save(os.path.join(output_dir, filename), array)
    semantic_bytes = sum(array.nbytes for array in arrays.values()) + sum(
        codebook.nbytes for codebook in codebooks
    )
    covered = valid.any(axis=1)
    manifest = {
        "format_version": 1,
        "representation": "hierarchical_independent_group_codebooks",
        "method": "group_first_gaussian_refined_hierarchical_memory",
        "variant": f"composite_group_key_weight_{group_key_weight:.2f}",
        "num_gaussians": int(num_gaussians),
        "num_spatial_groups": int(offset),
        "feature_dim": int(reference_manifest["feature_dim"]),
        "top_m": 4,
        "resident_slots_required": 4,
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_reliability": "point_group_reliability.npy",
        "point_group_entropy": "point_group_entropy.npy",
        "point_group_quantization_error": "point_group_quantization_error.npy",
        "point_group_quantization_error_scale": 2.0 / 255.0,
        "invalid_id": invalid_id,
        "id_dtype": str(stored_ids.dtype),
        "weight_dtype": "uint8_group_membership",
        "level_codebooks": level_specs,
        "covered_fraction": float(covered.mean()),
        "mean_ids_per_covered_gaussian": float(valid[covered].sum(axis=1).mean()),
        "usable_slot_fraction": float((valid & (point_reliability > 0.0)).mean()),
        "usable_covered_fraction": float((valid & (point_reliability > 0.0)).any(axis=1).mean()),
        "group_addressing": {
            "address": "unique (persistent spatial Group ID, Gaussian semantic token ID)",
            "semantic_value": "fresh per-level composite code ID",
            "decode": "Group-first bounded support with Gaussian-token refinement inside the support",
            "group_key_weight": float(group_key_weight),
            "peer_levels": True,
            "training": training,
            "statistics": decomposition["statistics"],
        },
        "storage": {
            "total_semantic_bytes": int(semantic_bytes),
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
            "shared_vocabulary_bytes_unique": int(sum(codebook.nbytes for codebook in codebooks)),
        },
        "reproducibility": {"seed": int(args.seed), "fixed": True},
        "leakage_control": {
            "evaluation_queries_or_labels_used": False,
            "training_views_only": True,
        },
        "source": {
            "reference_memory": os.path.abspath(reference_dir),
            "reference_method": reference_manifest.get("method"),
        },
        "args": vars(args),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a47_audit_dir", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--reference_memory", default=None)
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
    parser.add_argument("--minimum_owner_membership", type=float, default=0.05)
    parser.add_argument("--boundary_margin", type=float, default=0.20)
    parser.add_argument("--boundary_reliability_floor", type=float, default=0.25)
    parser.add_argument("--exterior_semantic_weight", type=float, default=0.35)
    parser.add_argument("--reliable_support_views", type=float, default=10.0)
    parser.add_argument("--minimum_group_reliability", type=float, default=0.05)
    parser.add_argument("--source_consistency_margin", type=float, default=0.0)
    parser.add_argument("--codes_per_level", nargs=4, type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--kmeans_iterations", type=int, default=25)
    parser.add_argument("--train_samples", type=int, default=200000)
    parser.add_argument("--assignment_chunk_size", type=int, default=8192)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    final_name = "group_conditioned_refine_memory" if args.reference_memory else "residual_key_memory"
    final_manifest = os.path.join(output_dir, final_name, "manifest.json")
    if os.path.isfile(final_manifest) and not args.force:
        print(f"Reuse full Group-addressed memory: {output_dir}")
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
        raise ValueError("A47 source contract is incompatible with A51")
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
    decomposition = decompose_overlapping_groups(
        persistent, views, ring_views, contact_graph, args
    )
    core_manifest = write_memory(
        os.path.join(output_dir, "core_key_memory"),
        decomposition,
        gaussian_atom_ids,
        "core_keys",
        args,
    )
    residual_manifest = write_memory(
        os.path.join(output_dir, "residual_key_memory"),
        decomposition,
        gaussian_atom_ids,
        "residual_keys",
        args,
    )
    gated_manifests = {}
    composite_manifests = {}
    if args.reference_memory:
        decomposition = add_split_consistency_keys(
            decomposition,
            os.path.abspath(args.reference_memory),
            gaussian_atom_ids,
            contact_graph,
            args,
        )
        for key_name in ("gated_core_keys", "gated_residual_keys"):
            memory_name = key_name.replace("keys", "key_memory")
            gated_manifests[memory_name] = write_memory(
                os.path.join(output_dir, memory_name),
                decomposition,
                gaussian_atom_ids,
                key_name,
                args,
            )
        for name, weight in (
            ("composite_refine_memory", 0.0),
            ("group_conditioned_refine_memory", 0.2),
        ):
            composite_manifests[name] = write_composite_refinement_memory(
                os.path.join(output_dir, name),
                decomposition,
                gaussian_atom_ids,
                os.path.abspath(args.reference_memory),
                "gated_residual_keys",
                weight,
                args,
            )
    summary = {
        "experiment": "A51_full_group_addressed_hierarchical_memory",
        "scene": "ramen",
        "seed": args.seed,
        "full_training_views": len(views),
        "spatial_groups": int(len(persistent["profiles"])),
        "core_memory": {
            "path": "core_key_memory",
            "covered_fraction": core_manifest["covered_fraction"],
        },
        "residual_memory": {
            "path": "residual_key_memory",
            "covered_fraction": residual_manifest["covered_fraction"],
        },
        "split_consistency_memories": {
            name: {
                "path": name,
                "covered_fraction": manifest["covered_fraction"],
            }
            for name, manifest in gated_manifests.items()
        },
        "composite_refinement_memories": {
            name: {
                "path": name,
                "covered_fraction": manifest["covered_fraction"],
                "num_composite_groups": manifest["num_spatial_groups"],
            }
            for name, manifest in composite_manifests.items()
        },
        "source": {
            "a47_audit_dir": os.path.abspath(args.a47_audit_dir),
            "a47_manifest_sha256": file_sha256(a47_manifest_path),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        },
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(summary, output, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
