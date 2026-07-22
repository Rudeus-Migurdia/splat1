#!/usr/bin/env python
"""Audit a geometry-conditioned global partition of persistent SAM tracklets."""

import argparse
import json
import os
import time

import numpy as np

from build_multi_hypothesis_entity_tomography import load_prepared_views, noisy_or, normalize_rows
from build_persistent_entity_tomography import (
    evaluate_persistent_model,
    file_sha256,
    fit_persistent_slots,
    incidence_entries,
    match_level_slots,
)


class UnionFind:
    def __init__(self, size):
        self.parent = np.arange(size, dtype=np.int32)
        self.members = [[index] for index in range(size)]

    def find(self, index):
        root = int(index)
        while self.parent[root] != root:
            root = int(self.parent[root])
        while self.parent[index] != index:
            parent = int(self.parent[index])
            self.parent[index] = root
            index = parent
        return root

    def union(self, first, second):
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return first
        if len(self.members[first]) < len(self.members[second]):
            first, second = second, first
        self.parent[second] = first
        self.members[first].extend(self.members[second])
        self.members[second] = []
        return first


def quaternion_minor_axes(quaternions, scaling):
    """Return the rotation axis associated with each Gaussian's smallest scale."""
    quaternion = np.asarray(quaternions, dtype=np.float64)
    quaternion /= np.maximum(np.linalg.norm(quaternion, axis=1, keepdims=True), 1e-12)
    w, x, y, z = quaternion.T
    rotations = np.empty((len(quaternion), 3, 3), dtype=np.float64)
    rotations[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rotations[:, 0, 1] = 2 * (x * y - w * z)
    rotations[:, 0, 2] = 2 * (x * z + w * y)
    rotations[:, 1, 0] = 2 * (x * y + w * z)
    rotations[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rotations[:, 1, 2] = 2 * (y * z - w * x)
    rotations[:, 2, 0] = 2 * (x * z - w * y)
    rotations[:, 2, 1] = 2 * (y * z + w * x)
    rotations[:, 2, 2] = 1 - 2 * (x * x + y * y)
    minor = np.argmin(scaling, axis=1)
    return rotations[np.arange(len(rotations)), :, minor]


def aggregate_gaussian_geometry(xyz, log_scaling, quaternion, raw_opacity, atom_ids):
    """Aggregate Gaussian covariance cues into fixed spatial atoms."""
    xyz = np.asarray(xyz, dtype=np.float64)
    scaling = np.exp(np.asarray(log_scaling, dtype=np.float64))
    opacity = 1.0 / (1.0 + np.exp(-np.asarray(raw_opacity, dtype=np.float64).reshape(-1)))
    atom_ids = np.asarray(atom_ids, dtype=np.int64)
    if len(xyz) != len(atom_ids):
        raise ValueError("Gaussian geometry and atom assignment lengths differ")
    atom_count = int(atom_ids.max()) + 1
    weights = np.maximum(opacity, 1e-4)
    weight_sum = np.bincount(atom_ids, weights=weights, minlength=atom_count)

    def weighted_columns(values):
        values = np.asarray(values, dtype=np.float64)
        result = np.stack(
            [np.bincount(atom_ids, weights=weights * values[:, col], minlength=atom_count)
             for col in range(values.shape[1])],
            axis=1,
        )
        return result / np.maximum(weight_sum[:, None], 1e-12)

    centroid = weighted_columns(xyz)
    mean_scaling = weighted_columns(scaling)
    radius = np.sqrt(np.mean(mean_scaling * mean_scaling, axis=1))
    minor_axes = quaternion_minor_axes(quaternion, scaling)
    outer = np.einsum("ni,nj->nij", minor_axes, minor_axes).reshape(len(xyz), 9)
    orientation_tensor = weighted_columns(outer).reshape(atom_count, 3, 3)
    _, eigenvectors = np.linalg.eigh(orientation_tensor)
    normal = eigenvectors[:, :, -1]
    mean_opacity = np.bincount(atom_ids, weights=weights * opacity, minlength=atom_count)
    mean_opacity /= np.maximum(weight_sum, 1e-12)
    counts = np.bincount(atom_ids, minlength=atom_count)
    return {
        "centroid": centroid.astype(np.float32),
        "normal": normal.astype(np.float32),
        "radius": radius.astype(np.float32),
        "opacity": mean_opacity.astype(np.float32),
        "gaussian_count": counts.astype(np.int32),
    }


def load_atom_geometry(checkpoint_path, atom_ids_path):
    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = checkpoint[0] if isinstance(checkpoint, tuple) and len(checkpoint) == 2 else checkpoint
    if not isinstance(model, (tuple, list)) or len(model) not in (12, 13):
        raise ValueError("Unsupported Gaussian checkpoint contract")
    atom_ids = np.load(atom_ids_path)
    return aggregate_gaussian_geometry(
        model[1].detach().numpy(),
        model[4].detach().numpy(),
        model[5].detach().numpy(),
        model[6].detach().numpy(),
        atom_ids,
    )


def build_atom_contact_graph(atom_geometry, neighbors=8):
    from scipy.sparse import coo_matrix
    from scipy.spatial import cKDTree

    centroid = atom_geometry["centroid"]
    normal = atom_geometry["normal"]
    radius = atom_geometry["radius"]
    opacity = atom_geometry["opacity"]
    distances, indices = cKDTree(centroid).query(centroid, k=min(neighbors + 1, len(centroid)))
    rows = np.repeat(np.arange(len(centroid)), indices.shape[1] - 1)
    columns = indices[:, 1:].reshape(-1)
    distances = distances[:, 1:].reshape(-1)
    scale = radius[rows] + radius[columns]
    proximity = np.exp(-distances / np.maximum(4.0 * scale, 1e-6))
    orientation = np.abs(np.sum(normal[rows] * normal[columns], axis=1))
    opacity_agreement = np.exp(-np.abs(opacity[rows] - opacity[columns]) / 0.25)
    values = proximity * (0.65 * orientation + 0.35 * opacity_agreement)
    graph = coo_matrix((values, (rows, columns)), shape=(len(centroid), len(centroid)))
    graph = graph.maximum(graph.T).tocsr()
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def tracklet_geometry(model, atom_geometry):
    weights = np.maximum(model["profiles"], 0.0).astype(np.float64)
    weights *= np.maximum(atom_geometry["opacity"], 1e-3)[None]
    mass = np.maximum(weights.sum(axis=1), 1e-8)
    centroid = weights @ atom_geometry["centroid"] / mass[:, None]
    second = weights @ (atom_geometry["centroid"] ** 2) / mass[:, None]
    variance = np.maximum(second - centroid * centroid, 0.0)
    atom_radius = weights @ (atom_geometry["radius"] ** 2) / mass
    spread = np.sqrt(variance.sum(axis=1) + atom_radius)
    opacity = weights @ atom_geometry["opacity"] / mass
    orientation = np.einsum(
        "sa,aij->sij",
        weights,
        np.einsum("ai,aj->aij", atom_geometry["normal"], atom_geometry["normal"]),
    )
    _, eigenvectors = np.linalg.eigh(orientation)
    normal = eigenvectors[:, :, -1]
    return {
        "centroid": centroid.astype(np.float32),
        "spread": spread.astype(np.float32),
        "opacity": opacity.astype(np.float32),
        "normal": normal.astype(np.float32),
    }


def pair_evidence(model, atom_geometry, contact_graph, coverage_threshold):
    from scipy.sparse import csr_matrix

    binary = (model["profiles"] >= coverage_threshold).astype(np.float32)
    support = binary.sum(axis=1)
    intersection = binary @ binary.T
    union = support[:, None] + support[None] - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0)
    sparse_support = csr_matrix(binary)
    contact = (sparse_support @ contact_graph @ sparse_support.T).toarray()
    contact /= np.maximum(8.0 * np.minimum(support[:, None], support[None]), 1.0)
    semantic = model["descriptors"] @ model["descriptors"].T
    geometry = tracklet_geometry(model, atom_geometry)
    delta = geometry["centroid"][:, None] - geometry["centroid"][None]
    distance = np.linalg.norm(delta, axis=2)
    extent = geometry["spread"][:, None] + geometry["spread"][None]
    proximity = np.exp(-distance / np.maximum(extent, 1e-6))
    orientation = np.abs(geometry["normal"] @ geometry["normal"].T)
    opacity = np.exp(
        -np.abs(geometry["opacity"][:, None] - geometry["opacity"][None]) / 0.25
    )
    continuity = 0.50 * proximity + 0.30 * orientation + 0.20 * opacity
    return {
        "overlap": overlap,
        "contact": contact,
        "semantic": semantic,
        "continuity": continuity,
    }


def geometry_conditioned_partition(model, atom_geometry, contact_graph, args):
    evidence = pair_evidence(model, atom_geometry, contact_graph, args.coverage_threshold)
    edges = []
    for level in range(4):
        slots = np.flatnonzero(model["levels"] == level)
        for local_first, first in enumerate(slots):
            for second in slots[local_first + 1:]:
                semantic = float(evidence["semantic"][first, second])
                overlap = float(evidence["overlap"][first, second])
                contact = float(evidence["contact"][first, second])
                continuity = float(evidence["continuity"][first, second])
                if semantic < args.partition_minimum_semantic_cosine:
                    continue
                if continuity < args.partition_minimum_geometry_continuity:
                    continue
                if max(overlap / 0.50, contact / 0.08) < args.partition_minimum_boundary_support:
                    continue
                semantic_cost = np.clip((1.0 - semantic) / 0.20, 0.0, 1.0)
                boundary_cost = 1.0 - np.clip(max(overlap / 0.50, contact / 0.08), 0.0, 1.0)
                geometry_cost = 1.0 - continuity
                inconsistency = (
                    args.partition_semantic_cost_weight * semantic_cost
                    + args.partition_boundary_cost_weight * boundary_cost
                    + args.partition_geometry_cost_weight * geometry_cost
                )
                gain = args.partition_entity_count_penalty - inconsistency
                if gain > 0.0:
                    edges.append((gain, int(first), int(second), semantic, overlap, contact, continuity))

    union_find = UnionFind(len(model["profiles"]))
    accepted = []
    for edge in sorted(edges, reverse=True):
        _, first, second, _, _, _, _ = edge
        first_root = union_find.find(first)
        second_root = union_find.find(second)
        if first_root == second_root:
            continue
        union_find.union(first_root, second_root)
        accepted.append(edge)

    components = [members for members in union_find.members if members]
    partitioned = []
    for members in components:
        members = np.asarray(members, dtype=np.int64)
        utility = np.maximum(model["utility"][members], 1e-6)
        descriptor = (utility[:, None] * model["descriptors"][members]).sum(axis=0)
        descriptor /= max(float(np.linalg.norm(descriptor)), 1e-8)
        partitioned.append(
            {
                "profile": noisy_or(model["profiles"][members]),
                "descriptor": descriptor,
                "support_views": int(model["support_views"][members].max()),
                "level": int(model["levels"][members[0]]),
                "utility": float(model["utility"][members].sum()),
                "members": members.tolist(),
            }
        )
    partitioned.sort(key=lambda item: (item["level"], -item["utility"]))
    result = {
        "profiles": np.stack([item["profile"] for item in partitioned]).astype(np.float32),
        "descriptors": normalize_rows(np.stack([item["descriptor"] for item in partitioned])),
        "support_views": np.asarray([item["support_views"] for item in partitioned], dtype=np.int32),
        "levels": np.asarray([item["level"] for item in partitioned], dtype=np.int8),
        "utility": np.asarray([item["utility"] for item in partitioned], dtype=np.float32),
        "capacity_saturated": False,
        "statistics": {
            "input_tracklets": int(len(model["profiles"])),
            "candidate_positive_mdl_edges": len(edges),
            "accepted_merges": len(accepted),
            "partition_entities": len(partitioned),
            "merged_entities": sum(len(item["members"]) > 1 for item in partitioned),
            "maximum_component_size": max(len(item["members"]) for item in partitioned),
            "entities_per_level": np.bincount(
                np.asarray([item["level"] for item in partitioned]), minlength=4
            ).astype(int).tolist(),
        },
        "components": [item["members"] for item in partitioned],
        "accepted_edges": [
            {
                "mdl_gain": float(edge[0]),
                "first_tracklet": edge[1],
                "second_tracklet": edge[2],
                "semantic_cosine": edge[3],
                "profile_jaccard": edge[4],
                "geometry_contact": edge[5],
                "geometry_continuity": edge[6],
            }
            for edge in accepted
        ],
    }
    return result


def save_partition(path, model):
    np.savez_compressed(
        path,
        profiles=model["profiles"],
        descriptors=model["descriptors"],
        support_views=model["support_views"],
        levels=model["levels"],
        utility=model["utility"],
    )


def make_gate(metrics, args):
    checks = {
        "improves_a48_heldout_nll": metrics["relative_nll_improvement_over_a48"] >= args.minimum_nll_improvement_over_a48,
        "improves_uncapped_tracklets": metrics["relative_nll_improvement_over_uncapped"] >= args.minimum_nll_improvement_over_uncapped,
        "split_partition_stability": metrics["median_matched_jaccard"] >= args.minimum_split_stability,
        "enough_stable_entities": metrics["stable_entities"] >= args.minimum_stable_entities,
        "slot_count_agreement": metrics["entity_count_agreement"] >= args.minimum_entity_count_agreement,
        "capacity_not_saturated": not metrics["capacity_saturated"],
        "union_mass_not_saturated": metrics["mdl_union_mass_fraction"] <= args.maximum_union_mass_fraction,
        "unresolved_certificate_written": metrics["unresolved_certificate_written"],
        "no_queries_labels_or_codebooks": metrics["no_queries_labels_or_codebooks"],
    }
    passed = all(checks.values())
    return {
        "pass": bool(passed),
        "decision": "PROCEED_TO_A49_1_CONTINUOUS_ENTITY_SEMANTICS" if passed else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS",
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a47_audit_dir", required=True)
    parser.add_argument("--a48_audit_dir", required=True)
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
    parser.add_argument("--evaluation_candidates", type=int, default=6)
    parser.add_argument("--union_relative_nll_penalty", type=float, default=0.05)
    parser.add_argument("--unresolved_association_threshold", type=float, default=0.20)
    parser.add_argument("--atom_neighbors", type=int, default=8)
    parser.add_argument("--partition_minimum_semantic_cosine", type=float, default=0.85)
    parser.add_argument("--partition_minimum_geometry_continuity", type=float, default=0.40)
    parser.add_argument("--partition_minimum_boundary_support", type=float, default=0.20)
    parser.add_argument("--partition_entity_count_penalty", type=float, default=0.42)
    parser.add_argument("--partition_semantic_cost_weight", type=float, default=0.45)
    parser.add_argument("--partition_boundary_cost_weight", type=float, default=0.35)
    parser.add_argument("--partition_geometry_cost_weight", type=float, default=0.20)
    parser.add_argument("--minimum_nll_improvement_over_a48", type=float, default=0.0)
    parser.add_argument("--minimum_nll_improvement_over_uncapped", type=float, default=0.0)
    parser.add_argument("--minimum_split_stability", type=float, default=0.80)
    parser.add_argument("--minimum_stable_entities", type=int, default=53)
    parser.add_argument("--minimum_entity_count_agreement", type=float, default=0.80)
    parser.add_argument("--maximum_union_mass_fraction", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse geometry-conditioned partition: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    with open(os.path.join(args.a47_audit_dir, "manifest.json")) as source:
        a47_manifest = json.load(source)
    with open(os.path.join(args.a48_audit_dir, "manifest.json")) as source:
        a48_manifest = json.load(source)
    contract = a47_manifest["source_contract"]
    if not (
        contract["training_views_only"]
        and contract["raw_overlapping_proposals"]
        and contract["odd_even_independent_fit"]
        and not contract["evaluation_queries_or_labels_used"]
        and not contract["codebooks_trained"]
        and int(contract["fixed_seed"]) == args.seed
    ):
        raise ValueError("A47 source contract is incompatible with A49")
    checkpoint_sha = file_sha256(args.geometry_checkpoint)
    if checkpoint_sha != a47_manifest["inputs"]["geometry_checkpoint_sha256"]:
        raise ValueError("Geometry checkpoint does not match A47 atom ownership")

    entries = incidence_entries(args.a47_audit_dir)
    views = load_prepared_views(args.a47_audit_dir, entries)
    atom_ids_path = os.path.join(args.a47_audit_dir, "gaussian_atom_ids.npy")
    atom_geometry = load_atom_geometry(args.geometry_checkpoint, atom_ids_path)
    contact_graph = build_atom_contact_graph(atom_geometry, args.atom_neighbors)
    odd_raw = fit_persistent_slots(views, 0, args)
    even_raw = fit_persistent_slots(views, 1, args)
    raw_odd_to_even = evaluate_persistent_model(odd_raw, views, 1, args)
    raw_even_to_odd = evaluate_persistent_model(even_raw, views, 0, args)
    raw_matches, _ = match_level_slots(
        odd_raw, even_raw, args.minimum_persistence_views, args.minimum_split_stability
    )
    odd = geometry_conditioned_partition(odd_raw, atom_geometry, contact_graph, args)
    even = geometry_conditioned_partition(even_raw, atom_geometry, contact_graph, args)

    odd_to_even = evaluate_persistent_model(odd, views, 1, args)
    even_to_odd = evaluate_persistent_model(even, views, 0, args)
    matches, unresolved_slots = match_level_slots(
        odd, even, args.minimum_persistence_views, args.minimum_split_stability
    )
    unresolved_path = os.path.join(output_dir, "unresolved_partition_certificate.json")
    with open(unresolved_path, "w") as output:
        json.dump(
            {
                "unresolved_entities": unresolved_slots,
                "odd_to_even_unresolved_proposals": odd_to_even["unresolved"],
                "even_to_odd_unresolved_proposals": even_to_odd["unresolved"],
                "forced_assignment_forbidden": True,
            },
            output,
            indent=2,
        )
    save_partition(os.path.join(output_dir, "odd_partition.npz"), odd)
    save_partition(os.path.join(output_dir, "even_partition.npz"), even)
    with open(os.path.join(output_dir, "partition_audit.json"), "w") as output:
        json.dump(
            {
                "odd_components": odd["components"],
                "even_components": even["components"],
                "odd_accepted_edges": odd["accepted_edges"],
                "even_accepted_edges": even["accepted_edges"],
            },
            output,
            indent=2,
        )

    nll_names = odd_to_even["mean_nll"].keys()
    mean_nll = {
        name: 0.5 * (odd_to_even["mean_nll"][name] + even_to_odd["mean_nll"][name])
        for name in nll_names
    }
    a48_nll = a48_manifest["metrics"]["mean_heldout_balanced_mask_nll"]["persistent_mdl_union"]
    uncapped_nll = 0.5 * (
        raw_odd_to_even["mean_nll"]["persistent_mdl_union"]
        + raw_even_to_odd["mean_nll"]["persistent_mdl_union"]
    )
    raw_jaccards = np.asarray([item["jaccard"] for item in raw_matches], dtype=np.float32)
    jaccards = np.asarray([item["jaccard"] for item in matches], dtype=np.float32)
    stable = [item for item in matches if item["stable"]]
    odd_count = len(odd["profiles"])
    even_count = len(even["profiles"])
    union_mass = odd_to_even["mdl_union_mass"] + even_to_odd["mdl_union_mass"]
    positive_mass = odd_to_even["positive_mass"] + even_to_odd["positive_mass"]
    metrics = {
        "mean_heldout_balanced_mask_nll": mean_nll,
        "a48_reference_persistent_mdl_union_nll": a48_nll,
        "uncapped_tracklet_reference_persistent_mdl_union_nll": float(uncapped_nll),
        "relative_nll_improvement_over_a48": float(
            (a48_nll - mean_nll["persistent_mdl_union"]) / max(a48_nll, 1e-8)
        ),
        "relative_nll_improvement_over_uncapped": float(
            (uncapped_nll - mean_nll["persistent_mdl_union"]) / max(uncapped_nll, 1e-8)
        ),
        "uncapped_tracklet_median_matched_jaccard": float(np.median(raw_jaccards)) if raw_jaccards.size else 0.0,
        "uncapped_tracklet_stable_entities": sum(
            item["jaccard"] >= args.minimum_split_stability for item in raw_matches
        ),
        "median_matched_jaccard": float(np.median(jaccards)) if jaccards.size else 0.0,
        "mean_matched_jaccard": float(jaccards.mean()) if jaccards.size else 0.0,
        "matched_entities": len(matches),
        "stable_entities": len(stable),
        "entity_count_agreement": min(odd_count, even_count) / max(odd_count, even_count),
        "capacity_saturated": False,
        "mdl_union_mass_fraction": float(union_mass / max(positive_mass, 1e-8)),
        "unresolved_entities": len(unresolved_slots),
        "unresolved_certificate_written": os.path.isfile(unresolved_path),
        "no_queries_labels_or_codebooks": True,
    }
    gate = make_gate(metrics, args)
    manifest = {
        "format_version": 1,
        "experiment": "A49.0_geometry_conditioned_global_tracklet_partition",
        "scene": "ramen",
        "seed": args.seed,
        "representation": "level_preserving_geometry_conditioned_persistent_entities",
        "inputs": {
            "a47_audit_dir": os.path.abspath(args.a47_audit_dir),
            "a48_audit_dir": os.path.abspath(args.a48_audit_dir),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
            "geometry_checkpoint_sha256": checkpoint_sha,
            "gaussian_atom_ids_sha256": file_sha256(atom_ids_path),
        },
        "source_contract": {
            "training_views_only": True,
            "odd_even_independent_fit_and_partition": True,
            "same_level_partition_only": True,
            "gaussian_geometry_used": True,
            "evaluation_queries_or_labels_used": False,
            "codebooks_trained": False,
            "fixed_seed": args.seed,
        },
        "odd_model": {"entities": odd_count, "statistics": odd["statistics"]},
        "even_model": {"entities": even_count, "statistics": even["statistics"]},
        "metrics": metrics,
        "gate": gate,
        "parameters": vars(args),
        "runtime_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    with open(os.path.join(output_dir, "gate.json"), "w") as output:
        json.dump(gate, output, indent=2)
    print(json.dumps({"odd": manifest["odd_model"], "even": manifest["even_model"], "metrics": metrics, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
