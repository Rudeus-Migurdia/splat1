#!/usr/bin/env python
"""Fit persistence-constrained entity tracks with an MDL union penalty."""

import argparse
import hashlib
import json
import os
import re
import time

import numpy as np

from build_multi_hypothesis_entity_tomography import (
    balanced_bernoulli_nll,
    load_prepared_views,
    noisy_or,
    normalize_rows,
    soft_jaccard_batch,
)


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ViewExclusiveUnionFind:
    """Union-find that forbids two distinct proposals from one view in a track."""

    def __init__(self, view_ids):
        self.parent = np.arange(len(view_ids), dtype=np.int32)
        self.size = np.ones(len(view_ids), dtype=np.int32)
        self.views = [{int(view_id)} for view_id in view_ids]
        self.rejected_view_conflicts = 0

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
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return True
        if self.views[first_root] & self.views[second_root]:
            self.rejected_view_conflicts += 1
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        self.views[first_root].update(self.views[second_root])
        self.views[second_root].clear()
        return True


def reciprocal_match_edges(
    first,
    second,
    coverage_threshold,
    minimum_spatial_jaccard,
    minimum_semantic_cosine,
    spatial_weight,
    minimum_association,
):
    edges = []
    for level in range(4):
        first_ids = np.flatnonzero(first["levels"] == level)
        second_ids = np.flatnonzero(second["levels"] == level)
        if not first_ids.size or not second_ids.size:
            continue
        first_binary = (first["coverage"][first_ids] >= coverage_threshold).astype(np.float32)
        second_binary = (second["coverage"][second_ids] >= coverage_threshold).astype(np.float32)
        intersection = first_binary @ second_binary.T
        union = (
            first_binary.sum(axis=1)[:, None]
            + second_binary.sum(axis=1)[None]
            - intersection
        )
        spatial = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0.0,
        )
        semantic = first["descriptors"][first_ids] @ second["descriptors"][second_ids].T
        normalized_semantic = np.clip((semantic - 0.5) / 0.5, 0.0, 1.0)
        association = spatial_weight * spatial + (1.0 - spatial_weight) * normalized_semantic
        row_best = association.argmax(axis=1)
        column_best = association.argmax(axis=0)
        for row, column in enumerate(row_best):
            if int(column_best[column]) != row:
                continue
            if spatial[row, column] < minimum_spatial_jaccard:
                continue
            if semantic[row, column] < minimum_semantic_cosine:
                continue
            if association[row, column] < minimum_association:
                continue
            edges.append(
                (
                    float(association[row, column]),
                    int(first_ids[row]),
                    int(second_ids[column]),
                    float(spatial[row, column]),
                    float(semantic[row, column]),
                )
            )
    return edges


def profile_jaccard(first, second, threshold=0.5):
    first = np.asarray(first) >= threshold
    second = np.asarray(second) >= threshold
    intersection = int(np.logical_and(first, second).sum())
    union = int(np.logical_or(first, second).sum())
    return intersection / max(union, 1)


def fit_persistent_slots(views, train_parity, args):
    train_views = (
        list(views)
        if train_parity is None
        else [view for view in views if view["view_index"] % 2 == train_parity]
    )
    node_refs = []
    view_offsets = {}
    node_view_ids = []
    for train_index, view in enumerate(train_views):
        start = len(node_refs)
        for proposal_index in range(view["coverage"].shape[0]):
            node_refs.append((train_index, proposal_index))
            node_view_ids.append(view["view_index"])
        view_offsets[train_index] = start

    edges = []
    for first_index, first in enumerate(train_views):
        for gap in range(1, args.temporal_neighbors + 1):
            second_index = first_index + gap
            if second_index >= len(train_views):
                continue
            second = train_views[second_index]
            local_edges = reciprocal_match_edges(
                first,
                second,
                args.coverage_threshold,
                args.minimum_spatial_jaccard,
                args.minimum_semantic_cosine,
                args.spatial_weight,
                args.minimum_association,
            )
            for association, first_local, second_local, spatial, semantic in local_edges:
                edges.append(
                    (
                        association,
                        view_offsets[first_index] + first_local,
                        view_offsets[second_index] + second_local,
                        spatial,
                        semantic,
                    )
                )

    union_find = ViewExclusiveUnionFind(node_view_ids)
    accepted_edges = 0
    for _, first_node, second_node, _, _ in sorted(edges, reverse=True):
        if union_find.union(first_node, second_node):
            accepted_edges += 1
    components = {}
    for node_index in range(len(node_refs)):
        components.setdefault(union_find.find(node_index), []).append(node_index)

    candidates = []
    rejected_ephemeral = 0
    for nodes in components.values():
        support_views = {node_view_ids[node] for node in nodes}
        if len(support_views) < args.minimum_persistence_views:
            rejected_ephemeral += 1
            continue
        levels = []
        descriptor_sum = np.zeros(512, dtype=np.float32)
        coverage_sum = np.zeros(views[0]["coverage"].shape[1], dtype=np.float32)
        observations = np.zeros_like(coverage_sum)
        total_mass = 0.0
        quality_sum = 0.0
        for node in nodes:
            train_index, proposal_index = node_refs[node]
            view = train_views[train_index]
            target = view["coverage"][proposal_index]
            visible = view["visibility"] > args.minimum_visibility
            quality = max(float(view["quality"][proposal_index]), 1e-4)
            coverage_sum += quality * target * visible
            observations += quality * visible
            descriptor_sum += quality * view["descriptors"][proposal_index]
            total_mass += float((target * view["visibility"]).sum())
            quality_sum += quality
            levels.append(int(view["levels"][proposal_index]))
        level = int(np.bincount(levels, minlength=4).argmax())
        profile = np.divide(
            coverage_sum,
            observations,
            out=np.zeros_like(coverage_sum),
            where=observations > 1e-8,
        )
        descriptor = descriptor_sum / max(float(np.linalg.norm(descriptor_sum)), 1e-8)
        mean_quality = quality_sum / len(nodes)
        utility = len(support_views) * mean_quality * np.log1p(total_mass / len(nodes))
        candidates.append(
            {
                "profile": profile,
                "descriptor": descriptor,
                "support_views": len(support_views),
                "level": level,
                "utility": float(utility),
                "nodes": len(nodes),
                "mass": total_mass,
                "members": [
                    {
                        "view_index": int(train_views[node_refs[node][0]]["view_index"]),
                        "proposal_index": int(node_refs[node][1]),
                    }
                    for node in nodes
                ],
            }
        )

    kept = []
    redundant_pruned = 0
    for candidate in sorted(candidates, key=lambda item: (-item["utility"], item["level"])):
        redundant = False
        for existing in kept:
            if candidate["level"] != existing["level"]:
                continue
            if float(candidate["descriptor"] @ existing["descriptor"]) < args.merge_semantic_cosine:
                continue
            if profile_jaccard(candidate["profile"], existing["profile"]) >= args.merge_jaccard:
                redundant = True
                break
        if redundant:
            redundant_pruned += 1
        else:
            kept.append(candidate)
    capacity_saturated = len(kept) > args.maximum_slots
    kept = kept[: args.maximum_slots]
    if not kept:
        split_name = "all views" if train_parity is None else f"split {train_parity}"
        raise RuntimeError(f"No persistent slots survived for {split_name}")
    return {
        "profiles": np.stack([item["profile"] for item in kept]).astype(np.float32),
        "descriptors": normalize_rows(np.stack([item["descriptor"] for item in kept])),
        "support_views": np.asarray([item["support_views"] for item in kept], dtype=np.int32),
        "levels": np.asarray([item["level"] for item in kept], dtype=np.int8),
        "utility": np.asarray([item["utility"] for item in kept], dtype=np.float32),
        "members": [item["members"] for item in kept],
        "capacity_saturated": capacity_saturated,
        "statistics": {
            "nodes": len(node_refs),
            "candidate_edges": len(edges),
            "accepted_edges": accepted_edges,
            "rejected_view_conflict_edges": union_find.rejected_view_conflicts,
            "raw_components": len(components),
            "ephemeral_components_rejected": rejected_ephemeral,
            "persistent_components": len(candidates),
            "redundant_components_pruned": redundant_pruned,
            "retained_slots": len(kept),
            "slots_per_level": np.bincount(
                np.asarray([item["level"] for item in kept]), minlength=4
            ).astype(int).tolist(),
        },
    }


def hard_profiles_by_level(model):
    profiles = model["profiles"]
    hard = np.full_like(profiles, 0.02, dtype=np.float32)
    for level in range(4):
        slots = np.flatnonzero(model["levels"] == level)
        if not slots.size:
            continue
        values = profiles[slots]
        local_owner = values.argmax(axis=0)
        confidence = values.max(axis=0)
        atoms = np.flatnonzero(confidence >= 0.25)
        hard[slots[local_owner[atoms]], atoms] = 0.98
    return hard


def evaluate_persistent_model(model, views, heldout_parity, args):
    profiles = model["profiles"]
    descriptors = model["descriptors"]
    hard_profiles = hard_profiles_by_level(model)
    totals = {
        "hard_persistent_single_id": 0.0,
        "persistent_single_path": 0.0,
        "persistent_unpenalized_union": 0.0,
        "persistent_mdl_union": 0.0,
    }
    proposal_weight = 0.0
    positive_mass = 0.0
    union_mass = 0.0
    union_count = 0
    unresolved = []
    evaluated = 0
    for view in views:
        if view["view_index"] % 2 != heldout_parity:
            continue
        visible = view["visibility"] > args.minimum_visibility
        for proposal_index, target in enumerate(view["coverage"]):
            level = int(view["levels"][proposal_index])
            level_slots = np.flatnonzero(model["levels"] == level)
            if not level_slots.size:
                unresolved.append(
                    {
                        "view_index": int(view["view_index"]),
                        "proposal_index": int(proposal_index),
                        "level": level,
                        "reason": "no_persistent_slot_at_level",
                    }
                )
                continue
            level_profiles = profiles[level_slots]
            jaccard = soft_jaccard_batch(level_profiles, target, visible)
            semantic = descriptors[level_slots] @ view["descriptors"][proposal_index]
            association = args.spatial_weight * jaccard + (1.0 - args.spatial_weight) * np.clip(
                (semantic - 0.5) / 0.5, 0.0, 1.0
            )
            candidate_count = min(args.evaluation_candidates, level_slots.size)
            local = np.argpartition(-association, candidate_count - 1)[:candidate_count]
            local = local[np.argsort(-association[local], kind="stable")]
            candidates = level_slots[local]
            hard_nll = min(
                balanced_bernoulli_nll(target, hard_profiles[int(slot)], view["visibility"])
                for slot in candidates
            )
            single_values = [
                balanced_bernoulli_nll(target, profiles[int(slot)], view["visibility"])
                for slot in candidates
            ]
            single_nll = min(single_values)
            best_single = int(candidates[int(np.argmin(single_values))])
            pair_nll = single_nll
            for first_index in range(len(candidates)):
                for second_index in range(first_index + 1, len(candidates)):
                    prediction = noisy_or(profiles[candidates[[first_index, second_index]]])
                    value = balanced_bernoulli_nll(target, prediction, view["visibility"])
                    pair_nll = min(pair_nll, value)
            relative_pair_gain = (single_nll - pair_nll) / max(single_nll, 1e-8)
            mdl_nll = pair_nll if relative_pair_gain >= args.union_relative_nll_penalty else single_nll
            weight = max(float(view["quality"][proposal_index]), 1e-4)
            mass = float((target * view["visibility"]).sum())
            totals["hard_persistent_single_id"] += weight * hard_nll
            totals["persistent_single_path"] += weight * single_nll
            totals["persistent_unpenalized_union"] += weight * pair_nll
            totals["persistent_mdl_union"] += weight * mdl_nll
            proposal_weight += weight
            positive_mass += mass
            evaluated += 1
            if mdl_nll < single_nll:
                union_count += 1
                union_mass += mass
            if association[local[0]] < args.unresolved_association_threshold:
                unresolved.append(
                    {
                        "view_index": int(view["view_index"]),
                        "proposal_index": int(proposal_index),
                        "level": level,
                        "best_slot": best_single,
                        "association": float(association[local[0]]),
                        "reason": "low_association",
                    }
                )
    return {
        "mean_nll": {key: value / max(proposal_weight, 1e-8) for key, value in totals.items()},
        "evaluated_proposals": evaluated,
        "proposal_weight": proposal_weight,
        "positive_mass": positive_mass,
        "mdl_union_mass": union_mass,
        "mdl_union_mass_fraction": union_mass / max(positive_mass, 1e-8),
        "mdl_union_proposals": union_count,
        "unresolved": unresolved,
    }


def match_level_slots(first, second, minimum_views, stability_threshold):
    from scipy.optimize import linear_sum_assignment

    matches = []
    unresolved = []
    for level in range(4):
        first_ids = np.flatnonzero(
            (first["levels"] == level) & (first["support_views"] >= minimum_views)
        )
        second_ids = np.flatnonzero(
            (second["levels"] == level) & (second["support_views"] >= minimum_views)
        )
        if not first_ids.size or not second_ids.size:
            for split, ids in (("odd", first_ids), ("even", second_ids)):
                unresolved.extend(
                    {"split": split, "slot": int(slot), "level": level, "best_jaccard": 0.0}
                    for slot in ids
                )
            continue
        first_binary = (first["profiles"][first_ids] >= 0.5).astype(np.float32)
        second_binary = (second["profiles"][second_ids] >= 0.5).astype(np.float32)
        intersection = first_binary @ second_binary.T
        union = first_binary.sum(axis=1)[:, None] + second_binary.sum(axis=1)[None] - intersection
        jaccard = np.divide(
            intersection,
            union,
            out=np.zeros_like(intersection, dtype=np.float32),
            where=union > 0.0,
        )
        rows, columns = linear_sum_assignment(-jaccard)
        matched_first = set()
        matched_second = set()
        for row, column in zip(rows, columns):
            value = float(jaccard[row, column])
            first_slot = int(first_ids[row])
            second_slot = int(second_ids[column])
            matches.append(
                {
                    "odd_slot": first_slot,
                    "even_slot": second_slot,
                    "level": level,
                    "jaccard": value,
                    "odd_support_views": int(first["support_views"][first_slot]),
                    "even_support_views": int(second["support_views"][second_slot]),
                    "stable": value >= stability_threshold,
                }
            )
            matched_first.add(first_slot)
            matched_second.add(second_slot)
        for split, ids, matched, matrix, opposite, axis in (
            ("odd", first_ids, matched_first, jaccard, second_ids, 1),
            ("even", second_ids, matched_second, jaccard, first_ids, 0),
        ):
            for local_index, slot in enumerate(ids):
                match = next(
                    (item for item in matches if item.get(f"{split}_slot") == int(slot)), None
                )
                if match is not None and match["jaccard"] >= stability_threshold:
                    continue
                values = matrix[local_index] if axis == 1 else matrix[:, local_index]
                best = int(values.argmax()) if values.size else -1
                unresolved.append(
                    {
                        "split": split,
                        "slot": int(slot),
                        "level": level,
                        "best_opposite_slot": int(opposite[best]) if best >= 0 else None,
                        "best_jaccard": float(values[best]) if best >= 0 else 0.0,
                    }
                )
    return matches, unresolved


def make_gate(metrics, args):
    checks = {
        "heldout_nll_improvement": metrics["relative_nll_improvement"] >= args.minimum_nll_improvement,
        "split_slot_stability": metrics["median_matched_jaccard"] >= args.minimum_split_stability,
        "enough_stable_slots": metrics["stable_slots"] >= args.minimum_stable_slots,
        "persistent_support_contract": metrics["minimum_slot_support_views"] >= args.minimum_persistence_views,
        "slot_count_agreement": metrics["slot_count_agreement"] >= args.minimum_slot_count_agreement,
        "capacity_not_saturated": not metrics["capacity_saturated"],
        "union_mass_not_trivial": metrics["mdl_union_mass_fraction"] >= args.minimum_union_mass_fraction,
        "union_mass_not_saturated": metrics["mdl_union_mass_fraction"] <= args.maximum_union_mass_fraction,
        "unresolved_certificate_written": metrics["unresolved_certificate_written"],
    }
    passed = all(checks.values())
    return {
        "pass": bool(passed),
        "decision": "PROCEED_TO_A48_1_CONTINUOUS_ENTITY_SEMANTICS" if passed else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": {
            "minimum_relative_nll_improvement": args.minimum_nll_improvement,
            "minimum_median_split_jaccard": args.minimum_split_stability,
            "minimum_stable_slots": args.minimum_stable_slots,
            "minimum_persistence_views": args.minimum_persistence_views,
            "minimum_slot_count_agreement": args.minimum_slot_count_agreement,
            "minimum_union_mass_fraction": args.minimum_union_mass_fraction,
            "maximum_union_mass_fraction": args.maximum_union_mass_fraction,
        },
    }


def incidence_entries(a47_audit_dir):
    directory = os.path.join(a47_audit_dir, "incidence_views")
    entries = []
    pattern = re.compile(r"^(\d{4})_(.+)\.npz$")
    for name in sorted(os.listdir(directory)):
        match = pattern.match(name)
        if match:
            entries.append(
                {
                    "view_index": int(match.group(1)),
                    "image_name": match.group(2),
                    "file": os.path.join("incidence_views", name),
                }
            )
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--a47_audit_dir", required=True)
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
    parser.add_argument("--maximum_slots", type=int, default=256)
    parser.add_argument("--evaluation_candidates", type=int, default=6)
    parser.add_argument("--union_relative_nll_penalty", type=float, default=0.05)
    parser.add_argument("--unresolved_association_threshold", type=float, default=0.20)
    parser.add_argument("--minimum_nll_improvement", type=float, default=0.10)
    parser.add_argument("--minimum_split_stability", type=float, default=0.80)
    parser.add_argument("--minimum_stable_slots", type=int, default=8)
    parser.add_argument("--minimum_slot_count_agreement", type=float, default=0.80)
    parser.add_argument("--minimum_union_mass_fraction", type=float, default=0.01)
    parser.add_argument("--maximum_union_mass_fraction", type=float, default=0.50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse persistent entity tomography: {output_dir}")
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
        and contract["visible_owner_censoring"]
        and not contract["evaluation_queries_or_labels_used"]
        and not contract["codebooks_trained"]
        and int(contract["fixed_seed"]) == args.seed
    ):
        raise ValueError("A47 source contract is incompatible with A48")
    entries = incidence_entries(args.a47_audit_dir)
    if len(entries) != 131:
        raise ValueError(f"A48 requires 131 incidence views, found {len(entries)}")
    views = load_prepared_views(args.a47_audit_dir, entries)
    odd = fit_persistent_slots(views, 0, args)
    even = fit_persistent_slots(views, 1, args)
    odd_to_even = evaluate_persistent_model(odd, views, 1, args)
    even_to_odd = evaluate_persistent_model(even, views, 0, args)
    matches, unresolved_slots = match_level_slots(
        odd, even, args.minimum_persistence_views, args.minimum_split_stability
    )
    unresolved_path = os.path.join(output_dir, "unresolved_slot_certificate.json")
    with open(unresolved_path, "w") as output:
        json.dump(
            {
                "unresolved_slots": unresolved_slots,
                "odd_to_even_unresolved_proposals": odd_to_even["unresolved"],
                "even_to_odd_unresolved_proposals": even_to_odd["unresolved"],
                "hard_assignment_for_unresolved_forbidden": True,
            },
            output,
            indent=2,
        )

    names = odd_to_even["mean_nll"].keys()
    mean_nll = {
        name: 0.5 * (odd_to_even["mean_nll"][name] + even_to_odd["mean_nll"][name])
        for name in names
    }
    improvement = (
        mean_nll["hard_persistent_single_id"] - mean_nll["persistent_mdl_union"]
    ) / max(mean_nll["hard_persistent_single_id"], 1e-8)
    jaccards = np.asarray([item["jaccard"] for item in matches], dtype=np.float32)
    stable = [item for item in matches if item["stable"]]
    odd_count = int(odd["profiles"].shape[0])
    even_count = int(even["profiles"].shape[0])
    union_mass = odd_to_even["mdl_union_mass"] + even_to_odd["mdl_union_mass"]
    positive_mass = odd_to_even["positive_mass"] + even_to_odd["positive_mass"]
    metrics = {
        "mean_heldout_balanced_mask_nll": mean_nll,
        "a47_reference_nll": a47_manifest["metrics"]["mean_heldout_balanced_mask_nll"],
        "relative_nll_improvement": float(improvement),
        "median_matched_jaccard": float(np.median(jaccards)) if jaccards.size else 0.0,
        "mean_matched_jaccard": float(jaccards.mean()) if jaccards.size else 0.0,
        "matched_slots": len(matches),
        "stable_slots": len(stable),
        "minimum_slot_support_views": int(
            min(odd["support_views"].min(), even["support_views"].min())
        ),
        "slot_count_agreement": min(odd_count, even_count) / max(odd_count, even_count),
        "capacity_saturated": bool(odd["capacity_saturated"] or even["capacity_saturated"]),
        "mdl_union_mass_fraction": float(union_mass / max(positive_mass, 1e-8)),
        "unresolved_slots": len(unresolved_slots),
        "unresolved_certificate_written": os.path.isfile(unresolved_path),
    }
    gate = make_gate(metrics, args)
    manifest = {
        "format_version": 1,
        "experiment": "A48.0_persistent_birth_mdl_entity_tomography",
        "representation": "level_preserving_persistent_entity_tracks",
        "scene": "ramen",
        "seed": args.seed,
        "odd_model": {
            "slots": odd_count,
            "capacity_saturated": odd["capacity_saturated"],
            "statistics": odd["statistics"],
        },
        "even_model": {
            "slots": even_count,
            "capacity_saturated": even["capacity_saturated"],
            "statistics": even["statistics"],
        },
        "heldout_directions": {
            "odd_train_even_eval": {key: value for key, value in odd_to_even.items() if key != "unresolved"},
            "even_train_odd_eval": {key: value for key, value in even_to_odd.items() if key != "unresolved"},
        },
        "slot_matching": {"matches": matches},
        "metrics": metrics,
        "gate": gate,
        "inputs": {
            "a47_audit_dir": os.path.abspath(args.a47_audit_dir),
            "a47_manifest_sha256": file_sha256(a47_manifest_path),
        },
        "source_contract": {
            "training_views_only": True,
            "raw_overlapping_proposals_reused_read_only": True,
            "raw_top45_incidence_reused_read_only": True,
            "level_preserving_tracks": True,
            "reciprocal_temporal_association": True,
            "minimum_three_view_birth_persistence": True,
            "view_exclusive_track_constraint": True,
            "mdl_union_penalty": args.union_relative_nll_penalty,
            "odd_even_independent_fit": True,
            "evaluation_queries_or_labels_used": False,
            "codebooks_trained": False,
            "fixed_seed": args.seed,
        },
        "artifacts": {"unresolved_slot_certificate": os.path.basename(unresolved_path)},
        "args": vars(args),
        "elapsed_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    with open(os.path.join(output_dir, "gate.json"), "w") as output:
        json.dump(gate, output, indent=2)
    print(json.dumps({"metrics": metrics, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
