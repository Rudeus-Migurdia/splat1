#!/usr/bin/env python
"""Audit multi-hypothesis entity factorization from raw overlapping proposals."""

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_packed_masks(packed_masks, flat_indices):
    packed_masks = np.asarray(packed_masks, dtype=np.uint8)
    flat_indices = np.asarray(flat_indices, dtype=np.int64)
    byte_indices = flat_indices // 8
    bit_indices = flat_indices % 8
    return ((packed_masks[:, byte_indices] >> bit_indices[None]) & 1).astype(bool)


def build_spatial_atoms(xyz, target_atoms, minimum_resolution=8, maximum_resolution=128):
    xyz = np.asarray(xyz, dtype=np.float32)
    lower = xyz.min(axis=0)
    span = np.maximum(xyz.max(axis=0) - lower, 1e-8)
    normalized = np.clip((xyz - lower) / span, 0.0, 1.0 - 1e-7)
    candidates = []
    lower_resolution = minimum_resolution
    upper_resolution = maximum_resolution
    while lower_resolution <= upper_resolution:
        resolution = (lower_resolution + upper_resolution) // 2
        coordinates = np.floor(normalized * resolution).astype(np.int32)
        keys = (
            coordinates[:, 0].astype(np.int64) * resolution * resolution
            + coordinates[:, 1].astype(np.int64) * resolution
            + coordinates[:, 2]
        )
        unique_count = np.unique(keys).size
        candidates.append((abs(unique_count - target_atoms), resolution, unique_count))
        if unique_count < target_atoms:
            lower_resolution = resolution + 1
        elif unique_count > target_atoms:
            upper_resolution = resolution - 1
        else:
            break
    _, resolution, _ = min(candidates)
    coordinates = np.floor(normalized * resolution).astype(np.int32)
    keys = (
        coordinates[:, 0].astype(np.int64) * resolution * resolution
        + coordinates[:, 1].astype(np.int64) * resolution
        + coordinates[:, 2]
    )
    _, atom_ids = np.unique(keys, return_inverse=True)
    return atom_ids.astype(np.int32), {
        "target_atoms": int(target_atoms),
        "grid_resolution": int(resolution),
        "num_atoms": int(atom_ids.max()) + 1,
        "lower_bound": lower.tolist(),
        "upper_bound": (lower + span).tolist(),
    }


def noisy_or(profiles):
    profiles = np.asarray(profiles, dtype=np.float32)
    if profiles.ndim == 1:
        return profiles
    return 1.0 - np.prod(1.0 - np.clip(profiles, 0.0, 1.0), axis=0)


def balanced_bernoulli_nll(target, prediction, visibility, epsilon=1e-4):
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), epsilon, 1.0 - epsilon)
    visibility = np.maximum(np.asarray(visibility, dtype=np.float32), 0.0)
    positive = visibility * target
    negative = visibility * (1.0 - target)
    positive_nll = -(positive * np.log(prediction)).sum() / max(float(positive.sum()), epsilon)
    negative_nll = -(negative * np.log1p(-prediction)).sum() / max(float(negative.sum()), epsilon)
    return 0.5 * (float(positive_nll) + float(negative_nll))


def soft_jaccard_batch(profiles, target, visible):
    profiles = np.asarray(profiles, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    visible = np.asarray(visible, dtype=bool)
    if not visible.any():
        return np.zeros(profiles.shape[0], dtype=np.float32)
    first = profiles[:, visible]
    second = target[visible][None]
    intersection = np.minimum(first, second).sum(axis=1)
    union = np.maximum(first, second).sum(axis=1)
    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union > 0.0)


def normalize_rows(values):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-8)


def select_proposals(payload, maximum_per_level, minimum_area_fraction, maximum_area_fraction):
    levels = payload["levels"].astype(np.int64)
    quality = payload["quality_score"].astype(np.float32)
    area = payload["area"].astype(np.float32)
    image_area = float(int(payload["image_height"]) * int(payload["image_width"]))
    area_fraction = area / image_area
    selected = []
    for level in range(4):
        candidates = np.flatnonzero(
            (levels == level)
            & (area_fraction >= minimum_area_fraction)
            & (area_fraction <= maximum_area_fraction)
        )
        order = np.lexsort((candidates, -quality[candidates]))
        selected.extend(candidates[order[:maximum_per_level]].tolist())
    return np.asarray(selected, dtype=np.int64)


def prepare_incidence_views(args, xyz, cache_manifest, proposal_manifest, output_dir):
    import torch
    from scipy.sparse import csr_matrix

    os.makedirs(output_dir, exist_ok=True)
    atom_ids, atom_contract = build_spatial_atoms(xyz, args.target_atoms)
    atom_path = os.path.join(output_dir, "gaussian_atom_ids.npy")
    np.save(atom_path, atom_ids)
    num_atoms = atom_contract["num_atoms"]
    proposal_entries = {item["image_name"]: item for item in proposal_manifest["views"]}
    prepared_dir = os.path.join(output_dir, "incidence_views")
    os.makedirs(prepared_dir, exist_ok=True)
    prepared_entries = []
    total_raw = 0
    total_selected = 0

    for view_index, cache_entry in enumerate(cache_manifest["views"]):
        image_name = cache_entry["image_name"]
        proposal_entry = proposal_entries.get(image_name)
        if proposal_entry is None:
            raise ValueError(f"No raw proposals for cached view {image_name}")
        output_path = os.path.join(prepared_dir, f"{view_index:04d}_{image_name}.npz")
        if os.path.isfile(output_path):
            with np.load(output_path) as prepared:
                prepared_entries.append(
                    {
                        "view_index": view_index,
                        "image_name": image_name,
                        "file": os.path.relpath(output_path, output_dir),
                        "num_proposals": int(prepared["coverage"].shape[0]),
                    }
                )
                total_selected += int(prepared["coverage"].shape[0])
            total_raw += int(proposal_entry["num_proposals"])
            continue

        proposal_path = os.path.join(args.proposal_dir, proposal_entry["file"])
        cache_path = os.path.join(args.cache_dir, cache_entry["cache"])
        with np.load(proposal_path) as proposal:
            selected = select_proposals(
                proposal,
                args.maximum_proposals_per_level,
                args.minimum_area_fraction,
                args.maximum_area_fraction,
            )
            descriptors = proposal["descriptors"][selected].astype(np.float32)
            levels = proposal["levels"][selected].astype(np.uint8)
            quality = proposal["quality_score"][selected].astype(np.float32)
            areas = proposal["area"][selected].astype(np.int32)
            packed = proposal["packed_masks"][selected]
            image_height = int(proposal["image_height"])
            image_width = int(proposal["image_width"])
            total_raw += int(proposal["levels"].shape[0])

        cache = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (int(cache["image_height"]), int(cache["image_width"])) != (image_height, image_width):
            raise ValueError(f"Proposal/cache resolution mismatch for {image_name}")
        sampled = cache["sampled_flat_indices"].numpy().astype(np.int64, copy=False)
        point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
        point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
        sampled_masks = sample_packed_masks(packed, sampled)

        pixel_indices = np.repeat(np.arange(point_ids.shape[0], dtype=np.int32), point_ids.shape[1])
        gaussian_indices = point_ids.reshape(-1)
        contribution = point_weights.reshape(-1)
        valid = (
            (gaussian_indices >= 0)
            & (gaussian_indices < atom_ids.shape[0])
            & np.isfinite(contribution)
            & (contribution > 0.0)
        )
        contribution_matrix = csr_matrix(
            (
                contribution[valid],
                (pixel_indices[valid], atom_ids[gaussian_indices[valid]]),
            ),
            shape=(point_ids.shape[0], num_atoms),
            dtype=np.float32,
        )
        contribution_matrix.sum_duplicates()
        visibility = np.asarray(contribution_matrix.sum(axis=0)).reshape(-1).astype(np.float32)
        incidence = (csr_matrix(sampled_masks.astype(np.float32)) @ contribution_matrix).toarray()
        coverage = np.divide(
            incidence,
            visibility[None],
            out=np.zeros_like(incidence, dtype=np.float32),
            where=visibility[None] > 1e-8,
        )
        coverage = np.clip(coverage, 0.0, 1.0)
        temporary = output_path + ".tmp"
        with open(temporary, "wb") as output:
            np.savez_compressed(
                output,
                coverage=coverage.astype(np.float16),
                visibility=visibility.astype(np.float16),
                descriptors=normalize_rows(descriptors).astype(np.float16),
                levels=levels,
                quality=quality.astype(np.float16),
                area=areas,
                raw_proposal_indices=selected.astype(np.int32),
            )
        os.replace(temporary, output_path)
        total_selected += selected.size
        prepared_entries.append(
            {
                "view_index": view_index,
                "image_name": image_name,
                "file": os.path.relpath(output_path, output_dir),
                "num_proposals": int(selected.size),
            }
        )
        print(
            f"[{view_index + 1}/{len(cache_manifest['views'])}] {image_name} "
            f"raw={proposal_entry['num_proposals']} selected={selected.size}",
            flush=True,
        )
        del cache, contribution_matrix, incidence, coverage, sampled_masks

    return prepared_entries, atom_contract, {
        "raw_proposals": total_raw,
        "selected_proposals": total_selected,
        "selection_fraction": total_selected / max(total_raw, 1),
        "gaussian_atom_ids": os.path.basename(atom_path),
    }


def load_prepared_views(output_dir, entries):
    views = []
    for entry in entries:
        with np.load(os.path.join(output_dir, entry["file"])) as payload:
            views.append(
                {
                    "view_index": int(entry["view_index"]),
                    "image_name": entry["image_name"],
                    "coverage": payload["coverage"].astype(np.float32),
                    "visibility": payload["visibility"].astype(np.float32),
                    "descriptors": normalize_rows(payload["descriptors"].astype(np.float32)),
                    "quality": payload["quality"].astype(np.float32),
                    "levels": payload["levels"].astype(np.int64),
                }
            )
    return views


def current_profiles(state):
    count = state["num_slots"]
    return np.divide(
        state["sums"][:count],
        state["observations"][:count],
        out=np.zeros_like(state["sums"][:count]),
        where=state["observations"][:count] > 1e-8,
    )


def current_descriptors(state):
    return normalize_rows(state["descriptor_sums"][: state["num_slots"]])


def add_slot(state, target, descriptor, visible, view_index, is_birth):
    slot = state["num_slots"]
    if slot >= state["sums"].shape[0]:
        return None
    state["num_slots"] += 1
    state["sums"][slot] = target * visible
    state["observations"][slot] = visible.astype(np.float32)
    state["descriptor_sums"][slot] = descriptor
    state["support_views"][slot].add(view_index)
    state["birth"][slot] = bool(is_birth)
    return slot


def update_slot(state, slot, target, descriptor, visible, view_index, weight=1.0):
    state["sums"][slot] += weight * target * visible
    state["observations"][slot] += weight * visible
    state["descriptor_sums"][slot] += weight * descriptor
    state["support_views"][slot].add(view_index)


def fit_entity_slots(views, train_parity, args):
    num_atoms = views[0]["coverage"].shape[1]
    state = {
        "sums": np.zeros((args.maximum_slots, num_atoms), dtype=np.float32),
        "observations": np.zeros((args.maximum_slots, num_atoms), dtype=np.float32),
        "descriptor_sums": np.zeros((args.maximum_slots, 512), dtype=np.float32),
        "support_views": [set() for _ in range(args.maximum_slots)],
        "birth": np.zeros(args.maximum_slots, dtype=bool),
        "num_slots": 0,
        "union_pairs": set(),
        "assignments": 0,
        "union_assignments": 0,
        "birth_assignments": 0,
    }
    for view in views:
        if view["view_index"] % 2 != train_parity:
            continue
        visible = view["visibility"] > args.minimum_visibility
        order = np.argsort(-view["quality"], kind="stable")
        for proposal_index in order:
            target = view["coverage"][proposal_index]
            descriptor = view["descriptors"][proposal_index]
            if state["num_slots"] == 0:
                add_slot(state, target, descriptor, visible, view["view_index"], False)
                state["assignments"] += 1
                continue
            profiles = current_profiles(state)
            descriptors = current_descriptors(state)
            jaccard = soft_jaccard_batch(profiles, target, visible)
            semantic = descriptors @ descriptor
            association = args.spatial_weight * jaccard + (1.0 - args.spatial_weight) * semantic
            candidate_count = min(args.association_candidates, state["num_slots"])
            candidates = np.argpartition(-association, candidate_count - 1)[:candidate_count]
            candidates = candidates[np.argsort(-association[candidates], kind="stable")]
            best = int(candidates[0])
            single_prediction = profiles[best]
            single_nll = balanced_bernoulli_nll(target, single_prediction, view["visibility"])
            selected_pair = None
            pair_nll = single_nll
            for second_value in candidates[1:]:
                second = int(second_value)
                prediction = noisy_or(profiles[[best, second]])
                value = balanced_bernoulli_nll(target, prediction, view["visibility"])
                if value < pair_nll:
                    pair_nll = value
                    selected_pair = (best, second)
            relative_pair_gain = (single_nll - pair_nll) / max(single_nll, 1e-8)
            residual = np.clip(target - single_prediction, 0.0, 1.0)
            residual_fraction = float((residual * view["visibility"]).sum()) / max(
                float((target * view["visibility"]).sum()), 1e-8
            )
            should_birth = (
                association[best] < args.birth_association_threshold
                or residual_fraction >= args.birth_residual_fraction
            )
            if selected_pair is not None and relative_pair_gain >= args.minimum_union_nll_gain:
                first, second = selected_pair
                first_target = np.clip(
                    (target - profiles[second]) / np.maximum(1.0 - profiles[second], 1e-4), 0.0, 1.0
                )
                second_target = np.clip(
                    (target - profiles[first]) / np.maximum(1.0 - profiles[first], 1e-4), 0.0, 1.0
                )
                update_slot(state, first, first_target, descriptor, visible, view["view_index"], 0.5)
                update_slot(state, second, second_target, descriptor, visible, view["view_index"], 0.5)
                state["union_pairs"].add(tuple(sorted((first, second))))
                state["union_assignments"] += 1
            else:
                update_slot(state, best, target, descriptor, visible, view["view_index"])
            if should_birth and state["num_slots"] < args.maximum_slots:
                birth_target = residual if association[best] >= args.birth_association_threshold else target
                if float((birth_target * view["visibility"]).sum()) > args.minimum_birth_mass:
                    add_slot(
                        state,
                        birth_target,
                        descriptor,
                        visible,
                        view["view_index"],
                        True,
                    )
                    state["birth_assignments"] += 1
            state["assignments"] += 1
    count = state["num_slots"]
    return {
        "profiles": current_profiles(state),
        "descriptors": current_descriptors(state),
        "support_views": np.asarray([len(value) for value in state["support_views"][:count]], dtype=np.int32),
        "birth": state["birth"][:count].copy(),
        "union_pairs": sorted(state["union_pairs"]),
        "assignments": state["assignments"],
        "union_assignments": state["union_assignments"],
        "birth_assignments": state["birth_assignments"],
    }


def hard_owner_profiles(profiles):
    hard = np.full_like(profiles, 0.02, dtype=np.float32)
    if not profiles.size:
        return hard
    owners = profiles.argmax(axis=0)
    confidence = profiles.max(axis=0)
    atoms = np.flatnonzero(confidence >= 0.25)
    hard[owners[atoms], atoms] = 0.98
    return hard


def evaluate_split(model, views, heldout_parity, args):
    profiles = model["profiles"]
    descriptors = model["descriptors"]
    hard_profiles = hard_owner_profiles(profiles)
    learned_pairs = {tuple(pair) for pair in model["union_pairs"]}
    totals = {name: 0.0 for name in ("hard_single_id", "pair_graph", "single_path_noisy_or", "multi_hypothesis_noisy_or")}
    proposal_weight = 0.0
    positive_mass_total = 0.0
    nontrivial_mass = 0.0
    union_count = 0
    birth_count = 0
    unresolved = []
    for view in views:
        if view["view_index"] % 2 != heldout_parity:
            continue
        visible = view["visibility"] > args.minimum_visibility
        for proposal_index, target in enumerate(view["coverage"]):
            descriptor = view["descriptors"][proposal_index]
            jaccard = soft_jaccard_batch(profiles, target, visible)
            semantic = descriptors @ descriptor
            association = args.spatial_weight * jaccard + (1.0 - args.spatial_weight) * semantic
            candidate_count = min(args.evaluation_candidates, profiles.shape[0])
            candidates = np.argpartition(-association, candidate_count - 1)[:candidate_count]
            candidates = candidates[np.argsort(-association[candidates], kind="stable")]
            weight = max(float(view["quality"][proposal_index]), 1e-4)
            mass = float((target * view["visibility"]).sum())

            hard_values = [
                balanced_bernoulli_nll(target, hard_profiles[int(slot)], view["visibility"])
                for slot in candidates
            ]
            soft_values = [
                balanced_bernoulli_nll(target, profiles[int(slot)], view["visibility"])
                for slot in candidates
            ]
            hard_nll = min(hard_values)
            single_nll = min(soft_values)
            best_single = int(candidates[int(np.argmin(soft_values))])
            pair_nll = single_nll
            multi_nll = single_nll
            best_multi = (best_single,)
            for first_index in range(len(candidates)):
                for second_index in range(first_index + 1, len(candidates)):
                    pair = tuple(sorted((int(candidates[first_index]), int(candidates[second_index]))))
                    prediction = noisy_or(profiles[list(pair)])
                    value = balanced_bernoulli_nll(target, prediction, view["visibility"])
                    if pair in learned_pairs and value < pair_nll:
                        pair_nll = value
                    if value < multi_nll:
                        multi_nll = value
                        best_multi = pair
            totals["hard_single_id"] += weight * hard_nll
            totals["pair_graph"] += weight * pair_nll
            totals["single_path_noisy_or"] += weight * single_nll
            totals["multi_hypothesis_noisy_or"] += weight * multi_nll
            proposal_weight += weight
            positive_mass_total += mass
            is_union = len(best_multi) == 2
            uses_birth = bool(model["birth"][list(best_multi)].any())
            if is_union:
                union_count += 1
            if uses_birth:
                birth_count += 1
            if is_union or uses_birth:
                nontrivial_mass += mass
            if association[best_single] < args.unresolved_association_threshold:
                unresolved.append(
                    {
                        "view_index": int(view["view_index"]),
                        "image_name": view["image_name"],
                        "proposal_index": int(proposal_index),
                        "best_slot": best_single,
                        "association": float(association[best_single]),
                    }
                )
    return {
        "mean_nll": {key: value / max(proposal_weight, 1e-8) for key, value in totals.items()},
        "proposal_weight": proposal_weight,
        "positive_mass": positive_mass_total,
        "nontrivial_mass": nontrivial_mass,
        "nontrivial_mass_fraction": nontrivial_mass / max(positive_mass_total, 1e-8),
        "union_proposals": union_count,
        "birth_slot_proposals": birth_count,
        "unresolved_proposals": unresolved,
    }


def match_split_slots(first, second, minimum_views, stability_threshold):
    from scipy.optimize import linear_sum_assignment

    first_ids = np.flatnonzero(first["support_views"] >= minimum_views)
    second_ids = np.flatnonzero(second["support_views"] >= minimum_views)
    if not first_ids.size or not second_ids.size:
        return [], [], {"eligible_first": int(first_ids.size), "eligible_second": int(second_ids.size)}
    first_binary = (first["profiles"][first_ids] >= 0.5).astype(np.int32)
    second_binary = (second["profiles"][second_ids] >= 0.5).astype(np.int32)
    intersection = first_binary @ second_binary.T
    union = first_binary.sum(axis=1)[:, None] + second_binary.sum(axis=1)[None] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection, dtype=np.float32),
        where=union > 0,
    )
    rows, columns = linear_sum_assignment(-jaccard)
    matches = []
    matched_first = set()
    matched_second = set()
    for row, column in zip(rows, columns):
        first_slot = int(first_ids[row])
        second_slot = int(second_ids[column])
        value = float(jaccard[row, column])
        matches.append(
            {
                "odd_slot": first_slot,
                "even_slot": second_slot,
                "jaccard": value,
                "odd_support_views": int(first["support_views"][first_slot]),
                "even_support_views": int(second["support_views"][second_slot]),
                "stable": value >= stability_threshold,
            }
        )
        matched_first.add(first_slot)
        matched_second.add(second_slot)
    unresolved = []
    for split_name, slot_ids, matched, matrix, opposite_ids, axis in (
        ("odd", first_ids, matched_first, jaccard, second_ids, 1),
        ("even", second_ids, matched_second, jaccard, first_ids, 0),
    ):
        for local_index, slot in enumerate(slot_ids):
            if int(slot) in matched:
                match = next(item for item in matches if item[f"{split_name}_slot"] == int(slot))
                if match["jaccard"] >= stability_threshold:
                    continue
            values = matrix[local_index] if axis == 1 else matrix[:, local_index]
            best_index = int(values.argmax()) if values.size else -1
            unresolved.append(
                {
                    "split": split_name,
                    "slot": int(slot),
                    "best_opposite_slot": int(opposite_ids[best_index]) if best_index >= 0 else None,
                    "best_jaccard": float(values[best_index]) if best_index >= 0 else 0.0,
                }
            )
    return matches, unresolved, {
        "eligible_first": int(first_ids.size),
        "eligible_second": int(second_ids.size),
    }


def make_gate(metrics, minimum_nll_improvement, minimum_stability, minimum_nontrivial_mass):
    checks = {
        "heldout_mask_nll_improvement": metrics["relative_nll_improvement"] >= minimum_nll_improvement,
        "split_slot_stability": metrics["median_matched_jaccard"] >= minimum_stability,
        "stable_slots_exist": metrics["stable_slots"] > 0,
        "stable_slots_have_three_views_per_split": metrics["stable_slot_support_valid"],
        "nontrivial_union_or_birth_mass": metrics["nontrivial_mass_fraction"] >= minimum_nontrivial_mass,
        "unresolved_certificate_written": metrics["unresolved_certificate_written"],
    }
    passed = all(checks.values())
    return {
        "pass": bool(passed),
        "decision": "PROCEED_TO_A47_1_CONTINUOUS_ENTITY_SEMANTICS" if passed else "STOP_BEFORE_CONTINUOUS_SEMANTICS_AND_CODEBOOKS",
        "checks": {key: bool(value) for key, value in checks.items()},
        "thresholds": {
            "minimum_relative_nll_improvement": minimum_nll_improvement,
            "minimum_median_split_slot_jaccard": minimum_stability,
            "minimum_views_per_stable_slot_per_split": 3,
            "minimum_nontrivial_heldout_mass_fraction": minimum_nontrivial_mass,
        },
    }


def load_geometry(path, expected_count):
    import torch

    model_params, iteration = torch.load(path, map_location="cpu", weights_only=False)
    xyz = model_params[1].detach().float().numpy()
    if xyz.shape != (expected_count, 3):
        raise ValueError("Geometry checkpoint Gaussian count does not match cache")
    return np.asarray(xyz, dtype=np.float32), int(iteration)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--proposal_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--target_atoms", type=int, default=8192)
    parser.add_argument("--maximum_proposals_per_level", type=int, default=24)
    parser.add_argument("--minimum_area_fraction", type=float, default=0.001)
    parser.add_argument("--maximum_area_fraction", type=float, default=0.90)
    parser.add_argument("--maximum_slots", type=int, default=192)
    parser.add_argument("--association_candidates", type=int, default=6)
    parser.add_argument("--evaluation_candidates", type=int, default=6)
    parser.add_argument("--spatial_weight", type=float, default=0.80)
    parser.add_argument("--birth_association_threshold", type=float, default=0.30)
    parser.add_argument("--birth_residual_fraction", type=float, default=0.35)
    parser.add_argument("--minimum_birth_mass", type=float, default=1.0)
    parser.add_argument("--minimum_union_nll_gain", type=float, default=0.05)
    parser.add_argument("--unresolved_association_threshold", type=float, default=0.20)
    parser.add_argument("--minimum_visibility", type=float, default=1e-4)
    parser.add_argument("--minimum_split_views", type=int, default=3)
    parser.add_argument("--minimum_nll_improvement", type=float, default=0.10)
    parser.add_argument("--minimum_split_stability", type=float, default=0.80)
    parser.add_argument("--minimum_nontrivial_mass_fraction", type=float, default=0.01)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse A47 identifiability audit: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    cache_manifest_path = os.path.join(args.cache_dir, "manifest.json")
    proposal_manifest_path = os.path.join(args.proposal_dir, "manifest.json")
    with open(cache_manifest_path) as source:
        cache_manifest = json.load(source)
    with open(proposal_manifest_path) as source:
        proposal_manifest = json.load(source)
    if not cache_manifest.get("raw_contribution_weights") or int(cache_manifest.get("topk", 0)) < 45:
        raise ValueError("A47 requires raw top-45 T*alpha contribution weights")
    if proposal_manifest.get("representation") != "overlapping_pre_flatten_sam_proposals":
        raise ValueError("A47 requires raw overlapping pre-flatten proposals")
    if proposal_manifest["contract"]["custom_flattening_or_mask_nms_applied"]:
        raise ValueError("Raw proposal cache has already been flattened")
    if int(proposal_manifest["contract"]["seed"]) != args.seed:
        raise ValueError("Raw proposal seed does not match A47 seed")
    num_gaussians = int(cache_manifest["num_gaussians"])
    xyz, checkpoint_iteration = load_geometry(args.geometry_checkpoint, num_gaussians)
    entries, atom_contract, proposal_statistics = prepare_incidence_views(
        args, xyz, cache_manifest, proposal_manifest, output_dir
    )
    views = load_prepared_views(output_dir, entries)
    odd_model = fit_entity_slots(views, 0, args)
    even_model = fit_entity_slots(views, 1, args)
    odd_to_even = evaluate_split(odd_model, views, 1, args)
    even_to_odd = evaluate_split(even_model, views, 0, args)
    matches, unresolved_slots, match_counts = match_split_slots(
        odd_model, even_model, args.minimum_split_views, args.minimum_split_stability
    )
    unresolved_path = os.path.join(output_dir, "unresolved_slot_certificate.json")
    with open(unresolved_path, "w") as output:
        json.dump(
            {
                "unresolved_slots": unresolved_slots,
                "odd_to_even_unresolved_proposals": odd_to_even["unresolved_proposals"],
                "even_to_odd_unresolved_proposals": even_to_odd["unresolved_proposals"],
                "hard_assignment_for_unresolved_forbidden": True,
            },
            output,
            indent=2,
        )
    model_names = odd_to_even["mean_nll"].keys()
    mean_nll = {
        name: 0.5 * (odd_to_even["mean_nll"][name] + even_to_odd["mean_nll"][name])
        for name in model_names
    }
    improvement = (
        mean_nll["hard_single_id"] - mean_nll["multi_hypothesis_noisy_or"]
    ) / max(mean_nll["hard_single_id"], 1e-8)
    matched_jaccards = np.asarray([item["jaccard"] for item in matches], dtype=np.float32)
    stable = [item for item in matches if item["stable"]]
    nontrivial_mass = odd_to_even["nontrivial_mass"] + even_to_odd["nontrivial_mass"]
    positive_mass = odd_to_even["positive_mass"] + even_to_odd["positive_mass"]
    metrics = {
        "mean_heldout_balanced_mask_nll": mean_nll,
        "relative_nll_improvement": float(improvement),
        "median_matched_jaccard": float(np.median(matched_jaccards)) if matched_jaccards.size else 0.0,
        "mean_matched_jaccard": float(matched_jaccards.mean()) if matched_jaccards.size else 0.0,
        "matched_slots": len(matches),
        "stable_slots": len(stable),
        "stable_slot_support_valid": bool(
            stable
            and all(
                item["odd_support_views"] >= args.minimum_split_views
                and item["even_support_views"] >= args.minimum_split_views
                for item in stable
            )
        ),
        "nontrivial_mass_fraction": float(nontrivial_mass / max(positive_mass, 1e-8)),
        "unresolved_slots": len(unresolved_slots),
        "unresolved_certificate_written": os.path.isfile(unresolved_path),
    }
    gate = make_gate(
        metrics,
        args.minimum_nll_improvement,
        args.minimum_split_stability,
        args.minimum_nontrivial_mass_fraction,
    )
    manifest = {
        "format_version": 1,
        "experiment": "A47.0_raw_proposal_identifiability_audit",
        "representation": "multi_hypothesis_entity_tomography_diagnostic",
        "scene": proposal_manifest["scene"],
        "seed": args.seed,
        "num_views": len(views),
        "num_gaussians": num_gaussians,
        "spatial_atoms": atom_contract,
        "proposal_statistics": proposal_statistics,
        "models": {
            "hard_single_id": "one-hot atom ownership and one slot per proposal",
            "pair_graph": "single slot or a pair observed during training",
            "single_path_noisy_or": "soft atom ownership and one slot per proposal",
            "multi_hypothesis_noisy_or": "soft ownership, unrestricted two-slot union, adaptive training birth",
        },
        "odd_model": {
            "slots": int(odd_model["profiles"].shape[0]),
            "birth_slots": int(odd_model["birth"].sum()),
            "union_pairs": len(odd_model["union_pairs"]),
            "assignments": odd_model["assignments"],
        },
        "even_model": {
            "slots": int(even_model["profiles"].shape[0]),
            "birth_slots": int(even_model["birth"].sum()),
            "union_pairs": len(even_model["union_pairs"]),
            "assignments": even_model["assignments"],
        },
        "heldout_directions": {
            "odd_train_even_eval": {key: value for key, value in odd_to_even.items() if key != "unresolved_proposals"},
            "even_train_odd_eval": {key: value for key, value in even_to_odd.items() if key != "unresolved_proposals"},
        },
        "slot_matching": {"counts": match_counts, "matches": matches},
        "metrics": metrics,
        "gate": gate,
        "artifacts": {
            "gaussian_atom_ids": proposal_statistics["gaussian_atom_ids"],
            "unresolved_slot_certificate": os.path.basename(unresolved_path),
        },
        "inputs": {
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
            "geometry_checkpoint_sha256": file_sha256(args.geometry_checkpoint),
            "geometry_checkpoint_iteration": checkpoint_iteration,
            "cache_dir": os.path.abspath(args.cache_dir),
            "cache_manifest_sha256": file_sha256(cache_manifest_path),
            "proposal_dir": os.path.abspath(args.proposal_dir),
            "proposal_manifest_sha256": file_sha256(proposal_manifest_path),
        },
        "source_contract": {
            "training_views_only": True,
            "raw_overlapping_proposals": True,
            "raw_top45_talpha": True,
            "visible_owner_censoring": True,
            "odd_even_independent_fit": True,
            "hungarian_slot_alignment": True,
            "evaluation_queries_or_labels_used": False,
            "codebooks_trained": False,
            "fixed_seed": args.seed,
        },
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
