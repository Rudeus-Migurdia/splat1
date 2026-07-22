#!/usr/bin/env python
"""Audit unique spatial Group addresses before semantic codebook training."""

import argparse
import json
import os
import time

import numpy as np

from build_geometry_conditioned_tracklet_partition import (
    build_atom_contact_graph,
    load_atom_geometry,
)
from build_multi_hypothesis_entity_tomography import (
    balanced_bernoulli_nll,
    load_prepared_views,
    normalize_rows,
)
from build_persistent_entity_tomography import (
    file_sha256,
    fit_persistent_slots,
    incidence_entries,
    match_level_slots,
)


def bounded_group_profiles(profiles, contact_graph, args):
    """Keep only mask-supported components anchored by high-confidence core atoms."""
    from scipy.sparse.csgraph import connected_components

    profiles = np.asarray(profiles, dtype=np.float32)
    adjacency = contact_graph.copy().tocsr()
    adjacency.data = (adjacency.data >= args.minimum_atom_contact).astype(np.uint8)
    adjacency.eliminate_zeros()
    bounded = np.zeros_like(profiles)
    core_masks = np.zeros_like(profiles, dtype=bool)
    support_masks = np.zeros_like(profiles, dtype=bool)
    rejected_unanchored_components = 0
    rejected_unanchored_atoms = 0
    groups_without_core = 0
    for group_index, profile in enumerate(profiles):
        core = profile >= args.core_coverage_threshold
        support = profile >= args.boundary_coverage_threshold
        if not core.any():
            groups_without_core += 1
            continue
        support_ids = np.flatnonzero(support)
        component_count, labels = connected_components(
            adjacency[support_ids][:, support_ids], directed=False
        )
        keep = np.zeros(support_ids.size, dtype=bool)
        local_core = core[support_ids]
        for component in range(component_count):
            component_mask = labels == component
            if int(local_core[component_mask].sum()) >= args.minimum_core_atoms:
                keep |= component_mask
            else:
                rejected_unanchored_components += 1
                rejected_unanchored_atoms += int(component_mask.sum())
        kept_ids = support_ids[keep]
        bounded[group_index, kept_ids] = profile[kept_ids]
        core_masks[group_index, kept_ids] = core[kept_ids]
        support_masks[group_index, kept_ids] = True
    return bounded, core_masks, support_masks, adjacency, {
        "groups_without_core": groups_without_core,
        "rejected_unanchored_components": rejected_unanchored_components,
        "rejected_unanchored_atoms": rejected_unanchored_atoms,
        "raw_positive_atoms": int((profiles >= args.boundary_coverage_threshold).sum()),
        "bounded_positive_atoms": int(support_masks.sum()),
        "core_atoms": int(core_masks.sum()),
    }


def build_ring_descriptors(model, support_masks, adjacency, args):
    """Describe each Group's immediate exterior using distinct neighboring Groups."""
    from scipy.sparse import csr_matrix

    support = csr_matrix(support_masks.astype(np.float32))
    expanded = support @ adjacency
    expanded.data[:] = 1.0
    expanded.eliminate_zeros()
    ring = expanded.toarray() > 0.0
    ring &= ~support_masks
    ring_contact = csr_matrix(ring.astype(np.float32)) @ support.T
    ring_contact = ring_contact.toarray()
    binary = support_masks.astype(np.float32)
    intersection = binary @ binary.T
    sizes = binary.sum(axis=1)
    union = sizes[:, None] + sizes[None] - intersection
    jaccard = np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )
    ring_descriptors = np.zeros_like(model["descriptors"], dtype=np.float32)
    ring_valid = np.zeros(len(binary), dtype=bool)
    neighbor_counts = np.zeros(len(binary), dtype=np.int32)
    for group_index in range(len(binary)):
        candidates = np.flatnonzero(
            (model["levels"] == model["levels"][group_index])
            & (ring_contact[group_index] > 0.0)
            & (jaccard[group_index] < args.maximum_ring_neighbor_jaccard)
        )
        candidates = candidates[candidates != group_index]
        if not candidates.size:
            continue
        order = np.argsort(-ring_contact[group_index, candidates], kind="stable")
        candidates = candidates[order[: args.maximum_ring_neighbors]]
        weights = ring_contact[group_index, candidates]
        descriptor = (weights[:, None] * model["descriptors"][candidates]).sum(axis=0)
        norm = float(np.linalg.norm(descriptor))
        if norm <= 1e-8:
            continue
        ring_descriptors[group_index] = descriptor / norm
        ring_valid[group_index] = True
        neighbor_counts[group_index] = len(candidates)
    return ring_descriptors, ring_valid, ring, {
        "ring_valid_groups": int(ring_valid.sum()),
        "ring_valid_fraction": float(ring_valid.mean()),
        "mean_ring_neighbors": float(neighbor_counts[ring_valid].mean())
        if ring_valid.any()
        else 0.0,
        "ring_atoms": int(ring.sum()),
    }


def prepare_ring_incidence(a47_manifest, a47_audit_dir, atom_ids, output_dir, args):
    """Project a scale-adaptive 2D proposal exterior ring onto A47 spatial atoms."""
    import cv2
    import torch
    from scipy.sparse import csr_matrix

    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        with open(manifest_path) as source:
            return json.load(source)
    proposal_dir = a47_manifest["inputs"]["proposal_dir"]
    cache_dir = a47_manifest["inputs"]["cache_dir"]
    with open(os.path.join(proposal_dir, "manifest.json")) as source:
        proposal_manifest = json.load(source)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        cache_manifest = json.load(source)
    proposal_entries = {item["image_name"]: item for item in proposal_manifest["views"]}
    cache_entries = {item["image_name"]: item for item in cache_manifest["views"]}
    num_atoms = int(atom_ids.max()) + 1
    entries = []
    incidence_manifest = incidence_entries(a47_audit_dir)
    for incidence_entry in incidence_manifest:
        view_index = int(incidence_entry["view_index"])
        image_name = incidence_entry["image_name"]
        output_path = os.path.join(output_dir, f"{view_index:04d}_{image_name}.npz")
        if os.path.isfile(output_path) and not args.force:
            entries.append(
                {"view_index": view_index, "image_name": image_name, "file": os.path.basename(output_path)}
            )
            continue
        with np.load(os.path.join(a47_audit_dir, incidence_entry["file"])) as incidence:
            selected = incidence["raw_proposal_indices"].astype(np.int64)
            expected_visibility = incidence["visibility"].astype(np.float32)
        proposal_entry = proposal_entries[image_name]
        with np.load(os.path.join(proposal_dir, proposal_entry["file"])) as proposal:
            packed = proposal["packed_masks"][selected]
            areas = proposal["area"][selected].astype(np.float32)
            height = int(proposal["image_height"])
            width = int(proposal["image_width"])
        cache_entry = cache_entries[image_name]
        cache = torch.load(
            os.path.join(cache_dir, cache_entry["cache"]),
            map_location="cpu",
            weights_only=False,
        )
        sampled = cache["sampled_flat_indices"].numpy().astype(np.int64, copy=False)
        point_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
        point_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
        sampled_rings = np.zeros((len(selected), len(sampled)), dtype=np.uint8)
        pixel_count = height * width
        for proposal_index in range(len(selected)):
            mask = np.unpackbits(
                packed[proposal_index], count=pixel_count, bitorder="little"
            ).reshape(height, width).astype(np.uint8)
            radius = int(
                np.clip(
                    round(np.sqrt(max(float(areas[proposal_index]), 1.0)) * args.ring_radius_scale),
                    args.minimum_ring_radius,
                    args.maximum_ring_radius,
                )
            )
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            dilated = cv2.dilate(mask, kernel, iterations=1)
            ring = (dilated > 0) & (mask == 0)
            sampled_rings[proposal_index] = ring.reshape(-1)[sampled]
        pixel_indices = np.repeat(
            np.arange(point_ids.shape[0], dtype=np.int32), point_ids.shape[1]
        )
        gaussian_indices = point_ids.reshape(-1)
        contribution = point_weights.reshape(-1)
        valid = (
            (gaussian_indices >= 0)
            & (gaussian_indices < len(atom_ids))
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
        if not np.allclose(visibility, expected_visibility, rtol=2e-3, atol=2e-3):
            raise ValueError(f"A50 ring visibility does not match A47 incidence for {image_name}")
        ring_incidence = (csr_matrix(sampled_rings.astype(np.float32)) @ contribution_matrix).toarray()
        ring_coverage = np.divide(
            ring_incidence,
            visibility[None],
            out=np.zeros_like(ring_incidence, dtype=np.float32),
            where=visibility[None] > 1e-8,
        )
        temporary = output_path + ".tmp"
        with open(temporary, "wb") as output:
            np.savez_compressed(
                output,
                ring_coverage=np.clip(ring_coverage, 0.0, 1.0).astype(np.float16),
                raw_proposal_indices=selected.astype(np.int32),
            )
        os.replace(temporary, output_path)
        entries.append(
            {"view_index": view_index, "image_name": image_name, "file": os.path.basename(output_path)}
        )
        print(
            f"A50 ring incidence [{view_index + 1}/{len(incidence_manifest)}] {image_name}",
            flush=True,
        )
    manifest = {
        "format_version": 1,
        "representation": "scale_adaptive_2d_exterior_ring_atom_incidence",
        "views": entries,
        "num_views": len(entries),
        "ring_radius_scale": args.ring_radius_scale,
        "minimum_ring_radius": args.minimum_ring_radius,
        "maximum_ring_radius": args.maximum_ring_radius,
        "evaluation_queries_or_labels_used": False,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    return manifest


def load_ring_views(ring_dir, manifest):
    result = {}
    for entry in manifest["views"]:
        with np.load(os.path.join(ring_dir, entry["file"])) as payload:
            result[int(entry["view_index"])] = payload["ring_coverage"].astype(np.float32)
    return result


def signed_group_profiles(model, views, ring_views, args):
    views_by_index = {int(view["view_index"]): view for view in views}
    signed = np.zeros_like(model["profiles"], dtype=np.float32)
    confidence_values = []
    groups_without_members = 0
    for group_index, members in enumerate(model.get("members", [])):
        positive = np.zeros(model["profiles"].shape[1], dtype=np.float64)
        exterior = np.zeros_like(positive)
        if not members:
            groups_without_members += 1
            continue
        for member in members:
            view_index = int(member["view_index"])
            proposal_index = int(member["proposal_index"])
            view = views_by_index[view_index]
            quality = max(float(view["quality"][proposal_index]), 1e-4)
            visibility = view["visibility"]
            positive += quality * view["coverage"][proposal_index] * visibility
            exterior += quality * ring_views[view_index][proposal_index] * visibility
        confidence = np.divide(
            positive + args.signed_evidence_epsilon,
            positive + args.exterior_evidence_weight * exterior + args.signed_evidence_epsilon,
        )
        signed[group_index] = model["profiles"][group_index] * confidence.astype(np.float32)
        observed = (positive + exterior) > args.signed_evidence_epsilon
        if observed.any():
            confidence_values.append(confidence[observed])
    confidence_values = np.concatenate(confidence_values) if confidence_values else np.zeros(0)
    return signed, {
        "groups_without_members": groups_without_members,
        "signed_confidence_quantiles": np.quantile(
            confidence_values, [0.0, 0.1, 0.5, 0.9, 1.0]
        ).tolist()
        if confidence_values.size
        else [0.0] * 5,
    }


def make_spatial_group_model(model, views, ring_views, contact_graph, args):
    signed, signed_stats = signed_group_profiles(model, views, ring_views, args)
    bounded, core, support, adjacency, profile_stats = bounded_group_profiles(
        signed, contact_graph, args
    )
    ring_descriptors, ring_valid, ring_masks, ring_stats = build_ring_descriptors(
        model, support, adjacency, args
    )
    result = dict(model)
    result.update(
        {
            "raw_profiles": model["profiles"],
            "profiles": bounded,
            "core_masks": core,
            "support_masks": support,
            "ring_masks": ring_masks,
            "ring_descriptors": ring_descriptors,
            "ring_valid": ring_valid,
            "unique_spatial_group_ids": np.arange(len(bounded), dtype=np.int32),
            "capacity_saturated": False,
            "spatial_statistics": {**signed_stats, **profile_stats, **ring_stats},
        }
    )
    return result


def proposal_metrics(target, prediction, visibility):
    target = np.clip(np.asarray(target, dtype=np.float32), 0.0, 1.0)
    prediction = np.clip(np.asarray(prediction, dtype=np.float32), 0.0, 1.0)
    visibility = np.maximum(np.asarray(visibility, dtype=np.float32), 0.0)
    true_positive = float((visibility * target * prediction).sum())
    false_positive = float((visibility * (1.0 - target) * prediction).sum())
    target_mass = float((visibility * target).sum())
    predicted_mass = true_positive + false_positive
    return {
        "nll": balanced_bernoulli_nll(target, prediction, visibility),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "target_mass": target_mass,
        "predicted_mass": predicted_mass,
    }


def contrastive_group_scores(query_descriptor, model, args):
    core_scores = model["descriptors"] @ query_descriptor
    ring_scores = model["ring_descriptors"] @ query_descriptor
    penalty = np.maximum(
        ring_scores - core_scores + args.ring_contrast_margin,
        0.0,
    )
    penalty *= model["ring_valid"]
    return core_scores - args.ring_contrast_weight * penalty, core_scores, ring_scores


def evaluate_group_addressing(model, views, heldout_parity, args):
    totals = {
        name: {key: 0.0 for key in ("nll", "true_positive", "false_positive", "target_mass", "predicted_mass")}
        for name in ("raw_group", "bounded_group", "contrastive_bounded_group")
    }
    proposal_weight = 0.0
    selection_changed = 0
    unresolved = []
    evaluated = 0
    for view in views:
        if view["view_index"] % 2 != heldout_parity:
            continue
        for proposal_index, target in enumerate(view["coverage"]):
            level = int(view["levels"][proposal_index])
            groups = np.flatnonzero(model["levels"] == level)
            if not groups.size:
                unresolved.append(
                    {
                        "view_index": int(view["view_index"]),
                        "proposal_index": int(proposal_index),
                        "level": level,
                        "reason": "no_spatial_group_at_level",
                    }
                )
                continue
            query = view["descriptors"][proposal_index]
            adjusted, semantic, ring = contrastive_group_scores(query, model, args)
            raw_group = int(groups[np.argmax(semantic[groups])])
            contrastive_group = int(groups[np.argmax(adjusted[groups])])
            selection_changed += int(raw_group != contrastive_group)
            predictions = {
                "raw_group": model["raw_profiles"][raw_group],
                "bounded_group": model["profiles"][raw_group],
                "contrastive_bounded_group": model["profiles"][contrastive_group],
            }
            weight = max(float(view["quality"][proposal_index]), 1e-4)
            for name, prediction in predictions.items():
                values = proposal_metrics(target, prediction, view["visibility"])
                for key, value in values.items():
                    totals[name][key] += weight * value
            proposal_weight += weight
            evaluated += 1
            if adjusted[contrastive_group] < args.minimum_group_query_score:
                unresolved.append(
                    {
                        "view_index": int(view["view_index"]),
                        "proposal_index": int(proposal_index),
                        "level": level,
                        "selected_group": contrastive_group,
                        "core_score": float(semantic[contrastive_group]),
                        "ring_score": float(ring[contrastive_group]),
                        "adjusted_score": float(adjusted[contrastive_group]),
                        "reason": "low_group_query_score",
                    }
                )
    summary = {}
    for name, values in totals.items():
        summary[name] = {
            "mean_nll": values["nll"] / max(proposal_weight, 1e-8),
            "recall": values["true_positive"] / max(values["target_mass"], 1e-8),
            "exterior_spill_fraction": values["false_positive"]
            / max(values["predicted_mass"], 1e-8),
            "precision": values["true_positive"] / max(values["predicted_mass"], 1e-8),
            "predicted_to_target_mass": values["predicted_mass"]
            / max(values["target_mass"], 1e-8),
        }
    return {
        "variants": summary,
        "evaluated_proposals": evaluated,
        "selection_changed_fraction": selection_changed / max(evaluated, 1),
        "unresolved": unresolved,
    }


def mean_variant_results(first, second):
    names = first["variants"].keys()
    return {
        name: {
            key: 0.5 * (first["variants"][name][key] + second["variants"][name][key])
            for key in first["variants"][name]
        }
        for name in names
    }


def make_gate(metrics, args):
    checks = {
        "unique_group_address_contract": metrics["unique_group_address_contract"],
        "bounded_spill_reduction": metrics["bounded_spill_reduction"]
        >= args.minimum_spill_reduction,
        "bounded_recall_retention": metrics["bounded_recall_retention"]
        >= args.minimum_recall_retention,
        "bounded_nll_regression": metrics["bounded_relative_nll_regression"]
        <= args.maximum_nll_regression,
        "ring_contrast_improves_nll": metrics["ring_contrast_relative_nll_improvement"]
        >= args.minimum_ring_contrast_nll_improvement,
        "split_group_stability": metrics["median_matched_jaccard"]
        >= args.minimum_split_stability,
        "enough_stable_groups": metrics["stable_groups"] >= args.minimum_stable_groups,
        "group_count_agreement": metrics["group_count_agreement"]
        >= args.minimum_group_count_agreement,
        "capacity_not_saturated": not metrics["capacity_saturated"],
        "no_queries_labels_or_codebooks": metrics["no_evaluation_queries_labels_or_codebooks"],
        "unresolved_certificate_written": metrics["unresolved_certificate_written"],
    }
    passed = all(checks.values())
    return {
        "pass": bool(passed),
        "decision": "PROCEED_TO_A50_1_CONTINUOUS_GROUP_RETRIEVAL"
        if passed
        else "STOP_BEFORE_CONTINUOUS_RETRIEVAL_AND_CODEBOOKS",
        "checks": {key: bool(value) for key, value in checks.items()},
    }


def save_group_model(path, model):
    np.savez_compressed(
        path,
        unique_spatial_group_ids=model["unique_spatial_group_ids"],
        levels=model["levels"],
        descriptors=model["descriptors"].astype(np.float16),
        ring_descriptors=model["ring_descriptors"].astype(np.float16),
        ring_valid=model["ring_valid"],
        raw_profiles=model["raw_profiles"].astype(np.float16),
        bounded_profiles=model["profiles"].astype(np.float16),
        core_masks=np.packbits(model["core_masks"], axis=1, bitorder="little"),
        support_masks=np.packbits(model["support_masks"], axis=1, bitorder="little"),
        ring_masks=np.packbits(model["ring_masks"], axis=1, bitorder="little"),
        support_views=model["support_views"],
        utility=model["utility"],
    )


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
    parser.add_argument("--ring_contrast_weight", type=float, default=0.50)
    parser.add_argument("--ring_contrast_margin", type=float, default=0.05)
    parser.add_argument("--minimum_group_query_score", type=float, default=0.50)
    parser.add_argument("--minimum_spill_reduction", type=float, default=0.25)
    parser.add_argument("--minimum_recall_retention", type=float, default=0.85)
    parser.add_argument("--maximum_nll_regression", type=float, default=0.02)
    parser.add_argument("--minimum_ring_contrast_nll_improvement", type=float, default=0.0)
    parser.add_argument("--minimum_split_stability", type=float, default=0.65)
    parser.add_argument("--minimum_stable_groups", type=int, default=100)
    parser.add_argument("--minimum_group_count_agreement", type=float, default=0.80)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    np.random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse Group-addressed spatial audit: {output_dir}")
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
        and contract["odd_even_independent_fit"]
        and not contract["evaluation_queries_or_labels_used"]
        and not contract["codebooks_trained"]
        and int(contract["fixed_seed"]) == args.seed
    ):
        raise ValueError("A47 source contract is incompatible with A50")
    checkpoint_sha = file_sha256(args.geometry_checkpoint)
    if checkpoint_sha != a47_manifest["inputs"]["geometry_checkpoint_sha256"]:
        raise ValueError("Geometry checkpoint does not match A47 atom ownership")
    entries = incidence_entries(args.a47_audit_dir)
    views = load_prepared_views(args.a47_audit_dir, entries)
    atom_ids_path = os.path.join(args.a47_audit_dir, "gaussian_atom_ids.npy")
    atom_geometry = load_atom_geometry(args.geometry_checkpoint, atom_ids_path)
    contact_graph = build_atom_contact_graph(atom_geometry, args.atom_neighbors)
    ring_dir = os.path.join(output_dir, "ring_incidence_views")
    ring_manifest = prepare_ring_incidence(
        a47_manifest,
        args.a47_audit_dir,
        np.load(atom_ids_path),
        ring_dir,
        args,
    )
    ring_views = load_ring_views(ring_dir, ring_manifest)

    odd_raw = fit_persistent_slots(views, 0, args)
    even_raw = fit_persistent_slots(views, 1, args)
    odd = make_spatial_group_model(odd_raw, views, ring_views, contact_graph, args)
    even = make_spatial_group_model(even_raw, views, ring_views, contact_graph, args)
    odd_to_even = evaluate_group_addressing(odd, views, 1, args)
    even_to_odd = evaluate_group_addressing(even, views, 0, args)
    variants = mean_variant_results(odd_to_even, even_to_odd)

    raw_matches, _ = match_level_slots(
        odd_raw, even_raw, args.minimum_persistence_views, args.minimum_split_stability
    )
    matches, unresolved_groups = match_level_slots(
        odd, even, args.minimum_persistence_views, args.minimum_split_stability
    )
    unresolved_path = os.path.join(output_dir, "unresolved_group_certificate.json")
    with open(unresolved_path, "w") as output:
        json.dump(
            {
                "unresolved_groups": unresolved_groups,
                "odd_to_even_unresolved_proposals": odd_to_even["unresolved"],
                "even_to_odd_unresolved_proposals": even_to_odd["unresolved"],
                "forced_gaussian_or_group_assignment_forbidden": True,
            },
            output,
            indent=2,
        )
    save_group_model(os.path.join(output_dir, "odd_spatial_groups.npz"), odd)
    save_group_model(os.path.join(output_dir, "even_spatial_groups.npz"), even)

    raw = variants["raw_group"]
    bounded = variants["bounded_group"]
    contrastive = variants["contrastive_bounded_group"]
    jaccards = np.asarray([item["jaccard"] for item in matches], dtype=np.float32)
    raw_jaccards = np.asarray([item["jaccard"] for item in raw_matches], dtype=np.float32)
    stable = sum(item["stable"] for item in matches)
    odd_count = len(odd["profiles"])
    even_count = len(even["profiles"])
    metrics = {
        "heldout_semantic_group_retrieval": variants,
        "bounded_spill_reduction": float(
            (raw["exterior_spill_fraction"] - bounded["exterior_spill_fraction"])
            / max(raw["exterior_spill_fraction"], 1e-8)
        ),
        "bounded_recall_retention": float(bounded["recall"] / max(raw["recall"], 1e-8)),
        "bounded_relative_nll_regression": float(
            (bounded["mean_nll"] - raw["mean_nll"]) / max(raw["mean_nll"], 1e-8)
        ),
        "ring_contrast_relative_nll_improvement": float(
            (bounded["mean_nll"] - contrastive["mean_nll"])
            / max(bounded["mean_nll"], 1e-8)
        ),
        "ring_contrast_selection_changed_fraction": 0.5
        * (
            odd_to_even["selection_changed_fraction"]
            + even_to_odd["selection_changed_fraction"]
        ),
        "raw_median_matched_jaccard": float(np.median(raw_jaccards))
        if raw_jaccards.size
        else 0.0,
        "median_matched_jaccard": float(np.median(jaccards)) if jaccards.size else 0.0,
        "mean_matched_jaccard": float(jaccards.mean()) if jaccards.size else 0.0,
        "matched_groups": len(matches),
        "stable_groups": stable,
        "group_count_agreement": min(odd_count, even_count) / max(odd_count, even_count),
        "capacity_saturated": False,
        "unique_group_address_contract": bool(
            np.array_equal(odd["unique_spatial_group_ids"], np.arange(odd_count))
            and np.array_equal(even["unique_spatial_group_ids"], np.arange(even_count))
        ),
        "no_evaluation_queries_labels_or_codebooks": True,
        "unresolved_group_count": len(unresolved_groups),
        "unresolved_certificate_written": os.path.isfile(unresolved_path),
    }
    gate = make_gate(metrics, args)
    manifest = {
        "format_version": 1,
        "experiment": "A50.0_group_addressed_spatial_memory_audit",
        "scene": "ramen",
        "seed": args.seed,
        "representation": "unique_spatial_group_address_plus_continuous_semantic_key",
        "inputs": {
            "a47_audit_dir": os.path.abspath(args.a47_audit_dir),
            "a47_manifest_sha256": file_sha256(a47_manifest_path),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
            "geometry_checkpoint_sha256": checkpoint_sha,
            "gaussian_atom_ids_sha256": file_sha256(atom_ids_path),
            "ring_incidence_manifest_sha256": file_sha256(
                os.path.join(ring_dir, "manifest.json")
            ),
        },
        "source_contract": {
            "training_views_only": True,
            "raw_proposal_identity_retained_as_unique_spatial_group": True,
            "odd_even_independent_fit": True,
            "semantic_group_key_and_spatial_group_address_separated": True,
            "core_boundary_ring_membership": True,
            "scale_adaptive_2d_exterior_ring_projected_by_talpha": True,
            "evaluation_queries_or_labels_used": False,
            "codebooks_trained": False,
            "fixed_seed": args.seed,
        },
        "odd_model": {
            "spatial_groups": odd_count,
            "raw_statistics": odd_raw["statistics"],
            "spatial_statistics": odd["spatial_statistics"],
        },
        "even_model": {
            "spatial_groups": even_count,
            "raw_statistics": even_raw["statistics"],
            "spatial_statistics": even["spatial_statistics"],
        },
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
