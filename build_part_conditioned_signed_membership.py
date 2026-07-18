#!/usr/bin/env python
"""Gate fine semantic IDs with track-free, part-conditioned mask ownership."""

import hashlib
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
from scipy import ndimage


def manifest_fingerprint(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def link(source, destination):
    if os.path.lexists(destination):
        os.unlink(destination)
    os.symlink(os.path.abspath(source), destination)


def mask_interior_confidence(segmentation, distance_pixels, boundary_floor):
    boundary = np.zeros(segmentation.shape, dtype=bool)
    vertical = segmentation[1:] != segmentation[:-1]
    horizontal = segmentation[:, 1:] != segmentation[:, :-1]
    boundary[1:] |= vertical
    boundary[:-1] |= vertical
    boundary[:, 1:] |= horizontal
    boundary[:, :-1] |= horizontal
    distance = ndimage.distance_transform_edt(~boundary)
    return (
        boundary_floor
        + (1.0 - boundary_floor)
        * np.clip(distance / distance_pixels, 0.0, 1.0)
    ).astype(np.float32)


def per_view_part_ownership(
    point_ids,
    point_weights,
    segment_ids,
    interior_weights,
    point_part_ids,
    num_parts,
):
    """Return per-Gaussian foreground/total mass under each part's winning mask."""
    point_ids = np.asarray(point_ids, dtype=np.int64)
    point_weights = np.asarray(point_weights, dtype=np.float32)
    segment_ids = np.asarray(segment_ids, dtype=np.int64)
    interior_weights = np.asarray(interior_weights, dtype=np.float32)
    if point_ids.ndim != 2 or point_weights.shape != point_ids.shape:
        raise ValueError("Point IDs and weights must have matching [P, K] shapes")
    if segment_ids.shape != (point_ids.shape[0],):
        raise ValueError("Segment IDs must have one value per sampled pixel")
    if interior_weights.shape != segment_ids.shape:
        raise ValueError("Interior weights must match sampled segments")

    num_gaussians = point_part_ids.size
    pixels = np.repeat(np.arange(point_ids.shape[0]), point_ids.shape[1])
    points = point_ids.reshape(-1)
    weights = point_weights.reshape(-1)
    segments = segment_ids[pixels]
    interiors = interior_weights[pixels]
    point_valid = (points >= 0) & (points < num_gaussians)
    safe_points = np.clip(points, 0, max(0, num_gaussians - 1))
    parts = point_part_ids[safe_points]
    valid = (
        point_valid
        & (parts >= 0)
        & (parts < num_parts)
        & (segments >= 0)
        & (weights > 0.0)
        & (interiors > 0.0)
    )
    if not valid.any():
        zeros = np.zeros(num_gaussians, dtype=np.float32)
        return zeros, zeros, np.zeros(num_gaussians, dtype=bool), {
            "observed_parts": 0,
            "positive_margin_parts": 0,
            "mean_part_signed_margin": 0.0,
        }
    points = points[valid]
    parts = parts[valid]
    segments = segments[valid]
    weights = weights[valid] * interiors[valid]

    num_segments = int(segments.max()) + 1
    pairs = parts * num_segments + segments
    unique_pairs, inverse = np.unique(pairs, return_inverse=True)
    pair_mass = np.bincount(inverse, weights=weights).astype(np.float32)
    pair_parts = unique_pairs // num_segments
    pair_segments = unique_pairs % num_segments

    part_total = np.bincount(
        pair_parts, weights=pair_mass, minlength=num_parts
    ).astype(np.float32)
    order = np.lexsort((pair_segments, -pair_mass, pair_parts))
    ordered_parts = pair_parts[order]
    first = np.r_[True, ordered_parts[1:] != ordered_parts[:-1]]
    winners = order[first]
    winner_parts = pair_parts[winners]
    dominant_segments = np.full(num_parts, -1, dtype=np.int64)
    dominant_mass = np.zeros(num_parts, dtype=np.float32)
    dominant_segments[winner_parts] = pair_segments[winners]
    dominant_mass[winner_parts] = pair_mass[winners]
    signed_margin = np.zeros(num_parts, dtype=np.float32)
    observed_parts = part_total > 0.0
    signed_margin[observed_parts] = np.clip(
        (2.0 * dominant_mass[observed_parts] - part_total[observed_parts])
        / np.maximum(part_total[observed_parts], 1e-12),
        0.0,
        1.0,
    )

    contribution_confidence = signed_margin[parts]
    accepted = contribution_confidence > 0.0
    points = points[accepted]
    parts = parts[accepted]
    segments = segments[accepted]
    weights = weights[accepted] * contribution_confidence[accepted]
    foreground = segments == dominant_segments[parts]
    total_mass = np.bincount(
        points, weights=weights, minlength=num_gaussians
    ).astype(np.float32)
    foreground_mass = np.bincount(
        points[foreground], weights=weights[foreground], minlength=num_gaussians
    ).astype(np.float32)
    return foreground_mass, total_mass, total_mass > 0.0, {
        "observed_parts": int(observed_parts.sum()),
        "positive_margin_parts": int((signed_margin > 0.0).sum()),
        "mean_part_signed_margin": float(
            signed_margin[observed_parts].mean() if observed_parts.any() else 0.0
        ),
    }


def split_membership(foreground, total, view_counts, min_split_views):
    membership = foreground / np.maximum(total, 1e-12)
    signed = np.clip(2.0 * membership - 1.0, 0.0, 1.0)
    support = (view_counts >= min_split_views) & (total > 0.0)
    both = support.all(axis=0)
    balance = 2.0 * np.minimum(total[0], total[1]) / np.maximum(
        total[0] + total[1], 1e-12
    )
    stability = 1.0 - np.abs(membership[0] - membership[1])
    reliability = np.sqrt(np.clip(balance, 0.0, 1.0)) * np.clip(
        stability, 0.0, 1.0
    )
    reliability *= np.clip(
        np.minimum(view_counts[0], view_counts[1]) / float(min_split_views),
        0.0,
        1.0,
    )
    reliability[~both] = 0.0
    cross_split_signed = np.sqrt(signed[0] * signed[1])
    return membership, cross_split_signed, reliability, both


def positive_prediction_metrics(first, second, valid):
    valid = np.asarray(valid, dtype=bool)
    first = np.asarray(first, dtype=bool)[valid]
    second = np.asarray(second, dtype=bool)[valid]
    true_positive = int((first & second).sum())
    predicted = int(first.sum())
    target = int(second.sum())
    precision = true_positive / max(1, predicted)
    recall = true_positive / max(1, target)
    return {
        "num_points": int(valid.sum()),
        "target_positive_fraction": float(second.mean()) if second.size else 0.0,
        "a20_all_positive_precision": float(second.mean()) if second.size else 0.0,
        "a20_all_positive_recall": 1.0 if target else 0.0,
        "split0_to_split1_precision": float(precision),
        "split0_to_split1_recall": float(recall),
        "split0_to_split1_f1": float(
            2.0 * precision * recall / max(precision + recall, 1e-12)
        ),
        "split_positive_agreement": float((first == second).mean())
        if second.size
        else 0.0,
    }


def main():
    import torch

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_dir", required=True)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--feature_dir", required=True)
    parser.add_argument("--feature_levels", nargs="+", type=int, default=[1, 2])
    parser.add_argument("--part_interior_support", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--boundary_threshold", type=float, default=0.75)
    parser.add_argument("--interior_distance", type=float, default=4.0)
    parser.add_argument("--interior_floor", type=float, default=0.25)
    parser.add_argument("--min_split_views", type=int, default=3)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not args.feature_levels or min(args.feature_levels) < 0:
        raise ValueError("Feature levels must be non-negative")
    if len(set(args.feature_levels)) != len(args.feature_levels):
        raise ValueError("Feature levels must be unique")
    if not 0.0 < args.boundary_threshold <= 1.0:
        raise ValueError("Boundary threshold must be in (0, 1]")
    if args.interior_distance <= 0.0 or not 0.0 <= args.interior_floor <= 1.0:
        raise ValueError("Mask interior parameters are invalid")
    if args.min_split_views <= 0 or args.max_views < 0:
        raise ValueError("View parameters are invalid")

    output_dir = os.path.abspath(args.output_dir)
    output_manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(output_manifest_path) and not args.force:
        print(f"Reuse part-conditioned membership: {output_dir}")
        return

    source_dir = os.path.abspath(args.source_artifact_dir)
    source_manifest_path = os.path.join(source_dir, "manifest.json")
    with open(source_manifest_path) as source:
        source_manifest = json.load(source)
    required_modalities = {"base", "part", "fine"}
    if not required_modalities.issubset(source_manifest.get("vocabulary_modalities", [])):
        raise ValueError("A23 requires an A20 base+part+fine artifact")
    point_ids = np.load(
        os.path.join(source_dir, source_manifest["point_group_ids"])
    )
    point_weights = np.load(
        os.path.join(source_dir, source_manifest["point_group_weights"])
    )
    invalid = int(source_manifest["invalid_id"])
    point_valid = point_ids != invalid
    if point_ids.ndim != 2 or point_ids.shape[1] < 2:
        raise ValueError("A23 requires part and fine slots")
    num_gaussians = point_ids.shape[0]
    part_ids = point_ids[:, 0].astype(np.int64)
    part_ids[~point_valid[:, 0]] = -1
    fine_valid = point_valid[:, 1] & (point_weights[:, 1] > 0)
    num_parts = int(part_ids.max()) + 1

    interior_support = np.load(os.path.abspath(args.part_interior_support)).astype(
        np.float32
    )
    if interior_support.shape != (num_gaussians,):
        raise ValueError("Part interior support does not match the A20 artifact")
    boundary_fine = fine_valid & (interior_support < args.boundary_threshold)

    cache_dir = os.path.abspath(args.cache_dir)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        cache_manifest = json.load(source)
    if int(cache_manifest["num_gaussians"]) != num_gaussians:
        raise ValueError("Contribution cache and A20 artifact do not match")
    if not cache_manifest.get("raw_contribution_weights", False):
        raise ValueError("A23 requires raw T*alpha contribution weights")
    if int(cache_manifest.get("topk", 0)) < 45:
        raise ValueError("A23 requires at least top-45 ray contributors")
    entries = cache_manifest["views"]
    if args.max_views:
        entries = entries[: args.max_views]
    if not entries:
        raise ValueError("Contribution cache has no views")

    feature_dir = os.path.abspath(args.feature_dir)
    statistics = {
        level: {
            "foreground": np.zeros((2, num_gaussians), dtype=np.float32),
            "total": np.zeros((2, num_gaussians), dtype=np.float32),
            "views": np.zeros((2, num_gaussians), dtype=np.uint16),
            "view_diagnostics": [],
        }
        for level in args.feature_levels
    }
    for entry_index, entry in enumerate(entries):
        cache = torch.load(os.path.join(cache_dir, entry["cache"]), map_location="cpu")
        cached_ids = cache["point_ids"].numpy().astype(np.int64, copy=False)
        cached_weights = cache["point_weights"].numpy().astype(np.float32, copy=False)
        sampled_flat = cache["sampled_flat_indices"].numpy().astype(np.int64, copy=False)
        split_index = int(entry.get("view_index", entry_index)) % 2
        segmentations = np.load(
            os.path.join(feature_dir, entry["image_name"] + "_s.npy"), mmap_mode="r"
        )
        for level in args.feature_levels:
            if level >= segmentations.shape[0]:
                raise ValueError(
                    f"Feature level {level} is unavailable for {entry['image_name']}"
                )
            segmentation = np.asarray(segmentations[level])
            if segmentation.size <= int(sampled_flat.max()):
                raise ValueError("Sampled pixels exceed the multiscale segmentation")
            sampled_segments = segmentation.reshape(-1)[sampled_flat].astype(np.int64)
            interior = mask_interior_confidence(
                segmentation, args.interior_distance, args.interior_floor
            ).reshape(-1)[sampled_flat]
            foreground, total, observed, diagnostics = per_view_part_ownership(
                cached_ids,
                cached_weights,
                sampled_segments,
                interior,
                part_ids,
                num_parts,
            )
            state = statistics[level]
            state["foreground"][split_index] += foreground
            state["total"][split_index] += total
            counts = state["views"][split_index].astype(np.uint32)
            counts += observed.astype(np.uint32)
            state["views"][split_index] = np.minimum(
                counts, np.iinfo(np.uint16).max
            ).astype(np.uint16)
            state["view_diagnostics"].append(diagnostics)
        print(
            json.dumps(
                {
                    "view": entry_index + 1,
                    "total_views": len(entries),
                    "image": entry["image_name"],
                }
            ),
            flush=True,
        )

    negative_confidences = []
    reliable_across_levels = np.ones(num_gaussians, dtype=bool)
    positive_first = np.zeros(num_gaussians, dtype=bool)
    positive_second = np.zeros(num_gaussians, dtype=bool)
    level_summaries = {}
    output_arrays = {}
    for level in args.feature_levels:
        state = statistics[level]
        membership, signed, reliability, both = split_membership(
            state["foreground"], state["total"], state["views"], args.min_split_views
        )
        negative_confidences.append(reliability * (1.0 - signed))
        reliable_across_levels &= both
        positive_first |= both & (membership[0] > 0.5)
        positive_second |= both & (membership[1] > 0.5)
        level_summaries[str(level)] = {
            "supported_boundary_fine_fraction": float(
                (both & boundary_fine).sum() / max(1, boundary_fine.sum())
            ),
            "mean_membership_split0_boundary_fine": float(
                membership[0, boundary_fine & both].mean()
                if (boundary_fine & both).any()
                else 0.0
            ),
            "mean_membership_split1_boundary_fine": float(
                membership[1, boundary_fine & both].mean()
                if (boundary_fine & both).any()
                else 0.0
            ),
            "mean_reliability_boundary_fine": float(
                reliability[boundary_fine & both].mean()
                if (boundary_fine & both).any()
                else 0.0
            ),
            "mean_observed_parts_per_view": float(
                np.mean([row["observed_parts"] for row in state["view_diagnostics"]])
            ),
            "mean_positive_margin_parts_per_view": float(
                np.mean(
                    [row["positive_margin_parts"] for row in state["view_diagnostics"]]
                )
            ),
        }
        output_arrays[f"level_{level}_split_membership.npy"] = membership.astype(
            np.float16
        )
        output_arrays[f"level_{level}_reliability.npy"] = reliability.astype(
            np.float16
        )

    # Suppress only when every SAM scale supplies stable negative evidence.
    negative_confidence = np.min(np.stack(negative_confidences), axis=0)
    gate = np.ones(num_gaussians, dtype=np.float32)
    gated = boundary_fine & reliable_across_levels
    gate[gated] = 1.0 - negative_confidence[gated]
    updated_weights = point_weights.copy()
    updated_weights[gated, 1] = np.rint(
        point_weights[gated, 1].astype(np.float32) * gate[gated]
    ).clip(0, 255).astype(np.uint8)

    os.makedirs(output_dir, exist_ok=True)
    for name in (
        "shared_vocabulary.npy",
        "group_semantic_code_ids.npy",
        "group_reliability.npy",
        "point_group_ids.npy",
    ):
        link(os.path.join(source_dir, name), os.path.join(output_dir, name))
    np.save(os.path.join(output_dir, "point_group_weights.npy"), updated_weights)
    np.save(
        os.path.join(output_dir, "entity_membership_gate.npy"),
        np.rint(gate * 255.0).astype(np.uint8),
    )
    for name, values in output_arrays.items():
        np.save(os.path.join(output_dir, name), values)

    held_out_valid = boundary_fine & reliable_across_levels
    held_out_metrics = positive_prediction_metrics(
        positive_first, positive_second, held_out_valid
    )
    changed = updated_weights[:, 1] != point_weights[:, 1]
    metadata_bytes = int(
        updated_weights.nbytes
        + gate.astype(np.uint8).nbytes
        + sum(values.nbytes for values in output_arrays.values())
    )
    manifest = dict(source_manifest)
    manifest.update(
        {
            "format_version": max(3, int(source_manifest.get("format_version", 1))),
            "method": "part_conditioned_multiscale_signed_membership",
            "point_group_weights": "point_group_weights.npy",
            "weight_dtype": "uint8_a20_membership_times_multiscale_signed_gate",
            "entity_membership_gate": "entity_membership_gate.npy",
            "membership": {
                "feature_levels": args.feature_levels,
                "num_views": len(entries),
                "boundary_fine_points": int(boundary_fine.sum()),
                "reliable_boundary_fine_points": int(held_out_valid.sum()),
                "changed_fine_points": int(changed.sum()),
                "changed_fraction_of_fine_points": float(
                    changed.sum() / max(1, fine_valid.sum())
                ),
                "mean_gate_on_changed": float(gate[changed].mean())
                if changed.any()
                else 1.0,
                "level_summaries": level_summaries,
                "held_out_split_membership": held_out_metrics,
            },
            "module_codebook_contract": {
                **source_manifest.get("module_codebook_contract", {}),
                "enabled_modules": [
                    "A14_base",
                    "A18_part",
                    "A20_fine_part",
                    "A23_part_conditioned_signed_membership",
                ],
                "codebook_reuse_reason": (
                    "A23 changes only training-derived point membership weights; "
                    "base/part/fine feature targets and semantic IDs are byte-identical to A20"
                ),
                "a20_manifest_sha256": manifest_fingerprint(source_manifest_path),
            },
            "storage": {
                **source_manifest["storage"],
                "a23_membership_metadata_bytes": metadata_bytes,
            },
            "source": {
                **source_manifest.get("source", {}),
                "a20_artifact_dir": source_dir,
                "contribution_cache": cache_dir,
                "multiscale_feature_dir": feature_dir,
                "part_interior_support": os.path.abspath(args.part_interior_support),
                "leakage_control": (
                    "training cameras, SAM masks, raw T*alpha contributions, and A20 part IDs only"
                ),
            },
            "args": vars(args),
        }
    )
    with open(output_manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest["membership"], indent=2))


if __name__ == "__main__":
    main()
