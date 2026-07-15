#!/usr/bin/env python
"""Teacher-guided semantic splitting for heterogeneous 3D groups.

This is an offline artifact transform.  Runtime keeps the same multi-group
evaluator and never decodes PQ features.
"""

import json
import os
from argparse import ArgumentParser

import numpy as np

from distill_group_tokens_from_drsplat import decode_pq_checkpoint, l2_normalize
from quantize_multigroup_codebook import spherical_kmeans


def packed_memberships(top_group_ids, top_group_scores):
    num_points, slots = top_group_ids.shape
    flat_groups = top_group_ids.reshape(-1)
    flat_scores = top_group_scores.reshape(-1)
    flat_positions = np.arange(flat_groups.size, dtype=np.int64)
    valid = flat_groups >= 0
    flat_groups = flat_groups[valid]
    flat_scores = flat_scores[valid]
    flat_positions = flat_positions[valid]
    order = np.argsort(flat_groups, kind="stable")
    flat_groups = flat_groups[order]
    flat_scores = flat_scores[order]
    flat_positions = flat_positions[order]
    offsets = np.zeros(int(flat_groups.max()) + 2 if flat_groups.size else 1, dtype=np.int64)
    if flat_groups.size:
        offsets[1:] = np.cumsum(np.bincount(flat_groups, minlength=offsets.size - 1))
    return flat_scores, flat_positions, offsets, slots


def group_statistics(decoded, valid_teacher, scores, positions, offsets, slots):
    num_groups = offsets.size - 1
    agreement = np.ones(num_groups, dtype=np.float32)
    counts = np.zeros(num_groups, dtype=np.int32)
    centers = np.zeros((num_groups, decoded.shape[1]), dtype=np.float32)
    for group_id in range(num_groups):
        start, end = offsets[group_id], offsets[group_id + 1]
        if start == end:
            continue
        point_ids = positions[start:end] // slots
        keep = valid_teacher[point_ids]
        if not np.any(keep):
            continue
        point_ids = point_ids[keep]
        weights = scores[start:end][keep].clip(min=0.0)
        features = decoded[point_ids]
        center = l2_normalize((features * weights[:, None]).sum(axis=0, keepdims=True))[0]
        centers[group_id] = center
        total = float(weights.sum())
        agreement[group_id] = float((features @ center * weights).sum() / max(total, 1e-9))
        counts[group_id] = int(point_ids.size)
    return centers, agreement, counts


def main():
    parser = ArgumentParser(description="Split semantically heterogeneous lifted groups with an offline PQ teacher.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--parent_features", required=True)
    parser.add_argument("--drsplat_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--min_points", type=int, default=1024)
    parser.add_argument("--min_dispersion", type=float, default=0.05)
    parser.add_argument("--max_splits", type=int, default=32)
    parser.add_argument("--child_teacher_weight", type=float, default=0.75)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    if args.min_points < 2 or args.max_splits < 0:
        raise ValueError("--min_points must be at least 2 and --max_splits must be non-negative.")
    if not (0.0 <= args.min_dispersion <= 1.0 and 0.0 <= args.child_teacher_weight <= 1.0):
        raise ValueError("--min_dispersion and --child_teacher_weight must be in [0, 1].")

    os.makedirs(args.output_dir, exist_ok=True)
    assignments_path = os.path.join(args.artifact_dir, "point_group_assignments.npz")
    assignments = np.load(assignments_path)
    top_group_ids = assignments["top_group_ids"].astype(np.int32)
    top_group_scores = assignments["top_group_scores"].astype(np.float32)
    parent_features = l2_normalize(np.load(args.parent_features).astype(np.float32))
    if top_group_ids.size and int(top_group_ids.max()) >= parent_features.shape[0]:
        raise ValueError("Assignments reference ids outside parent_features.")

    decoded, valid_teacher = decode_pq_checkpoint(args.drsplat_checkpoint, args.pq_index)
    if decoded.shape[0] != top_group_ids.shape[0]:
        raise ValueError("Teacher checkpoint Gaussian count does not match point_group_assignments.")
    scores, positions, offsets, slots = packed_memberships(top_group_ids, top_group_scores)
    centers, agreement, counts = group_statistics(decoded, valid_teacher, scores, positions, offsets, slots)
    dispersion = 1.0 - agreement
    candidates = np.flatnonzero((counts >= args.min_points) & (dispersion >= args.min_dispersion))
    candidates = candidates[np.argsort(-dispersion[candidates], kind="stable")]
    selected = candidates[: args.max_splits]

    refined_assignments = {name: assignments[name].copy() for name in assignments.files}
    refined_ids = refined_assignments["top_group_ids"].reshape(-1)
    refined_features = [parent_features]
    split_rows = []
    next_group_id = parent_features.shape[0]
    for split_rank, group_id in enumerate(selected):
        start, end = offsets[group_id], offsets[group_id + 1]
        member_positions = positions[start:end]
        member_points = member_positions // slots
        keep = valid_teacher[member_points]
        if int(keep.sum()) < args.min_points:
            continue
        member_positions = member_positions[keep]
        member_points = member_points[keep]
        member_scores = scores[start:end][keep].clip(min=0.0)
        member_features = decoded[member_points]
        child_centers, child_ids = spherical_kmeans(
            member_features,
            num_codes=2,
            iterations=args.iterations,
            seed=args.seed + 7919 * (split_rank + 1),
            sample_weight=member_scores,
        )
        child_features = l2_normalize(
            (1.0 - args.child_teacher_weight) * parent_features[group_id][None, :]
            + args.child_teacher_weight * child_centers
        )
        child_global_ids = np.array([next_group_id, next_group_id + 1], dtype=np.int32)
        refined_ids[member_positions] = child_global_ids[child_ids]
        refined_features.append(child_features.astype(np.float32))
        child_counts = np.bincount(child_ids, minlength=2).astype(np.int32)
        split_rows.append(
            {
                "parent_group": int(group_id),
                "child_groups": [int(value) for value in child_global_ids],
                "parent_agreement": float(agreement[group_id]),
                "parent_dispersion": float(dispersion[group_id]),
                "member_points": int(member_points.size),
                "child_counts": [int(value) for value in child_counts],
            }
        )
        next_group_id += 2

    refined_assignments["top_group_ids"] = refined_ids.reshape(top_group_ids.shape).astype(np.int32)
    refined_features = np.concatenate(refined_features, axis=0)
    np.save(os.path.join(args.output_dir, "group_features_refined.npy"), refined_features.astype(np.float32))
    np.savez_compressed(os.path.join(args.output_dir, "point_group_assignments_refined.npz"), **refined_assignments)
    np.save(os.path.join(args.output_dir, "group_teacher_agreement.npy"), agreement)
    np.save(os.path.join(args.output_dir, "group_teacher_dispersion.npy"), dispersion)
    np.save(os.path.join(args.output_dir, "group_teacher_member_count.npy"), counts)

    summary = {
        "num_parent_groups": int(parent_features.shape[0]),
        "num_refined_groups": int(refined_features.shape[0]),
        "selected_splits": int(len(split_rows)),
        "min_points": int(args.min_points),
        "min_dispersion": float(args.min_dispersion),
        "child_teacher_weight": float(args.child_teacher_weight),
        "mean_parent_agreement": float(agreement.mean()),
        "mean_parent_dispersion": float(dispersion.mean()),
        "splits": split_rows,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "teacher_group_refinement_summary.json"), "w") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
