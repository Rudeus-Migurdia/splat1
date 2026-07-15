#!/usr/bin/env python
import json
import os
from argparse import ArgumentParser

import faiss
import numpy as np
import torch


def l2_normalize(x, eps=1e-9):
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), eps)


def decode_pq_checkpoint(checkpoint_path, pq_index_path):
    model_params, _ = torch.load(checkpoint_path, map_location="cpu")
    if len(model_params) != 13:
        raise ValueError(f"Expected 13-tuple Dr.Splat checkpoint, got {len(model_params)}")
    language_feature = model_params[7]
    feature_i16 = language_feature.to(torch.int16)
    invalid_neg = torch.all(feature_i16 == -1, dim=-1)
    invalid_255 = torch.all(feature_i16 == 255, dim=-1)
    valid = (~(invalid_neg | invalid_255)).numpy()

    decoded = np.zeros((language_feature.shape[0], 512), dtype=np.float32)
    pq_index = faiss.read_index(pq_index_path)
    valid_ids = np.flatnonzero(valid)
    for start in range(0, valid_ids.shape[0], 65536):
        ids = valid_ids[start : start + 65536]
        codes = language_feature[ids].numpy().astype("uint8", copy=False)
        decoded[ids] = pq_index.sa_decode(codes).astype(np.float32)
    decoded = l2_normalize(decoded)
    decoded[~valid] = 0.0
    return decoded, valid


def aggregate_teacher_to_groups(decoded_features, valid_teacher, assignments_path, num_groups, score_power):
    assignments = np.load(assignments_path)
    top_group_ids = assignments["top_group_ids"].astype(np.int64)
    top_group_scores = assignments["top_group_scores"].astype(np.float32)

    accum = np.zeros((num_groups, decoded_features.shape[1]), dtype=np.float32)
    weight_sum = np.zeros((num_groups,), dtype=np.float32)

    valid_points = np.flatnonzero(valid_teacher)
    for start in range(0, valid_points.shape[0], 32768):
        ids = valid_points[start : start + 32768]
        gids = top_group_ids[ids]
        scores = np.maximum(top_group_scores[ids], 0.0) ** score_power
        point_feats = decoded_features[ids]
        for slot in range(gids.shape[1]):
            valid = gids[:, slot] >= 0
            if not np.any(valid):
                continue
            np.add.at(accum, gids[valid, slot], point_feats[valid] * scores[valid, slot : slot + 1])
            np.add.at(weight_sum, gids[valid, slot], scores[valid, slot])

    teacher = np.zeros_like(accum)
    valid_groups = weight_sum > 1e-9
    teacher[valid_groups] = accum[valid_groups] / weight_sum[valid_groups, None]
    teacher = l2_normalize(teacher)
    return teacher, valid_groups, weight_sum


def robust_aggregate_teacher_to_groups(
    decoded_features,
    valid_teacher,
    assignments_path,
    initial_teacher,
    score_power,
    iterations,
    agreement_power,
    margin_floor,
):
    """IRLS-style spherical teacher aggregation for ambiguous multi-group assignments."""
    assignments = np.load(assignments_path)
    top_group_ids = assignments["top_group_ids"].astype(np.int64)
    top_group_scores = assignments["top_group_scores"].astype(np.float32)
    best_score = assignments["best_score"].astype(np.float32)
    second_score = assignments["second_score"].astype(np.float32)
    feature_weight_sum = assignments["feature_weight_sum"].astype(np.float32)
    num_groups = initial_teacher.shape[0]

    margin = (best_score - second_score) / np.maximum(best_score, 1e-6)
    margin = np.clip(margin, 0.0, 1.0)
    observed = feature_weight_sum[valid_teacher]
    support_scale = float(np.percentile(observed, 90.0)) if observed.size else 1.0
    support = np.sqrt(np.clip(feature_weight_sum / max(support_scale, 1e-6), 0.0, 1.0))
    point_confidence = (float(margin_floor) + (1.0 - float(margin_floor)) * margin) * support

    teacher = initial_teacher.copy()
    final_weight_sum = np.zeros((num_groups,), dtype=np.float32)
    final_agreement = np.zeros((num_groups,), dtype=np.float32)
    valid_points = np.flatnonzero(valid_teacher)
    for _ in range(iterations):
        accum = np.zeros_like(teacher)
        weight_sum = np.zeros((num_groups,), dtype=np.float32)
        agreement_sum = np.zeros((num_groups,), dtype=np.float32)
        for start in range(0, valid_points.shape[0], 32768):
            ids = valid_points[start : start + 32768]
            gids = top_group_ids[ids]
            point_feats = decoded_features[ids]
            base_scores = np.maximum(top_group_scores[ids], 0.0) ** score_power
            reliability = point_confidence[ids]
            for slot in range(gids.shape[1]):
                valid = gids[:, slot] >= 0
                if not np.any(valid):
                    continue
                slot_groups = gids[valid, slot]
                agreement = np.sum(point_feats[valid] * teacher[slot_groups], axis=1)
                agreement = np.clip(agreement, 0.0, 1.0)
                weights = base_scores[valid, slot] * reliability[valid] * np.power(agreement, agreement_power)
                np.add.at(accum, slot_groups, point_feats[valid] * weights[:, None])
                np.add.at(weight_sum, slot_groups, weights)
                np.add.at(agreement_sum, slot_groups, weights * agreement)
        valid_groups = weight_sum > 1e-9
        updated = teacher.copy()
        updated[valid_groups] = accum[valid_groups] / weight_sum[valid_groups, None]
        teacher = l2_normalize(updated)
        final_weight_sum = weight_sum
        final_agreement[valid_groups] = agreement_sum[valid_groups] / weight_sum[valid_groups]
    return teacher, final_weight_sum > 1e-9, final_weight_sum, final_agreement


def main():
    parser = ArgumentParser(description="Distill Dr.Splat PQ semantics into lightweight group tokens.")
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--drsplat_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--score_power", type=float, default=1.0)
    parser.add_argument("--teacher_weights", nargs="+", type=float, default=[0.25, 0.5, 0.75])
    parser.add_argument("--robust_iterations", type=int, default=0)
    parser.add_argument("--robust_agreement_power", type=float, default=2.0)
    parser.add_argument("--robust_margin_floor", type=float, default=0.25)
    parser.add_argument("--adaptive_teacher_blend", action="store_true")
    parser.add_argument("--min_teacher_blend", type=float, default=0.25)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    group_features = np.load(os.path.join(args.artifact_dir, "group_features.npy")).astype(np.float32)
    group_features = l2_normalize(group_features)
    assignments_path = os.path.join(args.artifact_dir, "point_group_assignments.npz")

    decoded, valid_teacher = decode_pq_checkpoint(args.drsplat_checkpoint, args.pq_index)
    teacher, valid_groups, weight_sum = aggregate_teacher_to_groups(
        decoded,
        valid_teacher,
        assignments_path,
        group_features.shape[0],
        args.score_power,
    )
    teacher_filled = teacher.copy()
    teacher_filled[~valid_groups] = group_features[~valid_groups]

    np.save(os.path.join(args.output_dir, "group_teacher_features.npy"), teacher_filled.astype(np.float32))
    np.save(os.path.join(args.output_dir, "group_teacher_valid.npy"), valid_groups)
    np.save(os.path.join(args.output_dir, "group_teacher_weight_sum.npy"), weight_sum)

    summaries = []
    for weight in args.teacher_weights:
        blended = l2_normalize((1.0 - weight) * group_features + weight * teacher_filled).astype(np.float32)
        name = f"group_features_teacher_w{str(weight).replace('.', 'p')}.npy"
        np.save(os.path.join(args.output_dir, name), blended)
        cosine_to_group = np.sum(blended * group_features, axis=1)
        cosine_to_teacher = np.sum(blended * teacher_filled, axis=1)
        summaries.append(
            {
                "teacher_weight": float(weight),
                "feature_path": os.path.join(args.output_dir, name),
                "mean_cosine_to_group": float(np.mean(cosine_to_group)),
                "mean_cosine_to_teacher": float(np.mean(cosine_to_teacher)),
            }
        )

    robust_summary = None
    if args.robust_iterations > 0:
        if args.robust_agreement_power < 0.0:
            raise ValueError("--robust_agreement_power must be non-negative.")
        if not (0.0 <= args.robust_margin_floor <= 1.0):
            raise ValueError("--robust_margin_floor must be in [0, 1].")
        if not (0.0 <= args.min_teacher_blend <= 1.0):
            raise ValueError("--min_teacher_blend must be in [0, 1].")
        robust_teacher, robust_valid, robust_weight_sum, robust_agreement = robust_aggregate_teacher_to_groups(
            decoded,
            valid_teacher,
            assignments_path,
            teacher_filled,
            args.score_power,
            args.robust_iterations,
            args.robust_agreement_power,
            args.robust_margin_floor,
        )
        robust_teacher[~robust_valid] = group_features[~robust_valid]
        support_scale = float(np.percentile(robust_weight_sum[robust_valid], 90.0)) if robust_valid.any() else 1.0
        support = np.sqrt(np.clip(robust_weight_sum / max(support_scale, 1e-6), 0.0, 1.0))
        group_confidence = np.clip(robust_agreement * support, 0.0, 1.0)
        np.save(os.path.join(args.output_dir, "group_teacher_features_robust.npy"), robust_teacher.astype(np.float32))
        np.save(os.path.join(args.output_dir, "group_teacher_robust_valid.npy"), robust_valid)
        np.save(os.path.join(args.output_dir, "group_teacher_robust_weight_sum.npy"), robust_weight_sum)
        np.save(os.path.join(args.output_dir, "group_teacher_robust_agreement.npy"), robust_agreement)
        np.save(os.path.join(args.output_dir, "group_teacher_robust_confidence.npy"), group_confidence)
        robust_blends = []
        for weight in args.teacher_weights:
            if args.adaptive_teacher_blend:
                effective_weight = float(weight) * (
                    float(args.min_teacher_blend) + (1.0 - float(args.min_teacher_blend)) * group_confidence
                )
            else:
                effective_weight = np.full(group_features.shape[0], float(weight), dtype=np.float32)
            blended = l2_normalize(
                (1.0 - effective_weight[:, None]) * group_features + effective_weight[:, None] * robust_teacher
            ).astype(np.float32)
            name = f"group_features_robust_w{str(weight).replace('.', 'p')}.npy"
            np.save(os.path.join(args.output_dir, name), blended)
            robust_blends.append(
                {
                    "teacher_weight": float(weight),
                    "feature_path": os.path.join(args.output_dir, name),
                    "mean_effective_teacher_weight": float(effective_weight.mean()),
                    "mean_cosine_to_teacher": float(np.mean(np.sum(blended * robust_teacher, axis=1))),
                }
            )
        robust_summary = {
            "iterations": int(args.robust_iterations),
            "agreement_power": float(args.robust_agreement_power),
            "margin_floor": float(args.robust_margin_floor),
            "adaptive_teacher_blend": bool(args.adaptive_teacher_blend),
            "mean_group_agreement": float(robust_agreement[robust_valid].mean()) if robust_valid.any() else 0.0,
            "mean_group_confidence": float(group_confidence[robust_valid].mean()) if robust_valid.any() else 0.0,
            "blends": robust_blends,
        }

    summary = {
        "num_groups": int(group_features.shape[0]),
        "feature_dim": int(group_features.shape[1]),
        "valid_teacher_groups": int(valid_groups.sum()),
        "valid_teacher_group_ratio": float(valid_groups.mean()),
        "mean_teacher_weight_sum": float(weight_sum[valid_groups].mean()) if valid_groups.any() else 0.0,
        "score_power": float(args.score_power),
        "blends": summaries,
        "robust": robust_summary,
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "teacher_distill_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
