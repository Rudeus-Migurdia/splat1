#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from evaluation.openclip_encoder import OpenCLIPNetwork
from eval_lerf_ovs_miou import (
    decode_pq_language_features,
    load_checkpoint_into_gaussians,
    load_lerf_labels,
    polygons_to_mask,
)
from eval_lerf_ovs_multigroup_miou import (
    calibrate_frame_scores,
    load_multigroup_tokens,
    point_activation_from_groups,
)
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def evaluate_thresholds(frame_scores, frame_gts, thresholds):
    results = []
    for threshold in thresholds:
        intersection = 0
        union = 0
        for image_name, score in frame_scores.items():
            pred = score > threshold
            gt = frame_gts[image_name]
            intersection += int(np.logical_and(pred, gt).sum())
            union += int(np.logical_or(pred, gt).sum())
        results.append({"threshold": threshold, "iou": float(intersection / union) if union else 0.0})
    return results


def activation_specificity(activation, low, mid, high):
    values = activation.squeeze(-1).float()
    if values.numel() == 0:
        return 0.0
    quantiles = torch.tensor([low, mid, high], dtype=torch.float32, device=values.device) / 100.0
    q_low, q_mid, q_high = torch.quantile(values, quantiles)
    specificity = (q_high - q_mid) / (q_high - q_low).clamp_min(1e-6)
    return float(specificity.clamp(0.0, 1.0).item())


def activation_agreement(first, second):
    first = first.squeeze(-1).float()
    second = second.squeeze(-1).float()
    valid = torch.isfinite(first) & torch.isfinite(second)
    if int(valid.sum()) < 2:
        return 0.0
    first = first[valid] - first[valid].mean()
    second = second[valid] - second[valid].mean()
    denominator = first.norm() * second.norm()
    if float(denominator) <= 1e-9:
        return 0.0
    return float((torch.dot(first, second) / denominator).clamp(-1.0, 1.0).item())


def estimate_hybrid_alpha(baseline_activation, group_activation, args):
    if args.hybrid_gate_mode == "fixed":
        return float(args.hybrid_alpha), None
    baseline_specificity = activation_specificity(
        baseline_activation,
        args.gate_quantile_low,
        args.gate_quantile_mid,
        args.gate_quantile_high,
    )
    group_specificity = activation_specificity(
        group_activation,
        args.gate_quantile_low,
        args.gate_quantile_mid,
        args.gate_quantile_high,
    )
    agreement = activation_agreement(baseline_activation, group_activation)
    specificity_delta = group_specificity - baseline_specificity
    specificity_signal = -specificity_delta if args.hybrid_gate_mode == "inverse_specificity" else specificity_delta
    signal = (
        specificity_signal
        + float(args.gate_agreement_weight) * (agreement - float(args.gate_agreement_center))
        - float(args.gate_bias)
    )
    gate = float(torch.sigmoid(torch.tensor(signal / args.gate_temperature)).item())
    alpha = args.hybrid_alpha_min + (args.hybrid_alpha - args.hybrid_alpha_min) * gate
    diagnostics = {
        "baseline_specificity": baseline_specificity,
        "group_specificity": group_specificity,
        "specificity_delta": specificity_delta,
        "branch_agreement": agreement,
        "gate_signal": signal,
        "gate": gate,
    }
    return float(alpha), diagnostics


def point_group_confidence(top_group_ids, top_group_scores, mode, floor, power):
    valid = top_group_ids >= 0
    covered = valid.any(dim=1, keepdim=True)
    if mode == "none":
        return torch.ones_like(covered, dtype=torch.float32)
    if mode == "coverage":
        return covered.float()
    scores = torch.where(valid, top_group_scores.clamp_min(0.0), torch.zeros_like(top_group_scores))
    best = scores.max(dim=1, keepdim=True).values
    if scores.shape[1] > 1:
        second = torch.topk(scores, k=2, dim=1).values[:, 1:2]
    else:
        second = torch.zeros_like(best)
    margin = ((best - second) / best.clamp_min(1e-9)).clamp(0.0, 1.0)
    confidence = float(floor) + (1.0 - float(floor)) * margin.pow(float(power))
    return torch.where(covered, confidence, torch.zeros_like(confidence))


def blend_hybrid_activations(baseline, group, point_alpha, mode):
    if mode == "convex":
        return point_alpha * group + (1.0 - point_alpha) * baseline
    if mode == "additive":
        return baseline + point_alpha * group
    if mode == "positive_residual":
        return baseline + point_alpha * (group - baseline).clamp_min(0.0)
    raise ValueError(f"Unsupported hybrid blend mode: {mode}")


def main():
    parser = ArgumentParser(description="Evaluate a Dr.Splat + multi-group-token hybrid on LeRF-OVS mIoU.")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--drsplat_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--group_features", required=True)
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--group_aggregation", choices=["weighted", "weighted_maxblend", "max"], default="weighted")
    parser.add_argument("--score_power", type=float, default=1.0)
    parser.add_argument("--blend_alpha", type=float, default=0.75)
    parser.add_argument("--hybrid_alpha", type=float, default=0.5, help="Weight for multi-group activation; baseline gets 1-alpha.")
    parser.add_argument(
        "--hybrid_blend_mode",
        choices=["convex", "additive", "positive_residual"],
        default="convex",
        help="Fuse group evidence by interpolation, addition, or a non-destructive positive residual.",
    )
    parser.add_argument(
        "--hybrid_gate_mode",
        choices=["fixed", "relative_specificity", "inverse_specificity"],
        default="fixed",
        help="Use query-wise branch specificity to gate the group residual, or preserve fixed-alpha behavior.",
    )
    parser.add_argument("--hybrid_alpha_min", type=float, default=0.0)
    parser.add_argument("--gate_temperature", type=float, default=0.1)
    parser.add_argument("--gate_bias", type=float, default=0.0)
    parser.add_argument("--gate_agreement_weight", type=float, default=0.0)
    parser.add_argument("--gate_agreement_center", type=float, default=0.5)
    parser.add_argument("--gate_quantile_low", type=float, default=1.0)
    parser.add_argument("--gate_quantile_mid", type=float, default=90.0)
    parser.add_argument("--gate_quantile_high", type=float, default=99.0)
    parser.add_argument(
        "--point_group_gate_mode",
        choices=["none", "coverage", "assignment_margin"],
        default="none",
        help="Keep the main branch unchanged where group evidence is missing or ambiguous.",
    )
    parser.add_argument("--point_group_gate_floor", type=float, default=0.0)
    parser.add_argument("--point_group_gate_power", type=float, default=1.0)
    parser.add_argument("--eval_topk", type=int, default=0)
    parser.add_argument(
        "--score_calibration",
        choices=["none", "frame_minmax", "frame_percentile", "category_percentile"],
        default="none",
    )
    parser.add_argument("--calibration_low", type=float, default=1.0)
    parser.add_argument("--calibration_high", type=float, default=99.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if not (0.0 <= args.hybrid_alpha_min <= args.hybrid_alpha <= 1.0):
        raise ValueError("Hybrid alpha bounds must satisfy 0 <= min <= max <= 1.")
    if args.gate_temperature <= 0.0:
        raise ValueError("--gate_temperature must be positive.")
    if not (0.0 <= args.gate_quantile_low < args.gate_quantile_mid < args.gate_quantile_high <= 100.0):
        raise ValueError("Gate quantiles must satisfy 0 <= low < mid < high <= 100.")
    if not (0.0 <= args.point_group_gate_floor <= 1.0):
        raise ValueError("--point_group_gate_floor must be in [0, 1].")
    if args.point_group_gate_power < 0.0:
        raise ValueError("--point_group_gate_power must be non-negative.")

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = SimpleNamespace(include_feature=False)

    labels, categories = load_lerf_labels(args.label_dir)
    os.makedirs(args.output, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_checkpoint_into_gaussians(scene.gaussians, args.drsplat_checkpoint)
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    missing_cameras = sorted(set(labels) - set(cameras))
    if missing_cameras:
        raise ValueError(f"Missing labeled cameras in scene: {missing_cameras}")

    pq_index = faiss.read_index(args.pq_index)
    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    decoded_features, valid_baseline = decode_pq_language_features(scene.gaussians, pq_index)
    group_features, top_group_ids, top_group_scores = load_multigroup_tokens(args.group_features, args.assignments)

    if top_group_ids.shape[0] != scene.gaussians.get_xyz.shape[0]:
        raise ValueError("Assignment point count does not match Gaussian count.")
    point_confidence = point_group_confidence(
        top_group_ids,
        top_group_scores,
        args.point_group_gate_mode,
        args.point_group_gate_floor,
        args.point_group_gate_power,
    )

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    thresholds = sorted(set(args.thresholds))
    per_category = {}

    with torch.no_grad():
        for category_idx, category in enumerate(tqdm(categories, desc="Evaluating hybrid categories")):
            baseline_activation = torch.zeros((scene.gaussians.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
            if valid_baseline.any():
                baseline_activation[valid_baseline] = clip_model.get_activation(decoded_features[valid_baseline], category_idx)

            group_activation = clip_model.get_activation(group_features, category_idx)
            point_group_activation = point_activation_from_groups(
                group_activation,
                top_group_ids,
                top_group_scores,
                args.group_aggregation,
                args.score_power,
                args.blend_alpha,
                args.eval_topk,
            )
            effective_alpha, gate_diagnostics = estimate_hybrid_alpha(
                baseline_activation,
                point_group_activation,
                args,
            )
            point_alpha = effective_alpha * point_confidence
            activation = blend_hybrid_activations(
                baseline_activation,
                point_group_activation,
                point_alpha,
                args.hybrid_blend_mode,
            )

            frame_scores = {}
            frame_gts = {}
            for image_name, label_data in labels.items():
                if category not in label_data["objects"]:
                    continue
                camera = cameras[image_name]
                rendered = render(
                    camera,
                    scene.gaussians,
                    pipe,
                    bg,
                    opt,
                    override_color=activation.repeat(1, 3),
                )["render"]
                score = rendered.mean(dim=0).detach().cpu().numpy()
                gt = polygons_to_mask(label_data["objects"][category], label_data["width"], label_data["height"])
                frame_scores[image_name] = score
                frame_gts[image_name] = gt

            if not frame_scores:
                continue
            frame_scores = calibrate_frame_scores(
                frame_scores,
                args.score_calibration,
                args.calibration_low,
                args.calibration_high,
            )

            threshold_results = evaluate_thresholds(frame_scores, frame_gts, thresholds)
            best = max(threshold_results, key=lambda item: item["iou"])
            per_category[category] = {
                "best_iou": best["iou"],
                "best_threshold": best["threshold"],
                "num_frames": len(frame_scores),
                "effective_hybrid_alpha": effective_alpha,
                "mean_point_hybrid_alpha": float(point_alpha.mean().item()),
                "gate_diagnostics": gate_diagnostics,
                "thresholds": threshold_results,
            }

    ious = [item["best_iou"] for item in per_category.values()]
    global_threshold_summary = []
    for threshold in thresholds:
        threshold_ious = []
        threshold_acc = []
        for item in per_category.values():
            match = next(result for result in item["thresholds"] if result["threshold"] == threshold)
            threshold_ious.append(match["iou"])
            threshold_acc.append(match["iou"] >= 0.25)
        global_threshold_summary.append(
            {
                "threshold": threshold,
                "mIoU": float(np.mean(threshold_ious)) if threshold_ious else 0.0,
                "mAcc@0.25": float(np.mean(threshold_acc)) if threshold_acc else 0.0,
            }
        )

    results = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "drsplat_checkpoint": os.path.abspath(args.drsplat_checkpoint),
        "checkpoint_iteration": int(checkpoint_iteration),
        "pq_index": os.path.abspath(args.pq_index),
        "group_features": os.path.abspath(args.group_features),
        "assignments": os.path.abspath(args.assignments),
        "group_aggregation": args.group_aggregation,
        "score_power": float(args.score_power),
        "blend_alpha": float(args.blend_alpha),
        "hybrid_alpha": float(args.hybrid_alpha),
        "hybrid_blend_mode": args.hybrid_blend_mode,
        "hybrid_gate_mode": args.hybrid_gate_mode,
        "hybrid_alpha_min": float(args.hybrid_alpha_min),
        "gate_temperature": float(args.gate_temperature),
        "gate_bias": float(args.gate_bias),
        "gate_agreement_weight": float(args.gate_agreement_weight),
        "gate_agreement_center": float(args.gate_agreement_center),
        "gate_quantiles": [
            float(args.gate_quantile_low),
            float(args.gate_quantile_mid),
            float(args.gate_quantile_high),
        ],
        "point_group_gate_mode": args.point_group_gate_mode,
        "point_group_gate_floor": float(args.point_group_gate_floor),
        "point_group_gate_power": float(args.point_group_gate_power),
        "point_group_coverage": float((point_confidence > 0).float().mean().item()),
        "eval_topk": int(args.eval_topk),
        "score_calibration": args.score_calibration,
        "calibration_low": float(args.calibration_low),
        "calibration_high": float(args.calibration_high),
        "num_categories": len(per_category),
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc@0.25": float(np.mean([iou >= 0.25 for iou in ious])) if ious else 0.0,
        "global_threshold_summary": global_threshold_summary,
        "best_global_threshold": max(global_threshold_summary, key=lambda item: item["mIoU"]) if global_threshold_summary else None,
        "per_category": per_category,
        "note": "Hybrid upper-bound probe: blends Dr.Splat PQ activations with multi-group token activations.",
    }
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
