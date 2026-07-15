"""Shared metrics for the LeRF-OVS 3D object-selection protocol."""

from collections import defaultdict

import numpy as np


PROTOCOL_NAME = "drsplat_3d_selection"


def binary_iou(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    if prediction.shape != ground_truth.shape:
        raise ValueError("Prediction and ground truth must have matching shapes")
    union = np.logical_or(prediction, ground_truth).sum()
    if union == 0:
        return 0.0
    intersection = np.logical_and(prediction, ground_truth).sum()
    return float(intersection / union)


def validate_selection_thresholds(thresholds):
    values = sorted(set(float(value) for value in thresholds))
    if not values:
        raise ValueError("At least one Gaussian selection threshold is required")
    if values[0] < 0.0 or values[-1] > 1.0:
        raise ValueError("Gaussian selection thresholds must be in [0, 1]")
    return values


def summarize_samples(samples):
    if not samples:
        return {
            "num_samples": 0,
            "mIoU": 0.0,
            "mAcc@0.25": 0.0,
            "mAcc@0.5": 0.0,
            "per_category": {},
        }

    ious = np.asarray([sample["iou"] for sample in samples], dtype=np.float64)
    category_ious = defaultdict(list)
    for sample in samples:
        category_ious[sample["category"]].append(float(sample["iou"]))
    return {
        "num_samples": len(samples),
        "mIoU": float(ious.mean()),
        # OpenGaussian's reference evaluator uses strict inequalities.
        "mAcc@0.25": float((ious > 0.25).mean()),
        "mAcc@0.5": float((ious > 0.5).mean()),
        "per_category": {
            category: float(np.mean(values))
            for category, values in sorted(category_ious.items())
        },
    }


def summarize_threshold_grid(samples_by_threshold):
    summaries = []
    for threshold in sorted(samples_by_threshold):
        summary = summarize_samples(samples_by_threshold[threshold])
        summary["selection_threshold"] = float(threshold)
        summary["samples"] = list(samples_by_threshold[threshold])
        summaries.append(summary)
    return summaries


def best_threshold_summary(threshold_summaries):
    if not threshold_summaries:
        return None
    return max(
        threshold_summaries,
        key=lambda item: (item["mIoU"], -item["selection_threshold"]),
    )


def summarize_method_scenes(scene_results):
    """Choose one method threshold by mean scene mIoU, as in Dr.Splat."""
    if not scene_results:
        raise ValueError("At least one scene result is required")

    by_scene = {}
    common_thresholds = None
    for scene, result in scene_results.items():
        if result.get("evaluation_protocol") != PROTOCOL_NAME:
            raise ValueError(f"{scene} is not a {PROTOCOL_NAME} result")
        grid = {
            float(item["selection_threshold"]): item
            for item in result.get("threshold_summary", [])
        }
        if not grid:
            raise ValueError(f"{scene} has no threshold summary")
        by_scene[scene] = grid
        thresholds = set(grid)
        common_thresholds = (
            thresholds if common_thresholds is None else common_thresholds & thresholds
        )

    if not common_thresholds:
        raise ValueError("Scene results do not share a selection threshold")

    method_grid = []
    for threshold in sorted(common_thresholds):
        rows = [grid[threshold] for grid in by_scene.values()]
        method_grid.append(
            {
                "selection_threshold": float(threshold),
                "mean_scene_mIoU": float(np.mean([row["mIoU"] for row in rows])),
                "mean_scene_mAcc@0.25": float(
                    np.mean([row["mAcc@0.25"] for row in rows])
                ),
                "mean_scene_mAcc@0.5": float(
                    np.mean([row["mAcc@0.5"] for row in rows])
                ),
            }
        )

    best = max(
        method_grid,
        key=lambda item: (item["mean_scene_mIoU"], -item["selection_threshold"]),
    )
    selected_threshold = best["selection_threshold"]
    selected_scenes = {
        scene: {
            key: value
            for key, value in grid[selected_threshold].items()
            if key != "samples"
        }
        for scene, grid in sorted(by_scene.items())
    }
    return {
        "evaluation_protocol": PROTOCOL_NAME,
        "selection_threshold": selected_threshold,
        "mIoU": best["mean_scene_mIoU"],
        "mAcc@0.25": best["mean_scene_mAcc@0.25"],
        "mAcc@0.5": best["mean_scene_mAcc@0.5"],
        "scenes": selected_scenes,
        "threshold_summary": method_grid,
    }
