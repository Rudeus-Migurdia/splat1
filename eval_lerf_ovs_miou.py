#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from evaluation.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from lerf_ovs_paper_protocol import (
    PROTOCOL_NAME,
    best_threshold_summary,
    binary_iou,
    summarize_threshold_grid,
    validate_selection_thresholds,
)
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def as_frozen_parameter(value):
    tensor = value.detach().to("cuda")
    return nn.Parameter(tensor, requires_grad=False)


def load_checkpoint_into_gaussians(gaussians, checkpoint_path):
    model_params, checkpoint_iteration = torch.load(checkpoint_path, map_location="cuda")
    if len(model_params) == 13:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            language_feature,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            _opt_dict,
            spatial_lr_scale,
        ) = model_params
    else:
        raise ValueError(
            f"Expected a Dr.Splat checkpoint with language features, got tuple length {len(model_params)}"
        )

    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = as_frozen_parameter(xyz)
    gaussians._features_dc = as_frozen_parameter(features_dc)
    gaussians._features_rest = as_frozen_parameter(features_rest)
    gaussians._scaling = as_frozen_parameter(scaling)
    gaussians._rotation = as_frozen_parameter(rotation)
    gaussians._opacity = as_frozen_parameter(opacity)
    gaussians._language_feature = as_frozen_parameter(language_feature)
    gaussians.max_radii2D = max_radii2D.detach().to("cuda")
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().to("cuda")
    gaussians.denom = denom.detach().to("cuda")
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians.optimizer = None
    return checkpoint_iteration


def load_lerf_labels(label_dir):
    labels = {}
    categories = set()
    for name in sorted(os.listdir(label_dir)):
        if not name.endswith(".json"):
            continue
        path = os.path.join(label_dir, name)
        with open(path) as f:
            data = json.load(f)
        info = data["info"]
        image_name = os.path.splitext(info["name"])[0]
        height = int(info["height"])
        width = int(info["width"])
        per_category = {}
        for obj in data.get("objects", []):
            category = obj.get("category", "").strip()
            polygon = obj.get("segmentation", [])
            if not category or len(polygon) < 3:
                continue
            categories.add(category)
            per_category.setdefault(category, []).append(polygon)
        labels[image_name] = {"width": width, "height": height, "objects": per_category}
    if not labels:
        raise ValueError(f"No label json files found in {label_dir}")
    return labels, sorted(categories)


def polygons_to_mask(polygons, width, height):
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        points = [(float(x), float(y)) for x, y in polygon]
        draw.polygon(points, outline=1, fill=1)
    return np.asarray(image, dtype=bool)


def decode_pq_language_features(gaussians, pq_index):
    features = gaussians._language_feature.detach()
    feature_i16 = features.to(torch.int16)
    invalid_neg = torch.all(feature_i16 == -1, dim=-1)
    invalid_255 = torch.all(feature_i16 == 255, dim=-1)
    valid = ~(invalid_neg | invalid_255)

    decoded = torch.zeros((features.shape[0], 512), dtype=torch.float32, device="cuda")
    if valid.any():
        encoded_np = features[valid].detach().cpu().numpy().astype("uint8", copy=False)
        decoded_np = pq_index.sa_decode(encoded_np)
        decoded[valid] = torch.from_numpy(decoded_np).to("cuda", dtype=torch.float32)
        decoded[valid] /= decoded[valid].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return decoded, valid


def save_visualization(rgb, score, pred, gt, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb_np = (rgb.clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    heat = np.zeros_like(rgb_np)
    score_np = np.clip(score, 0.0, 1.0)
    heat[..., 0] = (score_np * 255).astype(np.uint8)
    pred_rgb = rgb_np.copy()
    pred_rgb[pred] = (0.45 * pred_rgb[pred] + 0.55 * np.array([255, 0, 0])).astype(np.uint8)
    gt_rgb = rgb_np.copy()
    gt_rgb[gt] = (0.45 * gt_rgb[gt] + 0.55 * np.array([0, 255, 0])).astype(np.uint8)
    panel = np.concatenate([rgb_np, heat, pred_rgb, gt_rgb], axis=1)
    Image.fromarray(panel).save(out_path)


def normalize_score_map(score, mode, low, high, category_bounds=None):
    if mode == "none":
        return score
    if mode == "frame_minmax":
        lo = float(np.min(score))
        hi = float(np.max(score))
    elif mode == "frame_percentile":
        lo, hi = np.percentile(score, [low, high])
        lo = float(lo)
        hi = float(hi)
    elif mode == "category_percentile":
        if category_bounds is None:
            raise ValueError("category_percentile requires category_bounds")
        lo, hi = category_bounds
    else:
        raise ValueError(f"Unknown score calibration: {mode}")
    return np.clip((score - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def calibrate_frame_scores(frame_scores, mode, low, high):
    if mode in ("none", "frame_minmax", "frame_percentile"):
        return {
            image_name: normalize_score_map(score, mode, low, high)
            for image_name, score in frame_scores.items()
        }
    if mode == "category_percentile":
        values = np.concatenate([score.reshape(-1) for score in frame_scores.values()])
        bounds = tuple(float(v) for v in np.percentile(values, [low, high]))
        return {
            image_name: normalize_score_map(score, mode, low, high, bounds)
            for image_name, score in frame_scores.items()
        }
    raise ValueError(f"Unknown score calibration: {mode}")


def resize_rendered_map(score, target_shape):
    if score.shape == target_shape:
        return score
    score_image = Image.fromarray((np.clip(score, 0.0, 1.0) * 255).astype(np.uint8))
    return np.asarray(
        score_image.resize((target_shape[1], target_shape[0]), Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0


def evaluate_paper_3d_selection(
    scene,
    pipe,
    opt,
    labels,
    categories,
    cameras,
    activation_provider,
    thresholds,
    occupancy_threshold,
    output_dir,
    save_visualizations=False,
    max_visualizations=32,
):
    """Threshold Gaussian activations first, then score rendered occupancy masks."""
    thresholds = validate_selection_thresholds(thresholds)
    samples_by_threshold = {threshold: [] for threshold in thresholds}
    background = torch.zeros(3, dtype=torch.float32, device="cuda")

    with torch.no_grad():
        for category_index, category in enumerate(
            tqdm(categories, desc="Paper 3D-selection evaluation")
        ):
            activation = activation_provider(category_index).reshape(-1, 1)
            frame_ground_truth = {}
            for image_name, label_data in labels.items():
                if category not in label_data["objects"]:
                    continue
                frame_ground_truth[image_name] = polygons_to_mask(
                    label_data["objects"][category],
                    label_data["width"],
                    label_data["height"],
                )

            for batch_start in range(0, len(thresholds), 3):
                batch = thresholds[batch_start : batch_start + 3]
                selection_colors = torch.zeros(
                    (activation.shape[0], 3),
                    dtype=torch.float32,
                    device=activation.device,
                )
                for channel, threshold in enumerate(batch):
                    selection_colors[:, channel] = (activation[:, 0] > threshold).float()

                for image_name, ground_truth in frame_ground_truth.items():
                    rendered = render(
                        cameras[image_name],
                        scene.gaussians,
                        pipe,
                        background,
                        opt,
                        override_color=selection_colors,
                    )["render"]
                    for channel, threshold in enumerate(batch):
                        occupancy = rendered[channel].detach().cpu().numpy()
                        occupancy = resize_rendered_map(occupancy, ground_truth.shape)
                        prediction = occupancy > occupancy_threshold
                        samples_by_threshold[threshold].append(
                            {
                                "image_name": image_name,
                                "category": category,
                                "iou": binary_iou(prediction, ground_truth),
                            }
                        )

    threshold_summary = summarize_threshold_grid(samples_by_threshold)
    best_scene = best_threshold_summary(threshold_summary)
    best_scene_compact = (
        {key: value for key, value in best_scene.items() if key != "samples"}
        if best_scene is not None
        else None
    )

    if save_visualizations and best_scene is not None:
        visualization_count = 0
        selected_threshold = best_scene["selection_threshold"]
        with torch.no_grad():
            for category_index, category in enumerate(categories):
                if visualization_count >= max_visualizations:
                    break
                activation = activation_provider(category_index).reshape(-1, 1)
                selected = (activation > selected_threshold).float().repeat(1, 3)
                for image_name, label_data in labels.items():
                    if category not in label_data["objects"]:
                        continue
                    ground_truth = polygons_to_mask(
                        label_data["objects"][category],
                        label_data["width"],
                        label_data["height"],
                    )
                    rendered = render(
                        cameras[image_name],
                        scene.gaussians,
                        pipe,
                        background,
                        opt,
                        override_color=selected,
                    )["render"].mean(dim=0).detach().cpu().numpy()
                    occupancy = resize_rendered_map(rendered, ground_truth.shape)
                    prediction = occupancy > occupancy_threshold
                    safe_category = category.replace("/", "_").replace(" ", "_")
                    save_visualization(
                        cameras[image_name].original_image.detach().cpu(),
                        occupancy,
                        prediction,
                        ground_truth,
                        os.path.join(
                            output_dir,
                            "visualizations",
                            f"{image_name}_{safe_category}.png",
                        ),
                    )
                    visualization_count += 1
                    if visualization_count >= max_visualizations:
                        break

    return {
        "evaluation_protocol": PROTOCOL_NAME,
        "selection_space": "per_gaussian_openclip_relevancy",
        "selection_operator": ">",
        "occupancy_threshold": float(occupancy_threshold),
        "occupancy_operator": ">",
        "threshold_summary": threshold_summary,
        "best_scene_threshold_diagnostic": best_scene_compact,
        "note": (
            "Paper-style LeRF-OVS 3D object selection: threshold Gaussian relevancy "
            "before rendering, binarize rendered occupancy, and average IoU over "
            "annotation-view samples. Use scripts/summarize_lerf_ovs_paper.py on all "
            "four scenes to select one threshold per method and obtain final scores."
        ),
    }


def main():
    parser = ArgumentParser(description="Evaluate LeRF-OVS mIoU for a Dr.Splat checkpoint")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--evaluation_protocol",
        choices=["diagnostic", PROTOCOL_NAME],
        default=PROTOCOL_NAME,
        help="Paper 3D selection is the default; diagnostic must be requested explicitly.",
    )
    parser.add_argument(
        "--score_calibration",
        choices=["none", "frame_minmax", "frame_percentile", "category_percentile"],
        default="none",
    )
    parser.add_argument("--calibration_low", type=float, default=1.0)
    parser.add_argument("--calibration_high", type=float, default=99.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument(
        "--selection_thresholds",
        nargs="+",
        type=float,
        default=[value / 100.0 for value in range(10, 91, 5)],
        help="Per-Gaussian relevancy grid used only by drsplat_3d_selection.",
    )
    parser.add_argument("--occupancy_threshold", type=float, default=0.7)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--max_visualizations", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.evaluation_protocol == PROTOCOL_NAME:
        if args.score_calibration != "none":
            raise ValueError("Paper 3D-selection evaluation forbids score calibration")
        if not 0.0 <= args.occupancy_threshold <= 1.0:
            raise ValueError("--occupancy_threshold must be in [0, 1]")

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = SimpleNamespace(include_feature=False)

    labels, categories = load_lerf_labels(args.label_dir)
    default_eval_name = (
        "lerf_ovs_paper_selection"
        if args.evaluation_protocol == PROTOCOL_NAME
        else "lerf_ovs_miou"
    )
    output_dir = args.output or os.path.join(dataset.model_path, "eval", default_eval_name)
    os.makedirs(output_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_checkpoint_into_gaussians(scene.gaussians, args.checkpoint)
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    missing_cameras = sorted(set(labels) - set(cameras))
    if missing_cameras:
        raise ValueError(f"Missing labeled cameras in scene: {missing_cameras}")

    pq_index = faiss.read_index(args.pq_index)
    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    decoded_features, valid_gaussians = decode_pq_language_features(scene.gaussians, pq_index)

    if args.evaluation_protocol == PROTOCOL_NAME:
        def activation_provider(category_index):
            activation = torch.zeros(
                (scene.gaussians.get_xyz.shape[0], 1),
                dtype=torch.float32,
                device="cuda",
            )
            if valid_gaussians.any():
                activation[valid_gaussians] = clip_model.get_activation(
                    decoded_features[valid_gaussians], category_index
                )
            return activation

        results = evaluate_paper_3d_selection(
            scene,
            pipe,
            opt,
            labels,
            categories,
            cameras,
            activation_provider,
            args.selection_thresholds,
            args.occupancy_threshold,
            output_dir,
            args.save_visualizations,
            args.max_visualizations,
        )
        results.update(
            {
                "source_path": dataset.source_path,
                "model_path": dataset.model_path,
                "checkpoint": os.path.abspath(args.checkpoint),
                "checkpoint_iteration": int(checkpoint_iteration),
                "label_dir": os.path.abspath(args.label_dir),
                "num_label_frames": len(labels),
                "num_categories": len(categories),
            }
        )
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        print(json.dumps(results, indent=2))
        print(f"Saved metrics to {metrics_path}")
        return

    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    thresholds = sorted(set(args.thresholds))
    per_category = {}
    vis_count = 0

    with torch.no_grad():
        rgb_cache = {}
        for category_idx, category in enumerate(tqdm(categories, desc="Evaluating categories")):
            activation = torch.zeros((scene.gaussians.get_xyz.shape[0], 1), dtype=torch.float32, device="cuda")
            if valid_gaussians.any():
                activation[valid_gaussians] = clip_model.get_activation(decoded_features[valid_gaussians], category_idx)

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
                gt = polygons_to_mask(
                    label_data["objects"][category],
                    label_data["width"],
                    label_data["height"],
                )
                if score.shape != gt.shape:
                    score_img = Image.fromarray((np.clip(score, 0.0, 1.0) * 255).astype(np.uint8))
                    score = np.asarray(score_img.resize((gt.shape[1], gt.shape[0]), Image.BILINEAR), dtype=np.float32) / 255.0
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

            threshold_results = []
            for threshold in thresholds:
                intersection = 0
                union = 0
                for image_name, score in frame_scores.items():
                    pred = score > threshold
                    gt = frame_gts[image_name]
                    intersection += int(np.logical_and(pred, gt).sum())
                    union += int(np.logical_or(pred, gt).sum())
                iou = float(intersection / union) if union else 0.0
                threshold_results.append({"threshold": threshold, "iou": iou})

            best = max(threshold_results, key=lambda item: item["iou"])
            per_category[category] = {
                "best_iou": best["iou"],
                "best_threshold": best["threshold"],
                "num_frames": len(frame_scores),
                "thresholds": threshold_results,
            }

            if args.save_visualizations and vis_count < args.max_visualizations:
                for image_name, score in frame_scores.items():
                    if vis_count >= args.max_visualizations:
                        break
                    pred = score > best["threshold"]
                    gt = frame_gts[image_name]
                    if image_name not in rgb_cache:
                        rgb_cache[image_name] = cameras[image_name].original_image.detach().cpu()
                    safe_category = category.replace("/", "_").replace(" ", "_")
                    save_visualization(
                        rgb_cache[image_name],
                        score,
                        pred,
                        gt,
                        os.path.join(output_dir, "visualizations", f"{image_name}_{safe_category}.png"),
                    )
                    vis_count += 1

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
        "checkpoint": os.path.abspath(args.checkpoint),
        "checkpoint_iteration": int(checkpoint_iteration),
        "label_dir": os.path.abspath(args.label_dir),
        "num_label_frames": len(labels),
        "num_categories": len(per_category),
        "score_calibration": args.score_calibration,
        "calibration_low": float(args.calibration_low),
        "calibration_high": float(args.calibration_high),
        "thresholds": thresholds,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc@0.25": float(np.mean([iou >= 0.25 for iou in ious])) if ious else 0.0,
        "global_threshold_summary": global_threshold_summary,
        "best_global_threshold": max(global_threshold_summary, key=lambda item: item["mIoU"])
        if global_threshold_summary
        else None,
        "per_category": per_category,
        "note": "This evaluator renders continuous CLIP relevancy maps from Dr.Splat PQ language features and reports each category at its best threshold over the provided grid.",
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
