#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from eval_lerf_ovs_miou import calibrate_frame_scores, load_lerf_labels, polygons_to_mask
from evaluation.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from scene import GaussianModel, Scene
from semantic_field_utils import load_geometry_checkpoint, load_semantic_codec
from utils.general_utils import safe_state


def decode_semantic_field(codec, semantic_features, valid, chunk_size):
    decoded = torch.zeros(
        (semantic_features.shape[0], 512),
        dtype=torch.float16,
        device="cuda",
    )
    valid_indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
    with torch.no_grad():
        for start in tqdm(range(0, valid_indices.numel(), chunk_size), desc="Decoding semantic field"):
            indices = valid_indices[start : start + chunk_size]
            latents = semantic_features[indices].float()
            decoded[indices] = codec.decode(latents).to(torch.float16)
    return decoded


def evaluate_thresholds(frame_scores, frame_gts, thresholds):
    results = []
    for threshold in thresholds:
        intersection = 0
        union = 0
        for image_name, score in frame_scores.items():
            prediction = score > threshold
            ground_truth = frame_gts[image_name]
            intersection += int(np.logical_and(prediction, ground_truth).sum())
            union += int(np.logical_or(prediction, ground_truth).sum())
        results.append(
            {
                "threshold": threshold,
                "iou": float(intersection / union) if union else 0.0,
            }
        )
    return results


def main():
    parser = ArgumentParser(description="Evaluate a self-trained low-dimensional Gaussian semantic field.")
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--semantic_artifact", required=True)
    parser.add_argument("--codec", default=None)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--decode_chunk_size", type=int, default=65536)
    parser.add_argument(
        "--score_calibration",
        choices=["none", "frame_minmax", "frame_percentile", "category_percentile"],
        default="none",
    )
    parser.add_argument("--calibration_low", type=float, default=1.0)
    parser.add_argument("--calibration_high", type=float, default=99.0)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9],
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.decode_chunk_size <= 0:
        raise ValueError("--decode_chunk_size must be positive")
    safe_state(args.quiet)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    opt = SimpleNamespace(include_feature=False)
    os.makedirs(args.output, exist_ok=True)

    labels, categories = load_lerf_labels(args.label_dir)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(scene.gaussians, args.geometry_checkpoint)
    cameras = {
        camera.image_name: camera
        for camera in scene.getTrainCameras() + scene.getTestCameras()
    }
    missing_cameras = sorted(set(labels) - set(cameras))
    if missing_cameras:
        raise ValueError(f"Missing labeled cameras in scene: {missing_cameras}")

    artifact = torch.load(args.semantic_artifact, map_location="cpu")
    semantic_features = artifact["semantic_features"].float().cuda()
    support_weights = artifact["support_weights"].float().cuda()
    if semantic_features.shape[0] != scene.gaussians.get_xyz.shape[0]:
        raise ValueError("Semantic artifact Gaussian count does not match geometry")
    codec_path = args.codec or artifact.get("codec")
    if not codec_path:
        raise ValueError("No semantic codec path provided")
    codec, codec_payload = load_semantic_codec(codec_path, device="cuda")
    if semantic_features.shape[1] != codec.semantic_dim:
        raise ValueError("Semantic artifact dimension does not match codec")
    valid = (support_weights > 0) & (semantic_features.norm(dim=-1) > 1e-8)
    decoded_features = decode_semantic_field(
        codec,
        semantic_features,
        valid,
        chunk_size=args.decode_chunk_size,
    )
    del semantic_features

    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    thresholds = sorted(set(args.thresholds))
    per_category = {}

    with torch.no_grad():
        for category_index, category in enumerate(tqdm(categories, desc="Evaluating semantic categories")):
            activation = torch.zeros(
                (scene.gaussians.get_xyz.shape[0], 1),
                dtype=torch.float32,
                device="cuda",
            )
            for start in range(0, decoded_features.shape[0], args.decode_chunk_size):
                end = min(start + args.decode_chunk_size, decoded_features.shape[0])
                chunk_valid = valid[start:end]
                if chunk_valid.any():
                    chunk_activation = clip_model.get_activation(
                        decoded_features[start:end][chunk_valid],
                        category_index,
                    )
                    activation[start:end][chunk_valid] = chunk_activation.float()

            frame_scores = {}
            frame_ground_truth = {}
            for image_name, label_data in labels.items():
                if category not in label_data["objects"]:
                    continue
                rendered = render(
                    cameras[image_name],
                    scene.gaussians,
                    pipe,
                    background,
                    opt,
                    override_color=activation.repeat(1, 3),
                )["render"]
                score = rendered.mean(dim=0).detach().cpu().numpy()
                ground_truth = polygons_to_mask(
                    label_data["objects"][category],
                    label_data["width"],
                    label_data["height"],
                )
                if score.shape != ground_truth.shape:
                    score_image = Image.fromarray(
                        (np.clip(score, 0.0, 1.0) * 255).astype(np.uint8)
                    )
                    score = np.asarray(
                        score_image.resize(
                            (ground_truth.shape[1], ground_truth.shape[0]),
                            Image.BILINEAR,
                        ),
                        dtype=np.float32,
                    ) / 255.0
                frame_scores[image_name] = score
                frame_ground_truth[image_name] = ground_truth

            if not frame_scores:
                continue
            frame_scores = calibrate_frame_scores(
                frame_scores,
                args.score_calibration,
                args.calibration_low,
                args.calibration_high,
            )
            threshold_results = evaluate_thresholds(frame_scores, frame_ground_truth, thresholds)
            best = max(threshold_results, key=lambda item: item["iou"])
            per_category[category] = {
                "best_iou": best["iou"],
                "best_threshold": best["threshold"],
                "num_frames": len(frame_scores),
                "thresholds": threshold_results,
            }

    oracle_ious = [item["best_iou"] for item in per_category.values()]
    global_threshold_summary = []
    for threshold in thresholds:
        threshold_ious = []
        threshold_accuracy = []
        for item in per_category.values():
            match = next(result for result in item["thresholds"] if result["threshold"] == threshold)
            threshold_ious.append(match["iou"])
            threshold_accuracy.append(match["iou"] >= 0.25)
        global_threshold_summary.append(
            {
                "threshold": threshold,
                "mIoU": float(np.mean(threshold_ious)) if threshold_ious else 0.0,
                "mAcc@0.25": float(np.mean(threshold_accuracy)) if threshold_accuracy else 0.0,
            }
        )

    semantic_tensor = artifact["semantic_features"]
    semantic_bytes = int(semantic_tensor.numel() * semantic_tensor.element_size())
    results = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "semantic_artifact": os.path.abspath(args.semantic_artifact),
        "codec": os.path.abspath(codec_path),
        "semantic_dim": int(semantic_tensor.shape[1]),
        "semantic_storage_bytes": semantic_bytes,
        "semantic_storage_megabytes": semantic_bytes / (1024.0 ** 2),
        "codec_storage_bytes": os.path.getsize(codec_path),
        "supported_fraction": float(valid.float().mean().item()),
        "score_calibration": args.score_calibration,
        "calibration_low": float(args.calibration_low),
        "calibration_high": float(args.calibration_high),
        "num_categories": len(per_category),
        "mIoU": float(np.mean(oracle_ious)) if oracle_ious else 0.0,
        "mAcc@0.25": float(np.mean([iou >= 0.25 for iou in oracle_ious])) if oracle_ious else 0.0,
        "global_threshold_summary": global_threshold_summary,
        "best_global_threshold": max(global_threshold_summary, key=lambda item: item["mIoU"])
        if global_threshold_summary
        else None,
        "per_category": per_category,
        "training_config": artifact.get("config", {}),
        "codec_metadata": codec_payload.get("metadata", {}),
        "note": "Self-trained low-dimensional Gaussian semantic field; train-only nuisance is discarded.",
    }
    metrics_path = os.path.join(args.output, "metrics.json")
    with open(metrics_path, "w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
