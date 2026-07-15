#!/usr/bin/env python
"""Measure each Gaussian's maximum rendering contribution over training views."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import count_render
from scene import GaussianModel, Scene
from semantic_field_utils import load_geometry_checkpoint
from utils.general_utils import safe_state


def update_max_contribution(output, point_ids, point_weights, chunk_size=4_000_000):
    flat_ids = point_ids.reshape(-1)
    flat_weights = point_weights.reshape(-1)
    for start in range(0, flat_ids.numel(), chunk_size):
        stop = min(start + chunk_size, flat_ids.numel())
        ids = flat_ids[start:stop].long()
        weights = flat_weights[start:stop].float().clamp_min(0.0)
        valid = (ids >= 0) & (ids < output.numel()) & (weights > 0.0)
        if valid.any():
            output.scatter_reduce_(
                0,
                ids[valid],
                weights[valid],
                reduce="amax",
                include_self=True,
            )


def main():
    parser = ArgumentParser(description=__doc__)
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[5e-4])
    parser.add_argument("--chunk_size", type=int, default=4_000_000)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.chunk_size <= 0 or any(value < 0.0 for value in args.thresholds):
        raise ValueError("Chunk size must be positive and thresholds non-negative")

    safe_state(args.quiet)
    dataset = model_params.extract(args)
    pipe = pipeline_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(
        scene.gaussians,
        args.geometry_checkpoint,
    )
    cameras = scene.getTrainCameras()
    if args.max_views > 0:
        cameras = cameras[: args.max_views]
    if not cameras:
        raise ValueError("No training cameras found")

    num_gaussians = int(scene.gaussians.get_xyz.shape[0])
    maximum = torch.zeros(num_gaussians, dtype=torch.float32, device="cuda")
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    for camera in tqdm(cameras, desc="Maximum Gaussian contribution"):
        render_package = count_render(camera, scene.gaussians, pipe, background)
        update_max_contribution(
            maximum,
            render_package["per_pixel_gaussian_ids"],
            render_package["per_pixel_gaussian_contributions"],
            args.chunk_size,
        )
        del render_package
        torch.cuda.empty_cache()

    values = maximum.cpu().numpy()
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "max_contribution.npy"), values)
    threshold_stats = {}
    for threshold in sorted(set(args.thresholds)):
        keep = values > threshold
        tag = f"{threshold:.0e}".replace("+", "")
        filename = f"keep_gt_{tag}.npy"
        np.save(os.path.join(output_dir, filename), keep)
        threshold_stats[str(threshold)] = {
            "mask": filename,
            "num_kept": int(keep.sum()),
            "kept_fraction": float(keep.mean()),
        }
    positive = values[values > 0.0]
    manifest = {
        "format_version": 1,
        "representation": "training_view_max_gaussian_contribution",
        "source_path": dataset.source_path,
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "num_gaussians": num_gaussians,
        "num_training_views": len(cameras),
        "max_contribution": "max_contribution.npy",
        "positive_fraction": float((values > 0.0).mean()),
        "positive_quantiles": {
            str(quantile): float(np.quantile(positive, quantile))
            for quantile in (0.01, 0.05, 0.5, 0.95, 0.99)
        }
        if positive.size
        else {},
        "thresholds": threshold_stats,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
