#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#

import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim


def _as_frozen_parameter(value):
    tensor = value.detach().to("cuda")
    return nn.Parameter(tensor, requires_grad=False)


def load_checkpoint_into_gaussians(gaussians, checkpoint_path):
    model_params, checkpoint_iteration = torch.load(checkpoint_path, map_location="cuda")

    if len(model_params) == 12:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            _opt_dict,
            spatial_lr_scale,
        ) = model_params
        language_feature = None
    elif len(model_params) == 13:
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
        raise ValueError(f"Unsupported checkpoint tuple length: {len(model_params)}")

    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = _as_frozen_parameter(xyz)
    gaussians._features_dc = _as_frozen_parameter(features_dc)
    gaussians._features_rest = _as_frozen_parameter(features_rest)
    gaussians._scaling = _as_frozen_parameter(scaling)
    gaussians._rotation = _as_frozen_parameter(rotation)
    gaussians._opacity = _as_frozen_parameter(opacity)
    gaussians._language_feature = (
        _as_frozen_parameter(language_feature) if language_feature is not None else None
    )
    gaussians.max_radii2D = max_radii2D.detach().to("cuda")
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().to("cuda")
    gaussians.denom = denom.detach().to("cuda")
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians.optimizer = None
    return checkpoint_iteration


def save_image(tensor, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    image = torch.clamp(tensor.detach(), 0.0, 1.0)
    image = image.permute(1, 2, 0).cpu().numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)


def save_comparison(rendered, gt, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    diff = torch.abs(rendered - gt).mean(dim=0, keepdim=True).repeat(3, 1, 1)
    comparison = torch.cat(
        (
            torch.clamp(rendered, 0.0, 1.0),
            torch.clamp(gt, 0.0, 1.0),
            torch.clamp(diff * 4.0, 0.0, 1.0),
        ),
        dim=2,
    )
    save_image(comparison, path)


def evaluate_split(name, cameras, scene, pipe, background, opt, output_dir, args):
    if not cameras:
        print(f"Skipping {name}: no cameras")
        return None, []

    split_dir = os.path.join(output_dir, name)
    per_view = []
    totals = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}

    for idx, camera in enumerate(tqdm(cameras, desc=f"Evaluating {name}")):
        rendered = torch.clamp(render(camera, scene.gaussians, pipe, background, opt)["render"], 0.0, 1.0)
        gt = torch.clamp(camera.original_image.to("cuda"), 0.0, 1.0)

        metrics = {
            "image_name": camera.image_name,
            "l1": float(l1_loss(rendered, gt).detach().cpu()),
            "psnr": float(psnr(rendered[None], gt[None]).mean().detach().cpu()),
            "ssim": float(ssim(rendered[None], gt[None]).mean().detach().cpu()),
        }
        per_view.append(metrics)
        for key in totals:
            totals[key] += metrics[key]

        if args.save_images and idx < args.max_save:
            save_image(rendered, os.path.join(split_dir, "renders", f"{idx:05d}_{camera.image_name}.png"))
            save_image(gt, os.path.join(split_dir, "gt", f"{idx:05d}_{camera.image_name}.png"))
            save_comparison(
                rendered,
                gt,
                os.path.join(split_dir, "comparison", f"{idx:05d}_{camera.image_name}.png"),
            )

    count = len(per_view)
    summary = {key: totals[key] / count for key in totals}
    summary["count"] = count
    return summary, per_view


def default_output_dir(model_path, checkpoint_path, iteration):
    if checkpoint_path:
        checkpoint_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
        return os.path.join(model_path, "eval", checkpoint_name)
    return os.path.join(model_path, "eval", f"iteration_{iteration}")


def main():
    parser = ArgumentParser(description="Evaluate a vanilla 3DGS model with PSNR, SSIM, and L1")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=30000, type=int)
    parser.add_argument("--checkpoint", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--max_save", default=16, type=int)
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = SimpleNamespace(include_feature=False)

    if not dataset.model_path:
        raise ValueError("--model_path/-m is required")
    os.makedirs(dataset.model_path, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    load_iteration = None if args.checkpoint else args.iteration
    scene = Scene(dataset, gaussians, load_iteration=load_iteration, shuffle=False)

    loaded_iteration = args.iteration
    if args.checkpoint:
        loaded_iteration = load_checkpoint_into_gaussians(scene.gaussians, args.checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    output_dir = args.output or default_output_dir(dataset.model_path, args.checkpoint, loaded_iteration)
    os.makedirs(output_dir, exist_ok=True)

    results = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "checkpoint": os.path.abspath(args.checkpoint) if args.checkpoint else None,
        "iteration": int(loaded_iteration),
        "eval_split_enabled": bool(dataset.eval),
        "splits": {},
    }
    per_view_results = {}

    with torch.no_grad():
        if not args.skip_train:
            summary, per_view = evaluate_split(
                "train",
                scene.getTrainCameras(),
                scene,
                pipe,
                background,
                opt,
                output_dir,
                args,
            )
            if summary:
                results["splits"]["train"] = summary
                per_view_results["train"] = per_view

        if not args.skip_test:
            summary, per_view = evaluate_split(
                "test",
                scene.getTestCameras(),
                scene,
                pipe,
                background,
                opt,
                output_dir,
                args,
            )
            if summary:
                results["splits"]["test"] = summary
                per_view_results["test"] = per_view

    metrics_path = os.path.join(output_dir, "metrics.json")
    per_view_path = os.path.join(output_dir, "per_view_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    with open(per_view_path, "w") as f:
        json.dump(per_view_results, f, indent=2)

    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
