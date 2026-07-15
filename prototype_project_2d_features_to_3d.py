#!/usr/bin/env python
import json
import gc
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def as_frozen_parameter(value):
    return nn.Parameter(value.detach().to("cuda"), requires_grad=False)


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
            opt_dict,
            spatial_lr_scale,
        ) = model_params
    elif len(model_params) == 13:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            _language_feature,
            max_radii2D,
            xyz_gradient_accum,
            denom,
            opt_dict,
            spatial_lr_scale,
        ) = model_params
    else:
        raise ValueError(f"Unsupported checkpoint tuple length: {len(model_params)}")

    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = as_frozen_parameter(xyz)
    gaussians._features_dc = as_frozen_parameter(features_dc)
    gaussians._features_rest = as_frozen_parameter(features_rest)
    gaussians._scaling = as_frozen_parameter(scaling)
    gaussians._rotation = as_frozen_parameter(rotation)
    gaussians._opacity = as_frozen_parameter(opacity)
    gaussians._language_feature = None
    gaussians.max_radii2D = max_radii2D.detach().to("cuda")
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().to("cuda")
    gaussians.denom = denom.detach().to("cuda")
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians.optimizer = None
    return opt_dict, checkpoint_iteration


def capture_with_language_feature(gaussians, language_feature, opt_dict):
    return (
        gaussians.active_sh_degree,
        gaussians._xyz.detach(),
        gaussians._features_dc.detach(),
        gaussians._features_rest.detach(),
        gaussians._scaling.detach(),
        gaussians._rotation.detach(),
        gaussians._opacity.detach(),
        language_feature.detach(),
        gaussians.max_radii2D.detach(),
        gaussians.xyz_gradient_accum.detach(),
        gaussians.denom.detach(),
        opt_dict,
        gaussians.spatial_lr_scale,
    )


def project_points(camera, xyz):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    clip = xyz_h @ camera.full_proj_transform
    w = clip[:, 3]
    ndc = clip[:, :3] / w[:, None].clamp_min(1e-7)
    width = int(camera.image_width)
    height = int(camera.image_height)
    px = ((ndc[:, 0] + 1.0) * 0.5 * width).long()
    py = ((1.0 - ndc[:, 1]) * 0.5 * height).long()
    valid = (
        (w > 0)
        & (ndc[:, 2] >= 0.0)
        & (ndc[:, 2] <= 1.0)
        & (px >= 0)
        & (px < width)
        & (py >= 0)
        & (py < height)
    )
    return px, py, ndc[:, 2], valid


def visible_center_points(camera, xyz, z_tolerance):
    px, py, z, valid = project_points(camera, xyz)
    if not valid.any():
        return px, py, valid

    width = int(camera.image_width)
    height = int(camera.image_height)
    linear = py[valid] * width + px[valid]
    z_valid = z[valid]
    zbuf = torch.full((height * width,), float("inf"), dtype=z.dtype, device=z.device)
    zbuf.scatter_reduce_(0, linear, z_valid, reduce="amin", include_self=True)
    visible_valid = z_valid <= (zbuf[linear] + z_tolerance)
    visible = torch.zeros_like(valid)
    valid_idx = valid.nonzero(as_tuple=False).flatten()
    visible[valid_idx[visible_valid]] = True
    return px, py, visible


def accumulate_projected_features(scene, dataset_path, feature_level, min_votes, z_tolerance):
    xyz = scene.gaussians.get_xyz.detach()
    num_points = xyz.shape[0]
    feature_sum = torch.zeros((num_points, 512), dtype=torch.float32, device="cuda")
    vote_count = torch.zeros((num_points,), dtype=torch.int32, device="cuda")
    used_views = 0

    feature_dir = os.path.join(dataset_path, "language_features")
    cameras = scene.getTrainCameras()
    for camera in tqdm(cameras, desc="Projecting 2D mask features to 3D"):
        feature_path = os.path.join(feature_dir, camera.image_name + "_f.npy")
        segment_path = os.path.join(feature_dir, camera.image_name + "_s.npy")
        if not (os.path.exists(feature_path) and os.path.exists(segment_path)):
            continue

        px, py, visible = visible_center_points(camera, xyz, z_tolerance)
        if not visible.any():
            continue

        seg_np = np.load(segment_path, mmap_mode="r")[feature_level]
        feat_np = np.load(feature_path)
        seg = torch.from_numpy(np.asarray(seg_np)).to("cuda", dtype=torch.long)
        feats = torch.from_numpy(feat_np).to("cuda", dtype=torch.float32)
        feats = feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        visible_idx = visible.nonzero(as_tuple=False).flatten()
        seg_ids = seg[py[visible_idx], px[visible_idx]]
        has_seg = (seg_ids >= 0) & (seg_ids < feats.shape[0])
        if not has_seg.any():
            continue

        target_idx = visible_idx[has_seg]
        point_feats = feats[seg_ids[has_seg]]
        feature_sum.index_add_(0, target_idx, point_feats)
        vote_count.index_add_(
            0,
            target_idx,
            torch.ones((target_idx.shape[0],), dtype=torch.int32, device="cuda"),
        )
        used_views += 1

    valid_points = vote_count >= min_votes
    point_features = torch.zeros_like(feature_sum)
    if valid_points.any():
        point_features[valid_points] = feature_sum[valid_points] / vote_count[valid_points, None].to(torch.float32)
        point_features[valid_points] /= point_features[valid_points].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    stats = {
        "num_points": int(num_points),
        "used_views": int(used_views),
        "valid_points": int(valid_points.sum().item()),
        "valid_point_ratio": float(valid_points.float().mean().item()),
        "min_votes": int(min_votes),
        "mean_votes_valid": float(vote_count[valid_points].float().mean().item()) if valid_points.any() else 0.0,
        "median_votes_valid": float(vote_count[valid_points].float().median().item()) if valid_points.any() else 0.0,
    }
    return point_features, valid_points, vote_count, stats


def encode_with_pq(point_features, valid_points, pq_index):
    code_size = pq_index.sa_code_size() if hasattr(pq_index, "sa_code_size") else pq_index.code_size
    codes = torch.full((point_features.shape[0], code_size), 255, dtype=torch.uint8, device="cuda")
    if valid_points.any():
        features_np = point_features[valid_points].detach().cpu().numpy().astype("float32", copy=False)
        encoded_np = pq_index.sa_encode(features_np)
        if encoded_np.shape[1] != code_size:
            code_size = encoded_np.shape[1]
            codes = torch.full((point_features.shape[0], code_size), 255, dtype=torch.uint8, device="cuda")
        codes[valid_points] = torch.from_numpy(encoded_np).to("cuda", dtype=torch.uint8)
    return codes


def write_cfg_args(args, output_model):
    cfg = Namespace(
        sh_degree=3,
        source_path=os.path.abspath(args.source_path),
        model_path=os.path.abspath(output_model),
        language_features_name="language_features_dim3",
        images="images",
        resolution=-1,
        white_background=False,
        feature_level=args.feature_level,
        data_device="cuda",
        eval=False,
    )
    with open(os.path.join(output_model, "cfg_args"), "w") as f:
        f.write(str(cfg))


def run_eval(args, output_model, checkpoint_path):
    if not args.eval_label_dir:
        return None
    cmd = [
        sys.executable,
        "-u",
        "eval_lerf_ovs_miou.py",
        "-s",
        os.path.abspath(args.source_path),
        "-m",
        os.path.abspath(output_model),
        "--checkpoint",
        os.path.abspath(checkpoint_path),
        "--label_dir",
        os.path.abspath(args.eval_label_dir),
        "--pq_index",
        os.path.abspath(args.pq_index),
        "--thresholds",
        "0.1",
        "0.15",
        "0.2",
        "0.25",
        "0.3",
        "0.35",
        "0.4",
        "0.45",
        "0.5",
        "0.55",
        "0.6",
        "0.65",
        "0.7",
        "0.75",
        "0.8",
        "0.85",
        "0.9",
        "--save_visualizations",
        "--max_visualizations",
        "24",
    ]
    print("Running eval:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    metrics_path = os.path.join(output_model, "eval", "lerf_ovs_miou", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def main():
    parser = ArgumentParser(description="Prototype 2D mask-feature projection to 3D Gaussian PQ codes")
    model = ModelParams(parser)
    _pipeline = PipelineParams(parser)
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--min_votes", type=int, default=2)
    parser.add_argument("--z_tolerance", type=float, default=1e-4)
    parser.add_argument("--batch_points", type=int, default=32768)
    parser.add_argument("--eval_label_dir", default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    os.makedirs(args.output_model, exist_ok=True)
    dataset = model.extract(args)
    dataset.model_path = os.path.abspath(args.output_model)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    opt_dict, checkpoint_iteration = load_checkpoint_into_gaussians(scene.gaussians, args.base_checkpoint)

    point_features, valid_points, vote_count, stats = accumulate_projected_features(
        scene,
        dataset.source_path,
        args.feature_level,
        args.min_votes,
        args.z_tolerance,
    )

    pq_index = faiss.read_index(args.pq_index)
    pq_codes = encode_with_pq(point_features, valid_points, pq_index)
    scene.gaussians._language_feature = as_frozen_parameter(pq_codes)

    checkpoint_path = os.path.join(args.output_model, "chkpnt0.pth")
    torch.save((capture_with_language_feature(scene.gaussians, pq_codes, opt_dict), 0), checkpoint_path)
    scene.save(0)
    write_cfg_args(args, args.output_model)

    report = {
        "method": "project_2d_sam_clip_features_to_3d_gaussian_centers_mean_pq",
        "source_path": dataset.source_path,
        "base_checkpoint": os.path.abspath(args.base_checkpoint),
        "base_checkpoint_iteration": int(checkpoint_iteration),
        "output_model": os.path.abspath(args.output_model),
        "checkpoint": os.path.abspath(checkpoint_path),
        "pq_index": os.path.abspath(args.pq_index),
        "feature_level": int(args.feature_level),
        "projection": {
            "visibility": "center projection with per-pixel z-buffer, no differentiable 3D rendering",
            "z_tolerance": float(args.z_tolerance),
        },
        "aggregation": {
            "strategy": "mean of L2-normalized 2D SAM-mask CLIP features per Gaussian, then L2 normalize and PQ encode",
            "min_votes": int(args.min_votes),
        },
        "stats": stats,
    }

    del point_features, valid_points, vote_count, pq_codes, scene, gaussians, pq_index
    torch.cuda.empty_cache()
    gc.collect()

    eval_metrics = run_eval(args, args.output_model, checkpoint_path)
    if eval_metrics:
        report["eval"] = {
            "metrics_path": os.path.join(args.output_model, "eval", "lerf_ovs_miou", "metrics.json"),
            "oracle_per_category_best_mIoU": eval_metrics.get("mIoU"),
            "oracle_per_category_best_mAcc@0.25": eval_metrics.get("mAcc@0.25"),
            "best_global_threshold": eval_metrics.get("best_global_threshold"),
        }

    report_path = os.path.join(args.output_model, "projection_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"Saved report to {report_path}")


if __name__ == "__main__":
    main()
