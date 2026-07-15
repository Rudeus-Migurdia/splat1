#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser, Namespace

import faiss
import numpy as np
import torch
from torch import nn

from arguments import ModelParams
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def as_frozen_parameter(value):
    return nn.Parameter(value.detach().to("cuda"), requires_grad=False)


def restore_checkpoint(gaussians, checkpoint_path):
    model_params, iteration = torch.load(checkpoint_path, map_location="cuda")
    if len(model_params) != 13:
        raise ValueError(f"Expected checkpoint with language features, got tuple length {len(model_params)}")
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
        opt_dict,
        spatial_lr_scale,
    ) = model_params

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
    return opt_dict, iteration


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


def decode_pq(codes, pq_index):
    feature_i16 = codes.to(torch.int16)
    invalid_neg = torch.all(feature_i16 == -1, dim=-1)
    invalid_255 = torch.all(feature_i16 == 255, dim=-1)
    valid = ~(invalid_neg | invalid_255)
    decoded = torch.zeros((codes.shape[0], 512), dtype=torch.float32, device="cuda")
    if valid.any():
        decoded_np = pq_index.sa_decode(codes[valid].detach().cpu().numpy().astype("uint8", copy=False))
        decoded[valid] = torch.from_numpy(decoded_np).to("cuda", dtype=torch.float32)
        decoded[valid] /= decoded[valid].norm(dim=-1, keepdim=True).clamp_min(1e-9)
    return decoded, valid


def encode_pq(features, valid, pq_index):
    code_size = pq_index.sa_code_size() if hasattr(pq_index, "sa_code_size") else pq_index.code_size
    codes = np.full((features.shape[0], code_size), 255, dtype=np.uint8)
    if valid.any():
        encoded = pq_index.sa_encode(features[valid].detach().cpu().numpy().astype(np.float32, copy=False))
        codes[valid.detach().cpu().numpy()] = encoded
    return torch.from_numpy(codes).to("cuda")


def main():
    parser = ArgumentParser(description="Blend baseline and group-lift PQ language features")
    model = ModelParams(parser)
    parser.add_argument("--baseline_checkpoint", required=True)
    parser.add_argument("--group_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--alpha", type=float, default=0.75, help="Weight for baseline feature where group feature is valid.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    dataset = model.extract(args)
    pq_index = faiss.read_index(args.pq_index)
    os.makedirs(args.output_model, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    opt_dict, baseline_iter = restore_checkpoint(scene.gaussians, args.baseline_checkpoint)
    baseline_codes = scene.gaussians._language_feature.detach()
    baseline_features, baseline_valid = decode_pq(baseline_codes, pq_index)

    group_gaussians = GaussianModel(dataset.sh_degree)
    group_scene = Scene(dataset, group_gaussians, shuffle=False)
    _group_opt, group_iter = restore_checkpoint(group_scene.gaussians, args.group_checkpoint)
    group_features, group_valid = decode_pq(group_scene.gaussians._language_feature.detach(), pq_index)

    if baseline_features.shape != group_features.shape:
        raise ValueError(f"Feature shape mismatch: {baseline_features.shape} vs {group_features.shape}")

    output_features = baseline_features.clone()
    blend_valid = baseline_valid & group_valid
    if blend_valid.any():
        blended = args.alpha * baseline_features[blend_valid] + (1.0 - args.alpha) * group_features[blend_valid]
        blended /= blended.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        output_features[blend_valid] = blended

    output_codes = encode_pq(output_features, baseline_valid, pq_index)
    scene.gaussians._language_feature = nn.Parameter(output_codes.detach(), requires_grad=False)

    with open(os.path.join(args.output_model, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))
    checkpoint_path = os.path.join(args.output_model, "chkpnt0.pth")
    torch.save((capture_with_language_feature(scene.gaussians, output_codes, opt_dict), 0), checkpoint_path)

    summary = {
        "method": "blend_language_checkpoints",
        "baseline_checkpoint": os.path.abspath(args.baseline_checkpoint),
        "group_checkpoint": os.path.abspath(args.group_checkpoint),
        "baseline_iteration": int(baseline_iter),
        "group_iteration": int(group_iter),
        "alpha": float(args.alpha),
        "num_gaussians": int(output_codes.shape[0]),
        "baseline_valid": int(baseline_valid.sum()),
        "group_valid": int(group_valid.sum()),
        "blend_valid": int(blend_valid.sum()),
        "blend_ratio": float(blend_valid.float().mean()),
    }
    with open(os.path.join(args.output_model, "blend_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
