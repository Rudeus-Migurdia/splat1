#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw

from arguments import ModelParams, PipelineParams
from gaussian_renderer import count_render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def load_checkpoint_into_gaussians(gaussians, checkpoint_path):
    model_params, _ = torch.load(checkpoint_path, map_location="cuda")
    if len(model_params) == 13:
        model_params = model_params[:7] + model_params[8:]
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
    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = torch.nn.Parameter(xyz.detach().to("cuda"), requires_grad=False)
    gaussians._features_dc = torch.nn.Parameter(features_dc.detach().to("cuda"), requires_grad=False)
    gaussians._features_rest = torch.nn.Parameter(features_rest.detach().to("cuda"), requires_grad=False)
    gaussians._scaling = torch.nn.Parameter(scaling.detach().to("cuda"), requires_grad=False)
    gaussians._rotation = torch.nn.Parameter(rotation.detach().to("cuda"), requires_grad=False)
    gaussians._opacity = torch.nn.Parameter(opacity.detach().to("cuda"), requires_grad=False)
    gaussians.max_radii2D = max_radii2D.detach().to("cuda")
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().to("cuda")
    gaussians.denom = denom.detach().to("cuda")
    gaussians.spatial_lr_scale = spatial_lr_scale


def polygon_mask(polygons, width, height):
    image = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(image)
    for polygon in polygons:
        draw.polygon([(float(x), float(y)) for x, y in polygon], outline=1, fill=1)
    return torch.from_numpy(np.asarray(image, dtype=bool)).cuda()


def project_variant(camera, xyz, variant):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    xyz_h = torch.cat([xyz, ones], dim=-1)
    if variant.startswith("row"):
        clip = xyz_h @ camera.full_proj_transform
    else:
        clip = xyz_h @ camera.full_proj_transform.T
    w = clip[:, 3]
    ndc = clip[:, :3] / (w[:, None] + 1e-7)
    width = int(camera.image_width)
    height = int(camera.image_height)
    px = ((ndc[:, 0] + 1.0) * 0.5 * width).long()
    if "flip_y" in variant:
        py = ((1.0 - ndc[:, 1]) * 0.5 * height).long()
    else:
        py = ((ndc[:, 1] + 1.0) * 0.5 * height).long()
    valid = (w > 0) & (ndc[:, 2] >= 0) & (ndc[:, 2] <= 1) & (px >= 0) & (px < width) & (py >= 0) & (py < height)
    return px, py, ndc[:, 2], valid


def visible_projection_ids(camera, xyz, mask, variant, z_tolerance):
    px, py, z, valid = project_variant(camera, xyz, variant)
    if not valid.any():
        return torch.empty(0, dtype=torch.long, device="cuda")
    width = int(camera.image_width)
    height = int(camera.image_height)
    linear = py[valid] * width + px[valid]
    z_valid = z[valid]
    zbuf = torch.full((height * width,), float("inf"), dtype=z.dtype, device=z.device)
    zbuf.scatter_reduce_(0, linear, z_valid, reduce="amin", include_self=True)
    close = z_valid <= zbuf[linear] + z_tolerance
    valid_idx = valid.nonzero(as_tuple=False).flatten()[close]
    inside = mask[py[valid_idx], px[valid_idx]]
    return valid_idx[inside].unique()


def raster_ids_for_mask(camera, gaussians, pipe, mask):
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    ids = count_render(camera, gaussians, pipe, bg)["per_pixel_gaussian_ids"].detach()
    if ids.ndim == 3:
        masked = ids[mask]
    else:
        masked = ids[:, mask].T
    masked = masked.reshape(-1)
    return masked[masked >= 0].long().unique()


def jaccard(a, b):
    if a.numel() == 0 and b.numel() == 0:
        return 1.0, 0, 0, 0
    a_cpu = set(a.detach().cpu().tolist())
    b_cpu = set(b.detach().cpu().tolist())
    inter = len(a_cpu & b_cpu)
    union = len(a_cpu | b_cpu)
    return (inter / union if union else 0.0), inter, len(a_cpu), len(b_cpu)


def main():
    parser = ArgumentParser()
    model = ModelParams(parser)
    pipe_params = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--max_objects", type=int, default=12)
    parser.add_argument("--z_tolerance", type=float, default=1e-4)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    safe_state(args.quiet)

    dataset = model.extract(args)
    os.makedirs(dataset.model_path, exist_ok=True)
    pipe = pipe_params.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    load_checkpoint_into_gaussians(scene.gaussians, args.checkpoint)
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    xyz = scene.gaussians.get_xyz.detach()

    variants = ["row_flip_y", "row_no_flip_y", "col_flip_y", "col_no_flip_y"]
    rows = []
    for json_name in sorted(n for n in os.listdir(args.label_dir) if n.endswith(".json")):
        data = json.load(open(os.path.join(args.label_dir, json_name)))
        image_name = os.path.splitext(data["info"]["name"])[0]
        camera = cameras[image_name]
        for obj in data["objects"][: args.max_objects]:
            category = obj["category"]
            mask = polygon_mask([obj["segmentation"]], int(camera.image_width), int(camera.image_height))
            raster = raster_ids_for_mask(camera, scene.gaussians, pipe, mask)
            item = {"image_name": image_name, "category": category, "raster_count": int(raster.numel())}
            for variant in variants:
                proj = visible_projection_ids(camera, xyz, mask, variant, args.z_tolerance)
                jac, inter, a, b = jaccard(raster, proj)
                item[variant] = {"jaccard": jac, "intersection": inter, "raster_count": a, "project_count": b}
            rows.append(item)

    summary = {}
    for variant in variants:
        summary[variant] = float(np.mean([row[variant]["jaccard"] for row in rows]))
    report = {"summary": summary, "rows": rows}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
