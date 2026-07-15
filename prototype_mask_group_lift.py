#!/usr/bin/env python
import gc
import json
import os
import subprocess
import sys
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from types import SimpleNamespace

import faiss
import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from gaussian_renderer import count_render
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


@dataclass
class MaskObservation:
    ids: np.ndarray
    weights: np.ndarray
    feature: np.ndarray
    mass: float
    view: str
    local_mask: int


@dataclass
class Group:
    ids: np.ndarray
    weights: np.ndarray
    feature_sum: np.ndarray
    feature: np.ndarray
    mass: float
    count: int


def normalize_feature(feature):
    feature = feature.astype(np.float32, copy=False)
    norm = np.linalg.norm(feature)
    if norm < 1e-9:
        return feature
    return feature / norm


def sort_sparse(ids, weights, keep):
    if ids.size == 0:
        return ids.astype(np.int32), weights.astype(np.float32)
    order = np.argsort(ids)
    ids = ids[order]
    weights = weights[order]
    unique_ids, starts = np.unique(ids, return_index=True)
    summed = np.add.reduceat(weights, starts).astype(np.float32)
    if keep > 0 and unique_ids.size > keep:
        top = np.argpartition(summed, -keep)[-keep:]
        unique_ids = unique_ids[top]
        summed = summed[top]
        order = np.argsort(unique_ids)
        unique_ids = unique_ids[order]
        summed = summed[order]
    total = float(summed.sum())
    if total > 1e-9:
        summed /= total
    return unique_ids.astype(np.int32), summed.astype(np.float32)


def weighted_jaccard(a_ids, a_w, b_ids, b_w):
    common, ia, ib = np.intersect1d(a_ids, b_ids, assume_unique=True, return_indices=True)
    if common.size == 0:
        return 0.0
    inter = float(np.minimum(a_w[ia], b_w[ib]).sum())
    union = float(a_w.sum() + b_w.sum() - inter)
    return inter / union if union > 1e-9 else 0.0


def merge_group(group, obs, keep):
    ids = np.concatenate([group.ids, obs.ids])
    weights = np.concatenate([group.weights * group.mass, obs.weights * obs.mass])
    group.ids, group.weights = sort_sparse(ids, weights, keep)
    group.mass += obs.mass
    group.count += 1
    group.feature_sum += obs.feature * obs.mass
    group.feature = normalize_feature(group.feature_sum)


def collect_mask_observations(scene, dataset, pipe, background, args):
    observations = []
    lf_path = os.path.join(dataset.source_path, "language_features")
    num_gaussians = scene.gaussians.get_xyz.shape[0]
    cameras = scene.getTrainCameras().copy()

    with torch.no_grad():
        for camera in tqdm(cameras, desc="Lifting 2D masks"):
            language_feature_name = os.path.join(lf_path, camera.image_name)
            seg_path = language_feature_name + "_s.npy"
            feat_path = language_feature_name + "_f.npy"
            if not os.path.exists(seg_path) or not os.path.exists(feat_path):
                continue
            feature_map = np.load(feat_path).astype(np.float32)
            if feature_map.size == 0:
                continue
            feature_map /= np.linalg.norm(feature_map, axis=-1, keepdims=True).clip(min=1e-9)

            render_pkg = count_render(camera, scene.gaussians, pipe, background)
            ids = render_pkg["per_pixel_gaussian_ids"].detach()
            contribution = render_pkg["per_pixel_gaussian_contributions"].detach()
            seg = torch.from_numpy(np.load(seg_path)).to(torch.int64)[args.feature_level].to(ids.device)
            mask_idx = (seg != -1).nonzero(as_tuple=True)
            if mask_idx[0].numel() == 0:
                continue

            contrib = contribution[mask_idx]
            ray_ids = ids[mask_idx]
            topk = min(args.topk, contrib.shape[1])
            weights, indices = torch.topk(contrib, topk, dim=1)
            ray_ids = torch.gather(ray_ids, 1, indices)
            local_masks = seg[mask_idx].repeat(topk, 1).T.reshape(-1)
            ray_ids = ray_ids.reshape(-1)
            weights = weights.reshape(-1)
            valid = ray_ids != -1
            if args.min_contribution > 0:
                valid &= weights >= args.min_contribution
            if not valid.any():
                continue

            local_masks_np = local_masks[valid].detach().cpu().numpy().astype(np.int32)
            ray_ids_np = ray_ids[valid].detach().cpu().numpy().astype(np.int32)
            weights_np = weights[valid].detach().cpu().numpy().astype(np.float32)
            for local_mask in np.unique(local_masks_np):
                if local_mask < 0 or local_mask >= feature_map.shape[0]:
                    continue
                sel = local_masks_np == local_mask
                if int(sel.sum()) < args.min_mask_samples:
                    continue
                mask_ids, mask_weights = sort_sparse(
                    ray_ids_np[sel],
                    weights_np[sel],
                    min(args.mask_keep_gaussians, num_gaussians),
                )
                if mask_ids.size < args.min_mask_gaussians:
                    continue
                mass = float(weights_np[sel].sum())
                observations.append(
                    MaskObservation(
                        ids=mask_ids,
                        weights=mask_weights,
                        feature=feature_map[int(local_mask)],
                        mass=mass,
                        view=camera.image_name,
                        local_mask=int(local_mask),
                    )
                )

            del render_pkg, ids, contribution, seg
            torch.cuda.empty_cache()

    return observations


def build_groups(observations, args):
    groups = []
    inverted = {}
    for obs in tqdm(observations, desc="Matching 3D groups"):
        candidate_ids = set()
        seed_order = np.argsort(obs.weights)[-args.candidate_seed_gaussians :]
        for gid in obs.ids[seed_order]:
            candidate_ids.update(inverted.get(int(gid), ()))

        best_gid = -1
        best_score = -1.0
        for gid in candidate_ids:
            group = groups[gid]
            overlap = weighted_jaccard(obs.ids, obs.weights, group.ids, group.weights)
            if overlap < args.min_group_iou:
                continue
            cosine = float(np.dot(obs.feature, group.feature))
            if cosine < args.min_group_cosine:
                continue
            score = args.iou_weight * overlap + (1.0 - args.iou_weight) * cosine
            if score > best_score:
                best_gid = gid
                best_score = score

        if best_gid >= 0 and best_score >= args.merge_score:
            merge_group(groups[best_gid], obs, args.group_keep_gaussians)
            group = groups[best_gid]
            for gid in group.ids[np.argsort(group.weights)[-args.index_group_gaussians :]]:
                inverted.setdefault(int(gid), []).append(best_gid)
        else:
            feature_sum = obs.feature * obs.mass
            group = Group(
                ids=obs.ids.copy(),
                weights=obs.weights.copy(),
                feature_sum=feature_sum.copy(),
                feature=normalize_feature(feature_sum),
                mass=obs.mass,
                count=1,
            )
            groups.append(group)
            new_gid = len(groups) - 1
            for gid in obs.ids[np.argsort(obs.weights)[-args.index_group_gaussians :]]:
                inverted.setdefault(int(gid), []).append(new_gid)

    return groups


def assign_group_features(num_gaussians, groups, args):
    features = np.zeros((num_gaussians, 512), dtype=np.float32)
    feature_sum = np.zeros((num_gaussians, 512), dtype=np.float32)
    feature_weight_sum = np.zeros(num_gaussians, dtype=np.float32)
    best_score = np.zeros(num_gaussians, dtype=np.float32)
    second_score = np.zeros(num_gaussians, dtype=np.float32)
    best_group = -np.ones(num_gaussians, dtype=np.int32)
    top_group_ids = -np.ones((num_gaussians, args.keep_point_groups), dtype=np.int32)
    top_group_scores = np.zeros((num_gaussians, args.keep_point_groups), dtype=np.float32)
    usable_groups = 0

    for gid, group in enumerate(groups):
        if group.count < args.min_group_observations:
            continue
        usable_groups += 1
        confidence = np.sqrt(float(group.count))
        scores = group.weights * confidence
        for point_id, score in zip(group.ids, scores):
            point_id = int(point_id)
            score = float(score)
            if score <= 0:
                continue
            if args.assignment_mode == "soft":
                weight = score ** args.soft_score_power
                feature_sum[point_id] += group.feature * weight
                feature_weight_sum[point_id] += weight
            if score > best_score[point_id]:
                second_score[point_id] = best_score[point_id]
                best_score[point_id] = score
                best_group[point_id] = gid
            elif score > second_score[point_id]:
                second_score[point_id] = score
            if score > top_group_scores[point_id, -1]:
                insert = int(np.searchsorted(-top_group_scores[point_id], -score, side="right"))
                insert = min(insert, args.keep_point_groups - 1)
                if insert < args.keep_point_groups - 1:
                    top_group_scores[point_id, insert + 1 :] = top_group_scores[point_id, insert:-1]
                    top_group_ids[point_id, insert + 1 :] = top_group_ids[point_id, insert:-1]
                top_group_scores[point_id, insert] = score
                top_group_ids[point_id, insert] = gid

    valid = best_group >= 0
    if args.min_assign_score > 0:
        valid &= best_score >= args.min_assign_score
    if args.min_assign_margin > 0:
        valid &= (best_score - second_score) >= args.min_assign_margin

    if args.assignment_mode == "soft":
        valid &= feature_weight_sum > 0
        features[valid] = feature_sum[valid] / feature_weight_sum[valid, None]
        norms = np.linalg.norm(features[valid], axis=-1, keepdims=True).clip(min=1e-9)
        features[valid] /= norms
    else:
        for gid, group in enumerate(groups):
            point_mask = valid & (best_group == gid)
            if point_mask.any():
                features[point_mask] = group.feature

    return features, valid, usable_groups, best_group, best_score, second_score, top_group_ids, top_group_scores, feature_weight_sum


def pack_group_metadata(groups):
    features = np.stack([group.feature for group in groups]).astype(np.float32) if groups else np.zeros((0, 512), dtype=np.float32)
    counts = np.asarray([group.count for group in groups], dtype=np.int32)
    masses = np.asarray([group.mass for group in groups], dtype=np.float32)
    sizes = np.asarray([group.ids.size for group in groups], dtype=np.int32)
    return features, counts, masses, sizes


def encode_with_pq(features, valid, pq_index):
    code_size = pq_index.sa_code_size() if hasattr(pq_index, "sa_code_size") else pq_index.code_size
    pq_codes = np.full((features.shape[0], code_size), 255, dtype=np.uint8)
    if valid.any():
        encoded = pq_index.sa_encode(features[valid].astype(np.float32, copy=False))
        pq_codes[valid] = encoded
    return torch.from_numpy(pq_codes).to("cuda")


def maybe_run_eval(args, output_model):
    if not args.run_eval:
        return None
    cmd = [
        sys.executable,
        "eval_lerf_ovs_miou.py",
        "-s",
        os.path.abspath(args.source_path),
        "-m",
        os.path.abspath(output_model),
        "--checkpoint",
        os.path.join(output_model, "chkpnt0.pth"),
        "--label_dir",
        os.path.abspath(args.label_dir),
        "--pq_index",
        os.path.abspath(args.pq_index),
        "--thresholds",
        *[str(value) for value in args.thresholds],
    ]
    if args.eval_visualizations:
        cmd.append("--save_visualizations")
        cmd.extend(["--max_visualizations", str(args.max_visualizations)])
    subprocess.run(cmd, check=True)
    metrics_path = os.path.join(output_model, "eval", "lerf_ovs_miou", "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    return None


def main():
    parser = ArgumentParser(description="MUSplat-style prototype: lift 2D masks to 3D groups and attach group PQ codes")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--start_checkpoint", required=True)
    parser.add_argument("--pq_index", required=True)
    parser.add_argument("--output_model", required=True)
    parser.add_argument("--label_dir", default=None)
    parser.add_argument("--topk", type=int, default=45)
    parser.add_argument("--mask_keep_gaussians", type=int, default=512)
    parser.add_argument("--group_keep_gaussians", type=int, default=2048)
    parser.add_argument("--candidate_seed_gaussians", type=int, default=48)
    parser.add_argument("--index_group_gaussians", type=int, default=96)
    parser.add_argument("--min_mask_samples", type=int, default=64)
    parser.add_argument("--min_mask_gaussians", type=int, default=12)
    parser.add_argument("--min_contribution", type=float, default=0.0)
    parser.add_argument("--min_group_iou", type=float, default=0.015)
    parser.add_argument("--min_group_cosine", type=float, default=0.70)
    parser.add_argument("--merge_score", type=float, default=0.32)
    parser.add_argument("--iou_weight", type=float, default=0.65)
    parser.add_argument("--min_group_observations", type=int, default=2)
    parser.add_argument("--min_assign_score", type=float, default=0.0)
    parser.add_argument("--min_assign_margin", type=float, default=0.0)
    parser.add_argument("--assignment_mode", choices=["hard", "soft"], default="hard")
    parser.add_argument("--keep_point_groups", type=int, default=4)
    parser.add_argument("--soft_score_power", type=float, default=1.0)
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--eval_visualizations", action="store_true")
    parser.add_argument("--max_visualizations", type=int, default=24)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9])
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)

    os.makedirs(args.output_model, exist_ok=True)
    with open(os.path.join(args.output_model, "cfg_args"), "w") as f:
        f.write(str(Namespace(**vars(args))))

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    opt_dict, checkpoint_iteration = load_checkpoint_into_gaussians(scene.gaussians, args.start_checkpoint)
    background = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0], dtype=torch.float32, device="cuda")

    observations = collect_mask_observations(scene, dataset, pipe, background, args)
    groups = build_groups(observations, args)
    features, valid, usable_groups, best_group, best_score, second_score, top_group_ids, top_group_scores, feature_weight_sum = assign_group_features(
        scene.gaussians.get_xyz.shape[0],
        groups,
        args,
    )

    pq_index = faiss.read_index(args.pq_index)
    pq_codes = encode_with_pq(features, valid, pq_index)
    scene.gaussians._language_feature = nn.Parameter(pq_codes.detach(), requires_grad=False)
    checkpoint_path = os.path.join(args.output_model, "chkpnt0.pth")
    torch.save((capture_with_language_feature(scene.gaussians, pq_codes, opt_dict), 0), checkpoint_path)
    group_features, group_counts, group_masses, group_sizes = pack_group_metadata(groups)
    np.save(os.path.join(args.output_model, "group_features.npy"), group_features)
    np.savez_compressed(
        os.path.join(args.output_model, "group_metadata.npz"),
        group_counts=group_counts,
        group_masses=group_masses,
        group_sizes=group_sizes,
    )
    np.savez_compressed(
        os.path.join(args.output_model, "point_group_assignments.npz"),
        top_group_ids=top_group_ids,
        top_group_scores=top_group_scores,
        best_group=best_group,
        best_score=best_score,
        second_score=second_score,
        feature_weight_sum=feature_weight_sum,
        valid=valid,
    )

    summary = {
        "method": "mask_group_lift",
        "source_checkpoint": os.path.abspath(args.start_checkpoint),
        "source_checkpoint_iteration": int(checkpoint_iteration),
        "num_gaussians": int(scene.gaussians.get_xyz.shape[0]),
        "num_mask_observations": int(len(observations)),
        "num_groups": int(len(groups)),
        "num_usable_groups": int(usable_groups),
        "valid_gaussians": int(valid.sum()),
        "valid_ratio": float(valid.mean()),
        "mean_best_score": float(best_score[valid].mean()) if valid.any() else 0.0,
        "mean_margin": float((best_score[valid] - second_score[valid]).mean()) if valid.any() else 0.0,
        "assignment_mode": args.assignment_mode,
        "keep_point_groups": int(args.keep_point_groups),
        "mean_feature_weight_sum": float(feature_weight_sum[valid].mean()) if valid.any() else 0.0,
        "mean_active_groups_per_valid_point": float((top_group_scores[valid] > 0).sum(axis=1).mean()) if valid.any() else 0.0,
        "group_features_path": os.path.abspath(os.path.join(args.output_model, "group_features.npy")),
        "group_metadata_path": os.path.abspath(os.path.join(args.output_model, "group_metadata.npz")),
        "point_group_assignments_path": os.path.abspath(os.path.join(args.output_model, "point_group_assignments.npz")),
        "args": vars(args),
    }
    with open(os.path.join(args.output_model, "group_lift_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    metrics = maybe_run_eval(args, args.output_model)
    if metrics is not None:
        summary["eval"] = {
            "best_global_threshold": metrics.get("best_global_threshold"),
            "oracle_mIoU": metrics.get("mIoU"),
            "oracle_mAcc@0.25": metrics.get("mAcc@0.25"),
        }
        with open(os.path.join(args.output_model, "group_lift_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    del pq_codes, scene, gaussians
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
