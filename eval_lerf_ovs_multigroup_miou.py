#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from evaluation.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state


def as_frozen_parameter(value):
    return nn.Parameter(value.detach().to("cuda"), requires_grad=False)


def load_geometry_checkpoint(gaussians, checkpoint_path):
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
            _opt_dict,
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
    Image.fromarray(np.concatenate([rgb_np, heat, pred_rgb, gt_rgb], axis=1)).save(out_path)


def load_group_assignments(assignments_path):
    assignments = np.load(assignments_path)
    top_group_ids = torch.from_numpy(assignments["top_group_ids"].astype(np.int64)).to("cuda")
    top_group_scores = torch.from_numpy(assignments["top_group_scores"].astype(np.float32)).to("cuda")
    return top_group_ids, top_group_scores


def load_multigroup_tokens(group_features_path, assignments_path):
    group_features = torch.from_numpy(np.load(group_features_path).astype(np.float32)).to("cuda")
    group_features /= group_features.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    top_group_ids, top_group_scores = load_group_assignments(assignments_path)
    return group_features, top_group_ids, top_group_scores


def load_coarse_tokens(coarse_features_path, group_to_coarse_path, num_groups):
    if not coarse_features_path and not group_to_coarse_path:
        return None, None
    if not coarse_features_path or not group_to_coarse_path:
        raise ValueError("--coarse_features and --group_to_coarse must be provided together.")
    coarse_features = torch.from_numpy(np.load(coarse_features_path).astype(np.float32)).to("cuda")
    coarse_features /= coarse_features.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    group_to_coarse = torch.from_numpy(np.load(group_to_coarse_path).astype(np.int64)).to("cuda")
    if group_to_coarse.numel() != num_groups:
        raise ValueError(
            f"group_to_coarse has {group_to_coarse.numel()} entries, but group_features has {num_groups} groups."
        )
    if group_to_coarse.numel() and int(group_to_coarse.max()) >= coarse_features.shape[0]:
        raise ValueError("group_to_coarse references ids outside coarse_features.")
    return coarse_features, group_to_coarse


def load_reverse_codebook(reverse_codebook_dir, num_groups):
    if not reverse_codebook_dir:
        return None
    codebook_path = os.path.join(reverse_codebook_dir, "codebook.npy")
    group_to_code_path = os.path.join(reverse_codebook_dir, "group_to_code.npy")
    residual_path = os.path.join(reverse_codebook_dir, "group_residuals.npy")
    for path in (codebook_path, group_to_code_path):
        if not os.path.isfile(path):
            raise ValueError(f"Missing reverse codebook artifact: {path}")

    codebook = torch.from_numpy(np.load(codebook_path).astype(np.float32)).to("cuda")
    codebook /= codebook.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    group_to_code = torch.from_numpy(np.load(group_to_code_path).astype(np.int64)).to("cuda")
    if group_to_code.numel() != num_groups:
        raise ValueError(
            f"reverse group_to_code has {group_to_code.numel()} entries, but group_features has {num_groups} groups."
        )
    if group_to_code.numel() and int(group_to_code.max()) >= codebook.shape[0]:
        raise ValueError("reverse group_to_code references ids outside codebook.")

    residuals = None
    if os.path.isfile(residual_path):
        residuals = torch.from_numpy(np.load(residual_path).astype(np.float32)).to("cuda")
        if residuals.shape[0] != num_groups:
            raise ValueError(
                f"reverse residuals have {residuals.shape[0]} entries, but group_features has {num_groups} groups."
            )
    stats = {}
    for name in ("group_usage", "group_residual_norm", "group_cosine_to_code"):
        path = os.path.join(reverse_codebook_dir, f"{name}.npy")
        if os.path.isfile(path):
            value = torch.from_numpy(np.load(path).astype(np.float32)).to("cuda")
            if value.numel() != num_groups:
                raise ValueError(
                    f"reverse {name} has {value.numel()} entries, but group_features has {num_groups} groups."
                )
            stats[name] = value
    return {
        "dir": os.path.abspath(reverse_codebook_dir),
        "codebook": codebook,
        "group_to_code": group_to_code,
        "residuals": residuals,
        "stats": stats,
    }


def load_dynamic_hierarchical_codebook(codebook_dir, num_groups=None):
    if not codebook_dir:
        return None
    required = (
        "coarse_codebook.npy",
        "fine_codebook.npy",
        "group_coarse_ids.npy",
        "group_fine_candidate_ids.npy",
        "group_fine_candidate_scores.npy",
        "fine_code_to_groups.npz",
    )
    for name in required:
        path = os.path.join(codebook_dir, name)
        if not os.path.isfile(path):
            raise ValueError(f"Missing dynamic hierarchical codebook artifact: {path}")

    coarse_codebook = torch.from_numpy(
        np.load(os.path.join(codebook_dir, "coarse_codebook.npy")).astype(np.float32)
    ).to("cuda")
    fine_codebook = torch.from_numpy(
        np.load(os.path.join(codebook_dir, "fine_codebook.npy")).astype(np.float32)
    ).to("cuda")
    coarse_codebook /= coarse_codebook.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    fine_codebook /= fine_codebook.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    group_coarse_ids = torch.from_numpy(
        np.load(os.path.join(codebook_dir, "group_coarse_ids.npy")).astype(np.int64)
    ).to("cuda")
    candidate_ids = torch.from_numpy(
        np.load(os.path.join(codebook_dir, "group_fine_candidate_ids.npy")).astype(np.int64)
    ).to("cuda")
    candidate_scores = torch.from_numpy(
        np.load(os.path.join(codebook_dir, "group_fine_candidate_scores.npy")).astype(np.float32)
    ).to("cuda")
    if num_groups is not None and (group_coarse_ids.numel() != num_groups or candidate_ids.shape[0] != num_groups):
        raise ValueError("Dynamic hierarchical group attachments do not match group_features.")
    if candidate_ids.ndim != 2 or candidate_scores.shape != candidate_ids.shape:
        raise ValueError("Dynamic hierarchical fine candidates must have matching [num_groups, top_m] shapes.")
    if group_coarse_ids.numel() and int(group_coarse_ids.max()) >= coarse_codebook.shape[0]:
        raise ValueError("Dynamic hierarchical coarse ids reference values outside coarse_codebook.")
    valid = candidate_ids >= 0
    if valid.any() and int(candidate_ids[valid].max()) >= fine_codebook.shape[0]:
        raise ValueError("Dynamic hierarchical fine ids reference values outside fine_codebook.")

    reverse = np.load(os.path.join(codebook_dir, "fine_code_to_groups.npz"))
    reverse_indices = torch.from_numpy(reverse["indices"].astype(np.int64)).to("cuda")
    reverse_offsets = torch.from_numpy(reverse["offsets"].astype(np.int64)).to("cuda")
    if reverse_offsets.numel() != fine_codebook.shape[0] + 1:
        raise ValueError("Dynamic hierarchical reverse offsets do not match fine_codebook.")
    return {
        "dir": os.path.abspath(codebook_dir),
        "coarse_codebook": coarse_codebook,
        "fine_codebook": fine_codebook,
        "group_coarse_ids": group_coarse_ids,
        "candidate_ids": candidate_ids,
        "candidate_scores": candidate_scores,
        "reverse_indices": reverse_indices,
        "reverse_offsets": reverse_offsets,
    }


def dynamic_reverse_group_mask(fine_activation, dynamic_codebook, top_codes):
    if top_codes <= 0 or top_codes >= fine_activation.shape[0]:
        return torch.ones_like(dynamic_codebook["group_coarse_ids"], dtype=torch.bool)
    selected_codes = torch.topk(fine_activation.squeeze(-1), k=top_codes).indices
    offsets = dynamic_codebook["reverse_offsets"]
    indices = dynamic_codebook["reverse_indices"]
    mounted_groups = [indices[offsets[code] : offsets[code + 1]] for code in selected_codes]
    mounted_groups = [groups for groups in mounted_groups if groups.numel()]
    mask = torch.zeros_like(dynamic_codebook["group_coarse_ids"], dtype=torch.bool)
    if mounted_groups:
        mask[torch.cat(mounted_groups)] = True
    return mask


def dynamic_hierarchical_group_activation(dynamic_codebook, clip_model, category_idx, args):
    """Read only discrete code vectors; teacher features are never consulted here."""
    coarse_activation = clip_model.get_activation(dynamic_codebook["coarse_codebook"], category_idx)
    fine_activation = clip_model.get_activation(dynamic_codebook["fine_codebook"], category_idx)
    candidate_ids = dynamic_codebook["candidate_ids"]
    candidate_scores = dynamic_codebook["candidate_scores"]
    valid = candidate_ids >= 0
    safe_ids = candidate_ids.clamp_min(0)
    candidate_activation = fine_activation[safe_ids].squeeze(-1)
    candidate_activation = torch.where(valid, candidate_activation, torch.zeros_like(candidate_activation))
    prior = torch.log(candidate_scores.clamp_min(1e-6)) * float(args.dynamic_candidate_prior_power)
    logits = candidate_activation / max(float(args.dynamic_fine_temperature), 1e-6) + prior
    logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
    weights = torch.softmax(logits, dim=1) * valid.float()
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
    fine_per_group = (candidate_activation * weights).sum(dim=1, keepdim=True)

    candidate_count = valid.sum(dim=1).clamp_min(1).float()
    entropy = -(weights * weights.clamp_min(1e-9).log()).sum(dim=1, keepdim=True)
    entropy_normalizer = candidate_count.log()
    entropy_normalizer = torch.where(candidate_count > 1.0, entropy_normalizer, torch.ones_like(entropy_normalizer))
    normalized_entropy = entropy / entropy_normalizer.unsqueeze(-1)
    specificity = (1.0 - normalized_entropy).clamp(0.0, 1.0)
    fine_blend = float(args.dynamic_fine_min_blend) + (
        float(args.dynamic_fine_max_blend) - float(args.dynamic_fine_min_blend)
    ) * specificity
    coarse_per_group = coarse_activation[dynamic_codebook["group_coarse_ids"]]
    group_activation = (1.0 - fine_blend) * coarse_per_group + fine_blend * fine_per_group
    group_mask = dynamic_reverse_group_mask(fine_activation, dynamic_codebook, args.dynamic_reverse_top_codes)
    return group_activation, group_mask, float(fine_blend.mean().item())


def make_reverse_group_features(reverse_codebook, residual_weight):
    base = reverse_codebook["codebook"][reverse_codebook["group_to_code"]]
    residuals = reverse_codebook["residuals"]
    if residuals is None or residual_weight == 0.0:
        return base
    features = base + float(residual_weight) * residuals
    return features / features.norm(dim=-1, keepdim=True).clamp_min(1e-9)


def reverse_group_mask(code_activation, group_to_code, top_codes):
    if top_codes <= 0 or top_codes >= code_activation.shape[0]:
        return torch.ones_like(group_to_code, dtype=torch.bool)
    selected = torch.topk(code_activation.squeeze(-1), k=top_codes).indices
    code_mask = torch.zeros(code_activation.shape[0], dtype=torch.bool, device=code_activation.device)
    code_mask[selected] = True
    return code_mask[group_to_code]


def make_reverse_group_prior(reverse_codebook, mode, power=1.0, residual_temperature=0.5):
    if reverse_codebook is None or mode == "none":
        return None
    stats = reverse_codebook.get("stats", {})
    priors = []
    if "usage" in mode:
        usage = stats.get("group_usage")
        if usage is None:
            raise ValueError("--reverse_group_prior requires group_usage.npy in --reverse_codebook_dir.")
        usage = usage.clamp_min(0.0)
        usage = usage / usage.max().clamp_min(1e-9)
        priors.append(usage)
    if "residual" in mode:
        residual_norm = stats.get("group_residual_norm")
        if residual_norm is None:
            raise ValueError("--reverse_group_prior requires group_residual_norm.npy in --reverse_codebook_dir.")
        priors.append(torch.exp(-residual_norm / max(float(residual_temperature), 1e-6)))
    if "cosine" in mode:
        cosine = stats.get("group_cosine_to_code")
        if cosine is None:
            raise ValueError("--reverse_group_prior requires group_cosine_to_code.npy in --reverse_codebook_dir.")
        priors.append(((cosine + 1.0) * 0.5).clamp(0.0, 1.0))
    if not priors:
        raise ValueError(f"Unknown reverse group prior: {mode}")
    prior = torch.ones_like(priors[0])
    for item in priors:
        prior = prior * item.clamp_min(1e-6)
    return prior.pow(float(power)).clamp_min(1e-6)


def normalize_token_activation(activation, mode, low, high, temperature=1.0, valid_mask=None):
    if mode == "none":
        return activation
    values = activation.squeeze(-1)
    selected = values
    if valid_mask is not None:
        selected = values[valid_mask]
    if selected.numel() == 0:
        return activation
    if mode == "token_minmax":
        lo = selected.min()
        hi = selected.max()
        normalized = (values - lo) / (hi - lo).clamp_min(1e-6)
        return normalized.clamp(0.0, 1.0).unsqueeze(-1)
    if mode == "token_percentile":
        q = torch.tensor([float(low), float(high)], dtype=torch.float32, device=selected.device) / 100.0
        lo, hi = torch.quantile(selected.float(), q).to(selected.dtype)
        normalized = (values - lo) / (hi - lo).clamp_min(1e-6)
        return normalized.clamp(0.0, 1.0).unsqueeze(-1)
    if mode == "token_zscore_sigmoid":
        centered = values - selected.mean()
        scaled = centered / selected.std(unbiased=False).clamp_min(1e-6)
        return torch.sigmoid(scaled / max(float(temperature), 1e-6)).unsqueeze(-1)
    raise ValueError(f"Unknown activation normalization: {mode}")


def point_activation_from_groups(
    group_activation,
    top_group_ids,
    top_group_scores,
    mode,
    score_power,
    blend_alpha,
    eval_topk,
    query_temperature=0.07,
    query_prior_power=1.0,
    group_valid_mask=None,
    group_score_prior=None,
):
    if eval_topk > 0:
        top_group_ids = top_group_ids[:, :eval_topk]
        top_group_scores = top_group_scores[:, :eval_topk]
    valid = top_group_ids >= 0
    safe_ids = top_group_ids.clamp_min(0)
    if group_valid_mask is not None:
        valid = valid & group_valid_mask[safe_ids]
    gathered = group_activation[safe_ids].squeeze(-1)
    gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
    scores = torch.where(valid, top_group_scores.clamp_min(0.0), torch.zeros_like(top_group_scores))
    if group_score_prior is not None:
        scores = scores * torch.where(valid, group_score_prior[safe_ids], torch.zeros_like(scores))

    if mode == "max":
        return gathered.max(dim=1, keepdim=True).values
    if mode == "score_max":
        return (gathered * scores.pow(score_power)).max(dim=1, keepdim=True).values
    if mode == "weighted":
        weights = scores.pow(score_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return (gathered * weights).sum(dim=1, keepdim=True)
    if mode == "weighted_maxblend":
        weights = scores.pow(score_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        weighted = (gathered * weights).sum(dim=1, keepdim=True)
        maxed = gathered.max(dim=1, keepdim=True).values
        return blend_alpha * weighted + (1.0 - blend_alpha) * maxed
    if mode == "noisy_or":
        weights = scores.pow(score_power)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        return 1.0 - torch.prod(1.0 - (gathered * weights).clamp(0.0, 1.0), dim=1, keepdim=True)
    if mode in ("query_softmax", "query_softmax_maxblend"):
        prior = scores.clamp_min(1e-6).log() * query_prior_power
        logits = gathered / max(float(query_temperature), 1e-6) + prior
        logits = torch.where(valid, logits, torch.full_like(logits, -1e9))
        weights = torch.softmax(logits, dim=1)
        weights = weights * valid.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-9)
        weighted = (gathered * weights).sum(dim=1, keepdim=True)
        if mode == "query_softmax":
            return weighted
        maxed = gathered.max(dim=1, keepdim=True).values
        return blend_alpha * weighted + (1.0 - blend_alpha) * maxed
    raise ValueError(f"Unknown aggregation mode: {mode}")


def point_activation_from_coarse_groups(
    coarse_activation,
    group_to_coarse,
    top_group_ids,
    top_group_scores,
    mode,
    score_power,
    blend_alpha,
    eval_topk,
    query_temperature=0.07,
    query_prior_power=1.0,
    group_valid_mask=None,
    group_score_prior=None,
):
    valid = top_group_ids >= 0
    safe_group_ids = top_group_ids.clamp_min(0)
    if group_valid_mask is not None:
        valid = valid & group_valid_mask[safe_group_ids]
    top_coarse_ids = group_to_coarse[safe_group_ids]
    top_coarse_ids = torch.where(valid, top_coarse_ids, torch.full_like(top_coarse_ids, -1))
    return point_activation_from_groups(
        coarse_activation,
        top_coarse_ids,
        top_group_scores,
        mode,
        score_power,
        blend_alpha,
        eval_topk,
        query_temperature,
        query_prior_power,
    )


def estimate_query_adaptive_coarse_blend(fine_group_activation, max_blend, min_blend, topk):
    values = fine_group_activation.squeeze(-1).float()
    if values.numel() == 0 or max_blend <= 0.0:
        return 0.0
    lo = values.min()
    hi = values.max()
    normalized = (values - lo) / (hi - lo).clamp_min(1e-9)
    k = min(max(2, topk), normalized.numel())
    top_values = torch.topk(normalized, k=k).values
    specificity = (top_values[0] - top_values[-1]).clamp(0.0, 1.0)
    blend = min_blend + (max_blend - min_blend) * (1.0 - specificity)
    return float(blend.clamp(0.0, max_blend).item())


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


def main():
    parser = ArgumentParser(description="Evaluate LeRF-OVS mIoU with per-Gaussian multi-group semantic tokens")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--checkpoint", required=True, help="Geometry checkpoint, 12-tuple 3DGS or 13-tuple Dr.Splat.")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument(
        "--group_features",
        default=None,
        help="Continuous group tokens for legacy paths. Dynamic discrete codebook evaluation does not load this file.",
    )
    parser.add_argument("--assignments", required=True)
    parser.add_argument("--coarse_features", default=None)
    parser.add_argument("--group_to_coarse", default=None)
    parser.add_argument(
        "--dynamic_hierarchical_codebook_dir",
        default=None,
        help="Fully discrete coarse/fine codebook artifact with top-M group attachments and code-to-group reverse mounts.",
    )
    parser.add_argument(
        "--dynamic_fine_temperature",
        type=float,
        default=0.10,
        help="Query temperature for selecting a group-attached fine code.",
    )
    parser.add_argument(
        "--dynamic_candidate_prior_power",
        type=float,
        default=0.25,
        help="Strength of the offline code-assignment cosine prior during dynamic fine-code routing.",
    )
    parser.add_argument(
        "--dynamic_fine_min_blend",
        type=float,
        default=0.10,
        help="Minimum per-group fine-code weight; ambiguous queries fall back toward the coarse code.",
    )
    parser.add_argument(
        "--dynamic_fine_max_blend",
        type=float,
        default=0.90,
        help="Maximum per-group fine-code weight for specific queries.",
    )
    parser.add_argument(
        "--dynamic_reverse_top_codes",
        type=int,
        default=0,
        help="Use code-to-group reverse mounting to keep only the query top-M fine codes. 0 keeps all.",
    )
    parser.add_argument(
        "--reverse_codebook_dir",
        default=None,
        help="Directory containing codebook.npy, group_to_code.npy, and optional group_residuals.npy for codeword-to-group routing.",
    )
    parser.add_argument(
        "--reverse_top_codes",
        type=int,
        default=0,
        help="Activate only groups mounted under the top-M query codewords. 0 keeps all codewords active.",
    )
    parser.add_argument(
        "--reverse_residual_weight",
        type=float,
        default=0.0,
        help="Residual weight for reconstructing group tokens from reverse codebook prototypes plus group residuals.",
    )
    parser.add_argument(
        "--reverse_code_blend",
        type=float,
        default=0.0,
        help="Blend each group activation with its parent codeword activation after residual refinement.",
    )
    parser.add_argument(
        "--reverse_group_prior",
        choices=["none", "usage", "residual", "cosine", "usage_residual", "usage_cosine", "residual_cosine", "usage_residual_cosine"],
        default="none",
        help="Use reverse-codebook statistics as an extra routing prior over groups.",
    )
    parser.add_argument(
        "--reverse_prior_power",
        type=float,
        default=1.0,
        help="Exponent applied to --reverse_group_prior before multiplying assignment scores.",
    )
    parser.add_argument(
        "--reverse_residual_temperature",
        type=float,
        default=0.5,
        help="Temperature for residual-based reverse group prior exp(-residual_norm / temperature).",
    )
    parser.add_argument(
        "--coarse_blend",
        type=float,
        default=0.0,
        help="Blend parent/coarse token activation into fine token activation. 0 keeps the original evaluator.",
    )
    parser.add_argument(
        "--coarse_blend_mode",
        choices=["fixed", "query_adaptive"],
        default="fixed",
        help="fixed uses --coarse_blend directly; query_adaptive lowers coarse weight for peaked fine-token queries.",
    )
    parser.add_argument(
        "--coarse_min_blend",
        type=float,
        default=0.0,
        help="Minimum coarse blend used by query_adaptive mode.",
    )
    parser.add_argument(
        "--coarse_specificity_topk",
        type=int,
        default=16,
        help="Top-K fine token activations used to estimate query specificity.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--aggregation",
        choices=[
            "max",
            "score_max",
            "weighted",
            "weighted_maxblend",
            "noisy_or",
            "query_softmax",
            "query_softmax_maxblend",
        ],
        default="max",
    )
    parser.add_argument("--score_power", type=float, default=1.0)
    parser.add_argument("--blend_alpha", type=float, default=0.75)
    parser.add_argument("--eval_topk", type=int, default=0, help="Use only the first K assigned group tokens; 0 keeps all.")
    parser.add_argument(
        "--query_temperature",
        type=float,
        default=0.07,
        help="Temperature for query-conditioned softmax over each Gaussian's group tokens.",
    )
    parser.add_argument(
        "--query_prior_power",
        type=float,
        default=1.0,
        help="Strength of geometric assignment scores as a prior in query-conditioned softmax aggregation.",
    )
    parser.add_argument(
        "--score_calibration",
        choices=["none", "frame_minmax", "frame_percentile", "category_percentile"],
        default="none",
    )
    parser.add_argument("--calibration_low", type=float, default=1.0)
    parser.add_argument("--calibration_high", type=float, default=99.0)
    parser.add_argument(
        "--activation_normalization",
        choices=["none", "token_minmax", "token_percentile", "token_zscore_sigmoid"],
        default="none",
        help="Normalize per-query token activations before Gaussian-level aggregation.",
    )
    parser.add_argument("--activation_norm_low", type=float, default=1.0)
    parser.add_argument("--activation_norm_high", type=float, default=99.0)
    parser.add_argument(
        "--activation_norm_temperature",
        type=float,
        default=1.0,
        help="Temperature used by token_zscore_sigmoid activation normalization.",
    )
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--max_visualizations", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = SimpleNamespace(include_feature=False)

    labels, categories = load_lerf_labels(args.label_dir)
    output_dir = args.output or os.path.join(dataset.model_path, "eval", "lerf_ovs_multigroup_miou")
    os.makedirs(output_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(scene.gaussians, args.checkpoint)
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    missing_cameras = sorted(set(labels) - set(cameras))
    if missing_cameras:
        raise ValueError(f"Missing labeled cameras in scene: {missing_cameras}")

    top_group_ids, top_group_scores = load_group_assignments(args.assignments)
    group_features = None
    if args.group_features:
        group_features = torch.from_numpy(np.load(args.group_features).astype(np.float32)).to("cuda")
        group_features /= group_features.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    if top_group_ids.shape[0] != scene.gaussians.get_xyz.shape[0]:
        raise ValueError(
            f"Assignment point count {top_group_ids.shape[0]} does not match Gaussian count {scene.gaussians.get_xyz.shape[0]}"
        )
    num_groups = int(group_features.shape[0]) if group_features is not None else None
    if group_features is not None and top_group_ids.numel() and int(top_group_ids.max()) >= num_groups:
        raise ValueError("Assignments reference group ids outside group_features.")
    if group_features is None and not args.dynamic_hierarchical_codebook_dir:
        raise ValueError("--group_features is required unless --dynamic_hierarchical_codebook_dir is provided.")
    coarse_features, group_to_coarse = load_coarse_tokens(
        args.coarse_features,
        args.group_to_coarse,
        num_groups,
    )
    reverse_codebook = load_reverse_codebook(args.reverse_codebook_dir, num_groups)
    dynamic_codebook = load_dynamic_hierarchical_codebook(
        args.dynamic_hierarchical_codebook_dir,
        num_groups,
    )
    if dynamic_codebook is not None:
        num_groups = int(dynamic_codebook["group_coarse_ids"].numel())
        if top_group_ids.numel() and int(top_group_ids.max()) >= num_groups:
            raise ValueError("Assignments reference group ids outside dynamic hierarchical codebook.")
    reverse_group_features = None
    reverse_group_prior = None
    if reverse_codebook is not None:
        reverse_group_features = make_reverse_group_features(reverse_codebook, args.reverse_residual_weight)
        reverse_group_prior = make_reverse_group_prior(
            reverse_codebook,
            args.reverse_group_prior,
            args.reverse_prior_power,
            args.reverse_residual_temperature,
        )
    if not (0.0 <= args.coarse_blend <= 1.0):
        raise ValueError("--coarse_blend must be in [0, 1].")
    if not (0.0 <= args.coarse_min_blend <= args.coarse_blend):
        raise ValueError("--coarse_min_blend must be in [0, --coarse_blend].")
    if args.query_temperature <= 0.0:
        raise ValueError("--query_temperature must be positive.")
    if args.reverse_top_codes < 0:
        raise ValueError("--reverse_top_codes must be non-negative.")
    if not (0.0 <= args.reverse_code_blend <= 1.0):
        raise ValueError("--reverse_code_blend must be in [0, 1].")
    if args.reverse_group_prior != "none" and reverse_codebook is None:
        raise ValueError("--reverse_group_prior requires --reverse_codebook_dir.")
    if args.reverse_prior_power < 0.0:
        raise ValueError("--reverse_prior_power must be non-negative.")
    if args.reverse_residual_temperature <= 0.0:
        raise ValueError("--reverse_residual_temperature must be positive.")
    if dynamic_codebook is not None and (coarse_features is not None or reverse_codebook is not None):
        raise ValueError("--dynamic_hierarchical_codebook_dir cannot be combined with legacy coarse/reverse artifacts.")
    if args.dynamic_fine_temperature <= 0.0:
        raise ValueError("--dynamic_fine_temperature must be positive.")
    if args.dynamic_candidate_prior_power < 0.0:
        raise ValueError("--dynamic_candidate_prior_power must be non-negative.")
    if not (0.0 <= args.dynamic_fine_min_blend <= args.dynamic_fine_max_blend <= 1.0):
        raise ValueError("Dynamic fine blend bounds must satisfy 0 <= min <= max <= 1.")
    if args.dynamic_reverse_top_codes < 0:
        raise ValueError("--dynamic_reverse_top_codes must be non-negative.")
    if args.activation_norm_temperature <= 0.0:
        raise ValueError("--activation_norm_temperature must be positive.")

    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    bg = torch.zeros(3, dtype=torch.float32, device="cuda")
    thresholds = sorted(set(args.thresholds))
    per_category = {}
    vis_count = 0

    with torch.no_grad():
        rgb_cache = {}
        for category_idx, category in enumerate(tqdm(categories, desc="Evaluating categories")):
            reverse_group_valid_mask = None
            effective_dynamic_fine_blend = None
            if dynamic_codebook is not None:
                group_activation, reverse_group_valid_mask, effective_dynamic_fine_blend = (
                    dynamic_hierarchical_group_activation(dynamic_codebook, clip_model, category_idx, args)
                )
            elif reverse_codebook is None:
                group_activation = clip_model.get_activation(group_features, category_idx)
            else:
                code_activation = clip_model.get_activation(reverse_codebook["codebook"], category_idx)
                reverse_group_valid_mask = reverse_group_mask(
                    code_activation,
                    reverse_codebook["group_to_code"],
                    args.reverse_top_codes,
                )
                group_activation = clip_model.get_activation(reverse_group_features, category_idx)
                if args.reverse_code_blend > 0.0:
                    mounted_code_activation = code_activation[reverse_codebook["group_to_code"]]
                    group_activation = (
                        (1.0 - args.reverse_code_blend) * group_activation
                        + args.reverse_code_blend * mounted_code_activation
                    )
            group_activation = normalize_token_activation(
                group_activation,
                args.activation_normalization,
                args.activation_norm_low,
                args.activation_norm_high,
                args.activation_norm_temperature,
                reverse_group_valid_mask,
            )
            effective_coarse_blend = 0.0
            activation = point_activation_from_groups(
                group_activation,
                top_group_ids,
                top_group_scores,
                args.aggregation,
                args.score_power,
                args.blend_alpha,
                args.eval_topk,
                args.query_temperature,
                args.query_prior_power,
                reverse_group_valid_mask,
                reverse_group_prior,
            )
            if coarse_features is not None and args.coarse_blend > 0.0:
                if args.coarse_blend_mode == "query_adaptive":
                    effective_coarse_blend = estimate_query_adaptive_coarse_blend(
                        group_activation,
                        args.coarse_blend,
                        args.coarse_min_blend,
                        args.coarse_specificity_topk,
                    )
                else:
                    effective_coarse_blend = args.coarse_blend
                coarse_group_activation = clip_model.get_activation(coarse_features, category_idx)
                coarse_activation = point_activation_from_coarse_groups(
                    coarse_group_activation,
                    group_to_coarse,
                    top_group_ids,
                    top_group_scores,
                    args.aggregation,
                    args.score_power,
                    args.blend_alpha,
                    args.eval_topk,
                    args.query_temperature,
                    args.query_prior_power,
                    reverse_group_valid_mask,
                )
                activation = (1.0 - effective_coarse_blend) * activation + effective_coarse_blend * coarse_activation

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
                "effective_coarse_blend": float(effective_coarse_blend),
                "effective_dynamic_fine_blend": effective_dynamic_fine_blend,
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
        "group_features": os.path.abspath(args.group_features) if args.group_features else None,
        "assignments": os.path.abspath(args.assignments),
        "coarse_features": os.path.abspath(args.coarse_features) if args.coarse_features else None,
        "group_to_coarse": os.path.abspath(args.group_to_coarse) if args.group_to_coarse else None,
        "dynamic_hierarchical_codebook_dir": dynamic_codebook["dir"] if dynamic_codebook else None,
        "dynamic_fine_temperature": float(args.dynamic_fine_temperature),
        "dynamic_candidate_prior_power": float(args.dynamic_candidate_prior_power),
        "dynamic_fine_min_blend": float(args.dynamic_fine_min_blend),
        "dynamic_fine_max_blend": float(args.dynamic_fine_max_blend),
        "dynamic_reverse_top_codes": int(args.dynamic_reverse_top_codes),
        "reverse_codebook_dir": os.path.abspath(args.reverse_codebook_dir) if args.reverse_codebook_dir else None,
        "reverse_top_codes": int(args.reverse_top_codes),
        "reverse_residual_weight": float(args.reverse_residual_weight),
        "reverse_code_blend": float(args.reverse_code_blend),
        "reverse_group_prior": args.reverse_group_prior,
        "reverse_prior_power": float(args.reverse_prior_power),
        "reverse_residual_temperature": float(args.reverse_residual_temperature),
        "coarse_blend": float(args.coarse_blend),
        "coarse_blend_mode": args.coarse_blend_mode,
        "coarse_min_blend": float(args.coarse_min_blend),
        "coarse_specificity_topk": int(args.coarse_specificity_topk),
        "aggregation": args.aggregation,
        "score_power": float(args.score_power),
        "blend_alpha": float(args.blend_alpha),
        "eval_topk": int(args.eval_topk),
        "query_temperature": float(args.query_temperature),
        "query_prior_power": float(args.query_prior_power),
        "score_calibration": args.score_calibration,
        "calibration_low": float(args.calibration_low),
        "calibration_high": float(args.calibration_high),
        "activation_normalization": args.activation_normalization,
        "activation_norm_low": float(args.activation_norm_low),
        "activation_norm_high": float(args.activation_norm_high),
        "activation_norm_temperature": float(args.activation_norm_temperature),
        "num_label_frames": len(labels),
        "num_categories": len(per_category),
        "num_groups": int(num_groups),
        "groups_per_point": int(top_group_ids.shape[1]),
        "thresholds": thresholds,
        "mIoU": float(np.mean(ious)) if ious else 0.0,
        "mAcc@0.25": float(np.mean([iou >= 0.25 for iou in ious])) if ious else 0.0,
        "global_threshold_summary": global_threshold_summary,
        "best_global_threshold": max(global_threshold_summary, key=lambda item: item["mIoU"])
        if global_threshold_summary
        else None,
        "per_category": per_category,
        "note": "This evaluator keeps multiple group-level semantic tokens per Gaussian and computes query activation at evaluation time.",
    }

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
