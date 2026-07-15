#!/usr/bin/env python
"""Fit a compact, A6-anchored semantic residual from cached 2D observations."""

import json
import os
import random
import sys
from argparse import ArgumentParser

import torch
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm


def l2_normalize(value):
    return F.normalize(value, dim=-1, eps=1e-6)


def weighted_cosine_loss(prediction, target, weight=None):
    loss = 1.0 - F.cosine_similarity(prediction, target, dim=-1)
    if weight is None:
        return loss.mean()
    weight = weight.float().clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-6)


def split_agreement_confidence(observation, split_target, valid, floor):
    cosine = F.cosine_similarity(observation, split_target, dim=-1)
    confidence = ((cosine - floor) / max(1e-6, 1.0 - floor)).clamp(0.0, 1.0)
    return torch.where(valid, confidence, torch.zeros_like(confidence)), cosine


class A6LowRankSemanticField(nn.Module):
    def __init__(self, base_features, valid_mask, rank, train_semantic_opacity):
        super().__init__()
        if base_features.ndim != 2:
            raise ValueError("base_features must have shape [N, D]")
        if valid_mask.shape != (base_features.shape[0],):
            raise ValueError("valid_mask must match base_features")
        if rank <= 0:
            raise ValueError("rank must be positive")
        self.register_buffer("base_features", base_features, persistent=False)
        self.register_buffer("valid_mask", valid_mask.bool(), persistent=False)
        self.residual_codes = nn.Embedding(base_features.shape[0], rank, sparse=True)
        self.residual_basis = nn.Parameter(torch.empty(rank, base_features.shape[1]))
        nn.init.zeros_(self.residual_codes.weight)
        nn.init.normal_(self.residual_basis, mean=0.0, std=0.01)
        self.opacity_log_scale = None
        if train_semantic_opacity:
            self.opacity_log_scale = nn.Embedding(base_features.shape[0], 1, sparse=True)
            nn.init.zeros_(self.opacity_log_scale.weight)

    def point_features(self, point_ids):
        base = self.base_features[point_ids].float()
        residual = self.residual_codes(point_ids) @ self.residual_basis
        return l2_normalize(base + residual)

    def point_gate(self, point_ids):
        if self.opacity_log_scale is None:
            return torch.ones_like(point_ids, dtype=torch.float32)
        return self.opacity_log_scale(point_ids).squeeze(-1).clamp(-4.0, 4.0).exp()

    def render(self, point_ids, point_weights):
        valid = point_ids >= 0
        safe_ids = torch.where(valid, point_ids, torch.zeros_like(point_ids))
        valid = valid & self.valid_mask[safe_ids]
        features = self.point_features(safe_ids)
        gates = self.point_gate(safe_ids)
        effective_weights = torch.where(
            valid,
            point_weights.float() * gates,
            torch.zeros_like(point_weights, dtype=torch.float32),
        )
        rendered = (features * effective_weights.unsqueeze(-1)).sum(dim=1)
        return l2_normalize(rendered), effective_weights.sum(dim=1) > 1e-6


def render_split_target(point_ids, point_weights, split_features, split_weights):
    valid = point_ids >= 0
    safe_ids = torch.where(valid, point_ids, torch.zeros_like(point_ids))
    valid = valid & (split_weights[safe_ids] > 0.0)
    features = l2_normalize(split_features[safe_ids].float())
    effective_weights = torch.where(
        valid,
        point_weights.float(),
        torch.zeros_like(point_weights, dtype=torch.float32),
    )
    rendered = (features * effective_weights.unsqueeze(-1)).sum(dim=1)
    return l2_normalize(rendered), effective_weights.sum(dim=1) > 1e-6


def weighted_segment_contrastive_loss(
    prediction,
    segment_ids,
    segment_features,
    weight,
    temperature,
):
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    logits = prediction @ l2_normalize(segment_features).T / temperature
    losses = F.cross_entropy(logits, segment_ids, reduction="none")
    return (losses * weight).sum() / weight.sum().clamp_min(1e-6)


def load_slim_view_caches(cache_dir, manifest, max_views=0):
    entries = manifest["views"]
    if max_views > 0:
        entries = entries[:max_views]
    caches = []
    for entry in tqdm(entries, desc="Loading slim observation caches"):
        source = torch.load(os.path.join(cache_dir, entry["cache"]), map_location="cpu")
        if source["point_ids"].shape[0] == 0:
            continue
        caches.append(
            {
                "view_index": int(source["view_index"]),
                "image_name": str(source["image_name"]),
                "point_ids": source["point_ids"].contiguous(),
                "point_weights": source["point_weights"].contiguous(),
                "segment_ids": source["segment_ids"].contiguous(),
                "feature_latents": source["feature_latents"].contiguous(),
            }
        )
        del source
    if not caches:
        raise ValueError("Observation manifest contains no non-empty views")
    return caches


@torch.no_grad()
def save_consensus(field, support, output_path, metadata, chunk_size=32768):
    num_gaussians, feature_dim = field.base_features.shape
    output = torch.empty((num_gaussians, feature_dim), dtype=torch.float16)
    cosine_sum = 0.0
    valid_count = 0
    for start in tqdm(range(0, num_gaussians, chunk_size), desc="Exporting consensus"):
        end = min(start + chunk_size, num_gaussians)
        point_ids = torch.arange(start, end, device=field.base_features.device)
        features = field.point_features(point_ids)
        valid = field.valid_mask[start:end]
        features = torch.where(valid.unsqueeze(-1), features, torch.zeros_like(features))
        output[start:end].copy_(features.cpu().half())
        if valid.any():
            base = l2_normalize(field.base_features[start:end][valid].float())
            cosine_sum += float(F.cosine_similarity(features[valid], base, dim=-1).sum())
            valid_count += int(valid.sum())

    payload = {
        "initial_features": output,
        "total_weights": support.float().cpu(),
        "metadata": metadata,
    }
    export_metrics = {
        "base_to_trained_cosine": cosine_sum / max(1, valid_count),
        "num_valid_gaussians": valid_count,
    }
    if field.opacity_log_scale is not None:
        raw_gate = field.opacity_log_scale.weight.detach().cpu().float().squeeze(-1).exp()
        valid_cpu = field.valid_mask.detach().cpu()
        raw_p95 = torch.quantile(raw_gate[valid_cpu], 0.95)
        # Preserve the initial unit score scale. The learned gate may suppress
        # uncertain points but cannot globally attenuate calibrated CLIP scores.
        semantic_opacity = raw_gate.clamp(0.0, 1.0)
        semantic_opacity[~valid_cpu] = 0.0
        payload["semantic_opacity"] = semantic_opacity.half()
        valid_opacity = semantic_opacity[valid_cpu]
        export_metrics.update(
            {
                "raw_semantic_opacity_p95": float(raw_p95),
                "semantic_opacity_mean": float(valid_opacity.mean()),
                "semantic_opacity_p10": float(torch.quantile(valid_opacity, 0.1)),
                "semantic_opacity_suppressed_fraction": float(
                    (valid_opacity < 0.9).float().mean()
                ),
            }
        )
    torch.save(payload, output_path)
    return export_metrics


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--split_consensus")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    parser.add_argument("--batch_pixels", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--code_lr", type=float, default=0.02)
    parser.add_argument("--basis_lr", type=float, default=0.002)
    parser.add_argument("--opacity_lr", type=float, default=0.01)
    parser.add_argument("--direct_weight", type=float, default=1.0)
    parser.add_argument("--lovo_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_weight", type=float, default=0.0)
    parser.add_argument("--contrastive_temperature", type=float, default=0.07)
    parser.add_argument("--agreement_floor", type=float, default=0.65)
    parser.add_argument("--direct_confidence_floor", type=float, default=0.25)
    parser.add_argument("--anchor_weight", type=float, default=0.1)
    parser.add_argument("--code_regularization", type=float, default=1e-4)
    parser.add_argument("--opacity_regularization", type=float, default=1e-2)
    parser.add_argument("--train_semantic_opacity", action="store_true")
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_export", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.iterations <= 0 or args.batch_pixels <= 0 or args.topk <= 0:
        raise ValueError("iterations, batch_pixels, and topk must be positive")
    if args.rank <= 0 or args.log_interval <= 0:
        raise ValueError("rank and log_interval must be positive")
    if not -1.0 <= args.agreement_floor < 1.0:
        raise ValueError("agreement_floor must be in [-1, 1)")
    if not 0.0 <= args.direct_confidence_floor <= 1.0:
        raise ValueError("direct_confidence_floor must be in [0, 1]")
    needs_split = args.lovo_weight > 0.0 or args.contrastive_weight > 0.0
    if needs_split and not args.split_consensus:
        raise ValueError("split_consensus is required by LOVO or contrastive training")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")
    output_dir = os.path.abspath(args.output_dir)
    artifact_path = os.path.join(output_dir, "consensus.pt")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if os.path.exists(artifact_path) and not args.force:
        print(f"Reuse existing artifact: {artifact_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    cache_dir = os.path.abspath(args.cache_dir)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        manifest = json.load(source)
    caches = load_slim_view_caches(cache_dir, manifest, args.max_views)

    base_payload = torch.load(os.path.abspath(args.base_consensus), map_location="cpu")
    base_features_cpu = base_payload["initial_features"]
    support = base_payload["total_weights"].float().cpu().contiguous()
    num_gaussians, feature_dim = base_features_cpu.shape
    if int(manifest["num_gaussians"]) != num_gaussians:
        raise ValueError("Cache and A6 base have different Gaussian counts")
    if int(manifest["semantic_dim"]) != feature_dim:
        raise ValueError("Cache and A6 base have different semantic dimensions")
    base_features = base_features_cpu.to(device, dtype=torch.float16)
    del base_payload, base_features_cpu
    valid_mask = support.to(device) > 0.0

    split_features = None
    split_weights = None
    if needs_split:
        split_payload = torch.load(os.path.abspath(args.split_consensus), map_location="cpu")
        split_features = split_payload["split_initial_features"].to(
            device, dtype=torch.float16
        )
        split_weights = split_payload["split_weights"].to(device, dtype=torch.float32)
        del split_payload
        if split_features.shape != (2, num_gaussians, feature_dim):
            raise ValueError("Expected exactly two split feature tables matching A6")

    field = A6LowRankSemanticField(
        base_features,
        valid_mask,
        args.rank,
        args.train_semantic_opacity,
    ).to(device)
    sparse_parameter_groups = [
        {"params": [field.residual_codes.weight], "lr": args.code_lr}
    ]
    if field.opacity_log_scale is not None:
        sparse_parameter_groups.append(
            {"params": [field.opacity_log_scale.weight], "lr": args.opacity_lr}
        )
    sparse_optimizer = torch.optim.SparseAdam(
        sparse_parameter_groups, lr=args.code_lr
    )
    basis_optimizer = torch.optim.AdamW(
        [field.residual_basis], lr=args.basis_lr, weight_decay=0.0
    )

    generator = torch.Generator().manual_seed(args.seed)
    history = []
    running = {
        "loss": 0.0,
        "direct": 0.0,
        "lovo": 0.0,
        "contrastive": 0.0,
        "anchor": 0.0,
        "code_regularization": 0.0,
        "opacity_regularization": 0.0,
        "split_agreement": 0.0,
        "split_valid_fraction": 0.0,
    }

    for iteration in range(1, args.iterations + 1):
        view_slot = int(torch.randint(len(caches), (1,), generator=generator))
        cache = caches[view_slot]
        count = cache["point_ids"].shape[0]
        indices = torch.randint(count, (args.batch_pixels,), generator=generator)
        point_ids = cache["point_ids"][indices, : args.topk].long().to(device)
        point_weights = cache["point_weights"][indices, : args.topk].float().to(device)
        segment_ids = cache["segment_ids"][indices].long().to(device)
        segment_features = cache["feature_latents"].float().to(device)
        targets = l2_normalize(segment_features[segment_ids])

        prediction, prediction_valid = field.render(point_ids, point_weights)
        confidence = torch.ones(args.batch_pixels, device=device)
        split_cosine = torch.zeros_like(confidence)
        split_valid = torch.zeros_like(confidence, dtype=torch.bool)
        split_target = None
        if needs_split:
            opposite_split = 1 - (int(cache["view_index"]) % 2)
            split_target, split_valid = render_split_target(
                point_ids,
                point_weights,
                split_features[opposite_split],
                split_weights[opposite_split],
            )
            confidence, split_cosine = split_agreement_confidence(
                targets,
                split_target,
                split_valid,
                args.agreement_floor,
            )

        direct_confidence = (
            args.direct_confidence_floor
            + (1.0 - args.direct_confidence_floor) * confidence
            if needs_split
            else confidence
        )
        direct_confidence = direct_confidence * prediction_valid.float()
        direct_loss = weighted_cosine_loss(prediction, targets, direct_confidence)
        lovo_loss = torch.zeros((), device=device)
        if args.lovo_weight > 0.0:
            lovo_loss = weighted_cosine_loss(prediction, split_target, confidence)
        contrastive_loss = torch.zeros((), device=device)
        if args.contrastive_weight > 0.0:
            contrastive_loss = weighted_segment_contrastive_loss(
                prediction,
                segment_ids,
                segment_features,
                confidence,
                args.contrastive_temperature,
            )

        valid_ids = point_ids[point_ids >= 0]
        unique_ids = torch.unique(valid_ids)
        unique_features = field.point_features(unique_ids)
        base_unique = l2_normalize(field.base_features[unique_ids].float())
        anchor_loss = weighted_cosine_loss(unique_features, base_unique)
        code_regularization = field.residual_codes(unique_ids).square().mean()
        opacity_regularization = torch.zeros((), device=device)
        if field.opacity_log_scale is not None:
            opacity_regularization = field.opacity_log_scale(unique_ids).square().mean()

        loss = (
            args.direct_weight * direct_loss
            + args.lovo_weight * lovo_loss
            + args.contrastive_weight * contrastive_loss
            + args.anchor_weight * anchor_loss
            + args.code_regularization * code_regularization
            + args.opacity_regularization * opacity_regularization
        )
        sparse_optimizer.zero_grad(set_to_none=True)
        basis_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        sparse_optimizer.step()
        basis_optimizer.step()

        values = {
            "loss": loss,
            "direct": direct_loss,
            "lovo": lovo_loss,
            "contrastive": contrastive_loss,
            "anchor": anchor_loss,
            "code_regularization": code_regularization,
            "opacity_regularization": opacity_regularization,
            "split_agreement": split_cosine[split_valid].mean()
            if split_valid.any()
            else torch.zeros((), device=device),
            "split_valid_fraction": split_valid.float().mean(),
        }
        for key, value in values.items():
            running[key] += float(value.detach())

        if iteration % args.log_interval == 0 or iteration == args.iterations:
            window = args.log_interval if iteration % args.log_interval == 0 else iteration % args.log_interval
            row = {"iteration": iteration}
            row.update({key: value / max(1, window) for key, value in running.items()})
            history.append(row)
            print(json.dumps(row), flush=True)
            running = {key: 0.0 for key in running}

    metadata = {
        "representation": "a6_low_rank_semantic_residual",
        "cache_dir": cache_dir,
        "base_consensus": os.path.abspath(args.base_consensus),
        "split_consensus": os.path.abspath(args.split_consensus)
        if args.split_consensus
        else None,
        "num_gaussians": num_gaussians,
        "feature_dim": feature_dim,
        "num_views": len(caches),
        "arguments": vars(args),
    }
    if args.skip_export:
        metrics = {"metadata": metadata, "history": history, "export": None}
        with open(metrics_path, "w") as output:
            json.dump(metrics, output, indent=2)
        print("Training smoke completed; export skipped by request")
        return
    export_metrics = save_consensus(field, support, artifact_path, metadata)
    metrics = {"metadata": metadata, "history": history, "export": export_metrics}
    with open(metrics_path, "w") as output:
        json.dump(metrics, output, indent=2)
    print(json.dumps(export_metrics, indent=2))
    print(f"Saved semantic residual consensus to {artifact_path}")


if __name__ == "__main__":
    main()
