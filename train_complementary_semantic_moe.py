#!/usr/bin/env python
"""Train a label-free Old/L2/L3 semantic MoE and export its mixed consensus."""

import json
import math
import os
import random
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EXPERT_NAMES = ("old", "l2", "l3")


def normalize(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_consensus(path):
    payload = torch.load(os.path.abspath(path), map_location="cpu")
    required = {
        "initial_features",
        "total_weights",
        "split_initial_features",
        "split_weights",
    }
    missing = required.difference(payload)
    if missing:
        raise ValueError(f"Consensus is missing fields: {sorted(missing)}")
    output = {name: payload[name].detach().cpu().contiguous() for name in required}
    if output["split_initial_features"].shape[:2] != (2, output["initial_features"].shape[0]):
        raise ValueError("Split feature table does not match the consensus")
    if output["split_initial_features"].shape[2] != output["initial_features"].shape[1]:
        raise ValueError("Split and consensus feature dimensions differ")
    if output["split_weights"].shape != output["split_initial_features"].shape[:2]:
        raise ValueError("Split weights do not match split features")
    if output["total_weights"].shape != (output["initial_features"].shape[0],):
        raise ValueError("Total weights do not match consensus features")
    return output


def split_reliability(split_features, split_weights, stability_floor):
    if split_features.ndim != 4 or split_features.shape[1] != 2:
        raise ValueError("Expected split features with shape [E, 2, B, D]")
    if split_weights.shape != split_features.shape[:3]:
        raise ValueError("Split weights must have shape [E, 2, B]")
    supported = (split_weights[:, 0] > 0) & (split_weights[:, 1] > 0)
    cosine = F.cosine_similarity(
        split_features[:, 0].float(), split_features[:, 1].float(), dim=-1
    )
    stability = ((cosine - stability_floor) / (1.0 - stability_floor)).clamp(0.0, 1.0)
    balance = (
        2.0
        * split_weights.min(dim=1).values
        / split_weights.sum(dim=1).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    reliability = stability * balance.sqrt()
    return torch.where(supported, reliability, torch.zeros_like(reliability)), supported


def pairwise_cosines(features, valid):
    if features.ndim != 3 or features.shape[1] != 3:
        raise ValueError("Expected features with shape [B, 3, D]")
    if valid.shape != features.shape[:2]:
        raise ValueError("Expert validity must have shape [B, 3]")
    pairs = ((0, 1), (0, 2), (1, 2))
    values = []
    for first, second in pairs:
        cosine = F.cosine_similarity(features[:, first], features[:, second], dim=-1)
        pair_valid = valid[:, first] & valid[:, second]
        values.append(torch.where(pair_valid, cosine, torch.zeros_like(cosine)))
    return torch.stack(values, dim=1)


def gate_inputs(features, reliability, valid):
    pair_cosine = pairwise_cosines(features, valid)
    pair_valid = torch.stack(
        (
            valid[:, 0] & valid[:, 1],
            valid[:, 0] & valid[:, 2],
            valid[:, 1] & valid[:, 2],
        ),
        dim=1,
    )
    agreement = (pair_cosine * pair_valid.float()).sum(dim=1) / pair_valid.sum(dim=1).clamp_min(1)
    boundary = torch.where(
        pair_valid[:, 2],
        (1.0 - pair_cosine[:, 2]).clamp(0.0, 1.0),
        torch.zeros_like(agreement),
    )
    source_fraction = valid.float().mean(dim=1)
    inputs = torch.cat(
        (
            reliability.transpose(0, 1),
            pair_cosine,
            boundary[:, None],
            agreement[:, None],
            source_fraction[:, None],
        ),
        dim=1,
    )
    return inputs, boundary


def role_prior(reliability, valid, boundary):
    if reliability.shape != valid.transpose(0, 1).shape:
        raise ValueError("Reliability must have shape [3, B]")
    factors = torch.stack(
        (
            1.0 - 0.50 * boundary,
            torch.ones_like(boundary),
            0.35 + 1.65 * boundary,
        ),
        dim=1,
    )
    scores = (reliability.transpose(0, 1) + 0.05) * factors * valid.float()
    scores = scores + 0.02 * valid.float()
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


def masked_softmax(logits, valid):
    if logits.shape != valid.shape:
        raise ValueError("Gate logits and validity must match")
    any_valid = valid.any(dim=1)
    masked = torch.where(valid, logits, torch.full_like(logits, -torch.inf))
    safe = torch.where(any_valid[:, None], masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=1)
    return torch.where(valid & any_valid[:, None], weights, torch.zeros_like(weights))


class ComplementarySemanticMoE(nn.Module):
    def __init__(self, feature_dim, rank=16, hidden_dim=32, adapter_scale=0.10):
        super().__init__()
        self.feature_dim = int(feature_dim)
        self.rank = int(rank)
        self.adapter_scale = float(adapter_scale)
        self.down = nn.ModuleList(
            nn.Linear(feature_dim, rank, bias=False) for _ in EXPERT_NAMES
        )
        self.up = nn.ModuleList(
            nn.Linear(rank, feature_dim, bias=False) for _ in EXPERT_NAMES
        )
        self.gate = nn.Sequential(
            nn.Linear(9, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(EXPERT_NAMES)),
        )
        for down, up in zip(self.down, self.up):
            nn.init.normal_(down.weight, mean=0.0, std=0.02)
            nn.init.zeros_(up.weight)
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def adapt(self, features):
        if features.ndim != 3 or features.shape[1] != len(EXPERT_NAMES):
            raise ValueError("Expert features must have shape [B, 3, D]")
        output = []
        for expert_id in range(len(EXPERT_NAMES)):
            value = features[:, expert_id]
            residual = self.up[expert_id](torch.tanh(self.down[expert_id](value)))
            output.append(normalize(value + self.adapter_scale * residual))
        return torch.stack(output, dim=1)

    def route(self, features, reliability, valid):
        inputs, boundary = gate_inputs(features, reliability, valid)
        prior = role_prior(reliability, valid, boundary)
        logits = self.gate(inputs) + prior.clamp_min(1e-8).log()
        return masked_softmax(logits, valid), prior, boundary


def mix_experts(features, weights, valid):
    effective = weights * valid.float()
    effective = effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mixed = (effective[:, :, None] * features).sum(dim=1)
    any_valid = valid.any(dim=1)
    mixed = torch.where(any_valid[:, None], normalize(mixed), torch.zeros_like(mixed))
    return mixed, effective


def complementarity_loss(adapted, raw, valid, margin=0.50):
    all_valid = valid.all(dim=1)
    if not all_valid.any():
        return adapted.sum() * 0.0, torch.zeros((), device=adapted.device)
    l2_delta = adapted[:, 1] - adapted[:, 0]
    l3_delta = adapted[:, 2] - adapted[:, 0]
    correlation = F.cosine_similarity(l2_delta, l3_delta, dim=-1).abs()
    disagreement = (
        1.0 - F.cosine_similarity(raw[:, 1], raw[:, 2], dim=-1)
    ).clamp(0.0, 1.0).detach()
    weight = disagreement * all_valid.float()
    loss = (F.relu(correlation - margin).square() * weight).sum() / weight.sum().clamp_min(1e-8)
    mean_correlation = (correlation * all_valid.float()).sum() / all_valid.float().sum().clamp_min(1.0)
    return loss, mean_correlation.detach()


def gather_batch(payloads, indices, device, stability_floor):
    raw = torch.stack(
        [payload["initial_features"][indices].float() for payload in payloads], dim=1
    ).to(device, non_blocking=True)
    raw = normalize(raw)
    split = torch.stack(
        [payload["split_initial_features"][:, indices].float() for payload in payloads],
        dim=0,
    ).to(device, non_blocking=True)
    split = normalize(split)
    split_weights = torch.stack(
        [payload["split_weights"][:, indices].float() for payload in payloads], dim=0
    ).to(device, non_blocking=True)
    total_weights = torch.stack(
        [payload["total_weights"][indices].float() for payload in payloads], dim=0
    ).to(device, non_blocking=True)
    valid = total_weights.transpose(0, 1) > 0
    reliability, split_supported = split_reliability(split, split_weights, stability_floor)
    routing_reliability = torch.where(
        split_supported,
        reliability,
        0.05 * valid.transpose(0, 1).float(),
    )
    return raw, split, split_weights, valid, routing_reliability, split_supported


def compute_losses(model, batch, args):
    raw, split, split_weights, valid, reliability, split_supported = batch
    weights, prior, _ = model.route(raw, reliability, valid)
    adapted = model.adapt(raw)
    mixed, _ = mix_experts(adapted, weights, valid)

    split_adapted = []
    split_mixed = []
    split_valid = []
    for split_id in range(2):
        current = model.adapt(split[:, split_id].transpose(0, 1))
        current_valid = split_weights[:, split_id].transpose(0, 1) > 0
        current_mixed, _ = mix_experts(current, weights, current_valid)
        split_adapted.append(current)
        split_mixed.append(current_mixed)
        split_valid.append(current_valid.any(dim=1))
    cross_valid = split_valid[0] & split_valid[1]
    consistency = (
        1.0 - F.cosine_similarity(split_mixed[0][cross_valid], split_mixed[1][cross_valid], dim=-1)
    ).mean()

    teacher, _ = mix_experts(raw, prior, valid)
    mixture_valid = valid.any(dim=1)
    fidelity = (
        1.0 - F.cosine_similarity(mixed[mixture_valid], teacher[mixture_valid], dim=-1)
    ).mean()

    expert_cosine = F.cosine_similarity(
        split_adapted[0], split_adapted[1], dim=-1
    ).transpose(0, 1)
    expert_weight = weights.transpose(0, 1) * split_supported.float()
    expert_consistency = (
        (1.0 - expert_cosine) * expert_weight
    ).sum() / expert_weight.sum().clamp_min(1e-8)

    complementary, residual_correlation = complementarity_loss(
        adapted, raw, valid, args.complement_margin
    )
    entropy = -(weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()).sum(dim=1)
    normalized_entropy = entropy / math.log(len(EXPERT_NAMES))
    entropy_floor = F.relu(args.minimum_gate_entropy - normalized_entropy).square()
    entropy_loss = entropy_floor[mixture_valid].mean()

    role_loss = (
        prior
        * (prior.clamp_min(1e-8).log() - weights.clamp_min(1e-8).log())
    ).sum(dim=1)[mixture_valid].mean()
    mean_weight = weights[mixture_valid].mean(dim=0)
    mean_prior = prior[mixture_valid].mean(dim=0).detach()
    balance_loss = (mean_weight - mean_prior).square().sum()
    adapter_drift = (
        1.0 - F.cosine_similarity(adapted[valid], raw[valid], dim=-1)
    ).mean()

    components = {
        "consistency": consistency,
        "fidelity": fidelity,
        "expert_consistency": expert_consistency,
        "complementarity": complementary,
        "role": role_loss,
        "entropy": entropy_loss,
        "balance": balance_loss,
        "adapter_drift": adapter_drift,
    }
    total = (
        args.consistency_weight * consistency
        + args.fidelity_weight * fidelity
        + args.expert_consistency_weight * expert_consistency
        + args.complementarity_weight * complementary
        + args.role_weight * role_loss
        + args.entropy_weight * entropy_loss
        + args.balance_weight * balance_loss
        + args.adapter_weight * adapter_drift
    )
    diagnostics = {
        "normalized_entropy": normalized_entropy[mixture_valid].mean().detach(),
        "residual_correlation": residual_correlation,
        "mean_weights": mean_weight.detach(),
    }
    return total, components, diagnostics


def training_indices(payloads, stability_floor):
    split_weights = torch.stack([payload["split_weights"] for payload in payloads], dim=0)
    total_weights = torch.stack([payload["total_weights"] for payload in payloads], dim=0)
    split_supported = (split_weights[:, 0] > 0) & (split_weights[:, 1] > 0)
    both_views = (split_weights[:, 0] > 0).any(dim=0) & (split_weights[:, 1] > 0).any(dim=0)
    valid_experts = (total_weights > 0).sum(dim=0)
    eligible = both_views & (valid_experts >= 2) & (split_supported.sum(dim=0) >= 1)
    return torch.nonzero(eligible, as_tuple=False).squeeze(1)


def train(model, payloads, indices, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 101)
    history = []
    model.train()
    for step in range(args.steps):
        positions = torch.randint(
            0, indices.numel(), (args.batch_size,), generator=generator
        )
        batch_indices = indices[positions]
        batch = gather_batch(payloads, batch_indices, device, args.stability_floor)
        optimizer.zero_grad(set_to_none=True)
        loss, components, diagnostics = compute_losses(model, batch, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_interval == 0 or step + 1 == args.steps:
            row = {
                "step": step,
                "total": float(loss.detach().cpu()),
                **{name: float(value.detach().cpu()) for name, value in components.items()},
                "normalized_entropy": float(diagnostics["normalized_entropy"].cpu()),
                "residual_correlation": float(diagnostics["residual_correlation"].cpu()),
                "mean_weights": [float(value) for value in diagnostics["mean_weights"].cpu()],
            }
            history.append(row)
            print(json.dumps(row), flush=True)
    return history


@torch.no_grad()
def export_consensus(model, payloads, output_path, args, device):
    model.eval()
    count, feature_dim = payloads[0]["initial_features"].shape
    output_features = torch.zeros((count, feature_dim), dtype=torch.float16)
    output_splits = torch.zeros((2, count, feature_dim), dtype=torch.float16)
    output_total_weights = torch.zeros(count, dtype=torch.float32)
    output_split_weights = torch.zeros((2, count), dtype=torch.float32)
    gate_table = np.zeros((count, len(EXPERT_NAMES)), dtype=np.float16)
    reliability_table = np.zeros((count, len(EXPERT_NAMES)), dtype=np.float16)
    valid_table = np.zeros((count, len(EXPERT_NAMES)), dtype=np.bool_)

    weight_sum = torch.zeros(len(EXPERT_NAMES), dtype=torch.float64)
    active_sum = torch.zeros(len(EXPERT_NAMES), dtype=torch.int64)
    dominant = torch.zeros(len(EXPERT_NAMES), dtype=torch.int64)
    entropy_sum = 0.0
    split_cosine_sum = 0.0
    split_count = 0
    valid_count = 0
    residual_correlation_sum = 0.0
    residual_correlation_count = 0

    for start in range(0, count, args.export_chunk):
        end = min(start + args.export_chunk, count)
        indices = torch.arange(start, end)
        raw, split, split_weights, valid, reliability, _ = gather_batch(
            payloads, indices, device, args.stability_floor
        )
        weights, _, _ = model.route(raw, reliability, valid)
        adapted = model.adapt(raw)
        mixed, _ = mix_experts(adapted, weights, valid)
        any_valid = valid.any(dim=1)

        mixed_splits = []
        mixed_split_weights = []
        for split_id in range(2):
            current = model.adapt(split[:, split_id].transpose(0, 1))
            current_valid = split_weights[:, split_id].transpose(0, 1) > 0
            current_mixed, current_gate = mix_experts(current, weights, current_valid)
            support = (
                current_gate * split_weights[:, split_id].transpose(0, 1)
            ).sum(dim=1)
            mixed_splits.append(current_mixed)
            mixed_split_weights.append(support)

        output_features[start:end] = mixed.cpu().to(torch.float16)
        output_total_weights[start:end] = any_valid.float().cpu()
        for split_id in range(2):
            output_splits[split_id, start:end] = mixed_splits[split_id].cpu().to(torch.float16)
            output_split_weights[split_id, start:end] = mixed_split_weights[split_id].cpu()
        gate_table[start:end] = weights.cpu().numpy().astype(np.float16)
        reliability_table[start:end] = reliability.transpose(0, 1).cpu().numpy().astype(np.float16)
        valid_table[start:end] = valid.cpu().numpy()

        selected = weights.argmax(dim=1)
        valid_weights = weights[any_valid]
        weight_sum += valid_weights.double().sum(dim=0).cpu()
        active_sum += (valid_weights > 0.10).sum(dim=0).cpu()
        dominant += torch.bincount(selected[any_valid].cpu(), minlength=3)
        entropy = -(valid_weights.clamp_min(1e-8) * valid_weights.clamp_min(1e-8).log()).sum(dim=1)
        entropy_sum += float((entropy / math.log(3)).sum().cpu())
        valid_count += int(any_valid.sum())
        cross_valid = (mixed_split_weights[0] > 0) & (mixed_split_weights[1] > 0)
        if cross_valid.any():
            split_cosine_sum += float(
                F.cosine_similarity(
                    mixed_splits[0][cross_valid], mixed_splits[1][cross_valid], dim=-1
                ).sum().cpu()
            )
            split_count += int(cross_valid.sum())
        all_valid = valid.all(dim=1)
        if all_valid.any():
            l2_delta = adapted[:, 1] - adapted[:, 0]
            l3_delta = adapted[:, 2] - adapted[:, 0]
            corr = F.cosine_similarity(l2_delta, l3_delta, dim=-1).abs()
            residual_correlation_sum += float(corr[all_valid].sum().cpu())
            residual_correlation_count += int(all_valid.sum())

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(
        {
            "initial_features": output_features,
            "total_weights": output_total_weights,
            "split_initial_features": output_splits,
            "split_weights": output_split_weights,
        },
        output_path,
    )
    artifact_dir = os.path.dirname(output_path)
    np.save(os.path.join(artifact_dir, "expert_weights.npy"), gate_table)
    np.save(os.path.join(artifact_dir, "expert_reliability.npy"), reliability_table)
    np.save(os.path.join(artifact_dir, "expert_valid.npy"), valid_table)
    return {
        "num_gaussians": count,
        "feature_dim": feature_dim,
        "valid_fraction": valid_count / max(1, count),
        "mean_expert_weights": [float(value / max(1, valid_count)) for value in weight_sum],
        "fraction_weight_above_0.10": [float(value / max(1, valid_count)) for value in active_sum],
        "dominant_expert_fraction": [float(value / max(1, valid_count)) for value in dominant],
        "mean_normalized_gate_entropy": entropy_sum / max(1, valid_count),
        "mean_mixed_split_cosine": split_cosine_sum / max(1, split_count),
        "mean_l2_l3_residual_abs_cosine": residual_correlation_sum
        / max(1, residual_correlation_count),
        "cross_split_supported_gaussians": split_count,
    }


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old_consensus", required=True)
    parser.add_argument("--l2_consensus", required=True)
    parser.add_argument("--l3_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--adapter_scale", type=float, default=0.10)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=2e-3)
    parser.add_argument("--stability_floor", type=float, default=0.50)
    parser.add_argument("--minimum_gate_entropy", type=float, default=0.55)
    parser.add_argument("--complement_margin", type=float, default=0.50)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--fidelity_weight", type=float, default=0.35)
    parser.add_argument("--expert_consistency_weight", type=float, default=0.15)
    parser.add_argument("--complementarity_weight", type=float, default=0.08)
    parser.add_argument("--role_weight", type=float, default=0.10)
    parser.add_argument("--entropy_weight", type=float, default=0.05)
    parser.add_argument("--balance_weight", type=float, default=0.02)
    parser.add_argument("--adapter_weight", type=float, default=0.05)
    parser.add_argument("--export_chunk", type=int, default=8192)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed < 0 or args.steps <= 0 or args.batch_size <= 0 or args.export_chunk <= 0:
        raise ValueError("Seed, steps, batch size, and export chunk must be valid")
    if not 0.0 <= args.stability_floor < 1.0:
        raise ValueError("Stability floor must be in [0, 1)")
    if not 0.0 <= args.minimum_gate_entropy <= 1.0:
        raise ValueError("Minimum gate entropy must be in [0, 1]")
    if not 0.0 <= args.complement_margin <= 1.0:
        raise ValueError("Complement margin must be in [0, 1]")
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        print(f"Reuse complementary semantic MoE: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_paths = [args.old_consensus, args.l2_consensus, args.l3_consensus]
    payloads = [load_consensus(path) for path in source_paths]
    shapes = [tuple(payload["initial_features"].shape) for payload in payloads]
    if len(set(shapes)) != 1:
        raise ValueError(f"Expert consensus shapes differ: {shapes}")
    indices = training_indices(payloads, args.stability_floor)
    if indices.numel() < args.batch_size:
        raise ValueError("Not enough multi-expert two-view samples to train the gate")

    model = ComplementarySemanticMoE(
        shapes[0][1], args.rank, args.hidden_dim, args.adapter_scale
    ).to(device)
    started = time.time()
    history = train(model, payloads, indices, args, device)
    checkpoint_path = os.path.join(output_dir, "moe_checkpoint.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "feature_dim": shapes[0][1],
            "expert_names": EXPERT_NAMES,
            "args": vars(args),
        },
        checkpoint_path,
    )
    consensus_path = os.path.join(output_dir, "consensus.pt")
    diagnostics = export_consensus(model, payloads, consensus_path, args, device)
    manifest = {
        "format_version": 1,
        "method": "complementary_three_expert_semantic_moe",
        "representation": "continuous_moe_consensus",
        "expert_names": list(EXPERT_NAMES),
        "sources": {name: os.path.abspath(path) for name, path in zip(EXPERT_NAMES, source_paths)},
        "gate": {
            "type": "softmax_continuous_multi_expert",
            "uses_evaluation_queries": False,
            "uses_ground_truth": False,
            "inputs": [
                "split_reliability_old_l2_l3",
                "pairwise_semantic_cosines",
                "l2_l3_disagreement",
                "mean_expert_agreement",
                "source_coverage",
            ],
        },
        "training_objectives": {
            "cross_view_mixture_consistency": args.consistency_weight,
            "semantic_fidelity": args.fidelity_weight,
            "per_expert_cross_view_consistency": args.expert_consistency_weight,
            "l2_l3_residual_complementarity": args.complementarity_weight,
            "scale_role_prior": args.role_weight,
            "minimum_gate_entropy": args.entropy_weight,
            "load_balance": args.balance_weight,
            "adapter_drift": args.adapter_weight,
        },
        "training_samples": int(indices.numel()),
        "history": history,
        "diagnostics": diagnostics,
        "artifacts": {
            "consensus": "consensus.pt",
            "checkpoint": "moe_checkpoint.pt",
            "expert_weights": "expert_weights.npy",
            "expert_reliability": "expert_reliability.npy",
            "expert_valid": "expert_valid.npy",
        },
        "seed": args.seed,
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
