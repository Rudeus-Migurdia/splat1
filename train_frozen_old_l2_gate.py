#!/usr/bin/env python
"""Train a conservative label-free gate over frozen Old and L2 experts."""

import json
import math
import os
import random
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


EXPERT_NAMES = ("old", "l2")


def normalize(value, eps=1e-8):
    return value / value.norm(dim=-1, keepdim=True).clamp_min(eps)


def set_deterministic_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


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
    count, feature_dim = output["initial_features"].shape
    if output["split_initial_features"].shape != (2, count, feature_dim):
        raise ValueError("Split feature table does not match the consensus")
    if output["split_weights"].shape != (2, count):
        raise ValueError("Split weights do not match split features")
    if output["total_weights"].shape != (count,):
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
        2.0 * split_weights.min(dim=1).values
        / split_weights.sum(dim=1).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    reliability = stability * balance.sqrt()
    return torch.where(supported, reliability, torch.zeros_like(reliability)), supported


def masked_softmax(logits, valid):
    if logits.shape != valid.shape:
        raise ValueError("Gate logits and validity must match")
    any_valid = valid.any(dim=1)
    masked = torch.where(valid, logits, torch.full_like(logits, -torch.inf))
    safe = torch.where(any_valid[:, None], masked, torch.zeros_like(masked))
    weights = torch.softmax(safe, dim=1)
    return torch.where(valid & any_valid[:, None], weights, torch.zeros_like(weights))


def reliability_prior(reliability, valid):
    if reliability.shape != valid.shape:
        raise ValueError("Reliability and expert validity must match")
    scores = (reliability + 0.05) * valid.float()
    return scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-8)


def gate_inputs(features, reliability, valid):
    if features.ndim != 3 or features.shape[1] != 2:
        raise ValueError("Expected frozen expert features with shape [B, 2, D]")
    cosine = F.cosine_similarity(features[:, 0], features[:, 1], dim=-1)
    cosine = torch.where(valid.all(dim=1), cosine, torch.zeros_like(cosine))
    return torch.cat((reliability, cosine[:, None], valid.float()), dim=1)


class FrozenOldL2Gate(nn.Module):
    def __init__(self, hidden_dim=16, max_logit_delta=0.50):
        super().__init__()
        self.max_logit_delta = float(max_logit_delta)
        self.network = nn.Sequential(
            nn.Linear(5, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, len(EXPERT_NAMES)),
        )
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, features, reliability, valid):
        prior = reliability_prior(reliability, valid)
        delta = self.max_logit_delta * torch.tanh(
            self.network(gate_inputs(features, reliability, valid))
        )
        logits = prior.clamp_min(1e-8).log() + delta
        return masked_softmax(logits, valid), prior


def mix_experts(features, weights, valid):
    effective = weights * valid.float()
    effective = effective / effective.sum(dim=1, keepdim=True).clamp_min(1e-8)
    mixed = (effective[:, :, None] * features).sum(dim=1)
    any_valid = valid.any(dim=1)
    mixed = torch.where(any_valid[:, None], normalize(mixed), torch.zeros_like(mixed))
    return mixed, effective


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
        split_supported, reliability, 0.05 * valid.transpose(0, 1).float()
    ).transpose(0, 1)
    return raw, split, split_weights, valid, routing_reliability, split_supported


def compute_losses(model, batch, args):
    raw, split, split_weights, valid, reliability, _ = batch
    weights, prior = model(raw, reliability, valid)
    mixed, _ = mix_experts(raw, weights, valid)
    teacher, _ = mix_experts(raw, prior, valid)

    split_mixed = []
    split_valid = []
    for split_id in range(2):
        current = split[:, split_id].transpose(0, 1)
        current_valid = split_weights[:, split_id].transpose(0, 1) > 0
        current_mixed, _ = mix_experts(current, weights, current_valid)
        split_mixed.append(current_mixed)
        split_valid.append(current_valid.any(dim=1))
    cross_valid = split_valid[0] & split_valid[1]
    if cross_valid.any():
        consistency = (
            1.0
            - F.cosine_similarity(
                split_mixed[0][cross_valid], split_mixed[1][cross_valid], dim=-1
            )
        ).mean()
    else:
        consistency = mixed.sum() * 0.0

    mixture_valid = valid.any(dim=1)
    fidelity = (
        1.0 - F.cosine_similarity(mixed[mixture_valid], teacher[mixture_valid], dim=-1)
    ).mean()
    role = (
        prior
        * (prior.clamp_min(1e-8).log() - weights.clamp_min(1e-8).log())
    ).sum(dim=1)[mixture_valid].mean()
    mean_weight = weights[mixture_valid].mean(dim=0)
    mean_prior = prior[mixture_valid].mean(dim=0).detach()
    balance = (mean_weight - mean_prior).square().sum()
    total = (
        args.consistency_weight * consistency
        + args.fidelity_weight * fidelity
        + args.role_weight * role
        + args.balance_weight * balance
    )
    entropy = -(
        weights[mixture_valid].clamp_min(1e-8)
        * weights[mixture_valid].clamp_min(1e-8).log()
    ).sum(dim=1) / math.log(2.0)
    diagnostics = {
        "consistency": consistency,
        "fidelity": fidelity,
        "role": role,
        "balance": balance,
        "normalized_entropy": entropy.mean(),
        "mean_weights": mean_weight,
        "mean_prior": mean_prior,
    }
    return total, diagnostics


def training_indices(payloads):
    split_weights = torch.stack([payload["split_weights"] for payload in payloads], dim=0)
    total_weights = torch.stack([payload["total_weights"] for payload in payloads], dim=0)
    both_views = (split_weights[:, 0] > 0).any(dim=0) & (split_weights[:, 1] > 0).any(dim=0)
    eligible = both_views & (total_weights > 0).any(dim=0)
    return torch.nonzero(eligible, as_tuple=False).squeeze(1)


def train(model, payloads, indices, args, device):
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed + 101)
    history = []
    model.train()
    for step in range(args.steps):
        positions = torch.randint(0, indices.numel(), (args.batch_size,), generator=generator)
        batch = gather_batch(
            payloads, indices[positions], device, args.stability_floor
        )
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = compute_losses(model, batch, args)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.log_interval == 0 or step + 1 == args.steps:
            row = {
                "step": step,
                "total": float(loss.detach().cpu()),
                **{
                    name: float(value.detach().cpu())
                    for name, value in diagnostics.items()
                    if value.ndim == 0
                },
                "mean_weights": [float(value) for value in diagnostics["mean_weights"].cpu()],
                "mean_prior": [float(value) for value in diagnostics["mean_prior"].cpu()],
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
    reliability_table = np.zeros_like(gate_table)
    valid_table = np.zeros((count, len(EXPERT_NAMES)), dtype=np.bool_)
    weight_sum = torch.zeros(len(EXPERT_NAMES), dtype=torch.float64)
    dominant = torch.zeros(len(EXPERT_NAMES), dtype=torch.int64)
    entropy_sum = 0.0
    valid_count = 0
    split_cosine_sum = 0.0
    split_count = 0

    for start in range(0, count, args.export_chunk):
        end = min(start + args.export_chunk, count)
        indices = torch.arange(start, end)
        raw, split, split_weights, valid, reliability, _ = gather_batch(
            payloads, indices, device, args.stability_floor
        )
        weights, _ = model(raw, reliability, valid)
        mixed, _ = mix_experts(raw, weights, valid)
        any_valid = valid.any(dim=1)
        mixed_splits = []
        mixed_split_weights = []
        for split_id in range(2):
            current = split[:, split_id].transpose(0, 1)
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
        reliability_table[start:end] = reliability.cpu().numpy().astype(np.float16)
        valid_table[start:end] = valid.cpu().numpy()

        valid_weights = weights[any_valid]
        weight_sum += valid_weights.double().sum(dim=0).cpu()
        dominant += torch.bincount(weights[any_valid].argmax(dim=1).cpu(), minlength=2)
        entropy = -(
            valid_weights.clamp_min(1e-8) * valid_weights.clamp_min(1e-8).log()
        ).sum(dim=1) / math.log(2.0)
        entropy_sum += float(entropy.sum().cpu())
        valid_count += int(any_valid.sum())
        cross_valid = (mixed_split_weights[0] > 0) & (mixed_split_weights[1] > 0)
        if cross_valid.any():
            split_cosine_sum += float(
                F.cosine_similarity(
                    mixed_splits[0][cross_valid], mixed_splits[1][cross_valid], dim=-1
                ).sum().cpu()
            )
            split_count += int(cross_valid.sum())

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
        "dominant_expert_fraction": [float(value / max(1, valid_count)) for value in dominant],
        "mean_normalized_gate_entropy": entropy_sum / max(1, valid_count),
        "mean_mixed_split_cosine": split_cosine_sum / max(1, split_count),
        "cross_split_supported_gaussians": split_count,
    }


def parse_args():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--old_consensus", required=True)
    parser.add_argument("--l2_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260717)
    parser.add_argument("--hidden_dim", type=int, default=16)
    parser.add_argument("--max_logit_delta", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=0.002)
    parser.add_argument("--stability_floor", type=float, default=0.50)
    parser.add_argument("--consistency_weight", type=float, default=1.0)
    parser.add_argument("--fidelity_weight", type=float, default=0.50)
    parser.add_argument("--role_weight", type=float, default=0.25)
    parser.add_argument("--balance_weight", type=float, default=0.05)
    parser.add_argument("--export_chunk", type=int, default=8192)
    parser.add_argument("--log_interval", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed < 0 or args.steps <= 0 or args.batch_size <= 0 or args.export_chunk <= 0:
        raise ValueError("Seed, steps, batch size, and export chunk must be valid")
    if not 0.0 <= args.stability_floor < 1.0:
        raise ValueError("Stability floor must be in [0, 1)")
    if not 0.0 <= args.max_logit_delta <= 2.0:
        raise ValueError("Maximum logit delta must be in [0, 2]")
    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path):
        print(f"Reuse frozen Old/L2 gate: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    set_deterministic_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    source_paths = [args.old_consensus, args.l2_consensus]
    payloads = [load_consensus(path) for path in source_paths]
    shapes = [tuple(payload["initial_features"].shape) for payload in payloads]
    if len(set(shapes)) != 1:
        raise ValueError(f"Expert consensus shapes differ: {shapes}")
    indices = training_indices(payloads)
    if indices.numel() == 0:
        raise ValueError("No two-view samples are available to train the gate")

    model = FrozenOldL2Gate(args.hidden_dim, args.max_logit_delta).to(device)
    started = time.time()
    history = train(model, payloads, indices, args, device)
    checkpoint_path = os.path.join(output_dir, "gate_checkpoint.pt")
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
        "method": "frozen_old_l2_conservative_gate",
        "representation": "continuous_frozen_expert_consensus",
        "expert_names": list(EXPERT_NAMES),
        "sources": {name: os.path.abspath(path) for name, path in zip(EXPERT_NAMES, source_paths)},
        "experts_frozen": True,
        "uses_evaluation_queries": False,
        "uses_ground_truth": False,
        "gate": {
            "type": "bounded_residual_over_split_reliability_prior",
            "inputs": [
                "old_split_reliability",
                "l2_split_reliability",
                "old_l2_cosine",
                "old_valid",
                "l2_valid",
            ],
            "max_logit_delta": args.max_logit_delta,
        },
        "training_objectives": {
            "cross_view_mixture_consistency": args.consistency_weight,
            "semantic_fidelity": args.fidelity_weight,
            "reliability_prior_kl": args.role_weight,
            "batch_balance_to_prior": args.balance_weight,
        },
        "training_samples": int(indices.numel()),
        "history": history,
        "diagnostics": diagnostics,
        "artifacts": {
            "consensus": "consensus.pt",
            "checkpoint": "gate_checkpoint.pt",
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
