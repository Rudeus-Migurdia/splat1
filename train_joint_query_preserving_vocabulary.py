#!/usr/bin/env python
"""Refine one fixed-ID vocabulary jointly for two semantic modes."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_gaussian_multilevel_codebook import ConsensusFeatureSource, l2_normalize


class FixedSharedAssignment:
    def __init__(self, artifact_dir):
        self.artifact_dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.artifact_dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != "gaussian_adaptive_shared_codebook":
            raise ValueError("Expected a Gaussian adaptive shared-codebook artifact")
        if self.manifest.get("storage_layout") != "base_plus_sparse_overflow":
            raise ValueError("Expected base_plus_sparse_overflow storage")

        num_gaussians = int(self.manifest["num_gaussians"])
        slots = int(self.manifest["id_slots"])
        invalid_id = int(self.manifest["invalid_id"])
        base_ids = self._load("point_code_ids").astype(np.int64)
        if base_ids.shape != (num_gaussians,):
            raise ValueError("Base code IDs do not match the Gaussian count")
        ids = np.full((num_gaussians, slots), -1, dtype=np.int32)
        weights = np.zeros((num_gaussians, slots), dtype=np.float32)
        base_valid = base_ids != invalid_id
        ids[base_valid, 0] = base_ids[base_valid].astype(np.int32)
        weights[base_valid, 0] = 1.0

        points = self._load("overflow_point_ids").astype(np.int64)
        overflow_slots = self._load("overflow_slots").astype(np.int64)
        overflow_ids = self._load("overflow_code_ids").astype(np.int64)
        overflow_weights = self._load("overflow_weights").astype(np.float32) / 255.0
        if not (
            points.shape
            == overflow_slots.shape
            == overflow_ids.shape
            == overflow_weights.shape
        ):
            raise ValueError("Sparse overflow arrays must have matching shapes")
        if points.size:
            if points.max() >= num_gaussians or overflow_slots.max() >= slots:
                raise ValueError("Sparse overflow index is outside the artifact")
            ids[points, overflow_slots] = overflow_ids.astype(np.int32)
            weights[points, overflow_slots] = overflow_weights

        self.ids = ids
        self.weights = weights
        self.valid_mask = self._load("valid_mask").astype(bool)
        if self.valid_mask.shape != (num_gaussians,):
            raise ValueError("Artifact valid mask does not match the Gaussian count")
        self.num_gaussians = num_gaussians
        self.feature_dim = int(self.manifest["feature_dim"])
        self.num_codes = int(self.manifest["num_codes"])

    def _load(self, key):
        return np.load(os.path.join(self.artifact_dir, self.manifest[key]))

    def batch(self, indices, device):
        ids = torch.from_numpy(self.ids[indices].astype(np.int64, copy=False)).to(device)
        weights = torch.from_numpy(
            self.weights[indices].astype(np.float32, copy=False)
        ).to(device)
        return ids, weights


def reconstruct_fixed_assignment(
    codebook, point_ids, point_weights, sparse_grad=False
):
    valid = point_ids >= 0
    selected = F.embedding(
        point_ids.clamp_min(0), codebook, sparse=sparse_grad
    )
    reconstruction = (
        selected * point_weights.unsqueeze(-1) * valid.unsqueeze(-1)
    ).sum(dim=1)
    return F.normalize(reconstruction, dim=-1)


def query_rank_losses(prediction, target, query_bank, temperature):
    prediction_logits = prediction @ query_bank.T / temperature
    with torch.no_grad():
        target_logits = target @ query_bank.T / temperature
        target_probabilities = torch.softmax(target_logits, dim=-1)
        top2 = target_logits.topk(k=2, dim=-1)
        top_ids = top2.indices
        target_margin = top2.values[:, 0] - top2.values[:, 1]
        confidence = (
            target_probabilities.gather(1, top_ids[:, :1]).squeeze(1)
            - target_probabilities.gather(1, top_ids[:, 1:2]).squeeze(1)
        ).clamp_min(1e-4)
    kl = F.kl_div(
        torch.log_softmax(prediction_logits, dim=-1),
        target_probabilities,
        reduction="batchmean",
    )
    prediction_top = prediction_logits.gather(1, top_ids)
    prediction_margin = prediction_top[:, 0] - prediction_top[:, 1]
    margin_error = F.smooth_l1_loss(
        prediction_margin,
        target_margin,
        reduction="none",
    )
    margin = (margin_error * confidence).sum() / confidence.sum().clamp_min(1e-8)
    return kl, margin


@torch.no_grad()
def evaluate_mode(
    assignment,
    source,
    codebook,
    query_bank,
    valid_indices,
    sample_count,
    batch_size,
    temperature,
    seed,
):
    rng = np.random.default_rng(seed)
    sample = rng.choice(
        valid_indices,
        min(sample_count, valid_indices.size),
        replace=False,
    )
    totals = {
        "cosine": 0.0,
        "query_kl": 0.0,
        "margin_mae": 0.0,
        "top1_agreement": 0.0,
    }
    count = 0
    device = codebook.device
    for start in range(0, sample.size, batch_size):
        indices = sample[start : start + batch_size]
        ids, weights = assignment.batch(indices, device)
        target = torch.from_numpy(source.read(indices)).float().to(device)
        prediction = reconstruct_fixed_assignment(codebook, ids, weights)
        prediction_logits = prediction @ query_bank.T / temperature
        target_logits = target @ query_bank.T / temperature
        target_probabilities = torch.softmax(target_logits, dim=-1)
        per_kl = F.kl_div(
            torch.log_softmax(prediction_logits, dim=-1),
            target_probabilities,
            reduction="none",
        ).sum(dim=-1)
        target_top2 = target_logits.topk(k=2, dim=-1)
        prediction_pair = prediction_logits.gather(1, target_top2.indices)
        prediction_margin = prediction_pair[:, 0] - prediction_pair[:, 1]
        target_margin = target_top2.values[:, 0] - target_top2.values[:, 1]
        batch_count = int(indices.size)
        totals["cosine"] += float(
            F.cosine_similarity(prediction, target, dim=-1).sum()
        )
        totals["query_kl"] += float(per_kl.sum())
        totals["margin_mae"] += float((prediction_margin - target_margin).abs().sum())
        totals["top1_agreement"] += float(
            (prediction_logits.argmax(dim=-1) == target_logits.argmax(dim=-1)).sum()
        )
        count += batch_count
    return {
        "mean_cosine": totals["cosine"] / max(1, count),
        "query_kl": totals["query_kl"] / max(1, count),
        "top2_margin_mae": totals["margin_mae"] / max(1, count),
        "top1_agreement": totals["top1_agreement"] / max(1, count),
        "num_samples": count,
    }


def validation_score(mode_metrics, query_kl_weight, query_margin_weight):
    values = []
    for metrics in mode_metrics.values():
        values.append(
            (1.0 - metrics["mean_cosine"])
            + query_kl_weight * metrics["query_kl"]
            + query_margin_weight * metrics["top2_margin_mae"]
        )
    return float(np.mean(values))


def load_mode(name, consensus_path, artifact_dir):
    source = ConsensusFeatureSource(consensus_path)
    assignment = FixedSharedAssignment(artifact_dir)
    if source.num_items != assignment.num_gaussians:
        raise ValueError(f"{name} consensus and assignment sizes differ")
    valid = np.asarray(source.valid_mask, dtype=bool) & assignment.valid_mask
    valid_indices = np.flatnonzero(valid)
    if valid_indices.size == 0:
        raise ValueError(f"{name} mode contains no jointly valid Gaussians")
    return {
        "name": name,
        "source": source,
        "assignment": assignment,
        "valid_indices": valid_indices,
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--base_consensus", required=True)
    parser.add_argument("--candidate_consensus", required=True)
    parser.add_argument("--base_artifact_dir", required=True)
    parser.add_argument("--candidate_artifact_dir", required=True)
    parser.add_argument("--initial_codebook", required=True)
    parser.add_argument("--query_bank", required=True)
    parser.add_argument("--iterations", type=int, default=1200)
    parser.add_argument("--batch_gaussians", type=int, default=4096)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--cosine_weight", type=float, default=1.0)
    parser.add_argument("--query_kl_weight", type=float, default=0.1)
    parser.add_argument("--query_margin_weight", type=float, default=0.05)
    parser.add_argument("--codebook_anchor_weight", type=float, default=0.01)
    parser.add_argument("--prediction_anchor_weight", type=float, default=0.05)
    parser.add_argument("--query_temperature", type=float, default=0.07)
    parser.add_argument("--validation_samples", type=int, default=65536)
    parser.add_argument("--selection_samples", type=int, default=16384)
    parser.add_argument("--validation_interval", type=int, default=100)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.iterations <= 0 or args.batch_gaussians <= 1:
        raise ValueError("Iteration and batch counts must be positive")
    if args.validation_samples <= 0 or args.selection_samples <= 0:
        raise ValueError("Validation sample counts must be positive")
    if args.validation_interval <= 0 or args.log_interval <= 0:
        raise ValueError("Validation and logging intervals must be positive")
    if args.batch_gaussians % 2:
        raise ValueError("--batch_gaussians must be even for balanced modes")
    if args.learning_rate <= 0.0 or args.query_temperature <= 0.0:
        raise ValueError("Learning rate and query temperature must be positive")
    for name in (
        "cosine_weight",
        "query_kl_weight",
        "query_margin_weight",
        "codebook_anchor_weight",
        "prediction_anchor_weight",
    ):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name} must be non-negative")

    output_dir = os.path.abspath(args.output_dir)
    codebook_output = os.path.join(output_dir, "codebook_shared.npy")
    metrics_output = os.path.join(output_dir, "training_metrics.json")
    if os.path.isfile(codebook_output) and os.path.isfile(metrics_output) and not args.force:
        print(f"Reuse joint query-preserving vocabulary: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)

    base = load_mode("base", args.base_consensus, args.base_artifact_dir)
    candidate = load_mode(
        "candidate", args.candidate_consensus, args.candidate_artifact_dir
    )
    modes = (base, candidate)
    feature_dim = base["assignment"].feature_dim
    num_codes = base["assignment"].num_codes
    for mode in modes:
        assignment = mode["assignment"]
        if assignment.feature_dim != feature_dim or assignment.num_codes != num_codes:
            raise ValueError("Semantic modes must use the same vocabulary shape")

    initial_values = np.load(os.path.abspath(args.initial_codebook)).astype(np.float32)
    if initial_values.shape != (num_codes, feature_dim):
        raise ValueError("Initial vocabulary shape does not match fixed assignments")
    initial_values = l2_normalize(initial_values)
    query_values = np.load(os.path.abspath(args.query_bank)).astype(np.float32)
    if query_values.ndim != 2 or query_values.shape[1] != feature_dim:
        raise ValueError("Query bank dimension does not match the vocabulary")

    device = torch.device("cuda")
    initial_codebook = torch.from_numpy(initial_values).float().to(device)
    codebook = torch.nn.Parameter(initial_codebook.clone())
    query_bank = F.normalize(torch.from_numpy(query_values).float().to(device), dim=-1)
    optimizer = torch.optim.SparseAdam([codebook], lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)

    initial_metrics = {
        mode["name"]: evaluate_mode(
            mode["assignment"],
            mode["source"],
            initial_codebook,
            query_bank,
            mode["valid_indices"],
            args.validation_samples,
            args.batch_gaussians,
            args.query_temperature,
            args.seed + 1000 + index,
        )
        for index, mode in enumerate(modes)
    }
    best_codebook = codebook.detach().clone()
    best_iteration = 0
    best_selection_metrics = {
        mode["name"]: evaluate_mode(
            mode["assignment"],
            mode["source"],
            codebook,
            query_bank,
            mode["valid_indices"],
            args.selection_samples,
            args.batch_gaussians,
            args.query_temperature,
            args.seed + 2000 + index,
        )
        for index, mode in enumerate(modes)
    }
    best_score = validation_score(
        best_selection_metrics,
        args.query_kl_weight,
        args.query_margin_weight,
    )

    history = []
    half_batch = args.batch_gaussians // 2
    for iteration in range(1, args.iterations + 1):
        predictions = []
        targets = []
        initial_predictions = []
        used_ids = []
        for mode in modes:
            indices = rng.choice(
                mode["valid_indices"], half_batch, replace=True
            )
            ids, weights = mode["assignment"].batch(indices, device)
            target = torch.from_numpy(mode["source"].read(indices)).float().to(device)
            predictions.append(
                reconstruct_fixed_assignment(
                    codebook, ids, weights, sparse_grad=True
                )
            )
            with torch.no_grad():
                initial_predictions.append(
                    reconstruct_fixed_assignment(initial_codebook, ids, weights)
                )
            targets.append(target)
            used_ids.append(ids[ids >= 0])

        prediction = torch.cat(predictions, dim=0)
        target = torch.cat(targets, dim=0)
        initial_prediction = torch.cat(initial_predictions, dim=0)
        cosine_loss = 1.0 - F.cosine_similarity(
            prediction, target, dim=-1
        ).mean()
        query_kl, query_margin = query_rank_losses(
            prediction,
            target,
            query_bank,
            args.query_temperature,
        )
        prediction_anchor = 1.0 - F.cosine_similarity(
            prediction, initial_prediction, dim=-1
        ).mean()
        unique_ids = torch.unique(torch.cat(used_ids))
        sampled_codewords = F.embedding(unique_ids, codebook, sparse=True)
        codebook_anchor = 1.0 - F.cosine_similarity(
            sampled_codewords, initial_codebook[unique_ids], dim=-1
        ).mean()
        loss = (
            args.cosine_weight * cosine_loss
            + args.query_kl_weight * query_kl
            + args.query_margin_weight * query_margin
            + args.codebook_anchor_weight * codebook_anchor
            + args.prediction_anchor_weight * prediction_anchor
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            codebook[unique_ids] = F.normalize(codebook[unique_ids], dim=-1)

        selection_metrics = None
        selection_score = None
        if (
            iteration % args.validation_interval == 0
            or iteration == args.iterations
        ):
            selection_metrics = {
                mode["name"]: evaluate_mode(
                    mode["assignment"],
                    mode["source"],
                    codebook,
                    query_bank,
                    mode["valid_indices"],
                    args.selection_samples,
                    args.batch_gaussians,
                    args.query_temperature,
                    args.seed + 2000 + index,
                )
                for index, mode in enumerate(modes)
            }
            selection_score = validation_score(
                selection_metrics,
                args.query_kl_weight,
                args.query_margin_weight,
            )
            if selection_score < best_score:
                best_score = selection_score
                best_iteration = iteration
                best_selection_metrics = selection_metrics
                best_codebook.copy_(codebook.detach())

        if iteration == 1 or iteration % args.log_interval == 0 or iteration == args.iterations:
            row = {
                "iteration": iteration,
                "loss": float(loss.detach()),
                "cosine_loss": float(cosine_loss.detach()),
                "query_kl": float(query_kl.detach()),
                "query_margin": float(query_margin.detach()),
                "codebook_anchor": float(codebook_anchor.detach()),
                "prediction_anchor": float(prediction_anchor.detach()),
            }
            if selection_score is not None:
                row["selection_score"] = selection_score
            history.append(row)
            print(json.dumps(row))

    with torch.no_grad():
        codebook.copy_(best_codebook)
    final_metrics = {
        mode["name"]: evaluate_mode(
            mode["assignment"],
            mode["source"],
            codebook,
            query_bank,
            mode["valid_indices"],
            args.validation_samples,
            args.batch_gaussians,
            args.query_temperature,
            args.seed + 1000 + index,
        )
        for index, mode in enumerate(modes)
    }
    values = codebook.detach().cpu().numpy().astype(np.float16)
    np.save(codebook_output, values)
    metrics = {
        "representation": "joint_query_margin_preserving_shared_vocabulary",
        "source": "training-view semantic prototypes only; no evaluation text or 3D labels",
        "initial_codebook": os.path.abspath(args.initial_codebook),
        "query_bank": os.path.abspath(args.query_bank),
        "base": {
            "consensus": os.path.abspath(args.base_consensus),
            "artifact": os.path.abspath(args.base_artifact_dir),
        },
        "candidate": {
            "consensus": os.path.abspath(args.candidate_consensus),
            "artifact": os.path.abspath(args.candidate_artifact_dir),
        },
        "initial_metrics": initial_metrics,
        "final_metrics": final_metrics,
        "checkpoint_selection": {
            "best_iteration": best_iteration,
            "best_score": best_score,
            "metrics": best_selection_metrics,
            "note": "Selected only on held-out training-view semantic prototypes.",
        },
        "history": history,
        "num_codes": num_codes,
        "feature_dim": feature_dim,
        "storage_bytes_fp16": int(values.nbytes),
        "args": vars(args),
    }
    with open(metrics_output, "w") as handle:
        json.dump(metrics, handle, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
