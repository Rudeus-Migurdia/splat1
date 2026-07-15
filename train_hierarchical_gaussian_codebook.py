#!/usr/bin/env python
"""Jointly train object-level and fine-level discrete Gaussian codebooks."""

import json
import os
import random
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F

from semantic_field_utils import l2_normalize, load_json, save_json
from train_gaussian_multilevel_codebook import (
    MultilevelGaussianCodebook,
    load_query_bank,
    query_distribution_kl,
    render_sampled_codebook,
)
from train_semantic_field import leave_one_view_out_target, load_view_caches
from utils.general_utils import safe_state


def render_composed_codebooks(fine, object_codebook, point_ids, point_weights, object_weight):
    valid = point_ids >= 0
    point_features = l2_normalize(
        fine(point_ids) + object_weight * object_codebook(point_ids)
    )
    weights = torch.where(valid, point_weights, torch.zeros_like(point_weights))
    return l2_normalize((point_features * weights.unsqueeze(-1)).sum(dim=1))


def load_identity_cache(cache_dir):
    manifest = load_json(os.path.join(cache_dir, "manifest.json"))
    if manifest.get("codec_type") != "identity" or int(manifest.get("semantic_dim", 0)) != 512:
        raise ValueError("Hierarchical codebooks require identity 512D observation caches")
    consensus = torch.load(os.path.join(cache_dir, manifest["consensus"]), map_location="cpu")
    return manifest, consensus, load_view_caches(cache_dir, manifest)


def next_batch(caches, cursor, batch_pixels, generator):
    cache = caches[cursor % len(caches)]
    indices = torch.randint(
        cache["point_ids"].shape[0],
        (min(batch_pixels, cache["point_ids"].shape[0]),),
        generator=generator,
    )
    return cache, indices, cursor + 1


def main():
    parser = ArgumentParser(
        description="Train compositional object/fine Gaussian codebooks without per-point continuous semantics."
    )
    parser.add_argument("--fine_cache_dir", required=True)
    parser.add_argument("--object_cache_dir", required=True)
    parser.add_argument("--fine_initial_codebook_dir", required=True)
    parser.add_argument("--object_initial_codebook_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--batch_pixels", type=int, default=4096)
    parser.add_argument("--codebook_lr", type=float, default=1e-3)
    parser.add_argument("--object_feature_weight", type=float, default=0.5)
    parser.add_argument("--object_loss_weight", type=float, default=0.5)
    parser.add_argument("--lovo_weight", type=float, default=0.5)
    parser.add_argument("--lovo_topk", type=int, default=4)
    parser.add_argument("--query_bank", default=None)
    parser.add_argument("--query_kl_weight", type=float, default=0.1)
    parser.add_argument("--lovo_query_kl_weight", type=float, default=0.1)
    parser.add_argument("--query_temperature", type=float, default=0.07)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.iterations <= 0 or args.batch_pixels <= 0:
        raise ValueError("Iterations and batch_pixels must be positive")
    if args.object_feature_weight < 0.0 or args.object_loss_weight < 0.0:
        raise ValueError("Object weights must be non-negative")
    if args.lovo_weight < 0.0 or args.lovo_topk <= 0:
        raise ValueError("LOVO settings are invalid")

    safe_state(args.quiet)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    output_dir = os.path.abspath(args.output)
    fine_deployment_dir = os.path.join(output_dir, "fine_artifact")
    object_deployment_dir = os.path.join(output_dir, "object_artifact")
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    if os.path.isfile(metrics_path) and not args.force:
        print(f"Reuse hierarchical codebook training: {metrics_path}")
        return
    os.makedirs(output_dir, exist_ok=True)

    fine_manifest, fine_consensus, fine_caches = load_identity_cache(
        os.path.abspath(args.fine_cache_dir)
    )
    object_manifest, _, object_caches = load_identity_cache(
        os.path.abspath(args.object_cache_dir)
    )
    if int(fine_manifest["num_gaussians"]) != int(object_manifest["num_gaussians"]):
        raise ValueError("Fine and object caches must reference the same Gaussian geometry")

    fine = MultilevelGaussianCodebook(args.fine_initial_codebook_dir).cuda()
    object_codebook = MultilevelGaussianCodebook(args.object_initial_codebook_dir).cuda()
    if fine.num_gaussians != object_codebook.num_gaussians or fine.feature_dim != object_codebook.feature_dim:
        raise ValueError("Fine and object codebooks must have matching Gaussian count and feature dimensions")
    query_bank = load_query_bank(args.query_bank, fine.feature_dim, "cuda")
    if (args.query_kl_weight > 0.0 or args.lovo_query_kl_weight > 0.0) and query_bank is None:
        raise ValueError("Positive query KL weights require --query_bank")

    optimizer = torch.optim.AdamW(
        list(fine.codebooks.parameters()) + list(object_codebook.codebooks.parameters()),
        lr=args.codebook_lr,
        weight_decay=1e-5,
    )
    fine_total_sums = fine_consensus["total_sums"].float().cuda()
    fine_total_weights = fine_consensus["total_weights"].float().cuda()
    generator = torch.Generator().manual_seed(args.seed)
    fine_cursor = 0
    object_cursor = 0
    history = []
    running = {name: 0.0 for name in ("loss", "fine", "object", "lovo", "query_kl", "lovo_query_kl")}

    for iteration in range(1, args.iterations + 1):
        fine_cache, fine_indices, fine_cursor = next_batch(
            fine_caches, fine_cursor, args.batch_pixels, generator
        )
        fine_ids = fine_cache["point_ids"][fine_indices].long().cuda(non_blocking=True)
        fine_weights = fine_cache["point_weights"][fine_indices].float().cuda(non_blocking=True)
        fine_segments = fine_cache["segment_ids"][fine_indices].long()
        fine_targets = fine_cache["feature_latents"][fine_segments].float().cuda(non_blocking=True)
        composed = render_composed_codebooks(
            fine, object_codebook, fine_ids, fine_weights, args.object_feature_weight
        )
        fine_loss = 1.0 - F.cosine_similarity(composed, fine_targets, dim=-1).mean()
        query_kl_loss = query_distribution_kl(
            composed, fine_targets, query_bank, args.query_temperature
        )
        loo_target, valid_loo = leave_one_view_out_target(
            fine_cache,
            fine_indices,
            fine_ids[:, : args.lovo_topk],
            fine_weights[:, : args.lovo_topk],
            fine_total_sums,
            fine_total_weights,
        )
        if valid_loo.any():
            lovo_loss = 1.0 - F.cosine_similarity(
                composed[valid_loo], loo_target[valid_loo], dim=-1
            ).mean()
            lovo_query_kl_loss = query_distribution_kl(
                composed[valid_loo], loo_target[valid_loo], query_bank, args.query_temperature
            )
        else:
            lovo_loss = composed.new_zeros(())
            lovo_query_kl_loss = composed.new_zeros(())

        object_cache, object_indices, object_cursor = next_batch(
            object_caches, object_cursor, args.batch_pixels, generator
        )
        object_ids = object_cache["point_ids"][object_indices].long().cuda(non_blocking=True)
        object_weights = object_cache["point_weights"][object_indices].float().cuda(non_blocking=True)
        object_segments = object_cache["segment_ids"][object_indices].long()
        object_targets = object_cache["feature_latents"][object_segments].float().cuda(non_blocking=True)
        object_prediction = render_sampled_codebook(object_codebook, object_ids, object_weights)
        object_loss = 1.0 - F.cosine_similarity(
            object_prediction, object_targets, dim=-1
        ).mean()

        loss = (
            fine_loss
            + args.object_loss_weight * object_loss
            + args.lovo_weight * lovo_loss
            + args.query_kl_weight * query_kl_loss
            + args.lovo_query_kl_weight * lovo_query_kl_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        values = {
            "loss": float(loss.detach()),
            "fine": float(fine_loss.detach()),
            "object": float(object_loss.detach()),
            "lovo": float(lovo_loss.detach()),
            "query_kl": float(query_kl_loss.detach()),
            "lovo_query_kl": float(lovo_query_kl_loss.detach()),
        }
        for name, value in values.items():
            running[name] += value
        if iteration % args.log_interval == 0 or iteration == args.iterations:
            divisor = args.log_interval if iteration % args.log_interval == 0 else iteration % args.log_interval
            row = {"iteration": iteration}
            row.update({name: value / max(1, divisor) for name, value in running.items()})
            history.append(row)
            print(json.dumps(row))
            running = {name: 0.0 for name in running}

    training_metadata = {"config": vars(args), "fine_cache_dir": os.path.abspath(args.fine_cache_dir), "object_cache_dir": os.path.abspath(args.object_cache_dir)}
    fine_manifest_out = fine.save_deployment_artifact(fine_deployment_dir, training_metadata)
    object_manifest_out = object_codebook.save_deployment_artifact(object_deployment_dir, training_metadata)
    storage_bytes = int(fine_manifest_out["storage"]["total_semantic_bytes"]) + int(object_manifest_out["storage"]["total_semantic_bytes"])
    results = {
        "representation": "hierarchical_object_fine_gaussian_codebook",
        "fine_artifact": fine_deployment_dir,
        "object_artifact": object_deployment_dir,
        "storage_bytes": storage_bytes,
        "storage_megabytes": storage_bytes / (1024.0 ** 2),
        "history": history,
        "config": vars(args),
    }
    save_json(metrics_path, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
