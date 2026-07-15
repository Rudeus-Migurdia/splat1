#!/usr/bin/env python
import json
import os
import sys
from argparse import ArgumentParser

import torch
from torch.nn import functional as F
from tqdm import tqdm

from eval_lerf_ovs_miou import load_lerf_labels
from evaluation.openclip_encoder import OpenCLIPNetwork
from semantic_field_utils import l2_normalize, load_json, load_semantic_codec, save_json
from train_gaussian_multilevel_codebook import (
    MultilevelGaussianCodebook,
    render_sampled_codebook,
)
from utils.general_utils import safe_state


def render_features(features, point_ids, point_weights):
    valid = point_ids >= 0
    safe_ids = torch.where(valid, point_ids, torch.zeros_like(point_ids))
    weights = torch.where(valid, point_weights, torch.zeros_like(point_weights))
    point_features = l2_normalize(features[safe_ids])
    return l2_normalize((point_features * weights.unsqueeze(-1)).sum(dim=1))


def render_lovo_target(cache, sample_indices, point_ids, point_weights, total_sums, total_weights):
    aggregate_indices_cpu = cache["pixel_aggregate_indices"][
        sample_indices, : point_ids.shape[1]
    ].long()
    aggregate_indices = aggregate_indices_cpu.cuda()
    valid = (point_ids >= 0) & (aggregate_indices >= 0)
    safe_ids = torch.where(valid, point_ids, torch.zeros_like(point_ids))
    safe_aggregate_indices = torch.where(
        aggregate_indices_cpu >= 0,
        aggregate_indices_cpu,
        torch.zeros_like(aggregate_indices_cpu),
    )
    current_sums = cache["aggregate_sums"][safe_aggregate_indices].cuda()
    current_weights = cache["aggregate_weights"][safe_aggregate_indices].cuda()
    loo_sums = total_sums[safe_ids] - current_sums
    loo_weights = total_weights[safe_ids] - current_weights
    valid &= loo_weights > 1e-6
    point_targets = l2_normalize(loo_sums / loo_weights.clamp_min(1e-6).unsqueeze(-1))
    effective_weights = torch.where(valid, point_weights, torch.zeros_like(point_weights))
    rendered = l2_normalize((point_targets * effective_weights.unsqueeze(-1)).sum(dim=1))
    return rendered, effective_weights.sum(dim=1) > 1e-6


def query_distribution(clip_model, decoded_features, num_categories):
    activations = torch.cat(
        [clip_model.get_activation(decoded_features, index) for index in range(num_categories)],
        dim=1,
    ).float()
    return activations / activations.sum(dim=1, keepdim=True).clamp_min(1e-8)


def symmetric_kl(first, second):
    first = first.clamp_min(1e-8)
    second = second.clamp_min(1e-8)
    return 0.5 * (
        (first * (first.log() - second.log())).sum(dim=1)
        + (second * (second.log() - first.log())).sum(dim=1)
    )


def main():
    parser = ArgumentParser(description="Measure held-out cross-view query consistency.")
    parser.add_argument("--cache_dir", required=True)
    semantic_source = parser.add_mutually_exclusive_group(required=True)
    semantic_source.add_argument("--semantic_artifact")
    semantic_source.add_argument("--codebook_dir")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples_per_view", type=int, default=256)
    parser.add_argument("--lovo_topk", type=int, default=4)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if args.samples_per_view <= 0 or args.lovo_topk <= 0:
        raise ValueError("samples_per_view and lovo_topk must be positive")
    safe_state(args.quiet)
    cache_dir = os.path.abspath(args.cache_dir)
    manifest = load_json(os.path.join(cache_dir, "manifest.json"))
    consensus = torch.load(os.path.join(cache_dir, manifest["consensus"]), map_location="cpu")
    artifact = None
    features = None
    codebook = None
    if args.semantic_artifact:
        artifact = torch.load(args.semantic_artifact, map_location="cpu")
        features = artifact["semantic_features"].float().cuda()
        codec_path = artifact.get("codec") or os.path.join(cache_dir, manifest["codec"])
    else:
        codebook = MultilevelGaussianCodebook(args.codebook_dir).cuda()
        if codebook.num_gaussians != int(manifest["num_gaussians"]):
            raise ValueError("Gaussian codebook does not match the observation cache")
        codec_path = os.path.join(cache_dir, manifest["codec"])
    codec, _codec_payload = load_semantic_codec(codec_path)
    _labels, categories = load_lerf_labels(args.label_dir)
    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    total_sums = consensus["total_sums"].float().cuda()
    total_weights = consensus["total_weights"].float().cuda()
    generator = torch.Generator().manual_seed(args.seed)

    totals = {
        "canonical_lovo_symmetric_kl": 0.0,
        "observation_lovo_symmetric_kl": 0.0,
        "canonical_observation_symmetric_kl": 0.0,
        "canonical_lovo_label_flips": 0,
        "observation_lovo_label_flips": 0,
        "canonical_observation_label_flips": 0,
        "count": 0,
    }
    entries = manifest["views"]
    if args.max_views > 0:
        entries = entries[: args.max_views]
    for entry in tqdm(entries, desc="Evaluating cross-view query consistency"):
        cache = torch.load(os.path.join(cache_dir, entry["cache"]), map_location="cpu")
        num_pixels = cache["point_ids"].shape[0]
        if num_pixels == 0:
            continue
        sample_count = min(args.samples_per_view, num_pixels)
        sample_indices = torch.randperm(num_pixels, generator=generator)[:sample_count]
        topk = min(args.lovo_topk, cache["point_ids"].shape[1])
        point_ids = cache["point_ids"][sample_indices, :topk].long().cuda()
        point_weights = cache["point_weights"][sample_indices, :topk].float().cuda()
        segment_ids = cache["segment_ids"][sample_indices].long()
        observations = cache["feature_latents"][segment_ids].float().cuda()
        if codebook is None:
            canonical = render_features(features, point_ids, point_weights)
        else:
            canonical = render_sampled_codebook(codebook, point_ids, point_weights)
        lovo, valid = render_lovo_target(
            cache,
            sample_indices,
            point_ids,
            point_weights,
            total_sums,
            total_weights,
        )
        if not valid.any():
            continue
        canonical = canonical[valid]
        observations = observations[valid]
        lovo = lovo[valid]
        decoded = codec.decode(torch.cat([canonical, observations, lovo], dim=0))
        count = canonical.shape[0]
        canonical_query, observation_query, lovo_query = query_distribution(
            clip_model,
            decoded,
            len(categories),
        ).split(count)

        totals["canonical_lovo_symmetric_kl"] += float(
            symmetric_kl(canonical_query, lovo_query).sum()
        )
        totals["observation_lovo_symmetric_kl"] += float(
            symmetric_kl(observation_query, lovo_query).sum()
        )
        totals["canonical_observation_symmetric_kl"] += float(
            symmetric_kl(canonical_query, observation_query).sum()
        )
        totals["canonical_lovo_label_flips"] += int(
            (canonical_query.argmax(dim=1) != lovo_query.argmax(dim=1)).sum()
        )
        totals["observation_lovo_label_flips"] += int(
            (observation_query.argmax(dim=1) != lovo_query.argmax(dim=1)).sum()
        )
        totals["canonical_observation_label_flips"] += int(
            (canonical_query.argmax(dim=1) != observation_query.argmax(dim=1)).sum()
        )
        totals["count"] += count

    count = max(1, totals["count"])
    results = {
        "cache_dir": cache_dir,
        "semantic_artifact": os.path.abspath(args.semantic_artifact)
        if args.semantic_artifact
        else None,
        "codebook_dir": os.path.abspath(args.codebook_dir) if args.codebook_dir else None,
        "categories": categories,
        "num_samples": totals["count"],
        "canonical_lovo_symmetric_kl": totals["canonical_lovo_symmetric_kl"] / count,
        "observation_lovo_symmetric_kl": totals["observation_lovo_symmetric_kl"] / count,
        "canonical_observation_symmetric_kl": totals["canonical_observation_symmetric_kl"] / count,
        "canonical_lovo_label_flip_rate": totals["canonical_lovo_label_flips"] / count,
        "observation_lovo_label_flip_rate": totals["observation_lovo_label_flips"] / count,
        "canonical_observation_label_flip_rate": totals["canonical_observation_label_flips"] / count,
        "samples_per_view": args.samples_per_view,
        "lovo_topk": args.lovo_topk,
    }
    save_json(args.output, results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
