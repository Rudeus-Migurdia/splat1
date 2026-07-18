#!/usr/bin/env python
"""Append stable L2 micro semantics inside A20 fine-part identities."""

import hashlib
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def manifest_fingerprint(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def normalize(values):
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)


def decode_group_features(vocabulary, semantic_ids, invalid):
    valid = semantic_ids != invalid
    safe = np.where(valid, semantic_ids, 0)
    decoded = (vocabulary[safe].astype(np.float32) * valid[..., None]).sum(axis=1)
    return normalize(decoded)


def fine_identity_mapping(part_ids, fine_ids, invalid):
    valid = (part_ids != invalid) & (fine_ids != invalid)
    parts = part_ids[valid].astype(np.int64)
    fine = fine_ids[valid].astype(np.int64)
    if not parts.size:
        raise ValueError("A20 artifact has no resident fine identities")
    order = np.argsort(parts, kind="stable")
    parts = parts[order]
    fine = fine[order]
    starts = np.r_[0, np.flatnonzero(parts[1:] != parts[:-1]) + 1]
    ends = np.r_[starts[1:], parts.size]
    selected_parts = parts[starts]
    selected_fine = fine[starts]
    for start, end, expected in zip(starts, ends, selected_fine):
        if np.any(fine[start:end] != expected):
            raise ValueError("One part identity maps to multiple A20 fine tokens")
    return selected_parts, selected_fine


def select_micro_tokens(
    reliability,
    disagreement,
    supported,
    minimum_reliability,
    minimum_disagreement,
):
    return (
        supported
        & (reliability >= minimum_reliability)
        & (disagreement >= minimum_disagreement)
    )


def quantiles(values):
    values = np.asarray(values)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    import torch
    from torch.nn import functional as F

    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a20_artifact_dir", required=True)
    parser.add_argument("--l2_view_cache_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stability_floor", type=float, default=0.5)
    parser.add_argument("--min_views_per_split", type=int, default=3)
    parser.add_argument("--min_reliability", type=float, default=0.6)
    parser.add_argument("--min_disagreement", type=float, default=0.05)
    parser.add_argument("--max_views", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not -1.0 < args.stability_floor < 1.0:
        raise ValueError("Stability floor must lie in (-1, 1)")
    if args.min_views_per_split <= 0 or args.max_views < 0:
        raise ValueError("View parameters are invalid")
    if not 0.0 <= args.min_reliability <= 1.0:
        raise ValueError("Minimum reliability must be in [0, 1]")
    if not 0.0 <= args.min_disagreement <= 2.0:
        raise ValueError("Minimum disagreement must be in [0, 2]")

    output_dir = os.path.abspath(args.output_dir)
    output_manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(output_manifest_path) and not args.force:
        print(f"Reuse multiscale micro codebook: {output_dir}")
        return

    a20_dir = os.path.abspath(args.a20_artifact_dir)
    a20_manifest_path = os.path.join(a20_dir, "manifest.json")
    with open(a20_manifest_path) as source:
        a20_manifest = json.load(source)
    required = {"base", "part", "fine"}
    if not required.issubset(a20_manifest.get("vocabulary_modalities", [])):
        raise ValueError("A24 requires an A20 base+part+fine artifact")
    vocabulary = np.load(
        os.path.join(a20_dir, a20_manifest["group_codebook"])
    ).astype(np.float16)
    semantic_ids = np.load(
        os.path.join(a20_dir, a20_manifest["group_semantic_code_ids"])
    )
    semantic_invalid = int(a20_manifest["semantic_invalid_id"])
    decoded = decode_group_features(vocabulary, semantic_ids, semantic_invalid)
    point_ids = np.load(
        os.path.join(a20_dir, a20_manifest["point_group_ids"])
    )
    point_weights = np.load(
        os.path.join(a20_dir, a20_manifest["point_group_weights"])
    )
    point_invalid = int(a20_manifest["invalid_id"])
    if point_ids.ndim != 2 or point_ids.shape[1] < 2:
        raise ValueError("A24 requires resident part and fine IDs")
    num_gaussians = int(point_ids.shape[0])
    selected_parts, selected_fine_tokens = fine_identity_mapping(
        point_ids[:, 0], point_ids[:, 1], point_invalid
    )
    if selected_fine_tokens.max() >= decoded.shape[0]:
        raise ValueError("Fine token IDs exceed the A20 semantic table")
    l1_fine_features = decoded[selected_fine_tokens]
    num_identities = int(selected_parts.size)
    part_to_identity = np.full(int(selected_parts.max()) + 1, -1, dtype=np.int64)
    part_to_identity[selected_parts] = np.arange(num_identities, dtype=np.int64)

    cache_dir = os.path.abspath(args.l2_view_cache_dir)
    cache_manifest_path = os.path.join(cache_dir, "manifest.json")
    with open(cache_manifest_path) as source:
        cache_manifest = json.load(source)
    expected = {
        "num_gaussians": num_gaussians,
        "feature_level": 2,
        "semantic_dim": int(vocabulary.shape[1]),
    }
    for name, value in expected.items():
        if int(cache_manifest.get(name, -1)) != value:
            raise ValueError(f"L2 cache {name} does not match A20")
    if not cache_manifest.get("compact_view_cache", False):
        raise ValueError("A24 expects a compact per-view L2 cache")
    if not cache_manifest.get("signed_segment_ownership", False):
        raise ValueError("A24 requires signed segment ownership")
    if not cache_manifest.get("raw_contribution_weights", False):
        raise ValueError("A24 requires raw T*alpha contribution weights")
    if int(cache_manifest.get("topk", 0)) < 45:
        raise ValueError("A24 requires at least top-45 contributors")
    entries = cache_manifest["views"]
    if args.max_views:
        entries = entries[: args.max_views]
    if not entries:
        raise ValueError("L2 cache has no views")

    device = torch.device(args.device)
    split_sums = torch.zeros(
        (2, num_identities, vocabulary.shape[1]), dtype=torch.float32, device=device
    )
    split_weights = torch.zeros(
        (2, num_identities), dtype=torch.float32, device=device
    )
    split_views = torch.zeros(
        (2, num_identities), dtype=torch.int32, device=device
    )
    for entry_index, entry in enumerate(entries):
        payload = torch.load(os.path.join(cache_dir, entry["cache"]), map_location="cpu")
        gaussian_ids = payload["aggregate_ids"].numpy().astype(np.int64, copy=False)
        valid_gaussians = (gaussian_ids >= 0) & (gaussian_ids < num_gaussians)
        safe_gaussians = np.clip(gaussian_ids, 0, num_gaussians - 1)
        part_tokens = point_ids[safe_gaussians, 0].astype(np.int64)
        part_valid = (
            (part_tokens != point_invalid)
            & (part_tokens >= 0)
            & (part_tokens < part_to_identity.size)
        )
        safe_parts = np.clip(part_tokens, 0, part_to_identity.size - 1)
        identities = part_to_identity[safe_parts]
        valid = valid_gaussians & part_valid & (identities >= 0)
        if valid.any():
            rows = torch.from_numpy(np.flatnonzero(valid)).long()
            identity_ids = torch.from_numpy(identities[valid]).long().to(device)
            observations = payload["aggregate_sums"].index_select(0, rows).to(
                device, dtype=torch.float32
            )
            weights = payload["aggregate_weights"].index_select(0, rows).to(
                device, dtype=torch.float32
            )
            split_index = int(entry.get("view_index", entry_index)) % 2
            split_sums[split_index].index_add_(0, identity_ids, observations)
            split_weights[split_index].index_add_(0, identity_ids, weights)
            split_views[split_index, torch.unique(identity_ids)] += 1
        print(
            json.dumps(
                {
                    "view": entry_index + 1,
                    "total_views": len(entries),
                    "image": entry["image_name"],
                }
            ),
            flush=True,
        )

    split_features = F.normalize(split_sums, dim=-1)
    supported = (
        (split_weights > 0.0).all(dim=0)
        & (split_views >= args.min_views_per_split).all(dim=0)
        & (split_sums.norm(dim=-1) > 0.0).all(dim=0)
    )
    cross_cosine = F.cosine_similarity(
        split_features[0], split_features[1], dim=-1
    )
    stability = (
        (cross_cosine - args.stability_floor) / (1.0 - args.stability_floor)
    ).clamp(0.0, 1.0)
    balance = (
        2.0 * torch.minimum(split_weights[0], split_weights[1])
        / (split_weights[0] + split_weights[1]).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    compactness_per_split = (
        split_sums.norm(dim=-1) / split_weights.clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    compactness = torch.sqrt(
        compactness_per_split[0] * compactness_per_split[1]
    )
    reliability = stability * torch.sqrt(balance) * compactness
    reliability[~supported] = 0.0
    micro_features = F.normalize(split_sums.sum(dim=0), dim=-1)
    l1_features_t = torch.from_numpy(l1_fine_features).to(device)
    disagreement = (
        1.0 - F.cosine_similarity(micro_features, l1_features_t, dim=-1)
    ).clamp(0.0, 2.0)
    selected = select_micro_tokens(
        reliability,
        disagreement,
        supported,
        args.min_reliability,
        args.min_disagreement,
    )
    selected_indices = torch.nonzero(selected, as_tuple=False).squeeze(1)
    if not selected_indices.numel():
        raise ValueError("No L2 micro identities pass the fixed stability gate")
    selected_np = selected_indices.cpu().numpy()
    micro_rows = micro_features[selected_indices].to(torch.float16).cpu().numpy()

    extended_vocabulary = np.concatenate((vocabulary, micro_rows), axis=0)
    semantic_dtype = (
        np.uint32
        if extended_vocabulary.shape[0] > np.iinfo(np.uint16).max
        else np.uint16
    )
    new_semantic_invalid = int(np.iinfo(semantic_dtype).max)
    semantic_rows = np.full(
        semantic_ids.shape, new_semantic_invalid, dtype=semantic_dtype
    )
    semantic_valid = semantic_ids != semantic_invalid
    semantic_rows[semantic_valid] = semantic_ids[semantic_valid].astype(semantic_dtype)
    micro_semantic = np.full(
        (selected_np.size, semantic_ids.shape[1]),
        new_semantic_invalid,
        dtype=semantic_dtype,
    )
    micro_semantic[:, 0] = (
        vocabulary.shape[0] + np.arange(selected_np.size, dtype=np.int64)
    ).astype(semantic_dtype)
    extended_semantic = np.concatenate((semantic_rows, micro_semantic), axis=0)

    total_tokens = int(extended_semantic.shape[0])
    point_dtype = (
        np.uint32 if total_tokens > np.iinfo(np.uint16).max else np.uint16
    )
    new_point_invalid = int(np.iinfo(point_dtype).max)
    output_ids = np.full(
        (num_gaussians, 3), new_point_invalid, dtype=point_dtype
    )
    output_weights = np.zeros((num_gaussians, 3), dtype=np.uint8)
    old_valid = point_ids != point_invalid
    output_ids[:, :2][old_valid] = point_ids[old_valid].astype(point_dtype)
    output_weights[:, :2] = point_weights[:, :2]

    identity_to_micro = np.full(num_identities, -1, dtype=np.int64)
    identity_to_micro[selected_np] = semantic_ids.shape[0] + np.arange(
        selected_np.size
    )
    point_parts = point_ids[:, 0].astype(np.int64)
    part_valid = (
        (point_parts != point_invalid)
        & (point_parts >= 0)
        & (point_parts < part_to_identity.size)
    )
    safe_parts = np.clip(point_parts, 0, part_to_identity.size - 1)
    point_identity = np.full(num_gaussians, -1, dtype=np.int64)
    point_identity[part_valid] = part_to_identity[safe_parts[part_valid]]
    micro_tokens = np.full(num_gaussians, -1, dtype=np.int64)
    identity_valid = point_identity >= 0
    micro_tokens[identity_valid] = identity_to_micro[point_identity[identity_valid]]
    micro_point_valid = micro_tokens >= 0
    output_ids[micro_point_valid, 2] = micro_tokens[micro_point_valid].astype(
        point_dtype
    )
    output_weights[micro_point_valid, 2] = 255

    old_reliability = np.load(
        os.path.join(a20_dir, a20_manifest["group_reliability"])
    ).astype(np.float32)
    extended_reliability = np.concatenate(
        (old_reliability, reliability[selected_indices].cpu().numpy())
    ).astype(np.float16)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), extended_vocabulary)
    np.save(
        os.path.join(output_dir, "group_semantic_code_ids.npy"), extended_semantic
    )
    np.save(
        os.path.join(output_dir, "group_reliability.npy"), extended_reliability
    )
    np.save(os.path.join(output_dir, "point_group_ids.npy"), output_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), output_weights)

    counts = dict(a20_manifest["modality_token_counts"])
    counts["micro"] = int(selected_np.size)
    semantic_bytes = int(
        extended_vocabulary.nbytes
        + extended_semantic.nbytes
        + extended_reliability.nbytes
        + output_ids.nbytes
        + output_weights.nbytes
    )
    manifest = {
        **a20_manifest,
        "format_version": max(3, int(a20_manifest.get("format_version", 1))),
        "method": "identity_preserving_multiscale_micro_modes",
        "num_group_codes": total_tokens,
        "top_m": 2,
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": new_semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "group_reliability": "group_reliability.npy",
        "invalid_id": new_point_invalid,
        "id_dtype": str(output_ids.dtype),
        "weight_dtype": "uint8_identity_membership",
        "vocabulary_modalities": ["base", "part", "fine", "micro"],
        "modality_token_counts": counts,
        "covered_fraction": float((output_weights > 0).any(axis=1).mean()),
        "mean_ids_per_covered_gaussian": float(
            (output_weights > 0).sum()
            / max(1, (output_weights > 0).any(axis=1).sum())
        ),
        "micro_selection": {
            "source_fine_identities": num_identities,
            "selected_tokens": int(selected_np.size),
            "selected_fraction": float(selected.float().mean().item()),
            "covered_gaussians": int(micro_point_valid.sum()),
            "covered_fraction": float(micro_point_valid.mean()),
            "cross_split_cosine": quantiles(cross_cosine[selected_indices].cpu().numpy()),
            "reliability": quantiles(reliability[selected_indices].cpu().numpy()),
            "disagreement_vs_l1": quantiles(
                disagreement[selected_indices].cpu().numpy()
            ),
            "compactness": quantiles(compactness[selected_indices].cpu().numpy()),
            "view_count_split0": quantiles(
                split_views[0, selected_indices].cpu().numpy()
            ),
            "view_count_split1": quantiles(
                split_views[1, selected_indices].cpu().numpy()
            ),
        },
        "vocabulary": {
            **a20_manifest["vocabulary"],
            "exact_micro_codes": int(selected_np.size),
            "total_codes": int(extended_vocabulary.shape[0]),
            "construction": (
                "A20 exact vocabulary plus one exact FP16 L2 row per stable, "
                "L1-disagreeing fine identity"
            ),
        },
        "continuous_discrete_contract": {
            "continuous_target": "normalized signed-L2 training-view micro centroid",
            "discrete_encoding": "one exact FP16 shared-vocabulary row",
            "reconstruction_cosine": 1.0,
            "ranking_gap_source": "FP16 roundoff only",
        },
        "module_codebook_contract": {
            "enabled_modules": [
                "A14_base",
                "A18_part",
                "A20_fine_part",
                "A24_multiscale_micro_identity",
            ],
            "feature_source": "signed multiscale L2 compact full-view cache",
            "identity_source": "A20 fine-part identities",
            "readout_slots": ["part", "fine", "micro"],
            "a20_manifest_sha256": manifest_fingerprint(a20_manifest_path),
            "l2_cache_manifest_sha256": manifest_fingerprint(cache_manifest_path),
        },
        "storage": {
            "total_semantic_bytes": semantic_bytes,
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
        },
        "source": {
            **a20_manifest.get("source", {}),
            "a20_artifact_dir": a20_dir,
            "l2_view_cache_dir": cache_dir,
            "leakage_control": (
                "training-view signed L2 observations and A20 identities only; "
                "no evaluation text, labels, 3D GT, or PQ teacher"
            ),
        },
        "args": vars(args),
    }
    with open(output_manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest["micro_selection"], indent=2))


if __name__ == "__main__":
    main()
