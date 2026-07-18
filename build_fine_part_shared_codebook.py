#!/usr/bin/env python
"""Append stable fine-scale part modes to an exact shared semantic vocabulary."""

import hashlib
import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_hierarchical_group_semantic_codebook import aggregate_split_groups


def select_fine_tokens(
    sizes,
    reliability,
    disagreement,
    supported,
    min_size,
    max_size,
    min_reliability,
    min_disagreement,
):
    """Select compact, stable fine modes without consulting evaluation queries."""
    return (
        supported
        & (sizes >= min_size)
        & (sizes <= max_size)
        & (reliability >= min_reliability)
        & (disagreement >= min_disagreement)
    )


def manifest_fingerprint(path):
    with open(path, "rb") as source:
        return hashlib.sha256(source.read()).hexdigest()


def quantiles(values):
    values = np.asarray(values)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--part_artifact_dir", required=True)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stability_floor", type=float, default=0.5)
    parser.add_argument("--min_group_size", type=int, default=3)
    parser.add_argument("--max_group_size", type=int, default=32)
    parser.add_argument("--min_reliability", type=float, default=0.6)
    parser.add_argument("--min_disagreement", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.min_group_size <= 1 or args.max_group_size < args.min_group_size:
        raise ValueError("Fine group size range is invalid")
    if not 0.0 <= args.min_reliability <= 1.0:
        raise ValueError("Minimum reliability must be in [0, 1]")
    if not 0.0 <= args.min_disagreement <= 2.0:
        raise ValueError("Minimum disagreement must be in [0, 2]")

    output_dir = os.path.abspath(args.output_dir)
    output_manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(output_manifest_path) and not args.force:
        print(f"Reuse fine-part shared codebook: {output_dir}")
        return

    part_dir = os.path.abspath(args.part_artifact_dir)
    part_manifest_path = os.path.join(part_dir, "manifest.json")
    with open(part_manifest_path) as source:
        part_manifest = json.load(source)
    if part_manifest.get("representation") != "shared_codebook_group_hierarchy":
        raise ValueError("Fine modes require an exact shared part-code artifact")

    vocabulary = np.load(
        os.path.join(part_dir, part_manifest["group_codebook"])
    ).astype(np.float16)
    old_semantic_ids = np.load(
        os.path.join(part_dir, part_manifest["group_semantic_code_ids"])
    ).astype(np.int64)
    old_semantic_invalid = int(part_manifest["semantic_invalid_id"])
    old_point_ids = np.load(
        os.path.join(part_dir, part_manifest["point_group_ids"])
    ).astype(np.int64)
    old_point_invalid = int(part_manifest["invalid_id"])
    old_point_ids[old_point_ids == old_point_invalid] = -1
    old_point_weights = np.load(
        os.path.join(part_dir, part_manifest["point_group_weights"])
    )
    old_reliability = np.load(
        os.path.join(part_dir, part_manifest["group_reliability"])
    ).astype(np.float32)
    if old_point_ids.ndim != 2 or old_point_ids.shape[1] < 1:
        raise ValueError("Part artifact must provide at least one group slot")
    if old_semantic_ids.shape[0] != old_reliability.size:
        raise ValueError("Part semantic and reliability tables do not match")

    part_token_ids = old_point_ids[:, 0]
    num_gaussians = int(part_token_ids.size)
    num_part_tokens = int(old_semantic_ids.shape[0])
    semantic_valid = old_semantic_ids != old_semantic_invalid
    safe_semantic_ids = np.where(semantic_valid, old_semantic_ids, 0)
    if semantic_valid.any() and int(safe_semantic_ids[semantic_valid].max()) >= vocabulary.shape[0]:
        raise ValueError("Part semantic IDs exceed the source vocabulary")
    part_features = (
        vocabulary[safe_semantic_ids].astype(np.float32)
        * semantic_valid[..., None]
    ).sum(axis=1)
    part_features /= np.maximum(
        np.linalg.norm(part_features, axis=-1, keepdims=True), 1e-8
    )

    fine_path = os.path.abspath(args.fine_consensus)
    fine = torch.load(fine_path, map_location="cpu")
    split_features = fine["split_initial_features"].detach().cpu()
    split_weights = fine["split_weights"].detach().cpu()
    if split_features.shape != (2, num_gaussians, vocabulary.shape[1]):
        raise ValueError("Fine consensus does not match the part artifact")
    if split_weights.shape != split_features.shape[:2]:
        raise ValueError("Fine split weights do not match fine features")

    split_results = [
        aggregate_split_groups(
            split_features[index],
            split_weights[index],
            part_token_ids,
            num_part_tokens,
            args.device,
            args.chunk_size,
        )
        for index in range(2)
    ]
    first, second = split_results
    supported = first[3] & second[3]
    cross_cosine = F.cosine_similarity(first[0].float(), second[0].float(), dim=-1)
    stability = ((cross_cosine - args.stability_floor) / (1.0 - args.stability_floor)).clamp(0.0, 1.0)
    balance = (
        2.0 * torch.minimum(first[1], second[1])
        / (first[1] + second[1]).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    compactness = torch.sqrt(first[2] * second[2])
    reliability = stability * torch.sqrt(balance) * compactness
    reliability[~supported] = 0.0
    fine_features = F.normalize(
        first[0].float() * first[1].unsqueeze(-1)
        + second[0].float() * second[1].unsqueeze(-1),
        dim=-1,
    )
    fine_features[~supported] = 0.0
    del fine, split_features, split_weights

    valid_points = part_token_ids >= 0
    sizes = np.bincount(
        part_token_ids[valid_points], minlength=num_part_tokens
    ).astype(np.int64)
    part_features_t = torch.from_numpy(part_features)
    part_supported = part_features_t.norm(dim=-1) > 0.0
    disagreement = (
        1.0 - F.cosine_similarity(fine_features, part_features_t, dim=-1)
    ).clamp(0.0, 2.0)
    selected = select_fine_tokens(
        torch.from_numpy(sizes),
        reliability,
        disagreement,
        supported & part_supported,
        args.min_group_size,
        args.max_group_size,
        args.min_reliability,
        args.min_disagreement,
    )
    selected_ids = torch.nonzero(selected, as_tuple=False).squeeze(1).numpy()
    if not selected_ids.size:
        raise ValueError("No fine part modes passed the fixed training-only gate")

    fine_rows = fine_features[selected].to(torch.float16).numpy()
    extended_vocabulary = np.concatenate((vocabulary, fine_rows), axis=0)
    semantic_dtype = (
        np.uint32
        if extended_vocabulary.shape[0] > np.iinfo(np.uint16).max
        else np.uint16
    )
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    old_rows = np.full(old_semantic_ids.shape, semantic_invalid, dtype=semantic_dtype)
    old_rows[semantic_valid] = old_semantic_ids[semantic_valid].astype(semantic_dtype)
    fine_rows_semantic = np.full(
        (selected_ids.size, old_semantic_ids.shape[1]),
        semantic_invalid,
        dtype=semantic_dtype,
    )
    fine_rows_semantic[:, 0] = (
        vocabulary.shape[0] + np.arange(selected_ids.size, dtype=np.int64)
    ).astype(semantic_dtype)
    semantic_ids = np.concatenate((old_rows, fine_rows_semantic), axis=0)

    total_group_tokens = int(semantic_ids.shape[0])
    point_dtype = (
        np.uint32 if total_group_tokens > np.iinfo(np.uint16).max else np.uint16
    )
    point_invalid = int(np.iinfo(point_dtype).max)
    point_ids = np.full((num_gaussians, 2), point_invalid, dtype=point_dtype)
    point_weights = np.zeros((num_gaussians, 2), dtype=np.uint8)
    part_valid = part_token_ids >= 0
    point_ids[part_valid, 0] = part_token_ids[part_valid].astype(point_dtype)
    point_weights[part_valid, 0] = old_point_weights[part_valid, 0]

    token_to_fine = np.full(num_part_tokens, -1, dtype=np.int64)
    token_to_fine[selected_ids] = num_part_tokens + np.arange(selected_ids.size)
    fine_point_tokens = np.full(num_gaussians, -1, dtype=np.int64)
    fine_point_tokens[part_valid] = token_to_fine[part_token_ids[part_valid]]
    fine_point_valid = fine_point_tokens >= 0
    point_ids[fine_point_valid, 1] = fine_point_tokens[fine_point_valid].astype(point_dtype)
    # The 3D part identity defines the support. Full membership is the progressive
    # expansion step; training-only reliability still gates the query-time gain.
    point_weights[fine_point_valid, 1] = 255
    group_reliability = np.concatenate(
        (old_reliability, reliability[selected].numpy().astype(np.float32))
    ).astype(np.float16)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), extended_vocabulary)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), semantic_ids)
    np.save(os.path.join(output_dir, "group_reliability.npy"), group_reliability)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)

    base_codes = int(part_manifest.get("vocabulary", {}).get("base_codes", 0))
    part_codes = int(part_manifest.get("vocabulary", {}).get("exact_part_codes", vocabulary.shape[0] - base_codes))
    semantic_bytes = int(
        extended_vocabulary.nbytes
        + semantic_ids.nbytes
        + group_reliability.nbytes
        + point_ids.nbytes
        + point_weights.nbytes
    )
    fine_manifest_path = os.path.join(os.path.dirname(fine_path), "manifest.json")
    manifest = {
        "format_version": 2,
        "representation": "shared_codebook_group_hierarchy",
        "method": "identity_preserving_fine_part_modes",
        "num_gaussians": num_gaussians,
        "num_group_codes": total_group_tokens,
        "feature_dim": int(vocabulary.shape[1]),
        "top_m": 1,
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "group_reliability": "group_reliability.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(point_ids.dtype),
        "weight_dtype": "uint8_identity_membership",
        "covered_fraction": float((point_weights > 0).any(axis=1).mean()),
        "fine_covered_fraction": float(fine_point_valid.mean()),
        "mean_ids_per_covered_gaussian": float(
            (point_weights > 0).sum() / max(1, (point_weights > 0).any(axis=1).sum())
        ),
        "vocabulary_modalities": ["base", "part", "fine"],
        "modality_token_counts": {
            "base": base_codes,
            "part": part_codes,
            "fine": int(selected_ids.size),
        },
        "vocabulary": {
            "base_codes": base_codes,
            "exact_part_codes": part_codes,
            "exact_fine_codes": int(selected_ids.size),
            "total_codes": int(extended_vocabulary.shape[0]),
            "construction": "A18 exact vocabulary plus one exact row per selected fine part mode",
        },
        "fine_selection": {
            "selected_tokens": int(selected_ids.size),
            "selected_fraction_of_part_tokens": float(selected.float().mean()),
            "covered_gaussians": int(fine_point_valid.sum()),
            "group_size_quantiles": quantiles(sizes[selected_ids]),
            "reliability_quantiles": quantiles(reliability[selected].numpy()),
            "disagreement_quantiles": quantiles(disagreement[selected].numpy()),
            "mean_cross_split_cosine": float(cross_cosine[selected].mean()),
            "mean_compactness": float(compactness[selected].mean()),
        },
        "continuous_discrete_contract": {
            "continuous_target": "normalized training-view fine part centroid",
            "discrete_encoding": "one exact FP16 shared-vocabulary row",
            "reconstruction_cosine": 1.0,
            "ranking_gap_source": "FP16 roundoff only",
        },
        "module_codebook_contract": {
            "enabled_modules": ["A14_base", "A18_part", "A20_fine_part"],
            "feature_source": "signed multiscale L1 split consensus",
            "identity_source": "A17 3D part token assignments",
            "readout_slots": ["part", "fine"],
            "part_manifest_sha256": manifest_fingerprint(part_manifest_path),
            "fine_manifest_sha256": (
                manifest_fingerprint(fine_manifest_path)
                if os.path.isfile(fine_manifest_path)
                else None
            ),
            "fine_consensus_bytes": int(os.path.getsize(fine_path)),
        },
        "storage": {
            "total_semantic_bytes": semantic_bytes,
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
        },
        "source": {
            "part_artifact_dir": part_dir,
            "fine_consensus": fine_path,
            "leakage_control": "training views, geometry-derived part IDs, and fixed gates only",
        },
        "args": vars(args),
    }
    with open(output_manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
