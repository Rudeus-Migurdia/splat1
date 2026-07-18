#!/usr/bin/env python
"""Aggregate part/object semantics and encode them in one shared vocabulary."""

import json
import os
import sys
import time
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F


def aggregate_split_groups(features, weights, point_groups, num_groups, device, chunk_size):
    feature_dim = int(features.shape[1])
    sums = torch.zeros((num_groups, feature_dim), dtype=torch.float32, device=device)
    totals = torch.zeros(num_groups, dtype=torch.float32, device=device)
    for start in range(0, point_groups.size, chunk_size):
        end = min(start + chunk_size, point_groups.size)
        groups = torch.from_numpy(point_groups[start:end].astype(np.int64, copy=False)).to(device)
        local_weights = weights[start:end].float().to(device)
        valid = (groups >= 0) & (local_weights > 0.0)
        if not valid.any():
            continue
        groups = groups[valid]
        local_weights = local_weights[valid]
        values = features[start:end][valid.cpu()].float().to(device)
        sums.index_add_(0, groups, values * local_weights.unsqueeze(-1))
        totals.index_add_(0, groups, local_weights)
    norms = sums.norm(dim=-1)
    centers = F.normalize(sums, dim=-1)
    compactness = (norms / totals.clamp_min(1e-8)).clamp(0.0, 1.0)
    valid = totals > 0.0
    centers[~valid] = 0.0
    return centers.to(torch.float16).cpu(), totals.cpu(), compactness.cpu(), valid.cpu()


def aggregate_source(payload, point_groups, num_groups, device, chunk_size, stability_floor):
    split_features = payload["split_initial_features"].detach().cpu()
    split_weights = payload["split_weights"].detach().cpu()
    if split_features.shape[0] != 2 or split_weights.shape != split_features.shape[:2]:
        raise ValueError("Group semantic source requires two matching view splits")
    split_results = [
        aggregate_split_groups(
            split_features[index],
            split_weights[index],
            point_groups,
            num_groups,
            device,
            chunk_size,
        )
        for index in range(2)
    ]
    first, second = split_results
    supported = first[3] & second[3]
    cross_cosine = F.cosine_similarity(first[0].float(), second[0].float(), dim=-1)
    stability = ((cross_cosine - stability_floor) / (1.0 - stability_floor)).clamp(0.0, 1.0)
    balance = (
        2.0
        * torch.minimum(first[1], second[1])
        / (first[1] + second[1]).clamp_min(1e-8)
    ).clamp(0.0, 1.0)
    compactness = torch.sqrt(first[2] * second[2])
    reliability = stability * torch.sqrt(balance) * compactness
    reliability[~supported] = 0.0
    combined = F.normalize(
        first[0].float() * first[1].unsqueeze(-1)
        + second[0].float() * second[1].unsqueeze(-1),
        dim=-1,
    )
    combined[~supported] = 0.0
    return {
        "features": combined.to(torch.float16),
        "reliability": reliability.float(),
        "supported": supported,
        "cross_cosine": cross_cosine.float(),
        "compactness": compactness.float(),
        "balance": balance.float(),
    }


def fuse_sources(old, auxiliary, max_aux_weight, temperature):
    old_valid = old["supported"]
    aux_valid = auxiliary["supported"]
    gate = torch.sigmoid(
        (auxiliary["reliability"] - old["reliability"]) / temperature
    )
    gate = torch.where(aux_valid, gate, torch.zeros_like(gate))
    gate = torch.where(~old_valid & aux_valid, torch.ones_like(gate), gate)
    values = old["features"].float() + (
        max_aux_weight * gate.unsqueeze(-1) * auxiliary["features"].float()
    )
    valid = old_valid | aux_valid
    values[~old_valid & aux_valid] = auxiliary["features"][~old_valid & aux_valid].float()
    values = F.normalize(values, dim=-1)
    values[~valid] = 0.0
    reliability = (1.0 - gate) * old["reliability"] + gate * auxiliary["reliability"]
    reliability = torch.where(old_valid & ~aux_valid, old["reliability"], reliability)
    reliability[~valid] = 0.0
    return values, reliability, valid, gate


def residual_assign_shared_vocabulary(features, codebook, device, chunk_size):
    import faiss

    vocabulary = np.ascontiguousarray(codebook.astype(np.float32))
    index = faiss.IndexFlatIP(vocabulary.shape[1])
    index.add(vocabulary)
    resources = None
    if device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    all_ids = np.empty((features.shape[0], 2), dtype=np.int32)
    cosines = np.empty(features.shape[0], dtype=np.float32)
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        targets = np.asarray(features[start:end], dtype=np.float32)
        targets /= np.maximum(np.linalg.norm(targets, axis=-1, keepdims=True), 1e-8)
        _, first = index.search(np.ascontiguousarray(targets), 1)
        first = first[:, 0]
        residual = targets - vocabulary[first]
        residual /= np.maximum(np.linalg.norm(residual, axis=-1, keepdims=True), 1e-8)
        _, second = index.search(np.ascontiguousarray(residual), 1)
        second = second[:, 0]
        reconstruction = vocabulary[first] + vocabulary[second]
        reconstruction /= np.maximum(
            np.linalg.norm(reconstruction, axis=-1, keepdims=True), 1e-8
        )
        all_ids[start:end, 0] = first
        all_ids[start:end, 1] = second
        cosines[start:end] = np.sum(targets * reconstruction, axis=-1)
    del resources
    return all_ids, cosines


def quantiles(values):
    values = np.asarray(values)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--hierarchy_dir", required=True)
    parser.add_argument("--old_consensus", required=True)
    parser.add_argument("--aux_consensus", required=True)
    parser.add_argument("--shared_vocabulary", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stability_floor", type=float, default=0.5)
    parser.add_argument("--min_reliability", type=float, default=0.25)
    parser.add_argument("--min_part_size", type=int, default=3)
    parser.add_argument("--min_object_size", type=int, default=8)
    parser.add_argument("--max_aux_weight", type=float, default=1.5)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--save_continuous_diagnostic", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if not -1.0 <= args.stability_floor < 1.0:
        raise ValueError("Stability floor must be in [-1, 1)")
    if not 0.0 <= args.min_reliability <= 1.0:
        raise ValueError("Minimum reliability must be in [0, 1]")
    if args.min_part_size <= 1 or args.min_object_size < args.min_part_size:
        raise ValueError("Object size must be at least part size")
    if args.max_aux_weight < 0.0 or args.temperature <= 0.0 or args.chunk_size <= 0:
        raise ValueError("Fusion and chunk parameters are invalid")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    continuous_manifest = os.path.join(
        output_dir, "continuous_diagnostic", "manifest.json"
    )
    if (
        os.path.isfile(manifest_path)
        and (not args.save_continuous_diagnostic or os.path.isfile(continuous_manifest))
        and not args.force
    ):
        print(f"Reuse hierarchical group semantic codebook: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()

    hierarchy_dir = os.path.abspath(args.hierarchy_dir)
    with open(os.path.join(hierarchy_dir, "manifest.json")) as source:
        hierarchy_manifest = json.load(source)
    part_ids = np.load(os.path.join(hierarchy_dir, "part_group_ids.npy")).astype(np.int64)
    object_ids = np.load(os.path.join(hierarchy_dir, "object_group_ids.npy")).astype(np.int64)
    if part_ids.shape != object_ids.shape:
        raise ValueError("Part and object assignments must match")
    num_gaussians = int(part_ids.size)
    old_payload = torch.load(os.path.abspath(args.old_consensus), map_location="cpu")
    aux_payload = torch.load(os.path.abspath(args.aux_consensus), map_location="cpu")
    if old_payload["split_initial_features"].shape[1] != num_gaussians:
        raise ValueError("Old consensus does not match the hierarchy")
    if aux_payload["split_initial_features"].shape[1] != num_gaussians:
        raise ValueError("Auxiliary consensus does not match the hierarchy")

    token_features = []
    token_reliability = []
    token_levels = []
    point_token_ids = np.full((num_gaussians, 2), -1, dtype=np.int64)
    level_diagnostics = {}
    for slot, (level, point_groups, min_size) in enumerate(
        (("part", part_ids, args.min_part_size), ("object", object_ids, args.min_object_size))
    ):
        valid_points = point_groups >= 0
        num_groups = int(point_groups[valid_points].max()) + 1 if valid_points.any() else 0
        sizes = np.bincount(point_groups[valid_points], minlength=num_groups)
        old = aggregate_source(
            old_payload,
            point_groups,
            num_groups,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        auxiliary = aggregate_source(
            aux_payload,
            point_groups,
            num_groups,
            args.device,
            args.chunk_size,
            args.stability_floor,
        )
        fused, reliability, supported, gate = fuse_sources(
            old, auxiliary, args.max_aux_weight, args.temperature
        )
        selected = supported & (torch.from_numpy(sizes) >= min_size) & (
            reliability >= args.min_reliability
        )
        selected_ids = torch.nonzero(selected, as_tuple=False).squeeze(1).numpy()
        token_map = np.full(num_groups, -1, dtype=np.int64)
        offset = sum(value.shape[0] for value in token_features)
        token_map[selected_ids] = np.arange(selected_ids.size, dtype=np.int64) + offset
        point_token_ids[valid_points, slot] = token_map[point_groups[valid_points]]
        token_features.append(fused[selected].to(torch.float16).numpy())
        token_reliability.append(reliability[selected].numpy())
        token_levels.append(np.full(selected_ids.size, slot, dtype=np.uint8))
        level_diagnostics[level] = {
            "num_groups": num_groups,
            "num_selected_tokens": int(selected_ids.size),
            "selected_fraction": float(selected.float().mean()) if num_groups else 0.0,
            "covered_gaussian_fraction": float((point_token_ids[:, slot] >= 0).mean()),
            "size_quantiles": quantiles(sizes) if sizes.size else {},
            "reliability_quantiles_selected": quantiles(reliability[selected].numpy())
            if selected.any()
            else {},
            "old_cross_cosine_mean_selected": float(old["cross_cosine"][selected].mean())
            if selected.any()
            else 0.0,
            "aux_cross_cosine_mean_selected": float(auxiliary["cross_cosine"][selected].mean())
            if selected.any()
            else 0.0,
            "mean_aux_gate_selected": float(gate[selected].mean()) if selected.any() else 0.0,
        }

    features = np.concatenate(token_features, axis=0)
    reliability = np.concatenate(token_reliability, axis=0)
    levels = np.concatenate(token_levels, axis=0)
    vocabulary_path = os.path.abspath(args.shared_vocabulary)
    vocabulary = np.load(vocabulary_path).astype(np.float32)
    semantic_ids, reconstruction_cosine = residual_assign_shared_vocabulary(
        features, vocabulary, args.device, args.chunk_size
    )
    if vocabulary.shape[0] > np.iinfo(np.uint16).max:
        semantic_dtype = np.uint32
    else:
        semantic_dtype = np.uint16
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    packed_semantic_ids = semantic_ids.astype(semantic_dtype)
    if features.shape[0] > np.iinfo(np.uint16).max:
        point_dtype = np.uint32
    else:
        point_dtype = np.uint16
    point_invalid = int(np.iinfo(point_dtype).max)
    packed_point_ids = np.full(point_token_ids.shape, point_invalid, dtype=point_dtype)
    point_valid = point_token_ids >= 0
    packed_point_ids[point_valid] = point_token_ids[point_valid].astype(point_dtype)
    packed_point_weights = np.where(point_valid, 255, 0).astype(np.uint8)
    packed_reliability = reliability.astype(np.float16)

    vocabulary_link = os.path.join(output_dir, "shared_vocabulary.npy")
    if os.path.lexists(vocabulary_link):
        os.unlink(vocabulary_link)
    os.symlink(vocabulary_path, vocabulary_link)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), packed_semantic_ids)
    np.save(os.path.join(output_dir, "group_level.npy"), levels)
    np.save(os.path.join(output_dir, "group_reliability.npy"), packed_reliability)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), packed_point_weights)
    if args.save_continuous_diagnostic:
        diagnostic_dir = os.path.join(output_dir, "continuous_diagnostic")
        os.makedirs(diagnostic_dir, exist_ok=True)
        continuous_features = features.astype(np.float16)
        np.save(os.path.join(diagnostic_dir, "group_codebook.npy"), continuous_features)
        for name in (
            "group_reliability.npy",
            "point_group_ids.npy",
            "point_group_weights.npy",
        ):
            link = os.path.join(diagnostic_dir, name)
            if os.path.lexists(link):
                os.unlink(link)
            os.symlink(os.path.join("..", name), link)
        continuous_bytes = int(
            continuous_features.nbytes
            + packed_reliability.nbytes
            + packed_point_ids.nbytes
            + packed_point_weights.nbytes
        )
        with open(continuous_manifest, "w") as output:
            json.dump(
                {
                    "format_version": 1,
                    "representation": "compact_group_hierarchy",
                    "num_gaussians": num_gaussians,
                    "num_group_codes": int(features.shape[0]),
                    "feature_dim": int(features.shape[1]),
                    "top_m": 2,
                    "group_codebook": "group_codebook.npy",
                    "point_group_ids": "point_group_ids.npy",
                    "point_group_weights": "point_group_weights.npy",
                    "group_reliability": "group_reliability.npy",
                    "invalid_id": point_invalid,
                    "storage": {
                        "total_semantic_bytes": continuous_bytes,
                        "note": "Continuous diagnostic upper bound; not a deployable result.",
                    },
                },
                output,
                indent=2,
            )
    semantic_bytes = int(
        packed_semantic_ids.nbytes
        + levels.nbytes
        + packed_reliability.nbytes
        + packed_point_ids.nbytes
        + packed_point_weights.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "shared_codebook_group_hierarchy",
        "num_gaussians": num_gaussians,
        "num_group_codes": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "top_m": 2,
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "group_reliability": "group_reliability.npy",
        "group_reliability_dtype": "float16",
        "group_level": "group_level.npy",
        "invalid_id": point_invalid,
        "id_dtype": str(packed_point_ids.dtype),
        "weight_dtype": "uint8_unit_membership",
        "covered_fraction": float(point_valid.any(axis=1).mean()),
        "mean_ids_per_covered_gaussian": float(point_valid[point_valid.any(axis=1)].sum(axis=1).mean()),
        "quantization": {
            "ids_per_group_token": 2,
            "mean_reconstruction_cosine": float(reconstruction_cosine.mean()),
            "reconstruction_cosine_quantiles": quantiles(reconstruction_cosine),
        },
        "levels": level_diagnostics,
        "storage": {
            "shared_vocabulary_bytes_unique": 0,
            "hierarchy_semantic_bytes": semantic_bytes,
            "total_semantic_bytes": semantic_bytes,
            "bytes_per_gaussian_amortized": float(semantic_bytes / num_gaussians),
            "note": "Shared vocabulary storage is already owned by A14 base/candidate artifacts.",
        },
        "source": {
            "hierarchy_dir": hierarchy_dir,
            "hierarchy_manifest": hierarchy_manifest.get("representation"),
            "old_consensus": os.path.abspath(args.old_consensus),
            "aux_consensus": os.path.abspath(args.aux_consensus),
            "shared_vocabulary": vocabulary_path,
            "leakage_control": "training split observations, geometry, RGB, and fixed A17 hierarchy only",
        },
        "elapsed_seconds": time.time() - started,
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
