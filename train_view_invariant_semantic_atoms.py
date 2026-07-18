#!/usr/bin/env python
"""Learn split-invariant fine atoms with adjacent-part hard negatives."""

import hashlib
import json
import math
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_gaussian_superpoint_support import build_knn, load_geometry
from build_hierarchical_group_semantic_codebook import aggregate_split_groups


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def decode_group_features(vocabulary, semantic_ids, invalid_id):
    valid = semantic_ids != invalid_id
    safe = np.where(valid, semantic_ids, 0)
    if valid.any() and int(safe[valid].max()) >= vocabulary.shape[0]:
        raise ValueError("Semantic IDs exceed the shared vocabulary")
    features = (vocabulary[safe].astype(np.float32) * valid[..., None]).sum(axis=1)
    features /= np.maximum(np.linalg.norm(features, axis=-1, keepdims=True), 1e-8)
    return features


def select_competitors(
    local_groups,
    part_groups,
    neighbors,
    fine_features,
    part_features,
):
    """Choose the most spatially supported semantically confusing neighbor."""
    source = np.repeat(local_groups[:, None], neighbors.shape[1], axis=1)
    neighbor_parts = part_groups[neighbors]
    own_parts = np.repeat(part_groups[:, None], neighbors.shape[1], axis=1)
    valid = (source >= 0) & (neighbor_parts >= 0) & (neighbor_parts != own_parts)
    source = source[valid].astype(np.int64)
    destination = neighbor_parts[valid].astype(np.int64)
    if not source.size:
        return np.full(fine_features.shape[0], -1, dtype=np.int64), np.zeros(
            fine_features.shape[0], dtype=np.int64
        )
    width = int(part_features.shape[0])
    packed = source * width + destination
    pairs, counts = np.unique(packed, return_counts=True)
    pair_source = pairs // width
    pair_destination = pairs % width
    cosine = np.sum(
        fine_features[pair_source] * part_features[pair_destination], axis=-1
    )
    utility = np.log1p(counts.astype(np.float32)) * (0.1 + np.clip(cosine, 0.0, 1.0))
    order = np.lexsort((-utility, pair_source))
    pair_source = pair_source[order]
    pair_destination = pair_destination[order]
    counts = counts[order]
    first = np.r_[True, pair_source[1:] != pair_source[:-1]]
    competitor = np.full(fine_features.shape[0], -1, dtype=np.int64)
    boundary_edges = np.zeros(fine_features.shape[0], dtype=np.int64)
    competitor[pair_source[first]] = pair_destination[first]
    boundary_edges[pair_source[first]] = counts[first]
    return competitor, boundary_edges


def train_atoms(
    split_first,
    split_second,
    anchor,
    competitor,
    competitor_valid,
    steps,
    learning_rate,
    contrastive_margin,
    push_weight,
    anchor_weight,
):
    atoms = torch.nn.Parameter(anchor.clone())
    optimizer = torch.optim.Adam([atoms], lr=learning_rate)
    history = []
    for step in range(steps):
        normalized = F.normalize(atoms, dim=-1)
        first_cosine = (normalized * split_first).sum(dim=-1)
        second_cosine = (normalized * split_second).sum(dim=-1)
        positive = torch.minimum(first_cosine, second_cosine)
        pull = 1.0 - 0.5 * (first_cosine + second_cosine)
        anchor_loss = 1.0 - (normalized * anchor).sum(dim=-1)
        negative = (normalized * competitor).sum(dim=-1)
        push = F.relu(negative - positive + contrastive_margin)
        push = torch.where(competitor_valid, push, torch.zeros_like(push))
        loss = pull.mean() + anchor_weight * anchor_loss.mean()
        if competitor_valid.any():
            loss = loss + push_weight * push[competitor_valid].mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            atoms.copy_(F.normalize(atoms, dim=-1))
        if step in {0, steps - 1} or (step + 1) % max(1, steps // 4) == 0:
            history.append(
                {
                    "step": step + 1,
                    "loss": float(loss.detach()),
                    "mean_pull": float(pull.mean().detach()),
                    "mean_anchor_loss": float(anchor_loss.mean().detach()),
                    "mean_push_valid": float(push[competitor_valid].mean().detach())
                    if competitor_valid.any()
                    else 0.0,
                }
            )
    return F.normalize(atoms.detach(), dim=-1), history


def atom_metrics(atoms, split_first, split_second, competitor, competitor_valid, margin):
    first = (atoms * split_first).sum(dim=-1)
    second = (atoms * split_second).sum(dim=-1)
    negative = (atoms * competitor).sum(dim=-1)
    positive = torch.minimum(first, second)
    valid_negative = negative[competitor_valid]
    valid_positive = positive[competitor_valid]
    return {
        "mean_split_atom_cosine": float(0.5 * (first.mean() + second.mean())),
        "minimum_split_atom_cosine_mean": float(positive.mean()),
        "mean_neighbor_cosine": float(valid_negative.mean())
        if valid_negative.numel()
        else 0.0,
        "mean_positive_minus_neighbor_margin": float(
            (valid_positive - valid_negative).mean()
        )
        if valid_negative.numel()
        else 0.0,
        "contrastive_margin_satisfied_fraction": float(
            ((valid_positive - valid_negative) >= margin).float().mean()
        )
        if valid_negative.numel()
        else 0.0,
    }


def nearest_ids(database, queries, device, chunk_size=8192):
    import faiss

    database = np.ascontiguousarray(database.astype(np.float32))
    queries = np.ascontiguousarray(queries.astype(np.float32))
    index = faiss.IndexFlatIP(database.shape[1])
    index.add(database)
    resources = None
    if device.startswith("cuda") and hasattr(faiss, "StandardGpuResources"):
        resources = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(resources, 0, index)
    result = np.empty(queries.shape[0], dtype=np.int64)
    for start in range(0, queries.shape[0], chunk_size):
        end = min(start + chunk_size, queries.shape[0])
        _, ids = index.search(queries[start:end], 1)
        result[start:end] = ids[:, 0]
    del resources
    return result


def assignment_metrics(atoms, split_first, split_second, competitor, valid, device):
    atoms_np = atoms.cpu().numpy().astype(np.float32)
    first_ids = nearest_ids(atoms_np, split_first.cpu().numpy(), device)
    second_ids = nearest_ids(atoms_np, split_second.cpu().numpy(), device)
    expected = np.arange(atoms_np.shape[0], dtype=np.int64)
    result = {
        "cross_split_same_code_fraction": float((first_ids == second_ids).mean()),
        "split0_identity_top1_fraction": float((first_ids == expected).mean()),
        "split1_identity_top1_fraction": float((second_ids == expected).mean()),
    }
    if valid.any():
        competitor_ids = nearest_ids(
            atoms_np,
            competitor[valid].cpu().numpy(),
            device,
        )
        result["neighbor_collision_fraction"] = float(
            (competitor_ids == expected[valid.cpu().numpy()]).mean()
        )
    else:
        result["neighbor_collision_fraction"] = 0.0
    return result


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--a20_artifact_dir", required=True)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--neighbors", type=int, default=16)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.03)
    parser.add_argument("--contrastive_margin", type=float, default=0.10)
    parser.add_argument("--push_weight", type=float, default=0.25)
    parser.add_argument("--anchor_weight", type=float, default=1.0)
    parser.add_argument("--chunk_size", type=int, default=32768)
    parser.add_argument("--knn_workers", type=int, default=4)
    parser.add_argument("--faiss_gpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    if args.neighbors <= 1 or args.steps <= 0 or args.learning_rate <= 0.0:
        raise ValueError("Atom training parameters are invalid")
    if args.contrastive_margin < 0.0 or args.push_weight < 0.0 or args.anchor_weight < 0.0:
        raise ValueError("Atom loss weights must be non-negative")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse view-invariant semantic atoms: {output_dir}")
        return

    source_dir = os.path.abspath(args.a20_artifact_dir)
    source_manifest_path = os.path.join(source_dir, "manifest.json")
    with open(source_manifest_path) as source:
        manifest = json.load(source)
    required_modalities = {"base", "part", "fine"}
    if not required_modalities.issubset(manifest.get("vocabulary_modalities", [])):
        raise ValueError("A21 requires the complete A20 base/part/fine vocabulary")

    vocabulary = np.load(
        os.path.join(source_dir, manifest["group_codebook"])
    ).astype(np.float16)
    semantic_ids = np.load(
        os.path.join(source_dir, manifest["group_semantic_code_ids"])
    ).astype(np.int64)
    semantic_invalid_source = int(manifest["semantic_invalid_id"])
    group_features = decode_group_features(
        vocabulary, semantic_ids, semantic_invalid_source
    )
    point_ids = np.load(
        os.path.join(source_dir, manifest["point_group_ids"])
    ).astype(np.int64)
    point_weights = np.load(
        os.path.join(source_dir, manifest["point_group_weights"])
    )
    point_invalid_source = int(manifest["invalid_id"])
    point_ids[point_ids == point_invalid_source] = -1
    group_reliability = np.load(
        os.path.join(source_dir, manifest["group_reliability"])
    ).astype(np.float32)
    fine_count = int(manifest["modality_token_counts"]["fine"])
    fine_start = int(semantic_ids.shape[0] - fine_count)
    fine_tokens = np.arange(fine_start, semantic_ids.shape[0], dtype=np.int64)
    if point_ids.shape[1] < 2 or fine_count <= 0:
        raise ValueError("A20 artifact does not contain fine group assignments")

    fine_to_part = np.full(fine_count, -1, dtype=np.int64)
    point_fine = point_ids[:, 1]
    point_part = point_ids[:, 0]
    fine_points = point_fine >= fine_start
    local_fine = point_fine[fine_points] - fine_start
    pairs = np.unique(
        np.stack((local_fine, point_part[fine_points]), axis=1), axis=0
    )
    if np.unique(pairs[:, 0]).size != fine_count:
        raise ValueError("Every fine atom must map to one part identity")
    counts = np.bincount(pairs[:, 0], minlength=fine_count)
    if (counts != 1).any():
        raise ValueError("A fine atom maps to multiple part identities")
    fine_to_part[pairs[:, 0]] = pairs[:, 1]

    part_to_local = np.full(semantic_ids.shape[0], -1, dtype=np.int64)
    part_to_local[fine_to_part] = np.arange(fine_count, dtype=np.int64)
    local_point_groups = np.full(point_part.shape, -1, dtype=np.int64)
    valid_part = point_part >= 0
    local_point_groups[valid_part] = part_to_local[point_part[valid_part]]

    fine_path = os.path.abspath(args.fine_consensus)
    fine_payload = torch.load(fine_path, map_location="cpu")
    split_features = fine_payload["split_initial_features"].detach().cpu()
    split_weights = fine_payload["split_weights"].detach().cpu()
    if split_features.shape[1] != point_ids.shape[0]:
        raise ValueError("Fine split consensus does not match A20")
    split_results = [
        aggregate_split_groups(
            split_features[index],
            split_weights[index],
            local_point_groups,
            fine_count,
            args.device,
            args.chunk_size,
        )
        for index in range(2)
    ]
    if not (split_results[0][3] & split_results[1][3]).all():
        raise ValueError("A20 fine atoms lost required two-split support")
    split_first = F.normalize(split_results[0][0].float(), dim=-1)
    split_second = F.normalize(split_results[1][0].float(), dim=-1)
    anchor = F.normalize(torch.from_numpy(group_features[fine_tokens]).float(), dim=-1)
    del fine_payload, split_features, split_weights

    xyz, _, _, checkpoint_iteration = load_geometry(
        args.geometry_checkpoint, point_ids.shape[0]
    )
    valid_global = np.flatnonzero(point_part >= 0)
    valid_parts = point_part[valid_global]
    valid_local = local_point_groups[valid_global]
    neighbors, _, resources, knn_backend = build_knn(
        xyz[valid_global],
        args.neighbors,
        args.chunk_size,
        args.faiss_gpu,
        args.knn_workers,
    )
    competitor_part, boundary_edges = select_competitors(
        valid_local,
        valid_parts,
        neighbors,
        anchor.numpy(),
        group_features,
    )
    del resources, neighbors, xyz
    competitor_valid_np = competitor_part >= 0
    safe_competitor = np.where(competitor_valid_np, competitor_part, 0)
    competitor = F.normalize(
        torch.from_numpy(group_features[safe_competitor]).float(), dim=-1
    )
    competitor_valid = torch.from_numpy(competitor_valid_np)

    device = torch.device(args.device)
    split_first_device = split_first.to(device)
    split_second_device = split_second.to(device)
    anchor_device = anchor.to(device)
    competitor_device = competitor.to(device)
    competitor_valid_device = competitor_valid.to(device)
    before = atom_metrics(
        anchor_device,
        split_first_device,
        split_second_device,
        competitor_device,
        competitor_valid_device,
        args.contrastive_margin,
    )
    trained, history = train_atoms(
        split_first_device,
        split_second_device,
        anchor_device,
        competitor_device,
        competitor_valid_device,
        args.steps,
        args.learning_rate,
        args.contrastive_margin,
        args.push_weight,
        args.anchor_weight,
    )
    after = atom_metrics(
        trained,
        split_first_device,
        split_second_device,
        competitor_device,
        competitor_valid_device,
        args.contrastive_margin,
    )
    before_assignment = assignment_metrics(
        anchor_device,
        split_first_device,
        split_second_device,
        competitor_device,
        competitor_valid_device,
        args.device,
    )
    after_assignment = assignment_metrics(
        trained,
        split_first_device,
        split_second_device,
        competitor_device,
        competitor_valid_device,
        args.device,
    )
    trained_np = trained.cpu().numpy().astype(np.float16)
    competitor_np = competitor[competitor_valid].numpy().astype(np.float16)

    semantic_dtype = np.uint32
    semantic_invalid = int(np.iinfo(semantic_dtype).max)
    packed_semantic = np.full(semantic_ids.shape, semantic_invalid, dtype=semantic_dtype)
    source_valid_semantic = semantic_ids != semantic_invalid_source
    packed_semantic[source_valid_semantic] = semantic_ids[source_valid_semantic].astype(
        semantic_dtype
    )
    fine_code_rows = packed_semantic[fine_tokens, 0].astype(np.int64)
    if (fine_code_rows == semantic_invalid).any():
        raise ValueError("Fine semantic atoms must use an exact first code row")
    updated_vocabulary = vocabulary.copy()
    updated_vocabulary[fine_code_rows] = trained_np

    competitor_code_offset = updated_vocabulary.shape[0]
    updated_vocabulary = np.concatenate((updated_vocabulary, competitor_np), axis=0)
    valid_competitor_local = np.flatnonzero(competitor_valid_np)
    competitor_token_offset = packed_semantic.shape[0]
    competitor_semantic = np.full(
        (valid_competitor_local.size, packed_semantic.shape[1]),
        semantic_invalid,
        dtype=semantic_dtype,
    )
    competitor_semantic[:, 0] = (
        competitor_code_offset
        + np.arange(valid_competitor_local.size, dtype=np.int64)
    ).astype(semantic_dtype)
    packed_semantic = np.concatenate((packed_semantic, competitor_semantic), axis=0)

    point_dtype = np.uint32
    point_invalid = int(np.iinfo(point_dtype).max)
    packed_point_ids = np.full(point_ids.shape, point_invalid, dtype=point_dtype)
    valid_point_ids = point_ids >= 0
    packed_point_ids[valid_point_ids] = point_ids[valid_point_ids].astype(point_dtype)
    point_competitor_ids = np.full(point_ids.shape, point_invalid, dtype=point_dtype)
    local_to_competitor_token = np.full(fine_count, -1, dtype=np.int64)
    local_to_competitor_token[valid_competitor_local] = (
        competitor_token_offset + np.arange(valid_competitor_local.size)
    )
    point_local = point_fine - fine_start
    valid_point_fine = (point_local >= 0) & (point_local < fine_count)
    competitor_point_token = np.full(point_local.shape, -1, dtype=np.int64)
    competitor_point_token[valid_point_fine] = local_to_competitor_token[
        point_local[valid_point_fine]
    ]
    valid_point_competitor = competitor_point_token >= 0
    point_competitor_ids[valid_point_competitor, 1] = competitor_point_token[
        valid_point_competitor
    ].astype(point_dtype)
    competitor_reliability = group_reliability[fine_tokens[valid_competitor_local]]
    packed_reliability = np.concatenate(
        (group_reliability, competitor_reliability), axis=0
    ).astype(np.float16)

    os.makedirs(output_dir, exist_ok=True)
    np.save(os.path.join(output_dir, "shared_vocabulary.npy"), updated_vocabulary)
    np.save(os.path.join(output_dir, "group_semantic_code_ids.npy"), packed_semantic)
    np.save(os.path.join(output_dir, "group_reliability.npy"), packed_reliability)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), packed_point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), point_weights)
    np.save(os.path.join(output_dir, "point_competitor_ids.npy"), point_competitor_ids)

    storage_bytes = int(
        updated_vocabulary.nbytes
        + packed_semantic.nbytes
        + packed_reliability.nbytes
        + packed_point_ids.nbytes
        + point_weights.nbytes
        + point_competitor_ids.nbytes
    )
    output_manifest = {
        **manifest,
        "format_version": 3,
        "method": "view_invariant_contrastive_semantic_atoms",
        "num_group_codes": int(packed_semantic.shape[0]),
        "group_codebook": "shared_vocabulary.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "semantic_invalid_id": semantic_invalid,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_competitor_ids": "point_competitor_ids.npy",
        "competitor_invalid_id": point_invalid,
        "group_reliability": "group_reliability.npy",
        "invalid_id": point_invalid,
        "id_dtype": "uint32",
        "vocabulary_modalities": ["base", "part", "fine", "competitor"],
        "modality_token_counts": {
            **manifest["modality_token_counts"],
            "competitor": int(valid_competitor_local.size),
        },
        "vocabulary": {
            **manifest.get("vocabulary", {}),
            "contrastively_updated_fine_codes": fine_count,
            "exact_competitor_codes": int(valid_competitor_local.size),
            "total_codes": int(updated_vocabulary.shape[0]),
            "construction": "A20 vocabulary with fine rows contrastively updated and exact adjacent-part competitors appended",
        },
        "atom_training": {
            "num_atoms": fine_count,
            "num_atoms_with_competitor": int(competitor_valid_np.sum()),
            "competitor_coverage": float(competitor_valid_np.mean()),
            "mean_boundary_edges": float(boundary_edges[competitor_valid_np].mean())
            if competitor_valid_np.any()
            else 0.0,
            "knn_backend": knn_backend,
            "before": {**before, **before_assignment},
            "after": {**after, **after_assignment},
            "history": history,
        },
        "continuous_discrete_contract": {
            "continuous_target": "optimized split-paired identity atom and exact adjacent-part competitor",
            "discrete_encoding": "one exact FP16 row per fine/competitor atom",
            "reconstruction_cosine": 1.0,
            "ranking_gap_source": "FP16 roundoff only",
        },
        "module_codebook_contract": {
            "enabled_modules": [
                "A14_base",
                "A18_part",
                "A20_fine_part",
                "A21_view_invariant_atom",
                "A21_competitor",
            ],
            "source_a20_manifest_sha256": sha256(source_manifest_path),
            "fine_consensus_sha256": sha256(fine_path),
            "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
            "geometry_checkpoint_iteration": checkpoint_iteration,
            "readout_slots": ["part", "fine"],
            "competitor_slot": "fine",
        },
        "storage": {
            "total_semantic_bytes": storage_bytes,
            "bytes_per_gaussian_amortized": float(
                storage_bytes / point_ids.shape[0]
            ),
        },
        "source": {
            "a20_artifact_dir": source_dir,
            "fine_consensus": fine_path,
            "leakage_control": "training splits, 3D adjacency, and fixed losses only; no evaluation queries or labels",
        },
        "args": vars(args),
    }
    with open(manifest_path, "w") as output:
        json.dump(output_manifest, output, indent=2)
    print(json.dumps(output_manifest, indent=2))


if __name__ == "__main__":
    main()
