#!/usr/bin/env python
"""Build label-free same-level counterfactual neighborhoods for four codebooks."""

import hashlib
import json
import os
import time
from argparse import ArgumentParser

import numpy as np
import torch


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def nearest_codeword_neighbors(codebook, neighbors, chunk_size, device):
    codebook = np.asarray(codebook, dtype=np.float32)
    if codebook.ndim != 2 or codebook.shape[0] <= neighbors:
        raise ValueError("Codebook must contain more rows than requested neighbors")
    if neighbors <= 0 or chunk_size <= 0:
        raise ValueError("Neighbor and chunk counts must be positive")
    unique_codebook, representative_ids, inverse = np.unique(
        codebook,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    if unique_codebook.shape[0] <= neighbors:
        raise ValueError("Codebook must contain enough distinct semantic prototypes")
    normalized = unique_codebook / np.maximum(
        np.linalg.norm(unique_codebook, axis=1, keepdims=True), 1e-8
    )
    reference = torch.from_numpy(normalized).to(device)
    unique_ids = np.empty((normalized.shape[0], neighbors), dtype=np.uint16)
    unique_cosine = np.empty((normalized.shape[0], neighbors), dtype=np.float16)
    for start in range(0, normalized.shape[0], chunk_size):
        end = min(start + chunk_size, normalized.shape[0])
        similarity = reference[start:end] @ reference.T
        rows = torch.arange(end - start, device=reference.device)
        similarity[rows, torch.arange(start, end, device=reference.device)] = -torch.inf
        values, indices = similarity.topk(neighbors, dim=1, largest=True, sorted=True)
        unique_ids[start:end] = representative_ids[
            indices.cpu().numpy()
        ].astype(np.uint16)
        unique_cosine[start:end] = values.cpu().numpy().astype(np.float16)
    return unique_ids[inverse], unique_cosine[inverse]


def quantiles(values):
    values = np.asarray(values, dtype=np.float32)
    return {
        str(q): float(np.quantile(values, q))
        for q in (0.0, 0.1, 0.5, 0.9, 0.99, 1.0)
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--memory_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--neighbors", type=int, default=8)
    parser.add_argument("--chunk_size", type=int, default=1024)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--expected_memory_seed", type=int, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.neighbors <= 0 or args.chunk_size <= 0:
        raise ValueError("Neighbor and chunk counts must be positive")

    memory_dir = os.path.abspath(args.memory_dir)
    memory_manifest_path = os.path.join(memory_dir, "manifest.json")
    with open(memory_manifest_path) as source:
        memory = json.load(source)
    if memory.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError("A45 requires four independent hierarchical codebooks")
    if int(memory.get("resident_slots_required", 0)) != 4:
        raise ValueError("A45 requires exactly four resident token slots")
    if int(memory["reproducibility"]["seed"]) != args.expected_memory_seed:
        raise ValueError("Resident memory seed does not match the fixed A45 seed")

    output_dir = os.path.abspath(args.output_dir)
    manifest_path = os.path.join(output_dir, "manifest.json")
    if os.path.isfile(manifest_path) and not args.force:
        print(f"Reuse counterfactual codebook neighborhoods: {output_dir}")
        return
    os.makedirs(output_dir, exist_ok=True)
    started = time.time()
    level_entries = []
    total_bytes = 0
    for level_spec in memory.get("level_codebooks", []):
        level = int(level_spec["level"])
        codebook_path = os.path.join(memory_dir, level_spec["codebook"])
        codebook = np.load(codebook_path).astype(np.float32)
        unique_count = int(np.unique(codebook, axis=0).shape[0])
        ids, cosine = nearest_codeword_neighbors(
            codebook,
            args.neighbors,
            args.chunk_size,
            args.device,
        )
        ids_name = f"level_{level}_neighbor_ids.npy"
        cosine_name = f"level_{level}_neighbor_cosine.npy"
        np.save(os.path.join(output_dir, ids_name), ids)
        np.save(os.path.join(output_dir, cosine_name), cosine)
        total_bytes += ids.nbytes + cosine.nbytes
        level_entries.append(
            {
                "level": level,
                "name": level_spec["name"],
                "num_codes": int(codebook.shape[0]),
                "neighbor_ids": ids_name,
                "neighbor_cosine": cosine_name,
                "codebook_sha256": file_sha256(codebook_path),
                "unique_exact_prototypes": unique_count,
                "exact_duplicate_fraction": float(
                    1.0 - unique_count / codebook.shape[0]
                ),
                "nearest_cosine_quantiles": quantiles(cosine[:, 0]),
                "farthest_retained_cosine_quantiles": quantiles(cosine[:, -1]),
            }
        )
    if [entry["level"] for entry in level_entries] != [0, 1, 2, 3]:
        raise ValueError("A45 requires ordered L0-L3 codebooks")

    manifest = {
        "format_version": 1,
        "representation": "hierarchical_codebook_counterfactual_neighborhoods",
        "method": "same_level_semantic_nearest_prototype_counterfactuals",
        "neighbors": args.neighbors,
        "levels": level_entries,
        "memory_dir": memory_dir,
        "memory_manifest_sha256": file_sha256(memory_manifest_path),
        "source_contract": {
            "four_peer_tokens_unchanged": True,
            "same_level_neighbors_only": True,
            "self_codeword_excluded": True,
            "evaluation_queries_or_labels_used": False,
            "deterministic_exact_cosine_search": True,
        },
        "storage": {
            "total_bytes": int(total_bytes),
            "bytes_per_codeword": float(
                total_bytes / sum(entry["num_codes"] for entry in level_entries)
            ),
        },
        "args": vars(args),
        "elapsed_seconds": time.time() - started,
    }
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
