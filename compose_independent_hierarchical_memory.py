#!/usr/bin/env python
"""Compose four peer-token levels from independent hierarchical memories."""

import json
import os
from argparse import ArgumentParser

import numpy as np


LEVEL_NAMES = ("sam_l0", "sam_l1", "sam_l2", "sam_l3")


def load_memory(path):
    path = os.path.abspath(path)
    with open(os.path.join(path, "manifest.json")) as handle:
        manifest = json.load(handle)
    if manifest.get("representation") != "hierarchical_independent_group_codebooks":
        raise ValueError(f"Unsupported source memory: {path}")
    entries = sorted(manifest.get("level_codebooks", []), key=lambda item: item["level"])
    if [int(item["level"]) for item in entries] != [0, 1, 2, 3]:
        raise ValueError(f"Source memory does not provide L0--L3: {path}")
    point_ids = np.load(os.path.join(path, manifest["point_group_ids"])).astype(
        np.int64
    )
    invalid = int(manifest["invalid_id"])
    if point_ids.shape[1] != 4 or np.any(point_ids == invalid):
        raise ValueError("Every source Gaussian must contain four resident IDs")
    semantic_ids = np.load(
        os.path.join(path, manifest["group_semantic_code_ids"])
    ).astype(np.int64)
    levels = np.load(os.path.join(path, manifest["group_level"])).astype(np.int64)
    if semantic_ids.shape != (levels.size, 1):
        raise ValueError("Source semantic IDs and levels are inconsistent")

    def optional(name, default, dtype):
        filename = manifest.get(name)
        if filename:
            return np.load(os.path.join(path, filename)).astype(dtype)
        return np.full(point_ids.shape, default, dtype=dtype)

    return {
        "path": path,
        "manifest": manifest,
        "entries": entries,
        "point_ids": point_ids,
        "semantic_ids": semantic_ids[:, 0],
        "levels": levels,
        "point_weights": optional("point_group_weights", 255, np.uint8),
        "point_reliability": optional("point_group_reliability", 1.0, np.float16),
        "point_source": optional("point_group_source", 255, np.uint8),
        "group_reliability": np.load(
            os.path.join(path, manifest["group_reliability"])
        ).astype(np.float16),
        "group_source": np.load(
            os.path.join(path, manifest["group_source"])
        ).astype(np.uint8),
    }


def compose_memories(source_paths, output_dir, seed):
    if len(source_paths) != 4:
        raise ValueError("Exactly one source memory is required for each of L0--L3")
    sources_by_path = {path: load_memory(path) for path in set(source_paths)}
    sources = [sources_by_path[path] for path in source_paths]
    num_gaussians = sources[0]["point_ids"].shape[0]
    feature_dim = int(sources[0]["manifest"]["feature_dim"])
    if any(source["point_ids"].shape[0] != num_gaussians for source in sources):
        raise ValueError("All source memories must describe the same Gaussians")
    if any(int(source["manifest"]["feature_dim"]) != feature_dim for source in sources):
        raise ValueError("All source memories must use the same feature dimension")

    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    codebooks = []
    local_point_ids = []
    point_weights = []
    point_reliability = []
    point_source = []
    group_reliability = []
    group_source = []
    level_entries = []
    offset = 0
    for level, (name, source) in enumerate(zip(LEVEL_NAMES, sources)):
        entry = source["entries"][level]
        codebook = np.load(os.path.join(source["path"], entry["codebook"]))
        if codebook.shape[1] != feature_dim:
            raise ValueError("Source level codebook has an invalid feature dimension")
        source_groups = source["point_ids"][:, level]
        if source_groups.max() >= source["semantic_ids"].size:
            raise ValueError("Source point IDs exceed the semantic token table")
        local_ids = source["semantic_ids"][source_groups]
        if local_ids.min() < 0 or local_ids.max() >= codebook.shape[0]:
            raise ValueError("Source point IDs do not resolve inside the selected level")
        level_mask = source["levels"] == level
        if int(level_mask.sum()) != codebook.shape[0]:
            raise ValueError("Selected source level is not a one-row-per-code vocabulary")

        filename = f"{name}_codebook.npy"
        np.save(os.path.join(output_dir, filename), codebook)
        codebooks.append(codebook)
        local_point_ids.append(local_ids)
        point_weights.append(source["point_weights"][:, level])
        point_reliability.append(source["point_reliability"][:, level])
        point_source.append(source["point_source"][:, level])
        group_reliability.append(source["group_reliability"][level_mask])
        group_source.append(source["group_source"][level_mask])
        level_entries.append(
            {
                "name": name,
                "level": level,
                "semantic_role": ("coarse", "object", "part", "micro")[level],
                "codebook": filename,
                "num_codes": int(codebook.shape[0]),
                "group_token_start": offset,
                "group_token_end": offset + int(codebook.shape[0]),
                "quantization": entry.get("quantization", "source_level_codebook"),
                "composed_from": source["path"],
            }
        )
        offset += int(codebook.shape[0])

    point_dtype = np.uint16 if offset < np.iinfo(np.uint16).max else np.uint32
    invalid_id = np.iinfo(point_dtype).max
    packed_point_ids = np.stack(
        [
            local_point_ids[level] + level_entries[level]["group_token_start"]
            for level in range(4)
        ],
        axis=1,
    ).astype(point_dtype)
    packed_weights = np.stack(point_weights, axis=1).astype(np.uint8)
    packed_reliability = np.stack(point_reliability, axis=1).astype(np.float16)
    packed_source = np.stack(point_source, axis=1).astype(np.uint8)
    semantic_invalid_id = np.iinfo(np.uint16).max
    semantic_ids = np.concatenate(
        [np.arange(codebook.shape[0], dtype=np.uint16) for codebook in codebooks]
    )[:, None]
    group_levels = np.concatenate(
        [
            np.full(codebook.shape[0], level, dtype=np.uint8)
            for level, codebook in enumerate(codebooks)
        ]
    )
    packed_group_reliability = np.concatenate(group_reliability).astype(np.float16)
    packed_group_source = np.concatenate(group_source).astype(np.uint8)
    group_parents = np.full(offset, -1, dtype=np.int32)

    arrays = {
        "group_semantic_code_ids.npy": semantic_ids,
        "group_level.npy": group_levels,
        "group_reliability.npy": packed_group_reliability,
        "group_source.npy": packed_group_source,
        "group_parent_ids.npy": group_parents,
        "point_group_ids.npy": packed_point_ids,
        "point_group_weights.npy": packed_weights,
        "point_group_reliability.npy": packed_reliability,
        "point_group_source.npy": packed_source,
    }
    for filename, values in arrays.items():
        np.save(os.path.join(output_dir, filename), values)
    storage_bytes = sum(value.nbytes for value in arrays.values()) + sum(
        codebook.nbytes for codebook in codebooks
    )
    usable = packed_reliability > 0.0
    manifest = {
        "format_version": 3,
        "representation": "hierarchical_independent_group_codebooks",
        "method": "a41_independent_level_memory_composition",
        "feature_dim": feature_dim,
        "num_gaussians": num_gaussians,
        "num_groups": offset,
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "group_source": "group_source.npy",
        "group_parent_ids": "group_parent_ids.npy",
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_reliability": "point_group_reliability.npy",
        "point_group_source": "point_group_source.npy",
        "invalid_id": int(invalid_id),
        "semantic_invalid_id": int(semantic_invalid_id),
        "id_dtype": str(packed_point_ids.dtype),
        "resident_slots_required": 4,
        "top_m": 4,
        "covered_fraction": 1.0,
        "usable_slot_fraction": float(usable.mean()),
        "usable_covered_fraction": float(usable.any(axis=1).mean()),
        "vocabulary_modalities": ["base", *LEVEL_NAMES],
        "modality_token_counts": {
            name: int(codebook.shape[0]) for name, codebook in zip(LEVEL_NAMES, codebooks)
        },
        "level_codebooks": level_entries,
        "codebook": {
            "layout": "four independent source-level codebooks",
            "query_readout": "four peer tokens with no parent preference or fixed level priority",
        },
        "storage": {
            "total_semantic_bytes": int(storage_bytes),
            "bytes_per_gaussian": float(storage_bytes / num_gaussians),
        },
        "source": {
            "level_memory_dirs": [source["path"] for source in sources],
            "composition": "whole-level codebook, assignment, reliability, and source fields",
            "leakage_control": "pretrained source artifacts only; no labels, text queries, or evaluation metrics",
        },
        "reproducibility": {"seed": int(seed), "deterministic_composition": True},
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--level_memory_dirs", nargs=4, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=20260719)
    args = parser.parse_args()
    manifest = compose_memories(args.level_memory_dirs, args.output_dir, args.seed)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
