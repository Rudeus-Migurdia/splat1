#!/usr/bin/env python
"""Validate that every enabled semantic module is represented in the vocabulary."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def validate_independent_hierarchical_memory(manifest, artifact_dir, required_modalities):
    declared = set(manifest.get("vocabulary_modalities", []))
    missing = set(required_modalities).difference(declared)
    if missing:
        raise ValueError(f"Vocabulary is missing enabled modalities: {sorted(missing)}")
    semantic_ids = np.load(
        os.path.join(artifact_dir, manifest["group_semantic_code_ids"]), mmap_mode="r"
    )
    levels = np.load(
        os.path.join(artifact_dir, manifest["group_level"]), mmap_mode="r"
    )
    if semantic_ids.ndim != 2 or semantic_ids.shape[1] != 1:
        raise ValueError("Hierarchical semantic IDs must have shape [G, 1]")
    if levels.shape != (semantic_ids.shape[0],):
        raise ValueError("Hierarchical group levels must match the semantic token table")
    level_codebooks = manifest.get("level_codebooks", [])
    if len(level_codebooks) != 4:
        raise ValueError("Hierarchical semantic memory must provide four level codebooks")
    invalid = int(manifest["semantic_invalid_id"])
    total_codes = 0
    for spec in level_codebooks:
        level = int(spec["level"])
        level_mask = levels == level
        codebook = np.load(
            os.path.join(artifact_dir, spec["codebook"]), mmap_mode="r"
        )
        local_ids = semantic_ids[level_mask, 0]
        if (local_ids == invalid).any():
            raise ValueError("Resident hierarchical token is missing its local semantic ID")
        if local_ids.size and (
            int(local_ids.min()) < 0 or int(local_ids.max()) >= codebook.shape[0]
        ):
            raise ValueError("Hierarchical semantic ID exceeds its declared level codebook")
        if int(spec["num_codes"]) != codebook.shape[0]:
            raise ValueError("Declared level codebook size does not match its file")
        total_codes += int(codebook.shape[0])
    point_ids = np.load(
        os.path.join(artifact_dir, manifest["point_group_ids"]), mmap_mode="r"
    )
    point_invalid = int(manifest["invalid_id"])
    point_valid = point_ids != point_invalid
    if point_ids.shape[1] != 4:
        raise ValueError("Hierarchical semantic memory requires four resident point-token slots")
    if point_valid.any() and int(point_ids[point_valid].max()) >= semantic_ids.shape[0]:
        raise ValueError("Gaussian group IDs exceed the hierarchical semantic token table")
    point_reliability_name = manifest.get("point_group_reliability")
    if point_reliability_name:
        point_reliability = np.load(
            os.path.join(artifact_dir, point_reliability_name), mmap_mode="r"
        )
        if point_reliability.shape != point_ids.shape:
            raise ValueError("Point reliability must match four resident point-token slots")
        if not np.isfinite(point_reliability).all() or (
            (point_reliability < 0.0) | (point_reliability > 1.0)
        ).any():
            raise ValueError("Point reliability must lie in [0, 1]")
    required_slots = int(manifest.get("resident_slots_required", 0))
    if required_slots:
        if required_slots != point_ids.shape[1]:
            raise ValueError("resident_slots_required must match point-token width")
        if not point_valid.all():
            raise ValueError("Fixed resident hierarchy requires one token ID in every slot")
    counts = manifest.get("modality_token_counts", {})
    for modality in required_modalities:
        if modality != "base" and int(counts.get(modality, 0)) <= 0:
            raise ValueError(f"Vocabulary modality has no tokens: {modality}")
    return {
        "artifact_dir": artifact_dir,
        "required_modalities": list(required_modalities),
        "declared_modalities": sorted(declared),
        "num_vocabulary_codes": total_codes,
        "num_semantic_tokens": int(semantic_ids.shape[0]),
        "num_gaussians": int(point_ids.shape[0]),
        "has_competitor_ids": False,
        "has_semantic_atom_ids": False,
        "independent_level_codebooks": True,
        "resident_slot_fraction": float(point_valid.mean()),
    }


def validate_contract(artifact_dir, required_modalities):
    artifact_dir = os.path.abspath(artifact_dir)
    with open(os.path.join(artifact_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("representation") == "hierarchical_independent_group_codebooks":
        return validate_independent_hierarchical_memory(
            manifest, artifact_dir, required_modalities
        )
    declared = set(manifest.get("vocabulary_modalities", []))
    missing = set(required_modalities).difference(declared)
    if missing:
        raise ValueError(f"Vocabulary is missing enabled modalities: {sorted(missing)}")
    codebook = np.load(os.path.join(artifact_dir, manifest["group_codebook"]), mmap_mode="r")
    semantic_ids = np.load(
        os.path.join(artifact_dir, manifest["group_semantic_code_ids"]), mmap_mode="r"
    )
    invalid = int(manifest["semantic_invalid_id"])
    valid = semantic_ids != invalid
    if valid.any() and int(semantic_ids[valid].max()) >= codebook.shape[0]:
        raise ValueError("Semantic module IDs exceed the declared shared vocabulary")
    atom_name = manifest.get("group_semantic_atom_code_ids")
    if "semantic_atom" in required_modalities and not atom_name:
        raise ValueError("Semantic atom modality requires group_semantic_atom_code_ids")
    if atom_name:
        atom_ids = np.load(
            os.path.join(artifact_dir, atom_name), mmap_mode="r"
        )
        atom_invalid = int(
            manifest.get("semantic_atom_invalid_id", invalid)
        )
        atom_valid = atom_ids != atom_invalid
        if atom_ids.shape != semantic_ids.shape:
            raise ValueError("Semantic atom IDs must match semantic token IDs")
        if atom_valid.any() and int(atom_ids[atom_valid].max()) >= codebook.shape[0]:
            raise ValueError("Semantic atom IDs exceed the declared shared vocabulary")
    point_ids = np.load(
        os.path.join(artifact_dir, manifest["point_group_ids"]), mmap_mode="r"
    )
    point_invalid = int(manifest["invalid_id"])
    point_valid = point_ids != point_invalid
    if point_valid.any() and int(point_ids[point_valid].max()) >= semantic_ids.shape[0]:
        raise ValueError("Gaussian group IDs exceed the semantic token table")
    if int(manifest.get("num_gaussians", point_ids.shape[0])) != point_ids.shape[0]:
        raise ValueError("Manifest Gaussian count does not match the point ID table")
    competitor_name = manifest.get("point_competitor_ids")
    if "competitor" in required_modalities and not competitor_name:
        raise ValueError("Competitor modality requires point_competitor_ids")
    if competitor_name:
        competitor_ids = np.load(
            os.path.join(artifact_dir, competitor_name), mmap_mode="r"
        )
        competitor_invalid = int(
            manifest.get("competitor_invalid_id", point_invalid)
        )
        competitor_valid = competitor_ids != competitor_invalid
        if competitor_ids.shape != point_ids.shape:
            raise ValueError("Competitor ID table must match point group IDs")
        if competitor_valid.any() and int(competitor_ids[competitor_valid].max()) >= semantic_ids.shape[0]:
            raise ValueError("Competitor group IDs exceed the semantic token table")
    coverage = manifest.get("modality_token_counts", {})
    for modality in required_modalities:
        if modality != "base" and int(coverage.get(modality, 0)) <= 0:
            raise ValueError(f"Vocabulary modality has no tokens: {modality}")
    return {
        "artifact_dir": artifact_dir,
        "required_modalities": list(required_modalities),
        "declared_modalities": sorted(declared),
        "num_vocabulary_codes": int(codebook.shape[0]),
        "num_semantic_tokens": int(semantic_ids.shape[0]),
        "num_gaussians": int(point_ids.shape[0]),
        "has_competitor_ids": bool(competitor_name),
        "has_semantic_atom_ids": bool(atom_name),
    }


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--artifact_dir", required=True)
    parser.add_argument("--required", nargs="+", required=True)
    args = parser.parse_args(sys.argv[1:])
    result = validate_contract(args.artifact_dir, args.required)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
