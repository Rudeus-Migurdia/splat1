#!/usr/bin/env python
"""Select reliable codebook sources supported by multiview SAM tracks."""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np


def build_source_mask(codebook_dir, group_hierarchy_dir, output_path, min_weight=1):
    codebook_dir = Path(codebook_dir).resolve()
    group_hierarchy_dir = Path(group_hierarchy_dir).resolve()
    with open(codebook_dir / "manifest.json") as source:
        codebook_manifest = json.load(source)
    with open(group_hierarchy_dir / "manifest.json") as source:
        group_manifest = json.load(source)

    valid_mask = np.load(codebook_dir / codebook_manifest["valid_mask"]).astype(bool)
    point_ids = np.load(group_hierarchy_dir / group_manifest["point_group_ids"])
    point_weights = np.load(group_hierarchy_dir / group_manifest["point_group_weights"])
    if point_ids.shape != point_weights.shape or point_ids.shape[0] != valid_mask.size:
        raise ValueError("Codebook and group hierarchy must describe the same Gaussians")
    invalid_id = int(group_manifest["invalid_id"])
    group_supported = (point_ids != invalid_id) & (point_weights >= int(min_weight))
    source_mask = valid_mask & group_supported.any(axis=1)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, source_mask)
    result = {
        "method": "multiview_sam_track_source_gate",
        "codebook_dir": str(codebook_dir),
        "group_hierarchy_dir": str(group_hierarchy_dir),
        "min_weight": int(min_weight),
        "num_gaussians": int(valid_mask.size),
        "codebook_valid_count": int(valid_mask.sum()),
        "source_count": int(source_mask.sum()),
        "source_fraction": float(source_mask.mean()),
        "source_fraction_of_codebook_valid": float(
            source_mask.sum() / max(1, valid_mask.sum())
        ),
    }
    with open(output_path.with_suffix(".json"), "w") as output:
        json.dump(result, output, indent=2)
    return result


def main():
    parser = ArgumentParser(
        description="Build a source-only confidence gate from multiview SAM tracks."
    )
    parser.add_argument("--codebook_dir", required=True)
    parser.add_argument("--group_hierarchy_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--min_weight", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.min_weight <= 255:
        raise ValueError("--min_weight must be in [0, 255]")
    print(
        json.dumps(
            build_source_mask(
                args.codebook_dir,
                args.group_hierarchy_dir,
                args.output_path,
                args.min_weight,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
