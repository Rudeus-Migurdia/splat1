#!/usr/bin/env python
"""Apply A18 linear part-interior support only to an A24 micro-ID slot."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def apply_micro_support(point_weights, support, slot=2):
    output = np.asarray(point_weights).copy()
    support = np.asarray(support, dtype=np.float32)
    if output.ndim != 2 or not 0 <= slot < output.shape[1]:
        raise ValueError("Micro slot is outside the point weight table")
    if support.shape != (output.shape[0],):
        raise ValueError("Interior support must match the Gaussian count")
    output[:, slot] = np.rint(
        output[:, slot].astype(np.float32) * np.clip(support, 0.0, 1.0)
    ).clip(0, 255).astype(output.dtype)
    return output


def link(source, destination):
    if os.path.lexists(destination):
        os.unlink(destination)
    os.symlink(os.path.abspath(source), destination)


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact_dir", required=True)
    parser.add_argument("--part_interior_support", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--micro_slot", type=int, default=2)
    args = parser.parse_args(sys.argv[1:])

    source_dir = os.path.abspath(args.source_artifact_dir)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if "micro" not in manifest.get("vocabulary_modalities", []):
        raise ValueError("Micro interior gating requires an A24 micro vocabulary")
    point_ids = np.load(os.path.join(source_dir, manifest["point_group_ids"]))
    point_weights = np.load(
        os.path.join(source_dir, manifest["point_group_weights"])
    )
    support = np.load(os.path.abspath(args.part_interior_support)).astype(np.float32)
    output_weights = apply_micro_support(point_weights, support, args.micro_slot)
    micro_valid = point_weights[:, args.micro_slot] > 0
    changed = output_weights[:, args.micro_slot] != point_weights[:, args.micro_slot]

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for key in (
        "group_codebook",
        "group_semantic_code_ids",
        "group_reliability",
        "point_group_ids",
    ):
        name = manifest[key]
        link(os.path.join(source_dir, name), os.path.join(output_dir, name))
    np.save(os.path.join(output_dir, "point_group_weights.npy"), output_weights)

    output_manifest = dict(manifest)
    output_manifest.update(
        {
            "method": "identity_preserving_multiscale_micro_interior",
            "point_group_weights": "point_group_weights.npy",
            "weight_dtype": "uint8_identity_membership_with_linear_micro_interior",
            "micro_interior": {
                "slot": int(args.micro_slot),
                "micro_points": int(micro_valid.sum()),
                "changed_points": int(changed.sum()),
                "changed_fraction_of_micro": float(
                    changed.sum() / max(1, micro_valid.sum())
                ),
                "mean_support_on_micro": float(support[micro_valid].mean())
                if micro_valid.any()
                else 0.0,
                "mean_output_weight_on_micro": float(
                    output_weights[micro_valid, args.micro_slot].mean() / 255.0
                )
                if micro_valid.any()
                else 0.0,
            },
            "module_codebook_contract": {
                **manifest.get("module_codebook_contract", {}),
                "enabled_modules": [
                    "A14_base",
                    "A18_part",
                    "A20_fine_part",
                    "A24_multiscale_micro_identity",
                    "A24_linear_micro_interior",
                ],
                "codebook_reuse_reason": (
                    "Only the micro point-membership scalar changes; all semantic "
                    "features, token IDs, and shared-vocabulary rows are unchanged"
                ),
            },
            "storage": {
                **manifest["storage"],
                "micro_interior_point_weight_bytes": int(output_weights.nbytes),
            },
            "source": {
                **manifest.get("source", {}),
                "micro_interior_parent": source_dir,
                "part_interior_support": os.path.abspath(
                    args.part_interior_support
                ),
            },
            "args": vars(args),
        }
    )
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(output_manifest, output, indent=2)
    print(json.dumps(output_manifest["micro_interior"], indent=2))


if __name__ == "__main__":
    main()
