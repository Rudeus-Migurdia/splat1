#!/usr/bin/env python
"""Replace a part attachment with its sparse boundary code where available."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from validate_semantic_vocabulary_contract import validate_contract


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(sys.argv[1:])
    source_dir = os.path.abspath(args.source_artifact)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    point_ids = np.load(os.path.join(source_dir, manifest["point_group_ids"]))
    point_weights = np.load(os.path.join(source_dir, manifest["point_group_weights"]))
    boundary = point_weights[:, 1] > 0
    replaced_weights = point_weights.copy()
    replaced_weights[boundary, 0] = 0
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    for name in (
        "shared_vocabulary.npy",
        "group_semantic_code_ids.npy",
        "group_reliability.npy",
    ):
        destination = os.path.join(output_dir, name)
        if os.path.lexists(destination):
            os.unlink(destination)
        os.symlink(os.path.join(source_dir, name), destination)
    np.save(os.path.join(output_dir, "point_group_ids.npy"), point_ids)
    np.save(os.path.join(output_dir, "point_group_weights.npy"), replaced_weights)
    manifest["covered_fraction"] = float((replaced_weights > 0).any(axis=1).mean())
    manifest["mean_ids_per_covered_gaussian"] = float(
        (replaced_weights[(replaced_weights > 0).any(axis=1)] > 0).sum(axis=1).mean()
    )
    manifest["boundary"] = {
        **manifest["boundary"],
        "readout": "replace_part_attachment_on_reproducible_boundary_mode",
        "num_replaced_part_attachments": int(boundary.sum()),
    }
    manifest["source"] = {
        **manifest["source"],
        "replacement_parent": source_dir,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    result = validate_contract(output_dir, ["base", "part", "boundary"])
    print(json.dumps({"replaced": int(boundary.sum()), "contract": result}, indent=2))


if __name__ == "__main__":
    main()
