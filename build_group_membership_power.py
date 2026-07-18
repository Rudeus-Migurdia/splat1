#!/usr/bin/env python
"""Apply a fixed power to a training-derived group membership confidence."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--source_artifact", required=True)
    parser.add_argument("--support", required=True)
    parser.add_argument("--power", type=float, default=2.0)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args(sys.argv[1:])
    if args.power <= 0.0:
        raise ValueError("Membership power must be positive")
    source_dir = os.path.abspath(args.source_artifact)
    with open(os.path.join(source_dir, "manifest.json")) as source:
        manifest = json.load(source)
    support = np.load(os.path.abspath(args.support)).astype(np.float32)
    point_ids = np.load(os.path.join(source_dir, "point_group_ids.npy"))
    point_weights = np.load(os.path.join(source_dir, "point_group_weights.npy"))
    if support.shape != (point_ids.shape[0],):
        raise ValueError("Support does not match the point table")
    powered = point_weights.copy()
    powered[:, 0] = np.rint(
        powered[:, 0].astype(np.float32) * support.clip(0.0, 1.0) ** args.power
    ).clip(0, 255).astype(np.uint8)
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
    np.save(os.path.join(output_dir, "point_group_weights.npy"), powered)
    manifest["covered_fraction"] = float((powered[:, 0] > 0).mean())
    manifest["source"] = {
        **manifest.get("source", {}),
        "membership_support": os.path.abspath(args.support),
        "membership_power": float(args.power),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({"covered_fraction": manifest["covered_fraction"], "power": args.power}, indent=2))


if __name__ == "__main__":
    main()
