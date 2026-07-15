#!/usr/bin/env python
"""Extract selected continuous semantics as an independent sparse hypothesis."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np
import torch
from torch.nn import functional as F

from build_split_consistency_fusion import split_reliability


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--fine_consensus", required=True)
    parser.add_argument("--aux_consensus", required=True)
    parser.add_argument("--selection_payload", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--temperature", type=float, default=0.05)
    parser.add_argument("--stability_floor", type=float, default=0.0)
    args = parser.parse_args(sys.argv[1:])
    if args.temperature <= 0.0:
        raise ValueError("temperature must be positive")

    selection_payload = torch.load(args.selection_payload, map_location="cpu")
    selected = selection_payload["fine_mask"].detach().cpu().bool()
    del selection_payload

    fine = torch.load(args.fine_consensus, map_location="cpu")
    if fine["initial_features"].shape[0] != selected.numel():
        raise ValueError("Fine consensus and selection mask do not match")
    fine_reliability, fine_supported = split_reliability(
        fine["split_initial_features"],
        fine["split_weights"],
        args.stability_floor,
    )
    point_ids = torch.nonzero(selected & fine_supported, as_tuple=False).squeeze(1)
    features = F.normalize(fine["initial_features"][point_ids].float(), dim=-1)
    selected_fine_reliability = fine_reliability[point_ids]
    num_gaussians = int(selected.numel())
    feature_dim = int(features.shape[1])
    del fine, fine_reliability, fine_supported, selected

    aux = torch.load(args.aux_consensus, map_location="cpu")
    if aux["initial_features"].shape != (num_gaussians, feature_dim):
        raise ValueError("Aux consensus does not match fine consensus")
    aux_reliability, _ = split_reliability(
        aux["split_initial_features"],
        aux["split_weights"],
        args.stability_floor,
    )
    selected_aux_reliability = aux_reliability[point_ids]
    del aux, aux_reliability

    margin = torch.sigmoid(
        (selected_fine_reliability - selected_aux_reliability) / args.temperature
    )
    reliability = (selected_fine_reliability * margin).clamp(0.0, 1.0)

    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    packed_features = features.numpy().astype(np.float16)
    packed_ids = point_ids.numpy().astype(np.uint32)
    packed_reliability = np.rint(reliability.numpy() * 255.0).astype(np.uint8)
    np.save(os.path.join(output_dir, "point_ids.npy"), packed_ids)
    np.save(os.path.join(output_dir, "features.npy"), packed_features)
    np.save(os.path.join(output_dir, "reliability.npy"), packed_reliability)
    storage_bytes = int(
        packed_ids.nbytes + packed_features.nbytes + packed_reliability.nbytes
    )
    manifest = {
        "format_version": 1,
        "representation": "sparse_continuous_semantic_hypothesis",
        "num_gaussians": num_gaussians,
        "num_hypotheses": int(point_ids.numel()),
        "selected_fraction": float(point_ids.numel() / num_gaussians),
        "feature_dim": feature_dim,
        "point_ids": "point_ids.npy",
        "features": "features.npy",
        "reliability": "reliability.npy",
        "feature_dtype": "float16",
        "id_dtype": "uint32",
        "reliability_dtype": "uint8",
        "mean_reliability": float(reliability.mean()),
        "storage": {
            "point_id_bytes": int(packed_ids.nbytes),
            "feature_bytes_fp16": int(packed_features.nbytes),
            "reliability_bytes": int(packed_reliability.nbytes),
            "total_semantic_bytes": storage_bytes,
        },
        "source": {
            "fine_consensus": os.path.abspath(args.fine_consensus),
            "aux_consensus": os.path.abspath(args.aux_consensus),
            "selection_payload": os.path.abspath(args.selection_payload),
            "temperature": args.temperature,
            "stability_floor": args.stability_floor,
            "reliability": "fine_split_reliability * sigmoid((fine_rel - aux_rel) / temperature)",
        },
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
