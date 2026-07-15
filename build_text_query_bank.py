#!/usr/bin/env python
"""Build a fixed generic text-anchor bank for query-ranking supervision."""

import json
import os
from argparse import ArgumentParser

import numpy as np


# Broad visual concepts only. This list deliberately excludes LeRF-OVS category names.
GENERIC_CONCEPTS = (
    "indoor object",
    "household item",
    "small everyday item",
    "large everyday item",
    "appliance",
    "piece of furniture",
    "hand tool",
    "eating utensil",
    "storage container",
    "drinking vessel",
    "food package",
    "book or document",
    "electronic device",
    "decorative object",
    "plant or natural object",
    "wall surface",
    "floor surface",
    "tabletop surface",
    "door or panel",
    "handle or knob",
    "edge or corner",
    "flat surface",
    "curved surface",
    "round shape",
    "rectangular shape",
    "cylindrical shape",
    "thin object",
    "thick object",
    "metal material",
    "plastic material",
    "glass material",
    "ceramic material",
    "wood material",
    "paper material",
    "fabric material",
    "stone material",
    "painted material",
    "transparent material",
    "reflective material",
    "matte material",
    "smooth texture",
    "rough texture",
    "patterned texture",
    "light color",
    "dark color",
    "warm color",
    "cool color",
    "red color",
    "green color",
    "blue color",
    "yellow color",
    "black color",
    "white color",
    "gray color",
    "brown color",
    "front-facing view",
    "side-facing view",
    "top-facing view",
    "close-up detail",
    "background region",
    "foreground region",
    "occluded region",
    "shadowed region",
    "brightly lit region",
)
PROMPT_TEMPLATES = (
    "a photo of {concept}",
    "a close-up photo of {concept}",
    "a 3D scene containing {concept}",
)


def normalize(values, eps=1e-8):
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), eps)


def build_text_query_bank(output, device="cuda"):
    import torch

    from evaluation.openclip_encoder import OpenCLIPNetwork

    encoder = OpenCLIPNetwork(device)
    phrases = [
        template.format(concept=concept)
        for concept in GENERIC_CONCEPTS
        for template in PROMPT_TEMPLATES
    ]
    with torch.no_grad():
        embeddings = encoder.encode_text(phrases, device).float()
        embeddings = embeddings.reshape(
            len(GENERIC_CONCEPTS), len(PROMPT_TEMPLATES), -1
        ).mean(dim=1)
        embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
    values = embeddings.cpu().numpy().astype(np.float16)
    output = os.path.abspath(output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.save(output, values)
    summary = {
        "output": output,
        "num_queries": int(values.shape[0]),
        "feature_dim": int(values.shape[1]),
        "concepts": list(GENERIC_CONCEPTS),
        "templates": list(PROMPT_TEMPLATES),
        "source": (
            "Fixed generic OpenCLIP text concepts; no scene-specific terms or "
            "LeRF-OVS evaluation category names are used."
        ),
    }
    with open(os.path.splitext(output)[0] + "_summary.json", "w") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main():
    parser = ArgumentParser(
        description="Encode a generic non-evaluation text query bank with OpenCLIP."
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    print(json.dumps(build_text_query_bank(args.output, args.device), indent=2))


if __name__ == "__main__":
    main()
