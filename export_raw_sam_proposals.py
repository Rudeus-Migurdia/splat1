#!/usr/bin/env python
"""Export overlapping pre-flatten SAM proposals and CLIP descriptors."""

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

import open_clip


LEVEL_NAMES = ("default", "s", "m", "l")


def manifest_view_names(path):
    if not path:
        return None
    with open(path) as source:
        manifest = json.load(source)
    names = [str(item["image_name"]) for item in manifest["views"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("View manifest must contain unique training image names")
    return set(names)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def image_resolution(image, requested_resolution):
    height, width = image.shape[:2]
    if requested_resolution == -1:
        scale = max(height / 1080.0, 1.0)
    else:
        scale = width / float(requested_resolution)
    return int(width / scale), int(height / scale)


def masked_tiles(image, masks):
    tiles = []
    for mask in masks:
        segmentation = mask["segmentation"]
        x, y, width, height = np.asarray(mask["bbox"], dtype=np.int32)
        x = max(x, 0)
        y = max(y, 0)
        width = max(width, 1)
        height = max(height, 1)
        crop = image[y : y + height, x : x + width].copy()
        crop_mask = segmentation[y : y + height, x : x + width]
        crop[~crop_mask] = 0
        side = max(crop.shape[0], crop.shape[1])
        padded = np.zeros((side, side, 3), dtype=np.uint8)
        y0 = (side - crop.shape[0]) // 2
        x0 = (side - crop.shape[1]) // 2
        padded[y0 : y0 + crop.shape[0], x0 : x0 + crop.shape[1]] = crop
        tiles.append(cv2.resize(padded, (224, 224), interpolation=cv2.INTER_LINEAR))
    if not tiles:
        return torch.empty((0, 3, 224, 224), dtype=torch.float32)
    return torch.from_numpy(np.stack(tiles)).permute(0, 3, 1, 2).float() / 255.0


class ClipDescriptorEncoder:
    def __init__(self, checkpoint):
        self.preprocess = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((224, 224)),
                torchvision.transforms.Normalize(
                    mean=[0.48145466, 0.4578275, 0.40821073],
                    std=[0.26862954, 0.26130258, 0.27577711],
                ),
            ]
        )
        self.model, _, _ = open_clip.create_model_and_transforms(
            "ViT-B-16", pretrained=checkpoint, precision="fp16"
        )
        self.model = self.model.eval().cuda()

    @torch.no_grad()
    def encode(self, tiles, batch_size):
        outputs = []
        for start in range(0, tiles.shape[0], batch_size):
            batch = self.preprocess(tiles[start : start + batch_size]).half().cuda()
            descriptor = self.model.encode_image(batch)
            descriptor = torch.nn.functional.normalize(descriptor.float(), dim=-1)
            outputs.append(descriptor.cpu().half())
        if not outputs:
            return np.empty((0, 512), dtype=np.float16)
        return torch.cat(outputs).numpy()


def atomic_savez(path, **arrays):
    temporary = path + ".tmp"
    with open(temporary, "wb") as output:
        np.savez_compressed(output, **arrays)
    os.replace(temporary, path)


def export_view(image_path, output_path, mask_generator, clip_encoder, resolution, batch_size):
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    width, height = image_resolution(bgr, resolution)
    bgr = cv2.resize(bgr, (width, height), interpolation=cv2.INTER_AREA)
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    proposal_levels = mask_generator.generate(image)

    all_masks = []
    all_levels = []
    for level, proposals in enumerate(proposal_levels):
        all_masks.extend(proposals)
        all_levels.extend([level] * len(proposals))
    if not all_masks:
        raise RuntimeError(f"SAM returned no proposals for {image_path.name}")

    segmentations = np.stack([item["segmentation"] for item in all_masks]).astype(bool)
    packed = np.packbits(segmentations.reshape(len(all_masks), -1), axis=1, bitorder="little")
    tiles = masked_tiles(image, all_masks)
    descriptors = clip_encoder.encode(tiles, batch_size)
    predicted_iou = np.asarray([item["predicted_iou"] for item in all_masks], np.float32)
    stability = np.asarray([item["stability_score"] for item in all_masks], np.float32)
    atomic_savez(
        output_path,
        packed_masks=packed,
        descriptors=descriptors,
        levels=np.asarray(all_levels, dtype=np.uint8),
        predicted_iou=predicted_iou,
        stability_score=stability,
        quality_score=predicted_iou * stability,
        area=np.asarray([item["area"] for item in all_masks], dtype=np.int32),
        bbox=np.asarray([item["bbox"] for item in all_masks], dtype=np.float32),
        point_coords=np.asarray(
            [item["point_coords"][0] for item in all_masks], dtype=np.float32
        ),
        image_height=np.asarray(height, dtype=np.int32),
        image_width=np.asarray(width, dtype=np.int32),
    )
    return {
        "image_name": image_path.stem,
        "file": os.path.relpath(output_path, os.path.dirname(os.path.dirname(output_path))),
        "num_proposals": len(all_masks),
        "per_level": [len(level) for level in proposal_levels],
        "height": height,
        "width": width,
        "sha256": file_sha256(output_path),
    }


def finalize_manifest(args):
    output_dir = os.path.abspath(args.output_dir)
    shard_paths = sorted(Path(output_dir).glob("shard_*.json"))
    if not shard_paths:
        raise FileNotFoundError(f"No shard manifests in {output_dir}")
    entries = []
    contracts = []
    for path in shard_paths:
        with open(path) as source:
            payload = json.load(source)
        entries.extend(payload["views"])
        contracts.append(payload["contract"])
    entries.sort(key=lambda item: item["image_name"])
    names = [item["image_name"] for item in entries]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate images across raw-proposal shards")
    requested = manifest_view_names(args.view_manifest)
    expected = sorted(
        path.stem
        for path in Path(args.dataset_path, "images").iterdir()
        if path.is_file() and (requested is None or path.stem in requested)
    )
    if names != expected:
        missing = sorted(set(expected) - set(names))
        extra = sorted(set(names) - set(expected))
        raise ValueError(f"Raw proposal set is incomplete: missing={missing}, extra={extra}")
    if any(contract != contracts[0] for contract in contracts[1:]):
        raise ValueError("Raw-proposal shard contracts do not match")
    manifest = {
        "format_version": 1,
        "representation": "overlapping_pre_flatten_sam_proposals",
        "scene": Path(args.dataset_path).name,
        "views": entries,
        "num_views": len(entries),
        "num_proposals": int(sum(item["num_proposals"] for item in entries)),
        "contract": contracts[0],
    }
    manifest_path = os.path.join(output_dir, "manifest.json")
    with open(manifest_path, "w") as output:
        json.dump(manifest, output, indent=2)
    print(json.dumps({k: manifest[k] for k in ("scene", "num_views", "num_proposals")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--clip_checkpoint", required=True)
    parser.add_argument("--view_manifest", default=None)
    parser.add_argument("--resolution", type=int, default=-1)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--clip_batch_size", type=int, default=64)
    parser.add_argument("--finalize", action="store_true")
    args = parser.parse_args()
    if args.finalize:
        finalize_manifest(args)
        return
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")
    if args.clip_batch_size <= 0:
        raise ValueError("clip_batch_size must be positive")

    seed_everything(args.seed)
    output_dir = os.path.abspath(args.output_dir)
    view_dir = os.path.join(output_dir, "views")
    os.makedirs(view_dir, exist_ok=True)
    requested = manifest_view_names(args.view_manifest)
    images = sorted(
        path
        for path in Path(args.dataset_path, "images").iterdir()
        if path.is_file() and (requested is None or path.stem in requested)
    )
    if requested is not None and {path.stem for path in images} != requested:
        missing = sorted(requested - {path.stem for path in images})
        raise ValueError(f"Training-view manifest references missing images: {missing}")
    images = [path for index, path in enumerate(images) if index % args.num_shards == args.shard_index]

    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint).cuda().eval()
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,
        pred_iou_thresh=0.7,
        box_nms_thresh=0.7,
        stability_score_thresh=0.85,
        crop_n_layers=1,
        crop_n_points_downscale_factor=1,
        min_mask_region_area=100,
        output_mode="binary_mask",
    )
    clip_encoder = ClipDescriptorEncoder(args.clip_checkpoint)
    started = time.time()
    entries = []
    for index, image_path in enumerate(images):
        output_path = os.path.join(view_dir, image_path.stem + ".npz")
        if os.path.isfile(output_path):
            with np.load(output_path) as payload:
                entry = {
                    "image_name": image_path.stem,
                    "file": os.path.relpath(output_path, output_dir),
                    "num_proposals": int(payload["levels"].shape[0]),
                    "per_level": np.bincount(payload["levels"], minlength=4).astype(int).tolist(),
                    "height": int(payload["image_height"]),
                    "width": int(payload["image_width"]),
                    "sha256": file_sha256(output_path),
                }
        else:
            entry = export_view(
                image_path, output_path, mask_generator, clip_encoder, args.resolution, args.clip_batch_size
            )
        entries.append(entry)
        print(
            f"[{index + 1}/{len(images)}] {entry['image_name']} proposals={entry['num_proposals']} "
            f"levels={entry['per_level']}",
            flush=True,
        )
    contract = {
        "seed": args.seed,
        "sam_checkpoint": os.path.abspath(args.sam_checkpoint),
        "sam_checkpoint_sha256": file_sha256(args.sam_checkpoint),
        "clip_checkpoint": os.path.abspath(args.clip_checkpoint),
        "clip_checkpoint_sha256": file_sha256(args.clip_checkpoint),
        "levels": list(LEVEL_NAMES),
        "custom_flattening_or_mask_nms_applied": False,
        "overlap_preserved_within_and_across_levels": True,
        "evaluation_queries_or_labels_used": False,
        "codebooks_trained": False,
        "resolution": args.resolution,
        "view_manifest": os.path.abspath(args.view_manifest)
        if args.view_manifest
        else None,
        "view_manifest_sha256": file_sha256(args.view_manifest)
        if args.view_manifest
        else None,
    }
    shard_manifest = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "views": entries,
        "contract": contract,
        "elapsed_seconds": time.time() - started,
    }
    with open(os.path.join(output_dir, f"shard_{args.shard_index:02d}.json"), "w") as output:
        json.dump(shard_manifest, output, indent=2)


if __name__ == "__main__":
    main()
