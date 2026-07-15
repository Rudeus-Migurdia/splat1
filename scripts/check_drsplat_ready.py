#!/usr/bin/env python3
"""Pre-flight checks for running the Dr.Splat baseline."""

from __future__ import annotations

import argparse
import importlib
import os
from pathlib import Path
import sys


REQUIRED_IMPORTS = [
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("cv2", "opencv-python"),
    ("faiss", "faiss-cpu"),
    ("open_clip", "open-clip-torch"),
    ("segment_anything", "segment-anything"),
    ("diff_gaussian_rasterization", "submodules/langsplat-rasterization"),
    ("simple_knn._C", "submodules/simple-knn"),
]


def ok(message: str) -> None:
    print(f"[OK] {message}")


def warn(message: str) -> None:
    print(f"[WARN] {message}")


def fail(message: str) -> None:
    print(f"[FAIL] {message}")


def check_file(path: Path, label: str, required: bool = True) -> bool:
    if path.exists():
        ok(f"{label}: {path}")
        return True
    if required:
        fail(f"{label} missing: {path}")
    else:
        warn(f"{label} not found yet: {path}")
    return False


def check_imports() -> int:
    failures = 0
    for module_name, package_name in REQUIRED_IMPORTS:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "")
            suffix = f" {version}" if version else ""
            ok(f"import {module_name}{suffix}")
        except Exception as exc:
            failures += 1
            fail(f"cannot import {module_name} ({package_name}): {type(exc).__name__}: {exc}")

    try:
        import torch

        if torch.cuda.is_available():
            ok(f"CUDA visible to torch: {torch.cuda.get_device_name(0)}")
        else:
            failures += 1
            fail("torch.cuda.is_available() is False")
    except Exception:
        pass
    return failures


def check_dataset(dataset: Path, stage: str) -> int:
    failures = 0
    if not check_file(dataset, "dataset directory"):
        return 1

    images_dir = dataset / "images"
    if check_file(images_dir, "images directory"):
        image_count = len(
            [
                p
                for p in images_dir.iterdir()
                if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            ]
        )
        if image_count:
            ok(f"images found: {image_count}")
        else:
            failures += 1
            fail(f"no jpg/png images found in {images_dir}")
    else:
        failures += 1

    sparse_dir = dataset / "sparse" / "0"
    transforms_train = dataset / "transforms_train.json"
    if sparse_dir.exists():
        has_cameras = (sparse_dir / "cameras.bin").exists() or (sparse_dir / "cameras.txt").exists()
        has_images = (sparse_dir / "images.bin").exists() or (sparse_dir / "images.txt").exists()
        if has_cameras and has_images:
            ok("COLMAP cameras/images found under sparse/0")
        else:
            failures += 1
            fail("sparse/0 exists but cameras/images files are incomplete")
    elif transforms_train.exists():
        ok("Blender transforms_train.json found")
    elif "scannet" in str(dataset).lower():
        ok("ScanNet-style path detected")
    else:
        failures += 1
        fail("dataset must contain sparse/0, transforms_train.json, or a ScanNet-style path")

    language_features = dataset / "language_features"
    if stage in {"train", "all"}:
        if check_file(language_features, "language_features directory"):
            f_count = len(list(language_features.glob("*_f.npy")))
            s_count = len(list(language_features.glob("*_s.npy")))
            if f_count and s_count:
                ok(f"language feature files found: {f_count} feature, {s_count} segmentation")
            else:
                failures += 1
                fail("language_features exists but *_f.npy / *_s.npy files are missing")
        else:
            failures += 1
    elif stage in {"preprocess", "all"}:
        check_file(language_features, "language_features directory", required=False)

    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, help="COLMAP/Blender/ScanNet scene path")
    parser.add_argument("--stage", choices=["3dgs", "preprocess", "train", "all"], default="all")
    parser.add_argument("--start-checkpoint", type=Path, help="Existing 3DGS chkpnt*.pth")
    parser.add_argument("--sam-checkpoint", type=Path, default=Path("ckpts/sam_vit_h_4b8939.pth"))
    parser.add_argument("--pq-index", type=Path, default=Path("ckpts/pq_index.faiss"))
    parser.add_argument("--skip-imports", action="store_true")
    args = parser.parse_args()

    failures = 0
    print(f"Python: {sys.version.split()[0]}")
    print(f"Repo: {Path.cwd()}")

    if not args.skip_imports:
        failures += check_imports()

    if args.stage in {"preprocess", "all"}:
        failures += 0 if check_file(args.sam_checkpoint, "SAM checkpoint") else 1

    if args.stage in {"train", "all"}:
        failures += 0 if check_file(args.pq_index, "PQ index") else 1
        if args.start_checkpoint is not None:
            failures += 0 if check_file(args.start_checkpoint, "3DGS start checkpoint") else 1
        else:
            warn("3DGS start checkpoint not supplied")

    if args.dataset is not None:
        failures += check_dataset(args.dataset, args.stage)
    else:
        warn("dataset path not supplied")

    if failures:
        print(f"\nReady check failed with {failures} issue(s).")
        return 1
    print("\nReady check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
