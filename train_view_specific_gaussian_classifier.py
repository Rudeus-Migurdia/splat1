#!/usr/bin/env python
"""Train a Proto-SaGa-style shared Gaussian latent with per-view classifiers."""

import json
import os
import sys
from argparse import ArgumentParser

import numpy as np

from build_multiview_mask_track_hierarchy import load_cache, normalize


def random_orthogonal_projection(input_dim, output_dim, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal((input_dim, output_dim)).astype(np.float32)
    projection, _ = np.linalg.qr(values)
    return projection[:, :output_dim].astype(np.float32)


def initialize_gaussian_latent(codebook_dir, latent_dim, seed, chunk_size=65536):
    with open(os.path.join(codebook_dir, "manifest.json")) as source:
        manifest = json.load(source)
    if manifest.get("representation") != "gaussian_multilevel_residual_codebook":
        raise ValueError("Initialization requires a residual Gaussian codebook")
    feature_dim = int(manifest["feature_dim"])
    projection = random_orthogonal_projection(feature_dim, latent_dim, seed)
    projected_codebooks = [
        np.load(os.path.join(codebook_dir, name)).astype(np.float32) @ projection
        for name in manifest["codebook_files"]
    ]
    point_ids = np.load(
        os.path.join(codebook_dir, manifest["point_code_ids"]), mmap_mode="r"
    )
    valid_mask = np.load(
        os.path.join(codebook_dir, manifest["valid_mask"]), mmap_mode="r"
    )
    invalid_id = int(manifest["invalid_id"])
    output = np.empty((point_ids.shape[0], latent_dim), dtype=np.float32)
    rng = np.random.default_rng(seed + 1)
    for start in range(0, point_ids.shape[0], chunk_size):
        end = min(start + chunk_size, point_ids.shape[0])
        ids = np.asarray(point_ids[start:end], dtype=np.int64)
        valid = np.array(valid_mask[start:end], dtype=bool, copy=True)
        valid &= np.all(ids != invalid_id, axis=1)
        values = rng.standard_normal((end - start, latent_dim)).astype(np.float32)
        values[valid] = 0.0
        for level, codebook in enumerate(projected_codebooks):
            values[valid] += codebook[ids[valid, level]]
        output[start:end] = normalize(values)
    return output, projection


def initialize_view_classifiers(cache_dir, entries, projection):
    offsets = [0]
    classifiers = []
    for entry in entries:
        cache = load_cache(cache_dir, entry)
        features = normalize(cache["feature_latents"].numpy())
        classifiers.append(normalize(features @ projection))
        offsets.append(offsets[-1] + features.shape[0])
    return np.concatenate(classifiers, axis=0).astype(np.float32), offsets


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--cache_dir", required=True)
    parser.add_argument("--codebook_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=10.0)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--pixels_per_step", type=int, default=2048)
    parser.add_argument("--steps_per_loaded_view", type=int, default=4)
    parser.add_argument("--feature_lr", type=float, default=2.5e-3)
    parser.add_argument("--classifier_lr", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--save_every", type=int, default=0)
    args = parser.parse_args(sys.argv[1:])
    if (
        args.latent_dim <= 0
        or args.temperature <= 0.0
        or args.iterations <= 0
        or args.pixels_per_step <= 0
        or args.steps_per_loaded_view <= 0
    ):
        raise ValueError("Training dimensions, temperature, and iteration counts must be positive")

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cache_dir = os.path.abspath(args.cache_dir)
    codebook_dir = os.path.abspath(args.codebook_dir)
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "manifest.json")) as source:
        cache_manifest = json.load(source)
    entries = cache_manifest["views"]
    num_gaussians = int(cache_manifest["num_gaussians"])

    initial_features, projection = initialize_gaussian_latent(
        codebook_dir, args.latent_dim, args.seed
    )
    initial_classifiers, view_offsets = initialize_view_classifiers(
        cache_dir, entries, projection
    )
    if initial_features.shape[0] != num_gaussians:
        raise ValueError("Codebook and observation cache use different Gaussian counts")

    embedding = torch.nn.Embedding.from_pretrained(
        torch.from_numpy(initial_features),
        freeze=False,
        sparse=True,
    ).cuda()
    classifiers = torch.nn.Parameter(torch.from_numpy(initial_classifiers).cuda())
    feature_optimizer = torch.optim.SparseAdam(
        [embedding.weight], lr=args.feature_lr
    )
    classifier_optimizer = torch.optim.Adam(
        [classifiers], lr=args.classifier_lr
    )
    rng = np.random.default_rng(args.seed)
    order = rng.permutation(len(entries))
    order_position = 0
    iteration = 0
    recent_losses = []

    while iteration < args.iterations:
        if order_position == len(order):
            order = rng.permutation(len(entries))
            order_position = 0
        view_index = int(order[order_position])
        order_position += 1
        cache = load_cache(cache_dir, entries[view_index])
        point_ids_np = cache["point_ids"].numpy().astype(np.int64, copy=False)
        point_weights_np = cache["point_weights"].numpy().astype(np.float32, copy=False)
        segment_ids_np = cache["segment_ids"].numpy().astype(np.int64, copy=False)
        valid_pixels = np.flatnonzero(segment_ids_np >= 0)
        classifier_start, classifier_end = view_offsets[view_index : view_index + 2]

        for _ in range(args.steps_per_loaded_view):
            if iteration >= args.iterations:
                break
            sample_size = min(args.pixels_per_step, valid_pixels.size)
            sample = rng.choice(valid_pixels, sample_size, replace=False)
            point_ids = torch.from_numpy(point_ids_np[sample].copy()).long().cuda()
            point_weights = torch.from_numpy(point_weights_np[sample].copy()).cuda()
            labels = torch.from_numpy(segment_ids_np[sample].copy()).long().cuda()
            contributor_valid = point_ids >= 0
            safe_ids = point_ids.clamp_min(0)

            feature_optimizer.zero_grad(set_to_none=True)
            classifier_optimizer.zero_grad(set_to_none=True)
            gaussian_features = torch.nn.functional.normalize(
                embedding(safe_ids), dim=-1
            )
            rendered = (
                gaussian_features
                * point_weights.unsqueeze(-1)
                * contributor_valid.unsqueeze(-1)
            ).sum(dim=1)
            rendered = torch.nn.functional.normalize(rendered, dim=-1)
            view_classifiers = torch.nn.functional.normalize(
                classifiers[classifier_start:classifier_end], dim=-1
            )
            logits = args.temperature * rendered @ view_classifiers.T
            loss = torch.nn.functional.cross_entropy(logits, labels)
            loss.backward()
            feature_optimizer.step()
            classifier_optimizer.step()
            recent_losses.append(float(loss.detach()))
            iteration += 1
            if iteration % 100 == 0 or iteration == args.iterations:
                print(
                    json.dumps(
                        {
                            "iteration": iteration,
                            "view": view_index,
                            "loss": float(np.mean(recent_losses[-100:])),
                        }
                    ),
                    flush=True,
                )

    with torch.no_grad():
        final_features = torch.nn.functional.normalize(
            embedding.weight, dim=-1
        ).cpu().numpy().astype(np.float16)
        final_classifiers = torch.nn.functional.normalize(
            classifiers, dim=-1
        ).cpu().numpy().astype(np.float16)
    np.save(os.path.join(output_dir, "gaussian_features.npy"), final_features)
    np.save(os.path.join(output_dir, "view_classifiers.npy"), final_classifiers)
    result = {
        "format_version": 1,
        "representation": "view_specific_gaussian_classifier",
        "num_gaussians": num_gaussians,
        "num_views": len(entries),
        "num_view_classes": int(final_classifiers.shape[0]),
        "latent_dim": args.latent_dim,
        "temperature": args.temperature,
        "gaussian_features": "gaussian_features.npy",
        "view_classifiers": "view_classifiers.npy",
        "view_offsets": [int(value) for value in view_offsets],
        "storage_bytes": int(final_features.nbytes + final_classifiers.nbytes),
        "source": {
            "cache_dir": cache_dir,
            "codebook_initialization": codebook_dir,
            "note": "Training uses only frozen per-view masks and raw Gaussian rendering contributions.",
        },
        "training": vars(args),
        "final_loss_mean_100": float(np.mean(recent_losses[-100:])),
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as output:
        json.dump(result, output, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
