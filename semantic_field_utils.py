#!/usr/bin/env python
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def l2_normalize(value, dim=-1, eps=1e-8):
    return F.normalize(value, dim=dim, eps=eps)


class SemanticAutoencoder(nn.Module):
    def __init__(self, semantic_dim=16, hidden_dims=(256, 64)):
        super().__init__()
        hidden_dims = tuple(int(value) for value in hidden_dims)
        encoder_dims = (512,) + hidden_dims + (int(semantic_dim),)
        decoder_dims = (int(semantic_dim),) + tuple(reversed(hidden_dims)) + (512,)
        self.semantic_dim = int(semantic_dim)
        self.hidden_dims = hidden_dims
        self.encoder = self._make_mlp(encoder_dims)
        self.decoder = self._make_mlp(decoder_dims)

    @staticmethod
    def _make_mlp(dims):
        layers = []
        for index, (input_dim, output_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(input_dim, output_dim))
            if index + 1 < len(dims) - 1:
                layers.append(nn.GELU())
        return nn.Sequential(*layers)

    def encode(self, features):
        return l2_normalize(self.encoder(l2_normalize(features)))

    def decode(self, latents):
        return l2_normalize(self.decoder(l2_normalize(latents)))

    def forward(self, features):
        return self.decode(self.encode(features))


class IdentitySemanticCodec(nn.Module):
    def __init__(self):
        super().__init__()
        self.semantic_dim = 512
        self.hidden_dims = ()

    def encode(self, features):
        return l2_normalize(features)

    def decode(self, latents):
        return l2_normalize(latents)

    def forward(self, features):
        return self.encode(features)


def save_semantic_codec(path, model, metadata=None):
    payload = {
        "codec_type": "identity" if isinstance(model, IdentitySemanticCodec) else "autoencoder",
        "semantic_dim": model.semantic_dim,
        "hidden_dims": list(model.hidden_dims),
        "state_dict": model.state_dict(),
        "metadata": metadata or {},
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(payload, path)


def load_semantic_codec(path, device="cuda"):
    payload = torch.load(path, map_location=device)
    if payload.get("codec_type", "autoencoder") == "identity":
        model = IdentitySemanticCodec().to(device)
    else:
        model = SemanticAutoencoder(
            semantic_dim=payload["semantic_dim"],
            hidden_dims=payload["hidden_dims"],
        ).to(device)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def inspect_mask_features(feature_dir):
    feature_paths = sorted(Path(feature_dir).glob("*_f.npy"))
    if not feature_paths:
        raise ValueError(f"No *_f.npy files found in {feature_dir}")
    total_features = 0
    for path in feature_paths:
        features = np.load(path, mmap_mode="r")
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"Expected Nx512 features in {path}, got {features.shape}")
        total_features += int(features.shape[0])
    return total_features, [str(path) for path in feature_paths]


def collect_mask_features(feature_dir, max_features=200000, seed=0):
    feature_paths = sorted(Path(feature_dir).glob("*_f.npy"))
    if not feature_paths:
        raise ValueError(f"No *_f.npy files found in {feature_dir}")
    arrays = []
    for path in feature_paths:
        features = np.load(path, mmap_mode="r")
        if features.ndim != 2 or features.shape[1] != 512:
            raise ValueError(f"Expected Nx512 features in {path}, got {features.shape}")
        arrays.append(np.asarray(features, dtype=np.float32))
    features = np.concatenate(arrays, axis=0)
    if max_features > 0 and features.shape[0] > max_features:
        generator = np.random.default_rng(seed)
        indices = generator.choice(features.shape[0], size=max_features, replace=False)
        features = features[indices]
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    return torch.from_numpy(features), [str(path) for path in feature_paths]


def sample_segment_pixels(segmentation, max_pixels, seed):
    flat = np.asarray(segmentation).reshape(-1)
    valid = np.flatnonzero(flat >= 0)
    if valid.size == 0:
        return np.empty((0,), dtype=np.int64)
    if max_pixels <= 0 or valid.size <= max_pixels:
        return valid.astype(np.int64, copy=False)

    generator = np.random.default_rng(seed)
    valid_segments = flat[valid]
    order = np.argsort(valid_segments, kind="stable")
    sorted_valid = valid[order]
    sorted_segments = valid_segments[order]
    segment_ids, starts = np.unique(sorted_segments, return_index=True)
    ends = np.concatenate([starts[1:], np.array([sorted_valid.size])])
    quota = max(1, max_pixels // max(1, segment_ids.size))
    selected = []
    for start, end in zip(starts, ends):
        candidates = sorted_valid[start:end]
        take = min(quota, candidates.size)
        selected.append(generator.choice(candidates, size=take, replace=False))
    selected = np.unique(np.concatenate(selected))
    if selected.size < max_pixels:
        selected_mask = np.zeros(flat.shape[0], dtype=bool)
        selected_mask[selected] = True
        remaining = valid[~selected_mask[valid]]
        take = min(max_pixels - selected.size, remaining.size)
        if take:
            selected = np.concatenate(
                [selected, generator.choice(remaining, size=take, replace=False)]
            )
    if selected.size > max_pixels:
        selected = generator.choice(selected, size=max_pixels, replace=False)
    generator.shuffle(selected)
    return selected.astype(np.int64, copy=False)


def as_frozen_parameter(value, device="cuda"):
    return nn.Parameter(value.detach().to(device), requires_grad=False)


def load_geometry_checkpoint(gaussians, checkpoint_path, device="cuda"):
    model_params, checkpoint_iteration = torch.load(checkpoint_path, map_location=device)
    if len(model_params) == 12:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            max_radii2d,
            xyz_gradient_accum,
            denom,
            _optimizer_state,
            spatial_lr_scale,
        ) = model_params
    elif len(model_params) == 13:
        (
            active_sh_degree,
            xyz,
            features_dc,
            features_rest,
            scaling,
            rotation,
            opacity,
            _language_feature,
            max_radii2d,
            xyz_gradient_accum,
            denom,
            _optimizer_state,
            spatial_lr_scale,
        ) = model_params
    else:
        raise ValueError(f"Unsupported checkpoint tuple length: {len(model_params)}")

    gaussians.active_sh_degree = active_sh_degree
    gaussians._xyz = as_frozen_parameter(xyz, device)
    gaussians._features_dc = as_frozen_parameter(features_dc, device)
    gaussians._features_rest = as_frozen_parameter(features_rest, device)
    gaussians._scaling = as_frozen_parameter(scaling, device)
    gaussians._rotation = as_frozen_parameter(rotation, device)
    gaussians._opacity = as_frozen_parameter(opacity, device)
    gaussians._language_feature = None
    gaussians.max_radii2D = max_radii2d.detach().to(device)
    gaussians.xyz_gradient_accum = xyz_gradient_accum.detach().to(device)
    gaussians.denom = denom.detach().to(device)
    gaussians.spatial_lr_scale = spatial_lr_scale
    gaussians.optimizer = None
    return int(checkpoint_iteration)


def save_json(path, payload):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as output:
        json.dump(payload, output, indent=2)


def load_json(path):
    with open(path) as source:
        return json.load(source)
