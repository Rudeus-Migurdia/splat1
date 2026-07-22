#!/usr/bin/env python
"""Evaluate a fully discrete per-Gaussian multi-level semantic codebook."""

import hashlib
import json
import os
import sys
from argparse import ArgumentParser
from types import SimpleNamespace

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F
from tqdm import tqdm

from arguments import ModelParams, PipelineParams
from eval_lerf_ovs_miou import (
    calibrate_frame_scores,
    evaluate_paper_3d_selection,
    load_lerf_labels,
    polygons_to_mask,
    save_visualization,
)
from evaluation.openclip_encoder import OpenCLIPNetwork
from gaussian_renderer import render
from lerf_ovs_paper_protocol import PROTOCOL_NAME
from scene import GaussianModel, Scene
from semantic_hypothesis_routing import (
    blend_contrastive_group_hypotheses,
    blend_group_hypotheses,
    blend_dual_code_hypotheses,
    blend_sparse_hypothesis,
    complete_scores_from_seeded_groups,
    fuse_calibrated_equal_query_tokens,
    fuse_calibrated_hierarchical_memory,
    fuse_equal_query_tokens,
    fuse_hierarchical_semantic_memory,
    fuse_information_gain_equal_query_tokens,
    fuse_global_sparse_group_retrieval,
    fuse_multiscale_set_relation_token_scores,
    fuse_quantization_aware_equal_query_tokens,
    fuse_query_conditioned_spatial_posterior,
    fuse_signed_relation_graph_scores,
    route_group_hypotheses,
)
from query_conditioned_spatial_posterior import (
    GroupAnisotropicGeometry,
    QueryConditionedSpatialPosterior,
)
from semantic_field_utils import load_geometry_checkpoint
from utils.general_utils import safe_state


class GaussianCodebookArtifact:
    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        representation = self.manifest.get("representation")
        if representation not in {
            "gaussian_multilevel_residual_codebook",
            "gaussian_adaptive_shared_codebook",
        }:
            raise ValueError("Unsupported Gaussian codebook representation")
        self.shared_codebook = representation == "gaussian_adaptive_shared_codebook"
        self.num_gaussians = int(self.manifest["num_gaussians"])
        self.feature_dim = int(self.manifest["feature_dim"])
        self.invalid_id = int(self.manifest["invalid_id"])
        loaded_ids = np.load(
            os.path.join(self.dir, self.manifest["point_code_ids"]),
            mmap_mode="r",
        )
        sparse_overflow = (
            self.shared_codebook
            and self.manifest.get("storage_layout") == "base_plus_sparse_overflow"
        )
        if sparse_overflow:
            if loaded_ids.shape != (self.num_gaussians,):
                raise ValueError("Sparse base IDs do not match the Gaussian count")
            self.point_code_ids = np.full(
                (self.num_gaussians, int(self.manifest["id_slots"])),
                self.invalid_id,
                dtype=loaded_ids.dtype,
            )
            self.point_code_weights = np.zeros(self.point_code_ids.shape, dtype=np.uint8)
            self.point_code_ids[:, 0] = loaded_ids
            self.point_code_weights[loaded_ids != self.invalid_id, 0] = 255
            overflow_points = np.load(
                os.path.join(self.dir, self.manifest["overflow_point_ids"])
            ).astype(np.int64)
            overflow_slots = np.load(
                os.path.join(self.dir, self.manifest["overflow_slots"])
            ).astype(np.int64)
            overflow_ids = np.load(
                os.path.join(self.dir, self.manifest["overflow_code_ids"])
            )
            overflow_weights = np.load(
                os.path.join(self.dir, self.manifest["overflow_weights"])
            )
            if not (
                overflow_points.shape
                == overflow_slots.shape
                == overflow_ids.shape
                == overflow_weights.shape
            ):
                raise ValueError("Sparse overflow arrays must have matching shapes")
            self.point_code_ids[overflow_points, overflow_slots] = overflow_ids
            self.point_code_weights[overflow_points, overflow_slots] = overflow_weights
        else:
            self.point_code_ids = loaded_ids
        self.valid_mask = np.load(
            os.path.join(self.dir, self.manifest["valid_mask"]),
            mmap_mode="r",
        )
        if not sparse_overflow:
            self.point_code_weights = None
        if self.shared_codebook and not sparse_overflow:
            weights_name = self.manifest.get("point_code_weights")
            if weights_name:
                self.point_code_weights = np.load(
                    os.path.join(self.dir, weights_name),
                    mmap_mode="r",
                )
            elif self.manifest.get("weight_dtype") != "implicit_unit":
                raise ValueError(
                    "Shared codebooks require point weights or weight_dtype=implicit_unit"
                )
        self.codebooks = []
        for name in self.manifest["codebook_files"]:
            codebook = torch.from_numpy(
                np.load(os.path.join(self.dir, name)).astype(np.float32)
            ).to(device)
            self.codebooks.append(codebook)
        expected_slots = (
            int(self.manifest["id_slots"])
            if self.shared_codebook
            else len(self.codebooks)
        )
        if self.point_code_ids.shape != (self.num_gaussians, expected_slots):
            raise ValueError("Point code IDs do not match the codebook manifest")
        if (
            self.shared_codebook
            and self.point_code_weights is not None
            and self.point_code_weights.shape != self.point_code_ids.shape
        ):
            raise ValueError("Adaptive code IDs and weights must have matching shapes")
        if self.valid_mask.shape != (self.num_gaussians,):
            raise ValueError("Valid mask does not match the codebook manifest")
        if any(codebook.shape[1] != self.feature_dim for codebook in self.codebooks):
            raise ValueError("Codebook dimensions do not match the manifest")

    @torch.no_grad()
    def reconstruct_range(self, start, end):
        output = torch.zeros(
            (end - start, self.feature_dim),
            dtype=torch.float32,
            device=self.codebooks[0].device,
        )
        valid_np = np.array(self.valid_mask[start:end], dtype=bool, copy=True)
        ids_np = np.asarray(self.point_code_ids[start:end], dtype=np.int64)
        if self.shared_codebook:
            valid_np &= np.any(ids_np != self.invalid_id, axis=1)
        else:
            valid_np &= np.all(ids_np != self.invalid_id, axis=1)
        if not valid_np.any():
            return output
        selected_ids = ids_np[valid_np]
        ids = torch.from_numpy(selected_ids).long().to(output.device)
        reconstruction = torch.zeros(
            (ids.shape[0], self.feature_dim), dtype=torch.float32, device=output.device
        )
        if self.shared_codebook:
            slot_valid = ids != self.invalid_id
            safe_ids = ids.masked_fill(~slot_valid, 0)
            if slot_valid.any() and int(safe_ids[slot_valid].max()) >= self.codebooks[0].shape[0]:
                raise ValueError("Point IDs exceed the shared codebook")
            if self.point_code_weights is None:
                reconstruction = (
                    self.codebooks[0][safe_ids] * slot_valid.unsqueeze(-1)
                ).sum(dim=1)
            else:
                weights_np = np.asarray(
                    self.point_code_weights[start:end][valid_np], dtype=np.float32
                ) / 255.0
                weights = torch.from_numpy(weights_np).to(output.device)
                reconstruction = (
                    self.codebooks[0][safe_ids]
                    * weights.unsqueeze(-1)
                    * slot_valid.unsqueeze(-1)
                ).sum(dim=1)
        else:
            for level, codebook in enumerate(self.codebooks):
                level_ids = ids[:, level]
                if level_ids.numel() and int(level_ids.max()) >= codebook.shape[0]:
                    raise ValueError(f"Level {level} point IDs exceed its codebook")
                reconstruction += codebook[level_ids]
        output[torch.from_numpy(np.flatnonzero(valid_np)).long().to(output.device)] = F.normalize(
            reconstruction, dim=-1
        )
        return output

    @torch.no_grad()
    def query_activation(
        self,
        clip_model,
        category_index,
        chunk_size=65536,
        object_codebook=None,
        object_feature_weight=0.0,
    ):
        output = torch.zeros((self.num_gaussians, 1), dtype=torch.float32, device=self.codebooks[0].device)
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            reconstruction = self.reconstruct_range(start, end)
            if object_codebook is not None and object_feature_weight > 0.0:
                reconstruction = F.normalize(
                    reconstruction
                    + object_feature_weight * object_codebook.reconstruct_range(start, end),
                    dim=-1,
                )
            valid = reconstruction.norm(dim=-1) > 0.0
            if valid.any():
                positions = torch.nonzero(valid, as_tuple=False).squeeze(1) + start
                output[positions] = clip_model.get_activation(
                    reconstruction[valid], category_index
                ).float()
        return output


class ConsensusFeatureArtifact:
    """Read-only full-precision upper bound from the cached 2D consensus source."""

    def __init__(self, consensus_path, device="cuda"):
        self.dir = os.path.abspath(consensus_path)
        payload = torch.load(self.dir, map_location="cpu")
        if "initial_features" not in payload:
            raise ValueError("Consensus payload is missing initial_features")
        self.features = payload["initial_features"].detach().cpu().contiguous()
        if self.features.ndim != 2:
            raise ValueError("Consensus initial_features must have shape [N, D]")
        support = payload.get("total_weights")
        if support is None:
            self.valid_mask = self.features.norm(dim=-1) > 0.0
        else:
            self.valid_mask = support.detach().cpu().reshape(-1) > 0.0
        if self.valid_mask.shape != (self.features.shape[0],):
            raise ValueError("Consensus support does not match initial_features")
        semantic_opacity = payload.get("semantic_opacity")
        if semantic_opacity is None:
            self.semantic_opacity = None
        else:
            self.semantic_opacity = (
                semantic_opacity.detach().cpu().float().reshape(-1).contiguous()
            )
            if self.semantic_opacity.shape != (self.features.shape[0],):
                raise ValueError("Consensus semantic_opacity does not match initial_features")
            if not torch.isfinite(self.semantic_opacity).all():
                raise ValueError("Consensus semantic_opacity must be finite")
            if (self.semantic_opacity < 0.0).any() or (self.semantic_opacity > 1.0).any():
                raise ValueError("Consensus semantic_opacity must be in [0, 1]")
        self.num_gaussians = int(self.features.shape[0])
        self.feature_dim = int(self.features.shape[1])
        storage_bytes = int(self.features.numel() * self.features.element_size())
        storage_bytes += int(self.valid_mask.numel() * self.valid_mask.element_size())
        if self.semantic_opacity is not None:
            storage_bytes += int(
                self.semantic_opacity.numel() * self.semantic_opacity.element_size()
            )
        self.manifest = {
            "representation": "continuous_consensus_upper_bound",
            "feature_dim": self.feature_dim,
            "num_gaussians": self.num_gaussians,
            "num_valid_gaussians": int(self.valid_mask.sum()),
            "valid_fraction": float(self.valid_mask.float().mean()),
            "has_semantic_opacity": self.semantic_opacity is not None,
            "source": {"type": "cached_2d_consensus", "path": self.dir},
            "storage": {"total_semantic_bytes": storage_bytes},
        }
        self.device = device
        self.route_base_features = None
        self.route_base_scores = None
        self.route_candidate_scores = None
        self.query_route_mode = "none"
        self.query_route_diagnostics = {}

    @torch.no_grad()
    def blend_with_consensus(
        self,
        base_path,
        candidate_weight,
        chunk_size=65536,
        weight_mode="constant",
        retain_base_for_query_route=False,
    ):
        if not 0.0 <= candidate_weight <= 1.0:
            raise ValueError("candidate_weight must be in [0, 1]")
        payload = torch.load(os.path.abspath(base_path), map_location="cpu")
        base_features = payload.get("initial_features")
        if base_features is None or base_features.shape != self.features.shape:
            raise ValueError("Blend base must match the candidate consensus features")
        if weight_mode == "constant":
            blend_gate = None
        else:
            blend_gate = payload.get("fusion_gate")
            if blend_gate is None or blend_gate.shape != (self.num_gaussians,):
                raise ValueError("Reliability-weighted blending requires base fusion_gate")
            blend_gate = blend_gate.detach().cpu().float().clamp(0.0, 1.0)
            if weight_mode == "inverse_base_gate":
                blend_gate = 1.0 - blend_gate
            elif weight_mode != "base_gate":
                raise ValueError(f"Unsupported consensus blend weight mode: {weight_mode}")
        blended = torch.empty_like(self.features)
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            base = F.normalize(base_features[start:end].float(), dim=-1)
            candidate = F.normalize(self.features[start:end].float(), dim=-1)
            weight = candidate_weight
            if blend_gate is not None:
                weight = candidate_weight * blend_gate[start:end].unsqueeze(-1)
            features = F.normalize(
                (1.0 - weight) * base + weight * candidate,
                dim=-1,
            )
            blended[start:end].copy_(features.to(blended.dtype))
        self.features = blended.contiguous()
        if retain_base_for_query_route:
            self.route_base_features = base_features.detach().cpu().contiguous()
        self.manifest["feature_blend"] = {
            "base_path": os.path.abspath(base_path),
            "candidate_weight": float(candidate_weight),
            "weight_mode": weight_mode,
        }

    @torch.no_grad()
    def prepare_query_routing(self, clip_model, num_categories, mode, chunk_size=65536):
        if mode not in {"margin_switch", "margin_positive", "query_positive"}:
            raise ValueError(f"Unsupported consensus query route: {mode}")
        if self.route_base_features is None:
            raise ValueError("Query routing requires a retained blend-base consensus")
        self.route_base_scores = torch.zeros(
            (self.num_gaussians, num_categories),
            dtype=torch.float32,
            device=self.device,
        )
        self.route_candidate_scores = torch.zeros_like(self.route_base_scores)
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            candidate = self.reconstruct_range(start, end)
            valid = candidate.norm(dim=-1) > 0.0
            if not valid.any():
                continue
            base = torch.zeros_like(candidate)
            base_values = self.route_base_features[start:end][valid.cpu()].to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            base[valid] = F.normalize(base_values, dim=-1)
            positions = torch.nonzero(valid, as_tuple=False).squeeze(1) + start
            for category_index in range(num_categories):
                self.route_base_scores[positions, category_index] = (
                    clip_model.get_activation(base[valid], category_index)
                    .float()
                    .squeeze(-1)
                )
                self.route_candidate_scores[positions, category_index] = (
                    clip_model.get_activation(candidate[valid], category_index)
                    .float()
                    .squeeze(-1)
                )
        self.route_base_features = None
        self.query_route_mode = mode
        self.manifest["query_route"] = {
            "mode": mode,
            "diagnostics": self.query_route_diagnostics,
        }

    @torch.no_grad()
    def reconstruct_range(self, start, end):
        output = torch.zeros(
            (end - start, self.feature_dim), dtype=torch.float32, device=self.device
        )
        valid = self.valid_mask[start:end]
        if valid.any():
            features = self.features[start:end][valid].to(
                self.device, dtype=torch.float32, non_blocking=True
            )
            output[valid.to(self.device)] = F.normalize(features, dim=-1)
        return output

    @torch.no_grad()
    def query_activation(
        self,
        clip_model,
        category_index,
        chunk_size=65536,
        object_codebook=None,
        object_feature_weight=0.0,
    ):
        if object_codebook is not None or object_feature_weight != 0.0:
            raise ValueError("Continuous consensus does not support object code composition")
        if self.query_route_mode != "none":
            output, selected = route_query_activation(
                self.route_base_scores,
                self.route_candidate_scores,
                category_index,
                self.query_route_mode,
            )
            self.query_route_diagnostics[str(category_index)] = {
                "candidate_fraction": float(selected.float().mean())
            }
            return output
        output = torch.zeros(
            (self.num_gaussians, 1), dtype=torch.float32, device=self.device
        )
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            reconstruction = self.reconstruct_range(start, end)
            valid = reconstruction.norm(dim=-1) > 0.0
            if valid.any():
                positions = torch.nonzero(valid, as_tuple=False).squeeze(1) + start
                activation = clip_model.get_activation(
                    reconstruction[valid], category_index
                ).float()
                if self.semantic_opacity is not None:
                    opacity = self.semantic_opacity[start:end][valid.cpu()].to(
                        self.device, dtype=torch.float32, non_blocking=True
                    )
                    activation = activation * opacity.unsqueeze(-1)
                output[positions] = activation
        return output


def route_query_activation(
    base_scores,
    candidate_scores,
    category_index,
    mode,
    candidate_mask=None,
):
    if base_scores.shape != candidate_scores.shape or base_scores.ndim != 2:
        raise ValueError("Query route score tables must have matching [N, C] shapes")
    if not 0 <= category_index < base_scores.shape[1]:
        raise ValueError("category_index is outside the route score table")
    if mode not in {
        "margin_switch",
        "margin_positive",
        "query_positive",
        "query_positive_blend",
    }:
        raise ValueError(f"Unsupported query route mode: {mode}")
    base_target = base_scores[:, category_index]
    candidate_target = candidate_scores[:, category_index]
    if mode in {"query_positive", "query_positive_blend"}:
        selected = candidate_target > base_target
        if mode == "query_positive_blend":
            reliability = (
                candidate_mask.float().clamp(0.0, 1.0)
                if candidate_mask is not None
                else torch.ones_like(base_target)
            )
            selected = selected & (reliability > 0.0)
            output = base_target + reliability * (candidate_target - base_target).clamp_min(0.0)
        else:
            if candidate_mask is not None:
                selected = selected & candidate_mask.bool()
            output = torch.where(selected, candidate_target, base_target)
        return output.unsqueeze(-1), selected
    if base_scores.shape[1] > 1:
        base_competitors = base_scores.clone()
        candidate_competitors = candidate_scores.clone()
        base_competitors[:, category_index] = -torch.inf
        candidate_competitors[:, category_index] = -torch.inf
        base_margin = base_target - base_competitors.max(dim=1).values
        candidate_margin = candidate_target - candidate_competitors.max(dim=1).values
    else:
        base_margin = base_target
        candidate_margin = candidate_target
    selected = candidate_margin > base_margin
    if candidate_mask is not None:
        selected = selected & candidate_mask
    if mode == "margin_switch":
        output = torch.where(selected, candidate_target, base_target)
    else:
        output = torch.where(
            selected,
            torch.maximum(candidate_target, base_target),
            base_target,
        )
    return output.unsqueeze(-1), selected


@torch.no_grad()
def precompute_artifact_query_scores(
    artifact,
    clip_model,
    num_categories,
    chunk_size=65536,
):
    device = getattr(artifact, "device", None)
    if device is None:
        device = artifact.codebooks[0].device
    scores = torch.zeros(
        (artifact.num_gaussians, num_categories),
        dtype=torch.float32,
        device=device,
    )
    for start in range(0, artifact.num_gaussians, chunk_size):
        end = min(start + chunk_size, artifact.num_gaussians)
        features = artifact.reconstruct_range(start, end)
        valid = features.norm(dim=-1) > 0.0
        if not valid.any():
            continue
        positions = torch.nonzero(valid, as_tuple=False).squeeze(1) + start
        for category_index in range(num_categories):
            scores[positions, category_index] = (
                clip_model.get_activation(features[valid], category_index)
                .float()
                .squeeze(-1)
            )
    return scores


class SparseSemanticHypothesis:
    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        self.device = torch.device(device)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        representation = self.manifest.get("representation")
        supported = {
            "sparse_continuous_semantic_hypothesis",
            "sparse_quantized_semantic_hypothesis",
        }
        if representation not in supported:
            raise ValueError("Unsupported sparse semantic hypothesis artifact")
        self.num_gaussians = int(self.manifest["num_gaussians"])
        self.feature_dim = int(self.manifest["feature_dim"])
        self.point_ids = torch.from_numpy(
            np.load(os.path.join(self.dir, self.manifest["point_ids"])).astype(np.int64)
        ).to(self.device)
        if representation == "sparse_continuous_semantic_hypothesis":
            feature_table = np.load(
                os.path.join(self.dir, self.manifest["features"])
            ).astype(np.float32)
        else:
            codebook = np.load(
                os.path.join(self.dir, self.manifest["codebook"])
            ).astype(np.float32)
            code_ids = np.load(
                os.path.join(self.dir, self.manifest["code_ids"])
            ).astype(np.int64)
            if codebook.ndim != 2 or codebook.shape[1] != self.feature_dim:
                raise ValueError("Sparse hypothesis codebook has an invalid shape")
            if code_ids.shape != tuple(self.point_ids.shape):
                raise ValueError("Sparse hypothesis code IDs do not match point IDs")
            if code_ids.size and (code_ids.min() < 0 or code_ids.max() >= codebook.shape[0]):
                raise ValueError("Sparse hypothesis code IDs are out of bounds")
            feature_table = codebook[code_ids]
        self.features = torch.from_numpy(feature_table).to(self.device)
        self.reliability = torch.from_numpy(
            np.load(os.path.join(self.dir, self.manifest["reliability"])).astype(np.float32)
            / 255.0
        ).to(self.device)
        if self.point_ids.numel() and (
            int(self.point_ids.min()) < 0
            or int(self.point_ids.max()) >= self.num_gaussians
            or torch.unique(self.point_ids).numel() != self.point_ids.numel()
        ):
            raise ValueError("Sparse hypothesis point IDs are invalid or duplicated")
        if self.features.shape != (self.point_ids.numel(), self.feature_dim):
            raise ValueError("Sparse hypothesis feature table does not match point IDs")
        if self.reliability.shape != self.point_ids.shape:
            raise ValueError("Sparse hypothesis reliability does not match point IDs")
        self.query_activations = None

    @torch.no_grad()
    def set_query_activations(self, clip_model, num_categories):
        self.query_activations = torch.cat(
            [
                clip_model.get_activation(self.features, category_index).float()
                for category_index in range(num_categories)
            ],
            dim=1,
        )

    @torch.no_grad()
    def candidate_tables(self, category_index, query_margin=False):
        if self.query_activations is None:
            raise ValueError("Sparse query activations have not been initialized")
        device = self.features.device
        scores = torch.zeros((self.num_gaussians, 1), dtype=torch.float32, device=device)
        reliability = torch.zeros_like(scores)
        valid = torch.zeros((self.num_gaussians, 1), dtype=torch.bool, device=device)
        scores[self.point_ids, 0] = self.query_activations[:, category_index]
        reliability[self.point_ids, 0] = self.reliability
        valid[self.point_ids, 0] = True
        specificity = None
        if query_margin:
            target = self.query_activations[:, category_index]
            competitors = self.query_activations.clone()
            competitors[:, category_index] = -torch.inf
            margin = (target - competitors.max(dim=1).values).clamp_min(0.0)
            specificity = torch.zeros_like(scores)
            specificity[self.point_ids, 0] = margin
        return scores, reliability, valid, specificity

    @property
    def storage_bytes(self):
        return int(self.manifest["storage"]["total_semantic_bytes"])

    @torch.no_grad()
    def query_activation(self, clip_model, category_index, chunk_size=65536, **_):
        output = torch.zeros((self.num_gaussians, 1), dtype=torch.float32, device=self.device)
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            reconstruction = self.reconstruct_range(start, end)
            valid = reconstruction.norm(dim=-1) > 0.0
            if valid.any():
                positions = torch.nonzero(valid, as_tuple=False).squeeze(1) + start
                output[positions] = clip_model.get_activation(
                    reconstruction[valid], category_index
                ).float()
        return output


class SignedGaussianRelationGraph:
    """Read-only compact local graph used only after query-token scoring."""

    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != (
            "multiview_local_signed_gaussian_relation_graph"
        ):
            raise ValueError("Unsupported Gaussian relation graph representation")
        self.num_gaussians = int(self.manifest["num_gaussians"])
        neighbor_ids = np.load(
            os.path.join(self.dir, self.manifest["neighbor_ids"])
        )
        packed_weights = np.load(
            os.path.join(self.dir, self.manifest["signed_relation_weights"])
        )
        if neighbor_ids.shape != packed_weights.shape:
            raise ValueError("Relation graph IDs and weights must match")
        if neighbor_ids.shape != (
            self.num_gaussians,
            int(self.manifest["neighbors"]),
        ):
            raise ValueError("Relation graph shape does not match its manifest")
        if neighbor_ids.dtype != np.int32 or packed_weights.dtype != np.int8:
            raise ValueError("Relation graph requires int32 IDs and int8 weights")
        if neighbor_ids.size and (
            int(neighbor_ids.min()) < 0
            or int(neighbor_ids.max()) >= self.num_gaussians
        ):
            raise ValueError("Relation graph neighbor IDs exceed the Gaussian count")
        scale = float(self.manifest["relation_scale"])
        if scale <= 0.0:
            raise ValueError("Relation graph scale must be positive")
        self.neighbor_ids = torch.from_numpy(neighbor_ids).to(device)
        self.signed_weights = (
            torch.from_numpy(packed_weights.astype(np.float32)) * scale
        ).to(device)

    @property
    def storage_bytes(self):
        return int(self.manifest["storage"]["total_bytes"])


class MultiscaleSetRelationGraph:
    """Sparse level-conditioned relations selected by odd/even training views."""

    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != (
            "heldout_multiscale_set_relation_diagnostic"
        ):
            raise ValueError("Unsupported multiscale set relation representation")
        if not self.manifest.get("source_contract", {}).get(
            "odd_even_heldout_evaluation"
        ):
            raise ValueError("Multiscale relations require odd/even validation")
        self.num_gaussians = int(self.manifest["num_gaussians"])
        relation_graph_dir = self.manifest["inputs"]["relation_graph_dir"]
        with open(os.path.join(relation_graph_dir, "manifest.json")) as source:
            relation_manifest = json.load(source)
        neighbor_ids = np.load(
            os.path.join(relation_graph_dir, relation_manifest["neighbor_ids"])
        )
        signatures = np.load(
            os.path.join(
                self.dir, self.manifest["artifacts"]["multiscale_relation_signature"]
            )
        )
        ambiguous = np.load(
            os.path.join(
                self.dir, self.manifest["artifacts"]["stable_set_ambiguous_edges"]
            )
        )
        expected_edges = (
            self.num_gaussians,
            int(self.manifest["neighbors"]),
        )
        if neighbor_ids.shape != expected_edges or neighbor_ids.dtype != np.int32:
            raise ValueError("Multiscale relation neighbor IDs violate the manifest")
        if signatures.shape != expected_edges + (4,) or signatures.dtype != np.int8:
            raise ValueError("Multiscale relation signatures must be int8 [N, K, 4]")
        if ambiguous.shape != expected_edges or ambiguous.dtype != np.bool_:
            raise ValueError("Stable ambiguity mask must be bool [N, K]")
        if signatures.size and (signatures.min() < -1 or signatures.max() > 1):
            raise ValueError("Multiscale relation signatures must be ternary")
        signatures = signatures * ambiguous[..., None]
        self.neighbor_ids = torch.from_numpy(neighbor_ids).to(device)
        self.signed_weights = torch.from_numpy(signatures.astype(np.float32)).to(device)
        self._storage_bytes = int(
            neighbor_ids.nbytes + signatures.nbytes + ambiguous.nbytes
        )

    @property
    def storage_bytes(self):
        return self._storage_bytes


class CounterfactualCodebookNeighborhoods:
    """Nearest same-level semantic prototypes used as query counterfactuals."""

    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != (
            "hierarchical_codebook_counterfactual_neighborhoods"
        ):
            raise ValueError("Unsupported counterfactual codebook neighborhood artifact")
        self.neighbor_ids = {}
        for level_spec in self.manifest.get("levels", []):
            level = int(level_spec["level"])
            ids = np.load(os.path.join(self.dir, level_spec["neighbor_ids"]))
            expected = (int(level_spec["num_codes"]), int(self.manifest["neighbors"]))
            if ids.shape != expected or ids.dtype != np.uint16:
                raise ValueError("Counterfactual neighbor IDs violate their manifest")
            if ids.size and int(ids.max()) >= expected[0]:
                raise ValueError("Counterfactual neighbor IDs exceed their level codebook")
            self.neighbor_ids[level] = torch.from_numpy(ids.astype(np.int64)).to(device)
        if sorted(self.neighbor_ids) != [0, 1, 2, 3]:
            raise ValueError("Counterfactual neighborhoods require exactly levels 0-3")

    def validate_memory(self, memory_dir):
        manifest_path = os.path.join(os.path.abspath(memory_dir), "manifest.json")
        with open(manifest_path, "rb") as source:
            digest = hashlib.sha256(source.read()).hexdigest()
        if digest != self.manifest.get("memory_manifest_sha256"):
            raise ValueError("Counterfactual neighborhoods do not match the resident memory")
        with open(manifest_path) as source:
            memory = json.load(source)
        declared = {int(item["level"]): item for item in memory["level_codebooks"]}
        for level_spec in self.manifest["levels"]:
            level = int(level_spec["level"])
            codebook_path = os.path.join(memory_dir, declared[level]["codebook"])
            with open(codebook_path, "rb") as source:
                codebook_digest = hashlib.sha256(source.read()).hexdigest()
            if codebook_digest != level_spec["codebook_sha256"]:
                raise ValueError(
                    f"Counterfactual neighborhoods do not match level {level} codebook"
                )

    @property
    def storage_bytes(self):
        return int(self.manifest["storage"]["total_bytes"])


class GroupHierarchy:
    def __init__(
        self,
        codebook_path=None,
        assignments_path=None,
        artifact_dir=None,
        device="cuda",
    ):
        self.artifact_dir = os.path.abspath(artifact_dir) if artifact_dir else None
        self.manifest = None
        group_local_semantic_ids = None
        level_code_priors = None
        if self.artifact_dir:
            with open(os.path.join(self.artifact_dir, "manifest.json")) as source:
                manifest = json.load(source)
            self.manifest = manifest
            representation = manifest.get("representation")
            if representation not in {
                "compact_group_hierarchy",
                "shared_codebook_group_hierarchy",
                "hierarchical_independent_group_codebooks",
            }:
                raise ValueError("Unsupported compact group hierarchy artifact")
            self.assignments_path = None
            independent_level_codebooks = (
                representation == "hierarchical_independent_group_codebooks"
            )
            level_codebooks = {} if independent_level_codebooks else None
            shared_group_codes = representation == "shared_codebook_group_hierarchy"
            group_levels = None
            if independent_level_codebooks:
                self.codebook_path = self.artifact_dir
                semantic_ids = np.load(
                    os.path.join(self.artifact_dir, manifest["group_semantic_code_ids"])
                ).astype(np.int64)
                if semantic_ids.ndim != 2 or semantic_ids.shape[1] != 1:
                    raise ValueError("Independent hierarchical semantic IDs must have shape [G, 1]")
                group_levels = np.load(
                    os.path.join(self.artifact_dir, manifest["group_level"])
                ).astype(np.int64)
                if group_levels.shape != (semantic_ids.shape[0],):
                    raise ValueError("Independent hierarchical group levels must match semantic tokens")
                semantic_invalid = int(manifest["semantic_invalid_id"])
                group_local_semantic_ids = semantic_ids[:, 0].copy()
                codebook = np.zeros(
                    (semantic_ids.shape[0], int(manifest["feature_dim"])),
                    dtype=np.float32,
                )
                declared_levels = manifest.get("level_codebooks", [])
                if len(declared_levels) != 4:
                    raise ValueError("Hierarchical semantic memory requires four declared level codebooks")
                for level_spec in declared_levels:
                    level = int(level_spec["level"])
                    level_mask = group_levels == level
                    level_codebook = np.load(
                        os.path.join(self.artifact_dir, level_spec["codebook"])
                    ).astype(np.float32)
                    level_codebook /= np.maximum(
                        np.linalg.norm(level_codebook, axis=-1, keepdims=True), 1e-8
                    )
                    level_codebooks[level] = level_codebook
                    local_ids = semantic_ids[level_mask, 0]
                    if np.any(local_ids == semantic_invalid):
                        raise ValueError("Resident hierarchical tokens must have a local semantic ID")
                    if local_ids.size and (
                        int(local_ids.min()) < 0 or int(local_ids.max()) >= level_codebook.shape[0]
                    ):
                        raise ValueError("Hierarchical semantic IDs exceed their level codebook")
                    codebook[level_mask] = level_codebook[local_ids]
                atom_codebook = np.zeros_like(codebook)
            else:
                self.codebook_path = os.path.join(
                    self.artifact_dir, manifest["group_codebook"]
                )
            if shared_group_codes:
                shared = np.load(self.codebook_path).astype(np.float32)
                semantic_ids = np.load(
                    os.path.join(self.artifact_dir, manifest["group_semantic_code_ids"])
                ).astype(np.int64)
                semantic_invalid = int(manifest["semantic_invalid_id"])
                semantic_valid = semantic_ids != semantic_invalid
                safe_ids = np.where(semantic_valid, semantic_ids, 0)
                if semantic_valid.any() and int(safe_ids[semantic_valid].max()) >= shared.shape[0]:
                    raise ValueError("Group semantic IDs exceed the shared vocabulary")
                codebook = (
                    shared[safe_ids] * semantic_valid[..., None]
                ).sum(axis=1)
                atom_name = manifest.get("group_semantic_atom_code_ids")
                if atom_name:
                    atom_ids = np.load(
                        os.path.join(self.artifact_dir, atom_name)
                    ).astype(np.int64)
                    atom_invalid = int(
                        manifest.get("semantic_atom_invalid_id", semantic_invalid)
                    )
                    atom_valid = atom_ids != atom_invalid
                    atom_safe = np.where(atom_valid, atom_ids, 0)
                    if atom_ids.shape != semantic_ids.shape:
                        raise ValueError("Semantic atom IDs must match group semantic IDs")
                    if atom_valid.any() and int(atom_safe[atom_valid].max()) >= shared.shape[0]:
                        raise ValueError("Semantic atom IDs exceed the shared vocabulary")
                    atom_codebook = (
                        shared[atom_safe] * atom_valid[..., None]
                    ).sum(axis=1)
                else:
                    atom_codebook = np.zeros_like(codebook)
            elif not independent_level_codebooks:
                codebook = np.load(self.codebook_path).astype(np.float32)
                atom_codebook = np.zeros_like(codebook)
            point_ids = np.load(
                os.path.join(self.artifact_dir, manifest["point_group_ids"])
            ).astype(np.int64)
            invalid_id = int(manifest["invalid_id"])
            point_ids[point_ids == invalid_id] = -1
            point_scores = np.load(
                os.path.join(self.artifact_dir, manifest["point_group_weights"])
            ).astype(np.float32) / 255.0
            competitor_name = manifest.get("point_competitor_ids")
            if competitor_name:
                competitor_ids = np.load(
                    os.path.join(self.artifact_dir, competitor_name)
                ).astype(np.int64)
                competitor_invalid = int(
                    manifest.get("competitor_invalid_id", invalid_id)
                )
                competitor_ids[competitor_ids == competitor_invalid] = -1
                if competitor_ids.shape != point_ids.shape:
                    raise ValueError("Competitor IDs must match positive group IDs")
            else:
                competitor_ids = np.full_like(point_ids, -1)
            self._storage_bytes = int(manifest["storage"]["total_semantic_bytes"])
            entropy_name = manifest.get("point_group_entropy")
            self.point_entropy = (
                torch.from_numpy(
                    np.load(os.path.join(self.artifact_dir, entropy_name)).astype(np.float32)
                    / 255.0
                ).to(device)
                if entropy_name
                else torch.zeros_like(torch.from_numpy(point_scores)).to(device)
            )
            point_reliability_name = manifest.get("point_group_reliability")
            point_reliability_values = (
                np.load(
                    os.path.join(self.artifact_dir, point_reliability_name)
                ).astype(np.float32)
                if point_reliability_name
                else np.ones_like(point_scores, dtype=np.float32)
            )
            self.point_reliability = torch.from_numpy(
                point_reliability_values
            ).to(device)
            if self.point_reliability.shape != self.point_entropy.shape:
                raise ValueError("Point reliability must match group ID slots")
            quantization_error_name = manifest.get(
                "point_group_quantization_error"
            )
            if quantization_error_name:
                quantization_error_scale = float(
                    manifest["point_group_quantization_error_scale"]
                )
                if quantization_error_scale <= 0.0:
                    raise ValueError("Point quantization-error scale must be positive")
                packed_quantization_error = np.load(
                    os.path.join(self.artifact_dir, quantization_error_name)
                )
                if packed_quantization_error.dtype != np.uint8:
                    raise ValueError("Point quantization error must use uint8 storage")
                self.point_quantization_error = torch.from_numpy(
                    packed_quantization_error.astype(np.float32)
                    * quantization_error_scale
                ).to(device)
                if self.point_quantization_error.shape != self.point_entropy.shape:
                    raise ValueError(
                        "Point quantization error must match group ID slots"
                    )
                point_levels = group_levels[np.maximum(point_ids, 0)]
                usable = (point_ids >= 0) & (point_reliability_values > 0.0)
                quantization_percentile = np.zeros(
                    packed_quantization_error.shape, dtype=np.float32
                )
                for level in np.unique(point_levels[usable]):
                    level_mask = usable & (point_levels == level)
                    counts = np.bincount(
                        packed_quantization_error[level_mask], minlength=256
                    ).astype(np.float64)
                    midpoint = (np.cumsum(counts) - 0.5 * counts) / max(
                        counts.sum(), 1.0
                    )
                    quantization_percentile[level_mask] = midpoint[
                        packed_quantization_error[level_mask]
                    ]
                self.point_quantization_error_percentile = torch.from_numpy(
                    quantization_percentile
                ).to(device)
            else:
                self.point_quantization_error = None
                self.point_quantization_error_percentile = None
            reliability_name = manifest.get("group_reliability")
            group_reliability_values = (
                np.load(os.path.join(self.artifact_dir, reliability_name)).astype(np.float32)
                if reliability_name
                else np.ones(codebook.shape[0], dtype=np.float32)
            )
            self.group_reliability = torch.from_numpy(group_reliability_values).to(device)
            if independent_level_codebooks:
                safe_point_ids = np.maximum(point_ids, 0)
                point_levels = group_levels[safe_point_ids]
                point_local_ids = group_local_semantic_ids[safe_point_ids]
                occupancy_weight = (
                    point_scores
                    * point_reliability_values
                    * group_reliability_values[safe_point_ids]
                )
                occupancy_weight = np.where(point_ids >= 0, occupancy_weight, 0.0)
                level_code_priors = {}
                for level, level_codebook in level_codebooks.items():
                    level_mask = (point_ids >= 0) & (point_levels == level)
                    counts = np.bincount(
                        point_local_ids[level_mask],
                        weights=occupancy_weight[level_mask],
                        minlength=level_codebook.shape[0],
                    ).astype(np.float64)
                    counts += 0.5
                    level_code_priors[level] = (counts / counts.sum()).astype(np.float32)
        else:
            self.codebook_path = os.path.abspath(codebook_path)
            self.assignments_path = os.path.abspath(assignments_path)
            codebook = np.load(self.codebook_path).astype(np.float32)
            assignments = np.load(self.assignments_path)
            point_ids = assignments["top_group_ids"].astype(np.int64)
            point_scores = assignments["top_group_scores"].astype(np.float32)
            self._storage_bytes = os.path.getsize(self.codebook_path) + os.path.getsize(
                self.assignments_path
            )
            self.point_entropy = torch.zeros_like(torch.from_numpy(point_scores)).to(device)
            self.point_reliability = torch.ones_like(
                torch.from_numpy(point_scores)
            ).to(device)
            self.point_quantization_error = None
            self.point_quantization_error_percentile = None
            self.group_reliability = torch.ones(
                codebook.shape[0], dtype=torch.float32, device=device
            )
            competitor_ids = np.full_like(point_ids, -1)
            atom_codebook = np.zeros_like(codebook)
            group_levels = None
            level_codebooks = None
            group_local_semantic_ids = None
            level_code_priors = None
        codebook /= np.maximum(np.linalg.norm(codebook, axis=-1, keepdims=True), 1e-8)
        self.codebook = torch.from_numpy(codebook).to(device)
        atom_codebook /= np.maximum(
            np.linalg.norm(atom_codebook, axis=-1, keepdims=True), 1e-8
        )
        self.atom_codebook = torch.from_numpy(atom_codebook).to(device)
        self.point_ids = torch.from_numpy(point_ids).to(device)
        self.point_scores = torch.from_numpy(point_scores).to(device)
        self.competitor_ids = torch.from_numpy(competitor_ids).to(device)
        if group_levels is None:
            group_levels = np.zeros(codebook.shape[0], dtype=np.int64)
        self.group_levels = torch.from_numpy(group_levels).to(device)
        self.group_local_semantic_ids = (
            torch.from_numpy(group_local_semantic_ids).to(device)
            if group_local_semantic_ids is not None
            else None
        )
        self.level_codebooks = (
            {
                level: torch.from_numpy(level_codebook).to(device)
                for level, level_codebook in level_codebooks.items()
            }
            if level_codebooks is not None
            else None
        )
        self.level_code_priors = (
            {
                level: torch.from_numpy(prior).to(device)
                for level, prior in level_code_priors.items()
            }
            if level_code_priors is not None
            else None
        )
        if self.point_ids.shape != self.point_scores.shape:
            raise ValueError("Group IDs and scores must have matching shapes")
        valid = self.point_ids >= 0
        if valid.any() and int(self.point_ids[valid].max()) >= self.codebook.shape[0]:
            raise ValueError("Group assignments reference IDs outside the shared codebook")
        competitor_valid = self.competitor_ids >= 0
        if competitor_valid.any() and int(self.competitor_ids[competitor_valid].max()) >= self.codebook.shape[0]:
            raise ValueError("Competitor assignments reference IDs outside the shared codebook")
        if self.group_levels.shape != (self.codebook.shape[0],):
            raise ValueError("Group levels must match the decoded group codebook")
        self.feature_agreement = None
        self.query_activations = None
        self.atom_query_activations = None

    @torch.no_grad()
    def set_query_activations(self, clip_model, num_categories):
        self.query_activations = torch.cat(
            [
                clip_model.get_activation(self.codebook, category_index).float()
                for category_index in range(num_categories)
            ],
            dim=1,
        )
        self.atom_query_activations = torch.cat(
            [
                clip_model.get_activation(self.atom_codebook, category_index).float()
                for category_index in range(num_categories)
            ],
            dim=1,
        )

    @torch.no_grad()
    def set_feature_agreement_gate(self, point_codebook, floor, power, chunk_size=65536):
        """Gate track residuals unless track and local codebook agree semantically."""
        if floor < 0.0:
            self.feature_agreement = None
            return
        agreement = torch.zeros(self.num_gaussians, dtype=torch.float32, device=self.codebook.device)
        for start in range(0, self.num_gaussians, chunk_size):
            end = min(start + chunk_size, self.num_gaussians)
            point_features = point_codebook.reconstruct_range(start, end)
            ids = self.point_ids[start:end, 0]
            valid = ids >= 0
            if not valid.any():
                continue
            track_features = self.codebook[ids.clamp_min(0)]
            cosine = F.cosine_similarity(point_features, track_features, dim=-1).clamp(-1.0, 1.0)
            scaled = ((cosine - floor) / max(1e-8, 1.0 - floor)).clamp(0.0, 1.0)
            agreement[start:end] = torch.where(valid, scaled.pow(power), torch.zeros_like(scaled))
        self.feature_agreement = agreement

    @torch.no_grad()
    def candidate_activation(self, clip_model, category_index, topk):
        code_activation = (
            self.query_activations[:, category_index]
            if self.query_activations is not None
            else clip_model.get_activation(self.codebook, category_index).squeeze(-1)
        )
        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        point_scores = self.point_scores[:, :topk] if topk > 0 else self.point_scores
        valid = point_ids >= 0
        gathered = code_activation[point_ids.clamp_min(0)]
        gathered = torch.where(valid, gathered, torch.zeros_like(gathered))
        scores = torch.where(
            valid,
            point_scores.clamp(0.0, 1.0),
            torch.zeros_like(point_scores),
        )
        return gathered, scores, valid

    @torch.no_grad()
    def candidate_competitor_activation(self, clip_model, category_index, topk):
        code_activation = (
            self.query_activations[:, category_index]
            if self.query_activations is not None
            else clip_model.get_activation(self.codebook, category_index).squeeze(-1)
        )
        point_ids = (
            self.competitor_ids[:, :topk]
            if topk > 0
            else self.competitor_ids
        )
        valid = point_ids >= 0
        gathered = code_activation[point_ids.clamp_min(0)]
        return torch.where(valid, gathered, torch.zeros_like(gathered)), valid

    @torch.no_grad()
    def candidate_atom_activation(self, clip_model, category_index, topk):
        atom_activation = (
            self.atom_query_activations[:, category_index]
            if self.atom_query_activations is not None
            else clip_model.get_activation(
                self.atom_codebook, category_index
            ).squeeze(-1)
        )
        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        valid = (point_ids >= 0) & (
            self.atom_codebook.norm(dim=-1)[point_ids.clamp_min(0)] > 0.0
        )
        gathered = atom_activation[point_ids.clamp_min(0)]
        return torch.where(valid, gathered, torch.zeros_like(gathered)), valid

    @torch.no_grad()
    def candidate_query_specificity(self, category_index, topk):
        if self.query_activations is None:
            raise ValueError("Query activations must be initialized before margin routing")
        target = self.query_activations[:, category_index]
        if self.query_activations.shape[1] > 1:
            competitors = self.query_activations.clone()
            competitors[:, category_index] = -torch.inf
            margin = (target - competitors.max(dim=1).values).clamp_min(0.0)
        else:
            margin = torch.ones_like(target)
        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        valid = point_ids >= 0
        gathered = margin[point_ids.clamp_min(0)]
        return torch.where(valid, gathered, torch.zeros_like(gathered))

    @torch.no_grad()
    def candidate_reliability(self, topk):
        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        memberships = self.point_scores[:, :topk] if topk > 0 else self.point_scores
        entropy = self.point_entropy[:, :topk] if topk > 0 else self.point_entropy
        point_reliability = (
            self.point_reliability[:, :topk]
            if topk > 0
            else self.point_reliability
        )
        valid = point_ids >= 0
        track_reliability = self.group_reliability[point_ids.clamp_min(0)]
        reliability = (
            memberships
            * point_reliability.clamp(0.0, 1.0)
            * track_reliability
            * (1.0 - entropy)
        )
        return torch.where(valid, reliability, torch.zeros_like(reliability))

    @torch.no_grad()
    def candidate_levels(self, topk):
        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        valid = point_ids >= 0
        levels = self.group_levels[point_ids.clamp_min(0)]
        return torch.where(valid, levels, torch.full_like(levels, -1))

    @torch.no_grad()
    def candidate_quantization_error(self, topk):
        if self.point_quantization_error is None:
            raise ValueError(
                "Quantization-aware routing requires a rebuilt memory with "
                "point_group_quantization_error"
            )
        return (
            self.point_quantization_error[:, :topk]
            if topk > 0
            else self.point_quantization_error
        )

    @torch.no_grad()
    def candidate_quantization_error_percentile(self, topk):
        if self.point_quantization_error_percentile is None:
            raise ValueError(
                "Level-normalized routing requires point quantization-error metadata"
            )
        return (
            self.point_quantization_error_percentile[:, :topk]
            if topk > 0
            else self.point_quantization_error_percentile
        )

    @torch.no_grad()
    def candidate_level_calibration(
        self,
        clip_model,
        category_index,
        candidate_scores,
        candidate_levels,
        valid,
    ):
        """Calibrate token scores against each level's complete vocabulary."""
        if self.level_codebooks is None:
            raise ValueError("Level calibration requires independent level codebooks")
        if not (
            candidate_scores.shape == candidate_levels.shape == valid.shape
        ):
            raise ValueError("Candidate calibration tensors must match")

        percentiles = torch.zeros_like(candidate_scores)
        tail_evidence = torch.zeros_like(candidate_scores)
        level_stats = {}
        for level in sorted(self.level_codebooks):
            level_mask = valid & (candidate_levels == level)
            reference = clip_model.get_activation(
                self.level_codebooks[level], category_index
            ).squeeze(-1).float()
            sorted_reference = reference.sort().values
            count = int(sorted_reference.numel())
            if level_mask.any():
                ranks = torch.searchsorted(
                    sorted_reference,
                    candidate_scores[level_mask].contiguous(),
                    right=True,
                ).to(candidate_scores.dtype)
                cdf = ranks / float(count + 1)
                tail_probability = (float(count) - ranks + 1.0) / float(count + 1)
                percentiles[level_mask] = cdf
                tail_evidence[level_mask] = -tail_probability.clamp_min(1e-12).log()
            level_stats[f"level_{level}"] = {
                "vocabulary_size": count,
                "reference_mean": float(reference.mean().item()),
                "reference_std": float(reference.std(unbiased=False).item()),
            }
        return percentiles, tail_evidence, level_stats

    @torch.no_grad()
    def candidate_information_gain(
        self,
        clip_model,
        category_index,
        candidate_scores,
        candidate_levels,
        valid,
        temperature,
        counterfactual_neighborhoods=None,
        topk=0,
    ):
        """Compute global and local query evidence for four peer codebooks."""
        if self.level_codebooks is None or self.level_code_priors is None:
            raise ValueError("Information gain requires independent level codebooks")
        if self.group_local_semantic_ids is None:
            raise ValueError("Information gain requires local per-level semantic IDs")
        if not (
            candidate_scores.shape == candidate_levels.shape == valid.shape
        ):
            raise ValueError("Information-gain candidate tensors must match")
        if temperature <= 0.0:
            raise ValueError("Information-gain temperature must be positive")

        point_ids = self.point_ids[:, :topk] if topk > 0 else self.point_ids
        local_ids = self.group_local_semantic_ids[point_ids.clamp_min(0)]
        global_gain = torch.zeros_like(candidate_scores)
        local_gain = (
            torch.zeros_like(candidate_scores)
            if counterfactual_neighborhoods is not None
            else None
        )
        level_stats = {}
        for level in sorted(self.level_codebooks):
            level_mask = valid & (candidate_levels == level)
            reference = clip_model.get_activation(
                self.level_codebooks[level], category_index
            ).squeeze(-1).float()
            prior = self.level_code_priors[level].float().clamp_min(1e-12)
            log_prior = prior.log()
            log_partition = torch.logsumexp(
                log_prior + reference / temperature,
                dim=0,
            )
            if level_mask.any():
                selected_ids = local_ids[level_mask]
                if int(selected_ids.min()) < 0 or int(selected_ids.max()) >= reference.numel():
                    raise ValueError("Candidate local IDs exceed their level codebook")
                global_gain[level_mask] = (
                    candidate_scores[level_mask] / temperature - log_partition
                )
                if counterfactual_neighborhoods is not None:
                    neighbors = counterfactual_neighborhoods.neighbor_ids[level][
                        selected_ids
                    ]
                    local_log_evidence = torch.logsumexp(
                        reference[neighbors] / temperature,
                        dim=1,
                    ) - np.log(float(neighbors.shape[1]))
                    local_gain[level_mask] = (
                        candidate_scores[level_mask] / temperature
                        - local_log_evidence
                    )
            level_stats[f"level_{level}"] = {
                "vocabulary_size": int(reference.numel()),
                "log_partition": float(log_partition.item()),
                "prior_entropy": float((-(prior * log_prior).sum()).item()),
                "reference_mean": float(reference.mean().item()),
                "reference_std": float(reference.std(unbiased=False).item()),
            }
        return global_gain, local_gain, level_stats

    @torch.no_grad()
    def point_activation(
        self,
        clip_model,
        category_index,
        topk,
        mode,
        score_power,
        gate_floor,
        gate_power,
        use_membership_confidence=False,
    ):
        gathered, scores, valid = self.candidate_activation(
            clip_model, category_index, topk
        )
        if mode == "max":
            activation = gathered.max(dim=1, keepdim=True).values
        elif mode == "weighted":
            weights = scores.pow(score_power)
            weights /= weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            activation = (gathered * weights).sum(dim=1, keepdim=True)
        else:
            raise ValueError(f"Unknown group aggregation mode: {mode}")

        covered = valid.any(dim=1)
        first = scores[:, 0] if scores.shape[1] else torch.zeros_like(covered, dtype=torch.float32)
        if scores.shape[1] > 1:
            second = scores[:, 1]
        else:
            second = torch.zeros_like(first)
        margin = ((first - second).clamp_min(0.0) / first.clamp_min(1e-8)).pow(gate_power)
        confidence = torch.where(
            covered,
            gate_floor + (1.0 - gate_floor) * margin,
            torch.zeros_like(margin),
        ).unsqueeze(-1)
        if use_membership_confidence:
            confidence = confidence * first.clamp(0.0, 1.0).unsqueeze(-1)
        if self.feature_agreement is not None:
            confidence = confidence * self.feature_agreement.unsqueeze(-1)
        return activation, confidence

    @property
    def num_gaussians(self):
        return int(self.point_ids.shape[0])

    @property
    def storage_bytes(self):
        return self._storage_bytes


def main():
    parser = ArgumentParser(
        description="Evaluate LeRF-OVS with large shared codebooks and compact per-Gaussian IDs."
    )
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    parser.add_argument("--geometry_checkpoint", required=True)
    parser.add_argument("--codebook_dir", default=None)
    parser.add_argument(
        "--consensus_path",
        default=None,
        help="Evaluate cached full-precision consensus features instead of a codebook artifact.",
    )
    parser.add_argument(
        "--object_codebook_dir",
        default=None,
        help="Optional coarse object-code artifact composed with --codebook_dir at query time.",
    )
    parser.add_argument("--object_feature_weight", type=float, default=0.0)
    parser.add_argument("--hypothesis_dir", default=None)
    parser.add_argument(
        "--hypothesis_readout",
        choices=["switch", "reliability_blend"],
        default="switch",
    )
    parser.add_argument("--hypothesis_query_margin", action="store_true")
    parser.add_argument("--label_dir", required=True)
    parser.add_argument("--group_codebook", default=None)
    parser.add_argument("--group_assignments", default=None)
    parser.add_argument(
        "--group_hierarchy_dir",
        default=None,
        help="Compact shared group codebook with uint16 IDs and uint8 weights.",
    )
    parser.add_argument(
        "--spatial_group_posterior_dir",
        default=None,
        help="Top-2 Group support applied after four-token semantic retrieval.",
    )
    parser.add_argument("--group_anisotropic_geometry_dir", default=None)
    parser.add_argument("--group_topk", type=int, default=0)
    parser.add_argument("--group_aggregation", choices=["weighted", "max"], default="weighted")
    parser.add_argument("--group_score_power", type=float, default=1.0)
    parser.add_argument(
        "--group_readout",
        choices=[
            "residual",
            "hypothesis",
            "hypothesis_blend",
            "contrastive_blend",
            "dual_agreement",
            "dual_contrastive",
            "hierarchical_memory",
            "calibrated_hierarchical_memory",
            "equal_query_softmax",
            "equal_query_max",
            "equal_query_margin_top2",
            "equal_query_percentile_max",
            "equal_query_tail_max",
            "equal_query_information_gain",
            "equal_query_counterfactual_information_gain",
            "equal_query_quantization_lcb",
            "equal_query_quantization_percentile_lcb",
            "equal_query_relation_graph",
            "equal_query_multiscale_set_relation",
            "equal_query_spatial_posterior",
            "equal_query_spatial_geodesic",
            "equal_query_global_sparsemax",
            "equal_query_global_entmax15",
            "equal_query_global_entmax15_geodesic",
            "equal_query_global_anchor_entmax15",
            "equal_query_global_anchor_null_entmax15",
            "equal_query_global_anchor_null_entmax15_geodesic",
            "equal_query_seeded_group_completion",
            "equal_query_anisotropic_group_completion",
            "equal_query_seed_anisotropic_group_completion",
            "equal_query_profile_group_completion",
        ],
        default="residual",
        help="Keep group semantics separate at query time or use the legacy residual.",
    )
    parser.add_argument(
        "--group_query_temperature",
        type=float,
        default=0.10,
        help="Temperature for query-aware fusion across resident hierarchy levels.",
    )
    parser.add_argument("--spatial_posterior_maximum_penalty", type=float, default=0.06)
    parser.add_argument("--spatial_posterior_ring_weight", type=float, default=1.0)
    parser.add_argument("--spatial_posterior_contrast_temperature", type=float, default=0.05)
    parser.add_argument("--spatial_posterior_core_membership", type=float, default=0.30)
    parser.add_argument("--spatial_posterior_entropy_relaxation", type=float, default=0.75)
    parser.add_argument("--spatial_posterior_geodesic_delta", type=float, default=0.05)
    parser.add_argument("--spatial_posterior_recovery_factor", type=float, default=0.20)
    parser.add_argument("--global_group_temperature", type=float, default=0.05)
    parser.add_argument("--global_group_semantic_weight", type=float, default=0.75)
    parser.add_argument("--global_group_ring_contrast_strength", type=float, default=0.50)
    parser.add_argument("--global_group_maximum_penalty", type=float, default=0.08)
    parser.add_argument("--global_group_entropy_relaxation", type=float, default=0.50)
    parser.add_argument("--global_group_anchor_quantile", type=float, default=0.20)
    parser.add_argument("--global_group_outside_quantile", type=float, default=0.75)
    parser.add_argument("--global_group_anchor_temperature", type=float, default=0.02)
    parser.add_argument(
        "--global_group_semantic_preservation_quantile",
        type=float,
        default=-1.0,
    )
    parser.add_argument("--group_completion_seed_quantile", type=float, default=0.75)
    parser.add_argument("--group_completion_seed_support", type=float, default=0.95)
    parser.add_argument("--group_completion_seed_score_floor", type=float, default=0.55)
    parser.add_argument("--group_completion_target_quantile", type=float, default=0.20)
    parser.add_argument("--group_completion_boundary_membership", type=float, default=0.05)
    parser.add_argument("--group_completion_semantic_delta", type=float, default=0.10)
    parser.add_argument("--group_completion_agreement_temperature", type=float, default=0.02)
    parser.add_argument("--group_completion_strength", type=float, default=0.75)
    parser.add_argument("--group_completion_max_expansion_ratio", type=float, default=2.0)
    parser.add_argument("--group_completion_minimum_seed_points", type=int, default=16)
    parser.add_argument("--group_completion_minimum_contact", type=float, default=0.05)
    parser.add_argument("--group_completion_maximum_hops", type=int, default=32)
    parser.add_argument("--group_completion_anisotropic_axis_floor", type=float, default=0.15)
    parser.add_argument("--group_completion_anisotropic_budget_floor", type=float, default=0.25)
    parser.add_argument("--group_completion_anisotropic_semantic_floor", type=float, default=0.50)
    parser.add_argument("--group_completion_anisotropic_direction_power", type=float, default=1.0)
    parser.add_argument("--group_completion_profile_quantile", type=float, default=0.90)
    parser.add_argument("--group_completion_profile_margin", type=float, default=0.03)
    parser.add_argument("--group_completion_profile_temperature", type=float, default=0.01)
    parser.add_argument("--group_completion_profile_minimum_slots", type=int, default=2)
    parser.add_argument(
        "--group_query_tie_margin",
        type=float,
        default=0.0,
        help="Blend the top two peer tokens when their adjusted-score gap is below this margin.",
    )
    parser.add_argument(
        "--group_quantization_uncertainty_scale",
        type=float,
        default=0.05,
        help="Map codebook chord error to a score-space interval radius.",
    )
    parser.add_argument("--group_relation_graph_dir", default=None)
    parser.add_argument(
        "--group_counterfactual_codebook_dir",
        default=None,
        help="Same-level nearest-codeword alternatives for counterfactual information gain.",
    )
    parser.add_argument(
        "--group_relation_positive_strength",
        type=float,
        default=0.20,
        help="Positive-edge score smoothing strength for one graph step.",
    )
    parser.add_argument(
        "--group_relation_negative_strength",
        type=float,
        default=0.10,
        help="Negative-edge boundary sharpening strength for one graph step.",
    )
    parser.add_argument(
        "--group_relation_maximum_delta",
        type=float,
        default=0.05,
        help="Maximum absolute query-score change from relation correction.",
    )
    parser.add_argument("--group_level_margin_threshold", type=float, default=0.25)
    parser.add_argument("--group_level_margin_temperature", type=float, default=0.10)
    parser.add_argument("--group_competitor_weight", type=float, default=1.0)
    parser.add_argument("--group_route_fraction", type=float, default=1.0)
    parser.add_argument(
        "--group_route_priority",
        choices=[
            "query_gain",
            "membership_gain",
            "query_margin_gain",
            "membership_margin_gain",
            "reliability_gain",
            "reliability_margin_gain",
        ],
        default="query_gain",
    )
    parser.add_argument("--rgr_alpha", type=float, default=0.0)
    parser.add_argument(
        "--rgr_mode",
        choices=["positive", "convex"],
        default="positive",
        help="Positive residual preserves local activation; convex mode also suppresses track-disagreed peaks.",
    )
    parser.add_argument("--point_gate_floor", type=float, default=0.1)
    parser.add_argument("--point_gate_power", type=float, default=1.0)
    parser.add_argument(
        "--group_membership_confidence",
        action="store_true",
        help="Multiply the group gate by the absolute point/group membership score.",
    )
    parser.add_argument(
        "--group_feature_agreement_floor",
        type=float,
        default=-1.0,
        help="Disable below zero; otherwise gate track residuals by point/track feature cosine.",
    )
    parser.add_argument("--group_feature_agreement_power", type=float, default=1.0)
    parser.add_argument("--activation_chunk", type=int, default=65536)
    parser.add_argument(
        "--ignore_consensus_semantic_opacity",
        action="store_true",
        help="Ablate a semantic-opacity table stored in a continuous consensus.",
    )
    parser.add_argument(
        "--consensus_semantic_opacity_scale",
        type=float,
        default=1.0,
        help="Training-derived scale correction applied before clipping semantic opacity to one.",
    )
    parser.add_argument(
        "--consensus_blend_base",
        default=None,
        help="Optional continuous A6 base interpolated with --consensus_path before readout.",
    )
    parser.add_argument("--consensus_candidate_weight", type=float, default=1.0)
    parser.add_argument(
        "--consensus_candidate_weight_mode",
        choices=["constant", "base_gate", "inverse_base_gate"],
        default="constant",
    )
    parser.add_argument(
        "--consensus_query_route",
        choices=["none", "margin_switch", "margin_positive", "query_positive"],
        default="none",
    )
    parser.add_argument(
        "--query_route_base_codebook_dir",
        default=None,
        help="Second ID table in the same vocabulary used for discrete query-margin routing.",
    )
    parser.add_argument(
        "--codebook_query_route",
        choices=[
            "none",
            "margin_switch",
            "margin_positive",
            "query_positive",
            "query_positive_blend",
        ],
        default="none",
    )
    parser.add_argument(
        "--query_route_candidate_mask",
        default=None,
        help="Optional training-derived boolean mask or [0,1] reliability for blend routing.",
    )
    parser.add_argument(
        "--semantic_keep_mask",
        default=None,
        help="Optional boolean NPY mask that neutralizes low-confidence Gaussians.",
    )
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--evaluation_protocol",
        choices=["diagnostic", PROTOCOL_NAME],
        default=PROTOCOL_NAME,
        help="Paper 3D selection is the default; diagnostic must be requested explicitly.",
    )
    parser.add_argument(
        "--score_calibration",
        choices=["none", "frame_minmax", "frame_percentile", "category_percentile"],
        default="none",
    )
    parser.add_argument("--calibration_low", type=float, default=1.0)
    parser.add_argument("--calibration_high", type=float, default=99.0)
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.25, 0.3, 0.35, 0.4, 0.45, 0.5],
    )
    parser.add_argument(
        "--selection_thresholds",
        nargs="+",
        type=float,
        default=[value / 100.0 for value in range(10, 91, 5)],
        help="Per-Gaussian relevancy grid used only by drsplat_3d_selection.",
    )
    parser.add_argument("--occupancy_threshold", type=float, default=0.7)
    parser.add_argument("--save_visualizations", action="store_true")
    parser.add_argument("--max_visualizations", type=int, default=32)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(sys.argv[1:])

    if bool(args.codebook_dir) == bool(args.consensus_path):
        raise ValueError("Provide exactly one of --codebook_dir or --consensus_path")
    if (args.group_codebook is None) != (args.group_assignments is None):
        raise ValueError("--group_codebook and --group_assignments must be provided together")
    if args.group_hierarchy_dir and args.group_codebook:
        raise ValueError(
            "--group_hierarchy_dir cannot be combined with legacy group files"
        )
    if not 0.0 <= args.rgr_alpha:
        raise ValueError("--rgr_alpha must be non-negative")
    if args.object_feature_weight < 0.0:
        raise ValueError("--object_feature_weight must be non-negative")
    if args.consensus_path and args.object_codebook_dir:
        raise ValueError("--object_codebook_dir requires --codebook_dir")
    if args.hypothesis_dir and (args.group_hierarchy_dir or args.group_codebook):
        raise ValueError("Sparse hypothesis and group hierarchy must be evaluated separately")
    if not 0.0 <= args.point_gate_floor <= 1.0:
        raise ValueError("--point_gate_floor must be in [0, 1]")
    if args.point_gate_power <= 0.0 or args.activation_chunk <= 0:
        raise ValueError("Gate power and activation chunk must be positive")
    if not -1.0 <= args.group_feature_agreement_floor < 1.0:
        raise ValueError("--group_feature_agreement_floor must be in [-1, 1)")
    if args.group_feature_agreement_power <= 0.0:
        raise ValueError("--group_feature_agreement_power must be positive")
    if args.consensus_semantic_opacity_scale <= 0.0:
        raise ValueError("--consensus_semantic_opacity_scale must be positive")
    if (
        args.ignore_consensus_semantic_opacity
        or args.consensus_semantic_opacity_scale != 1.0
    ) and not args.consensus_path:
        raise ValueError("Consensus semantic-opacity controls require --consensus_path")
    if args.consensus_blend_base and not args.consensus_path:
        raise ValueError("--consensus_blend_base requires --consensus_path")
    if args.consensus_query_route != "none" and not args.consensus_blend_base:
        raise ValueError("Consensus query routing requires --consensus_blend_base")
    if bool(args.query_route_base_codebook_dir) != (
        args.codebook_query_route != "none"
    ):
        raise ValueError(
            "Discrete query routing requires both --query_route_base_codebook_dir "
            "and --codebook_query_route"
        )
    if args.query_route_base_codebook_dir and not args.codebook_dir:
        raise ValueError("Discrete query routing requires --codebook_dir")
    if args.query_route_candidate_mask and not args.query_route_base_codebook_dir:
        raise ValueError("Candidate route mask requires discrete query routing")
    if not 0.0 <= args.consensus_candidate_weight <= 1.0:
        raise ValueError("--consensus_candidate_weight must be in [0, 1]")
    if not 0.0 <= args.group_route_fraction <= 1.0:
        raise ValueError("--group_route_fraction must be in [0, 1]")
    if args.group_competitor_weight < 0.0:
        raise ValueError("--group_competitor_weight must be non-negative")
    if args.group_query_temperature <= 0.0:
        raise ValueError("--group_query_temperature must be positive")
    spatial_readout = args.group_readout in {
        "equal_query_spatial_posterior",
        "equal_query_spatial_geodesic",
        "equal_query_global_sparsemax",
        "equal_query_global_entmax15",
        "equal_query_global_entmax15_geodesic",
        "equal_query_global_anchor_entmax15",
        "equal_query_global_anchor_null_entmax15",
        "equal_query_global_anchor_null_entmax15_geodesic",
        "equal_query_seeded_group_completion",
        "equal_query_anisotropic_group_completion",
        "equal_query_seed_anisotropic_group_completion",
        "equal_query_profile_group_completion",
    }
    if spatial_readout != bool(args.spatial_group_posterior_dir):
        raise ValueError(
            "A spatial-posterior readout and --spatial_group_posterior_dir are required together"
        )
    if args.spatial_posterior_maximum_penalty < 0.0:
        raise ValueError("--spatial_posterior_maximum_penalty must be non-negative")
    if args.spatial_posterior_contrast_temperature <= 0.0:
        raise ValueError("--spatial_posterior_contrast_temperature must be positive")
    if args.spatial_posterior_core_membership <= 0.0:
        raise ValueError("--spatial_posterior_core_membership must be positive")
    if not 0.0 <= args.spatial_posterior_entropy_relaxation <= 1.0:
        raise ValueError("--spatial_posterior_entropy_relaxation must be in [0, 1]")
    if not 0.0 <= args.spatial_posterior_recovery_factor <= 1.0:
        raise ValueError("--spatial_posterior_recovery_factor must be in [0, 1]")
    if args.global_group_temperature <= 0.0:
        raise ValueError("--global_group_temperature must be positive")
    if not 0.0 <= args.global_group_semantic_weight <= 1.0:
        raise ValueError("--global_group_semantic_weight must be in [0, 1]")
    if args.global_group_ring_contrast_strength < 0.0:
        raise ValueError("--global_group_ring_contrast_strength must be non-negative")
    if args.global_group_maximum_penalty < 0.0:
        raise ValueError("--global_group_maximum_penalty must be non-negative")
    if not 0.0 <= args.global_group_entropy_relaxation <= 1.0:
        raise ValueError("--global_group_entropy_relaxation must be in [0, 1]")
    if not 0.0 <= args.global_group_anchor_quantile <= 1.0:
        raise ValueError("--global_group_anchor_quantile must be in [0, 1]")
    if not 0.0 <= args.global_group_outside_quantile <= 1.0:
        raise ValueError("--global_group_outside_quantile must be in [0, 1]")
    if args.global_group_anchor_temperature <= 0.0:
        raise ValueError("--global_group_anchor_temperature must be positive")
    if not (
        args.global_group_semantic_preservation_quantile == -1.0
        or 0.0 <= args.global_group_semantic_preservation_quantile <= 1.0
    ):
        raise ValueError(
            "--global_group_semantic_preservation_quantile must be -1 or in [0, 1]"
        )
    if (
        args.group_readout in {
            "equal_query_seeded_group_completion",
            "equal_query_anisotropic_group_completion",
            "equal_query_seed_anisotropic_group_completion",
            "equal_query_profile_group_completion",
        }
        and args.global_group_semantic_preservation_quantile < 0.0
    ):
        raise ValueError(
            "Seeded Group completion requires a semantic-preservation quantile"
        )
    if not 0.0 <= args.group_completion_seed_quantile <= 1.0:
        raise ValueError("--group_completion_seed_quantile must be in [0, 1]")
    if not 0.0 <= args.group_completion_seed_support <= 1.0:
        raise ValueError("--group_completion_seed_support must be in [0, 1]")
    if not np.isfinite(args.group_completion_seed_score_floor):
        raise ValueError("--group_completion_seed_score_floor must be finite")
    if not 0.0 <= args.group_completion_target_quantile <= 1.0:
        raise ValueError("--group_completion_target_quantile must be in [0, 1]")
    if not 0.0 < args.group_completion_boundary_membership <= args.spatial_posterior_core_membership:
        raise ValueError("--group_completion_boundary_membership must be in (0, core]")
    if args.group_completion_semantic_delta < 0.0:
        raise ValueError("--group_completion_semantic_delta must be non-negative")
    if args.group_completion_agreement_temperature <= 0.0:
        raise ValueError("--group_completion_agreement_temperature must be positive")
    if not 0.0 <= args.group_completion_strength <= 1.0:
        raise ValueError("--group_completion_strength must be in [0, 1]")
    if args.group_completion_max_expansion_ratio < 0.0:
        raise ValueError("--group_completion_max_expansion_ratio must be non-negative")
    if args.group_completion_minimum_seed_points <= 0:
        raise ValueError("--group_completion_minimum_seed_points must be positive")
    if args.group_completion_minimum_contact < 0.0:
        raise ValueError("--group_completion_minimum_contact must be non-negative")
    if args.group_completion_maximum_hops <= 0:
        raise ValueError("--group_completion_maximum_hops must be positive")
    if not 0.0 < args.group_completion_anisotropic_axis_floor <= 1.0:
        raise ValueError("--group_completion_anisotropic_axis_floor must be in (0, 1]")
    if not 0.0 <= args.group_completion_anisotropic_budget_floor <= 1.0:
        raise ValueError("--group_completion_anisotropic_budget_floor must be in [0, 1]")
    if not 0.0 <= args.group_completion_anisotropic_semantic_floor <= 1.0:
        raise ValueError("--group_completion_anisotropic_semantic_floor must be in [0, 1]")
    if args.group_completion_anisotropic_direction_power <= 0.0:
        raise ValueError("--group_completion_anisotropic_direction_power must be positive")
    if not 0.0 <= args.group_completion_profile_quantile <= 1.0:
        raise ValueError("--group_completion_profile_quantile must be in [0, 1]")
    if args.group_completion_profile_margin < 0.0:
        raise ValueError("--group_completion_profile_margin must be non-negative")
    if args.group_completion_profile_temperature <= 0.0:
        raise ValueError("--group_completion_profile_temperature must be positive")
    if not 1 <= args.group_completion_profile_minimum_slots <= args.group_topk:
        raise ValueError("--group_completion_profile_minimum_slots must be in [1, group_topk]")
    anisotropic_readout = args.group_readout in {
        "equal_query_anisotropic_group_completion",
        "equal_query_seed_anisotropic_group_completion",
    }
    if anisotropic_readout != bool(args.group_anisotropic_geometry_dir):
        raise ValueError(
            "Anisotropic Group completion and --group_anisotropic_geometry_dir are required together"
        )
    if args.group_query_tie_margin < 0.0:
        raise ValueError("--group_query_tie_margin must be non-negative")
    if args.group_quantization_uncertainty_scale < 0.0:
        raise ValueError(
            "--group_quantization_uncertainty_scale must be non-negative"
        )
    if (
        args.group_relation_positive_strength < 0.0
        or args.group_relation_negative_strength < 0.0
    ):
        raise ValueError("Relation graph strengths must be non-negative")
    if args.group_relation_maximum_delta <= 0.0:
        raise ValueError("--group_relation_maximum_delta must be positive")
    relation_readout = args.group_readout in {
        "equal_query_relation_graph",
        "equal_query_multiscale_set_relation",
    }
    if relation_readout != bool(args.group_relation_graph_dir):
        raise ValueError(
            "A relation readout and --group_relation_graph_dir are required together"
        )
    counterfactual_readout = (
        args.group_readout == "equal_query_counterfactual_information_gain"
    )
    if counterfactual_readout != bool(args.group_counterfactual_codebook_dir):
        raise ValueError(
            "equal_query_counterfactual_information_gain and "
            "--group_counterfactual_codebook_dir are required together"
        )
    if args.group_level_margin_threshold < 0.0:
        raise ValueError("--group_level_margin_threshold must be non-negative")
    if args.group_level_margin_temperature <= 0.0:
        raise ValueError("--group_level_margin_temperature must be positive")
    if args.evaluation_protocol == PROTOCOL_NAME:
        if args.score_calibration != "none":
            raise ValueError("Paper 3D-selection evaluation forbids score calibration")
        if not 0.0 <= args.occupancy_threshold <= 1.0:
            raise ValueError("--occupancy_threshold must be in [0, 1]")

    safe_state(args.quiet)
    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    opt = SimpleNamespace(include_feature=False)
    labels, categories = load_lerf_labels(args.label_dir)
    default_eval_name = (
        "gaussian_codebook_paper_selection"
        if args.evaluation_protocol == PROTOCOL_NAME
        else "gaussian_codebook"
    )
    output_dir = args.output or os.path.join(dataset.model_path, "eval", default_eval_name)
    os.makedirs(output_dir, exist_ok=True)

    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)
    checkpoint_iteration = load_geometry_checkpoint(
        scene.gaussians,
        args.geometry_checkpoint,
    )
    cameras = {camera.image_name: camera for camera in scene.getTrainCameras()}
    missing_cameras = sorted(set(labels) - set(cameras))
    if missing_cameras:
        raise ValueError(f"Missing labeled cameras in scene: {missing_cameras}")

    codebook = (
        GaussianCodebookArtifact(args.codebook_dir)
        if args.codebook_dir
        else ConsensusFeatureArtifact(args.consensus_path)
    )
    if isinstance(codebook, ConsensusFeatureArtifact):
        if args.ignore_consensus_semantic_opacity:
            codebook.semantic_opacity = None
            codebook.manifest["semantic_opacity_readout"] = "ignored"
        elif codebook.semantic_opacity is not None:
            codebook.semantic_opacity = (
                codebook.semantic_opacity * args.consensus_semantic_opacity_scale
            ).clamp(0.0, 1.0)
            codebook.manifest["semantic_opacity_readout"] = "scaled_one_sided"
            codebook.manifest["semantic_opacity_scale"] = float(
                args.consensus_semantic_opacity_scale
            )
        if args.consensus_blend_base:
            codebook.blend_with_consensus(
                args.consensus_blend_base,
                args.consensus_candidate_weight,
                args.activation_chunk,
                args.consensus_candidate_weight_mode,
                args.consensus_query_route != "none",
            )
    if codebook.num_gaussians != scene.gaussians.get_xyz.shape[0]:
        raise ValueError("Gaussian codebook size does not match the geometry checkpoint")
    query_route_base_codebook = None
    if args.query_route_base_codebook_dir:
        query_route_base_codebook = GaussianCodebookArtifact(
            args.query_route_base_codebook_dir
        )
        if (
            query_route_base_codebook.num_gaussians != codebook.num_gaussians
            or query_route_base_codebook.feature_dim != codebook.feature_dim
        ):
            raise ValueError("Discrete query-route codebooks must match")
    query_route_candidate_mask = None
    if args.query_route_candidate_mask:
        route_mask = np.load(args.query_route_candidate_mask)
        if route_mask.shape != (codebook.num_gaussians,):
            raise ValueError("Candidate route mask does not match the Gaussian count")
        route_dtype = (
            np.float32
            if args.codebook_query_route == "query_positive_blend"
            else bool
        )
        query_route_candidate_mask = torch.from_numpy(
            np.asarray(route_mask, dtype=route_dtype)
        ).to("cuda")
        if args.codebook_query_route == "query_positive_blend":
            if not torch.isfinite(query_route_candidate_mask).all():
                raise ValueError("Candidate route reliability must be finite")
            if (
                (query_route_candidate_mask < 0.0).any()
                or (query_route_candidate_mask > 1.0).any()
            ):
                raise ValueError("Candidate route reliability must be in [0, 1]")
    semantic_keep_mask = None
    if args.semantic_keep_mask:
        keep_mask = np.load(args.semantic_keep_mask)
        if keep_mask.shape != (codebook.num_gaussians,):
            raise ValueError("Semantic keep mask does not match the Gaussian count")
        semantic_keep_mask = torch.from_numpy(
            np.asarray(keep_mask, dtype=bool)
        ).to("cuda")
    group_hierarchy = None
    spatial_posterior = None
    anisotropic_geometry = None
    relation_graph = None
    counterfactual_neighborhoods = None
    object_codebook = None
    sparse_hypothesis = None
    if args.object_codebook_dir:
        object_codebook = GaussianCodebookArtifact(args.object_codebook_dir)
        if (
            object_codebook.num_gaussians != codebook.num_gaussians
            or object_codebook.feature_dim != codebook.feature_dim
        ):
            raise ValueError("Object codebook must match the fine codebook geometry and feature dimension")
    if args.group_hierarchy_dir:
        group_hierarchy = GroupHierarchy(artifact_dir=args.group_hierarchy_dir)
    elif args.group_codebook:
        group_hierarchy = GroupHierarchy(args.group_codebook, args.group_assignments)
        if group_hierarchy.num_gaussians != codebook.num_gaussians:
            raise ValueError("Group assignments do not match the Gaussian codebook")
    if group_hierarchy is not None:
        if group_hierarchy.num_gaussians != codebook.num_gaussians:
            raise ValueError("Group hierarchy does not match the Gaussian codebook")
        group_hierarchy.set_feature_agreement_gate(
            codebook,
            args.group_feature_agreement_floor,
            args.group_feature_agreement_power,
            args.activation_chunk,
        )
    if args.spatial_group_posterior_dir:
        spatial_posterior = QueryConditionedSpatialPosterior(
            args.spatial_group_posterior_dir
        )
        if spatial_posterior.num_gaussians != codebook.num_gaussians:
            raise ValueError("Spatial posterior does not match the Gaussian codebook")
    if args.group_anisotropic_geometry_dir:
        anisotropic_geometry = GroupAnisotropicGeometry(
            args.group_anisotropic_geometry_dir
        )
        if spatial_posterior is None:
            raise ValueError("Anisotropic geometry requires a spatial posterior")
        if (
            anisotropic_geometry.num_gaussians != codebook.num_gaussians
            or anisotropic_geometry.num_atoms != spatial_posterior.num_atoms
            or anisotropic_geometry.num_groups != spatial_posterior.core_keys.shape[0]
        ):
            raise ValueError("Anisotropic geometry does not match the spatial posterior")
    if args.group_relation_graph_dir:
        if args.group_readout == "equal_query_multiscale_set_relation":
            relation_graph = MultiscaleSetRelationGraph(
                args.group_relation_graph_dir
            )
        else:
            relation_graph = SignedGaussianRelationGraph(
                args.group_relation_graph_dir
            )
        if relation_graph.num_gaussians != codebook.num_gaussians:
            raise ValueError("Relation graph does not match the Gaussian codebook")
    if args.group_counterfactual_codebook_dir:
        if group_hierarchy is None or group_hierarchy.artifact_dir is None:
            raise ValueError("Counterfactual codebooks require a resident hierarchy artifact")
        counterfactual_neighborhoods = CounterfactualCodebookNeighborhoods(
            args.group_counterfactual_codebook_dir
        )
        counterfactual_neighborhoods.validate_memory(group_hierarchy.artifact_dir)
    if args.hypothesis_dir:
        sparse_hypothesis = SparseSemanticHypothesis(args.hypothesis_dir)
        if (
            sparse_hypothesis.num_gaussians != codebook.num_gaussians
            or sparse_hypothesis.feature_dim != codebook.feature_dim
        ):
            raise ValueError("Sparse hypothesis must match the base codebook")

    clip_model = OpenCLIPNetwork("cuda")
    clip_model.set_positives(categories)
    if (
        isinstance(codebook, ConsensusFeatureArtifact)
        and args.consensus_query_route != "none"
    ):
        codebook.prepare_query_routing(
            clip_model,
            len(categories),
            args.consensus_query_route,
            args.activation_chunk,
        )
    route_base_scores = None
    route_candidate_scores = None
    discrete_route_diagnostics = {}
    if query_route_base_codebook is not None:
        route_base_scores = precompute_artifact_query_scores(
            query_route_base_codebook,
            clip_model,
            len(categories),
            args.activation_chunk,
        )
        route_candidate_scores = precompute_artifact_query_scores(
            codebook,
            clip_model,
            len(categories),
            args.activation_chunk,
        )
    if sparse_hypothesis is not None:
        sparse_hypothesis.set_query_activations(clip_model, len(categories))
    if group_hierarchy is not None and (
        "margin" in args.group_route_priority
        or args.group_readout in {"dual_agreement", "dual_contrastive"}
    ):
        group_hierarchy.set_query_activations(clip_model, len(categories))
    if spatial_posterior is not None:
        spatial_posterior.set_query_activations(clip_model, len(categories))
    background = torch.zeros(3, dtype=torch.float32, device="cuda")
    thresholds = sorted(set(args.thresholds))
    route_diagnostics = {}

    def activation_provider(category_index):
        if route_base_scores is not None:
            activation, selected = route_query_activation(
                route_base_scores,
                route_candidate_scores,
                category_index,
                args.codebook_query_route,
                query_route_candidate_mask,
            )
            discrete_route_diagnostics[categories[category_index]] = {
                "candidate_fraction": float(selected.float().mean())
            }
        else:
            activation = codebook.query_activation(
                clip_model,
                category_index,
                chunk_size=args.activation_chunk,
                object_codebook=object_codebook,
                object_feature_weight=args.object_feature_weight,
            )
        if sparse_hypothesis is not None:
            candidate, reliability, valid, specificity = sparse_hypothesis.candidate_tables(
                category_index,
                args.hypothesis_query_margin,
            )
            if args.hypothesis_readout == "switch":
                priority = "query_margin_gain" if args.hypothesis_query_margin else "query_gain"
                activation, diagnostics = route_group_hypotheses(
                    activation,
                    candidate,
                    torch.ones_like(reliability),
                    valid,
                    1.0,
                    priority,
                    specificity,
                    reliability,
                )
            else:
                activation, diagnostics = blend_sparse_hypothesis(
                    activation,
                    candidate,
                    reliability,
                    valid,
                    specificity,
                )
            route_diagnostics[categories[category_index]] = diagnostics
        elif group_hierarchy is not None and args.group_readout in {
            "hypothesis",
            "hypothesis_blend",
            "contrastive_blend",
            "dual_agreement",
            "dual_contrastive",
            "hierarchical_memory",
            "calibrated_hierarchical_memory",
            "equal_query_softmax",
            "equal_query_max",
            "equal_query_margin_top2",
            "equal_query_percentile_max",
            "equal_query_tail_max",
            "equal_query_information_gain",
            "equal_query_counterfactual_information_gain",
            "equal_query_quantization_lcb",
            "equal_query_quantization_percentile_lcb",
            "equal_query_relation_graph",
            "equal_query_multiscale_set_relation",
            "equal_query_spatial_posterior",
            "equal_query_spatial_geodesic",
            "equal_query_global_sparsemax",
            "equal_query_global_entmax15",
            "equal_query_global_entmax15_geodesic",
            "equal_query_global_anchor_entmax15",
            "equal_query_global_anchor_null_entmax15",
            "equal_query_global_anchor_null_entmax15_geodesic",
            "equal_query_seeded_group_completion",
            "equal_query_anisotropic_group_completion",
            "equal_query_seed_anisotropic_group_completion",
            "equal_query_profile_group_completion",
        }:
            candidate_scores, memberships, valid = group_hierarchy.candidate_activation(
                clip_model,
                category_index,
                args.group_topk,
            )
            query_specificity = (
                group_hierarchy.candidate_query_specificity(
                    category_index,
                    args.group_topk,
                )
                if "margin" in args.group_route_priority
                else None
            )
            candidate_reliability = (
                group_hierarchy.candidate_reliability(args.group_topk)
                if (
                    "reliability" in args.group_route_priority
                    or args.group_readout == "hypothesis_blend"
                    or args.group_readout == "hierarchical_memory"
                    or args.group_readout == "calibrated_hierarchical_memory"
                    or args.group_readout == "equal_query_softmax"
                    or args.group_readout == "equal_query_max"
                    or args.group_readout == "equal_query_margin_top2"
                    or args.group_readout == "equal_query_percentile_max"
                    or args.group_readout == "equal_query_tail_max"
                    or args.group_readout == "equal_query_information_gain"
                    or args.group_readout
                    == "equal_query_counterfactual_information_gain"
                    or args.group_readout == "equal_query_quantization_lcb"
                    or args.group_readout
                    == "equal_query_quantization_percentile_lcb"
                    or args.group_readout == "equal_query_relation_graph"
                    or args.group_readout == "equal_query_multiscale_set_relation"
                    or args.group_readout == "equal_query_spatial_posterior"
                    or args.group_readout == "equal_query_spatial_geodesic"
                    or args.group_readout == "equal_query_global_sparsemax"
                    or args.group_readout == "equal_query_global_entmax15"
                    or args.group_readout == "equal_query_global_entmax15_geodesic"
                    or args.group_readout == "equal_query_global_anchor_entmax15"
                    or args.group_readout
                    == "equal_query_global_anchor_null_entmax15"
                    or args.group_readout
                    == "equal_query_global_anchor_null_entmax15_geodesic"
                    or args.group_readout == "equal_query_seeded_group_completion"
                    or args.group_readout == "equal_query_anisotropic_group_completion"
                    or args.group_readout == "equal_query_seed_anisotropic_group_completion"
                    or args.group_readout == "equal_query_profile_group_completion"
                )
                else None
            )
            if args.group_readout in {"dual_agreement", "dual_contrastive"}:
                atom_scores, atom_valid = group_hierarchy.candidate_atom_activation(
                    clip_model, category_index, args.group_topk
                )
                competitor_scores, competitor_valid = (
                    group_hierarchy.candidate_competitor_activation(
                        clip_model, category_index, args.group_topk
                    )
                )
                dual_valid = valid & atom_valid
                activation, diagnostics = blend_dual_code_hypotheses(
                    activation,
                    atom_scores,
                    candidate_scores,
                    competitor_scores,
                    memberships,
                    candidate_reliability,
                    dual_valid,
                    competitor_valid,
                    args.group_readout == "dual_contrastive",
                )
            elif args.group_readout == "contrastive_blend":
                competitor_scores, competitor_valid = (
                    group_hierarchy.candidate_competitor_activation(
                        clip_model,
                        category_index,
                        args.group_topk,
                    )
                )
                activation, diagnostics = blend_contrastive_group_hypotheses(
                    activation,
                    candidate_scores,
                    competitor_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    competitor_valid,
                    args.group_competitor_weight,
                )
            elif args.group_readout == "hypothesis_blend":
                activation, diagnostics = blend_group_hypotheses(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                )
            elif args.group_readout == "hierarchical_memory":
                activation, diagnostics = fuse_hierarchical_semantic_memory(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    args.group_query_temperature,
                    group_hierarchy.candidate_levels(args.group_topk),
                )
            elif args.group_readout == "calibrated_hierarchical_memory":
                activation, diagnostics = fuse_calibrated_hierarchical_memory(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    args.group_query_temperature,
                    group_hierarchy.candidate_levels(args.group_topk),
                    args.group_level_margin_threshold,
                    args.group_level_margin_temperature,
                )
            elif args.group_readout in {
                "equal_query_global_sparsemax",
                "equal_query_global_entmax15",
                "equal_query_global_entmax15_geodesic",
                "equal_query_global_anchor_entmax15",
                "equal_query_global_anchor_null_entmax15",
                "equal_query_global_anchor_null_entmax15_geodesic",
                "equal_query_seeded_group_completion",
                "equal_query_anisotropic_group_completion",
                "equal_query_seed_anisotropic_group_completion",
                "equal_query_profile_group_completion",
            }:
                candidate_levels = group_hierarchy.candidate_levels(args.group_topk)
                mode = "sparsemax" if args.group_readout == "equal_query_global_sparsemax" else "entmax15"
                global_tables = spatial_posterior.global_candidate_tables(
                    category_index,
                    candidate_levels,
                    candidate_scores,
                    mode,
                    args.global_group_temperature,
                    args.global_group_semantic_weight,
                    args.global_group_ring_contrast_strength,
                )
                confidence, probability, spatial_membership, spatial_entropy, spatial_valid, group_stats = global_tables
                geodesic = args.group_readout in {
                    "equal_query_global_entmax15_geodesic",
                    "equal_query_global_anchor_null_entmax15_geodesic",
                }
                seeded_completion = args.group_readout in {
                    "equal_query_seeded_group_completion",
                    "equal_query_anisotropic_group_completion",
                    "equal_query_seed_anisotropic_group_completion",
                    "equal_query_profile_group_completion",
                }
                anisotropic_completion = args.group_readout in {
                    "equal_query_anisotropic_group_completion",
                    "equal_query_seed_anisotropic_group_completion",
                }
                seed_anisotropic_completion = (
                    args.group_readout == "equal_query_seed_anisotropic_group_completion"
                )
                profile_completion = (
                    args.group_readout == "equal_query_profile_group_completion"
                )
                anchored = "global_anchor" in args.group_readout or seeded_completion
                use_null_expert = "anchor_null" in args.group_readout
                activation, diagnostics = fuse_global_sparse_group_retrieval(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    args.group_query_temperature,
                    confidence,
                    probability,
                    spatial_membership,
                    spatial_entropy,
                    spatial_valid,
                    args.global_group_maximum_penalty,
                    args.spatial_posterior_core_membership,
                    args.global_group_entropy_relaxation,
                    gaussian_atom_ids=(
                        spatial_posterior.gaussian_atom_ids if geodesic else None
                    ),
                    atom_neighbor_ids=(
                        spatial_posterior.atom_neighbor_ids if geodesic else None
                    ),
                    geodesic_delta=args.spatial_posterior_geodesic_delta,
                    recovery_factor=args.spatial_posterior_recovery_factor,
                    anchor_quantile=(
                        args.global_group_anchor_quantile if anchored else None
                    ),
                    outside_quantile=args.global_group_outside_quantile,
                    anchor_temperature=args.global_group_anchor_temperature,
                    use_null_expert=use_null_expert,
                    semantic_preservation_quantile=(
                        args.global_group_semantic_preservation_quantile
                        if args.global_group_semantic_preservation_quantile >= 0.0
                        else None
                    ),
                )
                diagnostics["global_group_posterior"] = group_stats
                if seeded_completion:
                    group_ids, group_id_valid = spatial_posterior.candidate_group_ids(
                        candidate_levels
                    )
                    completion_valid = spatial_valid & group_id_valid
                    activation, completion_diagnostics = complete_scores_from_seeded_groups(
                        activation,
                        candidate_scores,
                        memberships,
                        candidate_reliability,
                        valid,
                        args.group_query_temperature,
                        group_ids,
                        confidence,
                        spatial_membership,
                        completion_valid,
                        spatial_posterior.gaussian_atom_ids,
                        spatial_posterior.atom_neighbor_ids,
                        spatial_posterior.atom_neighbor_weights,
                        args.spatial_posterior_core_membership,
                        args.group_completion_boundary_membership,
                        args.group_completion_seed_support,
                        args.group_completion_seed_quantile,
                        args.group_completion_seed_score_floor,
                        args.group_completion_target_quantile,
                        args.group_completion_semantic_delta,
                        args.group_completion_agreement_temperature,
                        args.group_completion_strength,
                        args.group_completion_max_expansion_ratio,
                        args.group_completion_minimum_seed_points,
                        args.group_completion_minimum_contact,
                        args.group_completion_maximum_hops,
                        atom_centroids=(
                            anisotropic_geometry.atom_centroids
                            if anisotropic_completion else None
                        ),
                        group_principal_axes=(
                            anisotropic_geometry.group_principal_axes
                            if anisotropic_completion else None
                        ),
                        group_axis_ratios=(
                            anisotropic_geometry.group_axis_ratios
                            if anisotropic_completion else None
                        ),
                        anisotropic_axis_floor=(
                            args.group_completion_anisotropic_axis_floor
                        ),
                        anisotropic_budget_floor=(
                            args.group_completion_anisotropic_budget_floor
                        ),
                        anisotropic_semantic_floor=(
                            args.group_completion_anisotropic_semantic_floor
                        ),
                        anisotropic_direction_power=(
                            args.group_completion_anisotropic_direction_power
                        ),
                        seed_conditioned_anisotropy=seed_anisotropic_completion,
                        token_profile_gate=profile_completion,
                        token_profile_quantile=args.group_completion_profile_quantile,
                        token_profile_margin=args.group_completion_profile_margin,
                        token_profile_temperature=args.group_completion_profile_temperature,
                        token_profile_minimum_slots=(
                            args.group_completion_profile_minimum_slots
                        ),
                    )
                    diagnostics["seeded_group_completion"] = completion_diagnostics
            elif args.group_readout in {
                "equal_query_spatial_posterior",
                "equal_query_spatial_geodesic",
            }:
                candidate_levels = group_hierarchy.candidate_levels(args.group_topk)
                spatial_tables = spatial_posterior.candidate_tables(
                    category_index, candidate_levels
                )
                geodesic = args.group_readout == "equal_query_spatial_geodesic"
                activation, diagnostics = fuse_query_conditioned_spatial_posterior(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    args.group_query_temperature,
                    *spatial_tables,
                    args.spatial_posterior_ring_weight,
                    args.spatial_posterior_contrast_temperature,
                    args.spatial_posterior_maximum_penalty,
                    args.spatial_posterior_core_membership,
                    args.spatial_posterior_entropy_relaxation,
                    gaussian_atom_ids=(
                        spatial_posterior.gaussian_atom_ids if geodesic else None
                    ),
                    atom_neighbor_ids=(
                        spatial_posterior.atom_neighbor_ids if geodesic else None
                    ),
                    geodesic_delta=args.spatial_posterior_geodesic_delta,
                    recovery_factor=args.spatial_posterior_recovery_factor,
                )
            elif args.group_readout in {
                "equal_query_softmax",
                "equal_query_max",
                "equal_query_margin_top2",
                "equal_query_relation_graph",
                "equal_query_multiscale_set_relation",
            }:
                candidate_levels = group_hierarchy.candidate_levels(args.group_topk)
                set_relation_diagnostics = None
                if args.group_readout == "equal_query_multiscale_set_relation":
                    selectable = (
                        valid
                        & (memberships.clamp(0.0, 1.0) > 0.0)
                        & (candidate_reliability.clamp(0.0, 1.0) > 0.0)
                    )
                    candidate_scores, set_relation_diagnostics = (
                        fuse_multiscale_set_relation_token_scores(
                            candidate_scores,
                            candidate_levels,
                            selectable,
                            relation_graph.neighbor_ids,
                            relation_graph.signed_weights,
                            args.group_relation_positive_strength,
                            args.group_relation_negative_strength,
                            args.group_relation_maximum_delta,
                        )
                    )
                activation, diagnostics = fuse_equal_query_tokens(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    args.group_query_temperature,
                    candidate_levels,
                    hard=args.group_readout != "equal_query_softmax",
                    tie_margin=(
                        args.group_query_tie_margin
                        if args.group_readout == "equal_query_margin_top2"
                        else 0.0
                    ),
                )
                if args.group_readout == "equal_query_relation_graph":
                    activation, relation_diagnostics = (
                        fuse_signed_relation_graph_scores(
                            activation,
                            relation_graph.neighbor_ids,
                            relation_graph.signed_weights,
                            args.group_relation_positive_strength,
                            args.group_relation_negative_strength,
                            args.group_relation_maximum_delta,
                        )
                    )
                    diagnostics["relation_graph"] = relation_diagnostics
                if set_relation_diagnostics is not None:
                    diagnostics["multiscale_set_relation"] = set_relation_diagnostics
            elif args.group_readout in {
                "equal_query_percentile_max",
                "equal_query_tail_max",
            }:
                candidate_levels = group_hierarchy.candidate_levels(args.group_topk)
                percentiles, tail_evidence, calibration_stats = (
                    group_hierarchy.candidate_level_calibration(
                        clip_model,
                        category_index,
                        candidate_scores,
                        candidate_levels,
                        valid,
                    )
                )
                calibration_mode = (
                    "percentile"
                    if args.group_readout == "equal_query_percentile_max"
                    else "tail_evidence"
                )
                calibration_scores = (
                    percentiles
                    if calibration_mode == "percentile"
                    else tail_evidence
                )
                activation, diagnostics = fuse_calibrated_equal_query_tokens(
                    activation,
                    candidate_scores,
                    calibration_scores,
                    memberships,
                    candidate_reliability,
                    valid,
                    calibration_mode,
                    args.group_query_temperature,
                    candidate_levels,
                )
                diagnostics["level_reference_distributions"] = calibration_stats
            elif args.group_readout in {
                "equal_query_information_gain",
                "equal_query_counterfactual_information_gain",
            }:
                candidate_levels = group_hierarchy.candidate_levels(args.group_topk)
                information_gain, local_gain, information_stats = (
                    group_hierarchy.candidate_information_gain(
                        clip_model,
                        category_index,
                        candidate_scores,
                        candidate_levels,
                        valid,
                        args.group_query_temperature,
                        counterfactual_neighborhoods=(
                            counterfactual_neighborhoods
                            if args.group_readout
                            == "equal_query_counterfactual_information_gain"
                            else None
                        ),
                        topk=args.group_topk,
                    )
                )
                activation, diagnostics = fuse_information_gain_equal_query_tokens(
                    activation,
                    candidate_scores,
                    information_gain,
                    memberships,
                    candidate_reliability,
                    valid,
                    candidate_levels,
                    local_gain,
                )
                diagnostics["information_gain_reference"] = information_stats
            elif args.group_readout in {
                "equal_query_quantization_lcb",
                "equal_query_quantization_percentile_lcb",
            }:
                percentile_error = (
                    args.group_readout
                    == "equal_query_quantization_percentile_lcb"
                )
                activation, diagnostics = fuse_quantization_aware_equal_query_tokens(
                    activation,
                    candidate_scores,
                    memberships,
                    candidate_reliability,
                    (
                        group_hierarchy.candidate_quantization_error_percentile(
                            args.group_topk
                        )
                        if percentile_error
                        else group_hierarchy.candidate_quantization_error(
                            args.group_topk
                        )
                    ),
                    valid,
                    args.group_query_temperature,
                    args.group_quantization_uncertainty_scale,
                    group_hierarchy.candidate_levels(args.group_topk),
                    uncertainty_measure_name=(
                        "within_level_error_midrank_percentile"
                        if percentile_error
                        else "chord_error"
                    ),
                )
            else:
                activation, diagnostics = route_group_hypotheses(
                    activation,
                    candidate_scores,
                    memberships,
                    valid,
                    args.group_route_fraction,
                    args.group_route_priority,
                    query_specificity,
                    candidate_reliability,
                )
            route_diagnostics[categories[category_index]] = diagnostics
        elif group_hierarchy is not None and args.rgr_alpha > 0.0:
            group_activation, confidence = group_hierarchy.point_activation(
                clip_model,
                category_index,
                args.group_topk,
                args.group_aggregation,
                args.group_score_power,
                args.point_gate_floor,
                args.point_gate_power,
                args.group_membership_confidence,
            )
            residual = group_activation - activation
            if args.rgr_mode == "positive":
                residual = F.relu(residual)
            activation = activation + args.rgr_alpha * confidence * residual
        if semantic_keep_mask is not None:
            activation = torch.where(
                semantic_keep_mask.unsqueeze(-1),
                activation,
                torch.zeros_like(activation),
            )
        return activation

    semantic_storage = int(codebook.manifest["storage"]["total_semantic_bytes"])
    if object_codebook is not None:
        semantic_storage += int(object_codebook.manifest["storage"]["total_semantic_bytes"])
    if group_hierarchy is not None:
        semantic_storage += int(group_hierarchy.storage_bytes)
    if spatial_posterior is not None:
        semantic_storage += int(spatial_posterior.storage_bytes)
    if anisotropic_geometry is not None:
        semantic_storage += int(anisotropic_geometry.storage_bytes)
    if relation_graph is not None:
        semantic_storage += relation_graph.storage_bytes
    if counterfactual_neighborhoods is not None:
        semantic_storage += counterfactual_neighborhoods.storage_bytes
    if sparse_hypothesis is not None:
        semantic_storage += sparse_hypothesis.storage_bytes
    if query_route_base_codebook is not None:
        semantic_storage += int(
            query_route_base_codebook.manifest["storage"]["total_semantic_bytes"]
        )

    if args.evaluation_protocol == PROTOCOL_NAME:
        results = evaluate_paper_3d_selection(
            scene,
            pipe,
            opt,
            labels,
            categories,
            cameras,
            activation_provider,
            args.selection_thresholds,
            args.occupancy_threshold,
            output_dir,
            args.save_visualizations,
            args.max_visualizations,
        )
        results.update(
            {
                "source_path": dataset.source_path,
                "model_path": dataset.model_path,
                "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
                "geometry_checkpoint_iteration": checkpoint_iteration,
                "codebook_dir": codebook.dir if args.codebook_dir else None,
                "consensus_path": codebook.dir if args.consensus_path else None,
                "codebook_manifest": codebook.manifest,
                "object_codebook_dir": object_codebook.dir if object_codebook else None,
                "object_codebook_manifest": object_codebook.manifest if object_codebook else None,
                "object_feature_weight": float(args.object_feature_weight),
                "query_route_base_codebook_dir": query_route_base_codebook.dir
                if query_route_base_codebook
                else None,
                "query_route_base_codebook_manifest": query_route_base_codebook.manifest
                if query_route_base_codebook
                else None,
                "codebook_query_route": args.codebook_query_route,
                "query_route_candidate_mask": os.path.abspath(
                    args.query_route_candidate_mask
                )
                if args.query_route_candidate_mask
                else None,
                "query_route_candidate_mask_fraction": float(
                    query_route_candidate_mask.float().mean()
                )
                if query_route_candidate_mask is not None
                else 1.0,
                "discrete_route_diagnostics": discrete_route_diagnostics,
                "hypothesis_dir": sparse_hypothesis.dir if sparse_hypothesis else None,
                "hypothesis_manifest": sparse_hypothesis.manifest if sparse_hypothesis else None,
                "hypothesis_readout": args.hypothesis_readout,
                "hypothesis_query_margin": bool(args.hypothesis_query_margin),
                "group_codebook": group_hierarchy.codebook_path if group_hierarchy else None,
                "group_assignments": group_hierarchy.assignments_path if group_hierarchy else None,
                "group_hierarchy_dir": group_hierarchy.artifact_dir if group_hierarchy else None,
                "group_relation_graph_dir": relation_graph.dir if relation_graph else None,
                "group_relation_graph_manifest": relation_graph.manifest
                if relation_graph
                else None,
                "group_counterfactual_codebook_dir": (
                    counterfactual_neighborhoods.dir
                    if counterfactual_neighborhoods
                    else None
                ),
                "group_counterfactual_codebook_manifest": (
                    counterfactual_neighborhoods.manifest
                    if counterfactual_neighborhoods
                    else None
                ),
                "group_relation_positive_strength": float(
                    args.group_relation_positive_strength
                ),
                "group_relation_negative_strength": float(
                    args.group_relation_negative_strength
                ),
                "group_relation_maximum_delta": float(
                    args.group_relation_maximum_delta
                ),
                "rgr_alpha": float(args.rgr_alpha),
                "rgr_mode": args.rgr_mode,
                "group_topk": int(args.group_topk),
                "group_aggregation": args.group_aggregation,
                "group_score_power": float(args.group_score_power),
                "point_gate_floor": float(args.point_gate_floor),
                "point_gate_power": float(args.point_gate_power),
                "group_membership_confidence": bool(args.group_membership_confidence),
                "group_readout": args.group_readout,
                "group_query_temperature": float(args.group_query_temperature),
                "group_query_tie_margin": float(args.group_query_tie_margin),
                "group_quantization_uncertainty_scale": float(
                    args.group_quantization_uncertainty_scale
                ),
                "group_level_margin_threshold": float(args.group_level_margin_threshold),
                "group_level_margin_temperature": float(args.group_level_margin_temperature),
                "group_competitor_weight": float(args.group_competitor_weight),
                "group_route_fraction": float(args.group_route_fraction),
                "group_route_priority": args.group_route_priority,
                "route_diagnostics": route_diagnostics,
                "group_feature_agreement_floor": float(
                    args.group_feature_agreement_floor
                ),
                "group_feature_agreement_power": float(
                    args.group_feature_agreement_power
                ),
                "semantic_storage_bytes": semantic_storage,
                "semantic_storage_megabytes": semantic_storage / (1024.0 ** 2),
                "semantic_keep_mask": os.path.abspath(args.semantic_keep_mask)
                if args.semantic_keep_mask
                else None,
                "semantic_kept_fraction": float(semantic_keep_mask.float().mean())
                if semantic_keep_mask is not None
                else 1.0,
                "label_dir": os.path.abspath(args.label_dir),
                "num_label_frames": len(labels),
                "num_categories": len(categories),
            }
        )
        metrics_path = os.path.join(output_dir, "metrics.json")
        with open(metrics_path, "w") as output:
            json.dump(results, output, indent=2)
        print(json.dumps(results, indent=2))
        print(f"Saved metrics to {metrics_path}")
        return

    per_category = {}
    visualization_count = 0

    with torch.no_grad():
        rgb_cache = {}
        for category_index, category in enumerate(tqdm(categories, desc="Evaluating categories")):
            activation = activation_provider(category_index)

            frame_scores = {}
            frame_ground_truth = {}
            for image_name, label_data in labels.items():
                if category not in label_data["objects"]:
                    continue
                camera = cameras[image_name]
                rendered = render(
                    camera,
                    scene.gaussians,
                    pipe,
                    background,
                    opt,
                    override_color=activation.repeat(1, 3),
                )["render"]
                score = rendered.mean(dim=0).detach().cpu().numpy()
                ground_truth = polygons_to_mask(
                    label_data["objects"][category],
                    label_data["width"],
                    label_data["height"],
                )
                if score.shape != ground_truth.shape:
                    score_image = Image.fromarray(
                        (np.clip(score, 0.0, 1.0) * 255).astype(np.uint8)
                    )
                    score = np.asarray(
                        score_image.resize(
                            (ground_truth.shape[1], ground_truth.shape[0]),
                            Image.BILINEAR,
                        ),
                        dtype=np.float32,
                    ) / 255.0
                frame_scores[image_name] = score
                frame_ground_truth[image_name] = ground_truth

            if not frame_scores:
                continue
            frame_scores = calibrate_frame_scores(
                frame_scores,
                args.score_calibration,
                args.calibration_low,
                args.calibration_high,
            )
            threshold_results = []
            for threshold in thresholds:
                intersection = 0
                union = 0
                for image_name, score in frame_scores.items():
                    prediction = score > threshold
                    ground_truth = frame_ground_truth[image_name]
                    intersection += int(np.logical_and(prediction, ground_truth).sum())
                    union += int(np.logical_or(prediction, ground_truth).sum())
                threshold_results.append(
                    {
                        "threshold": threshold,
                        "iou": float(intersection / union) if union else 0.0,
                    }
                )
            best = max(threshold_results, key=lambda item: item["iou"])
            per_category[category] = {
                "best_iou": best["iou"],
                "best_threshold": best["threshold"],
                "num_frames": len(frame_scores),
                "thresholds": threshold_results,
            }

            if args.save_visualizations and visualization_count < args.max_visualizations:
                for image_name, score in frame_scores.items():
                    if visualization_count >= args.max_visualizations:
                        break
                    if image_name not in rgb_cache:
                        rgb_cache[image_name] = cameras[image_name].original_image.detach().cpu()
                    safe_category = category.replace("/", "_").replace(" ", "_")
                    save_visualization(
                        rgb_cache[image_name],
                        score,
                        score > best["threshold"],
                        frame_ground_truth[image_name],
                        os.path.join(
                            output_dir,
                            "visualizations",
                            f"{image_name}_{safe_category}.png",
                        ),
                    )
                    visualization_count += 1

    oracle_ious = [item["best_iou"] for item in per_category.values()]
    global_threshold_summary = []
    for threshold in thresholds:
        threshold_ious = []
        for item in per_category.values():
            match = next(
                result for result in item["thresholds"] if result["threshold"] == threshold
            )
            threshold_ious.append(match["iou"])
        global_threshold_summary.append(
            {
                "threshold": threshold,
                "mIoU": float(np.mean(threshold_ious)) if threshold_ious else 0.0,
                "mAcc@0.25": float(np.mean(np.asarray(threshold_ious) >= 0.25))
                if threshold_ious
                else 0.0,
            }
        )

    results = {
        "source_path": dataset.source_path,
        "model_path": dataset.model_path,
        "geometry_checkpoint": os.path.abspath(args.geometry_checkpoint),
        "geometry_checkpoint_iteration": checkpoint_iteration,
        "codebook_dir": codebook.dir if args.codebook_dir else None,
        "consensus_path": codebook.dir if args.consensus_path else None,
        "codebook_manifest": codebook.manifest,
        "object_codebook_dir": object_codebook.dir if object_codebook else None,
        "object_codebook_manifest": object_codebook.manifest if object_codebook else None,
        "object_feature_weight": float(args.object_feature_weight),
        "hypothesis_dir": sparse_hypothesis.dir if sparse_hypothesis else None,
        "hypothesis_manifest": sparse_hypothesis.manifest if sparse_hypothesis else None,
        "hypothesis_readout": args.hypothesis_readout,
        "hypothesis_query_margin": bool(args.hypothesis_query_margin),
        "group_codebook": group_hierarchy.codebook_path if group_hierarchy else None,
        "group_assignments": group_hierarchy.assignments_path if group_hierarchy else None,
        "group_hierarchy_dir": group_hierarchy.artifact_dir if group_hierarchy else None,
        "spatial_group_posterior_dir": (
            spatial_posterior.dir if spatial_posterior else None
        ),
        "spatial_group_posterior_manifest": (
            spatial_posterior.manifest if spatial_posterior else None
        ),
        "group_anisotropic_geometry_dir": (
            anisotropic_geometry.dir if anisotropic_geometry else None
        ),
        "group_anisotropic_geometry_manifest": (
            anisotropic_geometry.manifest if anisotropic_geometry else None
        ),
        "group_counterfactual_codebook_dir": (
            counterfactual_neighborhoods.dir
            if counterfactual_neighborhoods
            else None
        ),
        "group_counterfactual_codebook_manifest": (
            counterfactual_neighborhoods.manifest
            if counterfactual_neighborhoods
            else None
        ),
        "rgr_alpha": float(args.rgr_alpha),
        "rgr_mode": args.rgr_mode,
        "point_gate_floor": float(args.point_gate_floor),
        "point_gate_power": float(args.point_gate_power),
        "group_readout": args.group_readout,
        "group_query_temperature": float(args.group_query_temperature),
        "group_query_tie_margin": float(args.group_query_tie_margin),
        "spatial_posterior_maximum_penalty": float(
            args.spatial_posterior_maximum_penalty
        ),
        "spatial_posterior_ring_weight": float(args.spatial_posterior_ring_weight),
        "spatial_posterior_contrast_temperature": float(
            args.spatial_posterior_contrast_temperature
        ),
        "spatial_posterior_core_membership": float(
            args.spatial_posterior_core_membership
        ),
        "spatial_posterior_entropy_relaxation": float(
            args.spatial_posterior_entropy_relaxation
        ),
        "spatial_posterior_geodesic_delta": float(
            args.spatial_posterior_geodesic_delta
        ),
        "spatial_posterior_recovery_factor": float(
            args.spatial_posterior_recovery_factor
        ),
        "global_group_temperature": float(args.global_group_temperature),
        "global_group_semantic_weight": float(args.global_group_semantic_weight),
        "global_group_ring_contrast_strength": float(
            args.global_group_ring_contrast_strength
        ),
        "global_group_maximum_penalty": float(args.global_group_maximum_penalty),
        "global_group_entropy_relaxation": float(
            args.global_group_entropy_relaxation
        ),
        "global_group_anchor_quantile": float(args.global_group_anchor_quantile),
        "global_group_outside_quantile": float(args.global_group_outside_quantile),
        "global_group_anchor_temperature": float(
            args.global_group_anchor_temperature
        ),
        "global_group_semantic_preservation_quantile": float(
            args.global_group_semantic_preservation_quantile
        ),
        "group_completion_seed_quantile": float(args.group_completion_seed_quantile),
        "group_completion_seed_support": float(args.group_completion_seed_support),
        "group_completion_seed_score_floor": float(
            args.group_completion_seed_score_floor
        ),
        "group_completion_target_quantile": float(
            args.group_completion_target_quantile
        ),
        "group_completion_boundary_membership": float(
            args.group_completion_boundary_membership
        ),
        "group_completion_semantic_delta": float(
            args.group_completion_semantic_delta
        ),
        "group_completion_agreement_temperature": float(
            args.group_completion_agreement_temperature
        ),
        "group_completion_strength": float(args.group_completion_strength),
        "group_completion_max_expansion_ratio": float(
            args.group_completion_max_expansion_ratio
        ),
        "group_completion_minimum_seed_points": int(
            args.group_completion_minimum_seed_points
        ),
        "group_completion_minimum_contact": float(
            args.group_completion_minimum_contact
        ),
        "group_completion_maximum_hops": int(
            args.group_completion_maximum_hops
        ),
        "group_completion_anisotropic_axis_floor": float(
            args.group_completion_anisotropic_axis_floor
        ),
        "group_completion_anisotropic_budget_floor": float(
            args.group_completion_anisotropic_budget_floor
        ),
        "group_completion_anisotropic_semantic_floor": float(
            args.group_completion_anisotropic_semantic_floor
        ),
        "group_completion_anisotropic_direction_power": float(
            args.group_completion_anisotropic_direction_power
        ),
        "group_completion_profile_quantile": float(
            args.group_completion_profile_quantile
        ),
        "group_completion_profile_margin": float(args.group_completion_profile_margin),
        "group_completion_profile_temperature": float(
            args.group_completion_profile_temperature
        ),
        "group_completion_profile_minimum_slots": int(
            args.group_completion_profile_minimum_slots
        ),
        "group_quantization_uncertainty_scale": float(
            args.group_quantization_uncertainty_scale
        ),
        "group_level_margin_threshold": float(args.group_level_margin_threshold),
        "group_level_margin_temperature": float(args.group_level_margin_temperature),
        "group_competitor_weight": float(args.group_competitor_weight),
        "group_route_fraction": float(args.group_route_fraction),
        "group_route_priority": args.group_route_priority,
        "route_diagnostics": route_diagnostics,
        "group_feature_agreement_floor": float(args.group_feature_agreement_floor),
        "group_feature_agreement_power": float(args.group_feature_agreement_power),
        "semantic_storage_bytes": semantic_storage,
        "semantic_storage_megabytes": semantic_storage / (1024.0 ** 2),
        "num_categories": len(per_category),
        "score_calibration": args.score_calibration,
        "calibration_low": float(args.calibration_low),
        "calibration_high": float(args.calibration_high),
        "mIoU": float(np.mean(oracle_ious)) if oracle_ious else 0.0,
        "mAcc@0.25": float(np.mean(np.asarray(oracle_ious) >= 0.25))
        if oracle_ious
        else 0.0,
        "global_threshold_summary": global_threshold_summary,
        "best_global_threshold": max(
            global_threshold_summary,
            key=lambda item: item["mIoU"],
        )
        if global_threshold_summary
        else None,
        "per_category": per_category,
        "note": (
            "Inference evaluates cached full-precision consensus features."
            if args.consensus_path
            else "Inference loads shared 512D codebooks plus compact per-Gaussian IDs; "
            "it does not load Dr.Splat PQ codes or per-Gaussian continuous semantics."
        ),
    }
    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as output:
        json.dump(results, output, indent=2)
    print(json.dumps(results, indent=2))
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
