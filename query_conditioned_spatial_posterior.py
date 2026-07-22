"""Runtime loader for query-conditioned top-2 spatial Group evidence."""

import json
import os

import numpy as np
import torch


def sparse_probability(logits, temperature, mode):
    """Map one level's evidence to a sparse probability simplex."""
    if logits.ndim != 1 or not logits.numel():
        raise ValueError("Sparse Group logits must be a non-empty vector")
    if temperature <= 0.0:
        raise ValueError("Sparse Group temperature must be positive")
    scaled = logits / temperature
    scaled = scaled - scaled.max()
    if mode == "sparsemax":
        sorted_values = scaled.sort(descending=True).values
        cumulative = sorted_values.cumsum(dim=0)
        ranks = torch.arange(
            1, len(sorted_values) + 1, device=logits.device, dtype=logits.dtype
        )
        support = 1.0 + ranks * sorted_values > cumulative
        count = int(support.sum().item())
        threshold = (cumulative[count - 1] - 1.0) / float(count)
        probability = (scaled - threshold).clamp_min(0.0)
    elif mode == "entmax15":
        alpha_minus_one = 0.5
        low = scaled.min() - 2.0
        high = scaled.max()
        for _ in range(40):
            threshold = 0.5 * (low + high)
            probability = (
                alpha_minus_one * (scaled - threshold)
            ).clamp_min(0.0).pow(2.0)
            if float(probability.sum().item()) > 1.0:
                low = threshold
            else:
                high = threshold
        probability = (
            alpha_minus_one * (scaled - high)
        ).clamp_min(0.0).pow(2.0)
    else:
        raise ValueError(f"Unknown sparse Group posterior: {mode}")
    return probability / probability.sum().clamp_min(1e-8)


class QueryConditionedSpatialPosterior:
    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != "query_conditioned_top2_spatial_group_posterior":
            raise ValueError("Unsupported spatial posterior artifact")

        def tensor(name, dtype=None):
            value = np.load(os.path.join(self.dir, self.manifest[name]))
            result = torch.from_numpy(value)
            return result.to(device=device, dtype=dtype) if dtype else result.to(device)

        self.core_keys = tensor("group_core_keys", torch.float32)
        self.ring_keys = tensor("group_ring_keys", torch.float32)
        self.ring_valid = tensor("group_ring_valid", torch.bool)
        self.group_levels = tensor("group_level", torch.long)
        self.group_reliability = tensor("group_reliability", torch.float32)
        self.point_ids = tensor("point_group_ids", torch.long)
        self.point_memberships = tensor("point_group_memberships", torch.float32)
        self.point_entropy = tensor("point_group_entropy", torch.float32)
        self.gaussian_atom_ids = tensor("gaussian_atom_ids", torch.long)
        self.atom_neighbor_ids = tensor("atom_neighbor_ids", torch.long)
        self.atom_neighbor_weights = tensor("atom_neighbor_weights", torch.float32)
        self.num_gaussians = int(self.manifest["num_gaussians"])
        self.num_atoms = int(self.manifest["num_atoms"])
        if self.point_ids.shape != (self.num_gaussians, 4, 2):
            raise ValueError("Spatial Group IDs must have shape [N, 4, 2]")
        if self.point_memberships.shape != self.point_ids.shape:
            raise ValueError("Spatial memberships must match Group IDs")
        if self.point_entropy.shape != (self.num_gaussians, 4):
            raise ValueError("Spatial entropy must have shape [N, 4]")
        self.core_activations = None
        self.ring_activations = None

    @torch.no_grad()
    def set_query_activations(self, clip_model, num_categories):
        self.core_activations = torch.cat(
            [clip_model.get_activation(self.core_keys, index).float() for index in range(num_categories)],
            dim=1,
        )
        self.ring_activations = torch.cat(
            [clip_model.get_activation(self.ring_keys, index).float() for index in range(num_categories)],
            dim=1,
        )

    @torch.no_grad()
    def candidate_tables(self, category_index, candidate_levels):
        if self.core_activations is None:
            raise ValueError("Spatial query activations have not been initialized")
        safe_levels = candidate_levels.clamp(0, 3)
        gather_ids = safe_levels.unsqueeze(-1).expand(-1, -1, 2)
        point_ids = self.point_ids.gather(1, gather_ids)
        memberships = self.point_memberships.gather(1, gather_ids)
        entropy = self.point_entropy.gather(1, safe_levels)
        valid = (candidate_levels >= 0).unsqueeze(-1) & (point_ids >= 0)
        safe_ids = point_ids.clamp_min(0)
        core = self.core_activations[:, category_index][safe_ids]
        ring = self.ring_activations[:, category_index][safe_ids]
        ring_valid = self.ring_valid[safe_ids]
        ring = torch.where(ring_valid, ring, core)
        reliability = self.group_reliability[safe_ids]
        reliability = reliability * ring_valid.to(reliability.dtype)
        core = torch.where(valid, core, torch.zeros_like(core))
        ring = torch.where(valid, ring, torch.zeros_like(ring))
        reliability = torch.where(valid, reliability, torch.zeros_like(reliability))
        memberships = torch.where(valid, memberships, torch.zeros_like(memberships))
        return core, ring, memberships, reliability, entropy, valid

    @torch.no_grad()
    def candidate_group_ids(self, candidate_levels):
        """Return the persistent Group IDs aligned with resident token slots."""
        safe_levels = candidate_levels.clamp(0, 3)
        gather_ids = safe_levels.unsqueeze(-1).expand(-1, -1, 2)
        point_ids = self.point_ids.gather(1, gather_ids)
        valid = (candidate_levels >= 0).unsqueeze(-1) & (point_ids >= 0)
        return point_ids, valid

    @torch.no_grad()
    def global_candidate_tables(
        self,
        category_index,
        candidate_levels,
        candidate_scores,
        mode,
        temperature,
        semantic_weight,
        ring_contrast_strength,
    ):
        """Globally rank all Groups per level using key and resident-token evidence."""
        if not 0.0 <= semantic_weight <= 1.0:
            raise ValueError("semantic_weight must be in [0, 1]")
        if ring_contrast_strength < 0.0:
            raise ValueError("ring_contrast_strength must be non-negative")
        if candidate_levels.shape != candidate_scores.shape:
            raise ValueError("Candidate levels and semantic scores must match")
        safe_levels = candidate_levels.clamp(0, 3)
        gather_ids = safe_levels.unsqueeze(-1).expand(-1, -1, 2)
        point_ids = self.point_ids.gather(1, gather_ids)
        memberships = self.point_memberships.gather(1, gather_ids)
        entropy = self.point_entropy.gather(1, safe_levels)
        valid = (candidate_levels >= 0).unsqueeze(-1) & (point_ids >= 0) & (memberships > 0.0)
        group_count = self.core_keys.shape[0]
        group_probability = torch.zeros(
            group_count, dtype=torch.float32, device=self.core_keys.device
        )
        level_stats = {}
        for level in range(4):
            slot_mask = (candidate_levels == level).unsqueeze(-1) & valid
            flat_valid = slot_mask.reshape(-1)
            ids = point_ids.reshape(-1)[flat_valid]
            weights = memberships.reshape(-1)[flat_valid].float()
            scores = candidate_scores.unsqueeze(-1).expand_as(memberships).reshape(-1)[flat_valid].float()
            semantic_sum = torch.zeros_like(group_probability)
            semantic_mass = torch.zeros_like(group_probability)
            semantic_sum.scatter_add_(0, ids, scores * weights)
            semantic_mass.scatter_add_(0, ids, weights)
            groups = torch.nonzero(
                (self.group_levels == level) & (semantic_mass > 0.0), as_tuple=False
            ).squeeze(1)
            semantic_mean = semantic_sum[groups] / semantic_mass[groups].clamp_min(1e-8)
            core = self.core_activations[groups, category_index]
            ring = self.ring_activations[groups, category_index]
            ring = torch.where(self.ring_valid[groups], ring, core)
            key_evidence = core + ring_contrast_strength * (core - ring)
            evidence = semantic_weight * semantic_mean + (1.0 - semantic_weight) * key_evidence
            evidence = evidence + temperature * self.group_reliability[groups].clamp_min(1e-6).log()
            probability = sparse_probability(evidence, temperature, mode)
            group_probability[groups] = probability
            active = probability > 1e-8
            level_stats[f"level_{level}"] = {
                "candidate_groups": int(len(groups)),
                "active_groups": int(active.sum().item()),
                "maximum_probability": float(probability.max().item()),
                "posterior_entropy": float(
                    (-(probability[active] * probability[active].log()).sum()).item()
                )
                if active.any()
                else 0.0,
                "evidence_min": float(evidence.min().item()),
                "evidence_max": float(evidence.max().item()),
            }
        safe_ids = point_ids.clamp_min(0)
        probability = group_probability[safe_ids]
        probability = torch.where(valid, probability, torch.zeros_like(probability))
        level_maximum = torch.zeros(4, dtype=torch.float32, device=probability.device)
        for level in range(4):
            groups = self.group_levels == level
            level_maximum[level] = group_probability[groups].max() if groups.any() else 1.0
        confidence = probability / level_maximum[safe_levels].unsqueeze(-1).clamp_min(1e-8)
        confidence = torch.where(valid, confidence, torch.zeros_like(confidence))
        return confidence, probability, memberships, entropy, valid, level_stats

    @property
    def storage_bytes(self):
        return int(self.manifest["storage_bytes"])


class GroupAnisotropicGeometry:
    """Query-independent 3D shape tensors for anisotropic Group completion."""

    def __init__(self, artifact_dir, device="cuda"):
        self.dir = os.path.abspath(artifact_dir)
        with open(os.path.join(self.dir, "manifest.json")) as source:
            self.manifest = json.load(source)
        if self.manifest.get("representation") != "group_anisotropic_propagation_geometry":
            raise ValueError("Unsupported Group anisotropic geometry artifact")

        def tensor(name, dtype=None):
            value = np.load(os.path.join(self.dir, self.manifest[name]))
            result = torch.from_numpy(value)
            return result.to(device=device, dtype=dtype) if dtype else result.to(device)

        self.atom_centroids = tensor("atom_centroids", torch.float32)
        self.group_principal_axes = tensor("group_principal_axes", torch.float32)
        self.group_axis_ratios = tensor("group_axis_ratios", torch.float32)
        self.group_linearity = tensor("group_linearity", torch.float32)
        self.group_planarity = tensor("group_planarity", torch.float32)
        self.group_atom_counts = tensor("group_atom_counts", torch.long)
        self.num_atoms = int(self.manifest["num_atoms"])
        self.num_groups = int(self.manifest["num_groups"])
        self.num_gaussians = int(self.manifest["num_gaussians"])
        if self.atom_centroids.shape != (self.num_atoms, 3):
            raise ValueError("Atom centroids must have shape [A, 3]")
        if self.group_principal_axes.shape != (self.num_groups, 3, 3):
            raise ValueError("Group principal axes must have shape [G, 3, 3]")
        if self.group_axis_ratios.shape != (self.num_groups, 3):
            raise ValueError("Group axis ratios must have shape [G, 3]")

    @property
    def storage_bytes(self):
        return int(self.manifest["storage_bytes"])
