import numpy as np
import pytest

torch = pytest.importorskip("torch")

from prepare_semantic_field import (
    mask_interior_confidence,
    surface_responsibility_weights,
)
from eval_lerf_ovs_gaussian_codebook_miou import (
    ConsensusFeatureArtifact,
    precompute_artifact_query_scores,
    route_query_activation,
)


def test_mask_interior_confidence_penalizes_segment_boundaries():
    segmentation = np.array(
        [
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    confidence = mask_interior_confidence(segmentation, 2.0, 0.5)
    assert confidence[1, 0] > confidence[1, 2]
    assert confidence.min() >= 0.5
    assert confidence.max() <= 1.0


def test_surface_responsibility_prefers_front_depth_with_bounded_kl():
    point_ids = torch.tensor([[0, 1]])
    point_weights = torch.tensor([[0.5, 0.5]])
    depths = torch.tensor([1.0, 2.0])
    adjusted, kl, ratio = surface_responsibility_weights(
        point_ids,
        point_weights,
        depths,
        torch.ones(1),
        max_kl=0.02,
    )
    assert adjusted[0, 0] > adjusted[0, 1]
    assert adjusted.sum().item() == pytest.approx(1.0)
    assert kl.item() <= 0.02001
    assert ratio.item() <= 5.01


def test_surface_responsibility_preserves_boundary_confidence_mass():
    adjusted, _, _ = surface_responsibility_weights(
        torch.tensor([[0, 1]]),
        torch.tensor([[0.8, 0.2]]),
        torch.tensor([1.0, 1.01]),
        torch.tensor([0.5]),
    )
    assert adjusted.sum().item() == pytest.approx(0.5)


def test_continuous_consensus_supports_paper_query_activation():
    artifact = ConsensusFeatureArtifact.__new__(ConsensusFeatureArtifact)
    artifact.features = torch.tensor([[3.0, 4.0], [0.0, 0.0]])
    artifact.valid_mask = torch.tensor([True, False])
    artifact.semantic_opacity = None
    artifact.num_gaussians = 2
    artifact.feature_dim = 2
    artifact.device = "cpu"

    class ClipStub:
        def get_activation(self, features, _category_index):
            return features[:, :1]

    activation = artifact.query_activation(ClipStub(), 0, chunk_size=1)
    assert torch.allclose(activation, torch.tensor([[0.6], [0.0]]))


def test_continuous_consensus_applies_semantic_opacity_to_selection_score():
    artifact = ConsensusFeatureArtifact.__new__(ConsensusFeatureArtifact)
    artifact.features = torch.tensor([[3.0, 4.0], [4.0, 3.0]])
    artifact.valid_mask = torch.tensor([True, True])
    artifact.semantic_opacity = torch.tensor([0.5, 1.0])
    artifact.num_gaussians = 2
    artifact.feature_dim = 2
    artifact.device = "cpu"

    class ClipStub:
        def get_activation(self, features, _category_index):
            return features[:, :1]

    activation = artifact.query_activation(ClipStub(), 0, chunk_size=1)
    assert torch.allclose(activation, torch.tensor([[0.3], [0.8]]))


def test_continuous_consensus_blends_normalized_features(tmp_path):
    base_path = tmp_path / "base.pt"
    torch.save({"initial_features": torch.tensor([[1.0, 0.0]])}, base_path)
    artifact = ConsensusFeatureArtifact.__new__(ConsensusFeatureArtifact)
    artifact.features = torch.tensor([[0.0, 2.0]])
    artifact.num_gaussians = 1
    artifact.manifest = {}
    artifact.blend_with_consensus(base_path, 0.5, chunk_size=1)
    expected = torch.nn.functional.normalize(torch.tensor([[1.0, 1.0]]), dim=-1)
    assert torch.allclose(artifact.features, expected)
    assert artifact.manifest["feature_blend"]["candidate_weight"] == 0.5


def test_continuous_consensus_blend_uses_a6_fusion_gate(tmp_path):
    base_path = tmp_path / "base.pt"
    torch.save(
        {
            "initial_features": torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
            "fusion_gate": torch.tensor([0.0, 1.0]),
        },
        base_path,
    )
    artifact = ConsensusFeatureArtifact.__new__(ConsensusFeatureArtifact)
    artifact.features = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    artifact.num_gaussians = 2
    artifact.manifest = {}
    artifact.blend_with_consensus(base_path, 0.5, chunk_size=1, weight_mode="base_gate")
    expected = torch.tensor([[1.0, 0.0], [2**-0.5, 2**-0.5]])
    assert torch.allclose(artifact.features, expected, atol=1e-6)


def test_query_margin_switch_uses_candidate_only_for_better_margin():
    base = torch.tensor([[0.8, 0.4], [0.7, 0.1]])
    candidate = torch.tensor([[0.7, 0.1], [0.8, 0.4]])
    output, selected = route_query_activation(base, candidate, 0, "margin_switch")
    assert selected.tolist() == [True, False]
    assert torch.allclose(output[:, 0], torch.tensor([0.7, 0.7]))


def test_query_margin_positive_never_reduces_base_activation():
    base = torch.tensor([[0.8, 0.4], [0.7, 0.1]])
    candidate = torch.tensor([[0.7, 0.1], [0.8, 0.4]])
    output, selected = route_query_activation(base, candidate, 0, "margin_positive")
    assert selected.tolist() == [True, False]
    assert torch.allclose(output[:, 0], torch.tensor([0.8, 0.7]))


def test_query_positive_uses_only_current_query_score():
    base = torch.tensor([[0.8, 0.9], [0.7, 0.1]])
    candidate = torch.tensor([[0.7, 0.1], [0.8, 0.9]])
    output, selected = route_query_activation(base, candidate, 0, "query_positive")
    assert selected.tolist() == [False, True]
    assert torch.allclose(output[:, 0], torch.tensor([0.8, 0.8]))


def test_query_positive_respects_training_candidate_mask():
    base = torch.tensor([[0.2, 0.9], [0.2, 0.1]])
    candidate = torch.tensor([[0.8, 0.1], [0.8, 0.9]])
    output, selected = route_query_activation(
        base,
        candidate,
        0,
        "query_positive",
        torch.tensor([False, True]),
    )
    assert selected.tolist() == [False, True]
    assert torch.allclose(output[:, 0], torch.tensor([0.2, 0.8]))


def test_query_positive_blend_scales_only_positive_candidate_gain():
    base = torch.tensor([[0.2], [0.2], [0.4]])
    candidate = torch.tensor([[0.6], [0.6], [0.3]])
    reliability = torch.tensor([0.0, 0.5, 1.0])
    output, selected = route_query_activation(
        base,
        candidate,
        0,
        "query_positive_blend",
        reliability,
    )
    assert torch.allclose(output.squeeze(1), torch.tensor([0.2, 0.4, 0.4]))
    assert selected.tolist() == [False, True, False]


def test_precompute_query_scores_supports_discrete_artifact_device():
    class Artifact:
        num_gaussians = 2
        codebooks = [torch.zeros(1)]

        @staticmethod
        def reconstruct_range(start, end):
            return torch.eye(2)[start:end]

    class ClipStub:
        @staticmethod
        def get_activation(features, category_index):
            return features[:, category_index : category_index + 1]

    scores = precompute_artifact_query_scores(Artifact(), ClipStub(), 2, chunk_size=1)
    assert torch.allclose(scores, torch.eye(2))
