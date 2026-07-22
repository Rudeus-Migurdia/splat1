import numpy as np
import pytest

torch = pytest.importorskip("torch")

from build_multiview_local_relation_graph import (  # noqa: E402
    dominant_gaussian_segments,
    finalize_signed_relations,
)


def test_dominant_segments_use_all_ray_contributors():
    cache = {
        "point_ids": torch.tensor([[0, 1], [0, 2]], dtype=torch.int32),
        "point_weights": torch.tensor([[0.6, 0.4], [0.2, 0.5]]),
        "segment_ids": torch.tensor([3, 4], dtype=torch.int32),
    }
    segments, confidence, visibility = dominant_gaussian_segments(cache, 3, 0.55)
    assert segments.tolist() == [3, 3, 4]
    assert confidence.tolist() == pytest.approx([0.75, 1.0, 1.0])
    assert np.all((visibility > 0.0) & (visibility < 1.0))


def test_split_relations_require_sign_consistency_and_minimum_views():
    positive = np.zeros((2, 1, 3), dtype=np.float32)
    negative = np.zeros_like(positive)
    observations = np.full((2, 1, 3), 4, dtype=np.uint8)
    positive[:, 0, 0] = [4.0, 3.0]
    negative[:, 0, 0] = [1.0, 1.0]
    positive[:, 0, 1] = [4.0, 1.0]
    negative[:, 0, 1] = [1.0, 4.0]
    positive[:, 0, 2] = [1.0, 1.0]
    negative[:, 0, 2] = [4.0, 4.0]
    observations[:, 0, 2] = [4, 2]

    relation, diagnostics = finalize_signed_relations(
        positive, negative, observations, 3, 0.05
    )
    assert relation[0, 0] > 0.0
    assert relation[0, 1] == 0.0
    assert relation[0, 2] == 0.0
    assert diagnostics["minimum_split_views"].tolist() == [[4.0, 4.0, 2.0]]
