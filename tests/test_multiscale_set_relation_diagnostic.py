import numpy as np

from build_multiscale_set_relation_diagnostic import (
    dominant_gaussian_segments_from_labels,
    evaluate_relation_models,
    evaluate_train_selected_conflicts,
    make_gate_decision,
)


def test_dominant_gaussian_segments_uses_weighted_mass_and_fraction():
    point_ids = np.array([[0, 1], [0, 1], [0, 1]], dtype=np.int64)
    point_weights = np.array(
        [[0.8, 0.1], [0.7, 0.1], [0.2, 0.8]], dtype=np.float32
    )
    labels = np.array([3, 3, 7], dtype=np.int64)

    segments, confidence, visibility = dominant_gaussian_segments_from_labels(
        point_ids, point_weights, labels, num_gaussians=3, minimum_fraction=0.6
    )

    assert segments.tolist() == [3, 7, -1]
    assert confidence[0] > 0.8
    assert confidence[1] == 0.8
    assert visibility[0] > 0.0
    assert visibility[1] > 0.0


def _stable_set_evidence():
    positive = np.zeros((2, 4, 2, 1), dtype=np.float32)
    negative = np.zeros_like(positive)
    counts = np.full((2, 4, 2, 1), 4, dtype=np.uint8)
    positive[:, 0, :, :] = 12.0
    negative[:, 1, :, :] = 12.0
    positive[:, 2, :, :] = 9.0
    negative[:, 3, :, :] = 9.0
    return positive, negative, counts


def test_set_relation_beats_fixed_relation_on_stable_multiscale_evidence():
    positive, negative, counts = _stable_set_evidence()
    metrics = evaluate_relation_models(positive, negative, counts, 3)

    assert metrics["relative_nll_improvement"] > 0.5
    assert metrics["relation_signature_agreement"] == 1.0
    assert metrics["stable_set_ambiguous_directed_edges"] == 2
    assert metrics["stable_set_ambiguous_fraction"] == 1.0

    conflicts = evaluate_train_selected_conflicts(positive, negative, counts, 3)
    assert conflicts["relative_nll_improvement"] > 0.5
    assert conflicts["relative_balanced_nll_improvement"] > 0.5
    assert conflicts["selected_directed_edges_by_train_split"] == [2, 2]
    assert conflicts["selection_jaccard"] == 1.0


def test_split_instability_is_detected():
    positive, negative, counts = _stable_set_evidence()
    positive[1], negative[1] = negative[1].copy(), positive[1].copy()
    metrics = evaluate_relation_models(positive, negative, counts, 3)

    assert metrics["relation_signature_agreement"] == 0.0
    assert metrics["stable_set_ambiguous_directed_edges"] == 0


def test_gate_requires_both_prediction_and_stability():
    passing = {
        "relative_nll_improvement": 0.11,
        "relation_signature_agreement": 0.81,
        "multilevel_directed_edges": 1,
    }
    failing = dict(passing, relation_signature_agreement=0.79)

    assert make_gate_decision(passing, 0.10, 0.80)["pass"]
    assert not make_gate_decision(failing, 0.10, 0.80)["pass"]
