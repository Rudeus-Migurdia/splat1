import numpy as np
import pytest

from lerf_ovs_paper_protocol import (
    PROTOCOL_NAME,
    best_threshold_summary,
    binary_iou,
    summarize_method_scenes,
    summarize_samples,
    validate_selection_thresholds,
)


def test_binary_iou_uses_union_and_handles_empty_masks():
    prediction = np.array([[1, 1], [0, 0]], dtype=bool)
    ground_truth = np.array([[1, 0], [1, 0]], dtype=bool)
    assert binary_iou(prediction, ground_truth) == pytest.approx(1.0 / 3.0)
    assert binary_iou(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0


def test_sample_accuracy_matches_reference_strict_thresholds():
    samples = [
        {"category": "a", "iou": 0.25},
        {"category": "a", "iou": 0.50},
        {"category": "b", "iou": 0.75},
    ]
    summary = summarize_samples(samples)
    assert summary["mIoU"] == pytest.approx(0.5)
    assert summary["mAcc@0.25"] == pytest.approx(2.0 / 3.0)
    assert summary["mAcc@0.5"] == pytest.approx(1.0 / 3.0)
    assert summary["per_category"]["a"] == pytest.approx(0.375)


def test_method_threshold_is_shared_across_scenes():
    scene_results = {
        "one": {
            "evaluation_protocol": PROTOCOL_NAME,
            "threshold_summary": [
                {"selection_threshold": 0.4, "mIoU": 0.9, "mAcc@0.25": 1.0, "mAcc@0.5": 1.0},
                {"selection_threshold": 0.5, "mIoU": 0.5, "mAcc@0.25": 1.0, "mAcc@0.5": 0.0},
            ],
        },
        "two": {
            "evaluation_protocol": PROTOCOL_NAME,
            "threshold_summary": [
                {"selection_threshold": 0.4, "mIoU": 0.0, "mAcc@0.25": 0.0, "mAcc@0.5": 0.0},
                {"selection_threshold": 0.5, "mIoU": 0.6, "mAcc@0.25": 1.0, "mAcc@0.5": 1.0},
            ],
        },
    }
    result = summarize_method_scenes(scene_results)
    assert result["selection_threshold"] == 0.5
    assert result["mIoU"] == pytest.approx(0.55)
    assert result["scenes"]["one"]["mIoU"] == 0.5


def test_threshold_validation_and_tie_break_are_deterministic():
    assert validate_selection_thresholds([0.5, 0.4, 0.5]) == [0.4, 0.5]
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        validate_selection_thresholds([1.1])
    best = best_threshold_summary(
        [
            {"selection_threshold": 0.5, "mIoU": 0.4},
            {"selection_threshold": 0.4, "mIoU": 0.4},
        ]
    )
    assert best["selection_threshold"] == 0.4
