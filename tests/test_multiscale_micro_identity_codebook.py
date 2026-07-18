import numpy as np

from build_multiscale_micro_identity_codebook import (
    fine_identity_mapping,
    select_micro_tokens,
)


def test_fine_identity_mapping_is_unique_per_part():
    parts, fine = fine_identity_mapping(
        np.array([2, 2, 7, 7, 9], dtype=np.uint16),
        np.array([11, 11, 15, 15, 65535], dtype=np.uint16),
        65535,
    )
    assert parts.tolist() == [2, 7]
    assert fine.tolist() == [11, 15]


def test_micro_selection_requires_stability_and_new_information():
    selected = select_micro_tokens(
        np.array([0.8, 0.8, 0.5]),
        np.array([0.1, 0.01, 0.2]),
        np.array([True, True, True]),
        0.6,
        0.05,
    )
    assert selected.tolist() == [True, False, False]
