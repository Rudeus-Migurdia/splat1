import numpy as np
import pytest

from build_hierarchical_semantic_memory import (
    AUXILIARY_SOURCE,
    INVALID_SOURCE,
    OLD_SOURCE,
    parent_group_lookup,
    resolve_group_source,
    validate_level_configuration,
)


def test_split_consistency_gate_selects_one_reliable_source_per_group():
    selected = resolve_group_source(
        np.array([0.60, 0.80, 0.00, 0.90, 0.20]),
        np.array([True, True, False, True, False]),
        np.array([0.80, 0.70, 0.70, 0.00, 0.00]),
        np.array([True, True, True, False, False]),
        margin=0.05,
    )
    assert selected.tolist() == [
        int(AUXILIARY_SOURCE),
        int(OLD_SOURCE),
        int(AUXILIARY_SOURCE),
        int(OLD_SOURCE),
        int(INVALID_SOURCE),
    ]


def test_child_groups_record_their_modal_parent():
    parents = np.array([0, 0, 1, 1, 1, 2, 2])
    children = np.array([0, 0, 1, 1, 1, 2, 2])
    assert parent_group_lookup(parents, children).tolist() == [0, 1, 2]


def test_level_configuration_requires_coarser_to_finer_capacity():
    validate_level_configuration(
        [0.76, 0.82, 0.87, 0.91], [2048, 512, 128, 32], [16, 8, 4, 2]
    )
    with pytest.raises(ValueError, match="maximum group sizes"):
        validate_level_configuration(
            [0.76, 0.82, 0.87, 0.91], [512, 2048, 128, 32], [16, 8, 4, 2]
        )
