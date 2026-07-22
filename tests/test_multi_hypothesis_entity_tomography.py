import numpy as np

from build_multi_hypothesis_entity_tomography import (
    balanced_bernoulli_nll,
    build_spatial_atoms,
    make_gate,
    noisy_or,
    sample_packed_masks,
)


def test_sample_packed_masks_preserves_overlap():
    masks = np.asarray(
        [
            [1, 1, 0, 0, 1, 0, 0, 0, 1],
            [0, 1, 1, 0, 1, 0, 0, 0, 0],
        ],
        dtype=np.uint8,
    )
    packed = np.packbits(masks, axis=1, bitorder="little")
    sampled = sample_packed_masks(packed, np.asarray([1, 4, 8]))
    np.testing.assert_array_equal(sampled, np.asarray([[1, 1, 1], [1, 1, 0]], dtype=bool))


def test_noisy_or_recovers_union_probability():
    first = np.asarray([0.8, 0.0, 0.5], dtype=np.float32)
    second = np.asarray([0.0, 0.6, 0.5], dtype=np.float32)
    np.testing.assert_allclose(noisy_or(np.stack([first, second])), [0.8, 0.6, 0.75])


def test_balanced_nll_prefers_matching_prediction():
    target = np.asarray([1.0, 0.0, 1.0, 0.0], dtype=np.float32)
    visibility = np.ones(4, dtype=np.float32)
    matching = balanced_bernoulli_nll(target, [0.9, 0.1, 0.9, 0.1], visibility)
    reversed_value = balanced_bernoulli_nll(target, [0.1, 0.9, 0.1, 0.9], visibility)
    assert matching < reversed_value


def test_spatial_atoms_are_deterministic_and_compact():
    xyz = np.random.RandomState(4).randn(200, 3).astype(np.float32)
    first, first_contract = build_spatial_atoms(xyz, target_atoms=32)
    second, second_contract = build_spatial_atoms(xyz, target_atoms=32)
    np.testing.assert_array_equal(first, second)
    assert first_contract == second_contract
    assert np.array_equal(np.unique(first), np.arange(first.max() + 1))


def test_gate_requires_all_preregistered_conditions():
    metrics = {
        "relative_nll_improvement": 0.11,
        "median_matched_jaccard": 0.81,
        "stable_slots": 2,
        "stable_slot_support_valid": True,
        "nontrivial_mass_fraction": 0.02,
        "unresolved_certificate_written": True,
    }
    assert make_gate(metrics, 0.10, 0.80, 0.01)["pass"]
    metrics["nontrivial_mass_fraction"] = 0.009
    assert not make_gate(metrics, 0.10, 0.80, 0.01)["pass"]
