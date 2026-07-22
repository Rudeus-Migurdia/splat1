from types import SimpleNamespace

import numpy as np

from build_full_group_addressed_memory import (
    aggregate_reference_memory,
    decompose_overlapping_groups,
)


def test_descriptor_consistency_and_group_decomposition(monkeypatch):
    profiles = np.array(
        [
            [0.9, 0.8, 0.0], [0.1, 0.7, 0.9],
            [0.8, 0.1, 0.0], [0.0, 0.2, 0.8],
            [0.7, 0.2, 0.1], [0.2, 0.8, 0.6],
            [0.6, 0.4, 0.1], [0.3, 0.5, 0.9],
        ],
        dtype=np.float32,
    )
    model = {
        "profiles": profiles,
        "descriptors": np.eye(8, 512, dtype=np.float32),
        "support_views": np.full(8, 3, dtype=np.int32),
        "levels": np.repeat(np.arange(4, dtype=np.int8), 2),
        "utility": np.ones(8, dtype=np.float32),
        "members": [[{"view_index": 0, "proposal_index": index}] for index in range(8)],
        "statistics": {},
    }
    views = [
        {
            "view_index": 0,
            "descriptors": np.eye(8, 512, dtype=np.float32),
            "quality": np.ones(8, dtype=np.float32),
        }
    ]
    monkeypatch.setattr(
        "build_full_group_addressed_memory.signed_group_profiles",
        lambda model, views, ring_views, args: (model["profiles"], {}),
    )
    monkeypatch.setattr(
        "build_full_group_addressed_memory.bounded_group_profiles",
        lambda signed, graph, args: (
            signed,
            signed >= 0.3,
            signed >= 0.05,
            graph,
            {},
        ),
    )
    monkeypatch.setattr(
        "build_full_group_addressed_memory.build_ring_descriptors",
        lambda model, support, graph, args: (
            np.zeros((8, 512), dtype=np.float32),
            np.zeros(8, dtype=bool),
            np.zeros_like(support),
            {},
        ),
    )
    args = SimpleNamespace(
        exterior_semantic_weight=0.35,
        minimum_owner_membership=0.05,
        boundary_reliability_floor=0.25,
        boundary_margin=0.2,
    )
    result = decompose_overlapping_groups(model, views, [], None, args)
    assert result["atom_group_ids"].tolist() == [
        [0, 2, 4, 6],
        [0, 3, 5, 7],
        [1, 3, 5, 7],
    ]
    assert np.all(result["atom_membership"] > 0)
    assert result["atom_group_ids"].shape == (3, 4)


def test_each_valid_atom_has_at_most_one_group_per_level():
    ids = np.array([[2, 7, -1, 9], [3, 8, 4, 10]], dtype=np.int32)
    assert ids.ndim == 2 and ids.shape[1] == 4
    assert np.all((ids >= -1) & (ids < 11))


def test_reference_aggregation_keeps_spatial_addresses_separate(tmp_path):
    level_specs = []
    semantic_ids = []
    group_levels = []
    point_ids = np.zeros((2, 4), dtype=np.uint16)
    offset = 0
    for level in range(4):
        codebook = np.zeros((2, 512), dtype=np.float32)
        codebook[0, 2 * level] = 1.0
        codebook[1, 2 * level + 1] = 1.0
        filename = f"level_{level}.npy"
        np.save(tmp_path / filename, codebook)
        level_specs.append({"level": level, "codebook": filename})
        semantic_ids.extend([[0], [1]])
        group_levels.extend([level, level])
        point_ids[:, level] = [offset, offset + 1]
        offset += 2
    np.save(tmp_path / "point_ids.npy", point_ids)
    np.save(tmp_path / "semantic_ids.npy", np.asarray(semantic_ids, dtype=np.uint16))
    np.save(tmp_path / "levels.npy", np.asarray(group_levels, dtype=np.uint8))
    (tmp_path / "manifest.json").write_text(
        __import__("json").dumps(
            {
                "representation": "hierarchical_independent_group_codebooks",
                "method": "test",
                "feature_dim": 512,
                "point_group_ids": "point_ids.npy",
                "group_semantic_code_ids": "semantic_ids.npy",
                "group_level": "levels.npy",
                "level_codebooks": level_specs,
            }
        )
    )
    decomposition = {
        "levels": np.repeat(np.arange(4), 2),
        "atom_group_ids": np.array([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.int32),
        "atom_membership": np.ones((2, 4), dtype=np.float32),
        "atom_reliability": np.ones((2, 4), dtype=np.float32),
    }
    keys, consistency, valid, _ = aggregate_reference_memory(
        str(tmp_path), decomposition, np.array([0, 1], dtype=np.int64)
    )
    assert valid.all()
    assert np.allclose(consistency, 1.0)
    assert np.argmax(keys, axis=1).tolist() == list(range(8))
