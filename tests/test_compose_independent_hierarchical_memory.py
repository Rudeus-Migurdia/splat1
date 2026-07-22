import json

import numpy as np

from compose_independent_hierarchical_memory import compose_memories


def _write_memory(root, prefix, reliability):
    root.mkdir()
    point_ids = np.array([[0, 2, 4, 6], [1, 3, 5, 7]], dtype=np.uint16)
    semantic_ids = np.tile(np.arange(2, dtype=np.uint16), 4)[:, None]
    levels = np.repeat(np.arange(4, dtype=np.uint8), 2)
    np.save(root / "point_group_ids.npy", point_ids)
    np.save(root / "point_group_weights.npy", np.full((2, 4), 255, np.uint8))
    np.save(root / "point_group_reliability.npy", np.full((2, 4), reliability, np.float16))
    np.save(root / "point_group_source.npy", np.full((2, 4), prefix, np.uint8))
    np.save(root / "group_semantic_code_ids.npy", semantic_ids)
    np.save(root / "group_level.npy", levels)
    np.save(root / "group_reliability.npy", np.full(8, reliability, np.float16))
    np.save(root / "group_source.npy", np.full(8, prefix, np.uint8))
    entries = []
    for level in range(4):
        filename = f"sam_l{level}_codebook.npy"
        values = np.full((2, 3), prefix + level, dtype=np.float16)
        np.save(root / filename, values)
        entries.append(
            {
                "name": f"sam_l{level}",
                "level": level,
                "codebook": filename,
                "num_codes": 2,
            }
        )
    manifest = {
        "representation": "hierarchical_independent_group_codebooks",
        "feature_dim": 3,
        "num_gaussians": 2,
        "point_group_ids": "point_group_ids.npy",
        "point_group_weights": "point_group_weights.npy",
        "point_group_reliability": "point_group_reliability.npy",
        "point_group_source": "point_group_source.npy",
        "group_semantic_code_ids": "group_semantic_code_ids.npy",
        "group_level": "group_level.npy",
        "group_reliability": "group_reliability.npy",
        "group_source": "group_source.npy",
        "invalid_id": 65535,
        "level_codebooks": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_compose_memories_preserves_whole_selected_levels(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "output"
    _write_memory(first, 10, 0.25)
    _write_memory(second, 20, 0.75)

    manifest = compose_memories(
        [str(first), str(second), str(first), str(second)], str(output), 7
    )

    ids = np.load(output / "point_group_ids.npy")
    reliability = np.load(output / "point_group_reliability.npy")
    sources = np.load(output / "point_group_source.npy")
    assert ids.tolist() == [[0, 2, 4, 6], [1, 3, 5, 7]]
    assert reliability[0].tolist() == [0.25, 0.75, 0.25, 0.75]
    assert sources[0].tolist() == [10, 20, 10, 20]
    assert [item["num_codes"] for item in manifest["level_codebooks"]] == [2, 2, 2, 2]
