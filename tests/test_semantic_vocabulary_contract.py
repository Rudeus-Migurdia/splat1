import json

import numpy as np
import pytest

from validate_semantic_vocabulary_contract import validate_contract


def test_contract_rejects_missing_enabled_modality(tmp_path):
    np.save(tmp_path / "vocab.npy", np.eye(2, dtype=np.float16))
    np.save(tmp_path / "ids.npy", np.array([[0, 1]], dtype=np.uint16))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "group_codebook": "vocab.npy",
                "group_semantic_code_ids": "ids.npy",
                "semantic_invalid_id": 65535,
                "vocabulary_modalities": ["base", "part"],
                "modality_token_counts": {"base": 2, "part": 1},
            }
        )
    )
    with pytest.raises(ValueError, match="boundary"):
        validate_contract(tmp_path, ["base", "part", "boundary"])


def test_contract_accepts_four_independent_hierarchical_codebooks(tmp_path):
    for level in range(4):
        np.save(tmp_path / f"l{level}.npy", np.eye(2, 3, dtype=np.float16))
    np.save(tmp_path / "semantic_ids.npy", np.array([[0], [1], [0], [1]], dtype=np.uint16))
    np.save(tmp_path / "levels.npy", np.array([0, 1, 2, 3], dtype=np.uint8))
    np.save(tmp_path / "point_ids.npy", np.array([[0, 1, 2, 3]], dtype=np.uint16))
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "representation": "hierarchical_independent_group_codebooks",
                "group_semantic_code_ids": "semantic_ids.npy",
                "semantic_invalid_id": 65535,
                "group_level": "levels.npy",
                "point_group_ids": "point_ids.npy",
                "invalid_id": 65535,
                "vocabulary_modalities": ["base", "sam_l0", "sam_l1", "sam_l2", "sam_l3"],
                "modality_token_counts": {"base": 0, "sam_l0": 1, "sam_l1": 1, "sam_l2": 1, "sam_l3": 1},
                "level_codebooks": [
                    {"level": level, "codebook": f"l{level}.npy", "num_codes": 2}
                    for level in range(4)
                ],
            }
        )
    )

    result = validate_contract(
        tmp_path, ["base", "sam_l0", "sam_l1", "sam_l2", "sam_l3"]
    )

    assert result["independent_level_codebooks"] is True
    assert result["num_vocabulary_codes"] == 8
