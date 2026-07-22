import numpy as np
import pytest

torch = pytest.importorskip("torch")

from build_counterfactual_codebook_neighborhoods import (  # noqa: E402
    nearest_codeword_neighbors,
)


def test_nearest_codeword_neighbors_excludes_self_and_is_exact():
    codebook = np.asarray(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [-1.0, 0.0],
        ],
        dtype=np.float32,
    )
    ids, cosine = nearest_codeword_neighbors(
        codebook, neighbors=2, chunk_size=2, device="cpu"
    )

    assert ids.dtype == np.uint16
    assert cosine.dtype == np.float16
    assert ids.shape == cosine.shape == (4, 2)
    assert all(index not in ids[index] for index in range(4))
    assert ids[0, 0] == 1
    assert ids[2, 0] == 1


def test_nearest_codeword_neighbors_rejects_oversized_neighborhood():
    with pytest.raises(ValueError, match="more rows"):
        nearest_codeword_neighbors(
            np.eye(2, dtype=np.float32), neighbors=2, chunk_size=1, device="cpu"
        )


def test_nearest_codeword_neighbors_skips_exact_duplicate_centers():
    codebook = np.asarray(
        [[1.0, 0.0], [1.0, 0.0], [0.9, 0.1], [0.0, 1.0]],
        dtype=np.float32,
    )
    ids, _ = nearest_codeword_neighbors(
        codebook, neighbors=1, chunk_size=2, device="cpu"
    )

    assert ids[0, 0] == 2
    assert ids[1, 0] == 2
