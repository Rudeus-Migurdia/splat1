import faiss
import numpy as np

from build_paired_shared_semantic_atoms import paired_assign


def test_paired_assignment_excludes_competitor_code():
    atoms = np.eye(3, dtype=np.float32)
    index = faiss.IndexFlatIP(3)
    index.add(atoms)
    anchors = np.array([[1.0, 0.9, 0.0]], dtype=np.float32)
    competitors = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    assigned, competitor, unresolved = paired_assign(
        index, anchors, competitors, topk=3
    )
    assert competitor.tolist() == [0]
    assert assigned.tolist() == [1]
    assert not unresolved.item()
