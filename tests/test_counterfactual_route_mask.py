import pytest

torch = pytest.importorskip("torch")

from build_counterfactual_route_mask import split_counterfactual_advantage


def test_split_counterfactual_advantage_requires_both_splits_to_improve():
    base = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    candidate = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    split_targets = torch.tensor(
        [
            [[0.0, 1.0], [0.0, 1.0]],
            [[0.0, 1.0], [1.0, 0.0]],
        ]
    )
    advantage = split_counterfactual_advantage(base, candidate, split_targets)
    assert advantage[0] > 0.0
    assert advantage[1] < 0.0
