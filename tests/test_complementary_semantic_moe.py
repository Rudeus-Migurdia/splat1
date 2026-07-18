import unittest

import torch

from train_complementary_semantic_moe import (
    complementarity_loss,
    gate_inputs,
    masked_softmax,
    role_prior,
)


class ComplementarySemanticMoETest(unittest.TestCase):
    def test_soft_gate_keeps_multiple_supported_experts_and_masks_missing_source(self):
        logits = torch.zeros((2, 3))
        valid = torch.tensor([[True, True, True], [True, False, True]])
        weights = masked_softmax(logits, valid)
        torch.testing.assert_close(
            weights,
            torch.tensor([[1 / 3, 1 / 3, 1 / 3], [0.5, 0.0, 0.5]]),
        )
        torch.testing.assert_close(weights.sum(dim=1), torch.ones(2))

    def test_boundary_role_prior_moves_mass_from_old_to_l3(self):
        reliability = torch.full((3, 2), 0.8)
        valid = torch.ones((2, 3), dtype=torch.bool)
        prior = role_prior(reliability, valid, torch.tensor([0.0, 1.0]))
        self.assertGreater(float(prior[0, 0]), float(prior[0, 2]))
        self.assertGreater(float(prior[1, 2]), float(prior[1, 0]))
        torch.testing.assert_close(prior.sum(dim=1), torch.ones(2))

    def test_gate_inputs_are_label_free_pairwise_statistics(self):
        features = torch.tensor(
            [[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32
        )
        reliability = torch.tensor([[0.9], [0.8], [0.7]])
        valid = torch.ones((1, 3), dtype=torch.bool)
        inputs, boundary = gate_inputs(features, reliability, valid)
        self.assertEqual(inputs.shape, (1, 9))
        self.assertAlmostEqual(float(boundary), 1.0)

    def test_complementarity_penalizes_duplicate_residual_directions(self):
        raw = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.8, 0.0, 0.6]]],
            dtype=torch.float32,
        )
        duplicate = torch.tensor(
            [[[1.0, 0.0, 0.0], [0.8, 0.6, 0.0], [0.7, 0.7, 0.0]]],
            dtype=torch.float32,
        )
        complementary = raw.clone()
        valid = torch.ones((1, 3), dtype=torch.bool)
        duplicate_loss, _ = complementarity_loss(duplicate, raw, valid, margin=0.2)
        complementary_loss, _ = complementarity_loss(complementary, raw, valid, margin=0.2)
        self.assertLess(float(complementary_loss), float(duplicate_loss))


if __name__ == "__main__":
    unittest.main()
