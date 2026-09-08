from __future__ import annotations

import unittest

import torch

from utils.blur import blur_direction


class BlurDirectionTest(unittest.TestCase):
    def test_projected_retain_gradient_is_orthogonal_without_epsilon(self) -> None:
        forget = torch.tensor([1.0, 2.0, -1.0])
        retain = torch.tensor([3.0, -2.0, 4.0])
        gamma = 0.7
        direction, _ = blur_direction(
            forget, retain, gamma=gamma, epsilon=0.0
        )
        projected_retain = direction - gamma * forget
        self.assertLess(float(torch.dot(projected_retain, forget).abs()), 1e-6)


    def test_epsilon_keeps_zero_forget_gradient_finite(self) -> None:
        forget = torch.zeros(4)
        retain = torch.arange(4, dtype=torch.float32)
        direction, diagnostics = blur_direction(
            forget, retain, gamma=1.0, epsilon=1e-12
        )
        torch.testing.assert_close(direction, retain)
        self.assertEqual(diagnostics["projection_coefficient"], 0.0)
        self.assertTrue(bool(torch.isfinite(direction).all()))


if __name__ == "__main__":
    unittest.main()
