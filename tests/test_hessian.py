from __future__ import annotations

import unittest

import torch

from utils.hessian import (
    assign_parameters,
    compute_hvp,
    estimate_min_eigenvalue,
    flatten_parameters,
)


class Quadratic(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = torch.nn.Parameter(torch.tensor([0.4, -0.2, 0.7]))
        self.register_buffer("matrix", torch.diag(torch.tensor([-2.0, 1.0, 4.0])))

    def loss(self) -> torch.Tensor:
        return 0.5 * self.theta @ self.matrix @ self.theta


class HessianTest(unittest.TestCase):
    def test_hvp_matches_exact_matrix_product(self) -> None:
        model = Quadratic()
        vector = torch.tensor([0.3, -0.5, 0.2])
        actual = compute_hvp(model, model.loss, vector)
        torch.testing.assert_close(actual, model.matrix @ vector)

    def test_lanczos_finds_smallest_eigenpair(self) -> None:
        model = Quadratic()
        value, vector = estimate_min_eigenvalue(
            model,
            model.loss,
            3,
            seed=7,
            tolerance=1e-10,
            return_eigenvector=True,
        )
        self.assertLess(abs(value - (-2.0)), 1e-5)
        rayleigh = float(vector @ (model.matrix @ vector))
        self.assertLess(abs(rayleigh - (-2.0)), 1e-5)

    def test_flatten_assign_round_trip(self) -> None:
        model = Quadratic()
        original = flatten_parameters(model)
        replacement = torch.tensor([1.0, 2.0, 3.0])
        assign_parameters(model, replacement)
        torch.testing.assert_close(flatten_parameters(model), replacement)
        assign_parameters(model, original)
        torch.testing.assert_close(flatten_parameters(model), original)


if __name__ == "__main__":
    unittest.main()
