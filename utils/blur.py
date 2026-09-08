"""BLUR projected first-order update in the trainable parameter subspace."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .common import trainable_parameters


def blur_direction(
    forget_gradient: torch.Tensor,
    retain_gradient: torch.Tensor,
    *,
    gamma: float,
    epsilon: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return gamma*g + (I - gg^T/(||g||^2+eps))*f.

    The epsilon makes the projection continuous and finite when the forget gradient
    is already tiny—the exact regime this experiment is designed to diagnose.
    """
    squared_norm = torch.dot(forget_gradient, forget_gradient)
    projection_coefficient = torch.dot(forget_gradient, retain_gradient) / (
        squared_norm + epsilon
    )
    projected_retain = retain_gradient - projection_coefficient * forget_gradient
    direction = gamma * forget_gradient + projected_retain
    diagnostics = {
        "projection_coefficient": float(projection_coefficient.detach().cpu()),
        "forget_grad_norm": float(torch.sqrt(squared_norm).detach().cpu()),
        "retain_grad_norm": float(torch.linalg.vector_norm(retain_gradient).detach().cpu()),
        "update_norm": float(torch.linalg.vector_norm(direction).detach().cpu()),
    }
    return direction, diagnostics


def _flatten_gradients(
    loss: torch.Tensor, parameters: list[torch.nn.Parameter]
) -> torch.Tensor:
    raw = torch.autograd.grad(loss, parameters, allow_unused=True)
    gradients = [
        torch.zeros_like(parameter) if gradient is None else gradient
        for parameter, gradient in zip(parameters, raw)
    ]
    return torch.cat([gradient.reshape(-1) for gradient in gradients]).detach()


@torch.no_grad()
def apply_flat_update(
    parameters: list[torch.nn.Parameter], direction: torch.Tensor, learning_rate: float
) -> None:
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        update = direction[offset : offset + count].view_as(parameter)
        parameter.add_(update, alpha=-learning_rate)
        offset += count
    if offset != direction.numel():
        raise ValueError("Update vector size does not match the trainable parameters.")


def blur_step(
    model: torch.nn.Module,
    forget_closure: Callable[[], torch.Tensor],
    retain_closure: Callable[[], torch.Tensor],
    *,
    learning_rate: float,
    gamma: float,
    epsilon: float = 1e-12,
    max_update_norm: float | None = None,
) -> dict[str, float]:
    """Compute independent objective gradients and apply one manual BLUR step."""
    parameters = trainable_parameters(model)
    forget_value = forget_closure()
    forget_gradient = _flatten_gradients(forget_value, parameters)
    retain_value = retain_closure()
    retain_gradient = _flatten_gradients(retain_value, parameters)
    direction, diagnostics = blur_direction(
        forget_gradient, retain_gradient, gamma=gamma, epsilon=epsilon
    )
    raw_update_norm = torch.linalg.vector_norm(direction)
    if max_update_norm is not None and raw_update_norm > max_update_norm:
        direction = direction * (max_update_norm / raw_update_norm)
        diagnostics["clipped"] = 1.0
        diagnostics["update_norm"] = float(max_update_norm)
    else:
        diagnostics["clipped"] = 0.0
    apply_flat_update(parameters, direction, learning_rate)
    diagnostics.update(
        forget_loss=float(forget_value.detach().cpu()),
        retain_loss=float(retain_value.detach().cpu()),
    )
    return diagnostics

