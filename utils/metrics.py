"""Forget/retain objectives and first-order stationarity diagnostics."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Callable, Iterable

import torch
import torch.nn.functional as F

from .common import move_batch, trainable_parameters


def sequence_log_prob(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean answer-token log probability for every sequence in a batch."""
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        use_cache=False,
    )
    logits = outputs.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    mask = labels.ne(-100)
    safe_labels = labels.masked_fill(~mask, 0)
    token_log_probs = F.log_softmax(logits, dim=-1).gather(
        dim=-1, index=safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    lengths = mask.sum(dim=-1).clamp_min(1)
    return (token_log_probs * mask).sum(dim=-1) / lengths


def retain_loss(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Standard answer-only causal language-model loss."""
    return -sequence_log_prob(model, batch).mean()


def forget_loss(
    model: torch.nn.Module,
    batch: dict[str, torch.Tensor],
    *,
    loss_type: str = "npo",
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the forget objective.

    NPO minimizes the probability ratio between the current policy and the frozen
    reference policy. For LoRA, disabling the adapter exposes that exact reference
    without keeping a duplicate 3B model in memory.
    """
    loss_type = loss_type.lower()
    if loss_type == "gradient_ascent":
        return -retain_loss(model, batch)
    if loss_type != "npo":
        raise ValueError(f"Unknown forget loss type: {loss_type}")
    if beta <= 0:
        raise ValueError("NPO beta must be positive.")

    current_log_prob = sequence_log_prob(model, batch)
    disable_adapter = getattr(model, "disable_adapter", None)
    context = disable_adapter() if disable_adapter is not None else nullcontext()
    with torch.no_grad(), context:
        reference_log_prob = sequence_log_prob(model, batch)
    log_ratio = current_log_prob - reference_log_prob
    return (-2.0 / beta * F.logsigmoid(-beta * log_ratio)).mean()


def make_objective_closure(
    model: torch.nn.Module,
    batches: Iterable[dict[str, torch.Tensor]],
    device: torch.device,
    objective: Callable[[torch.nn.Module, dict[str, torch.Tensor]], torch.Tensor],
) -> Callable[[], torch.Tensor]:
    """Create a deterministic average-loss closure suitable for repeated HVPs."""
    stored_batches = list(batches)

    def closure() -> torch.Tensor:
        losses = [objective(model, move_batch(batch, device)) for batch in stored_batches]
        return torch.stack(losses).mean()

    return closure


def gradients(
    loss: torch.Tensor,
    parameters: list[torch.nn.Parameter],
    *,
    create_graph: bool = False,
) -> list[torch.Tensor]:
    values = torch.autograd.grad(
        loss,
        parameters,
        create_graph=create_graph,
        allow_unused=True,
    )
    return [torch.zeros_like(parameter) if value is None else value for parameter, value in zip(parameters, values)]


def flatten_tensors(tensors: Iterable[torch.Tensor]) -> torch.Tensor:
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


def objective_gradient(
    model: torch.nn.Module, closure: Callable[[], torch.Tensor]
) -> tuple[float, torch.Tensor]:
    parameters = trainable_parameters(model)
    loss = closure()
    flat_gradient = flatten_tensors(gradients(loss, parameters)).detach()
    return float(loss.detach().cpu()), flat_gradient


def first_order_statistics(
    model: torch.nn.Module,
    forget_closure: Callable[[], torch.Tensor],
    retain_closure: Callable[[], torch.Tensor],
    epsilon: float = 1e-12,
) -> dict[str, float]:
    """Measure gradient norm and the best scalar KKT/bilevel residual.

    At a first-order constrained solution, grad(f) + lambda grad(g) should vanish.
    We report the least-squares lambda and its residual, not an arbitrary lambda.
    """
    parameters = trainable_parameters(model)
    forget_value = forget_closure()
    forget_gradient = flatten_tensors(gradients(forget_value, parameters)).detach()
    retain_value = retain_closure()
    retain_gradient = flatten_tensors(gradients(retain_value, parameters)).detach()

    denominator = torch.dot(forget_gradient, forget_gradient) + epsilon
    lagrange_multiplier = -torch.dot(retain_gradient, forget_gradient) / denominator
    residual = retain_gradient + lagrange_multiplier * forget_gradient
    return {
        "forget_loss": float(forget_value.detach().cpu()),
        "retain_loss": float(retain_value.detach().cpu()),
        "forget_grad_norm": float(torch.linalg.vector_norm(forget_gradient).cpu()),
        "retain_grad_norm": float(torch.linalg.vector_norm(retain_gradient).cpu()),
        "lagrange_multiplier": float(lagrange_multiplier.cpu()),
        "bilevel_residual": float(torch.linalg.vector_norm(residual).cpu()),
    }

