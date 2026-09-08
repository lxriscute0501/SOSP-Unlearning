"""Matrix-free Hessian-vector products and extremal-eigenvalue estimation.

The full Hessian of even a LoRA adapter is far too large to construct. Lanczos
only asks how the Hessian acts on a vector, so autograd can supply H @ v without
ever materializing an O(n^2) matrix.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import overload

import torch

from .common import trainable_parameters


def flatten_parameters(
    parameters_or_model: Iterable[torch.nn.Parameter] | torch.nn.Module,
) -> torch.Tensor:
    """Flatten the current values of trainable parameters into one detached vector."""
    if isinstance(parameters_or_model, torch.nn.Module):
        parameters = trainable_parameters(parameters_or_model)
    else:
        parameters = list(parameters_or_model)
    if not parameters:
        raise ValueError("Cannot flatten an empty parameter list.")
    return torch.cat([parameter.detach().reshape(-1) for parameter in parameters])


@torch.no_grad()
def assign_parameters(
    parameters_or_model: Iterable[torch.nn.Parameter] | torch.nn.Module,
    vector: torch.Tensor,
) -> None:
    """Copy a flat vector into the trainable parameters without changing their identity."""
    if isinstance(parameters_or_model, torch.nn.Module):
        parameters = trainable_parameters(parameters_or_model)
    else:
        parameters = list(parameters_or_model)
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        if offset + count > vector.numel():
            raise ValueError("Parameter vector is too short.")
        parameter.copy_(vector[offset : offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError("Parameter vector is longer than the trainable parameter list.")


def _vector_chunks(vector: torch.Tensor, parameters: list[torch.nn.Parameter]) -> list[torch.Tensor]:
    if vector.ndim != 1:
        raise ValueError("The HVP vector must be one-dimensional.")
    chunks: list[torch.Tensor] = []
    offset = 0
    for parameter in parameters:
        count = parameter.numel()
        chunks.append(vector[offset : offset + count].view_as(parameter))
        offset += count
    if offset != vector.numel():
        raise ValueError(
            f"HVP vector has {vector.numel()} elements, but parameters have {offset}."
        )
    return chunks


def compute_hvp(
    model: torch.nn.Module,
    loss: torch.Tensor | Callable[[], torch.Tensor],
    vector: torch.Tensor,
) -> torch.Tensor:
    """Compute H(loss) @ vector over trainable parameters using double autograd.

    A callable is preferred for repeated products because it creates a fresh graph
    on every invocation. A tensor is supported for a single standalone HVP.
    """
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    chunks = _vector_chunks(vector, parameters)
    loss_value = loss() if callable(loss) else loss
    first_gradients = torch.autograd.grad(
        loss_value,
        parameters,
        create_graph=True,
        retain_graph=True,
        allow_unused=True,
    )
    products = [
        (gradient * chunk.to(gradient.dtype)).sum()
        for gradient, chunk in zip(first_gradients, chunks)
        if gradient is not None
    ]
    if not products:
        return torch.zeros_like(vector)
    directional_derivative = torch.stack(products).sum()
    if not directional_derivative.requires_grad:
        return torch.zeros_like(vector)
    second_gradients = torch.autograd.grad(
        directional_derivative,
        parameters,
        retain_graph=not callable(loss),
        allow_unused=True,
    )
    flat = torch.cat(
        [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, second_gradients)
        ]
    )
    return flat.detach().to(device=vector.device, dtype=vector.dtype)


@overload
def estimate_min_eigenvalue(
    model: torch.nn.Module,
    loss: torch.Tensor | Callable[[], torch.Tensor],
    num_lanczos_steps: int,
    *,
    tolerance: float = ...,
    seed: int = ...,
    return_eigenvector: bool = False,
    reorthogonalize: bool = ...,
) -> float: ...


@overload
def estimate_min_eigenvalue(
    model: torch.nn.Module,
    loss: torch.Tensor | Callable[[], torch.Tensor],
    num_lanczos_steps: int,
    *,
    tolerance: float = ...,
    seed: int = ...,
    return_eigenvector: bool = True,
    reorthogonalize: bool = ...,
) -> tuple[float, torch.Tensor]: ...


def estimate_min_eigenvalue(
    model: torch.nn.Module,
    loss: torch.Tensor | Callable[[], torch.Tensor],
    num_lanczos_steps: int,
    *,
    tolerance: float = 1e-7,
    seed: int = 0,
    return_eigenvector: bool = False,
    reorthogonalize: bool = True,
) -> float | tuple[float, torch.Tensor]:
    """Estimate the algebraically smallest Hessian eigenpair with Lanczos.

    The returned vector is the minimum Ritz vector in the LoRA parameter space.
    Full reorthogonalization costs O(k*n) memory but materially improves reliability
    for the short (typically 10--30 step) runs used here.
    """
    if num_lanczos_steps < 1:
        raise ValueError("num_lanczos_steps must be at least one.")
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("The model has no trainable parameters.")
    parameter_count = sum(parameter.numel() for parameter in parameters)
    device = parameters[0].device
    cpu_generator = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(parameter_count, generator=cpu_generator, dtype=torch.float32).to(device)
    q /= torch.linalg.vector_norm(q).clamp_min(torch.finfo(q.dtype).eps)
    q_previous = torch.zeros_like(q)
    beta_previous = q.new_zeros(())

    basis: list[torch.Tensor] = []
    alphas: list[float] = []
    betas: list[float] = []
    for iteration in range(num_lanczos_steps):
        basis.append(q)
        z = compute_hvp(model, loss, q)
        if iteration:
            z = z - beta_previous * q_previous
        alpha = torch.dot(q, z)
        z = z - alpha * q

        # Finite-precision loss of orthogonality can create spurious duplicated
        # Ritz values. Two-pass modified Gram-Schmidt is cheap for small k.
        if reorthogonalize:
            for _ in range(2):
                for old_q in basis:
                    z = z - torch.dot(old_q, z) * old_q

        alphas.append(float(alpha.detach().cpu()))
        if iteration == num_lanczos_steps - 1:
            break
        beta = torch.linalg.vector_norm(z)
        if not torch.isfinite(beta):
            raise FloatingPointError("Lanczos encountered a non-finite residual norm.")
        if beta <= tolerance:
            break
        betas.append(float(beta.detach().cpu()))
        q_previous, q = q, z / beta
        beta_previous = beta

    size = len(alphas)
    tridiagonal = torch.diag(torch.tensor(alphas, dtype=torch.float64))
    if size > 1:
        off_diagonal = torch.tensor(betas[: size - 1], dtype=torch.float64)
        tridiagonal += torch.diag(off_diagonal, diagonal=1)
        tridiagonal += torch.diag(off_diagonal, diagonal=-1)
    eigenvalues, eigenvectors = torch.linalg.eigh(tridiagonal)
    minimum = float(eigenvalues[0])
    if not return_eigenvector:
        return minimum

    coefficients = eigenvectors[:, 0].to(device=device, dtype=basis[0].dtype)
    ritz_vector = torch.zeros_like(basis[0])
    for coefficient, basis_vector in zip(coefficients, basis):
        ritz_vector.add_(basis_vector, alpha=float(coefficient))
    ritz_vector /= torch.linalg.vector_norm(ritz_vector).clamp_min(
        torch.finfo(ritz_vector.dtype).eps
    )
    return minimum, ritz_vector.detach()

