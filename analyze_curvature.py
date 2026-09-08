#!/usr/bin/env python3
"""Diagnose Hessian curvature and identify FOSP-but-not-SOSP checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import os
from functools import partial
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/blur-matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from utils.common import (
    checkpoint_step,
    discover_checkpoints,
    load_config,
    load_model_and_tokenizer,
    save_json,
    set_seed,
    trainable_parameter_names,
    trainable_parameters,
)
from utils.data import fixed_batches, make_dataloader
from utils.hessian import estimate_min_eigenvalue
from utils.metrics import first_order_statistics, forget_loss, make_objective_closure, retain_loss


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    columns = [
        "checkpoint",
        "step",
        "forget_loss",
        "retain_loss",
        "forget_grad_norm",
        "retain_grad_norm",
        "lagrange_multiplier",
        "bilevel_residual",
        "lambda_min_Hg",
        "is_failure",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_plots(rows: list[dict[str, object]], results_dir: Path) -> None:
    if not rows:
        return
    steps = [int(row["step"]) for row in rows]
    gradient_norms = [float(row["forget_grad_norm"]) for row in rows]
    eigenvalues = [float(row["lambda_min_Hg"]) for row in rows]
    failures = [bool(row["is_failure"]) for row in rows]

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(steps, gradient_norms, marker="o")
    axis.set_yscale("log")
    axis.set(xlabel="Training step", ylabel=r"$\|\nabla g(\theta)\|$", title="Forget gradient norm")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "forget_grad_norm_vs_step.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(steps, eigenvalues, marker="o")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(xlabel="Training step", ylabel=r"$\lambda_{min}(H_g)$", title="Forget-loss minimum curvature")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "lambda_min_vs_step.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4.8))
    colors = ["crimson" if failure else "steelblue" for failure in failures]
    axis.scatter(gradient_norms, eigenvalues, c=colors, s=55, alpha=0.9)
    axis.set_xscale("log")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set(
        xlabel=r"$\|\nabla g(\theta)\|$",
        ylabel=r"$\lambda_{min}(H_g)$",
        title="First-order stationarity versus curvature",
    )
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(results_dir / "gradient_vs_curvature.png", dpi=180)
    plt.close(fig)


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed, config.get("deterministic", True))
    analysis_cfg = config["analysis"]
    checkpoints = discover_checkpoints(analysis_cfg["checkpoints_dir"])
    if not checkpoints:
        raise FileNotFoundError(
            f"No checkpoint_* directories found in {analysis_cfg['checkpoints_dir']}"
        )
    results_dir = Path(analysis_cfg["results_dir"])
    vector_dir = results_dir / "eigenvectors"
    failure_dir = results_dir / "failures"
    results_dir.mkdir(parents=True, exist_ok=True)
    vector_dir.mkdir(parents=True, exist_ok=True)

    threshold = float(analysis_cfg.get("fosp_grad_threshold", 1e-3))
    delta = float(analysis_cfg.get("negative_curvature_delta", 0.05))
    npo_cfg = config.get("forget_objective", {})
    forget_objective = partial(
        forget_loss,
        loss_type=npo_cfg.get("type", "npo"),
        beta=float(npo_cfg.get("beta", 0.1)),
    )
    fixed_forget = None
    fixed_retain = None
    rows: list[dict[str, object]] = []

    print("checkpoint | forget_grad_norm | lambda_min(Hg)")
    for checkpoint in checkpoints:
        model, tokenizer, device = load_model_and_tokenizer(
            config, checkpoint, for_training=False
        )
        model.eval()  # Dropout would otherwise make the Hessian operator stochastic.
        if fixed_forget is None:
            data_cfg = config["data"]
            batch_size = int(analysis_cfg.get("batch_size", 1))
            count = int(analysis_cfg.get("num_loss_batches", 4))
            forget_loader = make_dataloader(
                data_cfg["forget"],
                tokenizer,
                max_length=int(data_cfg.get("max_length", 256)),
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            )
            retain_loader = make_dataloader(
                data_cfg["retain"],
                tokenizer,
                max_length=int(data_cfg.get("max_length", 256)),
                batch_size=batch_size,
                shuffle=False,
                seed=seed,
            )
            fixed_forget = fixed_batches(forget_loader, count)
            fixed_retain = fixed_batches(retain_loader, count)

        forget_closure = make_objective_closure(model, fixed_forget, device, forget_objective)
        retain_closure = make_objective_closure(model, fixed_retain, device, retain_loss)
        statistics = first_order_statistics(
            model,
            forget_closure,
            retain_closure,
            epsilon=float(config["training"].get("projection_epsilon", 1e-12)),
        )
        minimum, eigenvector = estimate_min_eigenvalue(
            model,
            forget_closure,
            int(analysis_cfg.get("num_lanczos_steps", 12)),
            tolerance=float(analysis_cfg.get("lanczos_tolerance", 1e-7)),
            seed=seed + checkpoint_step(checkpoint),
            return_eigenvector=True,
            reorthogonalize=analysis_cfg.get("reorthogonalize", True),
        )
        is_failure = statistics["forget_grad_norm"] < threshold and minimum < -delta
        row: dict[str, object] = {
            "checkpoint": checkpoint.name,
            "step": checkpoint_step(checkpoint),
            **statistics,
            "lambda_min_Hg": minimum,
            "is_failure": is_failure,
        }
        rows.append(row)
        print(f"{checkpoint.name} | {statistics['forget_grad_norm']:.7g} | {minimum:.7g}")

        vector_payload = {
            "vector": eigenvector.cpu(),
            "lambda_min_Hg": minimum,
            "parameter_names": trainable_parameter_names(model),
            "parameter_shapes": [list(parameter.shape) for parameter in trainable_parameters(model)],
        }
        if is_failure or analysis_cfg.get("save_all_eigenvectors", False):
            torch.save(vector_payload, vector_dir / f"{checkpoint.name}.pt")
        if is_failure:
            selected_dir = failure_dir / checkpoint.name
            selected_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "checkpoint": str(checkpoint),
                    "statistics": row,
                    "trainable_state_dict": {
                        name: parameter.detach().cpu()
                        for name, parameter in model.named_parameters()
                        if parameter.requires_grad
                    },
                },
                selected_dir / "failure_checkpoint.pt",
            )
            print("Found FOSP but not SOSP checkpoint")
        save_json(results_dir / "fosp_statistics.json", rows)
        write_csv(rows, results_dir / "fosp_statistics.csv")
        del eigenvector, forget_closure, retain_closure
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    make_plots(rows, results_dir)
    save_json(
        results_dir / "failure_summary.json",
        {
            "fosp_grad_threshold": threshold,
            "negative_curvature_delta": delta,
            "num_checkpoints": len(rows),
            "failure_checkpoints": [row["checkpoint"] for row in rows if row["is_failure"]],
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    arguments = parser.parse_args()
    main(arguments.config)
