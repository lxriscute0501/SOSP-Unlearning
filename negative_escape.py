#!/usr/bin/env python3
"""Take oriented minimum-curvature steps from diagnosed FOSP failure checkpoints."""

from __future__ import annotations

import argparse
import csv
import gc
import json
from functools import partial
from pathlib import Path

import torch

from utils.common import (
    load_config,
    load_model_and_tokenizer,
    set_seed,
    trainable_parameter_names,
)
from utils.data import fixed_batches, make_dataloader
from utils.hessian import assign_parameters, flatten_parameters
from utils.metrics import forget_loss, make_objective_closure, objective_gradient


def write_outputs(rows: list[dict[str, object]], results_dir: Path) -> None:
    columns = [
        "checkpoint",
        "lambda_min",
        "alpha",
        "directional_derivative",
        "loss_before",
        "loss_after",
        "loss_change",
        "improved",
    ]
    with (results_dir / "negative_escape.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary = results_dir / "negative_curvature_results.json.tmp"
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(results_dir / "negative_curvature_results.json")


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed, config.get("deterministic", True))
    analysis_cfg = config["analysis"]
    results_dir = Path(analysis_cfg["results_dir"])
    statistics_path = results_dir / "fosp_statistics.json"
    if not statistics_path.exists():
        raise FileNotFoundError(
            f"Missing {statistics_path}; run analyze_curvature.py first."
        )
    statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
    failures = [row for row in statistics if row.get("is_failure", False)]
    rows: list[dict[str, object]] = []
    if not failures:
        write_outputs(rows, results_dir)
        print("No FOSP failure checkpoint met both configured thresholds; nothing to escape.")
        return

    data_cfg = config["data"]
    escape_cfg = config.get("negative_escape", {})
    npo_cfg = config.get("forget_objective", {})
    forget_objective = partial(
        forget_loss,
        loss_type=npo_cfg.get("type", "npo"),
        beta=float(npo_cfg.get("beta", 0.1)),
    )
    alphas = [float(value) for value in escape_cfg.get("alphas", [1e-4, 5e-4, 1e-3, 5e-3])]

    for failure in failures:
        checkpoint_name = str(failure["checkpoint"])
        checkpoint = Path(analysis_cfg["checkpoints_dir"]) / checkpoint_name
        vector_path = results_dir / "eigenvectors" / f"{checkpoint_name}.pt"
        if not vector_path.exists():
            raise FileNotFoundError(f"Missing eigenvector for selected checkpoint: {vector_path}")
        model, tokenizer, device = load_model_and_tokenizer(
            config, checkpoint, for_training=False
        )
        model.eval()
        loader = make_dataloader(
            data_cfg["forget"],
            tokenizer,
            max_length=int(data_cfg.get("max_length", 256)),
            batch_size=int(analysis_cfg.get("batch_size", 1)),
            shuffle=False,
            seed=seed,
        )
        batches = fixed_batches(loader, int(analysis_cfg.get("num_loss_batches", 4)))
        closure = make_objective_closure(model, batches, device, forget_objective)
        payload = torch.load(vector_path, map_location="cpu", weights_only=True)
        current_names = trainable_parameter_names(model)
        if payload["parameter_names"] != current_names:
            raise RuntimeError(
                "Trainable parameter ordering differs from curvature analysis; refusing "
                "to apply a misaligned eigenvector."
            )
        eigenvector = payload["vector"].to(device=device, dtype=torch.float32)
        eigenvector /= torch.linalg.vector_norm(eigenvector).clamp_min(
            torch.finfo(eigenvector.dtype).eps
        )
        loss_before, gradient = objective_gradient(model, closure)

        # Eigenvectors have arbitrary sign. Orient v so the (small) linear term is
        # non-positive; the negative quadratic term then reinforces the decrease.
        directional_derivative = torch.dot(gradient.float(), eigenvector)
        if directional_derivative > 0:
            eigenvector.neg_()
            directional_derivative.neg_()
        base_parameters = flatten_parameters(model).float()
        try:
            for alpha in alphas:
                assign_parameters(model, base_parameters + alpha * eigenvector)
                with torch.no_grad():
                    loss_after = float(closure().cpu())
                row = {
                    "checkpoint": checkpoint_name,
                    "lambda_min": float(payload["lambda_min_Hg"]),
                    "alpha": alpha,
                    "directional_derivative": float(directional_derivative.cpu()),
                    "loss_before": loss_before,
                    "loss_after": loss_after,
                    "loss_change": loss_after - loss_before,
                    "improved": loss_after < loss_before,
                }
                rows.append(row)
                print(
                    f"{checkpoint_name} alpha={alpha:g}: "
                    f"g_before={loss_before:.8g} g_after={loss_after:.8g} "
                    f"delta={loss_after - loss_before:+.4g}"
                )
        finally:
            assign_parameters(model, base_parameters)
        write_outputs(rows, results_dir)
        del closure, eigenvector, gradient, base_parameters
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    successful = sum(bool(row["improved"]) for row in rows)
    print(f"Negative-curvature decreases observed for {successful}/{len(rows)} tested steps.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    arguments = parser.parse_args()
    main(arguments.config)
