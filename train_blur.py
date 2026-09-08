#!/usr/bin/env python3
"""Train the BLUR-NPO first-order baseline and save diagnostic checkpoints."""

from __future__ import annotations

import argparse
import csv
import time
from functools import partial
from pathlib import Path

from utils.blur import blur_step
from utils.common import (
    cycle,
    load_config,
    load_model_and_tokenizer,
    move_batch,
    save_json,
    set_seed,
    trainable_parameter_names,
    trainable_parameters,
)
from utils.data import fixed_batches, make_dataloader
from utils.metrics import first_order_statistics, forget_loss, make_objective_closure, retain_loss


def write_statistics(rows: list[dict[str, float | int | str]], results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    save_json(results_dir / "fosp_statistics.json", rows)
    columns = [
        "checkpoint",
        "step",
        "forget_loss",
        "retain_loss",
        "forget_grad_norm",
        "retain_grad_norm",
        "lagrange_multiplier",
        "bilevel_residual",
    ]
    with (results_dir / "fosp_statistics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main(config_path: str) -> None:
    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed, config.get("deterministic", True))
    model, tokenizer, device = load_model_and_tokenizer(config, for_training=True)
    data_cfg = config["data"]
    training_cfg = config["training"]
    analysis_cfg = config["analysis"]

    forget_loader = make_dataloader(
        data_cfg["forget"],
        tokenizer,
        max_length=int(data_cfg.get("max_length", 256)),
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        seed=seed,
    )
    retain_loader = make_dataloader(
        data_cfg["retain"],
        tokenizer,
        max_length=int(data_cfg.get("max_length", 256)),
        batch_size=int(training_cfg.get("batch_size", 1)),
        shuffle=True,
        seed=seed + 1,
    )
    evaluation_batch_size = int(analysis_cfg.get("batch_size", 1))
    forget_eval_loader = make_dataloader(
        data_cfg["forget"],
        tokenizer,
        max_length=int(data_cfg.get("max_length", 256)),
        batch_size=evaluation_batch_size,
        shuffle=False,
        seed=seed,
    )
    retain_eval_loader = make_dataloader(
        data_cfg["retain"],
        tokenizer,
        max_length=int(data_cfg.get("max_length", 256)),
        batch_size=evaluation_batch_size,
        shuffle=False,
        seed=seed,
    )
    evaluation_batches = int(analysis_cfg.get("num_loss_batches", 4))
    fixed_forget = fixed_batches(forget_eval_loader, evaluation_batches)
    fixed_retain = fixed_batches(retain_eval_loader, evaluation_batches)

    npo_cfg = config.get("forget_objective", {})
    forget_objective = partial(
        forget_loss,
        loss_type=npo_cfg.get("type", "npo"),
        beta=float(npo_cfg.get("beta", 0.1)),
    )
    checkpoint_root = Path(training_cfg["output_dir"])
    results_dir = Path(analysis_cfg["results_dir"])
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(parameter.numel() for parameter in trainable_parameters(model))
    save_json(
        checkpoint_root / "run_metadata.json",
        {
            "model": config["model"]["name_or_path"],
            "seed": seed,
            "total_parameters": total_parameters,
            "trainable_parameters": trainable_count,
            "trainable_parameter_names": trainable_parameter_names(model),
            "config_path": config["_config_path"],
        },
    )
    print(
        f"Loaded {config['model']['name_or_path']} on {device}; "
        f"training {trainable_count:,}/{total_parameters:,} parameters."
    )

    rows: list[dict[str, float | int | str]] = []

    def save_checkpoint(step: int) -> None:
        checkpoint = checkpoint_root / f"checkpoint_{step}"
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint, safe_serialization=True)
        model_was_training = model.training
        model.eval()
        forget_closure = make_objective_closure(
            model, fixed_forget, device, forget_objective
        )
        retain_closure = make_objective_closure(model, fixed_retain, device, retain_loss)
        statistics = first_order_statistics(
            model,
            forget_closure,
            retain_closure,
            epsilon=float(training_cfg.get("projection_epsilon", 1e-12)),
        )
        row: dict[str, float | int | str] = {
            "checkpoint": checkpoint.name,
            "step": step,
            **statistics,
        }
        rows.append(row)
        save_json(checkpoint / "statistics.json", row)
        write_statistics(rows, results_dir)
        if model_was_training:
            model.train()
        print(
            f"Saved {checkpoint.name}: g={statistics['forget_loss']:.6g}, "
            f"||grad g||={statistics['forget_grad_norm']:.6g}, "
            f"residual={statistics['bilevel_residual']:.6g}"
        )

    if training_cfg.get("save_initial", True):
        save_checkpoint(0)

    forget_iterator = iter(cycle(forget_loader))
    retain_iterator = iter(cycle(retain_loader))
    num_steps = int(training_cfg.get("num_steps", 1000))
    save_every = int(training_cfg.get("save_every", 100))
    log_every = int(training_cfg.get("log_every", 10))
    start = time.time()
    model.train()
    for step in range(1, num_steps + 1):
        forget_batch = move_batch(next(forget_iterator), device)
        retain_batch = move_batch(next(retain_iterator), device)
        diagnostics = blur_step(
            model,
            lambda: forget_objective(model, forget_batch),
            lambda: retain_loss(model, retain_batch),
            learning_rate=float(training_cfg.get("learning_rate", 5e-5)),
            gamma=float(training_cfg.get("gamma", 1.0)),
            epsilon=float(training_cfg.get("projection_epsilon", 1e-12)),
            max_update_norm=training_cfg.get("max_update_norm"),
        )
        if step % log_every == 0 or step == 1:
            elapsed = time.time() - start
            print(
                f"step={step}/{num_steps} forget={diagnostics['forget_loss']:.6g} "
                f"retain={diagnostics['retain_loss']:.6g} "
                f"||u||={diagnostics['update_norm']:.6g} elapsed={elapsed:.1f}s"
            )
        if step % save_every == 0:
            save_checkpoint(step)
    if num_steps % save_every:
        save_checkpoint(num_steps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    arguments = parser.parse_args()
    main(arguments.config)

