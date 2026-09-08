"""Shared configuration, reproducibility, model, and checkpoint helpers."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import yaml
from peft import (
    LoraConfig,
    PeftModel,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_config(path: str | Path) -> dict[str, Any]:
    """Load YAML and resolve all experiment paths relative to the config file."""
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {config_path}")

    config["_config_path"] = str(config_path)
    config["_project_dir"] = str(config_path.parent)
    for section, keys in {
        "training": ("output_dir",),
        "analysis": ("results_dir", "checkpoints_dir"),
    }.items():
        for key in keys:
            value = config.get(section, {}).get(key)
            if value and not Path(value).expanduser().is_absolute():
                config[section][key] = str((config_path.parent / value).resolve())
    return config


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible checkpoint comparisons."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "The config requests CUDA, but CUDA is unavailable. A 3B experiment needs "
            "a suitable GPU; use model.device: cpu only for smoke tests."
        )
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("The config requests MPS, but MPS is unavailable.")
    return device


def _torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported model dtype: {name}") from exc


def load_model_and_tokenizer(
    config: dict[str, Any],
    adapter_path: str | Path | None = None,
    *,
    for_training: bool,
) -> tuple[torch.nn.Module, Any, torch.device]:
    """Load one base model and a trainable LoRA adapter.

    Only adapter parameters are trainable. This makes both the BLUR projection and
    the Hessian live in a tractable, scientifically explicit parameter subspace.
    """
    model_cfg = config["model"]
    lora_cfg = config["lora"]
    device = resolve_device(model_cfg.get("device", "cuda"))
    dtype = _torch_dtype(model_cfg.get("dtype", "bfloat16"))
    quantization = model_cfg.get("quantization", "none").lower()

    tokenizer_name = model_cfg.get("tokenizer_name_or_path", model_cfg["name_or_path"])
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        trust_remote_code=model_cfg.get("trust_remote_code", False),
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": model_cfg.get("trust_remote_code", False),
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        # Fused SDPA/FlashAttention kernels do not consistently implement the
        # derivative of backward, which HVPs require. Eager attention is slower
        # but gives a reliable double-autograd path across PyTorch versions.
        "attn_implementation": model_cfg.get("attn_implementation", "eager"),
    }
    if quantization == "4bit":
        if device.type != "cuda":
            raise RuntimeError("4-bit loading requires a CUDA device and bitsandbytes.")
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["device_map"] = {"": device.index or 0}

    base_model = AutoModelForCausalLM.from_pretrained(model_cfg["name_or_path"], **model_kwargs)
    base_model.config.use_cache = False
    if quantization != "4bit":
        base_model.to(device)
    else:
        base_model = prepare_model_for_kbit_training(
            base_model, use_gradient_checkpointing=False
        )

    if adapter_path is None:
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=int(lora_cfg.get("rank", 8)),
            lora_alpha=int(lora_cfg.get("alpha", 16)),
            lora_dropout=float(lora_cfg.get("dropout", 0.0)),
            target_modules=list(lora_cfg.get("target_modules", ["q_proj", "v_proj"])),
            bias=lora_cfg.get("bias", "none"),
        )
        model = get_peft_model(base_model, peft_config)
    else:
        model = PeftModel.from_pretrained(base_model, str(adapter_path), is_trainable=True)

    # Reentrant activation checkpointing is incompatible with autograd.grad. The
    # non-reentrant variant works for BLUR, while analysis loads without it.
    if for_training and model_cfg.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable parameters were found after LoRA initialization.")
    return model, tokenizer, device


def trainable_parameters(model: torch.nn.Module) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def trainable_parameter_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def move_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def cycle(loader: Iterable[Any]) -> Iterable[Any]:
    while True:
        yield from loader


def checkpoint_step(path: str | Path) -> int:
    name = Path(path).name
    try:
        return int(name.rsplit("_", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Checkpoint directory must end in an integer: {name}") from exc


def discover_checkpoints(path: str | Path) -> list[Path]:
    root = Path(path)
    checkpoints = [item for item in root.glob("checkpoint_*") if item.is_dir()]
    return sorted(checkpoints, key=checkpoint_step)


def save_json(path: str | Path, value: Any) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_path)
