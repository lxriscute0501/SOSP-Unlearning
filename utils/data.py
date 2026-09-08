"""TOFU loading and answer-only causal-LM preprocessing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


QUESTION_COLUMNS = ("question", "prompt", "query", "instruction")
ANSWER_COLUMNS = ("answer", "response", "completion", "output")


def _first_present(columns: list[str], candidates: tuple[str, ...], kind: str) -> str:
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Could not infer the {kind} column from columns: {columns}")


def load_split(spec: dict[str, Any]) -> Dataset:
    """Load a Hugging Face split or a local JSON/JSONL/CSV file."""
    path = str(spec["path"])
    local_path = Path(path).expanduser()
    kwargs: dict[str, Any] = {}
    if local_path.exists() and local_path.is_file():
        extension = local_path.suffix.lower()
        builder = "json" if extension in {".json", ".jsonl"} else extension.lstrip(".")
        kwargs["data_files"] = str(local_path)
        dataset = load_dataset(builder, split=spec.get("split", "train"), **kwargs)
    else:
        dataset = load_dataset(
            path,
            spec.get("name"),
            split=spec.get("split", "train"),
            trust_remote_code=spec.get("trust_remote_code", False),
        )
    max_examples = spec.get("max_examples")
    if max_examples is not None:
        dataset = dataset.select(range(min(int(max_examples), len(dataset))))
    return dataset


class AnswerOnlyDataset(torch.utils.data.Dataset):
    """Tokenize QA examples and mask prompt tokens in the labels."""

    def __init__(self, dataset: Dataset, tokenizer: Any, spec: dict[str, Any], max_length: int):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        columns = list(dataset.column_names)
        self.question_column = spec.get("question_column") or _first_present(
            columns, QUESTION_COLUMNS, "question"
        )
        self.answer_column = spec.get("answer_column") or _first_present(
            columns, ANSWER_COLUMNS, "answer"
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def _strings(self, question: str, answer: str) -> tuple[str, str]:
        if getattr(self.tokenizer, "chat_template", None):
            prompt = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"Question: {question}\nAnswer:"
        eos = self.tokenizer.eos_token or ""
        return prompt, f"{prompt}{answer}{eos}"

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.dataset[index]
        prompt, full_text = self._strings(
            str(row[self.question_column]), str(row[self.answer_column])
        )
        prompt_ids = self.tokenizer(
            prompt, add_special_tokens=False, truncation=True, max_length=self.max_length
        )["input_ids"]
        input_ids = self.tokenizer(
            full_text, add_special_tokens=False, truncation=True, max_length=self.max_length
        )["input_ids"]
        labels = list(input_ids)
        labels[: min(len(prompt_ids), len(labels))] = [-100] * min(len(prompt_ids), len(labels))
        if not any(label != -100 for label in labels):
            raise ValueError(
                "An example has no answer tokens after truncation. Increase data.max_length."
            )
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}


@dataclass
class CausalCollator:
    pad_token_id: int

    def __call__(self, rows: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_length = max(len(row["input_ids"]) for row in rows)
        batch: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
        for row in rows:
            padding = max_length - len(row["input_ids"])
            batch["input_ids"].append(row["input_ids"] + [self.pad_token_id] * padding)
            batch["attention_mask"].append(row["attention_mask"] + [0] * padding)
            batch["labels"].append(row["labels"] + [-100] * padding)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


def make_dataloader(
    spec: dict[str, Any],
    tokenizer: Any,
    *,
    max_length: int,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = AnswerOnlyDataset(load_split(spec), tokenizer, spec, max_length)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=int(spec.get("num_workers", 0)),
        pin_memory=torch.cuda.is_available(),
        collate_fn=CausalCollator(tokenizer.pad_token_id),
        drop_last=False,
    )


def fixed_batches(loader: DataLoader, count: int) -> list[dict[str, torch.Tensor]]:
    """Materialize a deterministic prefix used consistently by every checkpoint."""
    batches: list[dict[str, torch.Tensor]] = []
    for batch in loader:
        batches.append(batch)
        if len(batches) >= count:
            break
    if not batches:
        raise ValueError("The configured dataset is empty.")
    return batches

