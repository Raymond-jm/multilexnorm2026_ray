#!/usr/bin/env python3
"""Evaluate a ByT5 checkpoint on MultiLexNorm2026 validation.

This script applies the UFAL-style token-level generation setup:

1. For each validation token, build:
   left context <extra_id_0> raw_token <extra_id_1> right context
2. Generate one normalized token with the checkpoint.
3. Assemble token predictions back into sentences.
4. Compute LAI accuracy, model accuracy, and ERR.

This can evaluate either a real fine-tuned checkpoint or a tiny smoke-test
checkpoint.  If the checkpoint is tiny, do not interpret the score as a model
result; use it only to verify that the validation pipeline works.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ValidationTokenDataset(Dataset):
    """Token-level validation dataset with enough metadata to reassemble output."""

    def __init__(self, rows: list[dict[str, Any]], limit_sentences: int | None = None, lang: str | None = None):
        # For language-specific experiments, filter validation rows before
        # converting each token into a generation example.  This lets us compare
        # a Korean-specific model only on Korean validation sentences.
        if lang is not None:
            rows = [row for row in rows if row["lang"] == lang]
        if limit_sentences is not None:
            rows = rows[:limit_sentences]
        self.rows = rows
        self.examples = []

        for sentence_id, row in enumerate(rows):
            raw_tokens = row["raw"]
            norm_tokens = row["norm"]
            if len(raw_tokens) != len(norm_tokens):
                raise ValueError(
                    f"Length mismatch at sentence {sentence_id}: "
                    f"raw={len(raw_tokens)} norm={len(norm_tokens)}"
                )

            for token_id, (raw_token, gold_token) in enumerate(zip(raw_tokens, norm_tokens)):
                marked = (
                    raw_tokens[:token_id]
                    + ["<extra_id_0>", raw_token, "<extra_id_1>"]
                    + raw_tokens[token_id + 1 :]
                )
                self.examples.append(
                    {
                        "sentence_id": sentence_id,
                        "token_id": token_id,
                        "lang": row["lang"],
                        "raw_token": raw_token,
                        "gold_token": gold_token,
                        "input_text": " ".join(marked),
                    }
                )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.examples[idx]


def read_validation(data_dir: Path) -> list[dict[str, Any]]:
    """Read validation parquet rows."""
    path = data_dir / "data" / "validation-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def make_collate_fn(tokenizer, max_input_length: int):
    """Tokenize validation inputs while preserving metadata."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        input_texts = [item["input_text"] for item in batch]
        encoded = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "items": batch,
        }

    return collate


def compute_metrics(records: list[dict[str, Any]]) -> dict[str, float | int]:
    """Compute token-level LAI accuracy, model accuracy, and ERR."""
    total = len(records)
    changed = sum(1 for r in records if r["raw_token"] != r["gold_token"])
    correct = sum(1 for r in records if r["prediction"] == r["gold_token"])
    lai = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    err = (accuracy - lai) / (1.0 - lai) if total and lai < 1.0 else 0.0
    return {
        "total": total,
        "changed": changed,
        "correct": correct,
        "lai_accuracy": lai,
        "accuracy": accuracy,
        "err": err,
    }


def compute_by_lang(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    """Compute metrics separately by language."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["lang"], []).append(record)
    return {lang: compute_metrics(items) for lang, items in sorted(grouped.items())}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a compact validation report."""
    overall = summary["overall"]
    lines = [
        "# ByT5 Checkpoint Validation Evaluation",
        "",
        f"Checkpoint: `{summary['checkpoint']}`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total tokens | {overall['total']} |",
        f"| Changed tokens | {overall['changed']} |",
        f"| Correct tokens | {overall['correct']} |",
        f"| LAI accuracy | {overall['lai_accuracy'] * 100:.2f} |",
        f"| Model accuracy | {overall['accuracy'] * 100:.2f} |",
        f"| ERR | {overall['err'] * 100:.2f} |",
        "",
        "## Per-language Results",
        "",
        "| Lang | Tokens | Changed | LAI Acc | Model Acc | ERR |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for lang, stats in summary["by_lang"].items():
        lines.append(
            f"| {lang} | {stats['total']} | {stats['changed']} | "
            f"{stats['lai_accuracy'] * 100:.2f} | {stats['accuracy'] * 100:.2f} | "
            f"{stats['err'] * 100:.2f} |"
        )

    lines += [
        "",
        "## Sample Predictions",
        "",
    ]
    for record in summary["sample_predictions"]:
        lines.append(
            f"- `{record['lang']}` raw=`{record['raw_token']}` "
            f"gold=`{record['gold_token']}` pred=`{record['prediction']}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("outputs/byt5/tiny_finetune/checkpoint"),
        help="Checkpoint directory to evaluate.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/multilexnorm2026-dev-pub"),
        help="Downloaded MultiLexNorm2026 dataset snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/byt5/validation_eval"),
        help="Where predictions and summaries will be saved.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--limit-sentences",
        type=int,
        default=50,
        help="Evaluate only the first N validation sentences. Use -1 for all validation.",
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Optional language filter, e.g. ko. If omitted, evaluate all validation languages.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU evaluation. By default this script exits if CUDA is unavailable.",
    )
    args = parser.parse_args()

    limit_sentences = None if args.limit_sentences < 0 else args.limit_sentences
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_validation(args.data_dir)
    dataset = ValidationTokenDataset(rows, limit_sentences=limit_sentences, lang=args.lang)
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset produced zero token examples.")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available, so evaluation was stopped before starting. "
            "Fix the GPU/CUDA environment or pass --allow-cpu intentionally."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.checkpoint, use_safetensors=True).to(device)
    model.eval()

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(tokenizer, args.max_input_length),
    )

    print("ByT5 validation evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"device: {device}")
    print(f"lang: {args.lang or 'all'}")
    print(f"sentences: {len(dataset.rows)}")
    print(f"token examples: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")

    records = []
    progress = tqdm(dataloader, desc="generating", total=len(dataloader))
    with torch.no_grad():
        for batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            for item, prediction in zip(batch["items"], predictions):
                records.append(
                    {
                        "sentence_id": item["sentence_id"],
                        "token_id": item["token_id"],
                        "lang": item["lang"],
                        "raw_token": item["raw_token"],
                        "gold_token": item["gold_token"],
                        "prediction": prediction,
                    }
                )

    summary = {
        "checkpoint": str(args.checkpoint),
        "data_dir": str(args.data_dir),
        "lang": args.lang,
        "limit_sentences": limit_sentences,
        "overall": compute_metrics(records),
        "by_lang": compute_by_lang(records),
        "sample_predictions": records[:30],
    }

    predictions_path = args.output_dir / "predictions.jsonl"
    summary_json_path = args.output_dir / "summary.json"
    summary_md_path = args.output_dir / "summary.md"

    with predictions_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary_json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, summary_md_path)

    overall = summary["overall"]
    print("Validation evaluation complete")
    print(f"LAI accuracy: {overall['lai_accuracy'] * 100:.2f}")
    print(f"Model accuracy: {overall['accuracy'] * 100:.2f}")
    print(f"ERR: {overall['err'] * 100:.2f}")
    print(f"predictions: {predictions_path}")
    print(f"summary: {summary_md_path}")


if __name__ == "__main__":
    main()
