#!/usr/bin/env python3
"""Evaluate ByT5 with conservative copy-vs-edit margin decoding.

Korean has high leave-as-is accuracy, so over-editing hurts.  This evaluator
accepts a model edit only when the generated prediction is sufficiently more
likely than copying the raw token.

For each validation token:

1. Generate model_pred with ByT5.
2. Compute seq2seq negative log-likelihood for model_pred.
3. Compute seq2seq negative log-likelihood for raw_token.
4. margin = raw_nll - pred_nll

If margin >= threshold, accept model_pred.  Otherwise, copy raw_token.

Larger margin means the model prefers its generated prediction over copying the
raw token.  A threshold sweep lets us tune conservativeness on validation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class ValidationTokenDataset(Dataset):
    """Token-level validation examples with metadata."""

    def __init__(self, rows: list[dict[str, Any]], lang: str | None, limit_sentences: int | None):
        if lang is not None:
            rows = [row for row in rows if row["lang"] == lang]
        if limit_sentences is not None:
            rows = rows[:limit_sentences]

        self.rows = rows
        self.examples = []
        for sentence_id, row in enumerate(rows):
            raw_tokens = row["raw"]
            norm_tokens = row["norm"]
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
    return pl.read_parquet(data_dir / "data" / "validation-00000-of-00001.parquet").to_dicts()


def make_collate_fn(tokenizer, max_input_length: int):
    """Tokenize encoder inputs."""

    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        encoded = tokenizer(
            [item["input_text"] for item in batch],
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


def sequence_nll(model, tokenizer, input_ids, attention_mask, target_texts, max_target_length: int) -> torch.Tensor:
    """Compute per-example target negative log-likelihood.

    Returns the mean token NLL for each example, so targets of different byte
    lengths are more comparable.
    """
    targets = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    labels = targets["input_ids"].to(input_ids.device)
    decoder_attention_mask = targets["attention_mask"].to(input_ids.device)
    labels_for_model = labels.clone()
    labels_for_model[labels_for_model == tokenizer.pad_token_id] = -100

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels_for_model,
    )
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels_for_model.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)
    token_counts = decoder_attention_mask.sum(dim=1).clamp(min=1)
    return token_loss.sum(dim=1) / token_counts


def metrics(records: list[dict[str, Any]], pred_key: str) -> dict[str, float | int]:
    """Compute token-level metrics for a selected prediction field."""
    total = len(records)
    changed = sum(1 for r in records if r["raw_token"] != r["gold_token"])
    correct = sum(1 for r in records if r[pred_key] == r["gold_token"])
    lai = (total - changed) / total if total else 0.0
    acc = correct / total if total else 0.0
    err = (acc - lai) / (1 - lai) if total and lai < 1 else 0.0
    return {"total": total, "changed": changed, "correct": correct, "lai_accuracy": lai, "accuracy": acc, "err": err}


def apply_threshold(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Add conservative_prediction for one margin threshold."""
    out = []
    for r in records:
        item = dict(r)
        item["conservative_prediction"] = r["model_prediction"] if r["margin"] >= threshold else r["raw_token"]
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/multilexnorm2026-dev-pub"))
    parser.add_argument("--lang", default=None)
    parser.add_argument("--limit-sentences", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument(
        "--thresholds",
        default="-2,-1,-0.5,0,0.5,1,2,3,5",
        help="Comma-separated margin thresholds.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU evaluation. By default this script exits if CUDA is unavailable.",
    )
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    limit = None if args.limit_sentences < 0 else args.limit_sentences
    rows = read_validation(args.data_dir)
    dataset = ValidationTokenDataset(rows, lang=args.lang, limit_sentences=limit)
    if len(dataset) == 0:
        raise RuntimeError("No validation token examples.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
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

    print("ByT5 conservative margin evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"lang: {args.lang or 'all'}")
    print(f"tokens: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"thresholds: {thresholds}")

    records = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="generating+scoring"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                num_beams=1,
            )
            model_predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            raw_tokens = [item["raw_token"] for item in batch["items"]]

            pred_nll = sequence_nll(
                model, tokenizer, input_ids, attention_mask, model_predictions, args.max_target_length
            )
            raw_nll = sequence_nll(
                model, tokenizer, input_ids, attention_mask, raw_tokens, args.max_target_length
            )
            margins = raw_nll - pred_nll

            for item, model_pred, pred_loss, raw_loss, margin in zip(
                batch["items"], model_predictions, pred_nll, raw_nll, margins
            ):
                records.append(
                    {
                        "sentence_id": item["sentence_id"],
                        "token_id": item["token_id"],
                        "lang": item["lang"],
                        "raw_token": item["raw_token"],
                        "gold_token": item["gold_token"],
                        "model_prediction": model_pred,
                        "pred_nll": float(pred_loss.cpu()),
                        "raw_nll": float(raw_loss.cpu()),
                        "margin": float(margin.cpu()),
                    }
                )

    sweep = []
    for threshold in thresholds:
        thresholded = apply_threshold(records, threshold)
        m = metrics(thresholded, "conservative_prediction")
        changed_correct = sum(
            1
            for r in thresholded
            if r["raw_token"] != r["gold_token"] and r["conservative_prediction"] == r["gold_token"]
        )
        unchanged_overedit = sum(
            1
            for r in thresholded
            if r["raw_token"] == r["gold_token"] and r["conservative_prediction"] != r["raw_token"]
        )
        accepted_edits = sum(1 for r in thresholded if r["conservative_prediction"] != r["raw_token"])
        sweep.append(
            {
                "threshold": threshold,
                **m,
                "accepted_edits": accepted_edits,
                "changed_correct": changed_correct,
                "unchanged_overedit": unchanged_overedit,
            }
        )

    best = max(sweep, key=lambda x: x["err"])
    best_records = apply_threshold(records, best["threshold"])
    for src, dst in zip(records, best_records):
        src["best_conservative_prediction"] = dst["conservative_prediction"]

    records_path = args.output_dir / "margin_records.jsonl"
    sweep_path = args.output_dir / "threshold_sweep.json"
    with records_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    sweep_path.write_text(json.dumps({"sweep": sweep, "best": best}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Threshold sweep:")
    for row in sweep:
        print(
            f"threshold={row['threshold']:.2f} acc={row['accuracy']*100:.2f} "
            f"ERR={row['err']*100:.2f} accepted_edits={row['accepted_edits']} "
            f"changed_correct={row['changed_correct']} unchanged_overedit={row['unchanged_overedit']}"
        )
    print(f"best_threshold: {best['threshold']}")
    print(f"best_accuracy: {best['accuracy']*100:.2f}")
    print(f"best_ERR: {best['err']*100:.2f}")
    print(f"records: {records_path}")
    print(f"sweep: {sweep_path}")


if __name__ == "__main__":
    main()
