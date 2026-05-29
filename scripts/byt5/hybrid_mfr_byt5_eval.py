#!/usr/bin/env python3
"""Evaluate an MFR-anchored conservative ByT5 hybrid system.

The hybrid rule is:

    base_pred = MFR(raw_token)
    if ByT5 prediction is safe and margin >= threshold:
        final_pred = ByT5 prediction
    else:
        final_pred = base_pred

where:

    margin = raw_nll - byt5_pred_nll

This means ByT5 can override MFR only when the model prefers its generated
prediction over copying the raw token by enough margin.  MFR remains the safe
default because it is strong for seen lexical mappings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


class TokenDataset(Dataset):
    """Validation tokens with context-marked ByT5 inputs."""

    def __init__(self, rows: list[dict[str, Any]], lang: str | None):
        if lang is not None:
            rows = [row for row in rows if row["lang"] == lang]
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


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read train/validation split."""
    return pl.read_parquet(data_dir / "data" / f"{split}-00000-of-00001.parquet").to_dicts()


def build_mfr_counts(rows: list[dict[str, Any]], lang: str | None) -> dict[str, Counter[str]]:
    """Build MFR counts, optionally language-specific."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if lang is not None and row["lang"] != lang:
            continue
        for raw, norm in zip(row["raw"], row["norm"]):
            counts[raw][norm] += 1
    return counts


def mfr_predict(raw_token: str, counts: dict[str, Counter[str]]) -> str:
    """MFR prediction with raw fallback."""
    if raw_token not in counts:
        return raw_token
    return counts[raw_token].most_common(1)[0][0]


def make_collate_fn(tokenizer, max_input_length: int):
    """Tokenize context-marked inputs."""

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
    """Mean token negative log-likelihood per example."""
    targets = tokenizer(
        target_texts,
        padding=True,
        truncation=True,
        max_length=max_target_length,
        return_tensors="pt",
    )
    labels = targets["input_ids"].to(input_ids.device)
    target_mask = targets["attention_mask"].to(input_ids.device)
    labels_for_model = labels.clone()
    labels_for_model[labels_for_model == tokenizer.pad_token_id] = -100

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_for_model)
    logits = outputs.logits
    vocab_size = logits.shape[-1]
    token_loss = F.cross_entropy(
        logits.view(-1, vocab_size),
        labels_for_model.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(labels.shape)
    return token_loss.sum(dim=1) / target_mask.sum(dim=1).clamp(min=1)


def is_safe_prediction(prediction: str, raw_token: str, *, max_length_ratio: float, max_abs_length: int) -> bool:
    """Filter obviously unsafe generated strings."""
    pred = prediction.strip()
    if pred == "":
        return False
    if "\n" in pred or "\t" in pred:
        return False
    if len(pred) > max_abs_length:
        return False
    if len(raw_token) > 0 and len(pred) > max(len(raw_token) * max_length_ratio, len(raw_token) + 10):
        return False
    return True


def compute_metrics(records: list[dict[str, Any]], pred_key: str) -> dict[str, float | int]:
    """Compute LAI, accuracy, and ERR."""
    total = len(records)
    changed = sum(1 for r in records if r["raw_token"] != r["gold_token"])
    correct = sum(1 for r in records if r[pred_key] == r["gold_token"])
    lai = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    err = (accuracy - lai) / (1 - lai) if total and lai < 1 else 0.0
    return {"total": total, "changed": changed, "correct": correct, "lai_accuracy": lai, "accuracy": accuracy, "err": err}


def apply_hybrid(records: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    """Apply one threshold to produce final hybrid predictions."""
    out = []
    for record in records:
        item = dict(record)
        if record["safe"] and record["margin"] >= threshold:
            item["hybrid_prediction"] = record["byt5_prediction"]
            item["source"] = "byt5"
        else:
            item["hybrid_prediction"] = record["mfr_prediction"]
            item["source"] = "mfr"
        out.append(item)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/multilexnorm2026-dev-pub"))
    parser.add_argument("--lang", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--thresholds", default="-2,-1,-0.5,0,0.5,1,2,3,5")
    parser.add_argument("--max-length-ratio", type=float, default=3.0)
    parser.add_argument("--max-abs-length", type=int, default=40)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU evaluation. By default this script exits if CUDA is unavailable.",
    )
    args = parser.parse_args()

    thresholds = [float(x) for x in args.thresholds.split(",") if x.strip()]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = read_split(args.data_dir, "train")
    val_rows = read_split(args.data_dir, "validation")
    mfr_counts = build_mfr_counts(train_rows, args.lang)
    dataset = TokenDataset(val_rows, args.lang)
    if len(dataset) == 0:
        raise RuntimeError("No validation examples.")

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

    print("Hybrid MFR + ByT5 evaluation")
    print(f"checkpoint: {args.checkpoint}")
    print(f"lang: {args.lang or 'all'}")
    print(f"tokens: {len(dataset)}")
    print(f"mfr dictionary size: {len(mfr_counts)}")
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
            byt5_predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            raw_tokens = [item["raw_token"] for item in batch["items"]]
            pred_nll = sequence_nll(
                model, tokenizer, input_ids, attention_mask, byt5_predictions, args.max_target_length
            )
            raw_nll = sequence_nll(
                model, tokenizer, input_ids, attention_mask, raw_tokens, args.max_target_length
            )
            margins = raw_nll - pred_nll

            for item, byt5_pred, pred_loss, raw_loss, margin in zip(
                batch["items"], byt5_predictions, pred_nll, raw_nll, margins
            ):
                mfr_pred = mfr_predict(item["raw_token"], mfr_counts)
                records.append(
                    {
                        "sentence_id": item["sentence_id"],
                        "token_id": item["token_id"],
                        "lang": item["lang"],
                        "raw_token": item["raw_token"],
                        "gold_token": item["gold_token"],
                        "mfr_prediction": mfr_pred,
                        "byt5_prediction": byt5_pred,
                        "pred_nll": float(pred_loss.cpu()),
                        "raw_nll": float(raw_loss.cpu()),
                        "margin": float(margin.cpu()),
                        "safe": is_safe_prediction(
                            byt5_pred,
                            item["raw_token"],
                            max_length_ratio=args.max_length_ratio,
                            max_abs_length=args.max_abs_length,
                        ),
                    }
                )

    base_mfr = compute_metrics(records, "mfr_prediction")
    raw_copy_records = [dict(r, raw_prediction=r["raw_token"]) for r in records]
    base_lai = compute_metrics(raw_copy_records, "raw_prediction")
    byt5 = compute_metrics(records, "byt5_prediction")

    sweep = []
    for threshold in thresholds:
        hybrid_records = apply_hybrid(records, threshold)
        m = compute_metrics(hybrid_records, "hybrid_prediction")
        sweep.append(
            {
                "threshold": threshold,
                **m,
                "accepted_byt5": sum(1 for r in hybrid_records if r["source"] == "byt5"),
                "changed_correct": sum(
                    1
                    for r in hybrid_records
                    if r["raw_token"] != r["gold_token"] and r["hybrid_prediction"] == r["gold_token"]
                ),
                "unchanged_overedit": sum(
                    1
                    for r in hybrid_records
                    if r["raw_token"] == r["gold_token"] and r["hybrid_prediction"] != r["raw_token"]
                ),
            }
        )

    best = max(sweep, key=lambda row: row["err"])
    best_records = apply_hybrid(records, best["threshold"])

    records_path = args.output_dir / "hybrid_records.jsonl"
    best_predictions_path = args.output_dir / "best_hybrid_predictions.jsonl"
    summary_path = args.output_dir / "summary.json"

    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    with best_predictions_path.open("w", encoding="utf-8") as f:
        for record in best_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "checkpoint": str(args.checkpoint),
        "lang": args.lang,
        "base_lai": base_lai,
        "base_mfr": base_mfr,
        "byt5_direct": byt5,
        "sweep": sweep,
        "best": best,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Base LAI:", f"acc={base_lai['accuracy']*100:.2f}", f"ERR={base_lai['err']*100:.2f}")
    print("Base MFR:", f"acc={base_mfr['accuracy']*100:.2f}", f"ERR={base_mfr['err']*100:.2f}")
    print("Direct ByT5:", f"acc={byt5['accuracy']*100:.2f}", f"ERR={byt5['err']*100:.2f}")
    print("Threshold sweep:")
    for row in sweep:
        print(
            f"threshold={row['threshold']:.2f} acc={row['accuracy']*100:.2f} "
            f"ERR={row['err']*100:.2f} accepted_byt5={row['accepted_byt5']} "
            f"changed_correct={row['changed_correct']} unchanged_overedit={row['unchanged_overedit']}"
        )
    print(f"best_threshold: {best['threshold']}")
    print(f"best_accuracy: {best['accuracy']*100:.2f}")
    print(f"best_ERR: {best['err']*100:.2f}")
    print(f"summary: {summary_path}")
    print(f"records: {records_path}")


if __name__ == "__main__":
    main()
