#!/usr/bin/env python3
"""Analyze where the MFR baseline fails on MultiLexNorm2026 validation.

This script is intentionally small and explicit so the experiment can be
explained in the report.  It reproduces the baseline repository's MFR idea:

1. Build a dictionary from train: raw token -> most frequent norm token.
2. Predict validation tokens with that dictionary.
3. Split validation errors into seen/unseen raw token cases.

The key question is whether MFR errors often come from raw tokens that never
appeared in train.  If yes, this motivates a pretrained byte-level model such
as ByT5, because ByT5 can learn character/byte patterns instead of only
memorizing exact token mappings.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read one parquet split into Python dictionaries.

    Each row has:
    - raw: list[str]
    - norm: list[str]
    - lang: str
    """
    path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def build_mfr_counts(rows: list[dict[str, Any]]) -> dict[str, Counter[str]]:
    """Count norm replacements for each raw token over training rows."""
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        for raw, norm in zip(row["raw"], row["norm"]):
            counts[raw][norm] += 1
    return counts


def predict_token(raw: str, counts: dict[str, Counter[str]]) -> str:
    """Return the most frequent replacement, or copy unseen tokens."""
    if raw not in counts:
        return raw
    return counts[raw].most_common(1)[0][0]


def safe_err(accuracy: float, lai: float) -> float:
    """Compute ERR while avoiding division by zero for degenerate subsets."""
    denom = 1.0 - lai
    if denom == 0:
        return 0.0
    return (accuracy - lai) / denom


def summarize_bucket(stats: dict[str, int]) -> dict[str, float | int]:
    """Convert raw counts into accuracy/LAI/ERR for one bucket."""
    total = stats["total"]
    if total == 0:
        return {
            "total": 0,
            "changed": 0,
            "correct": 0,
            "lai_accuracy": 0.0,
            "accuracy": 0.0,
            "err": 0.0,
        }

    lai = (total - stats["changed"]) / total
    accuracy = stats["correct"] / total
    return {
        "total": total,
        "changed": stats["changed"],
        "correct": stats["correct"],
        "lai_accuracy": lai,
        "accuracy": accuracy,
        "err": safe_err(accuracy, lai),
    }


def add_token(stats: dict[str, int], raw: str, gold: str, pred: str) -> None:
    """Update token-level evaluation counters."""
    stats["total"] += 1
    if raw != gold:
        stats["changed"] += 1
    if pred == gold:
        stats["correct"] += 1


def analyze(train_rows: list[dict[str, Any]], val_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Run MFR prediction and collect seen/unseen error analysis."""
    counts = build_mfr_counts(train_rows)
    seen_raw_tokens = set(counts)

    overall = defaultdict(int)
    by_lang: dict[str, Any] = defaultdict(lambda: defaultdict(int))
    error_buckets = Counter()
    changed_token_buckets = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for sent_id, row in enumerate(val_rows):
        lang = row["lang"]
        for tok_id, (raw, gold) in enumerate(zip(row["raw"], row["norm"])):
            pred = predict_token(raw, counts)
            seen = raw in seen_raw_tokens
            changed = raw != gold
            correct = pred == gold

            add_token(overall, raw, gold, pred)
            add_token(by_lang[lang], raw, gold, pred)

            if changed:
                changed_token_buckets["seen_changed" if seen else "unseen_changed"] += 1

            if not correct:
                key = "seen_error" if seen else "unseen_error"
                error_buckets[key] += 1

                # Keep a few human-readable examples for the report.
                if len(examples[key]) < 30:
                    examples[key].append(
                        {
                            "lang": lang,
                            "sentence_id": sent_id,
                            "token_id": tok_id,
                            "raw": raw,
                            "gold": gold,
                            "pred": pred,
                            "sentence_raw": row["raw"],
                            "sentence_norm": row["norm"],
                        }
                    )

    summary = {
        "overall": summarize_bucket(overall),
        "by_lang": {lang: summarize_bucket(stats) for lang, stats in sorted(by_lang.items())},
        "mfr_dictionary_size": len(seen_raw_tokens),
        "error_buckets": dict(error_buckets),
        "changed_token_buckets": dict(changed_token_buckets),
        "examples": examples,
    }

    total_errors = sum(error_buckets.values())
    if total_errors:
        summary["error_bucket_percent"] = {
            key: value / total_errors for key, value in error_buckets.items()
        }
    else:
        summary["error_bucket_percent"] = {}

    total_changed = sum(changed_token_buckets.values())
    if total_changed:
        summary["changed_token_bucket_percent"] = {
            key: value / total_changed for key, value in changed_token_buckets.items()
        }
    else:
        summary["changed_token_bucket_percent"] = {}

    return summary


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown report for quick inspection."""
    overall = summary["overall"]
    lines = [
        "# MFR Validation Error Analysis",
        "",
        "Dataset: `multilexnorm2026-dev-pub`",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| Total tokens | {overall['total']} |",
        f"| Changed tokens | {overall['changed']} |",
        f"| Correct tokens | {overall['correct']} |",
        f"| LAI accuracy | {overall['lai_accuracy'] * 100:.2f} |",
        f"| MFR accuracy | {overall['accuracy'] * 100:.2f} |",
        f"| ERR | {overall['err'] * 100:.2f} |",
        f"| MFR dictionary size | {summary['mfr_dictionary_size']} |",
        "",
        "## Error Buckets",
        "",
        "| Bucket | Count | Percent of MFR errors |",
        "| --- | ---: | ---: |",
    ]

    for key, count in sorted(summary["error_buckets"].items()):
        pct = summary["error_bucket_percent"].get(key, 0.0) * 100
        lines.append(f"| {key} | {count} | {pct:.2f} |")

    lines += [
        "",
        "## Changed Token Buckets",
        "",
        "| Bucket | Count | Percent of changed tokens |",
        "| --- | ---: | ---: |",
    ]
    for key, count in sorted(summary["changed_token_buckets"].items()):
        pct = summary["changed_token_bucket_percent"].get(key, 0.0) * 100
        lines.append(f"| {key} | {count} | {pct:.2f} |")

    lines += [
        "",
        "## Per-language Results",
        "",
        "| Lang | Tokens | Changed | LAI Acc | MFR Acc | ERR |",
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
        "## Example Errors",
        "",
        "### Unseen Raw Token Errors",
        "",
    ]
    for ex in summary["examples"].get("unseen_error", [])[:10]:
        lines.append(
            f"- `{ex['lang']}` raw=`{ex['raw']}` gold=`{ex['gold']}` pred=`{ex['pred']}`"
        )

    lines += ["", "### Seen Raw Token Errors", ""]
    for ex in summary["examples"].get("seen_error", [])[:10]:
        lines.append(
            f"- `{ex['lang']}` raw=`{ex['raw']}` gold=`{ex['gold']}` pred=`{ex['pred']}`"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/multilexnorm2026-dev-pub"),
        help="Path to the downloaded Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/mfr/analysis"),
        help="Directory where JSON and Markdown summaries will be written.",
    )
    args = parser.parse_args()

    train_rows = read_split(args.data_dir, "train")
    val_rows = read_split(args.data_dir, "validation")
    summary = analyze(train_rows, val_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "mfr_validation_error_analysis.json"
    md_path = args.output_dir / "mfr_validation_error_analysis.md"

    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_markdown(summary, md_path)

    overall = summary["overall"]
    print("MFR validation analysis complete")
    print(f"LAI accuracy: {overall['lai_accuracy'] * 100:.2f}")
    print(f"MFR accuracy: {overall['accuracy'] * 100:.2f}")
    print(f"ERR: {overall['err'] * 100:.2f}")
    print(f"Wrote: {json_path}")
    print(f"Wrote: {md_path}")


if __name__ == "__main__":
    main()
