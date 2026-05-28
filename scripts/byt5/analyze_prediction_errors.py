#!/usr/bin/env python3
"""Analyze token-level prediction errors from evaluate_checkpoint.py output.

This script reads a predictions.jsonl file with raw_token, gold_token, and
prediction fields.  It separates errors into:

- unchanged_correct: raw == gold and prediction == gold
- unchanged_overedited: raw == gold but prediction changed it
- changed_correct: raw != gold and prediction == gold
- changed_copied_raw: raw != gold but prediction copied raw
- changed_wrong_other: raw != gold and prediction is neither raw nor gold

This is especially useful for checking whether a normalization model is
learning to edit changed tokens or merely copying/over-editing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def load_records(path: Path) -> list[dict[str, Any]]:
    """Load JSONL prediction records."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def bucket(record: dict[str, Any]) -> str:
    """Classify one prediction into an interpretable bucket."""
    raw = record["raw_token"]
    gold = record["gold_token"]
    pred = record["prediction"]
    changed = raw != gold
    correct = pred == gold
    copied = pred == raw

    if changed and correct:
        return "changed_correct"
    if changed and copied:
        return "changed_copied_raw"
    if changed:
        return "changed_wrong_other"
    if correct:
        return "unchanged_correct"
    return "unchanged_overedited"


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    """Write a compact Markdown error analysis."""
    lines = [
        "# Prediction Error Analysis",
        "",
        f"Predictions: `{summary['predictions_path']}`",
        "",
        "## Buckets",
        "",
        "| Bucket | Count | Percent |",
        "| --- | ---: | ---: |",
    ]
    total = summary["total"]
    for key, value in summary["buckets"].items():
        lines.append(f"| {key} | {value} | {value / total * 100:.2f} |")

    lines += [
        "",
        "## Changed/Unchanged Accuracy",
        "",
        "| Subset | Total | Correct | Accuracy |",
        "| --- | ---: | ---: | ---: |",
    ]
    for key in ["changed", "unchanged"]:
        stats = summary[key]
        acc = stats["correct"] / stats["total"] * 100 if stats["total"] else 0.0
        lines.append(f"| {key} | {stats['total']} | {stats['correct']} | {acc:.2f} |")

    lines += [
        "",
        "## Most Common Error Predictions",
        "",
        "| Prediction | Count |",
        "| --- | ---: |",
    ]
    for pred, count in summary["top_error_predictions"]:
        safe_pred = pred.replace("|", "\\|")
        lines.append(f"| `{safe_pred}` | {count} |")

    for section_key, title in [
        ("changed_error_examples", "Changed Token Error Examples"),
        ("unchanged_overedit_examples", "Unchanged Over-edit Examples"),
    ]:
        lines += ["", f"## {title}", ""]
        for record in summary[section_key]:
            lines.append(
                f"- raw=`{record['raw_token']}` gold=`{record['gold_token']}` "
                f"pred=`{record['prediction']}`"
            )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions",
        type=Path,
        required=True,
        help="Path to predictions.jsonl produced by evaluate_checkpoint.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where analysis JSON/Markdown files will be written.",
    )
    parser.add_argument("--num-examples", type=int, default=30)
    args = parser.parse_args()

    records = load_records(args.predictions)
    buckets = Counter(bucket(record) for record in records)

    changed = [r for r in records if r["raw_token"] != r["gold_token"]]
    unchanged = [r for r in records if r["raw_token"] == r["gold_token"]]
    errors = [r for r in records if r["prediction"] != r["gold_token"]]

    changed_error_examples = [
        r for r in changed if r["prediction"] != r["gold_token"]
    ][: args.num_examples]
    unchanged_overedit_examples = [
        r for r in unchanged if r["prediction"] != r["gold_token"]
    ][: args.num_examples]

    summary = {
        "predictions_path": str(args.predictions),
        "total": len(records),
        "buckets": dict(buckets),
        "changed": {
            "total": len(changed),
            "correct": sum(1 for r in changed if r["prediction"] == r["gold_token"]),
        },
        "unchanged": {
            "total": len(unchanged),
            "correct": sum(1 for r in unchanged if r["prediction"] == r["gold_token"]),
        },
        "top_error_predictions": Counter(r["prediction"] for r in errors).most_common(30),
        "changed_error_examples": changed_error_examples,
        "unchanged_overedit_examples": unchanged_overedit_examples,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "prediction_error_analysis.json"
    md_path = args.output_dir / "prediction_error_analysis.md"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(summary, md_path)

    print("Prediction error analysis complete")
    print(f"total: {summary['total']}")
    print(f"buckets: {summary['buckets']}")
    print(f"changed accuracy: {summary['changed']['correct']}/{summary['changed']['total']}")
    print(f"unchanged accuracy: {summary['unchanged']['correct']}/{summary['unchanged']['total']}")
    print(f"summary: {md_path}")


if __name__ == "__main__":
    main()
