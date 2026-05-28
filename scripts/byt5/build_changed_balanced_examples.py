#!/usr/bin/env python3
"""Build changed-focused ByT5 examples with unchanged downsampling.

For lexical normalization, most tokens are unchanged.  Training on every token
can make the model learn copying too strongly, while training only changed
tokens can make it over-edit.  This builder keeps all changed tokens and samples
a controlled number of unchanged tokens.

Example setting:

    --lang ko --unchanged-ratio 2

means:

    all Korean changed examples + 2 * (#changed examples) Korean unchanged examples

The output is still normal ByT5 JSONL and can be used with finetune_byt5.py.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import polars as pl


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read a parquet split into dictionaries."""
    path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def make_marked_input(raw_tokens: list[str], token_id: int) -> str:
    """Insert T5 sentinel tokens around the target token."""
    marked = (
        raw_tokens[:token_id]
        + ["<extra_id_0>", raw_tokens[token_id], "<extra_id_1>"]
        + raw_tokens[token_id + 1 :]
    )
    return " ".join(marked)


def collect_examples(rows: list[dict[str, Any]], lang: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return changed and unchanged examples for one language."""
    changed_examples = []
    unchanged_examples = []

    for sentence_id, row in enumerate(rows):
        if row["lang"] != lang:
            continue

        raw_tokens = row["raw"]
        norm_tokens = row["norm"]
        if len(raw_tokens) != len(norm_tokens):
            raise ValueError(
                f"Length mismatch for lang={lang}, sentence_id={sentence_id}: "
                f"raw={len(raw_tokens)} norm={len(norm_tokens)}"
            )

        for token_id, (raw_token, target_token) in enumerate(zip(raw_tokens, norm_tokens)):
            changed = raw_token != target_token
            example = {
                "lang": lang,
                "sentence_id": sentence_id,
                "token_id": token_id,
                "raw_token": raw_token,
                "target_token": target_token,
                "input_text": make_marked_input(raw_tokens, token_id),
                "target_text": target_token,
                "changed": changed,
            }
            if changed:
                changed_examples.append(example)
            else:
                unchanged_examples.append(example)

    return changed_examples, unchanged_examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    """Write examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/multilexnorm2026-dev-pub"),
        help="Path to downloaded dataset snapshot.",
    )
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--lang", required=True, help="Language code, e.g. ko.")
    parser.add_argument(
        "--unchanged-ratio",
        type=float,
        default=2.0,
        help="How many unchanged examples to sample per changed example.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSONL path. If omitted, a name is generated from lang/ratio.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = read_split(args.data_dir, args.split)
    changed, unchanged = collect_examples(rows, args.lang)
    if not changed:
        raise RuntimeError(f"No changed examples found for lang={args.lang}")

    rng = random.Random(args.seed)
    rng.shuffle(changed)
    rng.shuffle(unchanged)

    n_unchanged = min(len(unchanged), int(round(len(changed) * args.unchanged_ratio)))
    selected = changed + unchanged[:n_unchanged]
    rng.shuffle(selected)

    if args.output is None:
        ratio_name = str(args.unchanged_ratio).replace(".", "p")
        args.output = Path(f"sample_data/byt5/changed_balanced/{args.split}_{args.lang}_changed_unchanged{ratio_name}.jsonl")

    write_jsonl(args.output, selected)

    summary = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "lang": args.lang,
        "unchanged_ratio": args.unchanged_ratio,
        "total_changed_available": len(changed),
        "total_unchanged_available": len(unchanged),
        "written_total": len(selected),
        "written_changed": len(changed),
        "written_unchanged": n_unchanged,
        "output": str(args.output),
    }
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print("Changed-balanced ByT5 examples built")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
