#!/usr/bin/env python3
"""Build UFAL-style ByT5 token-level examples for MultiLexNorm2026.

This script does not train a model.  It only converts the official parquet
data into examples that can later be fed to ByT5 fine-tuning.

The UFAL 2021 idea is:

    input:  left context <extra_id_0> raw_token <extra_id_1> right context
    target: normalized_token

For example:

    raw sentence:    Jeg skaelver .
    target token:    skaelver
    input_text:      Jeg <extra_id_0> skaelver <extra_id_1> .
    target_text:     skælver

We keep sentence_id/token_id/lang so predictions can later be assembled back
into the original sentence order.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import polars as pl


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read one MultiLexNorm2026 parquet split."""
    path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def make_marked_input(raw_tokens: list[str], token_id: int) -> str:
    """Insert T5 sentinel tokens around the target raw token.

    ByT5 uses the same sentinel token strings as T5, e.g. <extra_id_0>.
    The model sees the full sentence context and learns to output only the
    normalized form of the marked token.
    """
    marked = (
        raw_tokens[:token_id]
        + ["<extra_id_0>", raw_tokens[token_id], "<extra_id_1>"]
        + raw_tokens[token_id + 1 :]
    )
    return " ".join(marked)


def iter_examples(
    rows: list[dict[str, Any]],
    *,
    only_changed: bool,
    max_examples: int | None,
) -> Iterable[dict[str, Any]]:
    """Yield token-level examples from sentence-level rows.

    only_changed=False reproduces the UFAL fine-tuning setup more directly:
    every token becomes one example, including tokens whose normalization is
    the same as the raw form.

    only_changed=True creates a smaller dataset focused on actual edits.  It
    may be useful for quick smoke tests, but it changes the training
    distribution and should be reported clearly if used for experiments.
    """
    n_examples = 0
    for sentence_id, row in enumerate(rows):
        raw_tokens = row["raw"]
        norm_tokens = row["norm"]
        lang = row["lang"]

        if len(raw_tokens) != len(norm_tokens):
            raise ValueError(
                f"Length mismatch at sentence {sentence_id}: "
                f"raw={len(raw_tokens)} norm={len(norm_tokens)}"
            )

        for token_id, (raw_token, target_token) in enumerate(zip(raw_tokens, norm_tokens)):
            changed = raw_token != target_token
            if only_changed and not changed:
                continue

            yield {
                "lang": lang,
                "sentence_id": sentence_id,
                "token_id": token_id,
                "raw_token": raw_token,
                "target_token": target_token,
                "input_text": make_marked_input(raw_tokens, token_id),
                "target_text": target_token,
                "changed": changed,
            }

            n_examples += 1
            if max_examples is not None and n_examples >= max_examples:
                return


def write_jsonl(examples: Iterable[dict[str, Any]], output_path: Path) -> int:
    """Write examples as JSON Lines and return the number of rows."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/multilexnorm2026-dev-pub"),
        help="Path to the downloaded Hugging Face dataset snapshot.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "validation", "test"],
        default="train",
        help="Dataset split to convert.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("sample_data/byt5/train_examples_sample.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Write only tokens where raw_token != target_token.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=200,
        help="Maximum examples to write. Use -1 to write all examples.",
    )
    args = parser.parse_args()

    max_examples = None if args.max_examples < 0 else args.max_examples
    rows = read_split(args.data_dir, args.split)
    examples = iter_examples(rows, only_changed=args.only_changed, max_examples=max_examples)
    n_written = write_jsonl(examples, args.output)

    print("ByT5 example builder complete")
    print(f"data_dir: {args.data_dir}")
    print(f"split: {args.split}")
    print(f"only_changed: {args.only_changed}")
    print(f"max_examples: {max_examples}")
    print(f"written: {n_written}")
    print(f"output: {args.output}")


if __name__ == "__main__":
    main()
