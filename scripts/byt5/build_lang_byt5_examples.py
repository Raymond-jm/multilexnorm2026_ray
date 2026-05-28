#!/usr/bin/env python3
"""Build language-specific ByT5 examples for MultiLexNorm2026.

This script creates one JSONL file per language so we can train routed
language-specific ByT5 models, similar in spirit to the UFAL 2021 setup.

Each example follows the same context-marked format:

    input_text:  left context <extra_id_0> raw_token <extra_id_1> right context
    target_text: norm_token

By default, it builds examples for all languages in the train split.
Use --lang ko to build only one language.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read a parquet split into dictionaries."""
    path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def make_marked_input(raw_tokens: list[str], token_id: int) -> str:
    """Insert T5 sentinel tokens around the target raw token."""
    marked = (
        raw_tokens[:token_id]
        + ["<extra_id_0>", raw_tokens[token_id], "<extra_id_1>"]
        + raw_tokens[token_id + 1 :]
    )
    return " ".join(marked)


def build_examples_for_language(
    rows: list[dict[str, Any]],
    lang: str,
    *,
    only_changed: bool,
) -> list[dict[str, Any]]:
    """Build all token-level examples for one language."""
    examples = []
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
            if only_changed and not changed:
                continue

            examples.append(
                {
                    "lang": lang,
                    "sentence_id": sentence_id,
                    "token_id": token_id,
                    "raw_token": raw_token,
                    "target_token": target_token,
                    "input_text": make_marked_input(raw_tokens, token_id),
                    "target_text": target_token,
                    "changed": changed,
                }
            )
    return examples


def maybe_sample(examples: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    """Shuffle and optionally downsample examples for one language."""
    rng = random.Random(seed)
    rng.shuffle(examples)
    if max_examples is not None:
        return examples[:max_examples]
    return examples


def write_jsonl(path: Path, examples: list[dict[str, Any]]) -> None:
    """Write one language's examples as JSONL."""
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
    parser.add_argument(
        "--lang",
        default="all",
        help="Language code to build, e.g. ko. Use 'all' for all languages.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("sample_data/byt5/lang"),
        help="Directory for language-specific JSONL files.",
    )
    parser.add_argument(
        "--max-examples-per-lang",
        type=int,
        default=-1,
        help="Maximum examples per language. -1 means use all.",
    )
    parser.add_argument(
        "--only-changed",
        action="store_true",
        help="Keep only tokens where raw_token != target_token.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    max_examples = None if args.max_examples_per_lang < 0 else args.max_examples_per_lang
    rows = read_split(args.data_dir, args.split)
    available_langs = sorted({row["lang"] for row in rows})
    langs = available_langs if args.lang == "all" else [args.lang]

    unknown = [lang for lang in langs if lang not in available_langs]
    if unknown:
        raise ValueError(f"Unknown language(s): {unknown}. Available: {available_langs}")

    summary = {
        "data_dir": str(args.data_dir),
        "split": args.split,
        "only_changed": args.only_changed,
        "max_examples_per_lang": max_examples,
        "languages": {},
    }

    print("Building language-specific ByT5 examples")
    print(f"split: {args.split}")
    print(f"languages: {', '.join(langs)}")
    print(f"output_dir: {args.output_dir}")

    for lang in langs:
        examples = build_examples_for_language(rows, lang, only_changed=args.only_changed)
        raw_count = len(examples)
        examples = maybe_sample(examples, max_examples=max_examples, seed=args.seed)
        output_path = args.output_dir / f"{args.split}_{lang}_examples.jsonl"
        write_jsonl(output_path, examples)

        changed_count = sum(1 for ex in examples if ex["changed"])
        sentence_ids = {ex["sentence_id"] for ex in examples}
        token_counts = Counter("changed" if ex["changed"] else "unchanged" for ex in examples)

        summary["languages"][lang] = {
            "raw_examples_before_sampling": raw_count,
            "written_examples": len(examples),
            "changed_examples": changed_count,
            "unchanged_examples": token_counts.get("unchanged", 0),
            "sentence_count_with_examples": len(sentence_ids),
            "output_path": str(output_path),
        }
        print(
            f"{lang}: written={len(examples)} changed={changed_count} "
            f"output={output_path}"
        )

    summary_path = args.output_dir / f"{args.split}_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
