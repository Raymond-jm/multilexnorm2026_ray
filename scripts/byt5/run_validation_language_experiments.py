#!/usr/bin/env python3
"""Run language-specific ByT5 training/evaluation and compare with MFR.

This script automates the experiment pattern we have been running manually:

1. Select only languages that have a public validation split.
2. Train one language-specific ByT5 checkpoint per language.
3. Evaluate that checkpoint on the same language's validation rows.
4. Compute MFR scores on the same validation language.
5. Write one comparison table for the report.

Important project convention:
- The user runs this script from the terminal.
- The language JSONL sample files are assumed to already exist under
  sample_data/byt5/lang.
- The default MFR comparison uses one multilingual dictionary built from the
  full train split, matching the provided baseline's spirit.  Use
  --mfr-mode language-specific only when intentionally comparing routed MFR.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def read_split(data_dir: Path, split: str) -> list[dict[str, Any]]:
    """Read one MultiLexNorm parquet split into Python dictionaries."""
    path = data_dir / "data" / f"{split}-00000-of-00001.parquet"
    return pl.read_parquet(path).to_dicts()


def build_mfr_counts(rows: list[dict[str, Any]], lang: str | None) -> dict[str, Counter[str]]:
    """Build raw-token -> most-frequent-normalization counts.

    lang=None means multilingual MFR: use all train rows.
    lang="de" means routed MFR: use only German train rows.
    """
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        if lang is not None and row["lang"] != lang:
            continue
        for raw_token, norm_token in zip(row["raw"], row["norm"]):
            counts[raw_token][norm_token] += 1
    return counts


def predict_mfr(raw_token: str, counts: dict[str, Counter[str]]) -> str:
    """Return MFR replacement; copy raw token if it was unseen in train."""
    if raw_token not in counts:
        return raw_token
    return counts[raw_token].most_common(1)[0][0]


def compute_err(accuracy: float, lai_accuracy: float) -> float:
    """Compute Error Reduction Rate with a guard for no-change subsets."""
    denominator = 1.0 - lai_accuracy
    if denominator == 0.0:
        return 0.0
    return (accuracy - lai_accuracy) / denominator


def compute_mfr_for_lang(
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    lang: str,
    mfr_mode: str,
) -> dict[str, float | int]:
    """Evaluate MFR on one validation language."""
    mfr_lang = None if mfr_mode == "global" else lang
    counts = build_mfr_counts(train_rows, mfr_lang)

    total = 0
    changed = 0
    correct = 0
    for row in validation_rows:
        if row["lang"] != lang:
            continue
        for raw_token, gold_token in zip(row["raw"], row["norm"]):
            prediction = predict_mfr(raw_token, counts)
            total += 1
            if raw_token != gold_token:
                changed += 1
            if prediction == gold_token:
                correct += 1

    lai_accuracy = (total - changed) / total if total else 0.0
    accuracy = correct / total if total else 0.0
    return {
        "total": total,
        "changed": changed,
        "correct": correct,
        "lai_accuracy": lai_accuracy,
        "accuracy": accuracy,
        "err": compute_err(accuracy, lai_accuracy),
        "dictionary_size": len(counts),
    }


def load_byt5_summary(summary_path: Path) -> dict[str, float | int] | None:
    """Load evaluate_checkpoint.py summary.json if it exists."""
    if not summary_path.exists():
        return None
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    return data["overall"]


def checkpoint_exists(checkpoint_dir: Path) -> bool:
    """Check whether a Hugging Face checkpoint directory looks complete."""
    return (
        checkpoint_dir.exists()
        and (checkpoint_dir / "config.json").exists()
        and (
            (checkpoint_dir / "pytorch_model.bin").exists()
            or (checkpoint_dir / "model.safetensors").exists()
        )
    )


def checkpoint_step(path: Path) -> int:
    """Return N for checkpoint_step_N directories, otherwise -1."""
    match = re.fullmatch(r"checkpoint_step_(\d+)", path.name)
    if not match:
        return -1
    return int(match.group(1))


def latest_intermediate_checkpoint(output_dir: Path) -> Path | None:
    """Find the newest checkpoint_step_N directory in an output directory."""
    candidates = [
        path
        for path in output_dir.glob("checkpoint_step_*")
        if path.is_dir() and checkpoint_exists(path) and checkpoint_step(path) >= 0
    ]
    if not candidates:
        return None
    return max(candidates, key=checkpoint_step)


def command_has_flag(command: str, flag: str) -> bool:
    """Check whether a command string already includes an argparse flag."""
    return flag in shlex.split(command)


def append_training_control_flags(
    command: str,
    *,
    save_every: int,
    resume_checkpoint: Path | None,
) -> str:
    """Append checkpointing/resume flags without changing YAML manually."""
    parts = shlex.split(command)
    if save_every > 0 and "--save-every" not in parts:
        parts += ["--save-every", str(save_every)]
    if resume_checkpoint is not None and "--resume-from-checkpoint" not in parts:
        parts += ["--resume-from-checkpoint", str(resume_checkpoint)]
    return shlex.join(parts)


def run_command(command: str, log_path: Path, *, capture_output: bool) -> None:
    """Run one command.

    By default, stdout/stderr stay connected to the real terminal.  This keeps
    tqdm progress bars compact, so the bar updates in place instead of printing
    many repeated progress lines.

    capture_output=True mirrors output into automation.log, but it can make
    tqdm look noisy because the child process no longer sees a real terminal.
    Use it only when the full raw terminal output must be archived.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n$ {command}", flush=True)

    with log_path.open("a", encoding="utf-8") as log_f:
        log_f.write(f"\n$ {command}\n")
        log_f.flush()

    if not capture_output:
        result = subprocess.run(shlex.split(command), check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Command failed with exit code {result.returncode}: {command}")
        return

    with log_path.open("a", encoding="utf-8") as log_f:
        process = subprocess.Popen(
            shlex.split(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_f.write(line)
            log_f.flush()
        returncode = process.wait()
    if returncode != 0:
        raise RuntimeError(f"Command failed with exit code {returncode}: {command}")


def format_pct(value: float | int | None) -> str:
    """Format a metric as percentage points for Markdown tables."""
    if value is None:
        return "TBD"
    return f"{float(value) * 100:.2f}"


def make_comparison_rows(
    cfg: dict[str, Any],
    train_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    langs: list[str],
    mfr_mode: str,
) -> list[dict[str, Any]]:
    """Collect current MFR and ByT5 metrics without running training."""
    rows = []
    for lang in langs:
        lang_cfg = cfg["languages"][lang]
        mfr = compute_mfr_for_lang(train_rows, validation_rows, lang, mfr_mode)
        summary_path = Path(lang_cfg["validation_output_dir"]) / "summary.json"
        byt5 = load_byt5_summary(summary_path)
        rows.append(
            {
                "lang": lang,
                "tokens": mfr["total"],
                "changed": mfr["changed"],
                "lai_accuracy": mfr["lai_accuracy"],
                "mfr_accuracy": mfr["accuracy"],
                "mfr_err": mfr["err"],
                "byt5_accuracy": None if byt5 is None else byt5["accuracy"],
                "byt5_err": None if byt5 is None else byt5["err"],
                "byt5_summary": str(summary_path) if summary_path.exists() else None,
                "delta_err_byt5_minus_mfr": (
                    None if byt5 is None else float(byt5["err"]) - float(mfr["err"])
                ),
            }
        )
    return rows


def macro_average(rows: list[dict[str, Any]], key: str) -> float | None:
    """Average a metric over languages where that metric exists."""
    values = [row[key] for row in rows if row[key] is not None]
    if not values:
        return None
    return sum(float(v) for v in values) / len(values)


def write_comparison(
    rows: list[dict[str, Any]],
    output_dir: Path,
    *,
    config_path: Path,
    mfr_mode: str,
) -> None:
    """Write JSON and Markdown summaries for report-friendly comparison."""
    output_dir.mkdir(parents=True, exist_ok=True)
    macro = {
        "mfr_err": macro_average(rows, "mfr_err"),
        "byt5_err_completed": macro_average(rows, "byt5_err"),
        "delta_err_byt5_minus_mfr_completed": macro_average(rows, "delta_err_byt5_minus_mfr"),
        "completed_byt5_languages": sum(1 for row in rows if row["byt5_err"] is not None),
        "total_languages": len(rows),
    }
    payload = {
        "config_path": str(config_path),
        "mfr_mode": mfr_mode,
        "macro": macro,
        "rows": rows,
    }

    json_path = output_dir / "comparison_summary.json"
    md_path = output_dir / "comparison_summary.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# Validation-language MFR vs ByT5 Comparison",
        "",
        f"Config: `{config_path}`",
        f"MFR mode: `{mfr_mode}`",
        "",
        "## Macro Summary",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| MFR macro ERR | {format_pct(macro['mfr_err'])} |",
        f"| ByT5 macro ERR over completed languages | {format_pct(macro['byt5_err_completed'])} |",
        f"| Completed ByT5 languages | {macro['completed_byt5_languages']} / {macro['total_languages']} |",
        "",
        "## Per-language Results",
        "",
        "| Lang | Tokens | Changed | LAI Acc | MFR Acc | MFR ERR | ByT5 Acc | ByT5 ERR | Delta ERR |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['lang']} | {row['tokens']} | {row['changed']} | "
            f"{format_pct(row['lai_accuracy'])} | {format_pct(row['mfr_accuracy'])} | "
            f"{format_pct(row['mfr_err'])} | {format_pct(row['byt5_accuracy'])} | "
            f"{format_pct(row['byt5_err'])} | {format_pct(row['delta_err_byt5_minus_mfr'])} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\ncomparison json: {json_path}")
    print(f"comparison md:   {md_path}")


def parse_langs(cfg: dict[str, Any], requested: str) -> list[str]:
    """Parse --langs while preserving config order."""
    available = cfg["validation_languages"]
    if requested == "all":
        return list(available)
    wanted = [item.strip() for item in requested.split(",") if item.strip()]
    unknown = [lang for lang in wanted if lang not in available]
    if unknown:
        raise ValueError(f"Unknown or non-validation language(s): {unknown}. Available: {available}")
    return wanted


def validate_example_counts(cfg: dict[str, Any], langs: list[str]) -> None:
    """Confirm YAML step counts still match existing sample files."""
    batch_size = int(cfg["defaults"]["batch_size"])
    for lang in langs:
        lang_cfg = cfg["languages"][lang]
        examples_path = Path(lang_cfg["train_examples"])
        if not examples_path.exists():
            raise FileNotFoundError(f"Missing sample file for {lang}: {examples_path}")
        line_count = sum(1 for _ in examples_path.open(encoding="utf-8"))
        expected_steps = math.ceil(line_count / batch_size)
        yaml_steps = int(lang_cfg["max_steps_1epoch_bs4"])
        if expected_steps != yaml_steps:
            raise ValueError(
                f"1-epoch step mismatch for {lang}: "
                f"lines={line_count}, batch_size={batch_size}, "
                f"expected={expected_steps}, yaml={yaml_steps}"
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/byt5/validation_languages.yaml"),
        help="YAML manifest with validation languages and commands.",
    )
    parser.add_argument(
        "--langs",
        default="all",
        help="Comma-separated validation languages, e.g. de,en,ko. Default: all.",
    )
    parser.add_argument(
        "--comparison-output-dir",
        type=Path,
        default=Path("outputs/byt5/validation_language_comparison"),
        help="Where the MFR vs ByT5 comparison table is written.",
    )
    parser.add_argument(
        "--mfr-mode",
        choices=["global", "language-specific"],
        default="global",
        help="global matches the multilingual MFR baseline; language-specific is routed MFR.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands and write current comparison without running train/eval.",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Do not train; only evaluate existing checkpoints and write comparison.",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Train only; useful if you want validation to run later.",
    )
    parser.add_argument(
        "--force-train",
        action="store_true",
        help="Retrain even if checkpoint/config files already exist.",
    )
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Re-run evaluation even if summary.json already exists.",
    )
    parser.add_argument(
        "--capture-child-output",
        action="store_true",
        help=(
            "Also write child process stdout/stderr into automation.log. "
            "This can make tqdm print many lines, so the default keeps output attached to the terminal."
        ),
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=None,
        help="Save intermediate training checkpoints every N steps. Defaults to YAML defaults.save_every.",
    )
    parser.add_argument(
        "--no-auto-resume",
        action="store_true",
        help="Do not resume from checkpoint_step_N even if one exists.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    langs = parse_langs(cfg, args.langs)
    validate_example_counts(cfg, langs)
    save_every = int(args.save_every if args.save_every is not None else cfg["defaults"].get("save_every", 5000))

    data_dir = Path(cfg["dataset"]["data_dir"])
    train_rows = read_split(data_dir, cfg["dataset"]["train_split"])
    validation_rows = read_split(data_dir, cfg["dataset"]["validation_split"])

    print("Validation-language ByT5 automation")
    print(f"languages: {', '.join(langs)}")
    print(f"mfr_mode: {args.mfr_mode}")
    print(f"dry_run: {args.dry_run}")

    for lang in langs:
        lang_cfg = cfg["languages"][lang]
        checkpoint_dir = Path(lang_cfg["output_dir"]) / "checkpoint"
        summary_path = Path(lang_cfg["validation_output_dir"]) / "summary.json"
        log_path = Path(lang_cfg["output_dir"]) / "automation.log"
        latest_checkpoint = None
        if not args.force_train and not args.no_auto_resume:
            latest_checkpoint = latest_intermediate_checkpoint(Path(lang_cfg["output_dir"]))

        train_command = append_training_control_flags(
            lang_cfg["commands"]["train"],
            save_every=save_every,
            resume_checkpoint=latest_checkpoint,
        )
        eval_command = lang_cfg["commands"]["evaluate"]

        print(f"\n== {lang} ==")
        print(f"checkpoint: {checkpoint_dir}")
        print(f"summary:    {summary_path}")
        print(f"latest intermediate: {latest_checkpoint or 'none'}")

        if args.dry_run:
            print(f"[dry-run train] {train_command}")
            print(f"[dry-run eval]  {eval_command}")
            continue

        if not args.skip_training:
            if checkpoint_exists(checkpoint_dir) and not args.force_train:
                print("training skipped: checkpoint already exists")
            else:
                run_command(train_command, log_path, capture_output=args.capture_child_output)

        if not args.skip_evaluation:
            if summary_path.exists() and not args.force_eval:
                print("evaluation skipped: summary.json already exists")
            else:
                if not checkpoint_exists(checkpoint_dir):
                    raise FileNotFoundError(f"Cannot evaluate {lang}; checkpoint missing: {checkpoint_dir}")
                run_command(eval_command, log_path, capture_output=args.capture_child_output)

        rows = make_comparison_rows(cfg, train_rows, validation_rows, langs, args.mfr_mode)
        write_comparison(rows, args.comparison_output_dir, config_path=args.config, mfr_mode=args.mfr_mode)

    rows = make_comparison_rows(cfg, train_rows, validation_rows, langs, args.mfr_mode)
    write_comparison(rows, args.comparison_output_dir, config_path=args.config, mfr_mode=args.mfr_mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        raise SystemExit(130)
