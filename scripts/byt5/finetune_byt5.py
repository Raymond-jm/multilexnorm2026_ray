#!/usr/bin/env python3
"""Controlled ByT5 fine-tuning for MultiLexNorm2026.

This is the first real ByT5 training script for our project.  It keeps the
UFAL-style context-marked token format:

    input:  left context <extra_id_0> raw_token <extra_id_1> right context
    target: norm_token

The script focuses on reproducibility and visibility:

- progress bar with tqdm
- terminal log every N steps, default 10
- JSONL train log
- run_config.json
- checkpoint saving

Validation ERR is computed by scripts/byt5/evaluate_checkpoint.py after this
script finishes.  Keeping training and evaluation separate makes it easier to
rerun evaluation on different validation sizes without retraining.
"""

from __future__ import annotations

import argparse
import json
import re
import random
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, get_linear_schedule_with_warmup


@dataclass
class Example:
    """One seq2seq training example."""

    input_text: str
    target_text: str
    lang: str
    changed: bool


class JsonlSeq2SeqDataset(Dataset):
    """Load JSONL examples produced by build_byt5_examples.py."""

    def __init__(self, path: Path, max_examples: int | None, seed: int):
        examples: list[Example] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                examples.append(
                    Example(
                        input_text=item["input_text"],
                        target_text=item["target_text"],
                        lang=item["lang"],
                        changed=bool(item["changed"]),
                    )
                )

        rng = random.Random(seed)
        rng.shuffle(examples)
        if max_examples is not None:
            examples = examples[:max_examples]

        if not examples:
            raise RuntimeError(f"No training examples loaded from {path}")

        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Example:
        return self.examples[idx]


def make_collate_fn(tokenizer, max_input_length: int, max_target_length: int):
    """Tokenize a batch and mask target padding tokens for seq2seq loss."""

    def collate(batch: list[Example]) -> dict[str, torch.Tensor]:
        input_texts = [ex.input_text for ex in batch]
        target_texts = [ex.target_text for ex in batch]

        inputs = tokenizer(
            input_texts,
            padding=True,
            truncation=True,
            max_length=max_input_length,
            return_tensors="pt",
        )
        targets = tokenizer(
            target_texts,
            padding=True,
            truncation=True,
            max_length=max_target_length,
            return_tensors="pt",
        )

        labels = targets["input_ids"]
        labels[labels == tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
        }

    return collate


def moving_average(values: list[float], window: int) -> float:
    """Average the last window values."""
    recent = values[-window:]
    return sum(recent) / len(recent)


def infer_step_from_checkpoint(path: Path) -> int:
    """Infer checkpoint step from a directory named checkpoint_step_N."""
    match = re.fullmatch(r"checkpoint_step_(\d+)", path.name)
    if not match:
        return 0
    return int(match.group(1))


def save_training_state(
    path: Path,
    *,
    step: int,
    optimizer_step: int,
    optimizer: torch.optim.Optimizer,
    scheduler,
    losses: list[float],
) -> None:
    """Save optimizer/scheduler progress so interrupted runs can resume."""
    torch.save(
        {
            "step": step,
            "optimizer_step": optimizer_step,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "losses": losses,
        },
        path / "training_state.pt",
    )


def load_training_state(path: Path) -> dict[str, object]:
    """Load training state across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--examples",
        type=Path,
        default=Path("sample_data/byt5/train_examples_50k.jsonl"),
        help="Training JSONL examples produced by build_byt5_examples.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/byt5/context_marked_50k_1000steps"),
        help="Directory for logs and checkpoint.",
    )
    parser.add_argument("--model-name", default="google/byt5-small")
    parser.add_argument("--max-examples", type=int, default=50000)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-input-length", type=int, default=256)
    parser.add_argument("--max-target-length", type=int, default=32)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=5000, help="Save intermediate checkpoint every N steps. 0 disables.")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=Path,
        default=None,
        help="Resume model weights and, when available, optimizer/scheduler state from checkpoint_step_N.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow CPU training. By default this script exits if CUDA is unavailable.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config_path = args.output_dir / "run_config.json"
    train_log_path = args.output_dir / "train_log.jsonl"
    config_path.write_text(json.dumps(vars(args), ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA is not available, so training was stopped before starting. "
            "Fix the GPU/CUDA environment or pass --allow-cpu intentionally."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_source = args.resume_from_checkpoint if args.resume_from_checkpoint is not None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_source, use_safetensors=True).to(device)
    model.train()

    dataset = JsonlSeq2SeqDataset(args.examples, max_examples=args.max_examples, seed=args.seed)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate_fn(tokenizer, args.max_input_length, args.max_target_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.max_steps,
    )

    losses: list[float] = []
    step = 0
    optimizer_step = 0
    if args.resume_from_checkpoint is not None:
        state_path = args.resume_from_checkpoint / "training_state.pt"
        if state_path.exists():
            state = load_training_state(state_path)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            losses = [float(value) for value in state.get("losses", [])]
            step = int(state["step"])
            optimizer_step = int(state["optimizer_step"])
        else:
            # Older intermediate checkpoints saved only model weights.  We can
            # still continue from those weights, but optimizer/scheduler state
            # starts fresh and the step is inferred from the directory name.
            step = infer_step_from_checkpoint(args.resume_from_checkpoint)
            optimizer_step = step // args.gradient_accumulation_steps

    print("ByT5 controlled fine-tuning")
    print(f"model: {args.model_name}")
    print(f"resume_from_checkpoint: {args.resume_from_checkpoint}")
    print(f"device: {device}")
    print(f"examples: {len(dataset)}")
    print(f"batch_size: {args.batch_size}")
    print(f"gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    print(f"max_steps: {args.max_steps}")
    print(f"log_every: {args.log_every}")
    print(f"save_every: {args.save_every}")
    print(f"output_dir: {args.output_dir}")

    start_time = time.time()
    progress = tqdm(total=args.max_steps, initial=step, desc="training")

    optimizer.zero_grad(set_to_none=True)
    log_mode = "a" if args.resume_from_checkpoint is not None else "w"
    with train_log_path.open(log_mode, encoding="utf-8") as log_f:
        while step < args.max_steps:
            for batch in dataloader:
                batch = {key: value.to(device) for key, value in batch.items()}
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation_steps
                loss.backward()

                step += 1
                raw_loss = float((loss * args.gradient_accumulation_steps).detach().cpu())
                losses.append(raw_loss)

                if step % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    optimizer_step += 1

                lr = scheduler.get_last_lr()[0]
                elapsed = time.time() - start_time
                avg_loss = moving_average(losses, min(args.log_every, len(losses)))
                record = {
                    "step": step,
                    "optimizer_step": optimizer_step,
                    "loss": raw_loss,
                    "avg_loss": avg_loss,
                    "lr": lr,
                    "elapsed_sec": elapsed,
                }
                log_f.write(json.dumps(record) + "\n")
                log_f.flush()

                progress.update(1)
                progress.set_postfix(loss=f"{raw_loss:.4f}", avg=f"{avg_loss:.4f}", lr=f"{lr:.2e}")

                if step == 1 or step % args.log_every == 0:
                    tqdm.write(
                        f"step={step} loss={raw_loss:.6f} "
                        f"avg_loss_{args.log_every}={avg_loss:.6f} lr={lr:.2e} "
                        f"elapsed={elapsed/60:.1f}m"
                    )

                if args.save_every > 0 and step % args.save_every == 0:
                    checkpoint_dir = args.output_dir / f"checkpoint_step_{step}"
                    model.save_pretrained(checkpoint_dir, safe_serialization=True)
                    tokenizer.save_pretrained(checkpoint_dir)
                    save_training_state(
                        checkpoint_dir,
                        step=step,
                        optimizer_step=optimizer_step,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        losses=losses,
                    )
                    tqdm.write(f"saved intermediate checkpoint: {checkpoint_dir}")

                if step >= args.max_steps:
                    break

    progress.close()
    final_checkpoint = args.output_dir / "checkpoint"
    model.save_pretrained(final_checkpoint, safe_serialization=True)
    tokenizer.save_pretrained(final_checkpoint)
    save_training_state(
        final_checkpoint,
        step=step,
        optimizer_step=optimizer_step,
        optimizer=optimizer,
        scheduler=scheduler,
        losses=losses,
    )

    print("ByT5 fine-tuning complete")
    print(f"first_loss: {losses[0]:.6f}")
    print(f"last_loss: {losses[-1]:.6f}")
    print(f"avg_last_{args.log_every}: {moving_average(losses, min(args.log_every, len(losses))):.6f}")
    print(f"log: {train_log_path}")
    print(f"checkpoint: {final_checkpoint}")


if __name__ == "__main__":
    main()
