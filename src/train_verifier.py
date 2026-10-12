"""Train the step verifier that powers reranking and flag-and-retry.

Labels come from rollouts you already ran: every recorded step is a
(prompt context, candidate answer, correct or not) triple. No human
annotation, which is what makes this affordable inside an eight-week project.

Example:
    python src/train_verifier.py \\
        --rollouts results/simulated-0.92-0.55__synthetic__greedy__seed0/rollouts.jsonl \\
        --base-model distilbert-base-uncased --out checkpoints/step-verifier
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_eval import read_rollouts  # noqa: E402


def build_dataset(rollout_paths: List[Path]) -> List[Dict[str, object]]:
    """One example per recorded step, balanced by downsampling the majority class."""
    positives: List[Dict[str, object]] = []
    negatives: List[Dict[str, object]] = []
    for path in rollout_paths:
        for rollout in read_rollouts(path):
            for step in rollout.steps:
                example = {
                    "context": f"[{rollout.family}] step {step.index} of {rollout.horizon}",
                    "candidate": step.prediction,
                    "label": int(step.correct),
                }
                (positives if step.correct else negatives).append(example)
    size = min(len(positives), len(negatives))
    if size == 0:
        raise ValueError(
            "need both correct and incorrect steps; run a harder setting or more tasks"
        )
    return positives[:size] + negatives[:size]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train a step verifier from rollout logs.")
    parser.add_argument("--rollouts", nargs="+", required=True)
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--out", default="checkpoints/step-verifier")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max-length", type=int, default=256)
    args = parser.parse_args(argv)

    import numpy as np
    from datasets import Dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    examples = build_dataset([Path(p) for p in args.rollouts])
    print(f"{len(examples)} balanced examples from {len(args.rollouts)} rollout files")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.base_model, num_labels=2)

    def tokenize(batch):
        return tokenizer(
            batch["context"],
            batch["candidate"],
            truncation=True,
            max_length=args.max_length,
        )

    dataset = Dataset.from_list(examples).shuffle(seed=0)
    splits = dataset.train_test_split(test_size=0.2, seed=0)
    tokenized = splits.map(tokenize, batched=True)

    def metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return {"accuracy": float((predictions == labels).mean())}

    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=args.out,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            learning_rate=args.lr,
            eval_strategy="epoch",
            save_strategy="no",
            logging_steps=25,
            report_to=[],
        ),
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        processing_class=tokenizer,
        compute_metrics=metrics,
    )
    trainer.train()
    print(trainer.evaluate())

    Path(args.out).mkdir(parents=True, exist_ok=True)
    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"saved verifier to {args.out}")
    print("use it with: python src/run_eval.py --amplifier rerank_k5 --verifier " + args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
