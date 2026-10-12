"""Rollout records and the step-level metrics that feed the horizon model.

The quantities computed here are exactly the inputs the inversion needs:

    p_hat             per-step accuracy, measured on steps that were reached
                      while the rollout was still on-track
    p_after_error     per-step accuracy after an upstream step was wrong
                      (the correlation term)
    recovery_rate     fraction of drifted rollouts that return to the gold
                      trajectory
    success_by_n      end-to-end success bucketed by horizon N
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


@dataclass
class StepRecord:
    index: int
    prediction: str
    gold: str
    correct: bool
    samples: int = 1
    verifier_score: float = float("nan")
    retried: bool = False
    prompt_tokens: int = 0
    output_tokens: int = 0


@dataclass
class Rollout:
    task_id: str
    family: str
    horizon: int
    model: str
    amplifier: str
    steps: List[StepRecord] = field(default_factory=list)

    @property
    def final_correct(self) -> bool:
        return bool(self.steps) and self.steps[-1].correct

    @property
    def all_steps_correct(self) -> bool:
        return bool(self.steps) and all(s.correct for s in self.steps)

    @property
    def first_error_index(self) -> int:
        for step in self.steps:
            if not step.correct:
                return step.index
        return -1

    @property
    def recovered(self) -> bool:
        """An error occurred, yet the final step still matched the gold answer."""
        return self.first_error_index > 0 and self.final_correct

    @property
    def total_tokens(self) -> int:
        return sum(s.prompt_tokens + s.output_tokens for s in self.steps)

    def to_json(self) -> str:
        payload = asdict(self)
        payload["final_correct"] = self.final_correct
        payload["total_tokens"] = self.total_tokens
        return json.dumps(payload)


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------


def extract_answer(text: str) -> str:
    """Pull a single step answer out of a model completion."""
    cleaned = text.strip().split("\n")[0].strip()
    cleaned = cleaned.rstrip(".").replace("$", "").replace(",", "")
    for prefix in ("result:", "answer:", "result is", "answer is"):
        low = cleaned.lower()
        if low.startswith(prefix):
            cleaned = cleaned[len(prefix) :].strip()
    numbers = _NUM_RE.findall(cleaned)
    if numbers:
        return numbers[-1]
    return cleaned


def step_correct(prediction: str, gold: str, tolerance: float = 1e-6) -> bool:
    """Numeric comparison when both sides parse as numbers, else string match."""
    pred, ref = prediction.strip(), gold.strip()
    try:
        return abs(float(pred) - float(ref)) <= tolerance
    except ValueError:
        return pred.lower() == ref.lower()


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


@dataclass
class StepStats:
    n_rollouts: int
    n_steps: int
    p_hat: float
    p_after_error: float
    recovery_rate: float
    end_to_end: float
    success_by_n: Dict[int, float]
    tokens_per_task: float
    tokens_per_solved_task: float

    def as_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        payload["success_by_n"] = {str(k): v for k, v in self.success_by_n.items()}
        return payload


def aggregate(rollouts: Sequence[Rollout]) -> StepStats:
    if not rollouts:
        raise ValueError("no rollouts to aggregate")

    on_track_total = on_track_correct = 0
    post_error_total = post_error_correct = 0
    drifted = recovered = 0
    by_n_total: Dict[int, int] = defaultdict(int)
    by_n_hit: Dict[int, int] = defaultdict(int)
    tokens = 0
    solved = 0

    for rollout in rollouts:
        saw_error = False
        for step in rollout.steps:
            if saw_error:
                post_error_total += 1
                post_error_correct += int(step.correct)
            else:
                on_track_total += 1
                on_track_correct += int(step.correct)
            if not step.correct:
                saw_error = True
        if saw_error:
            drifted += 1
            recovered += int(rollout.final_correct)
        by_n_total[rollout.horizon] += 1
        by_n_hit[rollout.horizon] += int(rollout.final_correct)
        tokens += rollout.total_tokens
        solved += int(rollout.final_correct)

    n_rollouts = len(rollouts)
    return StepStats(
        n_rollouts=n_rollouts,
        n_steps=on_track_total + post_error_total,
        p_hat=_safe_div(on_track_correct, on_track_total),
        p_after_error=_safe_div(post_error_correct, post_error_total),
        recovery_rate=_safe_div(recovered, drifted),
        end_to_end=_safe_div(solved, n_rollouts),
        success_by_n={n: _safe_div(by_n_hit[n], by_n_total[n]) for n in sorted(by_n_total)},
        tokens_per_task=_safe_div(tokens, n_rollouts),
        tokens_per_solved_task=_safe_div(tokens, solved),
    )


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else float("nan")


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def write_rollouts(rollouts: Sequence[Rollout], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for rollout in rollouts:
            handle.write(rollout.to_json() + "\n")


def read_rollouts(path: Path) -> List[Rollout]:
    rollouts: List[Rollout] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            payload.pop("final_correct", None)
            payload.pop("total_tokens", None)
            steps = [StepRecord(**step) for step in payload.pop("steps", [])]
            rollouts.append(Rollout(steps=steps, **payload))
    return rollouts
