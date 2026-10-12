"""Per-step competence amplifiers and the rollout loop.

Every amplifier has the same job: raise the probability that one step is
correct, at a measurable token cost. The rollout loop is deliberately strict.
After a step is produced it is appended to the history as-is, right or wrong,
so that the next step is conditioned on the agent's own state. That is what
makes error propagation visible instead of hidden.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, List, Optional, Protocol

from chain_tasks import ChainTask
from models import GenerationConfig, SimulatedStepper
from step_eval import Rollout, StepRecord, extract_answer, step_correct


class Verifier(Protocol):
    """Scores a candidate step answer in [0, 1]. Higher is better."""

    def score(self, prompt: str, candidate: str) -> float: ...


@dataclass
class AmplifierConfig:
    name: str = "greedy"
    k: int = 1
    temperature: float = 0.0
    retry_threshold: float = 0.0
    max_retries: int = 0
    retry_temperature: float = 0.8
    max_new_tokens: int = 48

    @property
    def uses_verifier(self) -> bool:
        return self.name in {"rerank", "rerank_retry", "retry"}


PRESETS = {
    "greedy": AmplifierConfig(name="greedy", k=1, temperature=0.0),
    "vote_k3": AmplifierConfig(name="vote", k=3, temperature=0.7),
    "vote_k5": AmplifierConfig(name="vote", k=5, temperature=0.7),
    "vote_k9": AmplifierConfig(name="vote", k=9, temperature=0.7),
    "rerank_k5": AmplifierConfig(name="rerank", k=5, temperature=0.7),
    "rerank_retry": AmplifierConfig(
        name="rerank_retry", k=5, temperature=0.7, retry_threshold=0.5, max_retries=1
    ),
}


def get_amplifier(name: str) -> AmplifierConfig:
    if name not in PRESETS:
        raise KeyError(f"unknown amplifier '{name}', choose from {sorted(PRESETS)}")
    return PRESETS[name]


# ---------------------------------------------------------------------------
# Step-level selection
# ---------------------------------------------------------------------------


def majority_vote(candidates: List[str]) -> str:
    """Most common normalised answer, ties broken by first occurrence."""
    normalised = [extract_answer(c) for c in candidates]
    counts = Counter(normalised)
    best_count = max(counts.values())
    for value in normalised:
        if counts[value] == best_count:
            return value
    return normalised[0]


def verifier_pick(prompt: str, candidates: List[str], verifier: Verifier) -> tuple:
    """Highest scoring candidate and its score."""
    scored = [(verifier.score(prompt, extract_answer(c)), extract_answer(c)) for c in candidates]
    score, choice = max(scored, key=lambda pair: pair[0])
    return choice, score


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------


def run_rollout(
    model,
    task: ChainTask,
    cfg: AmplifierConfig,
    verifier: Optional[Verifier] = None,
) -> Rollout:
    """Execute one task end to end under one amplifier setting."""
    if cfg.uses_verifier and verifier is None:
        raise ValueError(f"amplifier '{cfg.name}' needs a verifier")

    rollout = Rollout(
        task_id=task.task_id,
        family=task.family,
        horizon=task.horizon,
        model=getattr(model, "model_name", "unknown"),
        amplifier=cfg.name if cfg.k == 1 else f"{cfg.name}_k{cfg.k}",
    )
    if hasattr(model, "reset"):
        model.reset()
    history: List[str] = []
    saw_error = False

    for index, step in enumerate(task.steps):
        prompt = task.prompt_for(index, history)
        record = _execute_step(model, prompt, step.gold, cfg, verifier, saw_error)
        record.index = step.index
        rollout.steps.append(record)
        history.append(record.prediction)
        saw_error = saw_error or not record.correct

    return rollout


def _execute_step(
    model,
    prompt: str,
    gold: str,
    cfg: AmplifierConfig,
    verifier: Optional[Verifier],
    saw_error: bool,
) -> StepRecord:
    if isinstance(model, SimulatedStepper):
        prediction = model.step(gold, saw_error)
        return StepRecord(
            index=0,
            prediction=prediction,
            gold=gold,
            correct=step_correct(prediction, gold),
            samples=cfg.k,
            prompt_tokens=len(prompt.split()),
            output_tokens=cfg.k * 4,
        )

    gen = GenerationConfig(
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        num_samples=cfg.k,
    )
    candidates = model.generate(prompt, gen)
    output_tokens = sum(model.token_count(c) for c in candidates)
    score = float("nan")
    retried = False

    if cfg.name == "greedy" or cfg.k == 1:
        prediction = extract_answer(candidates[0])
    elif cfg.name == "vote":
        prediction = majority_vote(candidates)
    else:
        prediction, score = verifier_pick(prompt, candidates, verifier)

    retries = cfg.max_retries if verifier is not None else 0
    for _ in range(retries):
        if score == score and score >= cfg.retry_threshold:  # score is not NaN
            break
        retry_gen = GenerationConfig(
            max_new_tokens=cfg.max_new_tokens,
            temperature=cfg.retry_temperature,
            num_samples=cfg.k,
        )
        retry_candidates = model.generate(prompt, retry_gen)
        output_tokens += sum(model.token_count(c) for c in retry_candidates)
        retry_prediction, retry_score = verifier_pick(prompt, retry_candidates, verifier)
        retried = True
        if retry_score > score:
            prediction, score = retry_prediction, retry_score

    return StepRecord(
        index=0,
        prediction=prediction,
        gold=gold,
        correct=step_correct(prediction, gold),
        samples=cfg.k,
        verifier_score=score,
        retried=retried,
        prompt_tokens=model.token_count(prompt),
        output_tokens=output_tokens,
    )


# ---------------------------------------------------------------------------
# Verifiers
# ---------------------------------------------------------------------------


class TrainedStepVerifier:
    """Sequence classifier that scores (prompt, candidate step answer) pairs.

    Train one with src/train_verifier.py, which collects its labels from
    rollouts you have already run, so no human annotation is needed.
    """

    def __init__(self, checkpoint: str, device: str = "cpu", max_length: int = 512) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
        self.model.to(device).eval()
        self.device = device
        self.max_length = max_length
        self._torch = torch

    def score(self, prompt: str, candidate: str) -> float:
        inputs = self.tokenizer(
            prompt,
            candidate,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)
        with self._torch.no_grad():
            logits = self.model(**inputs).logits
        return float(self._torch.softmax(logits, dim=-1)[0, 1])


@dataclass
class OracleVerifier:
    """Scores against the gold answer. Upper bound only, never a result.

    Use it to measure the headroom of reranking: if an oracle verifier does
    not lift per-step accuracy much, a trained one never will.
    """

    gold_lookup: Callable[[str], str]

    def score(self, prompt: str, candidate: str) -> float:
        return 1.0 if step_correct(candidate, self.gold_lookup(prompt)) else 0.0
