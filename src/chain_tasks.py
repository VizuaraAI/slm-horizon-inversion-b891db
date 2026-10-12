"""Step-decomposable tasks with a known horizon N and checkable intermediates.

Every task exposes the same interface so that the amplifiers and the metrics
code never need to know which family they are looking at:

    task.task_id        unique string
    task.horizon        number of steps N
    task.steps          list of Step, each with a prompt and a gold answer
    task.final_answer   gold answer of the last step

Two families ship offline (synthetic chains, which let you dial N from 2 to 20)
and two download public datasets on first use (GSM8K and MuSiQue). Start with
the synthetic family while you debug, then move to the real data.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Step:
    index: int
    instruction: str
    gold: str
    kind: str = "compute"


@dataclass
class ChainTask:
    task_id: str
    family: str
    preamble: str
    steps: List[Step] = field(default_factory=list)
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def horizon(self) -> int:
        return len(self.steps)

    @property
    def final_answer(self) -> str:
        return self.steps[-1].gold if self.steps else ""

    def prompt_for(self, index: int, history: List[str]) -> str:
        """Prompt for step `index`, conditioned on what the agent produced before.

        History carries the agent's own outputs, not the gold ones. That is
        deliberate: error propagation is the phenomenon under study, so we never
        silently repair the context between steps.
        """
        lines = [self.preamble, ""]
        for prev, value in zip(self.steps[:index], history):
            lines.append(f"Step {prev.index}: {prev.instruction}")
            lines.append(f"Result {prev.index}: {value}")
        step = self.steps[index]
        lines.append(f"Step {step.index}: {step.instruction}")
        lines.append("Answer with the result of this step only, no explanation.")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Family 1: synthetic chains (offline, N controllable from 2 to 20)
# ---------------------------------------------------------------------------

_LOOKUP_TABLE = {
    "alpha": 3,
    "beta": 7,
    "gamma": 11,
    "delta": 13,
    "epsilon": 17,
}


def _op_add(value: int, rng: random.Random):
    k = rng.randint(2, 19)
    return value + k, f"add {k} to the previous result"


def _op_multiply(value: int, rng: random.Random):
    k = rng.randint(2, 7)
    return value * k, f"multiply the previous result by {k}"


def _op_lookup(value: int, rng: random.Random):
    key = rng.choice(sorted(_LOOKUP_TABLE))
    return value + _LOOKUP_TABLE[key], (
        f"add the table value of '{key}' to the previous result "
        f"(table: {_format_table()})"
    )


def _op_digit_sum(value: int, rng: random.Random):
    total = sum(int(ch) for ch in str(abs(value)))
    return total, "replace the previous result by the sum of its digits"


def _op_modulo(value: int, rng: random.Random):
    k = rng.choice([5, 7, 9, 10])
    return value % k, f"take the previous result modulo {k}"


_OPS: List[Callable[[int, random.Random], tuple]] = [
    _op_add,
    _op_multiply,
    _op_lookup,
    _op_digit_sum,
    _op_modulo,
]


def _format_table() -> str:
    return ", ".join(f"{k}={v}" for k, v in sorted(_LOOKUP_TABLE.items()))


def make_synthetic_chain(task_index: int, horizon: int, seed: int = 0) -> ChainTask:
    """Build one deterministic chain of `horizon` checkable arithmetic steps."""
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    rng = random.Random((seed * 100003) + task_index)
    start = rng.randint(4, 40)
    value = start
    steps: List[Step] = []
    for i in range(1, horizon + 1):
        op = rng.choice(_OPS)
        value, instruction = op(value, rng)
        steps.append(Step(index=i, instruction=instruction, gold=str(value)))
    return ChainTask(
        task_id=f"synthetic-n{horizon}-{task_index:04d}",
        family="synthetic",
        preamble=(
            f"You are executing a chain of {horizon} arithmetic steps. "
            f"The starting value, before step 1, is {start}."
        ),
        steps=steps,
        meta={"horizon": str(horizon), "seed": str(seed), "start": str(start)},
    )


def load_synthetic(
    horizons: Optional[List[int]] = None, per_horizon: int = 25, seed: int = 0
) -> List[ChainTask]:
    horizons = horizons or [2, 4, 8, 12, 16]
    tasks: List[ChainTask] = []
    for horizon in horizons:
        for i in range(per_horizon):
            tasks.append(make_synthetic_chain(i, horizon, seed=seed))
    return tasks


# ---------------------------------------------------------------------------
# Family 2: GSM8K, one step per annotated equation line
# ---------------------------------------------------------------------------

_CALC_RE = re.compile(r"<<[^>]*>>")
_FINAL_RE = re.compile(r"####\s*(.+)$")


def load_gsm8k(split: str = "test", limit: int = 200) -> List[ChainTask]:
    """Load GSM8K and split each reference solution into its equation lines.

    Requires `datasets`. The horizon of a GSM8K item is the number of
    intermediate equation lines, which is typically 2 to 8.
    """
    from datasets import load_dataset  # imported lazily so the module stays light

    raw = load_dataset("openai/gsm8k", "main", split=split)
    tasks: List[ChainTask] = []
    for i, row in enumerate(raw):
        if len(tasks) >= limit:
            break
        solution = row["answer"]
        final_match = _FINAL_RE.search(solution)
        if not final_match:
            continue
        body = solution.split("####")[0].strip().split("\n")
        steps: List[Step] = []
        for j, line in enumerate(body, start=1):
            clean = _CALC_RE.sub("", line).strip()
            gold = _last_number(clean)
            if gold is None:
                continue
            steps.append(
                Step(index=len(steps) + 1, instruction=clean, gold=gold, kind="reasoning")
            )
        if not steps:
            continue
        steps.append(
            Step(
                index=len(steps) + 1,
                instruction="state the final numeric answer to the question",
                gold=final_match.group(1).strip().replace(",", ""),
                kind="final",
            )
        )
        tasks.append(
            ChainTask(
                task_id=f"gsm8k-{split}-{i:05d}",
                family="gsm8k",
                preamble=f"Question: {row['question']}",
                steps=steps,
                meta={"horizon": str(len(steps))},
            )
        )
    return tasks


def _last_number(text: str) -> Optional[str]:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    return numbers[-1] if numbers else None


# ---------------------------------------------------------------------------
# Family 3: MuSiQue multi-hop question answering (2 to 4 retrieval hops)
# ---------------------------------------------------------------------------


def load_musique(split: str = "validation", limit: int = 150) -> List[ChainTask]:
    """Load MuSiQue, where each decomposition entry is one retrieval hop."""
    from datasets import load_dataset

    raw = load_dataset("dgslibisey/MuSiQue", split=split)
    tasks: List[ChainTask] = []
    for i, row in enumerate(raw):
        if len(tasks) >= limit:
            break
        hops = row.get("question_decomposition") or []
        if not hops:
            continue
        steps = [
            Step(
                index=j,
                instruction=str(hop.get("question", "")).strip(),
                gold=str(hop.get("answer", "")).strip(),
                kind="retrieval",
            )
            for j, hop in enumerate(hops, start=1)
            if hop.get("question") and hop.get("answer")
        ]
        if len(steps) < 2:
            continue
        tasks.append(
            ChainTask(
                task_id=f"musique-{split}-{i:05d}",
                family="musique",
                preamble=f"Multi-hop question: {row['question']}",
                steps=steps,
                meta={"horizon": str(len(steps))},
            )
        )
    return tasks


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TASK_FAMILIES: Dict[str, Callable[..., List[ChainTask]]] = {
    "synthetic": load_synthetic,
    "gsm8k": load_gsm8k,
    "musique": load_musique,
}


def get_tasks(family: str, **kwargs) -> List[ChainTask]:
    if family not in TASK_FAMILIES:
        raise KeyError(f"unknown family '{family}', choose from {sorted(TASK_FAMILIES)}")
    return TASK_FAMILIES[family](**kwargs)


if __name__ == "__main__":
    sample = make_synthetic_chain(0, horizon=4, seed=1)
    print(sample.task_id, "horizon", sample.horizon)
    print(sample.prompt_for(2, ["12", "19"]))
    print("gold:", [s.gold for s in sample.steps])
