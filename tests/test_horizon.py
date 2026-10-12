"""Tests for the horizon arithmetic and the step metrics.

These run in a second and they are the contract for the whole project: if the
inversion is wrong here, every number in the paper is wrong.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from amplifiers import majority_vote  # noqa: E402
from chain_tasks import make_synthetic_chain  # noqa: E402
from horizon import (  # noqa: E402
    MarkovChainModel,
    cheapest_budget_for,
    effective_horizon,
    fit_effective_horizon,
    horizon_at_target,
    naive_success,
    required_per_step,
)
from models import SimulatedStepper  # noqa: E402
from step_eval import Rollout, StepRecord, aggregate, extract_answer  # noqa: E402


def test_the_motivating_number():
    assert naive_success(0.9, 8) == pytest.approx(0.4305, abs=1e-4)


def test_inversion_round_trips():
    for target in (0.5, 0.8, 0.9):
        for n in (2, 8, 30):
            p = required_per_step(target, n)
            assert naive_success(p, n) == pytest.approx(target, abs=1e-9)


def test_long_horizons_demand_near_perfect_steps():
    assert required_per_step(0.8, 12) == pytest.approx(0.9816, abs=1e-4)
    assert required_per_step(0.8, 30) > required_per_step(0.8, 12)


def test_effective_horizon_detects_correlation():
    p_hat = 0.9
    worse_than_independent = naive_success(p_hat, 8) / 2
    assert effective_horizon(p_hat, worse_than_independent) > 8


def test_fit_recovers_planted_alpha():
    p_hat, alpha = 0.92, 1.4
    horizons = [2, 4, 8, 12, 16]
    observed = [p_hat ** (alpha * n) for n in horizons]
    fit = fit_effective_horizon(p_hat, horizons, observed)
    assert fit["alpha"] == pytest.approx(alpha, abs=1e-6)
    assert fit["mae"] < 1e-9


def test_horizon_at_target_is_monotone_in_accuracy():
    assert horizon_at_target(0.99, 0.8) > horizon_at_target(0.95, 0.8)
    assert horizon_at_target(0.95, 0.8) > horizon_at_target(0.90, 0.8)


def test_markov_recovery_beats_naive_product():
    p = 0.9
    no_recovery = MarkovChainModel(p=p, rho=0.0, r=0.0).success(8)
    with_recovery = MarkovChainModel(p=p, rho=0.8, r=0.6).success(8)
    assert no_recovery == pytest.approx(naive_success(p, 8), abs=1e-9)
    assert with_recovery > no_recovery


def test_cheapest_budget_prefers_the_cheap_one_that_clears():
    curve = {"greedy": 0.90, "vote_k5": 0.985, "rerank": 0.99}
    costs = {"greedy": 100.0, "vote_k5": 400.0, "rerank": 900.0}
    choice = cheapest_budget_for(curve, costs, target_success=0.8, n=12)
    assert choice["choice"] == "vote_k5"
    assert choice["required_p"] == pytest.approx(0.9816, abs=1e-4)


def test_cheapest_budget_reports_when_nothing_is_enough():
    choice = cheapest_budget_for({"greedy": 0.9}, {"greedy": 1.0}, 0.9, 30)
    assert choice["choice"] == ""


def test_synthetic_chain_has_checkable_steps():
    task = make_synthetic_chain(3, horizon=6, seed=1)
    assert task.horizon == 6
    assert all(step.gold for step in task.steps)
    prompt = task.prompt_for(2, ["11", "17"])
    assert "Result 1: 11" in prompt and "Result 2: 17" in prompt


def test_simulator_hits_its_planted_accuracy():
    stepper = SimulatedStepper(p_base=0.8, p_after_error=0.8, seed=7)
    hits = sum(stepper.step("42", saw_error=False) == "42" for _ in range(4000))
    assert 0.77 < hits / 4000 < 0.83


def test_extract_answer_and_voting():
    assert extract_answer("Result: 42.") == "42"
    assert extract_answer("the answer is -7\nnext step") == "-7"
    assert majority_vote(["12", "Result: 12", "13"]) == "12"


def test_aggregate_separates_on_track_from_post_error():
    steps = [
        StepRecord(1, "1", "1", True),
        StepRecord(2, "9", "2", False),
        StepRecord(3, "3", "3", True),
    ]
    stats = aggregate([Rollout("t", "synthetic", 3, "m", "greedy", steps)])
    assert stats.p_hat == pytest.approx(0.5)        # steps 1 and 2 were on-track
    assert stats.p_after_error == pytest.approx(1.0)  # step 3 came after the error
    assert stats.recovery_rate == pytest.approx(1.0)  # final step was correct
    assert stats.end_to_end == pytest.approx(1.0)
