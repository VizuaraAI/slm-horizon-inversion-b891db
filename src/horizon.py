"""Horizon arithmetic: forward prediction, effective horizon, and inversion.

This is the mathematical core of the project. Everything else in the repo
exists to feed these functions with measured numbers.

Three models of a chain of N steps:

1. Naive independence:      S_N = p ** N
2. Effective horizon:       S_N = p ** N_eff, where N_eff is fitted from data
3. Three-state Markov:      states are on-track, drift, failed, with a
                            recovery probability out of drift

The inversion is the point of the project: given a target end-to-end success
S* and a horizon N, what per-step accuracy p* is required?
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------
# Forward models
# ---------------------------------------------------------------------------


def naive_success(p: float, n: int) -> float:
    """End-to-end success under independent step failures: p ** n."""
    _check_prob(p, "p")
    if n < 0:
        raise ValueError("n must be non-negative")
    return float(p) ** int(n)


@dataclass(frozen=True)
class MarkovChainModel:
    """Three-state model of an agent rollout.

    p    : probability a step is executed correctly while on-track
    rho  : fraction of step errors that are recoverable (drift) rather than fatal
    r    : probability of returning to on-track from drift on the next step

    State order is (on-track, drift, failed). A rollout succeeds if it is
    on-track after the final step.
    """

    p: float
    rho: float = 0.5
    r: float = 0.3

    def transition_matrix(self) -> np.ndarray:
        p, rho, r = self.p, self.rho, self.r
        for name, value in (("p", p), ("rho", rho), ("r", r)):
            _check_prob(value, name)
        return np.array(
            [
                [p, (1.0 - p) * rho, (1.0 - p) * (1.0 - rho)],
                [r, (1.0 - r) * p, (1.0 - r) * (1.0 - p)],
                [0.0, 0.0, 1.0],
            ]
        )

    def success(self, n: int) -> float:
        """Probability of being on-track after n steps."""
        state = np.array([1.0, 0.0, 0.0])
        matrix = self.transition_matrix()
        for _ in range(int(n)):
            state = state @ matrix
        return float(state[0])

    def success_curve(self, horizons: Iterable[int]) -> np.ndarray:
        return np.array([self.success(n) for n in horizons])


# ---------------------------------------------------------------------------
# Effective horizon
# ---------------------------------------------------------------------------


def effective_horizon(p_hat: float, s_observed: float) -> float:
    """Exponent N_eff such that p_hat ** N_eff equals the observed success.

    N_eff > N means errors are correlated and compound faster than the naive
    product predicts. N_eff < N means the agent recovers from some of its own
    mistakes, so the chain is effectively shorter than it looks.
    """
    _check_prob(p_hat, "p_hat", strict_upper=True)
    _check_prob(s_observed, "s_observed", strict_lower=True)
    return float(np.log(s_observed) / np.log(p_hat))


def fit_effective_horizon(
    p_hat: float, horizons: Sequence[int], observed: Sequence[float]
) -> Dict[str, float]:
    """Least-squares fit of S_N = p_hat ** (alpha * N) over several horizons.

    alpha is the compounding slope: alpha = 1 recovers the naive product,
    alpha > 1 means super-multiplicative decay. Returns alpha, the implied
    N_eff at each horizon through alpha * N, and the fit residual.
    """
    _check_prob(p_hat, "p_hat", strict_upper=True)
    horizons_arr = np.asarray(horizons, dtype=float)
    observed_arr = np.asarray(observed, dtype=float)
    if horizons_arr.shape != observed_arr.shape:
        raise ValueError("horizons and observed must have the same length")
    mask = (observed_arr > 0.0) & (horizons_arr > 0.0)
    if mask.sum() < 1:
        raise ValueError("need at least one horizon with non-zero success")
    log_s = np.log(observed_arr[mask])
    log_p = np.log(p_hat)
    design = (horizons_arr[mask] * log_p).reshape(-1, 1)
    alpha, residuals, _, _ = np.linalg.lstsq(design, log_s, rcond=None)
    alpha_value = float(alpha[0])
    predicted = p_hat ** (alpha_value * horizons_arr)
    return {
        "alpha": alpha_value,
        "mae": float(np.mean(np.abs(predicted - observed_arr))),
        "sse_log": float(residuals[0]) if residuals.size else 0.0,
    }


# ---------------------------------------------------------------------------
# The inversion
# ---------------------------------------------------------------------------


def required_per_step(target_success: float, n: int, alpha: float = 1.0) -> float:
    """Per-step accuracy needed for `target_success` over `n` steps.

    With alpha = 1 this is the clean inverse of the naive product,
    p* = S* ** (1 / n). With a fitted alpha it becomes p* = S* ** (1 / (alpha * n)),
    which is the number the student should actually design against.
    """
    _check_prob(target_success, "target_success", strict_lower=True)
    if n <= 0:
        raise ValueError("n must be positive")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    return float(target_success ** (1.0 / (alpha * n)))


def competence_gap(p_measured: float, target_success: float, n: int, alpha: float = 1.0) -> float:
    """How far a measured per-step accuracy is from the requirement."""
    return required_per_step(target_success, n, alpha) - float(p_measured)


def horizon_at_target(
    p: float, target_success: float, alpha: float = 1.0, max_n: int = 512
) -> int:
    """Longest horizon still reaching `target_success` at per-step accuracy p.

    This is the headline metric of the paper (horizon@50, horizon@80).
    """
    _check_prob(p, "p")
    _check_prob(target_success, "target_success")
    best = 0
    for n in range(1, int(max_n) + 1):
        if p ** (alpha * n) >= target_success:
            best = n
        else:
            break
    return best


def cheapest_budget_for(
    competence_curve: Dict[str, float],
    budget_cost: Dict[str, float],
    target_success: float,
    n: int,
    alpha: float = 1.0,
) -> Dict[str, float]:
    """Pick the cheapest amplifier setting that clears the required accuracy.

    competence_curve maps an amplifier label (for example "vote_k5") to the
    per-step accuracy it delivers; budget_cost maps the same labels to token
    cost per task. Returns the chosen label plus the requirement it had to
    clear, or an empty choice if nothing in the curve is good enough.
    """
    needed = required_per_step(target_success, n, alpha)
    feasible = {k: v for k, v in competence_curve.items() if v >= needed}
    if not feasible:
        return {"choice": "", "required_p": needed, "achieved_p": 0.0, "cost": float("inf")}
    choice = min(feasible, key=lambda k: budget_cost.get(k, float("inf")))
    return {
        "choice": choice,
        "required_p": needed,
        "achieved_p": float(competence_curve[choice]),
        "cost": float(budget_cost.get(choice, float("nan"))),
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_prob(
    value: float, name: str, strict_lower: bool = False, strict_upper: bool = False
) -> None:
    value = float(value)
    low_ok = value > 0.0 if strict_lower else value >= 0.0
    high_ok = value < 1.0 if strict_upper else value <= 1.0
    if not (low_ok and high_ok):
        raise ValueError(f"{name} must be a probability in range, got {value}")


if __name__ == "__main__":
    print("naive 8 steps at 0.90:", round(naive_success(0.90, 8), 4))
    for n in (4, 8, 16, 32):
        print(
            f"horizon {n:>3}: p* for S*=0.85 is {required_per_step(0.85, n):.4f}",
            f"| p* for S*=0.50 is {required_per_step(0.50, n):.4f}",
        )
    print("horizon@80 at p=0.95:", horizon_at_target(0.95, 0.80))
