"""Entry point: fit the horizon model across runs and print the prescription.

Reads every results/*/summary.json produced by run_eval.py, fits the
compounding slope alpha per (model, family, amplifier), and then inverts:
for a target end-to-end success and horizon it reports the required per-step
accuracy, the competence gap, and the cheapest amplifier that clears it.

Example:
    python src/fit_and_invert.py --results results --target 0.8 --horizon 12
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))

from horizon import (  # noqa: E402
    cheapest_budget_for,
    competence_gap,
    fit_effective_horizon,
    horizon_at_target,
    required_per_step,
)
from utils import fmt_pct  # noqa: E402


def load_summaries(results_dir: Path) -> List[Dict]:
    summaries = []
    for path in sorted(results_dir.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_path"] = str(path.parent)
        summaries.append(payload)
    return summaries


def fit_one(summary: Dict) -> Dict:
    """Fit alpha for a single run from its success-by-horizon curve."""
    by_n = {int(k): v for k, v in summary["success_by_n"].items()}
    horizons = [n for n, s in by_n.items() if s > 0.0]
    observed = [by_n[n] for n in horizons]
    p_hat = summary["p_hat"]
    result = {"alpha": float("nan"), "mae": float("nan")}
    if horizons and 0.0 < p_hat < 1.0:
        result = fit_effective_horizon(p_hat, horizons, observed)
    return {
        "model": summary["model"],
        "family": summary["family"],
        "amplifier": summary["amplifier"],
        "p_hat": p_hat,
        "p_after_error": summary["p_after_error"],
        "recovery_rate": summary["recovery_rate"],
        "end_to_end": summary["end_to_end"],
        "tokens_per_solved": summary["tokens_per_solved_task"],
        "alpha": result["alpha"],
        "fit_mae": result["mae"],
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Fit the horizon model and invert it.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--target", type=float, default=0.8, help="target end-to-end success")
    parser.add_argument("--horizon", type=int, default=12, help="target horizon N")
    parser.add_argument("--csv", default=None, help="optional path to write the table as CSV")
    args = parser.parse_args(argv)

    summaries = load_summaries(Path(args.results))
    if not summaries:
        print(f"no summary.json found under {args.results}; run src/run_eval.py first")
        return 1

    rows = [fit_one(s) for s in summaries]

    header = (
        f"{'model':<28}{'family':<11}{'amplifier':<14}{'p_hat':>7}"
        f"{'alpha':>7}{'S_obs':>8}{'h@50':>6}{'h@80':>6}{'gap':>8}{'tok/win':>9}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        alpha = row["alpha"]
        usable = 0.0 < row["p_hat"] < 1.0 and alpha == alpha and alpha > 0
        h50 = horizon_at_target(row["p_hat"], 0.5, alpha) if usable else 0
        h80 = horizon_at_target(row["p_hat"], 0.8, alpha) if usable else 0
        gap = competence_gap(row["p_hat"], args.target, args.horizon, alpha) if usable else float("nan")
        print(
            f"{row['model'][:27]:<28}{row['family']:<11}{row['amplifier']:<14}"
            f"{fmt_pct(row['p_hat']):>7}{alpha:>7.2f}{fmt_pct(row['end_to_end']):>8}"
            f"{h50:>6}{h80:>6}{fmt_pct(gap):>8}{row['tokens_per_solved']:>9.0f}"
        )
        row["h50"], row["h80"], row["gap"] = h50, h80, gap

    print()
    needed = required_per_step(args.target, args.horizon)
    print(
        f"Naive inversion: to hit S* = {args.target} at N = {args.horizon} "
        f"you need p* = {needed:.4f} per step."
    )

    by_model: Dict[str, Dict[str, float]] = {}
    costs: Dict[str, Dict[str, float]] = {}
    for row in rows:
        by_model.setdefault(row["model"], {})[row["amplifier"]] = row["p_hat"]
        costs.setdefault(row["model"], {})[row["amplifier"]] = row["tokens_per_solved"]

    print("\nCheapest amplifier that clears the requirement, per model:")
    for model, curve in by_model.items():
        alphas = [r["alpha"] for r in rows if r["model"] == model and r["alpha"] == r["alpha"]]
        alpha = sum(alphas) / len(alphas) if alphas else 1.0
        choice = cheapest_budget_for(curve, costs[model], args.target, args.horizon, alpha)
        if choice["choice"]:
            print(
                f"  {model:<28} {choice['choice']:<14} "
                f"required p* = {choice['required_p']:.4f}, "
                f"achieved {choice['achieved_p']:.4f}, "
                f"{choice['cost']:.0f} tokens per solved task"
            )
        else:
            print(
                f"  {model:<28} nothing measured reaches p* = {choice['required_p']:.4f}; "
                f"shorten the horizon, or amplify harder"
            )

    if args.csv:
        import csv

        fields = list(rows[0].keys())
        with Path(args.csv).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nwrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
