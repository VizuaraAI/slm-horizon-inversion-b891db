"""Plots for the paper: the inversion curve and the measured success curves.

    python src/plots.py --results results --out figures/

Figure 1 of the paper shows the required per-step accuracy against horizon.
This script draws the same curve with your measured per-step accuracies
overlaid, so the competence gap in your write-up is a real measurement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from fit_and_invert import fit_one, load_summaries  # noqa: E402


def plot_requirement_curve(rows, out_path: Path, targets=(0.5, 0.8, 0.9)) -> None:
    horizons = np.arange(2, 33)
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for target in targets:
        ax.plot(horizons, target ** (1.0 / horizons), label=f"required for S* = {target}")
    for row in rows:
        if 0.0 < row["p_hat"] < 1.0:
            ax.axhline(row["p_hat"], linestyle="--", linewidth=0.8, alpha=0.6)
            ax.text(
                32,
                row["p_hat"],
                f"{row['model'][:18]} / {row['amplifier']}",
                fontsize=6,
                ha="right",
                va="bottom",
            )
    ax.set_xlabel("task horizon N (steps)")
    ax.set_ylabel("per-step accuracy")
    ax.set_title("Required per-step accuracy against measured accuracy")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_success_curves(summaries, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    for summary in summaries:
        by_n = {int(k): v for k, v in summary["success_by_n"].items()}
        if not by_n:
            continue
        ns = sorted(by_n)
        label = f"{summary['model'][:16]} / {summary['amplifier']}"
        ax.plot(ns, [by_n[n] for n in ns], marker="o", markersize=3, label=label)
        p_hat = summary["p_hat"]
        if 0.0 < p_hat < 1.0:
            ax.plot(ns, [p_hat ** n for n in ns], linestyle=":", linewidth=0.9, color="grey")
    ax.set_xlabel("task horizon N (steps)")
    ax.set_ylabel("end-to-end success")
    ax.set_title("Observed success against the naive p^N prediction (dotted)")
    ax.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Draw the project figures.")
    parser.add_argument("--results", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args(argv)

    summaries = load_summaries(Path(args.results))
    if not summaries:
        print(f"no results under {args.results}; run src/run_eval.py first")
        return 1
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [fit_one(s) for s in summaries]
    plot_requirement_curve(rows, out_dir / "requirement_curve.png")
    plot_success_curves(summaries, out_dir / "success_curves.png")
    print(f"wrote {out_dir}/requirement_curve.png and {out_dir}/success_curves.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
