"""Entry point: run one (model, task family, amplifier) cell of the grid.

Example:
    python src/run_eval.py --model simulated:0.92:0.55 --family synthetic \\
        --horizons 2,4,8,12 --per-horizon 40 --amplifier greedy

Writes two files into results/<run-name>/:
    rollouts.jsonl   one line per rollout, every step recorded
    summary.json     the aggregate statistics the horizon fit consumes
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from amplifiers import TrainedStepVerifier, get_amplifier, run_rollout  # noqa: E402
from chain_tasks import get_tasks  # noqa: E402
from models import load_model  # noqa: E402
from step_eval import aggregate, write_rollouts  # noqa: E402
from utils import load_config, progress, run_name, set_seed  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure per-step competence of one model.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default=None, help="hub id, or simulated:p_base:p_after")
    parser.add_argument("--family", default=None, choices=["synthetic", "gsm8k", "musique"])
    parser.add_argument("--amplifier", default=None)
    parser.add_argument("--horizons", default=None, help="comma separated, synthetic family only")
    parser.add_argument("--per-horizon", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="cap on tasks for real datasets")
    parser.add_argument("--verifier", default=None, help="path to a trained step verifier")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config, args)
    set_seed(cfg["seed"])

    model = load_model(cfg["model"])
    amplifier = get_amplifier(cfg["amplifier"])
    verifier = None
    if cfg.get("verifier"):
        verifier = TrainedStepVerifier(cfg["verifier"], device=cfg.get("device", "cpu"))

    if cfg["family"] == "synthetic":
        tasks = get_tasks(
            "synthetic",
            horizons=cfg["horizons"],
            per_horizon=cfg["per_horizon"],
            seed=cfg["seed"],
        )
    else:
        tasks = get_tasks(cfg["family"], limit=cfg["limit"])

    print(f"model={cfg['model']} family={cfg['family']} amplifier={cfg['amplifier']} "
          f"tasks={len(tasks)}")

    rollouts = []
    for task in progress(tasks, desc="rollouts"):
        rollouts.append(run_rollout(model, task, amplifier, verifier=verifier))

    stats = aggregate(rollouts)
    out_dir = Path(cfg["out_dir"]) / run_name(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_rollouts(rollouts, out_dir / "rollouts.jsonl")
    summary = {
        "model": cfg["model"],
        "family": cfg["family"],
        "amplifier": cfg["amplifier"],
        "seed": cfg["seed"],
        **stats.as_dict(),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nper-step accuracy      p_hat = {stats.p_hat:.4f}")
    print(f"post-error accuracy          = {stats.p_after_error:.4f}")
    print(f"recovery rate                = {stats.recovery_rate:.4f}")
    print(f"end-to-end success           = {stats.end_to_end:.4f}")
    print("success by horizon:")
    for horizon, value in stats.success_by_n.items():
        naive = stats.p_hat ** horizon
        print(f"  N = {horizon:>3}: observed {value:.3f}   naive p_hat^N {naive:.3f}")
    print(f"\nwrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
