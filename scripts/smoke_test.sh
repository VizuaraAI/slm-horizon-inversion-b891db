#!/usr/bin/env bash
# Day one check: no GPU, no downloads, about one minute.
# Plants a known per-step accuracy in the simulator and checks that the
# estimator recovers it, then fits and inverts the horizon model.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== unit tests =="
python -m pytest tests -q

echo
echo "== pure chain, one error is fatal (expect alpha close to 1.0) =="
python src/run_eval.py --model "simulated:0.92:0.0" --family synthetic \
  --horizons 2,4,8,12,16 --per-horizon 80 --amplifier greedy --seed 0

echo
echo "== self-conditioning decay (expect alpha above 1.0) =="
python src/run_eval.py --model "simulated:0.92:0.0:0.012" --family synthetic \
  --horizons 2,4,8,12,16 --per-horizon 80 --amplifier greedy --seed 0

echo
echo "== partial self-recovery (expect alpha below 1.0) =="
python src/run_eval.py --model "simulated:0.92:0.45" --family synthetic \
  --horizons 2,4,8,12,16 --per-horizon 80 --amplifier greedy --seed 0

echo
echo "== fit and invert =="
python src/fit_and_invert.py --results results --target 0.8 --horizon 12
