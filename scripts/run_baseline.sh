#!/usr/bin/env bash
# Baseline measurement for one real checkpoint: greedy chains on every family.
# This is the row of the paper table that everything else is compared against.
#
#   bash scripts/run_baseline.sh Qwen/Qwen2.5-1.5B-Instruct
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-1.5B-Instruct}"
SEED="${2:-0}"

echo "== synthetic chains, horizons 2 to 16 =="
python src/run_eval.py --model "$MODEL" --family synthetic \
  --horizons 2,4,8,12,16 --per-horizon 40 --amplifier greedy --seed "$SEED"

echo "== GSM8K, natural language reasoning steps =="
python src/run_eval.py --model "$MODEL" --family gsm8k \
  --limit 200 --amplifier greedy --seed "$SEED"

echo "== MuSiQue, multi-hop retrieval steps =="
python src/run_eval.py --model "$MODEL" --family musique \
  --limit 150 --amplifier greedy --seed "$SEED"

python src/fit_and_invert.py --results results --target 0.8 --horizon 12 \
  --csv results/baseline_table.csv
