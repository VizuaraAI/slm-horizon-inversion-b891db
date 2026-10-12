#!/usr/bin/env bash
# The amplifier sweep: the main results table of the paper.
# Each model is measured under every amplifier so that the competence curve
# (per-step accuracy against token cost) can be fitted and then inverted.
#
#   bash scripts/run_sweep.sh                    # default model list
#   bash scripts/run_sweep.sh "Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct"
set -euo pipefail
cd "$(dirname "$0")/.."

MODELS="${1:-Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen2.5-1.5B-Instruct meta-llama/Llama-3.2-3B-Instruct}"
VERIFIER="${VERIFIER:-checkpoints/step-verifier}"
SEED="${SEED:-0}"

for model in $MODELS; do
  for amp in greedy vote_k3 vote_k5 vote_k9; do
    echo "== $model / $amp =="
    python src/run_eval.py --model "$model" --family synthetic \
      --horizons 2,4,8,12,16 --per-horizon 40 --amplifier "$amp" --seed "$SEED"
  done

  if [ -d "$VERIFIER" ]; then
    for amp in rerank_k5 rerank_retry; do
      echo "== $model / $amp =="
      python src/run_eval.py --model "$model" --family synthetic \
        --horizons 2,4,8,12,16 --per-horizon 40 --amplifier "$amp" \
        --verifier "$VERIFIER" --seed "$SEED"
    done
  else
    echo "no verifier at $VERIFIER, skipping rerank conditions"
    echo "train one with: python src/train_verifier.py --rollouts results/*/rollouts.jsonl"
  fi
done

python src/fit_and_invert.py --results results --target 0.8 --horizon 12 \
  --csv results/sweep_table.csv
python src/plots.py --results results --out figures
