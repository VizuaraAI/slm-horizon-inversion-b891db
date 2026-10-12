# slm-horizon-inversion

Starter code for **Inverting the Horizon Arithmetic: what per-step competence
does a small language model need for a given task horizon?**

Eight chained components at 90 percent each gives a 43 percent end-to-end
ceiling. Everyone quotes that forward. This project solves it backwards:
given a target horizon `N` and a target end-to-end success rate `S*`, what
per-step accuracy `p*` does an SLM actually need, and what is the cheapest
way to buy it?

```
p* = (S*) ** (1 / (alpha * N))
```

`alpha` is the correction this repo measures. It is 1 when step errors behave
independently, above 1 when errors compound (conditioning on a corrupted
state makes later steps worse), and below 1 when the chain absorbs some of
its own mistakes.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

The horizon math, the synthetic tasks and the tests run on CPU with only
numpy, pyyaml and pytest installed. You need torch and transformers only when
you move to real checkpoints.

## First three things to run

```bash
# 1. One minute, no GPU, no downloads. Checks that the estimator recovers a
#    planted per-step accuracy and a planted alpha.
bash scripts/smoke_test.sh

# 2. Your first real measurement. Downloads a 1.5B checkpoint on first use.
bash scripts/run_baseline.sh Qwen/Qwen2.5-1.5B-Instruct

# 3. The inversion, which is the result the paper is built on.
python src/fit_and_invert.py --results results --target 0.8 --horizon 12
```

Step 3 prints one line per measured condition (per-step accuracy, fitted
alpha, observed end-to-end success, horizon@50, horizon@80, competence gap,
tokens per solved task) and then the prescription: the cheapest amplifier
that clears the required per-step accuracy, or an honest statement that
nothing you measured is good enough.

## Layout

```
configs/default.yaml      defaults for one evaluation cell
scripts/smoke_test.sh     CPU-only sanity check, run this first
scripts/run_baseline.sh   greedy chains for one checkpoint on all families
scripts/run_sweep.sh      the full model x amplifier grid, the paper table
src/horizon.py            the arithmetic: forward model, alpha fit, inversion
src/chain_tasks.py        step-decomposable tasks with labeled horizon N
src/models.py             SLM wrapper, plus a simulator for validating the math
src/amplifiers.py         greedy, voting, verifier reranking, flag-and-retry
src/step_eval.py          rollout records and the step-level metrics
src/run_eval.py           entry point: measure one (model, family, amplifier)
src/train_verifier.py     entry point: train the step verifier from rollouts
src/fit_and_invert.py     entry point: fit alpha, invert, print the prescription
src/plots.py              requirement curve and observed-vs-naive success curves
tests/test_horizon.py     fast tests, the contract for every number in the paper
experiments/log_template.md   one row per experiment, mapped to the paper table
```

## The measurement protocol

A rollout executes each step conditioned on **its own** previous outputs,
never on the gold ones. That is the whole point: error propagation has to be
visible, not quietly repaired between steps. From a set of rollouts the code
measures four things.

| quantity | meaning |
| --- | --- |
| `p_hat` | per-step accuracy, counted only on steps reached while still on-track |
| `p_after_error` | per-step accuracy after an upstream step was already wrong |
| `recovery_rate` | fraction of rollouts that erred somewhere and still ended correct |
| `alpha` | fitted exponent in `S_N = p_hat ** (alpha * N)` |

## Task families

- `synthetic`: offline chains of arithmetic and table lookups with `N`
  controllable from 2 to 20 and every intermediate value labeled. Start here,
  because it isolates horizon length from task difficulty.
- `gsm8k`: each annotated equation line is one step, `N` roughly 2 to 8.
- `musique`: each decomposition entry is one retrieval hop, `N` roughly 2 to 4,
  with non-numeric answers.

Nothing large is committed. Loaders download the public datasets on first use.

## Amplifiers

| preset | what it does | cost |
| --- | --- | --- |
| `greedy` | one sample per step, defines the baseline `p_hat` | 1x |
| `vote_k3` / `vote_k5` / `vote_k9` | majority vote over k samples per step | k x |
| `rerank_k5` | k candidates scored by the trained step verifier | k x plus verifier |
| `rerank_retry` | reranking, plus one hotter retry when the score is low | k to 2k x |

Train the verifier from rollouts you have already collected, so it costs no
annotation:

```bash
python src/train_verifier.py --rollouts results/*/rollouts.jsonl \
  --base-model distilbert-base-uncased --out checkpoints/step-verifier
```

## Where this is going

Every run you log fills a row of the results table in the paper draft. When
the grid is full, `fit_and_invert.py` produces the headline claim: the
per-step accuracy a given horizon demands, the gap your model has to close,
and the cheapest measured way to close it.
