# Instructions for the coding agent

You are working with Subramanian on an eight-week research project. The goal
is a workshop paper, not a product. Optimise for measurements that can be
defended, not for code that looks impressive.

## The research question in one line

Chained reliability is quoted forward (eight steps at 90 percent gives 43
percent end to end). This project inverts it: for a target horizon `N` and a
target end-to-end success `S*`, what per-step accuracy `p*` must a small
language model reach, and what is the cheapest way to buy it?

The claim lives in one formula, `p* = (S*) ** (1 / (alpha * N))`, where
`alpha` is fitted from measured success curves. Everything in this repo
either measures an input to that formula or validates its output.

## Project layout

- `src/horizon.py` is the mathematical core. Treat it as load bearing. Any
  change here needs a test in `tests/test_horizon.py` first.
- `src/chain_tasks.py` builds tasks whose horizon `N` is known and whose
  intermediate steps are checkable.
- `src/amplifiers.py` holds the rollout loop and the per-step amplifiers.
- `src/step_eval.py` turns rollouts into the four measured quantities.
- `src/run_eval.py`, `src/train_verifier.py`, `src/fit_and_invert.py` are the
  three entry points. Keep them thin.
- `experiments/log_template.md` mirrors the results table in the paper. Every
  run that produces a number goes in there the same day.

## Conventions

- Python 3.10 or newer, standard library plus the packages in
  `requirements.txt`. No new heavy dependency without a reason written down.
- Modules import flat (`from horizon import ...`), with `sys.path` set by the
  entry points. Do not convert this into an installed package mid-project.
- Heavy imports (torch, transformers, datasets) stay inside functions so the
  math and the tests keep running on a laptop.
- Every experiment writes `rollouts.jsonl` plus `summary.json` into a
  directory named by `utils.run_name`. Never overwrite a result by hand.
- Seeds are explicit and recorded in the summary. A number without a seed is
  not a result.
- No em dashes in code comments or prose written for this project.

## Rules that protect the science

1. **Never feed gold intermediates back into the context.** A rollout is
   conditioned on the model's own outputs. If you "fix" the state between
   steps, the measured `alpha` becomes meaningless.
2. **Measure `p_hat` only on on-track steps.** Post-error steps are a
   separate statistic (`p_after_error`). Mixing them hides the correlation
   effect this project exists to measure.
3. **Price everything in tokens.** An amplifier that raises accuracy is not
   interesting until the cost per solved task is next to it.
4. **The simulator is a test instrument, not a result.** Use it to check that
   the estimator recovers a planted `alpha`. Never report a simulated number
   as a finding.
5. **The oracle verifier is an upper bound only.** It is there to tell you
   whether reranking has headroom before you spend a week training one.

## Current state

The pipeline runs end to end on the simulator and on the synthetic family.
`scripts/smoke_test.sh` passes: the estimator recovers alpha near 1.0 for a
pure chain, above 1.0 with self-conditioning decay, and below 1.0 with
partial recovery.

## Next steps, in order

1. Run `scripts/run_baseline.sh` with Qwen2.5-1.5B-Instruct and get real
   `p_hat` and `alpha` numbers on the synthetic family. Expect prompt
   formatting to be the first problem; fix it in `ChainTask.prompt_for`.
2. Extend the answer matching in `step_eval.step_correct` for MuSiQue, where
   answers are entity strings rather than numbers. Keep the matcher strict
   and report what it rejects.
3. Collect rollouts, train the step verifier, and check it against the oracle
   verifier for headroom.
4. Fill the model by amplifier grid with `scripts/run_sweep.sh`, then run the
   inversion and write the numbers into the paper table.
5. Hold out one task family entirely, then test whether the fitted `alpha`
   predicts its success curve. That held-out test is the strongest claim the
   paper can make, so protect the held-out family from tuning.

## When adding code

- Write the test before the feature when it touches the horizon math.
- Prefer one clear function over a configurable one with five flags.
- If a result surprises you, suspect the instrumentation before the model,
  and check the rollout JSONL by hand.
