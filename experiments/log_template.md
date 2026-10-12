# Experiment log

One row per run. The columns are the columns of Table 1 in the paper, so a
filled log is a filled table. Copy the block at the bottom for each new run.

## Results table (mirrors Table 1 of the paper)

| date | run id | model | family | amplifier | N grid | p_hat (%) | p_hat^8 (%) | S_8 (%) | alpha | h@50 | h@80 | tokens / solved | seed | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | | | | | | | | | | | | | |

Fill `p_hat`, `S_8`, `alpha`, `h@50`, `h@80` and `tokens / solved` straight
from the output of `python src/fit_and_invert.py`. `p_hat^8` is the naive
independence prediction, and the gap between it and `S_8` is the whole paper.

## Inversion table (the prescription, Section 5 of the paper)

| target S* | horizon N | required p* | best measured p_hat | competence gap | cheapest amplifier that clears it | tokens / solved |
| --- | --- | --- | --- | --- | --- | --- |
| 0.80 | 8 | | | | | |
| 0.80 | 12 | | | | | |
| 0.50 | 20 | | | | | |

## Run entry

Copy one of these per run and keep it immediately below this heading.

```
### YYYY-MM-DD  run id: <results/ directory name>

Question
: what did this run exist to answer, in one sentence

Setup
: model, task family, horizons, tasks per horizon, amplifier, verifier
  checkpoint if any, seed, hardware, wall clock

Command
: the exact command line

Numbers
: p_hat, p_after_error, recovery_rate, S_N per horizon, alpha, fit MAE,
  tokens per solved task

Reading
: what it means for the inversion. Did alpha move? Did the competence gap
  shrink? Which paper section does this feed?

Surprises and doubts
: anything that looked wrong, any instrumentation you still distrust

Next
: the single next run this implies
```

## Standing checks before trusting a row

- [ ] The seed is recorded and the run directory was not overwritten.
- [ ] `p_hat` was measured on on-track steps only.
- [ ] No gold intermediate was fed back into the context.
- [ ] Token cost was recorded, not estimated after the fact.
- [ ] The same answer matcher was used as in the previous rows, or the change
      is noted in the row and the earlier rows were re-run.
- [ ] A simulated run is labeled as such and never reported as a finding.
