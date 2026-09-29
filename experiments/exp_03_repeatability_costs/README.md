# exp_03_repeatability_costs

Companion code for "Measuring agent reliability with repeated evals" and
"Fewer agent steps do not mean proportionally lower cost".

## Method

A single run per question is one sample, not the agent's accuracy. This
experiment runs every question `--repeats` times (optionally across several
agent profiles) and reports three independent things:

**Reliability** (`reliability.py`) -- mean success (pass@1), `pass^k` (right
every time), and `pass@k` (right at least once), computed per question and
resampled at the question level, not the attempt level.

**Variance** (`variance.py`) -- two separate signals, because an agent's score
comes from two moving parts (the agent writes SQL, the judge grades it):
whether the agent produced the same SQL on every repeat, and whether the judge
graded identical SQL the same way every time it recurred naturally across
repeats.

**Cost** (`pricing.py`, `costs.py`) -- LLM calls, input/output tokens, and
dollar cost per attempt and per 1,000 attempts, agent and judge tracked
separately, plus cost-of-pass (cost / success rate). "Cost" is reported as
several numbers on purpose -- calls, tokens, and dollars can each move by a
different amount for the same change.

## Run it

```bash
# Cheap smoke test on the toy set (2 questions x 2 repeats):
python -m experiments.exp_03_repeatability_costs.run --profiles toy --repeats 2 --limit 2
python -m experiments.exp_03_repeatability_costs.report output/results/exp_03_repeats_<stamp>.jsonl

# Compare agent versions on a real eval set (add profiles to
# config/experiment_profiles.yaml pointing at your own questions-*.jsonl):
python -m experiments.exp_03_repeatability_costs.run --profiles v1 v2 v3 --repeats 3
```

Requires provider API keys in `.env` for `run.py` (real agent + judge calls,
`repeats` times per question per profile -- costs scale linearly with
`repeats`, so start small). `report.py` itself makes no API calls.

`run.py` writes the combined attempts JSONL plus a `<file>.judge_usage.json`
sidecar with the judge's per-call token usage (captured via a small
`CostTrackingJudge` subclass local to this experiment, not a change to the
shared judge). `report.py` reads both.

## Interpreting the toy-set output

With only 2-4 questions and a couple of repeats, `pass^k`/`pass@k` and the
variance/cost numbers won't be statistically meaningful -- this is enough to
confirm the pipeline runs end to end and to see the shape of the report. Fill
in `pricing.py` with current list prices for whichever models you use, and
point `--profiles` at a real eval set (and real profiles) for numbers worth
reporting.
