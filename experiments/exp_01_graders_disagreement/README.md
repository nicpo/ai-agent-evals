# exp_01_graders_disagreement

Companion code for the post "Lessons from the mismatch between your
deterministic grader and the LLM judge" (EX vs. LLM judge -- 38% vs. 66%).

## Method

Grading a text-to-SQL agent with both execution accuracy (EX) and an LLM judge
produces two scores that rarely match. The instinct is to average them into one
adjusted number. Instead, look at all four cells of the disagreement matrix:

```
                   EX pass              EX fail
    Judge accepts  expected_success     valid_alternative
    Judge rejects  latent_bug           expected_failure
```

The diagonal is expected. The off-diagonal cells are where the eval teaches you
something:

- **latent_bug** (EX pass, judge reject): EX matched on this database, but the
  judge thinks the SQL logic would fail on data it hasn't seen. Look for a
  missing counterexample in your eval data.
- **valid_alternative** (EX fail, judge accept): EX's strict comparator
  rejected an answer the judge considers acceptable. Look at comparator
  policy, the gold SQL, or the output contract -- not the agent.

This experiment doesn't decide which grader is "right". It surfaces the
disagreement cases so you can read them.

## Run it

Uses the harness's existing `eval.runner` (agent + Tier 2b EX + Tier 5 judge)
and defaults to the toy eval set in `data/evals/toy/`:

```bash
python -m experiments.exp_01_graders_disagreement.run --profile toy
python -m experiments.exp_01_graders_disagreement.analyze output/results/<the-file-just-written>.jsonl
```

Add `--markdown` to `analyze.py` to also print the off-diagonal cases as short
case cards (question / gold SQL / agent SQL / verdict / reason).

To run against your own eval set instead of the toy set, add a profile to
`config/experiment_profiles.yaml` pointing at your `questions-*.jsonl` files
and pass `--profile <name>` to `run.py`. No eval-set content is embedded in
this experiment's code -- everything comes from whatever profile you pass.

## What you get

`analyze.py` writes a `<results>.disagreement.json` report with the matrix
counts/rates and, for the two off-diagonal cells, the full case list (id,
question, gold SQL, agent SQL, EX reason, judge verdict, judge reason) so you
can inspect them directly instead of re-deriving them from the raw results
file.

On the 4-question toy set this is a demonstration of the *method*, not a
finding -- there's nowhere near enough data for the matrix to be meaningful.
Point it at a real eval set to get a real signal.
