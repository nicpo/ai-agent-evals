# Experiment 02: When graders agree

Companion code for the post <a href="https://nicpo.github.io/2026/08/05/graders-agreement/">"When the graders agree, they may not have seen everything"</a>: a query that passed EX and got a `CORRECT` judge verdict, yet
failed under a valid database change.

## Method

1. **Run** the agent + EX + judge suite (or reuse an existing results file).
2. **Filter to dual passes**: cases where EX matched *and* the judge's
   Branch A false-positive check said `CORRECT`. Both graders agreed, on one
   database snapshot.
3. **Probe with scenarios**: `scenarios.py` discovers legal perturbations
   directly from the live schema (`PRAGMA table_info`) -- setting a nullable
   column to NULL on a cloned row, or duplicating a free-text value across two
   rows -- and materializes each one as its own temporary copy of the
   database. `probe.py` reruns the gold SQL and the agent's saved SQL against
   every (dual-pass case x scenario) pair.
4. **Manual review**: any scenario where the two queries now disagree is a
   *candidate* counterexample, not a confirmed bug. Check that the scenario is
   a legitimate state and that the gold SQL's behavior under it is the
   intended one before concluding the agent's SQL was wrong.

Because scenarios are discovered from the schema rather than hand-authored per
question, this works unchanged against any eval set that runs against the same
database -- there is nothing question-specific to plug in.

## Run it

```bash
python -m experiments.exp_02_graders_agreement.run --profile toy
# or, to reuse a run from experiment 1 / eval.runner without new LLM calls:
python -m experiments.exp_02_graders_agreement.run --results output/results/<file>.jsonl
```

This makes agent + judge LLM calls only in the first form (`run_suite`); the
scenario probing itself is pure SQL, no API calls.

## What you get

A JSON report listing every counterexample found: the case id/question, the
scenario name and description, and the execution-accuracy detail showing how
gold and agent SQL diverged. Treat these as a manual-review queue, not a
finished finding -- see the note field in the report itself.

On the toy set there are only 4 questions and (usually) very few dual passes,
so this mostly demonstrates the *method*. Point `--profile` at a real eval set
for a real search.
