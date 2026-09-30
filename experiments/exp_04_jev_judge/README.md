# exp_04_jev_judge

Companion code for the post on replacing the LLM judge with Jev, a small hosted
"System One" decision model from TypeSafe.

## Method

The harness grades each agent attempt with execution accuracy (EX) and a GPT
judge (Tier 5). This experiment adds Jev as a third scorer and compares the
three on the same attempts.

The agent and its grading are the existing harness, unchanged. `run.py` calls
`eval.runner.run_suite` (agent, EX, GPT judge), then sends each attempt that has
agent SQL to Jev. The branch follows EX, like the GPT judge:

- **Branch A** (EX match): does the passing SQL still have a logical flaw that
  would give a wrong answer on another plausible database state? Jev sees the
  question, schema, approved SQL and agent SQL.
- **Branch B** (EX mismatch): is the different result nevertheless a defensible
  answer? Jev also sees both result sets.

Jev answers one multiple-choice question per attempt: `A` CORRECT, `B` INCORRECT
or `C` CANNOT_DETERMINE. `CANNOT_DETERMINE` is an abstention and does not count as
a pass. Prompts are in `prompts/monolithic.json`; the model is pinned in
`config.json` and a response from any other model ID stops the run.

Two variants, chosen with `--variant`:

- `mono` (default): one call per attempt.
- `two-call`: three calls per attempt. `mono` as above, `narrow` (four yes/no
  probability questions on dimensions of the GPT rubric, `prompts/decomposed.json`),
  and `after` (the monolithic question again, with the narrow probabilities added
  to the state). This tests whether decomposing the judgment helps.

`report.py` compares EX, the GPT judge and Jev (both variants if present):

- pass rates, abstentions, agreement with the GPT judge, Jev cost and latency
- accuracy, false accepts and false rejects against your own labels (optional)
- a **cascade**: use Jev's verdict, but send cases where Jev abstains or its
  confidence is below `--escalate-below` (default 0.8) to the GPT judge verdict
  already in the same run. This shows how much of the GPT judge's accuracy a cheap
  first pass keeps and how many cases still need the expensive judge.

## Run it

All commands run from the repo root.

1. Set up the environment and database as in the root README (`pip install -r requirements.txt`, `python setup_db.py`).

2. Add your keys to `.env`: the agent and GPT judge providers for the profile you run, plus Jev:

   ```text
   TYPESAFE_API_KEY=...
   ```

3. Look at the requests Jev will receive. This makes no API calls and needs no keys:

   ```bash
   python -m experiments.exp_04_jev_judge.run --dry-run
   python -m experiments.exp_04_jev_judge.run --dry-run --variant two-call
   ```

4. Run the agent and judge the attempts. This makes real LLM calls (agent, GPT judge and Jev):

   ```bash
   python -m experiments.exp_04_jev_judge.run --profile toy
   # or, with the decomposed variant:
   python -m experiments.exp_04_jev_judge.run --profile toy --variant two-call
   ```

   It prints the path of the results file, `output/results/exp_04_jev_<stamp>.jsonl`.
   If the Jev pass is interrupted, rerun with `--out <that file>` to resume; finished attempts are skipped.
   Failed Jev calls are written to `<file>.failures.jsonl` and the run continues.

5. Report:

   ```bash
   python -m experiments.exp_04_jev_judge.report output/results/exp_04_jev_<stamp>.jsonl
   ```

6. Optional: measure accuracy against your own verdicts. Write a label template, fill in `label`
   (`correct`, `incorrect` or `cannot_determine`) for each attempt after reading its question, SQL and
   results, then rerun the report:

   ```bash
   python -m experiments.exp_04_jev_judge.report <jsonl> --write-label-template labels.jsonl
   python -m experiments.exp_04_jev_judge.report <jsonl> --labels labels.jsonl --escalate-below 0.8
   ```

   Label without looking at the Jev or GPT verdicts. The template does not show them.

Other options:

- Judge an existing run without rerunning the agent: `run --results output/results/output_<stamp>.jsonl`
  (works on any harness results file that includes the GPT judge, for example one written by exp_01).
- Your own eval set: `--profile <name>` (see `config/experiment_profiles.yaml`).
- Tests (no network, no keys): `python -m unittest discover -s tests`.

**Privacy and cost.** Each attempt's question, schema, SQL and result rows are sent to the
hosted Jev API. The toy set is synthetic; check that your own eval data may be sent before
pointing `--profile` at it. The Jev cost in the report is input tokens at the price in
`config.json`; check it is current. The agent and GPT judge calls are billed by their providers
as in any harness run, and `two-call` makes three Jev calls per attempt.

## What you get

- `output/results/exp_04_jev_<stamp>.jsonl`: one row per judged attempt with `key`, `id`, `profile`,
  `branch`, the GPT judge verdict, the question and SQL, and the Jev calls (`mono`, and for
  `two-call` also `narrow` and `after`) with verdict, confidence, probabilities, token usage,
  latency and request ID.
- `<file>.report.md` and `<file>.report.json` from `report.py`.
- Attempts with no agent SQL are not sent to Jev. They fail under every scorer, so the pass rates
  are over judged attempts only.

## Interpreting the toy-set output

The toy set has 4 questions. That is enough to check the pipeline runs end to end and to see the
shape of the report, but the pass rates, agreement and cascade numbers are not findings. Use a real
eval set, several repeats and your own labels before drawing conclusions. The study behind the post used 61
questions, three agent versions and three runs each.
