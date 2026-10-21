# Experiment 05: Self-improvement loop

A bounded reflection loop improves the agent's text configuration (system prompt and tool descriptions) and
checks the result on questions and graders the loop never touched.

## Question

Does a loop that selects dev failures, critiques them, proposes one prompt or tool-description edit,
evaluates it and keeps or rejects it produce a candidate that beats the unchanged baseline on held-out
questions and under an independent grader? How much of the in-loop gain is overfitting or grader gaming?

## What the study found

The study behind the post ran on the 61-question Roman Empire eval set plus 24 fresh held-out questions. Neither
set ships in this repo (`data/evals/` is git-ignored apart from the toy set), so the numbers below are context, not
something the toy set reproduces. All figures are percent of attempts judged correct. Test: 24 questions x 5
repeats. Regression: the 61 questions x 3 repeats (the loop trained and selected on them, so they are not held
out). Judges: Sol (`gpt-6.1-sol`, in the loop) and Kimi K2 Thinking (independent, chosen by calibration).

| Prompt | Test, Sol | Test, Kimi | Regression, Sol | Regression, Kimi |
|---|---|---|---|---|
| Original | 56.7 | 55.8 | 68.3 | 62.8 |
| Hand-fixed (loop start) | 64.2 | 59.2 | 74.3 | 69.4 |
| Loop winner | 81.7 | 75.8 | 92.3 | 79.8 |

- The gain of the loop winner over the hand-fixed prompt survives on fresh questions and under the independent grader:
  +17.5 points under Sol and +16.7 under Kimi on test (95% bootstrap CIs over question families exclude zero).
- The hand-written fix alone is worth +3 to +7 points, with CIs that include zero.
- Control: the hand-fixed prompt re-run unchanged scored 80.0% on dev (76.7% before) and 69.9% on validation (76.3% before). The
  keep rule would have rejected it, but the swing is as large as the +5 point bar. Treat one accepted edit as weak
  evidence.
- Power: 8 test families can only detect gains of roughly 15 points or more.

In the study, the hand-fixed prompt is the original plus one hand-written rule (the harness scores the last
`run_query`, so the last one must be the query that answers), and it was the loop's starting point.
This repo ships none of the three prompts as candidates: the loop starts from your profile's agent prompt and its output is yours to generate.

## Method

| Role | Model | What it does |
|---|---|---|
| Agent under test | Haiku 5.5 | Answers questions by writing SQL with `run_query` and `get_sample_rows`. Its prompt and tool descriptions are what gets improved. |
| Judge (in the loop, frozen) | Sol (`gpt-6.1-sol`) | Grades each attempt pass or fail: SQL results are compared with the reference query (EX), then the judge decides whether a differing query still answers the question. Its written reasoning is the feedback the proposer reads. |
| Proposer | Sonnet 5.5, medium effort | Reads a few failed dev attempts plus the judge's reasoning, writes a diagnosis, proposes one edit. |
| Keep rule | hard-coded, no LLM | Accepts an edit only if measured scores clear fixed thresholds. |
| Independent grader | any other-family model you choose | Not used in the loop. After it, re-grades the final arms' saved attempts to check the gains do not exist only under Sol. |

One iteration, up to 10, or until the $20 cap:

1. Take the incumbent (the first is the baseline, `c000_baseline`) and a few of its failed dev attempts.
2. The proposer writes a diagnosis and one edited candidate. A lint rejects edits that smuggle in answers: question
   ids, quoted database values, copied question text (8+ words), gold SQL (40+ characters), mentions of grading, or
   text over 3,500 characters. A lint failure gets one repair attempt; if it still fails the proposal is spent.
3. The candidate runs on dev and validation, 3 repeats each. Sol grades every attempt.
4. The keep rule compares with the incumbent. **Keep** if validation is at least +5 points, dev is no worse than
   -5 points, at most one question that passed 3 of 3 for the incumbent now fails 3 of 3, and the lint passes.

Data:

- **Dev**: the proposer sees these failures (question, tool calls, SQL, judge verdict and reasoning, never gold SQL).
- **Validation**: decides keep or reject. The proposer never sees its questions, traces or verdicts, only aggregate
  score changes of earlier proposals.
- **Dev and validation partition the profile's questions** and double as a regression suite: they shaped earlier prompts,
  so they say something about the loop's mechanics, not about generalization.
- **Test**: a separate profile of fresh questions, grouped in families of related variants (`family_id`), hashed
  when you run `loop init --test-profile ...`. Nothing in the loop reads it. At the end it compares `c000_baseline`
  (the profile's unchanged agent prompt, `agent/system_prompt_v1.md` for the shipped profiles) with the loop's winner.

Pre-registered outcomes: (a) the gain survives on test and under the independent grader, (b) the dev/validation gain
vanishes on test (overfitting), (c) the gain exists under Sol but not under the independent grader or EX (grader
gaming), (d) nothing is accepted.

## Run it

All commands run from the repo root. Everything is written to `output/exp_05/<profile>/` (git-ignored). Every
command takes `--profile` (default `toy`); the held-out set is `--test-profile` (the toy one is `toy_test`).

1. Set up the environment and database as in the root README (`pip install -r requirements.txt`, `python setup_db.py`).
   Put `ANTHROPIC_API_KEY` (agent, proposer) and `OPENAI_API_KEY` (Sol judge) in `.env`.

2. Offline tests, no keys or network:

   ```bash
   python -m unittest tests.test_exp05_loop
   ```

3. Split the profile's questions into dev and validation (deterministic, seed 20261015):

   ```bash
   python -m experiments.exp_05_self_improvement_loop.make_splits --profile toy
   ```

   Splits are fixed once the loop starts. For a real set, first run the baseline on all questions (3 repeats) and
   pass the records with `--attempts`, which balances the two sides by how often the baseline fails each question:

   ```bash
   python -m experiments.exp_05_self_improvement_loop.evaluate run --profile <profile> --candidate c000_baseline --split regression --repeats 3 --run-name baseline_probe
   python -m experiments.exp_05_self_improvement_loop.make_splits --profile <profile> --attempts output/exp_05/<profile>/runs/baseline_probe.jsonl
   ```

4. Dry run (a few cents): stub proposer, 2 questions per split, 1 repeat, separate sandbox folder.

   ```bash
   python -m experiments.exp_05_self_improvement_loop.loop run --dry-run
   ```

5. Record the run inputs, lock the test set and score the baseline (`c000_baseline`, 3 repeats on dev and validation):

   ```bash
   python -m experiments.exp_05_self_improvement_loop.loop init --test-profile toy_test
   ```

   This writes `manifest.json` (hashes of the questions, judge prompts, splits and test set). `loop run` refuses to
   continue if those inputs change, and the test set cannot be evaluated unless its hash matches. Lock it before the
   loop, not after.

6. Run the loop. It is resumable (rerun the same command), and `status` shows progress.

   ```bash
   python -m experiments.exp_05_self_improvement_loop.loop run --max-proposals 10 --cap 20
   python -m experiments.exp_05_self_improvement_loop.loop status
   ```

   Proposals are saved under `output/exp_05/<profile>/candidates/`, attempts under `loop/`, decisions in `loop_log.jsonl`.
   To check a candidate against the lint without spending anything: `loop lint NAME`.

7. Control: re-run the unchanged baseline and apply the keep rule to the pair. Any "keep" is a false accept caused by
   noise alone, so this shows how large the rule's noise is.

   ```bash
   python -m experiments.exp_05_self_improvement_loop.evaluate run --candidate c000_baseline --split dev --repeats 3 --run-name control_dev
   python -m experiments.exp_05_self_improvement_loop.evaluate run --candidate c000_baseline --split validation --repeats 3 --run-name control_validation
   python -m experiments.exp_05_self_improvement_loop.compare_control
   ```

8. Final arms (`c000_baseline` and the loop winner; needs at least one kept candidate) on the held-out test set (5 repeats) and the
   regression suite (3 repeats), then the report with 95% bootstrap CIs and paired difference:

   ```bash
   python -m experiments.exp_05_self_improvement_loop.final_eval plan   --test-profile toy_test
   python -m experiments.exp_05_self_improvement_loop.final_eval run    --test-profile toy_test
   python -m experiments.exp_05_self_improvement_loop.final_eval report --test-profile toy_test
   ```

9. Independent re-grade. Hand-label a sample of dev or validation attempts (from `loop/` or `final/`, never test
   attempts) without looking at any verdict, then calibrate a judge from another model family against your labels.
   Labels are JSONL, one line per attempt:

   ```json
   {"candidate": "c000_baseline", "split": "validation", "id": "toy_easy_001", "repeat": 1, "label": "correct"}
   ```

   ```bash
   pip install -r experiments/exp_05_self_improvement_loop/requirements.txt   # adds langchain-aws, for Bedrock models only
   python -m experiments.exp_05_self_improvement_loop.independent_grader calibrate --model bedrock_converse:moonshot.kimi-k2-thinking --labels labels.jsonl
   python -m experiments.exp_05_self_improvement_loop.independent_grader regrade   --model bedrock_converse:moonshot.kimi-k2-thinking --test-profile toy_test
   python -m experiments.exp_05_self_improvement_loop.independent_grader report    --model bedrock_converse:moonshot.kimi-k2-thinking --test-profile toy_test
   ```

   For Bedrock models (Kimi, Qwen, DeepSeek, GLM) add credentials to `.env`: either `AWS_BEARER_TOKEN_BEDROCK` (a Bedrock
   API key) or `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`, plus optionally `AWS_REGION` (default `us-east-1`).
   The model must be enabled for your account in that region, and model ids differ by region; use the id Bedrock shows
   for it, such as `moonshot.kimi-k2-thinking`. A model from another provider needs its own package and key instead.

   The judge may grade only if it agrees with your labels on at least 90% of the attempts where it gave a verdict;
   `regrade` refuses a model below that. If none reaches the bar, report that there is no independent grader rather
   than lowering the bar. The study labeled 196 attempts and tried five models; the best reached 94.3%. Pick a
   family unlike the in-loop judge (Sol and GPT-5.4 are both OpenAI models and may share blind spots).

### Reading the toy-set output

The toy profile has 4 questions (2 dev, 2 validation) and `toy_test` has 4 (2 families). That is enough to see the
pipeline run end to end and the shape of every report. It is not enough to conclude anything: one extra pass on
validation moves the score by more than the +5 point keep bar, and the baseline may pass every dev question, in which
case the loop stops with "no failed dev attempts left". Use your own eval set, with a held-out profile of at least a
few families of related questions, before drawing conclusions. The held-out profile needs `gold_sql`, `gold_result`
and a `family_id` per question (see `data/evals/toy/test/questions-test.jsonl`), and an entry in
`config/experiment_profiles.yaml` whose `eval_glob` finds it.

### Cost

The study spent about $25 in total: the loop about $11.6 (10 proposals at about $0.97 of evaluation plus $0.04 of
proposer calls each, the $0.93 baseline, and $0.49 for nine proposals later voided), the control $0.9, the final arms
$4.5, calibrating five candidate graders about $3, and the independent re-grade $5.4. Per attempt, the Sol judge
(about $0.003 to $0.004) costs roughly twice the Haiku agent ($0.0013 to $0.0019), so about $0.005 all-in. The toy runs cost cents. Spend is capped by `--cap`, which counts the baseline, proposer and
evaluation calls at the list prices in `models.py` (check they are current; a model without a price stops the run).
The independent grader's cost is not tracked: token usage is stored in every record.

## Files

- [`workspace.py`](workspace.py): which questions each split holds and where files go; test-set lock check.
- [`make_splits.py`](make_splits.py): dev/validation split.
- [`models.py`](models.py), [`candidates.py`](candidates.py), [`agent_loop.py`](agent_loop.py), [`evaluate.py`](evaluate.py): model entries and prices, the candidate format (the baseline is the profile's agent prompt; proposals are saved in the workspace), the agent run with candidate tool descriptions, and the evaluator (agent, EX, Sol judge, success rule).
- [`loop.py`](loop.py), [`proposer.py`](proposer.py), [`lint.py`](lint.py), [`packet.py`](packet.py), [`prompts/`](prompts/): the loop, the keep rule, the proposer prompts, the lint and the failure packets.
- [`compare_control.py`](compare_control.py), [`final_eval.py`](final_eval.py), [`independent_grader.py`](independent_grader.py): control, final arms with bootstrap CIs, independent grader.
- Tests: [`tests/test_exp05_loop.py`](../../tests/test_exp05_loop.py).

The agent, graders and eval code of the harness are used unchanged (`agent/`, `graders/`, `eval/`, `experiment_profiles.py`). The judge prompts
are the profile's judge bundle (`config/judge_prompts/default.yaml`), and EX compares against each question's `gold_sql`.

## Differences from the original study

The original code ran inside a private copy of the harness. This port keeps the method and changes what the
public repo does not have:

- The 61-question set, the 24-question test set, its drafting scripts and the human labels live elsewhere. Questions
  come from a profile, the held-out set is a second profile, and calibration uses labels you write.
- Gold SQL corrections that the study applied to the 61 questions are not part of this repo: gold is the `gold_sql` in your files.
- The study started from the hand-fixed prompt (the original plus one hand-written rule) and reused earlier runs of it as the baseline records. This port starts from the profile's agent prompt and `loop init` runs the baseline, so the first accepted edit may well be the same "last query answers" rule, and the final comparison has two arms (baseline and winner) instead of three.
- The original split balanced the 61 questions by those runs; `make_splits` does the same when you pass `--attempts`.
- The independent grader is generic (any LangChain model) instead of a Bedrock table, and the calibration extras (a format
  reminder for models that omit the verdict line, retrying unresolved cases, reparsing) were dropped.
- The study's dated amendment log (a length-cap fix after the first loop run, a stricter completeness check before keep
  decisions, parser and calibration notes) is not reproduced; this code is the final behavior.
