# AI Agent Evals: Reliability, Graders and Cost

An experimental evaluation harness for a text-to-SQL agent.

I use a small agent and fictional Roman Empire database to investigate how agent evaluation behaves in practice:

- [When do deterministic and LLM graders disagree?](experiments/exp_01_graders_disagreement/)
- [Can both graders agree on an incorrect answer?](experiments/exp_02_graders_agreement/)
- [How much does measured performance vary across repeated runs?](experiments/exp_03_repeatability_costs/)
- [How do agent architecture changes affect accuracy and inference cost?](experiments/exp_03_repeatability_costs/)
- [Can Jev, a decision model, replace the LLM judge?](experiments/exp_04_jev_judge/)
- [Can an agent improve its own prompt without gaming its eval?](experiments/exp_05_self_improvement_loop/)

[Setup](#setup) · [Run the harness](#run) · [Experiments and blog](https://nicpo.github.io/)

## Project summary

```text
                         Build eval
                             │
                             ▼
             ┌── deterministic graders
Agent ───────┤
             └── LLM judge
                             │
                ┌────────────┴────────────┐
                ▼                         ▼
           DISAGREE                    AGREE
                │                         │
          Why? What does            Are they both
          each measure?             still wrong?
                │                         │
                └────────────┬────────────┘
                             ▼
                       REPEATABILITY
                   Do results persist?
                             │
                             ▼
                           COST
                 Is the improvement worth it?
```

## Agent overview

The agent is a small text-to-SQL assistant. You give it a free-form question about the Roman Empire such as:

> “Which province collected the most tribute?”

It then tries to answer by writing SQL, running that SQL against the SQLite database, and turning the returned rows into a natural-language answer.

The workflow is:

```text
User question
  → inspect schema or sample data if needed
  → write SQL
  → run SQL
  → optionally fix a failed query
  → explain the result in English
```

Its tools are:

- `get_schema`: returns tables, columns, types and relationships.
- `get_sample_rows`: returns example records so the agent can discover actual values, such as permitted status names.
- `run_query`: executes a read-only SQL query and returns columns, rows, row count, or an error.

## Setup

### 1. Set up the environment
```bash
# create venv
python -m venv .venv

# activate it. PowerShell:
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# install packages
pip install -r requirements.txt
```

### 2. Add API keys
API keys live in a git-ignored `.env` file at the repo root. Copy the template
and fill in the providers you use:

```bash
# PowerShell/macOS/Linux
cp .env.example .env
```

Then edit the new `.env` file in the repository root (next to `config.py`).
Add the key(s) for the model provider you use:

```text
# For --model gpt-6-1-sol
OPENAI_API_KEY=sk-...

# For --model haiku-5-5 or sonnet-5-5
ANTHROPIC_API_KEY=sk-ant-...

# Only for experiments/exp_04_jev_judge (hosted Jev judge)
TYPESAFE_API_KEY=...
```
Experiment [`exp_05_self_improvement_loop`](experiments/exp_05_self_improvement_loop/) needs both `ANTHROPIC_API_KEY` (agent, proposer) and `OPENAI_API_KEY` (judge). Its optional independent grader can use any LangChain provider (for example `pip install langchain-aws` and AWS credentials for Bedrock).

The repository supports **OpenAI** and **Anthropic** out of the box: those are
the provider packages installed in `requirements.txt` and the model entries in
`config.py`. To use another LangChain provider, install its integration package,
add its standard API key to `.env`, and add a `provider:model` entry to
`MODELS` in `config.py`.

`config.py` loads `.env` automatically at import. A missing `.env` is a no-op,
so deterministic graders run without any keys.

### 3. Create DB

Build the SQLite DB `data/roman_empire.db` from the CSV seed files in `data/tables/`:

```bash
python setup_db.py
```

The script drops and recreates the database on each run, so if you make any changes to the database, back it up first.


## Run

```bash
# Verify deterministic graders only (no API key needed):
python -m eval.verify_graders

# Run the full suite (requires the agent/judge provider API keys in the environment):
python -m eval.runner
python -m eval.runner --profile v1
python -m eval.runner --profile v2 --no-judge
python -m eval.runner --profile v3 --difficulty easy --limit 5
python -m eval.runner --difficulty easy --limit 5
python -m eval.runner --no-judge
```

Results are written to `output/results/` (per-question traces and grades) and `output/summaries/` (aggregate metrics). Both are git-ignored.

Experiment profiles live in `config/experiment_profiles.yaml`. Each profile
selects repository-local agent and judge prompts, tools, models and harness
settings; results record the resolved profile plus prompt hashes.

### Prompt versioning

Agent prompts are files (`agent/system_prompt_v1.md`, `_v2`, `_v3`), and a profile points at one by path. Results
record the profile and the SHA-256 of each prompt used, so you can tell later which prompt produced which score.

To try a prompt change, don't edit a prompt file that earlier results were produced with. Otherwise the file no longer
matches those results, and a score from before the edit and a score from after it look like the same setup. Instead:

1. Copy the prompt: `cp agent/system_prompt_v3.md agent/system_prompt_v4.md`, then edit the copy.
2. Add a profile to `config/experiment_profiles.yaml` that points at it (copy an existing profile and change its
   name and `agent.prompt`):

   ```yaml
   v4:
     agent:
       model: haiku-5-5
       prompt: agent/system_prompt_v4.md
       tools: [run_query, get_sample_rows]
     judge:
       enabled: true
       model: gpt-6-1-sol
       prompts: config/judge_prompts/default.yaml
     harness:
       eval_glob: questions-*.jsonl
       max_tool_calls: 12
   ```

3. Run it: `python -m eval.runner --profile v4`.

Judge prompts work the same way: copy `config/judge_prompts/default.yaml` to a new file and point the profile's
`judge.prompts` at it.

## Toy eval set

`data/evals/toy/` contains 4 example cases (2 easy, 1 medium, 1 hard), split by difficulty into `questions-easy.jsonl`, `questions-medium.jsonl`, and `questions-hard.jsonl` - the same naming convention the canonical set uses. **This is a toy set only**. It exists so you can run the harness end-to-end without generating anything first, and so you can see the eval case structure (question, gold SQL, gold result and the grader metadata fields) in real files. It lives in a subdirectory so the default `questions-*.jsonl` glob (rooted at `data/evals/`) does not pick it up; point the runner at it explicitly:

```bash
python -m eval.runner --profile toy
```

The `toy` profile (in `config/experiment_profiles.yaml`) is identical to `v1` except it points `harness.eval_glob` at `toy/questions-*.jsonl` instead of the default `questions-*.jsonl`.

`data/evals/toy/test/` holds 4 more toy cases (2 families, with a `family_id` field) as a held-out set for [experiment 05](experiments/exp_05_self_improvement_loop/); the `toy_test` profile points at them.

For actual experiments, do not rely on this toy set. Create your own eval set (see below) sized and reviewed for the behaviors you want to measure.

## Generate gold eval set

The repository intentionally does not ship a canonical gold eval set. Use the
generator to create a **candidate** set from the local database, then review
every question, SQL query and returned result before promoting a case to the
canonical `data/evals/questions-*.jsonl` format.

Run `python setup_db.py` first so `data/roman_empire.db` exists.

The generator makes one LLM call. It sends the authored schema and five sample
rows from each table to the selected provider, requests natural-language
questions plus SQLite `SELECT` queries, executes each query using the local
read-only database, and writes only executable cases marked
`"review_status": "needs_review"`.

```bash
# Requires OPENAI_API_KEY in .env
python -m eval.generate_evals --model gpt-6-1-sol --count 10 \
  --out data/evals/candidates/roman_empire_candidates.jsonl

# Or use Anthropic, with ANTHROPIC_API_KEY in .env
python -m eval.generate_evals --model opus-5-5 --count 10 \
  --out data/evals/candidates/roman_empire_candidates.jsonl
```

After the questions were generated, **review them (very important! Gold quality = eval trustworthiness)**:

1. Does the question have one clear interpretation?
2. Does the SQL answer that question exactly, without unrequested columns or
   filters?
3. Are the saved `gold_result` rows the result you expect from the database?
4. Is the difficulty and adversarial label appropriate?
5. Does the case cover a useful behavior rather than duplicate another case?

Once you've approved, copy or edit a candidate into
`data/evals/questions-*.jsonl`. Generated candidates and the canonical eval
directory are git-ignored, so explicitly version an approved benchmark
elsewhere if you need scores to be reproducible across clones.

## Experiments

The code for experiments is in `experiments/`. Method details and options for each experiment is in each experiment's own README.

The results are described as a [series of blog posts](https://nicpo.github.io/).

### How to run experiments

[`experiments/README.md`](experiments/README.md) describes how to run each experiment. Out of the box, each experiment runs against the toy eval set (but you really should [create and review](#generate-gold-eval-set) an actual eval set).

### Writing your own experiment

An experiment is a small script on top of the harness. Here is the pattern used by `exp_01`'s `run.py`:

```python
from eval.runner import load_questions, run_suite
from experiment_profiles import load_profile

profile = load_profile("my_profile")           # defined in config/experiment_profiles.yaml
cases = load_questions(glob=profile.eval_glob)
results_path = run_suite(cases, profile=profile, run_judge=True)   # writes output/results/*.jsonl
```

Define what varies (prompt, tools, model, eval set) as a profile, run it through `run_suite`, then analyze the
results JSONL in your own script. Copy an `experiments/exp_0N_*/` folder as a starting point and keep its README
with the question, method and run commands.

## Changing models

The agent and the judge talk to LLMs through `agent/llm.py`, which wraps
LangChain's `init_chat_model`. Models live in the `MODELS` registry in
`config.py`; pick the agent and judge by registry key:

```python
# config.py
MODELS = {
    "haiku-5-5":  {
        "id": "anthropic:claude-haiku-5-5",
        "kwargs": {"max_tokens": 4096}
      },
    "sonnet-5-5": {
        "id": "anthropic:claude-sonnet-5-5",
        "kwargs": {"max_tokens": 4096},
    },
    "gpt-6-1-sol": {
        "id": "openai:gpt-6.1-sol",
        "kwargs": {"reasoning_effort": "medium"},
    },
    # ...
}
AGENT_MODEL = "haiku-5-5"      # the agent under test
JUDGE_MODEL = "gpt-6-1-sol"    # the LLM judge
```

The judge defaults to a different model family than the agent to avoid self-preference bias.

## Judge calibration (test-retest)

Judge calibration reruns the judge several times on cases where EX passed. It reports self-consistency: the average share of repeats agreeing with the majority verdict. For example, with 3 repeats, a case judged `CORRECT`, `CORRECT`, `INCORRECT` has a majority share of 2/3.

It measures judge stability, rather than accuracy against human labels. Also, it's only informative for a judge that samples non-deterministically (temperature > 0 or reasoning models).

To run:
```bash
python -m eval.calibrate_judge output/results/output_*.jsonl --judge-model gpt-6-1-sol --sample 18 --repeats 3
```

Parameters:

* `--sample`: how many distinct EX-passing cases to draw, stratified by difficulty to match the eval set's mix (default 18; if fewer candidates exist, it uses all of them).
* `--repeats`: how many times the judge is re-run on each sampled case, to see whether its verdict stays the same (default 3).


## LangSmith

To emit traces to LangSmith, set these environment variables. LangChain will emit them automatically:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=sql-agent-eval   # optional
```

## Tests

Offline unit tests (no network, no API keys):

```bash
python -m unittest discover -s tests
```

## Layout

```
ai-agent-evals/                # repo root (also the import root)
├── config.py              # paths + MODELS registry + agent/judge selection
├── experiment_profiles.py # loads config/experiment_profiles.yaml
├── setup_db.py            # builds data/roman_empire.db from data/tables/*.csv
├── agent/
│   ├── db.py              # read-only SQLite connection + authored schema metadata
│   ├── tools.py           # get_schema, run_query, get_sample_rows (+ LangChain tools)
│   ├── llm.py             # provider-agnostic chat-model factory
│   ├── agent.py           # run_agent(), AgentTrace, ToolCall
│   └── system_prompt*.md  # default prompt and the v1/v2/v3 variants used by profiles
├── graders/
│   ├── tool_call.py       # Tier 1
│   ├── sql_structural.py  # Tier 2 (sqlglot)
│   ├── execution.py       # Tier 2b (EX, ESM, join diagnostics)
│   └── llm_judge.py       # Tier 5 (branching rubrics)
├── eval/
│   ├── runner.py          # orchestrates all tiers, writes output/results/*.jsonl
│   ├── verify_graders.py  # offline check of Tiers 1/2/2b (no LLM)
│   ├── generate_evals.py  # LLM-generated candidate eval cases
│   ├── calibrate_judge.py # judge test-retest calibration
│   ├── metrics.py         # aggregate run metrics
│   └── replay.py, rejudge_replay.py  # re-grade saved attempts
├── config/                # experiment_profiles.yaml + judge prompt bundles
├── data/
│   ├── tables/            # CSV seed files for the database
│   ├── roman_empire.db    # generated by setup_db.py
│   └── evals/             # questions-*.jsonl (canonical set is git-ignored); toy/ ships with the repo
├── experiments/           # exp_XX: standalone experiments
├── tests/                 # offline unit tests
└── output/                # git-ignored
    ├── results/           # per-question traces + grades (.jsonl)
    ├── summaries/         # aggregate run metrics (.json)
    └── calibration/       # judge calibration reports (.txt)
```

Run modules from the repo root (the import root), e.g. `python -m eval.runner`.

The default profiles (`v1` to `v3`) read test cases from `data/evals/questions-*.jsonl`. That canonical set is not
shipped (the author's run used 61 questions); generate your own (see above) or use the `toy` profile.
