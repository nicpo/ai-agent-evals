# AI Agent Evals: Reliability, Graders, and Cost

An experimental evaluation harness for a text-to-SQL agent.

I use a small agent and fictional Roman Empire database to investigate how agent evaluation behaves in practice:

- When do deterministic and LLM graders disagree?
- Can both graders agree on an incorrect answer?
- How much does measured performance vary across repeated runs?
- How do agent architecture changes affect accuracy and inference cost?

[Project overview](#project-summary-diagram) · [Experiments and blog](#experiments) · [Run the harness](#run)

## Project summary diagram

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

## Key experiments

The experiments examine execution-versus-judge disagreement, whether grader agreement survives database perturbations, repeated-run reliability and cost accounting, and whether a small hosted decision model (Jev) can replace the LLM judge. See [Experiments](#experiments) for the runnable methodology and related blog-post code.

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

- `get_schema`: returns tables, columns, types, and relationships.
- `get_sample_rows`: returns example records so the agent can discover actual values, such as permitted status names.
- `run_query`: executes a read-only SQL query and returns columns, rows, row count, or an error.

## Setup

### 1. Set up the environment
```bash
python -m venv .venv
.venv/Scripts/activate          # Windows; use source .venv/bin/activate on POSIX
pip install -r requirements.txt
```

### 2. Add API keys
API keys live in a git-ignored `.env` file at the repo root. Copy the template
and fill in the providers you use:

```bash
# PowerShell
Copy-Item .env.example .env

# macOS/Linux
cp .env.example .env
```

Then edit the new `.env` file in the repository root (next to `config.py`).
Never commit it. Add the key for the model provider you select:

```text
# For --model gpt-5-4-mini, gpt-5-4, or gpt-5-4-nano
OPENAI_API_KEY=sk-...

# For --model haiku-4-5 or sonnet-4-6
ANTHROPIC_API_KEY=sk-ant-...

# Only for experiments/exp_04_jev_judge (hosted Jev judge)
TYPESAFE_API_KEY=...
```

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

# Judge calibration (test-retest)
python -m eval.calibrate_judge output/results/output_*.jsonl --judge-model gpt-5-4 --sample 15 --repeats 3
```

Experiment profiles live in `config/experiment_profiles.yaml`. Each profile
selects repository-local agent and judge prompts, tools, models, and harness
settings; results record the resolved profile plus prompt hashes.

## Toy eval set

`data/evals/toy/` contains 4 example cases (2 easy, 1 medium, 1 hard), split by difficulty into `questions-easy.jsonl`, `questions-medium.jsonl`, and `questions-hard.jsonl` - the same naming convention the canonical set uses. **This is a toy set only**. I t exists so you can run the harness end-to-end without generating anything first, and so you can see the eval case structure (question, gold SQL, gold result, and the grader metadata fields) in real files. It lives in a subdirectory so the default `questions-*.jsonl` glob (rooted at `data/evals/`) does not pick it up; point
the runner at it explicitly:

```bash
python -m eval.runner --profile toy
```

The `toy` profile (in `config/experiment_profiles.yaml`) is identical to `v1` except it points `harness.eval_glob` at `toy/questions-*.jsonl` instead of the default `questions-*.jsonl`.

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
python -m eval.generate_evals --model gpt-5-4-mini --count 10 \
  --out data/evals/candidates/roman_empire_candidates.jsonl

# Or use Anthropic, with ANTHROPIC_API_KEY in .env
python -m eval.generate_evals --model haiku-4-5 --count 10 \
  --out data/evals/candidates/roman_empire_candidates.jsonl
```

After the questions were generated, review them:

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

`experiments/` contains standalone code for the methodology behind a few blog
posts about this harness: the EX-vs-judge disagreement matrix, probing
graders' *agreement* with database perturbations, repeated-run
reliability/cost accounting, and Jev (a small hosted decision model) as the
semantic SQL judge next to EX and the GPT judge. Each one runs against the toy eval set by
default and takes a `--profile` flag to point at your own eval set instead.
See [`experiments/README.md`](experiments/README.md).

## Changing models

The agent and the judge talk to LLMs through `agent/llm.py`, which wraps
LangChain's `init_chat_model`. Models live in the `MODELS` registry in
`config.py`; pick the agent and judge by registry key:

```python
# config.py
MODELS = {
    "haiku-4-5":  {"id": "anthropic:claude-haiku-4-5", "kwargs": {"temperature": 0, "max_tokens": 4096}},
    "sonnet-4-6": {"id": "anthropic:claude-sonnet-4-6", "kwargs": {"temperature": 0, "max_tokens": 4096}},
    "gpt-5-4-mini": {"id": "openai:gpt-5.4-mini", "kwargs": {"reasoning_effort": "low"}},
    # ...
}
AGENT_MODEL = "sonnet-4-6"      # the agent under test
JUDGE_MODEL = "gpt-5-4-mini"    # the LLM judge
```

The judge defaults to a different model family than the agent to avoid self-preference bias.

## LangSmith

To emit traces to LangSmith, set these environment variables. LangChain will emit them automatically:
```
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=sql-agent-eval   # optional
```

## Layout

```
rome-eval-harness/             # repo root (also the import root)
├── config.py              # paths + MODELS registry + agent/judge selection
├── agent/
│   ├── db.py              # read-only SQLite connection + authored schema metadata
│   ├── tools.py           # get_schema, run_query, get_sample_rows (+ LangChain tools)
│   ├── llm.py             # provider-agnostic chat-model factory
│   ├── agent.py           # run_agent(), AgentTrace, ToolCall
│   └── system_prompt.md
├── graders/
│   ├── tool_call.py       # Tier 1
│   ├── sql_structural.py  # Tier 2 (sqlglot)
│   ├── execution.py       # Tier 2b (EX, ESM, join diagnostics)
│   └── llm_judge.py       # Tier 5 (branching rubrics)
├── eval/
│   ├── runner.py          # orchestrates all tiers, writes output/results/*.jsonl
│   └── verify_graders.py  # offline check of Tiers 1/2/2b (no LLM)
├── config/                # experiment_profiles.yaml + judge prompt bundles
├── data/                  # roman_empire.db + evals/questions-*.jsonl (toy set in evals/toy/)
├── experiments/           # exp_01 to exp_04: standalone blog-post experiments
├── tests/                 # offline unit tests: python -m unittest discover -s tests
└── output/
    ├── results/           # per-question traces + grades (.jsonl)
    ├── summaries/         # aggregate run metrics (.json)
    └── calibration/       # judge calibration reports (.txt)
```

Run modules from the repo root (the import root), e.g. `python -m eval.runner`.

The agent reads the existing database at `data/roman_empire.db` and the test
cases at `data/evals/questions-*.jsonl` (61 questions).
