# SQL Agent + Eval Harness

A text-to-SQL agent over a synthetic Roman Empire database, plus a tiered eval
harness (deterministic tool-call graders, structural SQL graders, gold-SQL
execution accuracy, and an LLM-as-judge).

## Provider flexibility

The agent and the judge talk to LLMs only through `agent/llm.py`, which wraps
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
│   └── llm_judge.py       # Tier 3 (branching rubrics)
├── eval/
│   ├── runner.py          # orchestrates all tiers, writes output/results/*.jsonl
│   └── verify_graders.py  # offline check of Tiers 1/2/2b (no LLM)
├── data/                  # roman_empire.db + evals/questions-*.jsonl
└── output/
    ├── results/           # per-question traces + grades (.jsonl)
    ├── summaries/         # aggregate run metrics (.json)
    └── calibration/       # judge calibration reports (.txt)
```

Run modules from the repo root (the import root), e.g. `python -m eval.runner`.

The agent reads the existing database at `data/roman_empire.db` and the test
cases at `data/evals/questions-*.jsonl` (61 questions).

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
cp .env.example .env            # then edit .env
```

`config.py` loads `.env` automatically at import (a missing `.env` is a no-op,
so the deterministic graders run without any keys).

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

# Full suite (requires the agent/judge provider API keys in the environment):
python -m eval.runner
python -m eval.runner --difficulty easy --limit 5
python -m eval.runner --no-judge
python -m eval.runner --agent-model haiku-4-5 --judge-model gpt-5-4

# Judge calibration (test-retest)
python -m eval.calibrate_judge output/results/output_*.jsonl --judge-model gpt-5-4 --sample 15 --repeats 3
```
