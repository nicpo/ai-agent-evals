"""Central configuration: filesystem paths and model selection.

Model selection is provider-agnostic. Each entry in ``MODELS`` carries a
LangChain ``provider:model`` id and the constructor kwargs for that model; the
id is passed straight to ``init_chat_model`` (see ``agent/llm.py``). Pick the
agent and judge by their friendly registry key (``AGENT_MODEL`` / ``JUDGE_MODEL``),
so switching providers is a one-line change here with no code changes elsewhere.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -------------------------------------------------------------------
# config.py lives at the repository root, so resolve everything relative to it.
# This keeps the harness working regardless of the current working directory.
REPO_ROOT = Path(__file__).resolve().parent

DATA_DIR = REPO_ROOT / "data"
DB_PATH = DATA_DIR / "roman_empire.db"
EVALS_DIR = DATA_DIR / "evals"
OUTPUT_DIR = REPO_ROOT / "output"
RESULTS_DIR = OUTPUT_DIR / "results"        # per-question traces + grades (.jsonl)
SUMMARIES_DIR = OUTPUT_DIR / "summaries"    # aggregate run metrics (.json)
CALIBRATION_DIR = OUTPUT_DIR / "calibration"  # judge calibration reports (.txt)

# --- Models ------------------------------------------------------------------
# Registry of selectable models. Keys are friendly names referenced by
# AGENT_MODEL / JUDGE_MODEL below. Each entry has:
#   "id"     -- LangChain "provider:model" string passed to init_chat_model.
#   "kwargs" -- constructor kwargs for that model (temperature, max_tokens,
#               reasoning_effort, ...), forwarded verbatim.
MODELS: dict[str, dict] = {
    # GPT-6.x reasoning models: function tools + reasoning_effort require
    # OpenAI's /v1/responses endpoint, so route them through it.
    "gpt-6-1-sol": {
        "id": "openai:gpt-6.1-sol",
        "kwargs": {"reasoning_effort": "medium"},
    },
    "haiku-4-5": {
        "id": "anthropic:claude-haiku-4-5",
        "kwargs": {"temperature": 0, "max_tokens": 4096},
    },
    "sonnet-5-5": {
        "id": "anthropic:claude-sonnet-5-5",
        "kwargs": {"temperature": 0, "max_tokens": 4096},
    },
}

# The agent under test and the LLM judge, selected by registry key. The judge is
# deliberately a *different model family* from the agent to avoid self-preference bias.
AGENT_MODEL = "haiku-4-5"  # or "gpt-6-1-sol"
JUDGE_MODEL = "gpt-6-1-sol" # or "opus-5-5"


def resolve_model(name: str) -> dict:
    """Return the MODELS entry for a friendly name, or raise a clear error."""
    try:
        return MODELS[name]
    except KeyError:
        raise KeyError(
            f"Unknown model {name!r}. Available: {', '.join(MODELS)}"
        ) from None


# API keys
# Keys live in a git-ignored .env file at the repo root, one KEY=VALUE per line:
#   ANTHROPIC_API_KEY=sk-ant-...
#   OPENAI_API_KEY=sk-...
# See .env.example for the template. Loaded via python-dotenv. Never commit .env.
ENV_FILE = REPO_ROOT / ".env"


def load_env(path: Path = ENV_FILE, *, override: bool = False) -> bool:
    """Load the .env file into ``os.environ`` via python-dotenv.

    LangChain's provider integrations read keys from the standard environment
    variables (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...), so this bridges
    the git-ignored .env file -> ``os.environ``. A missing .env -- or a missing
    python-dotenv install -- is a no-op, so the deterministic-grader path still
    imports config without keys or the package.

    Args:
        path: Path to the .env file (defaults to ``<repo root>/.env``).
        override: If True, .env values overwrite variables already set in the
            environment; if False (default), existing env vars win.

    Returns:
        True if a .env file was found and loaded, else False.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return False
    return load_dotenv(dotenv_path=path, override=override)


# Load .env at import time so keys are available before any model is built.
load_env()


# Glob used by the runner to discover the canonical eval files (the dated
# subfolders under data/evals/ are historical snapshots and are ignored).
EVAL_GLOB = "questions-*.jsonl"

# Circuit breaker: maximum number of tool calls before the agent loop aborts.
MAX_TOOL_CALLS = 12
PROFILE_CATALOG = REPO_ROOT / "config" / "experiment_profiles.yaml"

