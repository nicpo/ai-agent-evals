"""Model entries and prices for exp 05 (kept local: config.py and agent/ stay untouched).

Entries are passed to agent.llm.get_chat_model as dicts. Prices are USD per million tokens,
all input charged uncached. A model without a price is reported as cost None, and the loop's
budget guard refuses to start until it is filled in.
"""

from __future__ import annotations

import config

AGENT = {
    "id": "anthropic:claude-haiku-5-5",
    "kwargs": {"max_tokens": 4096},  # temperature is rejected by Haiku 5.5 (deprecated)
}
# Frozen in-loop judge: the same entry the rest of the harness uses.
JUDGE = config.MODELS["gpt-6-1-sol"]
PROPOSER = {"id": "anthropic:claude-sonnet-5-5", "kwargs": {"effort": "medium", "max_tokens": 16000}}

PRICES = {  # (input, output) USD per million tokens, checked 2026-10-16
    # Haiku 5.5 is billed higher ($0.50/$2.50) for prompts over 100k tokens; agent prompts here are far below that.
    "anthropic:claude-haiku-5-5": (0.10, 0.50),
    "anthropic:claude-sonnet-5-5": (2.0, 10.0),
    "openai:gpt-6.1-sol": (2.0, 10.0),
}


def cost(model_id: str, input_tokens: int, output_tokens: int) -> float | None:
    rate_in, rate_out = PRICES.get(model_id, (None, None))
    if rate_in is None or rate_out is None:
        return None
    return (input_tokens * rate_in + output_tokens * rate_out) / 1e6
