"""List-price token rates, keyed by the same registry names as ``config.MODELS``.

Kept separate from ``config.py`` because prices change independently of model
selection. Fill in current list prices for whichever models you actually run
this experiment with -- a model missing here just can't have its dollar cost
computed (token/call counts are still reported).

All rates are USD per 1,000,000 tokens.
"""

from __future__ import annotations

PRICING: dict[str, dict[str, float]] = {
    # Keys must match config.MODELS registry keys.
    "haiku-4-5": {"input_per_million": 1.0, "output_per_million": 5.0},
    "sonnet-4-6": {"input_per_million": 3.0, "output_per_million": 15.0},
    "gpt-5-4-nano": {"input_per_million": 0.05, "output_per_million": 0.4},
    "gpt-5-4-mini": {"input_per_million": 0.25, "output_per_million": 2.0},
    "gpt-5-4": {"input_per_million": 2.5, "output_per_million": 15.0},
}


def cost_usd(model_key: str, input_tokens: int, output_tokens: int) -> float | None:
    """Dollar cost for a token count under ``model_key``'s list price, or None if unpriced."""
    rates = PRICING.get(model_key)
    if rates is None:
        return None
    return (
        input_tokens * rates["input_per_million"] / 1_000_000
        + output_tokens * rates["output_per_million"] / 1_000_000
    )
