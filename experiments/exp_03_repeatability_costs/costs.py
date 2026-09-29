"""Token and dollar cost accounting for agent attempts and judge calls.

Method (see the costs post): "cost" is several numbers -- calls, input tokens,
output tokens, and dollars move independently, because every LLM call in an
agent loop resends the whole conversation so far. Report them separately
rather than reducing to one headline percentage. Also keep the judge's bill
separate from the agent's: grading cost scales with reruns just like agent
cost does, but it isn't the product's cost.
"""

from __future__ import annotations

from collections import defaultdict

from experiments.exp_03_repeatability_costs.pricing import cost_usd
from graders.llm_judge import Judge


class CostTrackingJudge(Judge):
    """A Judge that records each call's token usage instead of discarding it.

    ``graders.llm_judge.Judge._ask`` only returns response text; this
    subclass overrides it to also append the response's ``usage_metadata`` to
    ``self.usage_log``, without touching the shared harness module.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usage_log: list[dict] = []

    def _ask(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        response = self.model.invoke([HumanMessage(content=prompt)])
        self.usage_log.append(dict(getattr(response, "usage_metadata", None) or {}))
        return (response.text or "").strip()


def _profile_name(record: dict) -> str:
    return (record.get("profile") or {}).get("profile", "unknown")


def _model_keys(record: dict) -> tuple[str | None, str | None]:
    p = record.get("profile") or {}
    return p.get("agent_model"), p.get("judge_model")


def agent_costs(records: list[dict]) -> dict:
    """Per-profile mean tokens/calls and dollar cost per 1,000 attempts."""
    by_profile: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_profile[_profile_name(r)].append(r["trace"])

    out = {}
    for profile, traces in by_profile.items():
        n = len(traces)
        calls = sum(t.get("total_llm_calls", 0) for t in traces)
        input_tokens = sum(t.get("total_input_tokens", 0) for t in traces)
        output_tokens = sum(t.get("total_output_tokens", 0) for t in traces)
        agent_model = _model_keys(next(r for r in records if _profile_name(r) == profile))[0]

        per_1000_input = input_tokens / n * 1000 if n else 0
        per_1000_output = output_tokens / n * 1000 if n else 0
        input_cost = cost_usd(agent_model, per_1000_input, 0) if agent_model else None
        output_cost = cost_usd(agent_model, 0, per_1000_output) if agent_model else None

        out[profile] = {
            "n_attempts": n,
            "agent_model": agent_model,
            "mean_llm_calls": round(calls / n, 2) if n else None,
            "mean_input_tokens": round(input_tokens / n, 1) if n else None,
            "mean_output_tokens": round(output_tokens / n, 1) if n else None,
            "cost_per_1000_attempts_usd": {
                "input": round(input_cost, 2) if input_cost is not None else None,
                "output": round(output_cost, 2) if output_cost is not None else None,
                "total": round(input_cost + output_cost, 2)
                if input_cost is not None and output_cost is not None else None,
            },
        }
    return out


def cost_of_pass(cost_per_1000_usd: float | None, success_rate: float | None) -> float | None:
    """Cost per 1,000 *successful* attempts -- only meaningful with a way to detect success."""
    if cost_per_1000_usd is None or not success_rate:
        return None
    return round(cost_per_1000_usd / success_rate, 2)


def judge_costs(usage_by_profile: dict[str, list[dict]], judge_model_by_profile: dict[str, str]) -> dict:
    """Aggregate a CostTrackingJudge's usage_log per profile into calls/tokens/dollars."""
    out = {}
    for profile, usage_log in usage_by_profile.items():
        n = len(usage_log)
        input_tokens = sum(int(u.get("input_tokens", 0) or 0) for u in usage_log)
        output_tokens = sum(int(u.get("output_tokens", 0) or 0) for u in usage_log)
        model = judge_model_by_profile.get(profile)
        cost = cost_usd(model, input_tokens, output_tokens) if model else None
        out[profile] = {
            "judge_model": model,
            "n_calls": n,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": round(cost, 4) if cost is not None else None,
        }
    return out
