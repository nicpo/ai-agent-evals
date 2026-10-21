"""Critique + proposal step of the loop (Sonnet 5.5, medium effort), plus a stub for tests.

The proposer sees: the incumbent's prompt and tool descriptions, a packet of failed DEV
attempts (question, tool calls, final SQL and answer, verdict and reviewer reasoning), and a
one-line history of earlier proposals with aggregate score deltas. It never sees gold SQL
fields, validation attempts, or test data. The reviewer's reasoning can paraphrase what the
reference answer does; that is the intended grader feedback (GEPA style), and the lint plus
the final test-set and independent-grader evaluation are the checks against leakage.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from experiments.exp_05_self_improvement_loop import models
from experiments.exp_05_self_improvement_loop.candidates import ENABLED_TOOLS, Candidate
from experiments.exp_05_self_improvement_loop.lint import MAX_TOTAL_CHARS, candidate_text

PROMPTS = Path(__file__).resolve().parent / "prompts"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


@dataclass
class Proposal:
    summary: str
    prompt: str
    descriptions: dict
    critique: str
    usage: dict = field(default_factory=dict)  # {"input_tokens", "output_tokens"} summed over calls
    attempts: int = 1

    @property
    def cost(self) -> float | None:
        return models.cost(models.PROPOSER["id"], self.usage.get("input_tokens", 0), self.usage.get("output_tokens", 0))


def render_tools(c: Candidate) -> str:
    return "\n".join(f"- {name}: {c.descriptions[name]}" for name in ENABLED_TOOLS)


def fill(template: str, **values: str) -> str:
    for key, value in values.items():
        template = template.replace(f"<<{key.upper()}>>", value)
    return template


def parse_proposal_json(text: str) -> dict:
    cleaned = _FENCE.sub("", text.strip()).strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("no JSON object in proposer response")
    data = json.loads(cleaned[start:end + 1])
    tools = data.get("tools")
    if not (isinstance(data.get("prompt"), str) and isinstance(tools, dict)
            and set(tools) == set(ENABLED_TOOLS) and all(isinstance(v, str) for v in tools.values())):
        raise ValueError("proposal JSON must have 'prompt' and 'tools' for exactly " + ", ".join(ENABLED_TOOLS))
    return {"summary": str(data.get("summary", "")).strip(), "prompt": data["prompt"], "tools": tools}


def _invoke(model, prompt: str) -> tuple[str, dict]:
    """Stream (long thinking requests can drop non-streaming connections); return text and usage."""
    agg = None
    for chunk in model.stream(prompt):
        agg = chunk if agg is None else agg + chunk
    usage = dict(getattr(agg, "usage_metadata", None) or {})
    return (agg.text or "").strip(), usage


class SonnetProposer:
    def __init__(self, entry: dict = models.PROPOSER):
        from langchain.chat_models import init_chat_model

        kwargs = dict(entry["kwargs"])
        self.model = init_chat_model(entry["id"], **kwargs)
        self.critique_template = (PROMPTS / "critique.md").read_text(encoding="utf-8")
        self.propose_template = (PROMPTS / "propose.md").read_text(encoding="utf-8")

    def propose(self, incumbent: Candidate, failures: str, history: str, lint_feedback: list[str] | None = None,
                previous: Proposal | None = None) -> Proposal:
        usage = {"input_tokens": 0, "output_tokens": 0}

        def add(u: dict) -> None:
            usage["input_tokens"] += u.get("input_tokens", 0)
            usage["output_tokens"] += u.get("output_tokens", 0)

        if previous is None:
            critique, u = _invoke(self.model, fill(self.critique_template, prompt=incumbent.prompt,
                                                   tools=render_tools(incumbent), history=history, failures=failures))
            add(u)
        else:
            critique = previous.critique
            add(previous.usage)
        repair = ""
        if lint_feedback:
            repair = ("\nYour previous attempt was rejected for these reasons. Fix them and keep the same hypothesis:\n- "
                      + "\n- ".join(lint_feedback) + "\n")
        text, u = _invoke(self.model, fill(self.propose_template, prompt=incumbent.prompt, tools=render_tools(incumbent),
                                           critique=critique, history=history, repair=repair,
                                           limit=str(MAX_TOTAL_CHARS), length=str(len(candidate_text(incumbent)))))
        add(u)
        data = parse_proposal_json(text)
        return Proposal(data["summary"], data["prompt"], data["tools"], critique, usage,
                        attempts=(previous.attempts + 1) if previous else 1)


class StubProposer:
    """No API calls. Appends one harmless sentence; used by tests and dry runs."""

    def __init__(self, sentence: str = "Prefer the simplest query that answers the question."):
        self.sentence = sentence
        self.calls = 0

    def propose(self, incumbent: Candidate, failures: str, history: str, lint_feedback=None, previous=None) -> Proposal:
        self.calls += 1
        return Proposal("stub: append one sentence", incumbent.prompt.rstrip() + "\n\n" + self.sentence + "\n",
                        dict(incumbent.descriptions), "stub critique", {"input_tokens": 0, "output_tokens": 0},
                        attempts=(previous.attempts + 1) if previous else 1)
