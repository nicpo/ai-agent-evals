"""Failure packets: what the proposer sees about failed DEV attempts.

No gold SQL, no gold results, no validation or test data. Tool results are truncated.
"""

from __future__ import annotations

import json
import random


def _clip(text, n: int) -> str:
    text = text if isinstance(text, str) else json.dumps(text, default=str)
    return text if len(text) <= n else text[:n] + " ...[truncated]"


def render_attempt(rec: dict) -> str:
    lines = [f"Question: {rec['question']}"]
    for i, call in enumerate(rec["trace"]["tool_calls"], 1):
        result = call.get("result") or {}
        if result.get("error"):
            shown = f"ERROR: {_clip(result['error'], 200)}"
        else:
            rows = result.get("rows") or []
            shown = f"{result.get('row_count', len(rows))} rows; first rows: {_clip(rows[:3], 350)}"
        lines.append(f"  call {i}: {call['tool_name']}({_clip(call.get('parameters') or {}, 500)}) -> {shown}")
    if rec["trace"].get("error"):
        lines.append(f"  agent error: {_clip(rec['trace']['error'], 200)}")
    lines.append(f"Agent final SQL (last run_query): {_clip(rec.get('agent_sql') or '(none)', 800)}")
    lines.append(f"Agent final answer: {_clip(rec['trace'].get('final_answer') or '(none)', 600)}")
    verdict = rec.get("verdict") or "NO VERDICT (no SQL or agent error)"
    lines.append(f"Reviewer verdict: {verdict}")
    if rec.get("reason"):
        lines.append(f"Reviewer reasoning: {_clip(rec['reason'], 900)}")
    return "\n".join(lines)


def failed_by_question(records: list[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for r in records:
        if not r["success"]:
            out.setdefault(r["id"], []).append(r)
    return out


def select_failures(records: list[dict], seed: int, k: int = 4) -> list[dict]:
    """Up to k failed attempts from distinct questions, most-often-failing questions first."""
    rng = random.Random(seed)
    failed = failed_by_question(records)
    order = sorted(failed, key=lambda q: (-len(failed[q]), rng.random()))
    return [rng.choice(failed[q]) for q in order[:k]]


def render_packet(attempts: list[dict]) -> str:
    return "\n\n".join(f"--- Failed attempt {i} ---\n{render_attempt(r)}" for i, r in enumerate(attempts, 1))
