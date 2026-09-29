"""Probe dual-pass cases against scenario databases.

A "dual pass" is a case where execution accuracy (EX) matched *and* the LLM
judge's Branch A false-positive check said `CORRECT` -- both graders agreed,
on the original database. This module re-runs the gold SQL and the agent's
saved SQL against each scenario database (see ``scenarios.py``) and flags any
scenario where they now disagree.

A flagged mismatch is a **candidate**, not a confirmed bug: per the post, it
still needs a human to check the scenario is a legitimate database state and
that the gold SQL's behavior under it is actually the intended one before
concluding the agent's SQL is wrong.
"""

from __future__ import annotations

import os
from pathlib import Path

from agent import db as db_module
from experiments.exp_02_graders_agreement.scenarios import Scenario, build_scenario_db
from graders.execution import execution_accuracy


def _pred_sql(record: dict) -> str | None:
    queries = [tc for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"]
    return queries[-1]["parameters"].get("sql") if queries else None


def dual_pass_candidates(records: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """Records where EX matched and the judge's Branch A verdict was CORRECT."""
    out = []
    for r in records:
        ex = (r["graders"].get("tier2b") or {}).get("execution_accuracy") or {}
        tier5 = r["graders"].get("tier5") or {}
        if not ex.get("match") or tier5.get("false_positive_verdict") != "CORRECT":
            continue
        case = cases_by_id.get(r.get("id")) or {}
        gold_sql = case.get("gold_sql")
        agent_sql = _pred_sql(r)
        if gold_sql and agent_sql:
            out.append({
                "id": r.get("id"), "question": r.get("question"),
                "gold_sql": gold_sql, "agent_sql": agent_sql,
            })
    return out


def probe_candidate(candidate: dict, scenario: Scenario, source_db: Path) -> dict:
    """Run gold + agent SQL against one scenario DB; report match/mismatch."""
    scenario_db = build_scenario_db(source_db, scenario)
    try:
        conn = db_module.connect(scenario_db)
        try:
            result = execution_accuracy(candidate["gold_sql"], candidate["agent_sql"], conn)
        finally:
            conn.close()
    finally:
        os.unlink(scenario_db)

    return {
        "id": candidate["id"],
        "question": candidate["question"],
        "scenario": scenario.name,
        "scenario_description": scenario.description,
        "still_matches": result.get("match"),
        "execution_detail": result,
        "needs_manual_review": not result.get("match"),
    }


def probe_all(
    candidates: list[dict], scenarios: list[Scenario], source_db: Path
) -> dict:
    """Probe every (candidate x scenario) pair; return counterexamples + a summary."""
    probes = [probe_candidate(c, s, source_db) for c in candidates for s in scenarios]
    counterexamples = [p for p in probes if p["needs_manual_review"]]
    return {
        "n_candidates": len(candidates),
        "n_scenarios": len(scenarios),
        "n_probes": len(probes),
        "n_counterexamples": len(counterexamples),
        "counterexamples": counterexamples,
        "note": (
            "A counterexample means the agent SQL and gold SQL disagree under a "
            "database change the schema already permits. It is a candidate, not "
            "a confirmed bug -- verify the scenario is a legitimate state and "
            "check the gold SQL's behavior against the question wording before "
            "treating it as a real fault."
        ),
    }
