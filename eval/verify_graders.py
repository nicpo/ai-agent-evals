"""Verify the deterministic graders (Tiers 1/2/2b) against hand-built traces.

No LLM calls. Constructs AgentTraces by hand -- as if an agent had produced
them -- and runs the grader stack against the real SQLite database, asserting
the outcomes. Run with:

    python -m eval.verify_graders
"""

from __future__ import annotations

import json

from agent import db
from agent.agent import AgentTrace, ToolCall
from eval.runner import grade_trace, load_questions


def _schema_call(idx: int) -> ToolCall:
    return ToolCall("get_schema", {}, db.SCHEMA, idx)


def _query_call(idx: int, sql: str, conn) -> ToolCall:
    """Build a run_query ToolCall by actually executing against the DB."""
    from agent.tools import Toolbox

    tb = Toolbox(conn)
    return ToolCall("run_query", {"sql": sql}, tb.run_query(sql), idx)


def _case(cases: list[dict], cid: str) -> dict:
    return next(c for c in cases if c["id"] == cid)


def main() -> None:
    cases = load_questions()
    conn = db.connect()
    failures: list[str] = []

    def check(label: str, got, want):
        ok = got == want
        print(f"  {'PASS' if ok else 'FAIL'}  {label}: got={got!r} want={want!r}")
        if not ok:
            failures.append(label)

    # --- Scenario 1: clean correct run (schema -> correct query) ----------
    c1 = _case(cases, "easy_001")
    trace1 = AgentTrace(question=c1["question"])
    trace1.tool_calls = [
        _schema_call(0),
        _query_call(1, c1["gold_sql"], conn),
    ]
    trace1.final_answer = "Egypt has the highest annual tribute at 80,000 denarii; Dacia the lowest at 8,000."
    g1 = grade_trace(trace1, c1, conn, run_judge=False)
    print("\nScenario 1 — clean correct run (easy_001):")
    check("tier1.schema_before_query", g1["tier1"]["schema_before_query"], True)
    check("tier1.query_executed_without_error", g1["tier1"]["query_executed_without_error"], True)
    check("tier1.self_corrected", g1["tier1"]["self_corrected"], False)
    check("tier1.tool_call_count", g1["tier1"]["tool_call_count"], 2)
    check("tier2.is_parseable", g1["tier2"]["is_parseable"], True)
    check("tier2.hallucinated_references", g1["tier2"]["hallucinated_references"], [])
    check("tier2b.execution_accuracy.match", g1["tier2b"]["execution_accuracy"]["match"], True)

    # --- Scenario 2: self-correction (bad column -> fixed) ----------------
    c2 = _case(cases, "easy_001")
    trace2 = AgentTrace(question=c2["question"])
    bad_sql = "SELECT name, total_taxes FROM provinces ORDER BY total_taxes DESC"
    trace2.tool_calls = [
        _schema_call(0),
        _query_call(1, bad_sql, conn),          # errors: no such column
        _query_call(2, c2["gold_sql"], conn),   # corrected
    ]
    trace2.final_answer = "Egypt leads at 80,000 denarii."
    g2 = grade_trace(trace2, c2, conn, run_judge=False)
    print("\nScenario 2 — self-correction:")
    check("first query errored", trace2.tool_calls[1].result["error"] is not None, True)
    check("tier1.self_corrected", g2["tier1"]["self_corrected"], True)
    check("tier2b.execution_accuracy.match", g2["tier2b"]["execution_accuracy"]["match"], True)

    # --- Scenario 3: schema-hallucination trap (invalid column survives) --
    c3 = _case(cases, "easy_001")
    trace3 = AgentTrace(question=c3["question"])
    # Parses fine, executes? No -- 'total_taxes' doesn't exist, so it errors,
    # but Tier 2 still flags the hallucinated column structurally.
    trace3.tool_calls = [_query_call(0, bad_sql, conn)]
    g3 = grade_trace(trace3, c3, conn, run_judge=False)
    print("\nScenario 3 — schema hallucination trap:")
    check("tier1.schema_was_called", g3["tier1"]["schema_was_called"], False)
    check("tier2.hallucinated_references", g3["tier2"]["hallucinated_references"], ["column:total_taxes"])
    check("tier2b.execution_accuracy.match", g3["tier2b"]["execution_accuracy"]["match"], False)

    # --- Scenario 4: outer-join diagnostic (INNER where LEFT needed) ------
    # Find a case that requires an outer join, if present.
    outer = next((c for c in cases if c.get("requires_outer_join") and c.get("gold_sql")), None)
    if outer:
        inner_variant = outer["gold_sql"].replace("LEFT JOIN", "JOIN").replace("LEFT OUTER JOIN", "JOIN")
        trace4 = AgentTrace(question=outer["question"])
        trace4.tool_calls = [_schema_call(0), _query_call(1, inner_variant, conn)]
        g4 = grade_trace(trace4, outer, conn, run_judge=False)
        ex = g4["tier2b"]["execution_accuracy"]
        joins = g4["tier2b"].get("joins", {})
        print(f"\nScenario 4 — outer-join necessity ({outer['id']}):")
        check("tier2b.execution_accuracy.match (INNER should differ)", ex["match"], False)
        check("tier2b.joins.join_type_match", joins.get("join_type_match"), False)
    else:
        print("\nScenario 4 — skipped (no requires_outer_join case found).")

    conn.close()

    print("\n" + ("ALL CHECKS PASSED" if not failures else f"FAILURES: {failures}"))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
