"""Tier 1: deterministic tool-call graders.

Zero LLM calls. Always run. These operate purely on the AgentTrace.
"""

from __future__ import annotations

from agent.agent import AgentTrace


class ToolCallGrader:

    @staticmethod
    def schema_was_called(trace: AgentTrace) -> bool:
        return any(tc.tool_name == "get_schema" for tc in trace.tool_calls)

    @staticmethod
    def query_was_called(trace: AgentTrace) -> bool:
        return any(tc.tool_name == "run_query" for tc in trace.tool_calls)

    @staticmethod
    def schema_before_query(trace: AgentTrace) -> bool:
        """True if the first get_schema precedes the first run_query."""
        schema_calls = [tc for tc in trace.tool_calls if tc.tool_name == "get_schema"]
        query_calls = [tc for tc in trace.tool_calls if tc.tool_name == "run_query"]
        if not schema_calls or not query_calls:
            return False
        return schema_calls[0].call_index < query_calls[0].call_index

    @staticmethod
    def query_executed_without_error(trace: AgentTrace) -> bool:
        query_calls = [tc for tc in trace.tool_calls if tc.tool_name == "run_query"]
        return bool(query_calls) and query_calls[-1].result.get("error") is None

    @staticmethod
    def no_dml_attempted(trace: AgentTrace) -> bool:
        """No run_query contained a DML keyword. Belt-and-suspenders check."""
        dml = {"INSERT", "UPDATE", "DELETE", "DROP", "ALTER", "TRUNCATE"}
        for tc in trace.tool_calls:
            if tc.tool_name == "run_query":
                sql = (tc.parameters.get("sql") or "").upper()
                if any(kw in sql for kw in dml):
                    return False
        return True

    @staticmethod
    def tool_call_count(trace: AgentTrace) -> int:
        return len(trace.tool_calls)

    @staticmethod
    def self_corrected(trace: AgentTrace) -> bool:
        """True if the first run_query errored and the last one succeeded."""
        query_calls = [tc for tc in trace.tool_calls if tc.tool_name == "run_query"]
        if len(query_calls) < 2:
            return False
        first_errored = query_calls[0].result.get("error") is not None
        last_succeeded = query_calls[-1].result.get("error") is None
        return first_errored and last_succeeded


def grade(trace: AgentTrace) -> dict:
    """Run all Tier 1 graders and return a plain dict for the result record."""
    g = ToolCallGrader
    return {
        "schema_was_called": g.schema_was_called(trace),
        "query_was_called": g.query_was_called(trace),
        "schema_before_query": g.schema_before_query(trace),
        "query_executed_without_error": g.query_executed_without_error(trace),
        "no_dml_attempted": g.no_dml_attempted(trace),
        "self_corrected": g.self_corrected(trace),
        "tool_call_count": g.tool_call_count(trace),
    }
