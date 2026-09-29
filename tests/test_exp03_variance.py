from __future__ import annotations

import unittest

from experiments.exp_03_repeatability_costs.variance import (
    agent_sql_variance,
    judge_verdict_variance,
    normalize_sql,
)


def _record(id_, sql, branch=None, verdict=None, profile="p1"):
    graders: dict = {}
    if branch:
        tier5 = {"branch": branch}
        if branch == "A":
            tier5["false_positive_verdict"] = verdict
        else:
            tier5["false_negative_verdict"] = verdict
        graders["tier5"] = tier5
    return {
        "id": id_, "profile": {"profile": profile}, "graders": graders,
        "trace": {"tool_calls": [{"tool_name": "run_query", "parameters": {"sql": sql}}] if sql else []},
    }


class NormalizeSqlTest(unittest.TestCase):
    def test_whitespace_and_case_insensitive(self) -> None:
        a = normalize_sql("select  a,b FROM t WHERE x=1")
        b = normalize_sql("SELECT a, b\nFROM t\nWHERE x = 1")
        self.assertEqual(a, b)

    def test_unparseable_sql_falls_back_to_text_normalization(self) -> None:
        a = normalize_sql("SELECT FROM (broken")
        b = normalize_sql("select from (broken")
        self.assertEqual(a, b)

    def test_none_is_none(self) -> None:
        self.assertIsNone(normalize_sql(None))


class AgentVarianceTest(unittest.TestCase):
    def test_identical_repeats_counted(self) -> None:
        records = [
            _record("q1", "SELECT 1"), _record("q1", "select 1"),
            _record("q2", "SELECT 1"), _record("q2", "SELECT 2"),
        ]
        report = agent_sql_variance(records)
        self.assertEqual(report["p1"]["n_questions"], 2)
        self.assertEqual(report["p1"]["identical_across_repeats"], 1)
        self.assertEqual(report["p1"]["identical_rate"], 0.5)


class JudgeVarianceTest(unittest.TestCase):
    def test_disagreement_on_identical_sql_detected(self) -> None:
        records = [
            _record("q1", "SELECT 1", "A", "CORRECT"),
            _record("q1", "SELECT 1", "A", "FALSE_POSITIVE"),
            _record("q2", "SELECT 2", "A", "CORRECT"),
            _record("q2", "SELECT 2", "A", "CORRECT"),
        ]
        report = judge_verdict_variance(records)
        self.assertEqual(report["p1"]["n_comparable_groups"], 2)
        self.assertEqual(report["p1"]["disagreeing_groups"], 1)
        self.assertEqual(report["p1"]["disagreement_rate"], 0.5)

    def test_singleton_groups_excluded(self) -> None:
        records = [_record("q1", "SELECT 1", "A", "CORRECT")]
        report = judge_verdict_variance(records)
        self.assertEqual(report["p1"]["n_comparable_groups"], 0)
        self.assertIsNone(report["p1"]["disagreement_rate"])


if __name__ == "__main__":
    unittest.main()
