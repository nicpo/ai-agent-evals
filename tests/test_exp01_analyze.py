from __future__ import annotations

import unittest

from experiments.exp_01_graders_disagreement.analyze import build_matrix


def _record(id_, ex_match, branch, verdict, sql="SELECT 1"):
    graders = {"tier2b": {"execution_accuracy": {"match": ex_match, "reason": None}}}
    tier5 = {"branch": branch}
    if branch == "A":
        tier5["false_positive_verdict"] = verdict
        tier5["false_positive_reason"] = "r"
    else:
        tier5["false_negative_verdict"] = verdict
        tier5["false_negative_reason"] = "r"
    graders["tier5"] = tier5
    return {
        "id": id_,
        "question": f"q-{id_}",
        "graders": graders,
        "trace": {"tool_calls": [{"tool_name": "run_query", "parameters": {"sql": sql}}]},
    }


class DisagreementMatrixTest(unittest.TestCase):
    def test_classifies_all_four_cells_and_review(self) -> None:
        records = [
            _record("a", True, "A", "CORRECT"),
            _record("b", True, "A", "FALSE_POSITIVE"),
            _record("c", False, "B", "ACCEPTABLE"),
            _record("d", False, "B", "WRONG"),
            _record("e", False, "B", "GOLD_ERROR"),
        ]
        report = build_matrix(records, {})
        self.assertEqual(report["n_judged"], 5)
        m = report["matrix"]
        self.assertEqual(m["expected_success"]["count"], 1)
        self.assertEqual(m["latent_bug"]["count"], 1)
        self.assertEqual(m["valid_alternative"]["count"], 1)
        self.assertEqual(m["expected_failure"]["count"], 1)
        self.assertEqual(m["review"]["count"], 1)
        self.assertEqual(report["cases"]["latent_bug"][0]["id"], "b")
        self.assertEqual(report["cases"]["valid_alternative"][0]["id"], "c")

    def test_skips_records_without_ex_or_judge(self) -> None:
        no_ex = {"id": "x", "question": "q", "graders": {"tier5": {"branch": None}},
                  "trace": {"tool_calls": []}}
        report = build_matrix([no_ex], {})
        self.assertEqual(report["n_judged"], 0)

    def test_gold_sql_joined_from_cases_by_id(self) -> None:
        records = [_record("a", True, "A", "FALSE_POSITIVE")]
        report = build_matrix(records, {"a": {"gold_sql": "SELECT gold"}})
        self.assertEqual(report["cases"]["latent_bug"][0]["gold_sql"], "SELECT gold")


if __name__ == "__main__":
    unittest.main()
