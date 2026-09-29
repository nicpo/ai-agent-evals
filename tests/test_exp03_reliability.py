from __future__ import annotations

import unittest

from experiments.exp_03_repeatability_costs.reliability import (
    compute_reliability,
    group_by_question,
    is_success,
)


def _record(id_, run_index, ex_match, branch=None, verdict=None, profile="p1"):
    graders = {"tier2b": {"execution_accuracy": {"match": ex_match}}}
    if branch:
        tier5 = {"branch": branch}
        if branch == "A":
            tier5["false_positive_verdict"] = verdict
        else:
            tier5["false_negative_verdict"] = verdict
        graders["tier5"] = tier5
    return {
        "id": id_, "run_index": run_index, "graders": graders,
        "profile": {"profile": profile},
        "trace": {"tool_calls": []},
    }


class SuccessTest(unittest.TestCase):
    def test_ex_pass_without_judge_is_success(self) -> None:
        self.assertTrue(is_success(_record("a", 0, True)))

    def test_ex_pass_with_false_positive_is_failure(self) -> None:
        self.assertFalse(is_success(_record("a", 0, True, "A", "FALSE_POSITIVE")))

    def test_ex_fail_with_acceptable_is_success(self) -> None:
        self.assertTrue(is_success(_record("a", 0, False, "B", "ACCEPTABLE")))

    def test_ex_fail_with_wrong_is_failure(self) -> None:
        self.assertFalse(is_success(_record("a", 0, False, "B", "WRONG")))

    def test_no_ex_outcome_is_none(self) -> None:
        r = {"id": "a", "run_index": 0, "graders": {}, "trace": {"tool_calls": []}}
        self.assertIsNone(is_success(r))


class ReliabilityTest(unittest.TestCase):
    def test_pass_hat_k_and_pass_at_k(self) -> None:
        # q1: 3/3 success (pass^3 and pass@3 both true)
        # q2: 0/3 success (neither)
        # q3: 1/3 success (only pass@3)
        records = []
        for run in range(3):
            records.append(_record("q1", run, True))
            records.append(_record("q2", run, False, "B", "WRONG"))
        records.append(_record("q3", 0, True))
        records.append(_record("q3", 1, False, "B", "WRONG"))
        records.append(_record("q3", 2, False, "B", "WRONG"))

        report = compute_reliability(records)
        r = report["p1"]
        self.assertEqual(r["n_questions"], 3)
        self.assertEqual(r["n_attempts"], 9)
        self.assertAlmostEqual(r["mean_success"], 4 / 9, places=4)
        self.assertAlmostEqual(r["pass_hat_k"][3], 1 / 3, places=4)
        self.assertAlmostEqual(r["pass_at_k"][3], 2 / 3, places=4)

    def test_groups_separate_profiles(self) -> None:
        records = [_record("q1", 0, True, profile="a"), _record("q1", 0, False, "B", "WRONG", profile="b")]
        report = compute_reliability(records)
        self.assertEqual(set(report), {"a", "b"})
        self.assertEqual(report["a"]["mean_success"], 1.0)
        self.assertEqual(report["b"]["mean_success"], 0.0)

    def test_group_by_question_orders_by_run_index(self) -> None:
        records = [_record("q1", 1, False, "B", "WRONG"), _record("q1", 0, True)]
        grouped = group_by_question(records)
        self.assertEqual(grouped["p1"]["q1"], [True, False])


if __name__ == "__main__":
    unittest.main()
