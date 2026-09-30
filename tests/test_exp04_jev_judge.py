from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp_04_jev_judge.jev_client import JevClientError, JevResponse
from experiments.exp_04_jev_judge.jev_judge import (
    after_request, attempt_key, build_cases, judge_case, load_jsonl, load_prompt,
    narrow_request, parse_narrow, request_for_case, run_judgments,
)

CONFIG = {"model": "jev-test"}
SCHEMA = "t(a, b)"
QUESTIONS = {
    "q1": {"id": "q1", "question": "How many?", "gold_sql": "SELECT COUNT(*) AS n FROM t", "gold_result": [{"n": 2}]},
    "q2": {"id": "q2", "question": "Which?", "gold_sql": "SELECT a FROM t", "gold_result": [{"a": 1}]},
}


def _record(id_, match, sql="SELECT 1", rows=None, profile="toy", run_index=0, gpt="CORRECT"):
    calls = []
    if sql is not None:
        calls = [{"tool_name": "run_query", "parameters": {"sql": sql}, "result": {"rows": rows or []}}]
    graders = {"tier5": {"branch": "A" if match else "B", "false_positive_verdict": gpt}}
    if match is not None:
        graders["tier2b"] = {"execution_accuracy": {"match": match}}
    return {"id": id_, "profile": {"profile": profile}, "run_index": run_index,
            "trace": {"tool_calls": calls}, "graders": graders}


def _response(payload, latency=10.0):
    return JevResponse(payload=payload, status=200, request_id="req", latency_ms=latency)


class StubClient:
    """Answers choice questions with ``choice`` and noul questions with ``p``."""

    def __init__(self, choice="A", confidence=0.9, p=0.8, model="jev-test", fail_on=None):
        self.choice, self.confidence, self.p, self.model = choice, confidence, p, model
        self.fail_on, self.calls = fail_on, []

    def system_one(self, *, model, state, questions):
        self.calls.append({"model": model, "state": state, "questions": questions})
        if self.fail_on and state["agent_sql"] == self.fail_on:
            raise JevClientError("HTTP 500")
        answers = {}
        for name, spec in questions.items():
            if spec["type"] == "choice":
                rest = (1 - self.confidence) / 2
                probs = {o: (self.confidence if o == self.choice else rest) for o in "ABC"}
                answers[name] = {"type": "choice", "choice": self.choice,
                                 "confidence": self.confidence, "probabilities": probs}
            else:
                answers[name] = {"type": "noul", "noul": self.p}
        return _response({"model": self.model, "usage": {"input_tokens": 100, "output_tokens": 5},
                          "answers": answers})


class BuildCasesTest(unittest.TestCase):
    def test_branch_follows_ex_and_skips_unjudgeable_attempts(self):
        records = [
            _record("q1", True, "SELECT COUNT(*) AS n FROM t"),
            _record("q2", False, "SELECT 1", rows=[{"a": 9}]),
            _record("q1", None, sql=None),                     # no SQL
            _record("q2", None, "SELECT a FROM t"),            # SQL but no EX verdict
            _record("missing", True, "SELECT 1"),              # question not in eval set
        ]
        cases, skipped = build_cases(records, QUESTIONS, schema=SCHEMA)
        self.assertEqual(skipped, 3)
        self.assertEqual([c["branch"] for c in cases], ["A", "B"])
        self.assertEqual(cases[1]["agent_result"], [{"a": 9}])
        self.assertEqual(cases[1]["gold_result"], [{"a": 1}])
        self.assertEqual(cases[0]["gpt_judge"], "CORRECT")

    def test_attempt_key_separates_profile_and_repeat(self):
        keys = {attempt_key(_record("q1", True, profile=p, run_index=r)) for p in ("v1", "v2") for r in (0, 1)}
        self.assertEqual(len(keys), 4)


class RequestTest(unittest.TestCase):
    def setUp(self):
        self.prompt, self.decomposed = load_prompt("monolithic"), load_prompt("decomposed")
        records = [_record("q1", True, "SELECT COUNT(*) AS n FROM t"), _record("q2", False, "SELECT 1", rows=[{"a": 9}])]
        self.case_a, self.case_b = build_cases(records, QUESTIONS, schema=SCHEMA)[0]

    def test_branch_a_sends_no_result_sets(self):
        state, questions = request_for_case(self.case_a, self.prompt)
        self.assertEqual(set(state), {"user_question", "schema", "approved_sql", "agent_sql"})
        self.assertEqual(questions["semantic_correctness"]["type"], "choice")
        self.assertEqual(set(questions["semantic_correctness"]["criteria"]), {"A", "B", "C"})

    def test_branch_b_sends_both_result_sets(self):
        state, _ = request_for_case(self.case_b, self.prompt)
        self.assertEqual(state["approved_result"], [{"a": 1}])
        self.assertEqual(state["agent_result"], [{"a": 9}])

    def test_narrow_questions_depend_on_branch(self):
        self.assertIn("filtering", narrow_request(self.case_a, self.decomposed))
        self.assertIn("same_entities", narrow_request(self.case_b, self.decomposed))
        self.assertTrue(all(q["type"] == "noul" for q in narrow_request(self.case_a, self.decomposed).values()))

    def test_after_request_adds_assessments_and_suffix(self):
        narrow = {"probabilities": {"schema_alignment": 0.91234, "filtering": 0.5, "clauses": 0.2, "multiple_rows": 1.0}}
        state, questions = after_request(self.case_a, self.prompt, self.decomposed, narrow)
        self.assertEqual(len(state["narrow_assessments"]), 4)
        self.assertEqual(state["narrow_assessments"][0]["probability_yes"], 0.912)
        self.assertIn("narrow_assessments", questions["semantic_correctness"]["instructions"])
        # The base prompt must not be mutated for later cases.
        _, fresh = request_for_case(self.case_a, self.prompt)
        self.assertNotIn("narrow_assessments", fresh["semantic_correctness"]["instructions"])

    def test_parse_narrow_rejects_malformed_and_out_of_range(self):
        names = ["a", "b"]
        ok = _response({"model": "m", "usage": {}, "answers": {"a": {"noul": 0.1}, "b": {"noul": 1}}})
        self.assertEqual(parse_narrow(ok, names)["probabilities"], {"a": 0.1, "b": 1.0})
        with self.assertRaises(JevClientError):
            parse_narrow(_response({"model": "m", "usage": {}, "answers": {"a": {"noul": 0.1}}}), names)
        with self.assertRaises(JevClientError):
            parse_narrow(_response({"model": "m", "usage": {}, "answers": {"a": {"noul": 1.5}, "b": {"noul": 0}}}), names)


class JudgeCaseTest(unittest.TestCase):
    def setUp(self):
        self.prompt, self.decomposed = load_prompt("monolithic"), load_prompt("decomposed")
        self.case = build_cases([_record("q1", True, "SELECT COUNT(*) AS n FROM t")], QUESTIONS, schema=SCHEMA)[0][0]

    def test_label_map(self):
        for choice, verdict in (("A", "CORRECT"), ("B", "INCORRECT"), ("C", "CANNOT_DETERMINE")):
            result = judge_case(StubClient(choice=choice), CONFIG, self.case, "mono", self.prompt)
            self.assertEqual(result["mono"]["verdict"], verdict)

    def test_two_call_makes_three_calls_and_carries_narrow_answers(self):
        client = StubClient(p=0.7)
        result = judge_case(client, CONFIG, self.case, "two-call", self.prompt, self.decomposed)
        self.assertEqual(len(client.calls), 3)
        self.assertEqual(set(result), {"mono", "narrow", "after"})
        self.assertIn("narrow_assessments", client.calls[2]["state"])
        self.assertNotIn("narrow_assessments", client.calls[0]["state"])

    def test_model_mismatch_stops(self):
        with self.assertRaises(ValueError):
            judge_case(StubClient(model="other"), CONFIG, self.case, "mono", self.prompt)

    def test_unknown_variant(self):
        with self.assertRaises(ValueError):
            judge_case(StubClient(), CONFIG, self.case, "nope", self.prompt)


class RunJudgmentsTest(unittest.TestCase):
    def setUp(self):
        records = [_record("q1", True, "SELECT COUNT(*) AS n FROM t"), _record("q2", False, "SELECT 1", rows=[{"a": 9}])]
        self.cases, _ = build_cases(records, QUESTIONS, schema=SCHEMA)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out.jsonl"

    def test_writes_one_row_per_case_and_resumes(self):
        client = StubClient()
        counts = run_judgments(self.cases, client, CONFIG, "mono", self.out, log=lambda _: None)
        self.assertEqual(counts["judged"], 2)
        rows = load_jsonl(self.out)
        self.assertEqual([r["branch"] for r in rows], ["A", "B"])
        self.assertEqual(rows[0]["mono"]["verdict"], "CORRECT")
        self.assertEqual(rows[0]["agent_sql"], "SELECT COUNT(*) AS n FROM t")

        again = StubClient()
        counts = run_judgments(self.cases, again, CONFIG, "mono", self.out, log=lambda _: None)
        self.assertEqual((counts["judged"], counts["resumed"], again.calls), (0, 2, []))
        self.assertEqual(len(load_jsonl(self.out)), 2)

    def test_failure_is_recorded_and_run_continues(self):
        client = StubClient(fail_on="SELECT 1")
        counts = run_judgments(self.cases, client, CONFIG, "mono", self.out, log=lambda _: None)
        self.assertEqual((counts["judged"], counts["failed"]), (1, 1))
        failures = load_jsonl(self.out.with_suffix(".failures.jsonl"))
        self.assertEqual(failures[0]["key"], self.cases[1]["key"])

    def test_model_mismatch_aborts_without_writing(self):
        with self.assertRaises(ValueError):
            run_judgments(self.cases, StubClient(model="other"), CONFIG, "mono", self.out, log=lambda _: None)
        self.assertEqual(load_jsonl(self.out), [])


if __name__ == "__main__":
    unittest.main()
