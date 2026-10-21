"""Offline tests for exp 05 (no API calls, no keys): python -m unittest discover -s tests

They run on the toy profile (data/evals/toy/) with the agent and judge replaced by fakes. Files are
written to a temporary workspace, never to output/.
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from agent.agent import AgentTrace, ToolCall
from experiments.exp_05_self_improvement_loop import (
    candidates as cand, evaluate, final_eval, independent_grader as ig, loop, make_splits, models, packet,
)
from experiment_profiles import load_profile
from experiments.exp_05_self_improvement_loop.lint import lint
from experiments.exp_05_self_improvement_loop.proposer import StubProposer, parse_proposal_json
from experiments.exp_05_self_improvement_loop.workspace import Workspace


def make_ws(tmp: str, test_profile: str | None = "toy_test") -> Workspace:
    ws = Workspace("toy", test_profile, root=Path(tmp))
    ws.root.mkdir(parents=True, exist_ok=True)
    ws.splits_path.write_text(json.dumps(make_splits.make_splits(ws.profile_cases("toy"))), encoding="utf-8")
    return ws


BASE = Workspace("toy").candidate(cand.BASELINE)


def rec(qid, repeat, success, candidate=cand.BASELINE, sha=None, branch="B"):
    """A minimal attempt record: enough for scoring, cost, the packet and the independent grader."""
    return {"id": qid, "repeat": repeat, "success": success, "candidate": candidate,
            "candidate_sha256": sha or BASE.sha256, "question": f"question for {qid}",
            "agent_sql": "SELECT 1", "branch": branch, "verdict": "ACCEPTABLE" if success else "WRONG",
            "reason": "r", "ex_match": bool(success), "unresolved": False,
            "judge_usage": {"input_tokens": 1000, "output_tokens": 500},
            "trace": {"model": models.AGENT["id"], "total_input_tokens": 2000, "total_output_tokens": 200,
                      "tool_calls": [{"tool_name": "run_query", "parameters": {"sql": "SELECT 1"},
                                      "result": {"rows": [{"x": 1}], "row_count": 1}, "call_index": 0}],
                      "final_answer": "a", "error": None}}


class Splits(unittest.TestCase):
    def test_deterministic_disjoint_and_complete(self):
        cases = Workspace("toy").profile_cases("toy")
        a, b = make_splits.make_splits(cases), make_splits.make_splits(cases)
        self.assertEqual(a, b)
        self.assertFalse(set(a["dev"]) & set(a["validation"]))
        self.assertEqual(set(a["dev"]) | set(a["validation"]), {c["id"] for c in cases})
        self.assertEqual((len(a["dev"]), len(a["validation"])), (2, 2))

    def test_balances_by_baseline_pass_count(self):
        cases = [{"id": f"q{i}", "difficulty": "easy"} for i in range(8)]
        attempts = [{"id": f"q{i}", "success": int(i < 4)} for i in range(8)]  # q0-q3 pass, q4-q7 fail
        s = make_splits.make_splits(cases, attempts)
        for side in ("dev", "validation"):
            self.assertEqual(sum(int(q[1:]) < 4 for q in s[side]), 2)

    def test_attempts_must_cover_all_questions(self):
        with self.assertRaises(ValueError):
            make_splits.make_splits([{"id": "a"}, {"id": "b"}], [{"id": "a", "success": 1}])


class Lint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = Workspace("toy").profile_cases("toy")
        cls.base = BASE

    def _with(self, extra: str) -> cand.Candidate:
        return cand.Candidate("x", self.base.prompt + "\n" + extra, dict(self.base.descriptions))

    def test_shipped_candidates_pass(self):
        self.assertTrue(lint(BASE, self.cases).ok)

    def test_baseline_is_the_profiles_agent_prompt(self):
        self.assertEqual(BASE.prompt, load_profile("toy").agent_prompt)
        self.assertEqual(BASE.descriptions, cand.ORIGINAL_DESCRIPTIONS)

    def test_rejects_question_ids(self):
        self.assertFalse(lint(self._with("See toy_easy_001 for an example."), self.cases).ok)

    def test_rejects_grader_mentions(self):
        self.assertFalse(lint(self._with("The judge prefers short answers."), self.cases).ok)

    def test_rejects_copied_question_words(self):
        self.assertFalse(lint(self._with(self.cases[0]["question"]), self.cases).ok)

    def test_rejects_gold_sql_fragment(self):
        gold = max((c["gold_sql"] for c in self.cases), key=len)
        self.assertFalse(lint(self._with(gold[:80]), self.cases).ok)

    def test_rejects_database_value_literals(self):
        self.assertFalse(lint(self._with("Filter with status = 'governed'."), self.cases).ok)

    def test_length_limit_is_absolute(self):
        advice = "Check the data before filtering. "
        self.assertTrue(lint(self._with(advice * 40), self.cases).ok)
        self.assertFalse(lint(self._with(advice * 75), self.cases).ok)

    def test_accepts_general_advice(self):
        self.assertTrue(lint(self._with("Look at the distinct values of a column before filtering on it."),
                             self.cases).ok)


def _table(split_scores: dict) -> dict:
    return {s: [rec(q, r + 1, v[r]) for q, v in sc.items() for r in range(len(v))] for s, sc in split_scores.items()}


class KeepRule(unittest.TestCase):
    base = {"dev": {"a": [1, 1, 1], "b": [0, 0, 0], "c": [1, 0, 1]},
            "validation": {"d": [1, 1, 1], "e": [0, 0, 0], "f": [1, 0, 0], "g": [0, 0, 0]}}

    def test_keep(self):
        new = {"dev": dict(self.base["dev"]), "validation": {**self.base["validation"], "e": [1, 1, 0], "g": [1, 0, 0]}}
        self.assertTrue(loop.decide(_table(self.base), _table(new))["keep"])

    def test_reject_small_validation_gain(self):
        filler = {f"z{i}": [1, 1, 1] for i in range(20)}  # one extra pass is then +1.2 points, below the margin
        base = {"dev": self.base["dev"], "validation": {**self.base["validation"], **filler}}
        new = {"dev": dict(base["dev"]), "validation": {**base["validation"], "e": [1, 0, 0]}}
        d = loop.decide(_table(base), _table(new))
        self.assertFalse(d["keep"])
        self.assertTrue(any("validation" in r for r in d["reasons"]))

    def test_reject_dev_drop(self):
        new = {"dev": {"a": [1, 1, 1], "b": [0, 0, 0], "c": [0, 0, 0]},
               "validation": {**self.base["validation"], "e": [1, 1, 1], "g": [1, 1, 1]}}
        d = loop.decide(_table(self.base), _table(new))
        self.assertFalse(d["keep"])
        self.assertTrue(any("dev" in r for r in d["reasons"]))

    def test_reject_two_broken_questions(self):
        new = {"dev": {"a": [0, 0, 0], "b": [1, 1, 1], "c": [1, 0, 1]},
               "validation": {"d": [0, 0, 0], "e": [1, 1, 1], "f": [1, 1, 1], "g": [1, 1, 1]}}
        d = loop.decide(_table(self.base), _table(new))
        self.assertEqual(d["broken"], ["a", "d"])
        self.assertFalse(d["keep"])

    def test_rejects_unpaired_cells(self):
        inc = {"dev": [rec("a", 1, 0), rec("a", 2, 0)], "validation": [rec("d", 1, 0)]}
        new = {"dev": [rec("a", 1, 1)], "validation": [rec("d", 1, 1)]}
        with self.assertRaises(ValueError):
            loop.decide(inc, new)


class Proposer(unittest.TestCase):
    def test_parse_valid_and_invalid(self):
        ok = json.dumps({"summary": "s", "prompt": "p", "tools": {"run_query": "a", "get_sample_rows": "b"}})
        self.assertEqual(parse_proposal_json("```json\n" + ok + "\n```")["prompt"], "p")
        with self.assertRaises(ValueError):
            parse_proposal_json(json.dumps({"summary": "s", "prompt": "p", "tools": {"run_query": "a"}}))
        with self.assertRaises(ValueError):
            parse_proposal_json("no json here")


class Packet(unittest.TestCase):
    def test_packet_has_no_gold_sql_and_one_attempt_per_question(self):
        ws = Workspace("toy")
        cases = {c["id"]: c for c in ws.profile_cases("toy")}
        dev = [rec(q, r, 0) for q in cases for r in (1, 2, 3)]
        chosen = packet.select_failures(dev, seed=1, k=50)
        self.assertEqual(len({r["id"] for r in chosen}), len(chosen))
        text = packet.render_packet(chosen)
        for c in cases.values():
            self.assertNotIn(c["gold_sql"], text)


class WorkspaceAndManifest(unittest.TestCase):
    def test_test_set_is_locked_at_init_and_changes_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            with self.assertRaises(SystemExit):  # not locked yet
                ws.cases("test")
            with contextlib.redirect_stdout(io.StringIO()):
                loop.write_manifest(ws)
                loop.write_manifest(ws)  # idempotent
            self.assertEqual(len(ws.cases("test")), 4)
            loop.verify_manifest(ws)
            saved = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
            saved["test_set_sha256"] = "0" * 64
            ws.manifest_path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ws.cases("test")
            with self.assertRaises(SystemExit):
                loop.verify_manifest(ws)

    def test_verify_refuses_changed_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp, test_profile=None)
            with contextlib.redirect_stdout(io.StringIO()):
                loop.write_manifest(ws)
            loop.verify_manifest(ws)
            saved = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
            saved["inputs_sha256"]["splits.json"] = "0" * 64
            ws.manifest_path.write_text(json.dumps(saved), encoding="utf-8")
            with self.assertRaises(SystemExit):
                loop.verify_manifest(ws)

    def test_splits_cover_the_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            ids = {c["id"] for s in ("dev", "validation") for c in ws.cases(s)}
            self.assertEqual(ids, {c["id"] for c in ws.cases("regression")})
            self.assertEqual(set(ws.clusters("regression")), ids)


class FakeEvaluation(unittest.TestCase):
    """Evaluator.attempt with the agent and judge faked, against the real toy database and gold SQL."""

    @staticmethod
    def fake_agent(sql_for):
        def run(question, entry, toolbox, system_prompt, max_tool_calls=12):
            sql = sql_for(question)
            trace = AgentTrace(question=question, model=entry["id"], final_answer="done",
                               total_llm_calls=2, total_input_tokens=100, total_output_tokens=10)
            trace.tool_calls.append(ToolCall("run_query", {"sql": sql}, toolbox.run_query(sql), 0))
            return trace
        return run

    def evaluator(self, ws, sql_for, verdict):
        with mock.patch.object(evaluate, "get_chat_model", return_value=None):
            ev = evaluate.Evaluator(ws, BASE)
        ev._judge = lambda prompt: {"verdict": verdict, "reason": "fake", "usage": {"input_tokens": 5, "output_tokens": 1}}
        return ev, mock.patch.object(evaluate, "run_agent_loop", self.fake_agent(sql_for))

    def test_ex_match_goes_to_branch_a_and_a_miss_to_branch_b(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            case = ws.cases("regression")[0]
            ev, patch = self.evaluator(ws, lambda q: case["gold_sql"], "CORRECT")
            with patch:
                good = ev.attempt(case, 1)
            self.assertEqual((good["branch"], good["success"], good["ex_match"]), ("A", 1, True))
            ev, patch = self.evaluator(ws, lambda q: "SELECT 1 AS x", "WRONG")
            with patch:
                bad = ev.attempt(case, 1)
            self.assertEqual((bad["branch"], bad["success"], bad["ex_match"]), ("B", 0, False))
            ev, patch = self.evaluator(ws, lambda q: "SELECT 1 AS x", "ACCEPTABLE")
            with patch:
                self.assertEqual(ev.attempt(case, 1)["success"], 1)

    def test_unparseable_verdict_fails_and_is_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            case = ws.cases("regression")[0]
            ev, patch = self.evaluator(ws, lambda q: case["gold_sql"], None)
            with patch:
                r = ev.attempt(case, 1)
            self.assertEqual((r["success"], r["unresolved"]), (0, True))

    def test_run_resumes_and_summarizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            by_q = {c["question"]: c["gold_sql"] for c in ws.cases("regression")}
            ev, patch = self.evaluator(ws, lambda q: by_q[q], "CORRECT")
            path = ws.root / "runs" / "r.jsonl"
            with patch, contextlib.redirect_stdout(io.StringIO()):
                first = evaluate.run(ws, BASE, "regression", 2, path, workers=1, limit=2,
                                     evaluator=ev)
                again = evaluate.run(ws, BASE, "regression", 2, path, workers=1, limit=3,
                                     evaluator=ev)
            self.assertEqual((len(first), len(again)), (4, 6))  # the 4 finished attempts are not rerun
            s = evaluate.summarize(again)
            self.assertEqual((s["attempts"], s["success_rate"], s["ex_rate"]), (6, 1.0, 1.0))
            self.assertIsNotNone(s["cost_usd"])


class LoopEndToEnd(unittest.TestCase):
    """Fake evaluator derived from synthetic baseline records; no API calls."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.ws = make_ws(self.tmp.name)
        self.logs: list[str] = []
        splits = self.ws.splits()
        for split in loop.SPLITS:  # baseline: every dev question fails, every validation question fails
            rows = [rec(q, r, 0) for q in splits[split] for r in (1, 2, 3)]
            path = self.ws.loop_dir / f"{loop.BASELINE}__{split}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def tearDown(self):
        self.tmp.cleanup()

    def _evaluate(self, flip_val: bool):
        def fn(c, split, repeats, out_path, limit):
            rows = copy.deepcopy(loop.load_jsonl(self.ws.loop_dir / f"{loop.BASELINE}__{split}.jsonl"))
            for r in rows:
                r["candidate"], r["candidate_sha256"] = c.name, c.sha256
                if split == "validation" and flip_val:
                    r["success"] = 1
            out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return rows
        return fn

    def _loop(self, flip_val, proposer=None, evaluate_fn=None, **kw):
        return loop.Loop(self.ws, proposer or StubProposer(), evaluate_fn or self._evaluate(flip_val),
                         out=self.logs.append, **kw)

    def test_keeps_a_real_gain_and_logs(self):
        lp = self._loop(True, max_proposals=1)
        lp.run()
        st = lp.state()
        self.assertEqual(st["incumbent"], "c002")
        self.assertTrue(st["decisions"][1]["keep"])
        self.assertTrue((self.ws.candidates_dir / "c002" / "prompt.md").exists())
        self.assertEqual(loop.winner(self.ws), "c002")

    def test_rejects_when_no_gain(self):
        lp = self._loop(False, max_proposals=1)
        lp.run()
        self.assertEqual(lp.state()["incumbent"], loop.BASELINE)
        self.assertFalse(lp.state()["decisions"][1]["keep"])

    def test_budget_guard_stops_before_spending(self):
        lp = self._loop(True, max_proposals=5, cap=0.001)
        lp.run()
        self.assertEqual(lp.state()["proposals"], {})
        self.assertTrue(any("stopping: spent" in m for m in self.logs))

    def test_proposal_limit_and_resume(self):
        self._loop(False, max_proposals=2).run()
        lp2 = self._loop(False, max_proposals=2)  # fresh object, same log: nothing more to do
        lp2.run()
        self.assertEqual(len(lp2.state()["proposals"]), 2)

    def test_lint_failure_counts_and_is_not_evaluated(self):
        class Bad(StubProposer):
            def propose(self, *a, **k):
                p = super().propose(*a, **k)
                p.prompt += "\nUse status = 'governed'. See toy_easy_001."
                return p

        lp = self._loop(False, proposer=Bad(), max_proposals=1)
        lp.run()
        self.assertTrue(lp.state()["decisions"][1].get("lint_failed"))
        self.assertFalse((self.ws.loop_dir / "c002__dev.jsonl").exists())

    def test_incomplete_evaluation_has_no_decision_and_can_resume(self):
        complete = self._evaluate(True)
        calls = 0

        def interrupted(c, split, repeats, out_path, limit):
            nonlocal calls
            rows = complete(c, split, repeats, out_path, limit)
            calls += 1
            if calls == 2:
                rows = rows[:-1]
                out_path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            return rows

        with self.assertRaisesRegex(RuntimeError, "missing=1"):
            self._loop(True, evaluate_fn=interrupted, max_proposals=1).run()
        resumed = self._loop(True, evaluate_fn=complete, max_proposals=1)
        self.assertEqual(resumed.state()["decisions"], {})
        resumed.run()
        self.assertTrue(resumed.state()["decisions"][1]["keep"])

    def test_duplicate_attempt_has_no_decision(self):
        complete = self._evaluate(True)

        def duplicated(c, split, repeats, out_path, limit):
            rows = complete(c, split, repeats, out_path, limit)
            return rows + [rows[0]] if split == "validation" else rows

        lp = self._loop(True, evaluate_fn=duplicated, max_proposals=1)
        with self.assertRaisesRegex(RuntimeError, "duplicates=1"):
            lp.run()
        self.assertEqual(lp.state()["decisions"], {})

    def test_final_eval_needs_a_kept_candidate(self):
        self._loop(False, max_proposals=1).run()
        with self.assertRaises(SystemExit):
            final_eval.arms(self.ws)

    def test_final_eval_report_over_baseline_and_winner(self):
        self._loop(True, max_proposals=1).run()  # c002 becomes the winner
        manifest_ok = contextlib.redirect_stdout(io.StringIO())
        with manifest_ok:
            loop.write_manifest(self.ws)
        for split, repeats in final_eval.REPEATS.items():
            ids = [c["id"] for c in self.ws.cases(split)]
            for i, arm in enumerate(final_eval.arms(self.ws)):
                rows = [rec(q, r, int(i == 1), candidate=cand.BASELINE)
                        for q in ids for r in range(1, repeats + 1)]
                path = final_eval.path_for(self.ws, arm, split)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            final_eval.report(self.ws)
        text = out.getvalue()
        self.assertIn("== test (5 repeats; clusters resampled by family)", text)
        self.assertIn("c002 - c000_baseline: +100.0 points", text)
        self.assertNotIn("c001", text)
        self.assertNotIn("WARNING", text)


class IndependentGrader(unittest.TestCase):
    def test_parse_verdict_handles_markdown_and_headings(self):
        self.assertEqual(ig.parse_verdict("analysis\n**VERDICT:** CORRECT\nREASON: fine")["verdict"], "CORRECT")
        self.assertEqual(ig.parse_verdict("## Verdict\nVERDICT: WRONG\n...\nVERDICT: ACCEPTABLE")["verdict"], "ACCEPTABLE")
        self.assertIsNone(ig.parse_verdict("## Verdict\nlooks fine")["verdict"])

    def test_calibration_summary_and_threshold(self):
        recs = [{"branch": "A", "verdict": "CORRECT", "human": 1, "model": "m"},
                {"branch": "A", "verdict": "FALSE_POSITIVE", "human": 0, "model": "m"},
                {"branch": "B", "verdict": "ACCEPTABLE", "human": 0, "model": "m"},
                {"branch": "B", "verdict": None, "human": 1, "model": "m"}]
        s = ig.calibration_summary("m", recs)
        self.assertEqual((s["resolved"], s["unresolved"]), (3, 1))
        self.assertAlmostEqual(s["agreement"], 2 / 3)
        self.assertEqual(s["false_accept_rate"], 0.5)
        self.assertFalse(s["meets_threshold"])

    def test_test_set_labels_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            labels = Path(tmp) / "labels.jsonl"
            labels.write_text(json.dumps({"candidate": "c000_baseline", "split": "test", "id": "x", "repeat": 1,
                                          "label": "correct"}) + "\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                ig.cmd_calibrate(ws, "m:x", "x", labels, 1, 100)

    def test_regrade_refuses_uncalibrated_or_below_bar_models(self):
        with tempfile.TemporaryDirectory() as tmp:
            ws = make_ws(tmp)
            with self.assertRaises(SystemExit):
                ig.cmd_regrade(ws, "m:x", "x", 1, 100)
            ws.independent_dir.mkdir(parents=True)
            (ws.independent_dir / "calibration_x.json").write_text(json.dumps({"meets_threshold": False}), encoding="utf-8")
            with self.assertRaises(SystemExit):
                ig.cmd_regrade(ws, "m:x", "x", 1, 100)

    def test_label_for(self):
        self.assertEqual(ig.label_for("bedrock_converse:moonshot.kimi-k2-thinking"), "moonshot.kimi-k2-thinking")


class Bootstrap(unittest.TestCase):
    def test_ci_contains_the_point_estimate(self):
        import random

        p, lo, hi = final_eval.boot_ci([[1.0, 0.0], [1.0, 0.5], [0.5, 0.5], [1.0, 0.0]], random.Random(1))
        self.assertLessEqual(lo, p)
        self.assertGreaterEqual(hi, p)
        self.assertAlmostEqual(p, 0.625)


if __name__ == "__main__":
    unittest.main()
