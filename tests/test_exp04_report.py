from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from experiments.exp_04_jev_judge.report import (
    agreement, build_report, cascade_predictions, classification_metrics, format_markdown,
    resolve_labels, write_label_template,
)


def _row(id_, branch, gpt, verdict, confidence=0.9, after=None, profile="toy"):
    def call(v, c):
        return {"verdict": v, "confidence": c, "usage": {"input_tokens": 1000}, "latency_ms": 200.0}
    row = {"key": f"{id_}|{profile}|0", "id": id_, "profile": profile, "branch": branch,
           "gpt_judge": gpt, "mono": call(verdict, confidence), "question": "q",
           "gold_sql": "g", "agent_sql": "a"}
    if after:
        row["narrow"] = {"usage": {"input_tokens": 500}, "latency_ms": 100.0}
        row["after"] = call(*after)
    return row


ROWS = [
    _row("a", "A", "CORRECT", "CORRECT", 0.95),
    _row("b", "A", "FALSE_POSITIVE", "CORRECT", 0.6),          # Jev misses a latent bug
    _row("c", "B", "ACCEPTABLE", "INCORRECT", 0.55),           # Jev rejects a valid alternative
    _row("d", "B", "WRONG", "INCORRECT", 0.99),
    _row("e", "B", "GOLD_ERROR", "CANNOT_DETERMINE", 0.4),     # both abstain
]
LABELS = [{"id": "a", "label": "correct"}, {"id": "b", "label": "incorrect"},
          {"id": "c", "label": "correct"}, {"id": "d", "label": "incorrect"},
          {"id": "e", "label": "cannot_determine"}]


class MetricsTest(unittest.TestCase):
    def test_abstention_lowers_coverage_not_accuracy(self):
        reference = {"x": "correct", "y": "incorrect", "z": "correct"}
        metrics = classification_metrics(reference, {"x": 1, "y": 1, "z": None})
        self.assertEqual((metrics["covered"], metrics["unresolved"]), (2, 1))
        self.assertAlmostEqual(metrics["coverage"], 2 / 3)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertEqual((metrics["false_accepts"], metrics["false_rejects"]), (1, 0))

    def test_cannot_determine_labels_are_excluded(self):
        metrics = classification_metrics({"x": "cannot_determine", "y": "correct"}, {"x": 1, "y": 1})
        self.assertEqual(metrics["labeled"], 1)

    def test_agreement_ignores_abstentions(self):
        self.assertEqual(agreement({"a": 1, "b": 0, "c": None}, {"a": 1, "b": 1, "c": 1}),
                         {"n": 2, "agree": 1, "rate": 0.5})

    def test_resolve_labels_key_beats_id_and_rejects_bad_label(self):
        rows = [ROWS[0]]
        labels = [{"id": "a", "label": "correct"}, {"key": "a|toy|0", "label": "incorrect"}]
        self.assertEqual(resolve_labels(rows, labels), {"a|toy|0": "incorrect"})
        with self.assertRaises(ValueError):
            resolve_labels(rows, [{"id": "a", "label": "maybe"}])
        self.assertEqual(resolve_labels(rows, [{"id": "a", "label": ""}]), {})


class CascadeTest(unittest.TestCase):
    def test_low_confidence_and_abstentions_take_the_gpt_verdict(self):
        predictions, escalated = cascade_predictions(ROWS, "mono", 0.8)
        # a: confident keep (1). b: 0.6 -> GPT FALSE_POSITIVE (0). c: 0.55 -> GPT ACCEPTABLE (1).
        # d: confident keep (0). e: abstained -> GPT GOLD_ERROR (None).
        self.assertEqual(predictions, {"a|toy|0": 1, "b|toy|0": 0, "c|toy|0": 1, "d|toy|0": 0, "e|toy|0": None})
        self.assertAlmostEqual(escalated, 3 / 5)

    def test_threshold_zero_never_escalates_confident_verdicts(self):
        predictions, escalated = cascade_predictions(ROWS, "mono", 0.0)
        self.assertAlmostEqual(escalated, 1 / 5)  # only the abstention
        self.assertEqual(predictions["b|toy|0"], 1)


class ReportTest(unittest.TestCase):
    def test_without_labels(self):
        report = build_report(ROWS)
        self.assertEqual(list(report["scorers"]), ["EX", "GPT judge", "Jev (monolithic)"])
        self.assertAlmostEqual(report["scorers"]["EX"]["pass_rate"], 2 / 5)
        self.assertEqual(report["scorers"]["Jev (monolithic)"]["abstained"], 1)
        self.assertEqual(report["agreement_with_gpt"]["EX"]["n"], 4)
        self.assertNotIn("vs_labels", report["scorers"]["EX"])
        self.assertIn("Pass rate by agent profile", format_markdown(report))

    def test_with_labels(self):
        report = build_report(ROWS, LABELS, price_per_million=1.0)
        self.assertEqual(report["labeled"], 5)
        jev = report["scorers"]["Jev (monolithic)"]["vs_labels"]
        self.assertEqual((jev["false_accepts"], jev["false_rejects"]), (1, 1))
        self.assertEqual(report["scorers"]["EX"]["vs_labels"]["labeled"], 4)
        self.assertAlmostEqual(report["scorers"]["Jev (monolithic)"]["usd_per_1000"], 1.0)
        self.assertIn("False accepts", format_markdown(report))

    def test_two_call_column_appears_only_when_present(self):
        rows = [_row("a", "A", "CORRECT", "CORRECT", after=("INCORRECT", 0.7)),
                _row("b", "B", "WRONG", "INCORRECT", after=("INCORRECT", 0.9))]
        report = build_report(rows, price_per_million=1.0)
        self.assertIn("Jev (two-call)", report["scorers"])
        self.assertEqual(report["scorers"]["Jev (two-call)"]["pass_rate"], 0.0)
        # two-call cost = narrow (500) + after (1000) input tokens
        self.assertAlmostEqual(report["scorers"]["Jev (two-call)"]["usd_per_1000"], 1.5)
        self.assertNotIn("Jev (two-call)", build_report(ROWS)["scorers"])

    def test_empty_rows(self):
        with self.assertRaises(ValueError):
            build_report([])


class LabelTemplateTest(unittest.TestCase):
    def test_template_written_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "labels.jsonl"
            write_label_template(ROWS, path)
            lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(len(lines), 5)
            self.assertEqual(lines[0]["label"], "")
            self.assertEqual(lines[0]["agent_sql"], "a")
            with self.assertRaises(SystemExit):
                write_label_template(ROWS, path)


if __name__ == "__main__":
    unittest.main()
