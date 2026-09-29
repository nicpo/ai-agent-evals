"""Build the EX x LLM-judge disagreement matrix from a finished run.

Method (see the linked post): don't collapse EX and the judge into one
adjusted score. Instead look at the four cells of (EX pass/fail) x (judge
accept/reject):

                   EX pass              EX fail
    Judge accepts  expected_success     valid_alternative
    Judge rejects  latent_bug           expected_failure

The diagonal is expected and mostly uninteresting. The two off-diagonal cells
are where an eval starts teaching you something:

* latent_bug: EX passed on this database, but the judge thinks the SQL logic
  would break on plausible data it hasn't seen -- inspect for a missing
  counterexample in the eval data.
* valid_alternative: EX failed under a strict comparator, but the judge
  thinks the answer is acceptable -- inspect the comparator policy, the gold
  SQL, or the output contract.

Cases where the judge returned GOLD_ERROR/HUMAN_REVIEW land in a separate
``review`` bucket (needs a human, not a grader fix).

Usage:

    python -m experiments.exp_01_graders_disagreement.analyze output/results/output_*.jsonl
    python -m experiments.exp_01_graders_disagreement.analyze output/results/output_*.jsonl --markdown
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import load_records
from eval.runner import load_questions

_CELLS = ("expected_success", "latent_bug", "valid_alternative", "expected_failure")


def _pred_sql(record: dict) -> str | None:
    queries = [tc for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"]
    return queries[-1]["parameters"].get("sql") if queries else None


def _classify(record: dict, cases_by_id: dict[str, dict]) -> tuple[str | None, dict]:
    """Return (cell name or None, case detail dict) for one graded record."""
    ex = (record["graders"].get("tier2b") or {}).get("execution_accuracy")
    tier5 = record["graders"].get("tier5") or {}
    if ex is None or tier5.get("branch") is None:
        return None, {}

    case = cases_by_id.get(record.get("id")) or {}
    detail = {
        "id": record.get("id"),
        "question": record.get("question"),
        "gold_sql": case.get("gold_sql"),
        "agent_sql": _pred_sql(record),
        "ex_match": ex.get("match"),
        "ex_reason": ex.get("reason"),
    }

    if tier5["branch"] == "A":
        verdict = tier5.get("false_positive_verdict")
        detail["verdict"] = verdict
        detail["reason"] = tier5.get("false_positive_reason")
        if verdict == "CORRECT":
            return "expected_success", detail
        if verdict == "FALSE_POSITIVE":
            return "latent_bug", detail
        return None, detail

    verdict = tier5.get("false_negative_verdict")
    detail["verdict"] = verdict
    detail["reason"] = tier5.get("false_negative_reason")
    if verdict == "ACCEPTABLE":
        return "valid_alternative", detail
    if verdict == "WRONG":
        return "expected_failure", detail
    if verdict in {"GOLD_ERROR", "HUMAN_REVIEW"}:
        return "review", detail
    return None, detail


def build_matrix(records: list[dict], cases_by_id: dict[str, dict] | None = None) -> dict:
    """Classify every graded record into one of the disagreement-matrix cells.

    ``cases_by_id`` supplies ``gold_sql`` for the case-card output (results
    records don't store it, only the judge sees it at run time); pass ``{}``
    if the eval source isn't available and case cards will just omit it.
    """
    cases_by_id = cases_by_id or {}
    buckets: dict[str, list[dict]] = {cell: [] for cell in (*_CELLS, "review")}
    n_judged = 0
    for record in records:
        cell, detail = _classify(record, cases_by_id)
        if cell is None and not detail:
            continue  # no EX outcome or judge branch at all
        n_judged += 1
        if cell:
            buckets[cell].append(detail)

    n = n_judged
    matrix = {
        cell: {"count": len(buckets[cell]), "rate": round(len(buckets[cell]) / n, 4) if n else None}
        for cell in _CELLS
    }
    matrix["review"] = {
        "count": len(buckets["review"]),
        "rate": round(len(buckets["review"]) / n, 4) if n else None,
    }
    return {
        "n_judged": n_judged,
        "matrix": matrix,
        "cases": {cell: buckets[cell] for cell in ("latent_bug", "valid_alternative")},
        "all_cases": buckets,
    }


def format_matrix(report: dict) -> str:
    m = report["matrix"]
    n = report["n_judged"]

    def cell(name: str) -> str:
        c = m[name]
        return f"{c['count']}/{n} ({round(100 * c['rate']) if c['rate'] is not None else 0}%)"

    lines = [
        "=" * 60,
        f"EX x JUDGE DISAGREEMENT MATRIX  (n={n} judged cases)",
        "=" * 60,
        "",
        f"{'':18}{'EX pass':<22}{'EX fail':<22}",
        f"{'Judge accepts':18}{cell('expected_success'):<22}{cell('valid_alternative'):<22}",
        f"{'Judge rejects':18}{cell('latent_bug'):<22}{cell('expected_failure'):<22}",
        "",
        f"  review (GOLD_ERROR / HUMAN_REVIEW): {cell('review')}",
        "",
        "Off-diagonal cells are the interesting ones:",
        f"  latent_bug         -- EX passed, judge thinks the SQL logic is wrong: {cell('latent_bug')}",
        f"  valid_alternative  -- EX failed, judge thinks the answer is fine:     {cell('valid_alternative')}",
        "=" * 60,
    ]
    return "\n".join(lines)


def format_case_cards(report: dict) -> str:
    lines = []
    for cell in ("latent_bug", "valid_alternative"):
        cases = report["cases"][cell]
        if not cases:
            continue
        lines.append(f"## {cell} ({len(cases)} case(s))\n")
        for c in cases:
            lines += [
                f"### {c['id']}",
                f"- **Question:** {c['question']}",
                f"- **Gold SQL:** `{c.get('gold_sql')}`",
                f"- **Agent SQL:** `{c['agent_sql']}`",
                f"- **EX:** match={c['ex_match']} reason={c['ex_reason']}",
                f"- **Judge verdict:** {c['verdict']}",
                f"- **Judge reason:** {c['reason']}",
                "",
            ]
    return "\n".join(lines) if lines else "(no off-diagonal cases in this run)\n"


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", help="Path to a results JSONL from eval.runner / this experiment's run.py.")
    p.add_argument("--out", type=Path, help="Where to write the JSON report (default: alongside the results file).")
    p.add_argument("--markdown", action="store_true", help="Also print off-diagonal cases as markdown case cards.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    records = load_records(args.results)
    # Recursive glob so this works whether the run used the canonical eval set
    # or the toy set (which lives in a subdirectory precisely so it's excluded
    # from the canonical glob by default -- see eval.runner.load_questions).
    cases_by_id = {c["id"]: c for c in load_questions(glob="**/questions-*.jsonl")}
    report = build_matrix(records, cases_by_id)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(format_matrix(report))
    if args.markdown:
        print()
        print(format_case_cards(report))

    out_path = args.out or Path(args.results).with_suffix("").with_name(
        Path(args.results).stem + ".disagreement.json"
    )
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
