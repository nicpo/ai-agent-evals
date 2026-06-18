"""Aggregate metrics over a run's per-question result records.

A "record" is one entry as written by ``eval.runner.build_result_record``:
``{"id", "question", "difficulty", "adversarial", "trace", "graders"}``.

``compute_metrics`` is pure (no I/O) so it can be reused to recompute a summary
from an existing results JSONL:

    python -m eval.metrics output/results/output_2026-06-15_12-00.jsonl

Each bundle carries both raw counts and rates, with explicit ``n`` denominators
(they differ per metric: EX needs gold SQL, judge metrics need the judge to have
run, etc.). ``format_summary`` renders the annotated console report.
"""

from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

DIFFICULTIES = ["easy", "medium", "hard", "extra-hard"]


# --- small helpers -----------------------------------------------------------

def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def _mean(values) -> float | None:
    vals = [v for v in values if v is not None]
    return round(statistics.fmean(vals), 4) if vals else None


def _tier(record: dict, name: str) -> dict:
    return record["graders"].get(name) or {}


def _ex(record: dict) -> dict | None:
    """The execution_accuracy block, or None if EX was not evaluated."""
    return _tier(record, "tier2b").get("execution_accuracy")


def _esm(record: dict) -> dict | None:
    """The exact_set_match block, or None if not computed."""
    return _tier(record, "tier2b").get("exact_set_match")


def _run_query_count(record: dict) -> int:
    return sum(
        1 for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"
    )


def _branch_a(record: dict):
    return _tier(record, "tier5").get("false_positive_verdict")


def _branch_b(record: dict):
    return _tier(record, "tier5").get("false_negative_verdict")


def _correctness(records: list[dict]) -> dict:
    """EX- and judge-verdict-based correctness. Denominator: cases with an EX outcome."""
    evaluated = [r for r in records if _ex(r) is not None]
    n = len(evaluated)
    ex_pass = sum(1 for r in evaluated if _ex(r).get("match"))
    acceptable = sum(1 for r in evaluated if _branch_b(r) == "ACCEPTABLE")
    wrong = sum(1 for r in evaluated if _branch_b(r) == "WRONG")
    false_pos = sum(1 for r in evaluated if _branch_a(r) == "FALSE_POSITIVE")
    review = sum(1 for r in evaluated if _branch_b(r) in ("GOLD_ERROR", "HUMAN_REVIEW"))
    adjusted = ex_pass + acceptable
    return {
        "n": n,
        "ex_pass": ex_pass, "ex_pass_rate": _rate(ex_pass, n),
        "acceptable": acceptable,
        "adjusted_correct": adjusted, "adjusted_correct_rate": _rate(adjusted, n),
        "true_error": wrong, "true_error_rate": _rate(wrong, n),
        "false_positive": false_pos, "false_positive_rate": _rate(false_pos, n),
        "human_review": review, "human_review_rate": _rate(review, n),
    }


def _tool_use(records: list[dict]) -> dict:
    n = len(records)
    schema_first = sum(1 for r in records if _tier(r, "tier1").get("schema_before_query"))
    multi = sum(1 for r in records if _run_query_count(r) > 1)
    self_corr = sum(1 for r in records if _tier(r, "tier1").get("self_corrected"))
    return {
        "n": n,
        "schema_first": schema_first, "schema_first_rate": _rate(schema_first, n),
        "multi_query": multi, "multi_query_rate": _rate(multi, n),
        "self_correction": self_corr, "self_correction_rate": _rate(self_corr, n),
        "mean_tool_calls": _mean(_tier(r, "tier1").get("tool_call_count") for r in records),
    }


_CLAUSES = ("from", "group_by", "order_by", "where", "select")


def _sql_quality(records: list[dict]) -> dict:
    tier2 = [r for r in records if r["graders"].get("tier2")]
    n_sql = len(tier2)
    parseable = sum(1 for r in tier2 if r["graders"]["tier2"].get("is_parseable"))
    halluc = sum(1 for r in tier2 if r["graders"]["tier2"].get("hallucinated_references"))

    esm = [_esm(r) for r in records if _esm(r)]
    n_esm = len(esm)
    clauses = {}
    for c in _CLAUSES:
        k = sum(1 for e in esm if e.get(c))
        clauses[c] = {"count": k, "rate": _rate(k, n_esm)}
    return {
        "n_sql": n_sql,
        "parseable": parseable, "parseable_rate": _rate(parseable, n_sql),
        "hallucination": halluc, "hallucination_rate": _rate(halluc, n_sql),
        "n_esm": n_esm,
        "clauses": clauses,
    }


def _branches(records: list[dict]) -> dict:
    """Tier 4 judge branch verdicts (A = EX passed, B = EX failed)."""
    a = [r for r in records if _tier(r, "tier5").get("branch") == "A"]
    b = [r for r in records if _tier(r, "tier5").get("branch") == "B"]
    n_a, n_b = len(a), len(b)
    false_pos = sum(1 for r in a if _branch_a(r) == "FALSE_POSITIVE")
    acceptable = sum(1 for r in b if _branch_b(r) == "ACCEPTABLE")
    wrong = sum(1 for r in b if _branch_b(r) == "WRONG")
    review = sum(1 for r in b if _branch_b(r) in ("GOLD_ERROR", "HUMAN_REVIEW"))
    return {
        "n_a": n_a, "false_positive": false_pos, "false_positive_rate": _rate(false_pos, n_a),
        "n_b": n_b,
        "acceptable": acceptable, "acceptable_rate": _rate(acceptable, n_b),
        "wrong": wrong, "wrong_rate": _rate(wrong, n_b),
        "review": review, "review_rate": _rate(review, n_b),
    }


def _answer_quality(records: list[dict]) -> dict:
    faith = [_tier(r, "tier5").get("faithfulness_score") for r in records]
    unc = [_tier(r, "tier5").get("uncertainty_score") for r in records]
    align = [_tier(r, "tier5").get("question_alignment_score") for r in records]
    recovery = [_tier(r, "tier5").get("error_recovery_score") for r in records]
    return {
        "n": sum(1 for v in faith if v is not None),
        "mean_faithfulness": _mean(faith),
        "mean_uncertainty": _mean(unc),
        "mean_alignment": _mean(align),
        "mean_error_recovery": _mean(recovery),
        "n_error_recovery": sum(1 for v in recovery if v is not None),
    }


def _efficiency(records: list[dict]) -> dict:
    return {
        "mean_tokens": _mean(r["trace"].get("total_tokens") for r in records),
        "mean_llm_calls": _mean(r["trace"].get("total_llm_calls") for r in records),
    }


def _bundle(records: list[dict]) -> dict:
    """The full metric bundle for a set of records (overall or one difficulty)."""
    return {
        "n": len(records),
        "correctness": _correctness(records),
        "tool_use": _tool_use(records),
        "sql_quality": _sql_quality(records),
        "branches": _branches(records),
        "answer_quality": _answer_quality(records),
        "efficiency": _efficiency(records),
    }


def _model_name(records: list[dict]) -> str:
    if not records:
        return ""
    model = records[0].get("trace", {}).get("model", "") or ""
    return model.split(":", 1)[-1]  # strip provider prefix (anthropic:/openai:)


def compute_metrics(records: list[dict]) -> dict:
    """Compute the full summary: overall bundle + by-difficulty + adversarial."""
    summary = _bundle(records)
    summary["model"] = _model_name(records)
    summary["by_difficulty"] = {
        d: _bundle(subset)
        for d in DIFFICULTIES
        if (subset := [r for r in records if r.get("difficulty") == d])
    }
    adversarial = [r for r in records if r.get("adversarial")]
    summary["by_adversarial"] = _correctness(adversarial) if adversarial else None
    return summary


# --- presentation ------------------------------------------------------------

_W = 60                       # report width
_DESC = 37                    # column where descriptions begin


def _pct(count: int, denom: int) -> str:
    return f"{round(100 * count / denom)}%" if denom else ""


def _crow(label: str, count: int, denom: int, desc: str, mark: bool = False) -> str:
    """A count/percent metric row."""
    frac = f"{count}/{denom}" if denom else "n/a"
    arrow = " ←" if mark else ""
    prefix = f"  {label:<20}{frac:>5}{_pct(count, denom):>6}{arrow}"
    return f"{prefix:<{_DESC}}  {desc}"


def _vrow(label: str, value: str, desc: str) -> str:
    """A single-value metric row (means, tokens)."""
    prefix = f"  {label:<20}{value:>11}"
    return f"{prefix:<{_DESC}}  {desc}"


def _clause_row(name: str, count: int, denom: int, desc: str, mark: bool = False) -> str:
    frac = f"{count}/{denom}" if denom else "n/a"
    arrow = "←" if mark else " "
    prefix = f"    {name:<9}{frac:>5}{_pct(count, denom):>6}  {arrow}"
    return f"{prefix:<{_DESC}}  {desc}"


def _score(v) -> str:
    return f"{v:.2f} / 5" if v is not None else "n/a"


_CLAUSE_DESC = {
    "from": ("FROM", "correct tables used"),
    "group_by": ("GROUP BY", "correct grouping"),
    "order_by": ("ORDER BY", "correct sort"),
    "where": ("WHERE", "correct filters"),
    "select": ("SELECT", "correct columns (extra columns often acceptable)"),
}


def format_summary(s: dict) -> str:
    c = s["correctness"]
    t = s["tool_use"]
    q = s["sql_quality"]
    b = s["branches"]
    a = s["answer_quality"]
    e = s["efficiency"]
    model = s.get("model") or (s.get("meta") or {}).get("agent_model") or "model"
    sep = "-" * _W
    L = [
        "=" * _W,
        f"RUN SUMMARY — {model} | {s['n']} questions",
        "=" * _W,
        "",
    ]

    # --- headline ----------------------------------------------------------
    n = c["n"]
    L.append(
        f"  Adjusted correct:  {c['adjusted_correct']}/{n}  ({_pct(c['adjusted_correct'], n)})"
        f"      True errors: {c['true_error']}/{n}  ({_pct(c['true_error'], n)})"
    )
    L.append("  (questions answered correctly after accounting for cosmetic SQL differences)")

    # --- Tier 1 ------------------------------------------------------------
    L += [
        "", sep,
        "TIER 1 - TOOL USE",
        "  Did the agent use tools in the right order and with the right strategy?",
        "",
        _crow("Schema called first", t["schema_first"], t["n"], "agent read schema before writing SQL"),
        _crow("Self-correction", t["self_correction"], t["n"], "agent fixed a failed query on retry"),
        _crow("Multi-query", t["multi_query"], t["n"], "agent ran >1 query without an error (self-refinement)"),
        _vrow("Mean tool calls", _fmt(t["mean_tool_calls"], 1), "total tool calls per question"),
    ]

    # --- Tier 2 ------------------------------------------------------------
    L += [
        "", sep,
        "TIER 2 - SQL STRUCTURE",
        "  Is the SQL syntactically valid and free of invented table/column names?",
        "",
        _crow("Parseable", q["parseable"], q["n_sql"], "SQL ran without a syntax error"),
        _crow("Hallucinated refs", q["hallucination"], q["n_sql"],
              "agent invented a table or column name not in schema"),
        "",
        f"  Clause match vs gold:            how closely each SQL clause matched the gold query",
    ]
    n_esm = q["n_esm"]
    if n_esm:
        rates = {k: (q["clauses"][k]["rate"] or 0) for k in _CLAUSES}
        min_rate = min(rates.values())
        mark_key = next(k for k in _CLAUSES if rates[k] == min_rate) if min_rate < 1 else None
        for key in sorted(_CLAUSES, key=lambda k: (-rates[k], list(_CLAUSES).index(k))):
            name, desc = _CLAUSE_DESC[key]
            marked = key == mark_key
            if marked:
                desc = f"{desc} - primary failure point"
            L.append(_clause_row(name, q["clauses"][key]["count"], n_esm, desc, marked))
    else:
        L.append("    (no gold SQL available to compare clauses)")

    # --- Tier 2b -----------------------------------------------------------
    L += [
        "", sep,
        "TIER 2b - EXECUTION",
        "  Did the agent's query return the same rows and values as the gold query?",
        "",
        _crow("EX pass rate", c["ex_pass"], n, "result sets matched exactly"),
        _crow("Adjusted correct", c["adjusted_correct"], n,
              f"adds {c['acceptable']} ACCEPTABLE (extra columns, valid aliases)"),
        _crow("True errors", c["true_error"], n, "Branch B WRONG"),
    ]

    # --- Tier 4A -----------------------------------------------------------
    L += [
        "", sep,
        f"TIER 4A - SQL LOGIC  (Branch A - EX passed, n={b['n_a']})",
        "  When EX passed, did the SQL get the right answer for the right reasons?",
        "",
        _crow("False positive rate", b["false_positive"], b["n_a"],
              "EX passed but SQL logic was subtly wrong"),
    ]

    # --- Tier 4B -----------------------------------------------------------
    L += [
        "", sep,
        f"TIER 4B - FAILURE TRIAGE  (Branch B - EX failed, n={b['n_b']})",
        "  When EX failed, was it a real error or a cosmetic difference?",
        "",
        _crow("ACCEPTABLE", b["acceptable"], b["n_b"], "EX failed but answer was actually correct"),
        _crow("WRONG", b["wrong"], b["n_b"], "EX failed and answer was genuinely wrong"),
        _crow("GOLD_ERROR / REVIEW", b["review"], b["n_b"],
              "gold SQL appears incorrect, needs human check"),
    ]

    # --- Tier 5 answer quality --------------------------------------------
    rec_n = a["n_error_recovery"]
    L += [
        "", sep,
        f"TIER 5 - ANSWER QUALITY  (judge rubrics, n={a['n']})",
        "  Did the agent's natural language answer correctly represent what the data showed?",
        "",
        _vrow("Faithfulness", _score(a["mean_faithfulness"]),
              "answer matched the returned data without distortion"),
        _vrow("Uncertainty", _score(a["mean_uncertainty"]),
              "answer handled ties, nulls, and edge cases correctly"),
        _vrow("Alignment", _score(a["mean_alignment"]),
              "answer addressed what the question actually asked"),
        _vrow("Error recovery", _score(a["mean_error_recovery"]) if rec_n else "n/a",
              f"agent diagnosed and fixed its own SQL error (n={rec_n})"),
    ]

    # --- Efficiency --------------------------------------------------------
    L += [
        "", sep,
        "EFFICIENCY",
        "  How much compute did each question consume?",
        "",
        _vrow("Mean tokens", _fmt(e["mean_tokens"], 0, comma=True),
              "tokens used per question across all LLM calls"),
        _vrow("Mean LLM calls", _fmt(e["mean_llm_calls"], 1),
              "LLM calls per question (schema + query + answer = 3 baseline)"),
        "", "=" * _W,
    ]
    return "\n".join(L)


def _fmt(v, digits: int, comma: bool = False) -> str:
    if v is None:
        return "n/a"
    if comma:
        return f"{v:,.{digits}f}"
    return f"{v:.{digits}f}"


# --- IO ----------------------------------------------------------------------

def load_records(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv=None) -> None:
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: python -m eval.metrics <results.jsonl>")
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # render —, ←, ↑ on any console
    except Exception:
        pass
    summary = compute_metrics(load_records(argv[0]))
    print(format_summary(summary))


if __name__ == "__main__":
    main()
