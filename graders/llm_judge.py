"""Tier 5: LLM-as-judge rubrics.

Two kinds of judgement, conditioned on the EX outcome (per FLEX):

* SQL-level: Branch A (false-positive check, runs when EX passes, SQL only) and
  Branch B (false-negative check, runs when EX fails, both result sets shown).
* Answer-level: Faithfulness, Uncertainty Acknowledgment, Question Alignment --
  evaluate the agent's natural-language ``final_answer``. Error Recovery runs
  only when the trace self-corrected.

The judge is just another provider-agnostic chat model (see ``agent/llm.py``),
deliberately defaulting to a *different family* than the agent to avoid
self-preference bias. Prompts are taken verbatim from the spec.
"""

from __future__ import annotations

import json
import re

from agent.agent import AgentTrace
from agent.db import SCHEMA
from config import JUDGE_MODEL
from agent.llm import get_chat_model
from experiment_profiles import JudgePromptBundle, load_profile


# --- Prompt templates (verbatim from the spec) -------------------------------

BRANCH_A_FALSE_POSITIVE = """\
You are evaluating a SQL query that produced the correct result set.
Your task is to determine whether it did so for the right reasons,
or whether the match is coincidental on this particular database.

Question: {question}
Schema: {schema_summary}
Gold SQL: {gold_sql}
Agent SQL: {agent_sql}

Do not consider the result sets — they matched. Focus only on whether
the agent's SQL logic correctly translates the question.

Evaluate:

SCHEMA ALIGNMENT: Do the tables and columns used match the intent of
the question and the actual schema? Flag any column or table that is
technically valid but semantically wrong for this question.

CORRECT FILTERING: Do the WHERE clauses reflect all conditions stated
in the question? Consider whether a filter that works on this dataset
would fail if new rows were added. NOTE: defensive redundant filters
(e.g., IS NOT NULL on a non-nullable column, an extra ORDER BY that
doesn't alter row selection) are not false positives — only flag
filters that could exclude valid data on plausible database variations.

CLAUSE APPROPRIATENESS: Are GROUP BY, HAVING, ORDER BY, DISTINCT used
correctly? Flag missing clauses the question requires, or extra clauses
that could alter results on different data.

MULTIPLE ROW HANDLING: If the question implies a single result but
multiple rows could qualify (e.g., a tie), does the SQL handle this
correctly, or does it suppress rows that should appear?

VERDICT:
CORRECT — SQL logic correctly translates the question. Defensive
  redundant additions (IS NOT NULL on required fields, extra columns)
  are acceptable and should not trigger FALSE_POSITIVE.
FALSE_POSITIVE — SQL has a logical flaw that would produce incorrect
  results on plausible data variations. Describe the flaw and give a
  concrete example of data that would expose it.

VERDICT: <CORRECT or FALSE_POSITIVE>
REASON: <cite specific clauses or column references>
"""

BRANCH_B_FALSE_NEGATIVE = """\
You are evaluating a SQL query that produced a different result set
from the gold SQL. Your task is to determine whether the agent's query
is actually wrong, or whether the mismatch is acceptable.

Question: {question}
Schema: {schema_summary}
Gold SQL: {gold_sql}
Agent SQL: {agent_sql}
Gold result: {gold_result}
Agent result: {agent_result}

Evaluate:

OUTPUT STRUCTURE: Are differences in column selection, column order,
or extra/missing columns acceptable given how the question is phrased?
(e.g., question asks "which provinces" — agent returning name + region
while gold returns name only may be acceptable.)

VALUE REPRESENTATION: Are differences in numeric precision, rounding,
or format (e.g., 13400.0 vs 13400) meaningful or cosmetic?

MULTIPLE VALID INTERPRETATIONS: Does the question admit more than one
valid SQL translation? If the agent's interpretation is defensible,
the mismatch may reflect ambiguity rather than error.

GOLD SQL QUALITY: Is the gold SQL correct? If the agent's SQL is a
better or equally valid translation of the question, flag it.

VERDICT:
WRONG — Agent's SQL does not correctly answer the question.
ACCEPTABLE — Mismatch is cosmetic or reflects valid ambiguity.
  Describe why the agent's result is a valid answer.
GOLD_ERROR — Gold SQL appears incorrect. Flag for human review.
HUMAN_REVIEW — Cannot determine without domain expertise.

VERDICT: <WRONG, ACCEPTABLE, GOLD_ERROR, or HUMAN_REVIEW>
REASON: <cite specific differences and whether they alter the meaning
         of the answer>
"""

FAITHFULNESS = """\
You are evaluating a data assistant's response to a database question.

Question: {question}
Result set returned by the database: {agent_result_set}
Agent's final answer: {final_answer}

Score FAITHFULNESS — does the answer accurately represent the result set?

1 — Answer contains a claim that directly contradicts the result set: wrong entity
    name, wrong numeric value, wrong direction of comparison (e.g., reports highest
    when result shows lowest).
2 — Answer mischaracterizes a result in a significant way: reports a sum as an
    average, drops a tie, or makes an unsupported claim not derivable from the data.
3 — Answer mostly correct but omits something meaningful from the result set that
    the question asked for, or adds a minor unsupported inference.
4 — Answer accurately represents the main result but has a very minor imprecision
    that does not change meaning (e.g., rounds a value slightly, omits a secondary
    detail that wasn't explicitly asked for).
5 — Answer accurately and completely represents the result set with no omissions,
    no unsupported claims, and correctly handles any edge cases visible in the data
    (ties, nulls, zero-row results). If no edge cases exist, a fully accurate
    answer still scores 5.

SCORE: <1-5>
REASON: <cite a specific value or entity from the result set and how the answer
         handles it>
"""

UNCERTAINTY = """\
You are evaluating a data assistant's response to a database question.

Question: {question}
Result set returned by the database: {agent_result_set}
Agent's final answer: {final_answer}

Score UNCERTAINTY ACKNOWLEDGMENT — does the answer correctly handle ambiguity
visible in the result set?

1 — Answer states a definitive conclusion when the result set contains an obvious
    ambiguity: a tie between two rows, a null in the key column, a single-row
    result that may not generalize, or a result that is exactly zero.
2 — Answer notes that ambiguity exists but misidentifies what it is or why it matters.
3 — Answer does not flag ambiguity that exists in the data, but the claim it makes
    is still technically defensible — the omission is minor.
4 — Answer handles ambiguity correctly but incompletely: either identifies a tie
    without explaining its implication, or notes a caveat without fully articulating
    what it means for the result.
5 — Answer fully and correctly handles ambiguity. If ambiguity exists, it is named
    and its practical implication explained. If the data is unambiguous, the answer
    states a clear conclusion without unnecessary hedging. Either outcome scores 5
    when handled correctly.

SCORE: <1-5>
REASON: <identify the specific ambiguity in the result set, if any, and how the
         answer handles it — if no ambiguity exists, confirm this and explain
         why the score reflects it>
"""

ERROR_RECOVERY = """\
You are evaluating how a data assistant recovered from a query error.

Question: {question}
Failed SQL: {failed_sql}
Error message: {error_message}
Corrected SQL: {corrected_sql}

Score ERROR RECOVERY — did the agent correctly diagnose and fix the error?

1 — Agent resubmitted the same query unchanged, or made a change unrelated to
    the error message.
2 — Agent identified the error type correctly but the fix introduces a new
    structural problem (e.g., fixes a column name but breaks the join condition).
3 — Agent fixed the immediate error but did not address an underlying issue the
    error was symptomatic of (e.g., fixed the syntax but the logic is still wrong).
4 — Agent correctly diagnosed and fixed the error. Corrected SQL executes cleanly
    and the fix is directly traceable to the error message.
5 — Agent correctly diagnosed the error, fixed it, and the corrected SQL reflects
    improved understanding of the schema (e.g., ran SELECT DISTINCT to verify an
    enum value before re-filtering, rather than guessing again).

SCORE: <1-5>
REASON: <cite the specific error message and identify exactly what changed between
         the two queries>
"""

QUESTION_ALIGNMENT = """\
You are evaluating whether a data assistant answered the question that was asked.

Question: {question}
Agent's final answer: {final_answer}

Score QUESTION ALIGNMENT — does the answer address the question as stated?

1 — Answer addresses a clearly different question: asked for highest, returned
    lowest; asked for a count, returned a list; asked about one entity, answered
    about another.
2 — Answer addresses the main question but drops a meaningful qualifier from the
    original (e.g., "among active provinces" or "in the last recorded year" was
    silently ignored).
3 — Answer addresses the question but at the wrong level of granularity: asked for
    per-province breakdown, returned an empire-wide total; asked for the single
    top result, returned a ranked list.
4 — Answer addresses the question correctly but with a minor gap: a qualifier is
    honored but not explicitly acknowledged, or the granularity is right but
    slightly imprecise in phrasing.
5 — Answer directly and completely addresses the question as stated, at the right
    granularity, with all qualifiers honored. No gaps, no drift. A correct answer
    to a simple question with no qualifiers scores 5 without needing to restate
    anything explicitly.

SCORE: <1-5>
REASON: <quote the specific part of the question that was or was not addressed
         in the answer>
"""


# --- Parsing helpers ---------------------------------------------------------

def _schema_summary() -> str:
    """A compact textual schema for the judge prompts."""
    lines = []
    for t in SCHEMA["tables"]:
        cols = ", ".join(c["name"] for c in t["columns"])
        lines.append(f"{t['name']}({cols})")
    return "\n".join(lines)


def _parse_verdict(text: str) -> dict:
    verdict = None
    m = re.search(r"VERDICT:\s*([A-Z_]+)", text)
    if m:
        verdict = m.group(1)
    reason = _parse_reason(text)
    return {"verdict": verdict, "reason": reason, "raw": text}


def _parse_score(text: str) -> dict:
    score = None
    m = re.search(r"SCORE:\s*([1-5])", text)
    if m:
        score = int(m.group(1))
    return {"score": score, "reason": _parse_reason(text), "raw": text}


def _parse_reason(text: str) -> str:
    m = re.search(r"REASON:\s*(.+)", text, re.DOTALL)
    return m.group(1).strip() if m else ""


# --- Judge -------------------------------------------------------------------

class Judge:
    """Wraps a provider-agnostic chat model and runs the rubric prompts."""

    def __init__(self, model: str = JUDGE_MODEL, prompts: JudgePromptBundle | None = None):
        self.model_name = model  # registry key into config.MODELS
        self.prompts = prompts or load_profile().judge_prompts
        self._model = None  # built lazily so importing this module is cheap

    @property
    def model(self):
        if self._model is None:
            self._model = get_chat_model(self.model_name)
        return self._model

    def _ask(self, prompt: str) -> str:
        from langchain_core.messages import HumanMessage

        response = self.model.invoke([HumanMessage(content=prompt)])
        return (response.text or "").strip()

    # -- SQL-level (branch) --------------------------------------------------
    def false_positive_check(self, question, gold_sql, agent_sql) -> dict:
        prompt = self.prompts.templates["branch_a_false_positive"].format(
            question=question, schema_summary=_schema_summary(),
            gold_sql=gold_sql, agent_sql=agent_sql,
        )
        return _parse_verdict(self._ask(prompt))

    def false_negative_check(self, question, gold_sql, agent_sql, gold_result, agent_result) -> dict:
        prompt = self.prompts.templates["branch_b_false_negative"].format(
            question=question, schema_summary=_schema_summary(),
            gold_sql=gold_sql, agent_sql=agent_sql,
            gold_result=json.dumps(gold_result, default=str),
            agent_result=json.dumps(agent_result, default=str),
        )
        return _parse_verdict(self._ask(prompt))

    # -- Answer-level --------------------------------------------------------
    def faithfulness(self, question, agent_result_set, final_answer) -> dict:
        return _parse_score(self._ask(self.prompts.templates["faithfulness"].format(
            question=question,
            agent_result_set=json.dumps(agent_result_set, default=str),
            final_answer=final_answer,
        )))

    def uncertainty(self, question, agent_result_set, final_answer) -> dict:
        return _parse_score(self._ask(self.prompts.templates["uncertainty"].format(
            question=question,
            agent_result_set=json.dumps(agent_result_set, default=str),
            final_answer=final_answer,
        )))

    def question_alignment(self, question, final_answer) -> dict:
        return _parse_score(self._ask(self.prompts.templates["question_alignment"].format(
            question=question, final_answer=final_answer,
        )))

    def error_recovery(self, question, failed_sql, error_message, corrected_sql) -> dict:
        return _parse_score(self._ask(self.prompts.templates["error_recovery"].format(
            question=question, failed_sql=failed_sql,
            error_message=error_message, corrected_sql=corrected_sql,
        )))


def grade(
    trace: AgentTrace,
    question: str,
    gold_sql: str | None,
    ex_result: dict | None,
    gold_result: list | None = None,
    judge: Judge | None = None,
) -> dict:
    """Run the Tier 4 branching structure described in the spec.

    Args:
        trace: The agent trace (provides final_answer, SQL, self-correction).
        question: The original question.
        gold_sql: Gold SQL if available (None => SQL-level branches are skipped).
        ex_result: The Tier 2b execution_accuracy dict (provides match).
        gold_result: The gold result rows (from the test case), shown to the
            judge in Branch B.
        judge: A Judge instance; one is created from JUDGE_MODEL if not supplied.
    """
    judge = judge or Judge()
    out: dict = {"branch": None}
    agent_sql = trace.last_query_sql()
    agent_result_set = _last_query_rows(trace)

    # --- SQL-level branch (needs gold SQL and an EX outcome) ---------------
    short_circuit = False
    if gold_sql and agent_sql and ex_result is not None:
        if ex_result.get("match"):
            out["branch"] = "A"
            v = judge.false_positive_check(question, gold_sql, agent_sql)
            out["false_positive_verdict"] = v["verdict"]
            out["false_positive_reason"] = v["reason"]
        else:
            out["branch"] = "B"
            v = judge.false_negative_check(
                question, gold_sql, agent_sql,
                gold_result or [], agent_result_set,
            )
            out["false_negative_verdict"] = v["verdict"]
            out["false_negative_reason"] = v["reason"]
            # ACCEPTABLE / GOLD_ERROR -> human review; skip answer rubrics.
            if v["verdict"] in {"ACCEPTABLE", "GOLD_ERROR", "HUMAN_REVIEW"}:
                short_circuit = True

    # --- Answer-level rubrics (run when final_answer is non-empty) ---------
    if trace.final_answer and not short_circuit:
        f = judge.faithfulness(question, agent_result_set, trace.final_answer)
        out["faithfulness_score"], out["faithfulness_reason"] = f["score"], f["reason"]
        u = judge.uncertainty(question, agent_result_set, trace.final_answer)
        out["uncertainty_score"], out["uncertainty_reason"] = u["score"], u["reason"]
        q = judge.question_alignment(question, trace.final_answer)
        out["question_alignment_score"], out["question_alignment_reason"] = q["score"], q["reason"]

    # --- Error recovery (conditional on self-correction) -------------------
    recovery = _error_recovery_inputs(trace)
    if recovery:
        r = judge.error_recovery(question, **recovery)
        out["error_recovery_score"], out["error_recovery_reason"] = r["score"], r["reason"]
    else:
        out["error_recovery_score"], out["error_recovery_reason"] = None, "Not triggered — self_corrected was false."

    return out


def _last_query_rows(trace: AgentTrace) -> list:
    queries = [tc for tc in trace.tool_calls if tc.tool_name == "run_query"]
    if not queries:
        return []
    return queries[-1].result.get("rows", []) or []


def _error_recovery_inputs(trace: AgentTrace) -> dict | None:
    """If the agent self-corrected, return inputs for the Error Recovery rubric."""
    queries = [tc for tc in trace.tool_calls if tc.tool_name == "run_query"]
    if len(queries) < 2:
        return None
    if queries[0].result.get("error") is None or queries[-1].result.get("error") is not None:
        return None
    return {
        "failed_sql": queries[0].parameters.get("sql", ""),
        "error_message": queries[0].result.get("error", ""),
        "corrected_sql": queries[-1].parameters.get("sql", ""),
    }
