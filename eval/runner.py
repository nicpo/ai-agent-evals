"""Orchestrates the agent + all grader tiers and writes per-question results.

Usage (live run -- requires the relevant provider API key in the environment):

    python -m eval.runner                              # all questions, with judge
    python -m eval.runner --no-judge                   # skip Tier 5 (no judge calls)
    python -m eval.runner --difficulty easy --limit 5
    python -m eval.runner --agent-model haiku-4-5 --judge-model gpt-5-4-mini

The grading logic (``grade_trace``) is deliberately separable from the agent run
so traces produced elsewhere -- or fixtures -- can be graded without an LLM.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from agent import db
from agent.agent import AgentTrace, run_agent
from agent.tools import Toolbox
from config import EVALS_DIR, RESULTS_DIR, SUMMARIES_DIR
from experiment_profiles import ExperimentProfile, load_profile
from graders import execution, sql_structural, tool_call
from eval.metrics import compute_metrics, format_summary


def load_questions(evals_dir: Path = EVALS_DIR, glob: str = "questions-*.jsonl") -> list[dict]:
    """Load all test cases from the canonical questions-*.jsonl files."""
    cases: list[dict] = []
    for path in sorted(evals_dir.glob(glob)):
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    cases.append(json.loads(line))
    return cases


def grade_trace(
    trace: AgentTrace,
    testcase: dict,
    conn,
    judge=None,
    run_judge: bool = False,
) -> dict:
    """Run Tiers 1, 2, 2b and (optionally) 3 against a trace.

    Tiers terminate early per the spec: no SQL => no Tier 2/2b; no gold SQL =>
    no Tier 2b; judge only runs when requested and the case opts in.
    """
    question = testcase["question"]
    gold_sql = testcase.get("gold_sql")
    gold_result = testcase.get("gold_result")
    pred_sql = trace.last_query_sql()

    graders: dict = {"tier1": tool_call.grade(trace)}

    if pred_sql:
        graders["tier2"] = sql_structural.grade(pred_sql, db.SCHEMA)

    ex_result = None
    if gold_sql and pred_sql:
        graders["tier2b"] = execution.grade(gold_sql, pred_sql, conn)
        ex_result = graders["tier2b"].get("execution_accuracy")

    if run_judge and testcase.get("run_llm_judge"):
        from graders import llm_judge

        graders["tier5"] = llm_judge.grade(
            trace, question, gold_sql, ex_result, gold_result, judge
        )

    return graders


def build_result_record(trace: AgentTrace, testcase: dict, graders: dict, profile: ExperimentProfile) -> dict:
    return {
        "id": testcase.get("id"),
        "question": testcase["question"],
        "difficulty": testcase.get("difficulty"),
        "adversarial": testcase.get("adversarial", False),
        "trace": trace.to_dict(),
        "graders": graders,
        "profile": profile.metadata(),
    }


def run_suite(
    cases: list[dict],
    profile: ExperimentProfile | None = None,
    run_judge: bool | None = None,
    out_path: Path | None = None,
) -> Path:
    """Run the agent over every case, grade it, and write a results JSONL file.

    Args:
        cases: Test cases.
        agent_model: Registry key (config.MODELS) for the agent under test.
        judge_model: Registry key for the LLM judge.
        run_judge: Whether to run Tier 5.
        out_path: Where to write results (defaults to a timestamped file).
    """
    profile = profile or load_profile()
    run_judge = profile.judge_enabled if run_judge is None else run_judge
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    if out_path is None:
        out_path = RESULTS_DIR / f"output_{stamp}.jsonl"

    judge = None
    if run_judge:
        from graders.llm_judge import Judge

        judge = Judge(profile.judge_model, prompts=profile.judge_prompts)

    records: list[dict] = []
    conn = db.connect()
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for i, case in enumerate(cases, 1):
                toolbox = Toolbox.open(profile.enabled_tools)
                try:
                    trace = run_agent(case["question"], profile.agent_model, toolbox=toolbox,
                                      max_tool_calls=profile.max_tool_calls,
                                      system_prompt=profile.agent_prompt)
                finally:
                    toolbox.close()
                graders = grade_trace(trace, case, conn, judge=judge, run_judge=run_judge)
                record = build_result_record(trace, case, graders, profile)
                records.append(record)
                fh.write(json.dumps(record, default=str) + "\n")
                fh.flush()
                ex = (graders.get("tier2b") or {}).get("execution_accuracy", {})
                print(f"[{i}/{len(cases)}] {case.get('id')}: EX match={ex.get('match')}")
    finally:
        conn.close()

    # Aggregate metrics: print to screen and save to the summaries dir.
    summary = compute_metrics(records)
    summary["meta"] = {
        "run_file": out_path.name,
        "timestamp": stamp,
        **profile.metadata(),
        "judge_model": profile.judge_model if run_judge else None,
        "n_questions": len(records),
    }
    SUMMARIES_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = SUMMARIES_DIR / f"summary_{stamp}.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    try:
        sys.stdout.reconfigure(encoding="utf-8")  # render —, ←, ↑ on any console
    except Exception:
        pass
    print("\n" + format_summary(summary))
    print(f"\nTraces:  {out_path}\nSummary: {summary_path}")

    return out_path


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the SQL-agent eval suite.")
    p.add_argument("--difficulty", help="Filter to one difficulty tier.")
    p.add_argument("--limit", type=int, help="Cap the number of questions.")
    p.add_argument("--no-judge", action="store_true", help="Skip Tier 5 LLM judge.")
    p.add_argument("--profile", help="Named experiment profile (defaults to catalog default).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(f"Invalid experiment profile: {exc}") from exc
    cases = load_questions(glob=profile.eval_glob)
    if args.difficulty:
        cases = [c for c in cases if c.get("difficulty") == args.difficulty]
    if args.limit is not None:
        cases = cases[: args.limit]

    run_suite(
        cases,
        profile=profile,
        run_judge=False if args.no_judge else None,
    )


if __name__ == "__main__":
    main()
