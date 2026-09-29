"""Run each question multiple times, optionally across several agent profiles.

Uses the same primitives ``eval.runner.run_suite`` uses internally
(``run_agent`` + ``grade_trace`` + ``build_result_record``), but keeps every
attempt instead of writing one file per run, and tags each record with
``run_index`` so repeats of the same question can be grouped later.

Usage:

    python -m experiments.exp_03_repeatability_costs.run --profiles toy --repeats 3
    python -m experiments.exp_03_repeatability_costs.run --profiles v1 v2 v3 --repeats 3 --limit 10

Requires provider API keys in ``.env`` -- this makes real agent + judge calls,
``repeats`` times per question per profile. Start small (``--limit``,
``--repeats 2``) on the toy set before pointing this at a real eval set.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime

from agent import db
from agent.agent import run_agent
from agent.tools import Toolbox
from config import RESULTS_DIR
from experiment_profiles import load_profile
from eval.runner import build_result_record, grade_trace, load_questions
from experiments.exp_03_repeatability_costs.costs import CostTrackingJudge


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profiles", nargs="+", default=["toy"], help="Experiment profiles to compare.")
    p.add_argument("--repeats", type=int, default=3, help="Attempts per question per profile.")
    p.add_argument("--difficulty", help="Filter to one difficulty tier.")
    p.add_argument("--limit", type=int, help="Cap the number of questions (per profile).")
    p.add_argument("--no-judge", action="store_true", help="Skip the LLM judge (EX + tool-call grading only).")
    p.add_argument("--out", help="Output JSONL path (default: timestamped file under output/results/).")
    return p.parse_args(argv)


def run_repeats(profile_names: list[str], repeats: int, *, difficulty: str | None = None,
                 limit: int | None = None, run_judge: bool | None = None) -> tuple[list[dict], dict[str, list[dict]]]:
    """Run every question ``repeats`` times for each named profile.

    Returns ``(records, judge_usage_by_profile)``: the flat list of result
    records, each with ``run_index`` and the resolved ``profile.metadata()``
    (already attached by ``build_result_record``) so downstream analysis can
    group by ``(profile, question id)``; and each profile's judge token usage
    log (see ``costs.CostTrackingJudge``), for cost accounting.
    """
    records: list[dict] = []
    judge_usage_by_profile: dict[str, list[dict]] = {}
    conn = db.connect()
    try:
        for profile_name in profile_names:
            profile = load_profile(profile_name)
            cases = load_questions(glob=profile.eval_glob)
            if difficulty:
                cases = [c for c in cases if c.get("difficulty") == difficulty]
            if limit is not None:
                cases = cases[:limit]
            if not cases:
                raise SystemExit(f"No eval cases matched for profile {profile_name!r}.")

            use_judge = profile.judge_enabled if run_judge is None else run_judge
            judge = None
            if use_judge:
                judge = CostTrackingJudge(profile.judge_model, prompts=profile.judge_prompts)
                judge_usage_by_profile[profile.name] = judge.usage_log

            total = len(cases) * repeats
            done = 0
            for case in cases:
                for run_index in range(repeats):
                    toolbox = Toolbox.open(profile.enabled_tools)
                    try:
                        trace = run_agent(
                            case["question"], profile.agent_model, toolbox=toolbox,
                            max_tool_calls=profile.max_tool_calls,
                            system_prompt=profile.agent_prompt,
                        )
                    finally:
                        toolbox.close()
                    graders = grade_trace(trace, case, conn, judge=judge, run_judge=use_judge)
                    record = build_result_record(trace, case, graders, profile)
                    record["run_index"] = run_index
                    records.append(record)
                    done += 1
                    ex = (graders.get("tier2b") or {}).get("execution_accuracy", {})
                    print(f"[{profile_name} {done}/{total}] {case.get('id')} run={run_index}: "
                          f"EX match={ex.get('match')}")
    finally:
        conn.close()
    return records, judge_usage_by_profile


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        records, judge_usage_by_profile = run_repeats(
            args.profiles, args.repeats, difficulty=args.difficulty, limit=args.limit,
            run_judge=False if args.no_judge else None,
        )
    except ValueError as exc:
        raise SystemExit(f"Invalid experiment profile: {exc}") from exc

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if args.out:
        out_path = RESULTS_DIR / args.out if not args.out.startswith(str(RESULTS_DIR)) else args.out
    else:
        stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
        out_path = RESULTS_DIR / f"exp_03_repeats_{stamp}.jsonl"

    with open(out_path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, default=str) + "\n")

    usage_path = out_path.with_suffix(".judge_usage.json")
    usage_path.write_text(json.dumps(judge_usage_by_profile, indent=2), encoding="utf-8")

    print(f"\n{len(records)} attempts written to {out_path}")
    print(f"Judge token usage written to {usage_path}")


if __name__ == "__main__":
    main()
