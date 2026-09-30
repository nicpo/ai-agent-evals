"""Run the agent, then grade every attempt with Jev as the semantic SQL judge.

Stage 1 is the normal harness (``eval.runner.run_suite``): the agent answers each
question and is graded by EX and the GPT judge. Stage 2 sends each attempt that
has SQL to Jev (the hosted System One model) and writes one row per attempt
next to the EX and GPT verdicts. Feed the output to ``report.py``.

Usage:

    python -m experiments.exp_04_jev_judge.run --dry-run             # show Jev requests, no calls
    python -m experiments.exp_04_jev_judge.run                       # toy profile, monolithic judge
    python -m experiments.exp_04_jev_judge.run --variant two-call    # also the narrow + verdict-after calls
    python -m experiments.exp_04_jev_judge.run --results output/results/output_<stamp>.jsonl
                                                                     # judge an existing run, skip the agent
    python -m experiments.exp_04_jev_judge.run --out <jsonl>         # resume an interrupted Jev pass

Needs ``TYPESAFE_API_KEY`` plus the agent and GPT judge provider keys in ``.env``.
The question, schema, SQL and result rows of each attempt are sent to the hosted
API; the toy set is synthetic, your own eval set may not be.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path

from config import RESULTS_DIR
from eval.runner import load_questions, run_suite
from experiment_profiles import load_profile

from .jev_client import JevClient
from .jev_judge import (
    VARIANTS, build_cases, load_config, load_jsonl, load_prompt, request_for_case,
    narrow_request, run_judgments,
)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--profile", default="toy", help="Experiment profile (default: toy).")
    p.add_argument("--difficulty", help="Filter to one difficulty tier.")
    p.add_argument("--limit", type=int, help="Cap the number of questions.")
    p.add_argument("--variant", choices=VARIANTS, default="mono",
                   help="mono: one call per attempt. two-call: mono + narrow + verdict-after (3 calls).")
    p.add_argument("--results", type=Path, help="Existing results JSONL; skips the agent run.")
    p.add_argument("--out", type=Path, help="Output JSONL (an existing file is resumed).")
    p.add_argument("--dry-run", action="store_true", help="Print example Jev requests and exit.")
    return p.parse_args(argv)


def _dry_run(cases: list[dict], variant: str) -> None:
    prompt, decomposed = load_prompt("monolithic"), load_prompt("decomposed")
    for case in cases:
        state, questions = request_for_case(case, prompt)
        print(f"--- {case['key']} (branch {case['branch']}) ---")
        print(json.dumps({"state": state, "questions": questions}, indent=2, default=str))
        if variant == "two-call":
            print("narrow questions:")
            print(json.dumps(narrow_request(case, decomposed), indent=2))


def _example_cases(questions: list[dict]) -> list[dict]:
    """One branch-A and one branch-B case built from the first question, no agent needed."""
    from graders.llm_judge import _schema_summary

    base = questions[0]
    common = {"id": base["id"], "profile": "example", "run_index": 0, "question": base["question"],
              "schema": _schema_summary(), "gold_sql": base["gold_sql"],
              "gold_result": base.get("gold_result") or [], "gpt_judge": None}
    return [
        {**common, "key": f"{base['id']}|example-A|0", "branch": "A",
         "agent_sql": base["gold_sql"], "agent_result": []},
        {**common, "key": f"{base['id']}|example-B|0", "branch": "B",
         "agent_sql": "SELECT 1 AS n", "agent_result": [{"n": 1}]},
    ]


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(f"Invalid experiment profile: {exc}") from exc

    all_questions = load_questions(glob=profile.eval_glob)
    cases = [c for c in all_questions if not args.difficulty or c.get("difficulty") == args.difficulty]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No eval cases matched -- check --profile / --difficulty / --limit.")

    config = load_config()
    client = JevClient(config["base_url"], config["api_key_env"], config["timeout_seconds"])

    if args.dry_run:
        if args.results:
            judge_cases, _ = build_cases(load_jsonl(args.results), {q["id"]: q for q in all_questions})
            judge_cases = judge_cases[:1]
        else:
            judge_cases = _example_cases(cases)
        _dry_run(judge_cases, args.variant)
        return

    client.require_key()  # fail before spending on the agent run

    if args.results:
        results_path = args.results
    else:
        # The GPT judge must run: the report compares Jev against it.
        results_path = run_suite(cases, profile=profile, run_judge=True)

    records = load_jsonl(results_path)
    judge_cases, skipped = build_cases(records, {q["id"]: q for q in all_questions})
    if not judge_cases:
        raise SystemExit("No attempts with agent SQL, gold SQL and an EX verdict to judge.")

    out_path = args.out or RESULTS_DIR / f"exp_04_jev_{datetime.now().strftime('%Y-%m-%d_%H-%M')}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nJudging {len(judge_cases)} attempts with Jev ({args.variant}); "
          f"{skipped} without judgeable SQL skipped.")
    counts = run_judgments(judge_cases, client, config, args.variant, out_path)
    print(f"\n{counts}\nJev results: {out_path}\n"
          f"Next: python -m experiments.exp_04_jev_judge.report {out_path}")


if __name__ == "__main__":
    main()
