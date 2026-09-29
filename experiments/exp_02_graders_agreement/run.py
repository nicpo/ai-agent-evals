"""End-to-end: run the suite, find dual passes, probe them against scenarios.

Usage:

    python -m experiments.exp_02_graders_agreement.run --profile toy
    python -m experiments.exp_02_graders_agreement.run --results output/results/<existing>.jsonl

Requires provider API keys in ``.env`` unless ``--results`` points at an
existing results JSONL (already produced by ``eval.runner`` or this
experiment's own prior run), in which case no LLM calls are made here -- only
SQL execution against scenario databases.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from config import DB_PATH, RESULTS_DIR
from experiment_profiles import load_profile
from eval.metrics import load_records
from eval.runner import load_questions, run_suite

from experiments.exp_02_graders_agreement.probe import dual_pass_candidates, probe_all
from experiments.exp_02_graders_agreement.scenarios import discover_scenarios


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="toy", help="Experiment profile (default: toy).")
    p.add_argument("--results", type=Path, help="Reuse an existing results JSONL instead of running the suite.")
    p.add_argument("--difficulty", help="Filter to one difficulty tier (only used without --results).")
    p.add_argument("--limit", type=int, help="Cap the number of questions (only used without --results).")
    p.add_argument("--db", type=Path, default=DB_PATH, help="Database to build scenarios from.")
    p.add_argument("--out", type=Path, help="Where to write the JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)

    if args.results:
        results_path = args.results
    else:
        try:
            profile = load_profile(args.profile)
        except ValueError as exc:
            raise SystemExit(f"Invalid experiment profile: {exc}") from exc
        cases = load_questions(glob=profile.eval_glob)
        if args.difficulty:
            cases = [c for c in cases if c.get("difficulty") == args.difficulty]
        if args.limit is not None:
            cases = cases[: args.limit]
        if not cases:
            raise SystemExit("No eval cases matched -- check --profile / --difficulty / --limit.")
        results_path = run_suite(cases, profile=profile, run_judge=True)

    records = load_records(results_path)
    cases_by_id = {c["id"]: c for c in load_questions(glob="**/questions-*.jsonl")}
    candidates = dual_pass_candidates(records, cases_by_id)
    if not candidates:
        raise SystemExit(
            "No dual-pass cases (EX match + judge CORRECT) in this run -- "
            "nothing to probe."
        )

    import sqlite3

    conn = sqlite3.connect(args.db)
    try:
        scenarios = discover_scenarios(conn)
    finally:
        conn.close()

    report = probe_all(candidates, scenarios, args.db)
    report["source_results"] = str(results_path)

    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    out_path = args.out or RESULTS_DIR / f"exp_02_scenario_probe_{stamp}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print(f"Dual-pass candidates: {report['n_candidates']}")
    print(f"Scenarios:            {report['n_scenarios']}")
    print(f"Probes run:           {report['n_probes']}")
    print(f"Counterexamples:      {report['n_counterexamples']}  <- need manual review")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
