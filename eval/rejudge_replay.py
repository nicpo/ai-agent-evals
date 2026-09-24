"""Rejudge EX-transition cases from saved replay records, without rerunning agents.

Only the SQL-level judge branch made stale by an EX transition is rerun.  The
command writes an append-only adjudication JSONL and never modifies historical
results or deterministic replays.
"""
from __future__ import annotations

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
from typing import Any

from config import EVALS_DIR, JUDGE_MODEL, MODELS
from eval.replay import ReplayError, _cases, _jsonl, _sha256, apply_gold_corrections
from graders.llm_judge import Judge


def _old_verdict(record: dict[str, Any]) -> str | None:
    graders = record.get("historical_graders") or {}
    judge = graders.get("tier5") or graders.get("tier3") or {}
    return judge.get("false_negative_verdict")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--eval-dir", type=Path, default=EVALS_DIR)
    parser.add_argument("--db", type=Path, required=True)
    parser.add_argument("--gold-corrections", type=Path, required=True)
    parser.add_argument("--judge-model", choices=list(MODELS), default=JUDGE_MODEL)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing output: {args.out}")
    try:
        cases, _ = _cases(args.eval_dir)
        cases, applied = apply_gold_corrections(cases, args.gold_corrections, args.db)
        judge = Judge(args.judge_model)
        items: list[dict[str, Any]] = []
        for replay_path in args.replays:
            for record in _jsonl(replay_path):
                replay = record.get("replay") or {}
                if replay.get("transition") != "fail_to_pass":
                    continue
                case = cases.get(record.get("id"))
                queries = [call for call in record.get("trace", {}).get("tool_calls", [])
                           if call.get("tool_name") == "run_query"]
                if case is None or case.get("question") != record.get("question") or not queries:
                    raise ReplayError(f"Invalid transitioned replay record: {record.get('id')!r}")
                agent_sql = queries[-1].get("parameters", {}).get("sql")
                if not agent_sql:
                    raise ReplayError(f"No saved final SQL: {record.get('id')!r}")
                verdict = judge.false_positive_check(case["question"], case["gold_sql"], agent_sql)
                items.append({
                    "id": record["id"], "question": record["question"],
                    "replay_source": str(replay_path), "replay_source_sha256": _sha256(replay_path),
                    "gold_correction": replay.get("gold_correction"),
                    "old_branch": "B", "new_branch": "A", "old_verdict": _old_verdict(record),
                    "new_verdict": verdict.get("verdict"), "new_reason": verdict.get("reason"),
                    "judge_model": args.judge_model,
                    "prompt_sha256": hashlib.sha256(
                        judge.prompts.templates["branch_a_false_positive"].encode("utf-8")
                    ).hexdigest(),
                    "rejudged_at_utc": datetime.now(UTC).isoformat(), "agent_rerun": False,
                    "scope": "SQL-level Branch A false-positive check only",
                })
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, default=str) + "\n")
        counts: dict[str, int] = {}
        for item in items:
            label = item["new_verdict"] or "PARSE_ERROR"
            counts[label] = counts.get(label, 0) + 1
        args.out.with_suffix(".summary.json").write_text(json.dumps({
            "applied_gold_corrections": applied, "transitioned_cases_rejudged": len(items),
            "new_verdict_counts": counts, "judge_model": args.judge_model, "agent_rerun": False,
        }, indent=2), encoding="utf-8")
    except (OSError, ValueError, KeyError, ReplayError) as exc:
        raise SystemExit(f"Replay rejudge failed: {exc}") from exc


if __name__ == "__main__":
    main()
