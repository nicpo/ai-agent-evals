"""Regrade saved traces without rerunning an agent or LLM judge."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

from agent import db
from agent.agent import AgentTrace
from eval.runner import grade_trace


class ReplayError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        with path.open(encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise ReplayError(f"Cannot read JSONL {path}: {exc}") from exc


def _cases(evals_dir: Path) -> tuple[dict[str, dict], dict[str, str]]:
    paths = sorted(evals_dir.glob("questions-*.jsonl"))
    if not paths:
        raise ReplayError(f"No questions-*.jsonl files in {evals_dir}")
    index, hashes = {}, {}
    for path in paths:
        hashes[str(path)] = _sha256(path)
        for case in _jsonl(path):
            if not case.get("id") or case["id"] in index:
                raise ReplayError(f"Missing or duplicate testcase ID: {case.get('id')!r}")
            index[case["id"]] = case
    return index, hashes


def _question_sha256(question: str) -> str:
    return hashlib.sha256(question.encode("utf-8")).hexdigest()


def apply_gold_corrections(
    cases: dict[str, dict[str, Any]], corrections_path: Path, db_path: Path,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Apply approved SQL corrections and materialize their results from the frozen DB."""
    corrected = copy.deepcopy(cases)
    applied: list[dict[str, Any]] = []
    conn = db.connect(db_path)
    try:
        for correction in _jsonl(corrections_path):
            if correction.get("reviewer_decision") != "approved":
                continue
            case_id = correction.get("id")
            case = corrected.get(case_id)
            if case is None:
                raise ReplayError(f"Correction refers to missing case: {case_id!r}")
            if correction.get("question") != case.get("question"):
                raise ReplayError(f"Correction question mismatch: {case_id!r}")
            if correction.get("question_sha256") != _question_sha256(case["question"]):
                raise ReplayError(f"Correction question hash mismatch: {case_id!r}")
            if correction.get("old_gold_sql") != case.get("gold_sql"):
                raise ReplayError(f"Correction old_gold_sql mismatch: {case_id!r}")
            proposed_sql = correction.get("proposed_gold_sql")
            if not isinstance(proposed_sql, str) or not proposed_sql.strip():
                raise ReplayError(f"Correction has no proposed_gold_sql: {case_id!r}")
            try:
                cursor = conn.execute(proposed_sql)
                columns = [column[0] for column in cursor.description or ()]
                result = [dict(zip(columns, row)) for row in cursor.fetchall()]
            except Exception as exc:
                raise ReplayError(f"Correction SQL failed for {case_id!r}: {exc}") from exc
            case["gold_sql"] = proposed_sql
            case["gold_result"] = result
            case["_gold_correction"] = {
                "id": case_id,
                "question_sha256": correction["question_sha256"],
                "corrected_row_count": len(result),
            }
            applied.append(case["_gold_correction"])
    finally:
        conn.close()
    return corrected, applied


def _old_ex(record: dict) -> bool | None:
    ex = ((record.get("graders") or {}).get("tier2b") or {}).get("execution_accuracy")
    return ex.get("match") if ex else None


def _transition(old: bool | None, new: bool | None) -> str:
    if old is None or new is None: return "not_evaluable"
    if old and not new: return "pass_to_fail"
    if not old and new: return "fail_to_pass"
    return "unchanged"


def replay_records(records: list[dict], cases: dict[str, dict], db_path: Path, *, source_path: Path, source_sha256: str, gold_hashes: dict[str, str], corrections: list[dict[str, Any]] | None = None) -> tuple[list[dict], dict]:
    replayed, changed = [], []
    old_pass = new_pass = missing_sql = pred_errors = 0
    conn = db.connect(db_path)
    try:
        for record in records:
            case = cases.get(record.get("id"))
            if case is None or record.get("question") != case.get("question"):
                raise ReplayError(f"Missing or mismatched gold case: {record.get('id')!r}")
            trace = AgentTrace.from_dict(record["trace"])
            if trace.last_query_sql() is None: missing_sql += 1
            graders = grade_trace(trace, case, conn, run_judge=False)
            old, ex = _old_ex(record), (graders.get("tier2b") or {}).get("execution_accuracy")
            new = ex.get("match") if ex else None
            old_pass += old is True; new_pass += new is True
            pred_errors += bool(ex and ex.get("reason") == "pred_execution_error")
            transition = _transition(old, new)
            if transition != "unchanged": changed.append({"id": record.get("id"), "old_ex": old, "new_ex": new, "transition": transition, "reason": ex.get("reason") if ex else None})
            replayed.append({"id": record.get("id"), "question": record["question"], "difficulty": record.get("difficulty"), "adversarial": record.get("adversarial", False), "trace": record["trace"], "graders": graders, "historical_graders": record.get("graders") or {}, "replay": {"source_results": str(source_path), "source_results_sha256": source_sha256, "database": str(db_path), "database_sha256": _sha256(db_path), "gold_eval_sha256": gold_hashes, "gold_correction": case.get("_gold_correction"), "comparator_policy": "strict_positional_v1", "old_ex": old, "new_ex": new, "transition": transition, "judge_rerun": False}})
    finally:
        conn.close()
    return replayed, {"total": len(records), "old_ex_pass": old_pass, "new_ex_pass": new_pass, "fail_to_pass": sum(c["transition"] == "fail_to_pass" for c in changed), "pass_to_fail": sum(c["transition"] == "pass_to_fail" for c in changed), "unchanged": len(records) - len(changed), "missing_sql": missing_sql, "pred_execution_errors": pred_errors, "changed_cases": changed, "judge_rerun": False, "applied_gold_corrections": corrections or []}


def _write(out_dir: Path, label: str, records: list[dict], summary: dict) -> None:
    result, report = out_dir / f"{label}.replay.jsonl", out_dir / f"{label}.summary.json"
    if result.exists() or report.exists(): raise ReplayError(f"Refusing to overwrite replay output for {label}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with result.open("x", encoding="utf-8") as f:
        for record in records: f.write(json.dumps(record, default=str) + "\n")
    report.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def _manifest_runs(path: Path, versions: list[str]) -> list[tuple[str, Path, Path, Path]]:
    manifest, root = json.loads(path.read_text(encoding="utf-8")), path.parent.parent
    runs = {r["published_version"]: r for r in manifest["runs"]}
    wanted = list(runs) if versions == ["all"] else versions
    if any(v not in runs for v in wanted): raise ReplayError("Requested version absent from manifest")
    selected = []
    for version in wanted:
        run = runs[version]
        for artifact in [run["results"], run["database"], *run["gold_evals"]]:
            p = root / artifact["path"]
            if not p.is_file() or _sha256(p) != artifact["sha256"]: raise ReplayError(f"Manifest verification failed: {artifact['path']}")
        selected.append((version, root / run["results"]["path"], root / "data/evals", root / run["database"]["path"]))
    return selected


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--results", type=Path); p.add_argument("--eval-dir", type=Path); p.add_argument("--db", type=Path); p.add_argument("--label"); p.add_argument("--manifest", type=Path); p.add_argument("--versions", nargs="+", default=["all"]); p.add_argument("--gold-corrections", type=Path, help="Approved correction overlay to apply before deterministic replay."); args = p.parse_args(argv)
    manifest = args.manifest is not None
    if manifest == all((args.results, args.eval_dir, args.db, args.label)):
        raise SystemExit("Use --manifest, or all of --results --eval-dir --db --label.")
    try:
        runs = _manifest_runs(args.manifest.resolve(), args.versions) if manifest else [(args.label, args.results, args.eval_dir, args.db)]
        for label, results, eval_dir, database in runs:
            cases, hashes = _cases(eval_dir)
            applied = []
            if args.gold_corrections:
                cases, applied = apply_gold_corrections(cases, args.gold_corrections, database)
            records, summary = replay_records(_jsonl(results), cases, database, source_path=results, source_sha256=_sha256(results), gold_hashes=hashes, corrections=applied)
            _write(args.out_dir, label, records, summary)
    except (OSError, ValueError, KeyError, ReplayError) as exc:
        raise SystemExit(f"Replay failed: {exc}") from exc


if __name__ == "__main__":
    main()
