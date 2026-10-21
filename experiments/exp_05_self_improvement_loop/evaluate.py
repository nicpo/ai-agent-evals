"""Evaluate a candidate on a split: agent attempts, EX, frozen Sol judge, adjusted success.

Success (the judge-adjusted metric of exp 03, without the answer-quality rubrics):
  - agent error or no SQL -> fail
  - EX match -> branch A judge; success iff verdict == CORRECT
  - EX miss  -> branch B judge; success iff verdict == ACCEPTABLE
  - any other / unparseable verdict -> fail, flagged as unresolved
The judge prompts come from the profile's judge bundle (config/judge_prompts/default.yaml for the
shipped profiles). Gold is the question's `gold_sql`. Results are appended per attempt to a JSONL
file and a rerun skips finished attempts.

    python -m experiments.exp_05_self_improvement_loop.evaluate run --candidate c000_baseline \
        --split regression --repeats 2 [--limit N] [--workers 4] [--run-name NAME] [--profile toy]
"""

from __future__ import annotations

import argparse
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

from agent import db
from agent.llm import get_chat_model
from experiments.exp_05_self_improvement_loop import candidates as cand, models
from experiments.exp_05_self_improvement_loop.agent_loop import LoopToolbox, run_agent_loop
from experiments.exp_05_self_improvement_loop.workspace import SPLITS, Workspace, add_workspace_args, from_args
from graders.execution import execution_accuracy
from graders.llm_judge import _last_query_rows, _parse_verdict, _schema_summary

VALID = {"A": {"CORRECT", "FALSE_POSITIVE"}, "B": {"ACCEPTABLE", "WRONG"}}
SUCCESS_VERDICT = {"A": "CORRECT", "B": "ACCEPTABLE"}
_write_lock = threading.Lock()


def render_judge_prompt(templates: dict, branch: str, case: dict, agent_sql: str, agent_rows: list) -> str:
    values = {"question": case["question"], "schema_summary": _schema_summary(),
              "gold_sql": case["gold_sql"], "agent_sql": agent_sql}
    if branch == "B":
        values.update(gold_result=json.dumps(case.get("gold_result") or [], default=str),
                      agent_result=json.dumps(agent_rows, default=str))
    return templates["branch_a_false_positive" if branch == "A" else "branch_b_false_negative"].format(**values)


class Evaluator:
    def __init__(self, ws: Workspace, candidate: cand.Candidate, judge_entry: dict = models.JUDGE,
                 agent_entry: dict = models.AGENT):
        self.ws, self.candidate, self.agent_entry = ws, candidate, agent_entry
        self.templates = ws.judge_templates()
        self.judge_model = get_chat_model(judge_entry)

    def _judge(self, prompt: str) -> dict:
        from langchain_core.messages import HumanMessage

        last = None
        for _ in range(2):
            try:
                response = self.judge_model.invoke([HumanMessage(content=prompt)])
                raw = (response.text or "").strip()
                usage = dict(getattr(response, "usage_metadata", None) or {})
                return {**_parse_verdict(raw), "usage": usage}
            except Exception as exc:  # noqa: BLE001
                last = f"{type(exc).__name__}: {exc}"
        return {"error": last}

    def attempt(self, case: dict, repeat: int) -> dict:
        toolbox = LoopToolbox.open_with(cand.ENABLED_TOOLS, self.candidate.descriptions)
        try:
            trace = run_agent_loop(case["question"], self.agent_entry, toolbox, self.candidate.prompt,
                                   self.ws.max_tool_calls)
        finally:
            toolbox.close()
        agent_sql = trace.last_query_sql()
        rec = {"id": case["id"], "repeat": repeat, "candidate": self.candidate.name,
               "candidate_sha256": self.candidate.sha256, "difficulty": case.get("difficulty"),
               "question": case["question"], "agent_sql": agent_sql, "trace": trace.to_dict(),
               "ex_match": None, "branch": None, "verdict": None, "reason": None, "judge_usage": {},
               "unresolved": False, "success": 0, "recorded_at_utc": datetime.now(UTC).isoformat()}
        if trace.error or not agent_sql:
            return rec
        conn = db.connect()
        try:
            rec["ex_match"] = bool(execution_accuracy(case["gold_sql"], agent_sql, conn)["match"])
        finally:
            conn.close()
        branch = "A" if rec["ex_match"] else "B"
        prompt = render_judge_prompt(self.templates, branch, case, agent_sql, _last_query_rows(trace))
        verdict = self._judge(prompt)
        if "error" in verdict:
            rec["judge_error"] = verdict["error"]
            return rec
        rec.update(branch=branch, verdict=verdict["verdict"], reason=verdict["reason"],
                   judge_usage=verdict["usage"], unresolved=verdict["verdict"] not in VALID[branch])
        rec["success"] = int(verdict["verdict"] == SUCCESS_VERDICT[branch])
        return rec


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def run(ws: Workspace, candidate: cand.Candidate, split: str, repeats: int, out_path: Path, workers: int = 4,
        limit: int | None = None, agent_entry: dict = models.AGENT, evaluator=None) -> list[dict]:
    """Run missing attempts and return the records for exactly this split and repeat range.

    `evaluator` (anything with .attempt(case, repeat)) replaces the live one; the tests use it."""
    cases = ws.cases(split)[: limit or None]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {(r["id"], r["repeat"]) for r in load_records(out_path) if "judge_error" not in r}
    todo = [(c, r) for r in range(1, repeats + 1) for c in cases if (c["id"], r) not in done]
    print(f"{candidate.name} on {split}: {len(done)} done, {len(todo)} to run ({workers} workers)", flush=True)
    ev = evaluator or Evaluator(ws, candidate, agent_entry=agent_entry)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(ev.attempt, c, r): (c["id"], r) for c, r in todo}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if "judge_error" in rec:
                print("JUDGE ERROR", rec["id"], rec["judge_error"][:150], flush=True)
                continue  # not saved, so a rerun retries it
            with _write_lock, out_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, default=str) + "\n")
            if n % 10 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)
    wanted = {(c["id"], r) for c in cases for r in range(1, repeats + 1)}
    return [r for r in load_records(out_path) if (r["id"], r["repeat"]) in wanted]


def summarize(records: list[dict]) -> dict:
    if not records:
        return {"attempts": 0}
    n = len(records)
    a_in = sum(r["trace"]["total_input_tokens"] for r in records)
    a_out = sum(r["trace"]["total_output_tokens"] for r in records)
    j_in = sum((r["judge_usage"] or {}).get("input_tokens", 0) for r in records)
    j_out = sum((r["judge_usage"] or {}).get("output_tokens", 0) for r in records)
    by_q: dict[str, list[int]] = {}
    for r in records:
        by_q.setdefault(r["id"], []).append(r["success"])
    parts = [models.cost(records[0]["trace"]["model"], a_in, a_out), models.cost(models.JUDGE["id"], j_in, j_out)]
    return {
        "attempts": n, "questions": len(by_q),
        "success_rate": sum(r["success"] for r in records) / n,
        "ex_rate": sum(bool(r["ex_match"]) for r in records) / n,
        "agent_errors": sum(bool(r["trace"].get("error")) for r in records),
        "tool_limit_hits": sum(r["trace"].get("error") == "exceeded_max_tool_calls" for r in records),
        "no_sql": sum(not r["agent_sql"] for r in records),
        "unresolved_verdicts": sum(r["unresolved"] for r in records),
        "mean_tool_calls": sum(len(r["trace"]["tool_calls"]) for r in records) / n,
        "agent_tokens": {"input": a_in, "output": a_out}, "judge_tokens": {"input": j_in, "output": j_out},
        "cost_usd": None if None in parts else sum(parts),
        "never_pass": sorted(q for q, v in by_q.items() if sum(v) == 0),
        "unstable": sorted(q for q, v in by_q.items() if 0 < sum(v) < len(v)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("run", "summary"))
    add_workspace_args(ap)
    ap.add_argument("--candidate", default="c000_baseline")
    ap.add_argument("--split", choices=SPLITS, required=True)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--run-name")
    args = ap.parse_args()
    ws = from_args(args)
    candidate = ws.candidate(args.candidate)
    out_path = ws.root / "runs" / f"{args.run_name or f'{args.candidate}__{args.split}'}.jsonl"
    if args.command == "run":
        records = run(ws, candidate, args.split, args.repeats, out_path, args.workers, args.limit)
    else:
        records = load_records(out_path)
    print(json.dumps(summarize(records), indent=2))


if __name__ == "__main__":
    main()
