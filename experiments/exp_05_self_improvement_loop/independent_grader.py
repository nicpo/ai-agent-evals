"""Independent grader: calibrate a second judge on your labels, then re-grade the final arms with it.

The loop optimizes against one judge (Sol), so held-out questions alone cannot separate a better
agent from an agent that judge likes. After the loop, a judge from another model family re-grades
the saved attempts in final/ (same judge prompts, same success rule as evaluate.py). It is never
used inside the loop and never influences a decision. No new agent runs: only judge calls.

The model is any LangChain `provider:model` id that `init_chat_model` can build, for example
`bedrock_converse:moonshot.kimi-k2-thinking` (pip install -r experiments/exp_05_self_improvement_loop/requirements.txt;
Bedrock credentials in .env, see the README) or `openai:gpt-5.4`. Pick a family unlike the in-loop judge.

    python -m experiments.exp_05_self_improvement_loop.independent_grader calibrate --model ID --labels labels.jsonl
    python -m experiments.exp_05_self_improvement_loop.independent_grader regrade   --model ID --test-profile toy_test
    python -m experiments.exp_05_self_improvement_loop.independent_grader report    --model ID --test-profile toy_test

Calibration labels (JSONL, one line per attempt you judged by hand):
    {"candidate": "c000_baseline", "split": "validation", "id": "...", "repeat": 1, "label": "correct"}
`label` is `correct` or `incorrect`; `split` is dev, validation or regression (attempts under loop/ or
final/). Labelling attempts from the test set is refused: that would let you see the data you are about to
report on. Label without looking at the judges' verdicts. Selection rule, fixed before the final results
exist: use the judge only if it agrees with your labels on at least 90% of the cases where it gave a verdict.
`regrade` refuses a model below that bar, and a below-bar result is reported as "no independent grader".
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from experiments.exp_05_self_improvement_loop import evaluate, final_eval
from experiments.exp_05_self_improvement_loop.loop import SEED, load_jsonl
from experiments.exp_05_self_improvement_loop.workspace import Workspace, add_workspace_args, from_args

MIN_AGREEMENT = 0.90
MAX_TOKENS = 8192  # thinking models spend many tokens before the verdict
VALID = evaluate.VALID
SUCCESS_VERDICT = evaluate.SUCCESS_VERDICT
HUMAN_PASS = {"correct": 1, "incorrect": 0}
LABELS = "CORRECT|FALSE_POSITIVE|ACCEPTABLE|WRONG|GOLD_ERROR|HUMAN_REVIEW"
VERDICT_LINE = re.compile(rf"VERDICT[*_`\s]*:[*_`\s]*({LABELS})\b", re.I)
_lock = threading.Lock()


def parse_verdict(text: str) -> dict:
    """A verdict is the word VERDICT, a colon, then one of the known labels (markdown around them is ignored).
    Headings such as '## Verdict' do not count. If the model states the verdict more than once, the last
    statement wins. Anything else is unresolved (verdict None)."""
    matches = VERDICT_LINE.findall(text)
    r = re.search(r"REASON\s*:\s*[*_`]*\s*(.+)", text, re.I | re.S)
    return {"verdict": matches[-1].upper() if matches else None, "reason": r.group(1).strip() if r else "", "raw": text}


def get_model(model: str, max_tokens: int = MAX_TOKENS):
    from langchain.chat_models import init_chat_model

    kwargs = {"max_tokens": max_tokens}
    if model.startswith("bedrock"):  # Bedrock model ids are per region; default like the AWS CLI's common choice
        kwargs["region_name"] = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1"
    return init_chat_model(model, **kwargs)


def judge_once(model, prompt: str) -> dict:
    from langchain_core.messages import HumanMessage

    last = None
    for _ in range(3):
        try:
            response = model.invoke([HumanMessage(content=prompt)])
            raw = (response.text or "").strip()
            usage = dict(getattr(response, "usage_metadata", None) or {})
            return {**parse_verdict(raw), "usage": usage}
        except Exception as exc:  # noqa: BLE001
            last = f"{type(exc).__name__}: {exc}"
    return {"error": last}


def label_for(model: str) -> str:
    """File-name label for a model id."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "-", model.split(":", 1)[-1]).strip("-")


def calib_path(ws: Workspace, name: str) -> Path:
    return ws.independent_dir / f"calibration_{name}.jsonl"


def regrade_path(ws: Workspace, name: str, arm: str, split: str) -> Path:
    return ws.independent_dir / f"{name}__{arm}__{split}.jsonl"


def append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _lock, path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def run_pool(todo: list, work, workers: int, path: Path, key: str) -> None:
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(work, item) for item in todo]
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            if "error" in rec:
                print("ERROR", rec[key], rec["error"][:150], flush=True)
                continue  # not saved, so a rerun retries it
            append(path, rec)
            if n % 20 == 0 or n == len(todo):
                print(f"  {n}/{len(todo)}", flush=True)


def last_query_rows(trace: dict) -> list:
    queries = [c for c in trace["tool_calls"] if c["tool_name"] == "run_query"]
    return (queries[-1]["result"].get("rows", []) or []) if queries else []


def judge_prompt(ws: Workspace, templates: dict, rec: dict, case: dict) -> str:
    return evaluate.render_judge_prompt(templates, rec["branch"], case, rec["agent_sql"], last_query_rows(rec["trace"]))


# ---------------------------------------------------------------- calibration
def find_attempt(ws: Workspace, label: dict) -> dict:
    if label["split"] == "test":
        raise SystemExit("labels for test-set attempts are refused (see the module docstring)")
    for folder in (ws.loop_dir, ws.final_dir):
        for rec in load_jsonl(folder / f"{label['candidate']}__{label['split']}.jsonl"):
            if (rec["id"], rec["repeat"]) == (label["id"], label["repeat"]):
                return rec
    raise SystemExit(f"no saved attempt for label {label}")


def cmd_calibrate(ws: Workspace, model: str, name: str, labels_path: Path, workers: int, max_tokens: int) -> None:
    labels = load_jsonl(labels_path)
    if not labels:
        raise SystemExit(f"{labels_path} has no labels")
    if any(l["split"] == "test" for l in labels):
        raise SystemExit("labels for test-set attempts are refused (see the module docstring)")
    path = calib_path(ws, name)
    ref = lambda l: f"{l['candidate']}|{l['split']}|{l['id']}|{l['repeat']}"  # noqa: E731
    done = {r["ref"] for r in load_jsonl(path)}
    todo = [l for l in labels if ref(l) not in done]
    print(f"{name}: {len(done)} done, {len(todo)} to run", flush=True)
    judge, templates = get_model(model, max_tokens), ws.judge_templates()
    cases = {s: {c["id"]: c for c in ws.cases(s)} for s in {l["split"] for l in todo}}

    def work(label: dict) -> dict:
        rec = find_attempt(ws, label)
        if not rec["branch"]:  # no SQL or agent error: fails under every judge, nothing to calibrate
            return {"ref": ref(label), "error": "attempt has no SQL to judge; drop this label"}
        v = judge_once(judge, judge_prompt(ws, templates, rec, cases[label["split"]][label["id"]]))
        if "error" in v:
            return {"ref": ref(label), "error": v["error"]}
        return {"ref": ref(label), "branch": rec["branch"], "verdict": v["verdict"], "reason": v["reason"],
                "raw": v["raw"], "usage": v["usage"], "human": HUMAN_PASS[label["label"]], "model": model}

    run_pool(todo, work, workers, path, "ref")
    summary = calibration_summary(name, load_jsonl(path))
    ws.independent_dir.joinpath(f"calibration_{name}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_calibration(summary)


def calibration_summary(name: str, recs: list[dict]) -> dict:
    resolved = [r for r in recs if r["verdict"] in VALID[r["branch"]]]
    pred = lambda r: int(r["verdict"] == SUCCESS_VERDICT[r["branch"]])  # noqa: E731
    agree = sum(pred(r) == r["human"] for r in resolved)
    human_bad = [r for r in resolved if r["human"] == 0]
    human_good = [r for r in resolved if r["human"] == 1]
    return {
        "name": name, "model": recs[0]["model"] if recs else None, "cases": len(recs), "resolved": len(resolved),
        "agreement": agree / len(resolved) if resolved else None,
        "false_accept_rate": sum(pred(r) == 1 for r in human_bad) / len(human_bad) if human_bad else None,
        "false_reject_rate": sum(pred(r) == 0 for r in human_good) / len(human_good) if human_good else None,
        "unresolved": len(recs) - len(resolved),
        "meets_threshold": bool(resolved) and agree / len(resolved) >= MIN_AGREEMENT,
    }


def print_calibration(s: dict) -> None:
    pct = lambda x: "n/a" if x is None else f"{x * 100:.1f}%"  # noqa: E731
    print(f"  {s['name']:<28} agreement {pct(s['agreement'])} on {s['resolved']}/{s['cases']} resolved   "
          f"false-accept {pct(s['false_accept_rate'])}  false-reject {pct(s['false_reject_rate'])}  "
          f"{'MEETS' if s['meets_threshold'] else 'below'} the {MIN_AGREEMENT:.0%} bar")


def load_calibration(ws: Workspace, name: str) -> dict | None:
    path = ws.independent_dir / f"calibration_{name}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ---------------------------------------------------------------- regrade and report
def cmd_regrade(ws: Workspace, model: str, name: str, workers: int, max_tokens: int) -> None:
    calib = load_calibration(ws, name)
    if calib is None:
        raise SystemExit(f"no calibration for {name!r}; run calibrate first")
    if not calib["meets_threshold"]:
        raise SystemExit(f"{name} is below the {MIN_AGREEMENT:.0%} agreement bar; not allowed to grade")
    judge, templates = get_model(model, max_tokens), ws.judge_templates()
    for split in final_eval.REPEATS:
        cases = {c["id"]: c for c in ws.cases(split)}
        for arm in final_eval.arms(ws):
            path = regrade_path(ws, name, arm, split)
            done = {(r["id"], r["repeat"]) for r in load_jsonl(path)}
            todo = [r for r in load_jsonl(final_eval.path_for(ws, arm, split))
                    if r["branch"] and (r["id"], r["repeat"]) not in done]
            print(f"{arm} {split}: {len(done)} done, {len(todo)} to judge", flush=True)

            def work(rec: dict) -> dict:
                key = {"id": rec["id"], "repeat": rec["repeat"]}
                v = judge_once(judge, judge_prompt(ws, templates, rec, cases[rec["id"]]))
                if "error" in v:
                    return {**key, "error": v["error"]}
                return {**key, "branch": rec["branch"], "verdict": v["verdict"], "reason": v["reason"],
                        "raw": v["raw"], "usage": v["usage"], "model": model}

            run_pool(todo, work, workers, path, "id")


def merged(ws: Workspace, name: str, arm: str, split: str) -> list[dict]:
    """Final records with the independent verdict attached. Attempts the in-loop judge never judged
    (agent error, no SQL) fail under both."""
    indep = {(r["id"], r["repeat"]): r for r in load_jsonl(regrade_path(ws, name, arm, split))}
    out = []
    for rec in load_jsonl(final_eval.path_for(ws, arm, split)):
        v = indep.get((rec["id"], rec["repeat"]))
        out.append({**rec, "indep_success": int(v is not None and v["verdict"] == SUCCESS_VERDICT[v["branch"]]),
                    "indep_unresolved": bool(v) and v["verdict"] not in VALID[v["branch"]],
                    "indep_missing": rec["branch"] is not None and v is None})
    return out


def cmd_report(ws: Workspace, name: str) -> None:
    arms = final_eval.arms(ws)
    rng = random.Random(SEED)
    summary = load_calibration(ws, name)
    if summary:
        print_calibration(summary)
    for split in final_eval.REPEATS:
        data = {a: merged(ws, name, a, split) for a in arms}
        if not all(data.values()):
            print(f"\n== {split}: missing final results")
            continue
        clusters = ws.clusters(split)
        print(f"\n== {split} ({final_eval.REPEATS[split]} repeats; clusters resampled by "
              f"{'family' if split == 'test' else 'question'})")
        missing = sum(r["indep_missing"] for a in arms for r in data[a])
        if missing:
            print(f"  WARNING: {missing} attempts have no independent verdict yet; run regrade again")
        for field, label in (("success", "In-loop judge success"), ("indep_success", f"{name} success")):
            print(f"  {label}")
            final_eval.print_comparison(arms, {a: final_eval.cluster_means(data[a], clusters, field) for a in arms}, rng)
        print("  In-loop judge vs independent, per attempt")
        for a in arms:
            judged = [r for r in data[a] if r["branch"] and not r["indep_missing"]]
            agree = sum(r["success"] == r["indep_success"] for r in judged)
            loop_only = sum(r["success"] == 1 and r["indep_success"] == 0 for r in judged)
            indep_only = sum(r["success"] == 0 and r["indep_success"] == 1 for r in judged)
            unres = sum(r["indep_unresolved"] for r in judged)
            print(f"    {a:<22} agree {agree}/{len(judged)}   in-loop-pass only {loop_only}   "
                  f"independent-pass only {indep_only}   independent unresolved {unres}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("calibrate", "regrade", "report"))
    add_workspace_args(ap)
    ap.add_argument("--model", required=True, help="LangChain provider:model id of the independent judge")
    ap.add_argument("--name", help="label for result files (default: derived from --model)")
    ap.add_argument("--labels", type=Path, help="calibrate: your hand labels (JSONL)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=MAX_TOKENS,
                    help="output cap per judge call; raise it if a thinking model is cut off before the verdict")
    a = ap.parse_args()
    ws, name = from_args(a), a.name or label_for(a.model)
    if a.command == "calibrate":
        if not a.labels:
            ap.error("calibrate needs --labels")
        cmd_calibrate(ws, a.model, name, a.labels, a.workers, a.max_tokens)
    elif a.command == "regrade":
        cmd_regrade(ws, a.model, name, a.workers, a.max_tokens)
    else:
        cmd_report(ws, name)


if __name__ == "__main__":
    main()
