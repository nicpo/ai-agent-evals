"""Jev as a semantic SQL judge: request building, calls, and result rows.

Pure functions plus one driver (``run_judgments``) that takes any client with a
``system_one(model=, state=, questions=)`` method, so tests can pass a stub.

Two variants are judged per attempt:

    mono      one ``choice`` question (A correct / B incorrect / C cannot determine)
    two-call  ``mono`` plus ``narrow`` (four yes/no ``noul`` probability questions)
              and ``after`` (the monolithic question again, with the narrow
              probabilities added to the state)

Branch follows the harness EX grader, as in the GPT judge: EX match -> branch A
(is the passing SQL still flawed?), EX mismatch -> branch B (is the different
result nevertheless a defensible answer?). Branch B also sends both result sets.
"""

from __future__ import annotations

from datetime import UTC, datetime
import json
import math
from pathlib import Path
from typing import Callable

from .jev_client import JevClientError, validate_choice_response


HERE = Path(__file__).resolve().parent
VARIANTS = ("mono", "two-call")


def load_config() -> dict:
    return json.loads((HERE / "config.json").read_text(encoding="utf-8"))


def load_prompt(name: str) -> dict:
    return json.loads((HERE / "prompts" / f"{name}.json").read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# --- attempts -> judge cases --------------------------------------------------

def attempt_key(record: dict) -> str:
    """Stable id of one agent attempt: question id, profile, repeat index."""
    profile = (record.get("profile") or {}).get("profile", "")
    return f"{record.get('id')}|{profile}|{record.get('run_index', 0)}"


def _last_query(record: dict) -> dict | None:
    queries = [tc for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"]
    return queries[-1] if queries else None


def build_cases(records: list[dict], questions_by_id: dict[str, dict],
                schema: str | None = None) -> tuple[list[dict], int]:
    """One judge case per attempt that has agent SQL, gold SQL and an EX verdict.

    Returns ``(cases, skipped)``. Attempts without judgeable SQL are skipped;
    they fail under every scorer, so they are not sent to Jev.
    """
    if schema is None:
        from graders.llm_judge import _schema_summary

        schema = _schema_summary()
    cases, skipped = [], 0
    for record in records:
        question = questions_by_id.get(record.get("id"))
        ex = ((record.get("graders") or {}).get("tier2b") or {}).get("execution_accuracy")
        query = _last_query(record)
        agent_sql = ((query or {}).get("parameters") or {}).get("sql")
        if not (question and question.get("gold_sql") and agent_sql and ex is not None):
            skipped += 1
            continue
        branch = "A" if ex.get("match") else "B"
        cases.append({
            "key": attempt_key(record),
            "id": record["id"],
            "profile": (record.get("profile") or {}).get("profile"),
            "run_index": record.get("run_index", 0),
            "branch": branch,
            "question": question["question"],
            "schema": schema,
            "gold_sql": question["gold_sql"],
            "agent_sql": agent_sql,
            "gold_result": question.get("gold_result") or [],
            "agent_result": ((query.get("result") or {}).get("rows") or []),
            "gpt_judge": gpt_verdict(record),
        })
    return cases, skipped


def gpt_verdict(record: dict) -> str | None:
    """Raw verdict of the harness's GPT judge (Tier 5 SQL-level branch)."""
    tier5 = (record.get("graders") or {}).get("tier5") or {}
    return tier5.get("false_positive_verdict") or tier5.get("false_negative_verdict")


# --- requests -----------------------------------------------------------------

def request_for_case(case: dict, prompt: dict) -> tuple[dict, dict]:
    branch_prompt = prompt["branch_a" if case["branch"] == "A" else "branch_b"]
    state = {
        "user_question": case["question"],
        "schema": case["schema"],
        "approved_sql": case["gold_sql"],
        "agent_sql": case["agent_sql"],
    }
    if case["branch"] == "B":
        state["approved_result"] = case.get("gold_result") or []
        state["agent_result"] = case.get("agent_result") or []
    questions = {prompt["question_name"]: {
        "type": "choice", "instructions": branch_prompt["instructions"],
        "criteria": branch_prompt["criteria"],
    }}
    return state, questions


def _narrow_questions(case: dict, decomposed: dict) -> dict[str, str]:
    return decomposed["branch_a" if case["branch"] == "A" else "branch_b"]["questions"]


def narrow_request(case: dict, decomposed: dict) -> dict:
    return {name: {"type": "noul", "instructions": text}
            for name, text in _narrow_questions(case, decomposed).items()}


def parse_narrow(response, names: list[str]) -> dict:
    try:
        answers = response.payload["answers"]
        probabilities = {name: float(answers[name]["noul"]) for name in names}
        usage = response.payload["usage"]
        model = response.payload["model"]
    except (KeyError, TypeError, ValueError) as exc:
        raise JevClientError(f"Malformed noul response: {response.payload}") from exc
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities.values()):
        raise JevClientError(f"Invalid noul probabilities: {probabilities}")
    return {"probabilities": probabilities, "usage": usage, "model": model,
            "latency_ms": response.latency_ms, "request_id": response.request_id}


def after_request(case: dict, prompt: dict, decomposed: dict, narrow: dict) -> tuple[dict, dict]:
    state, questions = request_for_case(case, prompt)
    described = _narrow_questions(case, decomposed)
    state["narrow_assessments"] = [
        {"question": described[name], "probability_yes": round(p, 3)}
        for name, p in narrow["probabilities"].items()
    ]
    questions[prompt["question_name"]]["instructions"] += decomposed["verdict_after_narrow"]["instructions_suffix"]
    return state, questions


# --- calls --------------------------------------------------------------------

def _check_model(parsed: dict, config: dict) -> None:
    if parsed["model"] != config["model"]:
        raise ValueError(f"Model mismatch: API returned {parsed['model']!r}, pinned {config['model']!r}")


def _choice_call(client, config: dict, prompt: dict, state: dict, questions: dict) -> dict:
    response = client.system_one(model=config["model"], state=state, questions=questions)
    parsed = validate_choice_response(response, prompt["question_name"], set(prompt["label_map"]))
    _check_model(parsed, config)
    return {
        "verdict": prompt["label_map"][parsed["choice"]], "confidence": parsed["confidence"],
        "probabilities": parsed["probabilities"], "usage": parsed["usage"], "model": parsed["model"],
        "latency_ms": parsed["latency_ms"], "request_id": parsed["request_id"],
    }


def judge_case(client, config: dict, case: dict, variant: str,
               prompt: dict, decomposed: dict | None = None) -> dict:
    """Judge one case. Returns ``{"mono": ..}`` or ``{"mono", "narrow", "after"}``."""
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant {variant!r}; choose from {VARIANTS}")
    state, questions = request_for_case(case, prompt)
    result = {"mono": _choice_call(client, config, prompt, state, questions)}
    if variant == "two-call":
        names = list(_narrow_questions(case, decomposed))
        response = client.system_one(model=config["model"], state=state,
                                     questions=narrow_request(case, decomposed))
        narrow = parse_narrow(response, names)
        _check_model(narrow, config)
        result["narrow"] = narrow
        after_state, after_questions = after_request(case, prompt, decomposed, narrow)
        result["after"] = _choice_call(client, config, prompt, after_state, after_questions)
    return result


def run_judgments(cases: list[dict], client, config: dict, variant: str, out_path: Path,
                  log: Callable[[str], None] = print) -> dict:
    """Judge every case not already in ``out_path``; append one row per case.

    Resumable: rerun with the same ``out_path`` to continue. Transport and
    malformed-response errors are written to ``<out_path>.failures.jsonl`` and
    the run continues; a model-ID mismatch stops it.
    """
    prompt = load_prompt("monolithic")
    decomposed = load_prompt("decomposed") if variant == "two-call" else None
    done = {row["key"] for row in load_jsonl(out_path)}
    failures_path = out_path.with_suffix(".failures.jsonl")
    counts = {"judged": 0, "resumed": len(done), "failed": 0}
    with out_path.open("a", encoding="utf-8") as out:
        for index, case in enumerate(cases, 1):
            if case["key"] in done:
                continue
            try:
                calls = judge_case(client, config, case, variant, prompt, decomposed)
            except JevClientError as exc:
                counts["failed"] += 1
                with failures_path.open("a", encoding="utf-8") as fail:
                    fail.write(json.dumps({"key": case["key"], "error": str(exc)}) + "\n")
                log(f"[{index}/{len(cases)}] {case['key']}: FAILED ({exc})")
                continue
            row = {k: case[k] for k in ("key", "id", "profile", "run_index", "branch", "gpt_judge",
                                        "question", "gold_sql", "agent_sql")}
            row |= {"variant": variant, "requested_model": config["model"],
                    "recorded_at_utc": datetime.now(UTC).isoformat(), **calls}
            out.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            out.flush()
            counts["judged"] += 1
            log(f"[{index}/{len(cases)}] {case['key']}: jev={calls['mono']['verdict']}"
                + (f" two-call={calls['after']['verdict']}" if "after" in calls else ""))
    return counts
