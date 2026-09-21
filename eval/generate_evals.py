"""Generate *candidate* text-to-SQL eval cases for human review.

This module never changes the canonical eval set.  It asks a configured LLM to
propose natural-language questions and read-only SQLite SQL, validates the SQL
against the local database, and writes executable candidates with
``review_status: needs_review``.  A human must review every question/SQL/result
trio before copying it into ``data/evals/questions-*.jsonl``.

Example:

    python -m eval.generate_evals --model gpt-5-4-mini --count 12 \
        --out data/evals/candidates/roman_empire_candidates.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent import db
from agent.llm import get_chat_model
from agent.tools import Toolbox
from config import AGENT_MODEL, DB_PATH, MODELS


_DIFFICULTIES = {"easy", "medium", "hard", "extra-hard"}
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_response(text: str) -> list[dict[str, Any]]:
    """Extract the list of proposed cases from an LLM JSON response."""
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        # A few models add one explanatory sentence despite the JSON-only
        # instruction.  Recover only the outer object/array, never evaluate it.
        starts = [i for i in (cleaned.find("{"), cleaned.find("[")) if i >= 0]
        if not starts:
            raise ValueError("model response did not contain JSON") from None
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end <= start:
            raise ValueError("model response contained incomplete JSON") from None
        try:
            payload = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"could not parse model JSON: {exc}") from exc

    cases = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(cases, list) or not all(isinstance(case, dict) for case in cases):
        raise ValueError("model JSON must be an object with a 'cases' list")
    return cases


def _generation_prompt(count: int, schema: dict, samples: dict) -> str:
    """Build the constrained prompt sent to the generator model."""
    context = json.dumps({"schema": schema, "sample_rows": samples}, indent=2, default=str)
    return f"""You are creating candidate benchmark cases for a SQLite text-to-SQL
agent. Generate exactly {count} diverse cases using only the database context
below.

Each case must have a precise, natural-language question and one SQLite
read-only SELECT or WITH...SELECT query that answers exactly that question.
The question must not reveal SQL syntax, table names, or column names unless a
real user would naturally use them. Do not add unrequested output columns. Do
not write ambiguous questions. Include a mix of easy, medium, hard, and
extra-hard cases; use joins, aggregation, NULL/empty results, ties, and known
value discovery only when the database supports them. A valid empty result is
fine. Never use INSERT, UPDATE, DELETE, DDL, PRAGMA, ATTACH, or multiple SQL
statements.

Return JSON only, with this exact shape:
{{
  "cases": [
    {{
      "question": "...",
      "gold_sql": "SELECT ...",
      "difficulty": "easy|medium|hard|extra-hard",
      "adversarial": false,
      "rationale": "one short sentence explaining why this is a useful case"
    }}
  ]
}}

Database context:
{context}
"""


def _validated_record(
    proposal: dict[str, Any],
    index: int,
    prefix: str,
    toolbox: Toolbox,
    model: str,
    db_hash: str,
) -> dict[str, Any]:
    """Validate one model proposal and convert it to a reviewable eval record."""
    question = proposal.get("question")
    gold_sql = proposal.get("gold_sql")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("missing non-empty question")
    if not isinstance(gold_sql, str) or not gold_sql.strip():
        raise ValueError("missing non-empty gold_sql")

    result = toolbox.run_query(gold_sql)
    if result["error"]:
        raise ValueError(f"SQL did not validate: {result['error']}")

    difficulty = proposal.get("difficulty", "medium")
    if difficulty not in _DIFFICULTIES:
        difficulty = "medium"

    return {
        "id": f"{prefix}_{index:03d}",
        "question": question.strip(),
        "gold_sql": gold_sql.strip(),
        "gold_result": result["rows"],
        "difficulty": difficulty,
        "adversarial": bool(proposal.get("adversarial", False)),
        "run_llm_judge": True,
        "review_status": "needs_review",
        "generation": {
            "model": model,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "database_sha256": db_hash,
            "rationale": str(proposal.get("rationale", "")).strip(),
            "result_columns": result["columns"],
            "result_row_count": result["row_count"],
        },
    }


def generate_candidates(
    *, model: str, count: int, prefix: str, db_path: Path = DB_PATH
) -> tuple[list[dict[str, Any]], list[str]]:
    """Ask ``model`` for candidates and validate them without writing files."""
    if count < 1:
        raise ValueError("count must be at least 1")
    if count > 50:
        raise ValueError("count must be at most 50; generate and review smaller batches")

    db_hash = _file_sha256(db_path)
    toolbox = Toolbox(db.connect(db_path))
    try:
        samples = {
            table["name"]: toolbox.get_sample_rows(table["name"], n=5)["rows"]
            for table in db.SCHEMA["tables"]
        }
        response = get_chat_model(model).invoke(_generation_prompt(count, db.SCHEMA, samples))
        proposals = _json_response((response.text or "").strip())
        if len(proposals) != count:
            raise ValueError(f"model returned {len(proposals)} cases; expected {count}")

        candidates: list[dict[str, Any]] = []
        rejected: list[str] = []
        for index, proposal in enumerate(proposals, 1):
            try:
                candidates.append(
                    _validated_record(proposal, index, prefix, toolbox, model, db_hash)
                )
            except ValueError as exc:
                rejected.append(f"{prefix}_{index:03d}: {exc}")
        return candidates, rejected
    finally:
        toolbox.close()


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate human-review-required candidate text-to-SQL eval cases."
    )
    parser.add_argument("--model", default=AGENT_MODEL, choices=list(MODELS),
                        help="Configured LLM used to draft questions and gold SQL.")
    parser.add_argument("--count", type=int, default=10,
                        help="Number of candidates to request (1-50, default: 10).")
    parser.add_argument("--prefix", default="candidate",
                        help="ID prefix for generated cases (default: candidate).")
    parser.add_argument("--out", type=Path, required=True,
                        help="New JSONL path for candidates; existing files are never overwritten.")
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    if args.out.exists():
        raise SystemExit(f"Refusing to overwrite existing candidate file: {args.out}")

    try:
        candidates, rejected = generate_candidates(
            model=args.model, count=args.count, prefix=args.prefix
        )
    except (KeyError, ValueError) as exc:
        raise SystemExit(f"Generation failed: {exc}") from exc

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as fh:
        for candidate in candidates:
            fh.write(json.dumps(candidate, default=str) + "\n")

    print(f"Wrote {len(candidates)} review-required candidates to {args.out}")
    if rejected:
        print(f"Rejected {len(rejected)} invalid proposal(s):", file=sys.stderr)
        for rejection in rejected:
            print(f"  {rejection}", file=sys.stderr)
    print("Review every case before copying it into data/evals/questions-*.jsonl.")


if __name__ == "__main__":
    main()
