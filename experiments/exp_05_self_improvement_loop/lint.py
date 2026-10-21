"""Deterministic lint for candidate prompts / tool descriptions, run before spending eval money.

Rejects (violations):
  - question ids (easy_001, hard_012, ...) or test-family ids
  - mentions of grading or evaluation machinery
  - 8+ consecutive words copied from any of the profile's questions
  - 40+ characters of any gold SQL (whitespace/case normalized)
  - quoted database value literals ('governed', "active", ...): column values belong to the
    database, not the prompt; the agent can discover them with its tools
  - candidate text (prompt + tool descriptions) over MAX_TOTAL_CHARS
The test set is deliberately NOT passed in: nothing in the loop process reads test questions.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from functools import lru_cache

from agent import db
from config import DB_PATH
from experiments.exp_05_self_improvement_loop.candidates import Candidate

MAX_TOTAL_CHARS = 3500  # prompt + tool descriptions; about 2x the baseline prompt, rounded up
WINDOW_WORDS = 8
MIN_SQL_FRAGMENT = 40

_ID_RE = re.compile(r"\b(?:easy|medium|hard|extra|toy)_\w*\d{3}\b|\bfam\d{2}(?:_\w+)?\b", re.I)
_GRADER_RE = re.compile(
    r"\b(?:judge|grader|grading|graded|gold|ground truth|benchmark|evaluation|eval set|test set|"
    r"validation set|execution accuracy|scoring|score)\b", re.I)
_QUOTED_RE = re.compile(r"""(['"`])([^'"`\n]{2,40})\1""")


def _norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _norm_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.lower()).strip()


def candidate_text(c: Candidate) -> str:
    return "\n\n".join([c.prompt, *(c.descriptions[k] for k in sorted(c.descriptions))])


@lru_cache(maxsize=1)
def _db_values() -> frozenset[str]:
    """Distinct values of low-cardinality text columns, lower-cased."""
    conn = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
    try:
        values: set[str] = set()
        for table in db.TABLE_NAMES:
            cols = [(r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({table})")]
            for name, ctype in cols:
                if ctype.upper() not in ("TEXT", "VARCHAR", ""):
                    continue
                rows = [r[0] for r in conn.execute(f"SELECT DISTINCT {name} FROM {table}") if r[0] is not None]
                if 0 < len(rows) <= 30:
                    values.update(str(v).lower() for v in rows)
        return frozenset(values)
    finally:
        conn.close()


@dataclass
class LintResult:
    violations: list[str]

    @property
    def ok(self) -> bool:
        return not self.violations


def lint(candidate: Candidate, cases: list[dict]) -> LintResult:
    """`cases`: the questions the loop may have seen (the profile's full question set)."""
    text = candidate_text(candidate)
    out: list[str] = []

    for m in _ID_RE.finditer(text):
        out.append(f"question or family id: {m.group(0)}")
    for m in _GRADER_RE.finditer(text):
        out.append(f"mentions grading/evaluation: '{m.group(0)}'")

    words = _norm_words(text)
    windows = {tuple(words[i:i + WINDOW_WORDS]) for i in range(max(0, len(words) - WINDOW_WORDS + 1))}
    for case in cases:
        q = _norm_words(case["question"])
        if any(tuple(q[i:i + WINDOW_WORDS]) in windows for i in range(max(0, len(q) - WINDOW_WORDS + 1))):
            out.append("copies 8+ consecutive words from a regression question")
            break
    flat = _norm_sql(text)
    for case in cases:
        g = _norm_sql(case["gold_sql"])
        for i in range(0, max(1, len(g) - MIN_SQL_FRAGMENT + 1), 10):
            frag = g[i:i + MIN_SQL_FRAGMENT]
            if len(frag) == MIN_SQL_FRAGMENT and frag in flat:
                out.append("contains a 40+ character fragment of a gold SQL query")
                break

    values = _db_values()
    for m in _QUOTED_RE.finditer(text):
        if m.group(2).strip().lower() in values:
            out.append(f"quoted database value literal: {m.group(0)}")

    if len(text) > MAX_TOTAL_CHARS:
        out.append(f"too long: {len(text)} chars, limit {MAX_TOTAL_CHARS} (prompt plus both tool descriptions)")
    if not candidate.prompt.strip():
        out.append("empty system prompt")
    seen, unique = set(), []
    for v in out:
        if v not in seen:
            seen.add(v)
            unique.append(v)
    return LintResult(unique)
