"""Two independent sources of variance across repeated attempts.

Method (see the repeatability post): a score comes out of a pipeline with two
moving parts -- the agent writes SQL, then the judge grades it. Either can
vary between identical repeats, even at temperature 0. Mixing them together
misattributes measurement noise (the judge) to product behavior (the agent).

* **agent variance**: for each question, did every repeat produce the same
  SQL (ignoring whitespace/case/comments)?
* **judge variance**: when two *different* attempts happen to produce the same
  normalized SQL for the same question, did the judge grade it the same way
  both times? This only has signal where such a coincidence occurs naturally
  in the repeats already collected -- no extra judge calls are made here.
"""

from __future__ import annotations

from collections import defaultdict

import sqlglot


def _profile_name(record: dict) -> str:
    return (record.get("profile") or {}).get("profile", "unknown")


def _pred_sql(record: dict) -> str | None:
    queries = [tc for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"]
    return queries[-1]["parameters"].get("sql") if queries else None


def normalize_sql(sql: str | None) -> str | None:
    """Whitespace/case/comment-insensitive normalization via a parse + re-render.

    Falls back to a lowercased, whitespace-collapsed string if the SQL doesn't
    parse (e.g. a syntax error the agent produced) so it can still be compared.
    """
    if sql is None:
        return None
    try:
        return sqlglot.parse_one(sql, dialect="sqlite").sql(dialect="sqlite", normalize=True)
    except Exception:
        return " ".join(sql.lower().split())


def agent_sql_variance(records: list[dict]) -> dict:
    """Per profile: fraction of questions where every repeat's SQL normalized identically."""
    by_profile: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    for r in records:
        sql = normalize_sql(_pred_sql(r))
        if sql is not None:
            by_profile[_profile_name(r)][r.get("id")].add(sql)

    out = {}
    for profile, by_q in by_profile.items():
        n = len(by_q)
        identical = sum(1 for variants in by_q.values() if len(variants) == 1)
        out[profile] = {
            "n_questions": n,
            "identical_across_repeats": identical,
            "identical_rate": round(identical / n, 4) if n else None,
        }
    return out


def judge_verdict_variance(records: list[dict]) -> dict:
    """Per profile: disagreement rate among attempts sharing (question, normalized SQL).

    Groups attempts by (profile, question id, normalized SQL, judge branch) --
    branch A and B verdicts are on different scales (CORRECT/FALSE_POSITIVE vs.
    WRONG/ACCEPTABLE/...) so they're never pooled together -- and reports how
    often a group with >=2 members isn't unanimous. Groups of size 1 carry no
    variance signal and are excluded from the denominator.
    """
    groups: dict[str, dict[tuple, list[str | None]]] = defaultdict(lambda: defaultdict(list))
    for r in records:
        tier5 = r["graders"].get("tier5") or {}
        branch = tier5.get("branch")
        if branch is None:
            continue
        sql = normalize_sql(_pred_sql(r))
        verdict = tier5.get("false_positive_verdict") if branch == "A" else tier5.get("false_negative_verdict")
        key = (r.get("id"), sql, branch)
        groups[_profile_name(r)][key].append(verdict)

    out = {}
    for profile, by_key in groups.items():
        comparable = {k: v for k, v in by_key.items() if len(v) >= 2}
        n = len(comparable)
        disagreeing = sum(1 for v in comparable.values() if len(set(v)) > 1)
        out[profile] = {
            "n_comparable_groups": n,
            "disagreeing_groups": disagreeing,
            "disagreement_rate": round(disagreeing / n, 4) if n else None,
            "note": (
                "n_comparable_groups counts (question, normalized SQL, branch) "
                "combinations that recurred naturally across repeats; with few "
                "repeats on a small eval set this can be 0 -- that means no "
                "signal was available, not that the judge is perfectly stable."
            ),
        }
    return out
