"""Reliability metrics over repeated attempts: mean success, pass@k, pass^k.

Method (see the repeatability post): a single run per question gives one
sample from a distribution, not the agent's true accuracy. With ``k`` repeats
per question you get three different numbers, and which one matters depends
on how the agent is deployed:

* **mean success (pass@1)**: share of *all attempts* that succeeded -- the
  expected result of a single try.
* **pass^k**: share of questions that succeeded on *every one* of k tries --
  how often you can count on the agent.
* **pass@k**: share of questions that succeeded *at least once* in k tries --
  the ceiling retries could reach, real only if something can tell a right
  answer from a wrong one.

Questions are the unit of resampling here (not attempts): a question run 3
times counts once toward pass^k/pass@k, mirroring the post's "count
questions, not attempts."
"""

from __future__ import annotations

from collections import Counter, defaultdict


def is_success(record: dict) -> bool | None:
    """Adjusted correctness for one attempt: EX pass minus false positives, plus acceptable EX fails.

    Mirrors ``eval.metrics._correctness``'s per-run aggregate, but per attempt.
    Returns None if the attempt has no EX outcome (nothing to grade against).
    """
    ex = (record["graders"].get("tier2b") or {}).get("execution_accuracy")
    if ex is None:
        return None
    tier5 = record["graders"].get("tier5") or {}
    if ex.get("match"):
        return tier5.get("false_positive_verdict") != "FALSE_POSITIVE"
    return tier5.get("false_negative_verdict") == "ACCEPTABLE"


def _profile_name(record: dict) -> str:
    return (record.get("profile") or {}).get("profile", "unknown")


def group_by_question(records: list[dict]) -> dict[str, dict[str, list[bool]]]:
    """{profile: {question_id: [success_per_run_index_in_order]}}."""
    grouped: dict[str, dict[str, dict[int, bool | None]]] = defaultdict(lambda: defaultdict(dict))
    for r in records:
        profile = _profile_name(r)
        qid = r.get("id")
        run_index = r.get("run_index", 0)
        grouped[profile][qid][run_index] = is_success(r)

    out: dict[str, dict[str, list[bool]]] = {}
    for profile, by_q in grouped.items():
        out[profile] = {}
        for qid, by_run in by_q.items():
            ordered = [by_run[i] for i in sorted(by_run)]
            out[profile][qid] = [v for v in ordered if v is not None]
    return out


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_reliability(records: list[dict]) -> dict:
    """Per-profile mean success, pass^k, pass@k, and a pass-count histogram."""
    grouped = group_by_question(records)
    out: dict[str, dict] = {}
    for profile, by_q in grouped.items():
        attempts = [v for runs in by_q.values() for v in runs]
        n_attempts = len(attempts)
        mean_success = _rate(sum(attempts), n_attempts)

        # pass^k / pass@k only make sense over questions with the same k.
        by_k: dict[int, list[list[bool]]] = defaultdict(list)
        for runs in by_q.values():
            if runs:
                by_k[len(runs)].append(runs)

        pass_hat_k: dict[int, float | None] = {}
        pass_at_k: dict[int, float | None] = {}
        histogram: dict[int, dict[int, int]] = {}
        for k, groups in by_k.items():
            n_q = len(groups)
            pass_hat_k[k] = _rate(sum(all(g) for g in groups), n_q)
            pass_at_k[k] = _rate(sum(any(g) for g in groups), n_q)
            histogram[k] = dict(Counter(sum(g) for g in groups))

        out[profile] = {
            "n_attempts": n_attempts,
            "n_questions": len(by_q),
            "mean_success": mean_success,
            "pass_hat_k": pass_hat_k,
            "pass_at_k": pass_at_k,
            "pass_count_histogram": histogram,
        }
    return out


def format_reliability(report: dict) -> str:
    lines = ["=" * 60, "RELIABILITY (mean success / pass^k / pass@k)", "=" * 60]
    for profile, r in report.items():
        lines.append("")
        lines.append(f"{profile}  (n_questions={r['n_questions']}, n_attempts={r['n_attempts']})")
        ms = r["mean_success"]
        lines.append(f"  mean success (pass@1): {ms * 100:.1f}%" if ms is not None else "  mean success: n/a")
        for k in sorted(r["pass_hat_k"]):
            hat, at = r["pass_hat_k"][k], r["pass_at_k"][k]
            lines.append(
                f"  k={k}: pass^{k}={hat * 100:.1f}%  pass@{k}={at * 100:.1f}%"
                if hat is not None and at is not None else f"  k={k}: n/a"
            )
    lines.append("=" * 60)
    return "\n".join(lines)
