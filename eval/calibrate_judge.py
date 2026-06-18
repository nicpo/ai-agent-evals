"""Calibrate the Branch A (false-positive) judge by measuring self-consistency.

Branch A only runs on cases where EX passed (a passing result set that might still
be the *wrong* SQL). This tool samples those cases from a finished run, re-runs
``BRANCH_A_FALSE_POSITIVE`` N times on each, and reports how often the majority
verdict dominates — a proxy for how noisy the judge is.

    python -m eval.calibrate_judge output/results/output_2026-06-15_12-00.jsonl \
        --sample 18 --repeats 3 --judge-model gpt-5-4-mini

Interpretation: with ``repeats=3`` a 3-0 split is consistent and 2-1 is noisy;
self-consistency is the average majority share across the sampled cases (1.0 =
every case unanimous). This measures judge *stability*, not agreement with human
labels — and it is only meaningful if the judge samples non-deterministically
(reasoning models, or temperature > 0). A temperature-0 judge will look ~100%
consistent by construction.

The run's result records don't store ``gold_sql`` (only the judge needs it), so
each EX=True record is joined back to its eval question by ``id``.
"""

from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict

from config import CALIBRATION_DIR, JUDGE_MODEL, MODELS
from eval.metrics import load_records
from eval.runner import load_questions
from graders.llm_judge import Judge


def _check_deterministic_judge(model_key: str) -> None:
    """Warn loudly and ask for confirmation if the judge is temperature-0.

    A temperature-0 judge always gives the same answer, so self-consistency
    will be ~100 % by construction — a meaningless result.  Only sampling
    judges (OpenAI reasoning models, or temperature > 0) produce real signal.
    """
    kwargs = MODELS[model_key].get("kwargs", {})
    if kwargs.get("temperature") != 0:
        return  # sampling judge — fine to proceed

    bar = "!" * 70
    print(bar)
    print("WARNING: DETERMINISTIC JUDGE DETECTED — RESULTS WILL BE MEANINGLESS")
    print(bar)
    print()
    print(f"  Model '{model_key}' has temperature=0.")
    print("  A temperature-0 judge always produces the same verdict for the")
    print("  same input, so self-consistency will be ~100 % by construction.")
    print("  This tells you nothing about how noisy or reliable the judge is.")
    print()
    print("  For a real signal, use an OpenAI reasoning model (no temperature")
    print("  kwarg) or any model with temperature > 0.")
    print()
    response = input("  Proceed anyway? [y/N] ").strip().lower()
    if response not in {"y", "yes"}:
        raise SystemExit("Aborted.  Re-run with a sampling judge.")
    print()


def _pred_sql(record: dict) -> str | None:
    """The last run_query SQL from a stored trace dict (the agent's prediction)."""
    queries = [tc for tc in record["trace"]["tool_calls"] if tc["tool_name"] == "run_query"]
    return queries[-1]["parameters"].get("sql") if queries else None


def branch_a_candidates(records: list[dict], cases_by_id: dict[str, dict]) -> list[dict]:
    """Records that EX passed and have both gold SQL and a predicted query."""
    out = []
    for r in records:
        ex = (r["graders"].get("tier2b") or {}).get("execution_accuracy") or {}
        if not ex.get("match"):
            continue
        case = cases_by_id.get(r.get("id")) or {}
        gold = case.get("gold_sql")
        pred = _pred_sql(r)
        if gold and pred:
            out.append({
                "id": r.get("id"),
                "question": r["question"],
                "gold_sql": gold,
                "agent_sql": pred,
                "difficulty": case.get("difficulty", "unknown"),
            })
    return out


def _stratified_sample(
    candidates: list[dict],
    all_cases: list[dict],
    sample_size: int,
    rng: random.Random,
) -> list[dict]:
    """Sample proportionally to the difficulty distribution in the full eval set.

    Uses the largest-remainder method to turn fractional slot counts into integers
    that sum exactly to ``sample_size``.  When a bucket has fewer candidates than
    its target, the deficit is redistributed to buckets that still have capacity.
    """
    # Target proportions come from the *full* eval set, not just the EX=True pool.
    diff_counts = Counter(c.get("difficulty", "unknown") for c in all_cases)
    total = len(all_cases)

    # Largest-remainder allocation so slots sum to exactly sample_size.
    raw = {d: (cnt / total) * sample_size for d, cnt in diff_counts.items()}
    floored = {d: int(v) for d, v in raw.items()}
    remainder = sample_size - sum(floored.values())
    fracs = sorted(((raw[d] - floored[d], d) for d in raw), reverse=True)
    targets: dict[str, int] = dict(floored)
    for _, d in fracs[:remainder]:
        targets[d] += 1

    # Shuffle each difficulty bucket independently for reproducibility.
    by_diff: dict[str, list[dict]] = defaultdict(list)
    for c in candidates:
        by_diff[c["difficulty"]].append(c)
    for bucket in by_diff.values():
        rng.shuffle(bucket)

    # Draw up to the target from each bucket; collect unchosen candidates.
    chosen: list[dict] = []
    unchosen: list[dict] = []
    for d, bucket in by_diff.items():
        t = targets.get(d, 0)
        chosen.extend(bucket[:t])
        unchosen.extend(bucket[t:])

    # Fill any deficit (buckets with fewer candidates than their target) from
    # the unchosen pool across *other* difficulty buckets.
    shortfall = sample_size - len(chosen)
    if shortfall > 0 and unchosen:
        rng.shuffle(unchosen)
        chosen.extend(unchosen[:shortfall])

    return chosen


def calibrate(
    candidates: list[dict],
    judge: Judge,
    all_cases: list[dict],
    sample_size: int = 18,
    repeats: int = 3,
    seed: int = 0,
) -> dict:
    """Run Branch A ``repeats`` times per sampled case and score self-consistency.

    Sampling is stratified: slots are allocated proportionally to each
    difficulty tier's share of the full eval set (``all_cases``), so the
    calibration sample mirrors the real question distribution.
    """
    rng = random.Random(seed)
    if len(candidates) <= sample_size:
        sample = list(candidates)
    else:
        sample = _stratified_sample(candidates, all_cases, sample_size, rng)

    from tqdm import tqdm

    per_example = []
    desc = f"Branch A calibration ({len(sample)} ex × {repeats} calls each)"
    for ex in tqdm(sample, desc=desc, unit="ex"):
        verdicts = []
        for _ in range(repeats):
            v = judge.false_positive_check(ex["question"], ex["gold_sql"], ex["agent_sql"])
            verdicts.append(v["verdict"] or "PARSE_ERROR")
        counts = Counter(verdicts)
        label, majority = counts.most_common(1)[0]
        per_example.append({
            "id": ex["id"],
            "difficulty": ex["difficulty"],
            "verdicts": verdicts,
            "majority_label": label,
            "majority": majority,
            "fraction": majority / repeats,
            "unanimous": majority == repeats,
        })

    n = len(per_example)
    return {
        "n_candidates": len(candidates),
        "n_sampled": n,
        "repeats": repeats,
        "self_consistency": (sum(p["fraction"] for p in per_example) / n) if n else None,
        "unanimous_rate": (sum(p["unanimous"] for p in per_example) / n) if n else None,
        "sample_by_difficulty": dict(Counter(p["difficulty"] for p in per_example)),
        "per_example": per_example,
    }


def format_report(res: dict, judge_model: str) -> str:
    sep = "=" * 60
    by_diff = res.get("sample_by_difficulty", {})
    diff_str = "  ".join(f"{d}:{n}" for d, n in sorted(by_diff.items()))
    L = [
        sep,
        "JUDGE CALIBRATION - Branch A (false-positive check)",
        f"judge: {judge_model} | repeats: {res['repeats']} | "
        f"candidates (EX=True): {res['n_candidates']} | sampled: {res['n_sampled']}",
        f"sample by difficulty (proportional to eval set): {diff_str}",
        sep,
        "",
        f"  {'id':<12}{'difficulty':<12}{'majority':<22}votes",
    ]
    for p in res["per_example"]:
        maj = f"{p['majority_label']} ({p['majority']}/{res['repeats']})"
        tag = "" if p["unanimous"] else "   <- noisy"
        L.append(
            f"  {str(p['id']):<12}{p['difficulty']:<12}{maj:<22}"
            f"{', '.join(p['verdicts'])}{tag}"
        )

    sc, ur = res["self_consistency"], res["unanimous_rate"]
    L += [
        "",
        f"  Self-consistency (avg majority share): {sc * 100:.1f}%" if sc is not None else
        "  Self-consistency: n/a (no candidates)",
        f"  Unanimous cases:                       {ur * 100:.1f}%" if ur is not None else "",
        sep,
    ]
    return "\n".join(line for line in L if line is not None)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Calibrate the Branch A false-positive judge.")
    p.add_argument("results", help="Path to a run's results JSONL.")
    p.add_argument("--sample", type=int, default=18, help="Max EX=True cases to sample (15-20).")
    p.add_argument("--repeats", type=int, default=3, help="Judge runs per case.")
    p.add_argument("--judge-model", default=JUDGE_MODEL, choices=list(MODELS),
                   help="Judge model (key into config.MODELS).")
    p.add_argument("--seed", type=int, default=10, help="Sampling seed.")
    args = p.parse_args(argv)

    _check_deterministic_judge(args.judge_model)

    records = load_records(args.results)
    all_cases = load_questions()
    cases_by_id = {c["id"]: c for c in all_cases}
    candidates = branch_a_candidates(records, cases_by_id)
    if not candidates:
        raise SystemExit("No EX=True Branch A candidates with gold SQL in this run.")

    judge = Judge(args.judge_model)
    res = calibrate(candidates, judge, all_cases, args.sample, args.repeats, args.seed)
    report = format_report(res, args.judge_model)
    print(report)

    from pathlib import Path
    timestamp = Path(args.results).stem.removeprefix("output_")
    out_path = CALIBRATION_DIR / f"branch_a_{timestamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
