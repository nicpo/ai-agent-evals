"""Combine reliability, variance, and cost metrics into one report.

Usage:

    python -m experiments.exp_03_repeatability_costs.report output/results/exp_03_repeats_<stamp>.jsonl

Looks for a sidecar ``<results>.judge_usage.json`` (written by ``run.py``) next
to the results file to include judge cost; if it's missing, judge cost is
simply omitted (agent reliability/variance/cost still work).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from eval.metrics import load_records

from experiments.exp_03_repeatability_costs.costs import agent_costs, cost_of_pass, judge_costs
from experiments.exp_03_repeatability_costs.reliability import compute_reliability, format_reliability
from experiments.exp_03_repeatability_costs.variance import agent_sql_variance, judge_verdict_variance


def _judge_model_by_profile(records: list[dict]) -> dict[str, str]:
    out = {}
    for r in records:
        p = r.get("profile") or {}
        if p.get("profile") and p.get("judge_model"):
            out.setdefault(p["profile"], p["judge_model"])
    return out


def build_report(records: list[dict], judge_usage_by_profile: dict[str, list[dict]] | None = None) -> dict:
    reliability = compute_reliability(records)
    agent_variance = agent_sql_variance(records)
    judge_variance = judge_verdict_variance(records)
    a_costs = agent_costs(records)

    for profile, c in a_costs.items():
        total = c["cost_per_1000_attempts_usd"]["total"]
        c["cost_of_pass_per_1000_usd"] = cost_of_pass(total, reliability.get(profile, {}).get("mean_success"))

    j_costs = {}
    if judge_usage_by_profile:
        j_costs = judge_costs(judge_usage_by_profile, _judge_model_by_profile(records))

    return {
        "reliability": reliability,
        "agent_sql_variance": agent_variance,
        "judge_verdict_variance": judge_variance,
        "agent_costs": a_costs,
        "judge_costs": j_costs,
    }


def format_report(report: dict) -> str:
    lines = [format_reliability(report["reliability"]), ""]

    lines += ["=" * 60, "VARIANCE", "=" * 60]
    for profile, v in report["agent_sql_variance"].items():
        rate = v["identical_rate"]
        lines.append(
            f"{profile}: agent produced identical SQL across all repeats for "
            f"{v['identical_across_repeats']}/{v['n_questions']} questions "
            f"({rate * 100:.1f}%)" if rate is not None else f"{profile}: n/a"
        )
    for profile, v in report["judge_verdict_variance"].items():
        rate = v["disagreement_rate"]
        lines.append(
            f"{profile}: judge disagreed on {v['disagreeing_groups']}/{v['n_comparable_groups']} "
            f"identical-SQL groups ({rate * 100:.1f}%)" if rate is not None
            else f"{profile}: judge variance n/a (no repeated identical-SQL groups observed)"
        )

    lines += ["", "=" * 60, "COSTS", "=" * 60]
    for profile, c in report["agent_costs"].items():
        cost = c["cost_per_1000_attempts_usd"]
        lines.append(
            f"{profile} ({c['agent_model'] or 'unpriced model'}): "
            f"{c['mean_llm_calls']} LLM calls/attempt, "
            f"{c['mean_input_tokens']} in / {c['mean_output_tokens']} out tokens/attempt"
        )
        if cost["total"] is not None:
            lines.append(
                f"    ${cost['total']}/1000 attempts (${cost['input']} in + ${cost['output']} out)"
            )
            if c.get("cost_of_pass_per_1000_usd") is not None:
                lines.append(f"    cost-of-pass: ${c['cost_of_pass_per_1000_usd']}/1000 successes")
        else:
            lines.append("    (add this model to pricing.py for a dollar estimate)")
    for profile, j in report["judge_costs"].items():
        if j["cost_usd"] is not None:
            lines.append(f"{profile} judge ({j['judge_model']}): ${j['cost_usd']} for {j['n_calls']} calls")

    lines.append("=" * 60)
    return "\n".join(lines)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("results", help="Path to a repeats JSONL from this experiment's run.py.")
    p.add_argument("--judge-usage", type=Path, help="Sidecar judge-usage JSON (default: alongside results).")
    p.add_argument("--out", type=Path, help="Where to write the JSON report.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    records = load_records(args.results)

    usage_path = args.judge_usage or Path(args.results).with_suffix(".judge_usage.json")
    judge_usage = json.loads(usage_path.read_text(encoding="utf-8")) if usage_path.exists() else None

    report = build_report(records, judge_usage)

    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(format_report(report))

    out_path = args.out or Path(args.results).with_suffix(".report.json")
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"\nReport: {out_path}")


if __name__ == "__main__":
    main()
