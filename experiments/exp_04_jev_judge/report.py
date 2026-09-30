"""Compare Jev against EX and the GPT judge from an exp_04 results file. No API calls.

Usage:

    python -m experiments.exp_04_jev_judge.report output/results/exp_04_jev_<stamp>.jsonl
    python -m experiments.exp_04_jev_judge.report <jsonl> --write-label-template labels.jsonl
    python -m experiments.exp_04_jev_judge.report <jsonl> --labels labels.jsonl --escalate-below 0.8

Without labels you get pass rates, abstentions, agreement between scorers, cost and
latency. With ``--labels`` (your own correct/incorrect verdicts per attempt) you also
get accuracy, false accepts and false rejects for every scorer. The cascade section
sends Jev verdicts below the confidence threshold, and Jev abstentions, to the GPT
judge verdict from the same run.

A label file has one JSON object per line: ``{"key": "<attempt key>", "label": "correct"}``.
``label`` is ``correct``, ``incorrect`` or ``cannot_determine``; ``id`` (question id) may
replace ``key`` when every attempt of that question deserves the same label. Generate
a template with ``--write-label-template``.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from .jev_judge import load_config, load_jsonl


VALID_LABELS = {"correct", "incorrect", "cannot_determine"}
GPT_POSITIVE = {"CORRECT", "ACCEPTABLE"}
GPT_NEGATIVE = {"FALSE_POSITIVE", "WRONG"}


def gpt_label(raw: str | None) -> int | None:
    return 1 if raw in GPT_POSITIVE else 0 if raw in GPT_NEGATIVE else None


def jev_label(verdict: str | None) -> int | None:
    return 1 if verdict == "CORRECT" else 0 if verdict == "INCORRECT" else None


def human_label(label: str | None) -> int | None:
    return 1 if label == "correct" else 0 if label == "incorrect" else None


def jev_variants(rows: list[dict]) -> dict[str, str]:
    """Display name -> the row field holding that Jev call."""
    variants = {"Jev (monolithic)": "mono"}
    if rows and all("after" in row for row in rows):
        variants["Jev (two-call)"] = "after"
    return variants


def systems(rows: list[dict]) -> dict[str, dict[str, int | None]]:
    """Every scorer's pass/fail (1/0, or None when it abstained) per attempt key."""
    out = {
        "EX": {row["key"]: int(row["branch"] == "A") for row in rows},
        "GPT judge": {row["key"]: gpt_label(row.get("gpt_judge")) for row in rows},
    }
    for name, field in jev_variants(rows).items():
        out[name] = {row["key"]: jev_label(row[field]["verdict"]) for row in rows}
    return out


def resolve_labels(rows: list[dict], label_rows: list[dict]) -> dict[str, str]:
    """Attempt key -> label. A ``key`` label wins over an ``id`` label."""
    by_key = {r["key"]: r["label"] for r in label_rows if r.get("key") and r.get("label")}
    by_id = {r["id"]: r["label"] for r in label_rows if r.get("id") and not r.get("key") and r.get("label")}
    resolved = {}
    for row in rows:
        label = by_key.get(row["key"]) or by_id.get(row["id"])
        if label is None:
            continue
        if label not in VALID_LABELS:
            raise ValueError(f"Invalid label {label!r} for {row['key']}; use one of {sorted(VALID_LABELS)}")
        resolved[row["key"]] = label
    return resolved


def classification_metrics(reference: dict[str, str], predictions: dict[str, int | None]) -> dict:
    """Score predictions against labels; ``cannot_determine`` labels are excluded."""
    resolved = [k for k, label in reference.items() if human_label(label) is not None]
    covered = [k for k in resolved if predictions.get(k) is not None]
    wrong = [k for k in resolved if human_label(reference[k]) == 0]
    right = [k for k in resolved if human_label(reference[k]) == 1]
    false_accepts = sum(predictions.get(k) == 1 for k in wrong)
    false_rejects = sum(predictions.get(k) == 0 for k in right)
    agree = sum(predictions[k] == human_label(reference[k]) for k in covered)
    return {
        "labeled": len(resolved), "covered": len(covered),
        "coverage": len(covered) / len(resolved) if resolved else None,
        "accuracy": agree / len(covered) if covered else None,
        "false_accepts": false_accepts, "false_rejects": false_rejects,
        "unresolved": len(resolved) - len(covered),
    }


def agreement(a: dict, b: dict) -> dict:
    keys = [k for k in a if a[k] is not None and b.get(k) is not None]
    same = sum(a[k] == b[k] for k in keys)
    return {"n": len(keys), "agree": same, "rate": same / len(keys) if keys else None}


def cascade_predictions(rows: list[dict], field: str, threshold: float) -> tuple[dict, float]:
    """Jev verdict unless it abstained or its confidence is below ``threshold``; then the GPT verdict."""
    predictions, escalated = {}, 0
    for row in rows:
        call = row[field]
        if jev_label(call["verdict"]) is None or call["confidence"] < threshold:
            predictions[row["key"]] = gpt_label(row.get("gpt_judge"))
            escalated += 1
        else:
            predictions[row["key"]] = jev_label(call["verdict"])
    return predictions, escalated / len(rows) if rows else 0.0


def cost_and_latency(rows: list[dict], variant_field: str, price_per_million: float) -> dict:
    fields = ("mono",) if variant_field == "mono" else ("narrow", "after")
    tokens = [sum(row[f]["usage"]["input_tokens"] for f in fields) for row in rows]
    latency = [sum(row[f]["latency_ms"] for f in fields) for row in rows]
    return {"usd_per_1000": statistics.mean(tokens) * price_per_million / 1e6 * 1000,
            "mean_latency_s": statistics.mean(latency) / 1000}


def pass_rates(rows: list[dict], scorers: dict) -> dict:
    """Scorer -> profile -> pass rate (abstentions count as not passing)."""
    by_profile = defaultdict(list)
    for row in rows:
        by_profile[row["profile"]].append(row["key"])
    return {name: {profile: statistics.mean(int(preds.get(k) == 1) for k in keys)
                   for profile, keys in sorted(by_profile.items())}
            for name, preds in scorers.items()}


def build_report(rows: list[dict], label_rows: list[dict] | None = None,
                 escalate_below: float = 0.8, price_per_million: float = 0.0) -> dict:
    if not rows:
        raise ValueError("No Jev rows to report on.")
    scorers = systems(rows)
    variants = jev_variants(rows)
    report: dict = {"attempts": len(rows), "scorers": {}, "agreement_with_gpt": {}, "cascade": {},
                    "pass_rates": pass_rates(rows, scorers), "escalate_below": escalate_below}
    reference = resolve_labels(rows, label_rows) if label_rows else {}
    report["labeled"] = len(reference)

    for name, preds in scorers.items():
        entry = {"pass_rate": statistics.mean(int(v == 1) for v in preds.values()),
                 "abstained": sum(v is None for v in preds.values())}
        if reference:
            entry["vs_labels"] = classification_metrics(reference, preds)
        report["scorers"][name] = entry
        if name != "GPT judge":
            report["agreement_with_gpt"][name] = agreement(preds, scorers["GPT judge"])

    for name, field in variants.items():
        cascade, escalated = cascade_predictions(rows, "mono" if field == "mono" else "after", escalate_below)
        entry = {"escalated": escalated,
                 "agreement_with_gpt": agreement(cascade, scorers["GPT judge"])}
        if reference:
            entry["vs_labels"] = classification_metrics(reference, cascade)
        report["cascade"][name] = entry
        report["scorers"][name] |= cost_and_latency(rows, "mono" if field == "mono" else "two", price_per_million)
    return report


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def format_markdown(report: dict) -> str:
    lines = [f"# Experiment 04 report ({report['attempts']} judged attempts)", "",
             "Attempts without judgeable SQL were not sent to Jev and are not counted here. "
             "Abstentions (Jev `CANNOT_DETERMINE`, GPT `GOLD_ERROR`/`HUMAN_REVIEW`) count as not passing in pass rates.", ""]
    labeled = bool(report["labeled"])
    header = "| Scorer | Pass rate | Abstained | $ per 1,000 | Mean latency |"
    sep = "| --- | ---: | ---: | ---: | ---: |"
    if labeled:
        header = "| Scorer | Accuracy on labeled | Coverage | False accepts | False rejects | Pass rate | Abstained | $ per 1,000 | Mean latency |"
        sep = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"
    lines += [header, sep]
    for name, row in report["scorers"].items():
        cost = f"${row['usd_per_1000']:.2f}" if "usd_per_1000" in row else "n/a"
        latency = f"{row['mean_latency_s']:.1f} s" if "mean_latency_s" in row else "n/a"
        tail = f"{_pct(row['pass_rate'])} | {row['abstained']} | {cost} | {latency} |"
        if labeled:
            m = row["vs_labels"]
            lines.append(f"| {name} | {_pct(m['accuracy'])} | {_pct(m['coverage'])} | {m['false_accepts']} | "
                         f"{m['false_rejects']} | {tail}")
        else:
            lines.append(f"| {name} | {tail}")
    if labeled:
        lines += ["", f"Labeled attempts: {report['labeled']}. `cannot_determine` labels are excluded from accuracy."]

    lines += ["", "## Agreement with the GPT judge", "", "| Scorer | Compared | Agree | Rate |", "| --- | ---: | ---: | ---: |"]
    for name, row in report["agreement_with_gpt"].items():
        lines.append(f"| {name} | {row['n']} | {row['agree']} | {_pct(row['rate'])} |")

    lines += ["", f"## Cascade: Jev first, GPT judge when confidence < {report['escalate_below']}", "",
              "| First judge | Sent to GPT judge | Agreement with GPT judge |"
              + (" Accuracy on labeled | False accepts | False rejects |" if labeled else ""),
              "| --- | ---: | ---: |" + (" ---: | ---: | ---: |" if labeled else "")]
    for name, row in report["cascade"].items():
        line = f"| {name} | {_pct(row['escalated'])} | {_pct(row['agreement_with_gpt']['rate'])} |"
        if labeled:
            m = row["vs_labels"]
            line += f" {_pct(m['accuracy'])} | {m['false_accepts']} | {m['false_rejects']} |"
        lines.append(line)

    profiles = sorted({p for rates in report["pass_rates"].values() for p in rates})
    lines += ["", "## Pass rate by agent profile", "",
              "| Scorer | " + " | ".join(profiles) + " |", "| --- | " + " | ".join("---:" for _ in profiles) + " |"]
    for name, rates in report["pass_rates"].items():
        lines.append(f"| {name} | " + " | ".join(_pct(rates.get(p)) for p in profiles) + " |")
    return "\n".join(lines) + "\n"


def write_label_template(rows: list[dict], path: Path) -> None:
    """One line per attempt with the evidence a reviewer needs and an empty label."""
    if path.exists():
        raise SystemExit(f"{path} already exists; refusing to overwrite your labels.")
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps({"key": row["key"], "label": "", "reason": "", "question": row.get("question"),
                                 "gold_sql": row.get("gold_sql"), "agent_sql": row.get("agent_sql"),
                                 "branch": row["branch"]}) + "\n")


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("results", type=Path, help="exp_04 results JSONL written by run.py.")
    p.add_argument("--labels", type=Path, help="Your labels JSONL (see module docstring).")
    p.add_argument("--escalate-below", type=float, default=0.8,
                   help="Cascade threshold on Jev confidence (default 0.8).")
    p.add_argument("--write-label-template", type=Path, metavar="PATH",
                   help="Write an empty labels file for these attempts and exit.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    rows = load_jsonl(args.results)
    if not rows:
        raise SystemExit(f"No rows in {args.results}.")
    if args.write_label_template:
        write_label_template(rows, args.write_label_template)
        print(f"Wrote {args.write_label_template}. Fill in `label` (correct / incorrect / cannot_determine).")
        return
    labels = load_jsonl(args.labels) if args.labels else None
    report = build_report(rows, labels, args.escalate_below,
                          price_per_million=load_config()["input_usd_per_million_tokens"])
    markdown = format_markdown(report)
    print(markdown)
    md_path = args.results.with_suffix(".report.md")
    md_path.write_text(markdown, encoding="utf-8")
    args.results.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
