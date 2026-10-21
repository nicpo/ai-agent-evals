"""Final arms on the held-out test set and the regression suite.

Arms: c000_baseline (the profile's unchanged agent prompt) and the loop winner (the incumbent in the loop log).
Test: 5 repeats. Regression (all of the profile's questions, which the loop trained on): 3 repeats.
Results go to output/exp_05/<profile>/final/<candidate>__<split>.jsonl; reruns skip finished
attempts, so an interrupted run can simply be started again. Needs --test-profile, and that
profile must have been locked by `loop init --test-profile ...` before the loop ran.

    python -m experiments.exp_05_self_improvement_loop.final_eval plan   --test-profile toy_test
    python -m experiments.exp_05_self_improvement_loop.final_eval run    --test-profile toy_test [--splits test regression] [--workers 4]
    python -m experiments.exp_05_self_improvement_loop.final_eval report --test-profile toy_test

The report gives, per split and arm, the judge-based success rate and strict EX rate with 95% bootstrap
CIs (resampling families on test, questions on regression), plus paired differences between arms.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

from experiments.exp_05_self_improvement_loop import evaluate
from experiments.exp_05_self_improvement_loop.loop import BASELINE, SEED, load_jsonl, records_cost, winner
from experiments.exp_05_self_improvement_loop.workspace import Workspace, add_workspace_args, from_args

REPEATS = {"test": 5, "regression": 3}
N_BOOT = 10_000


def arms(ws: Workspace) -> list[str]:
    """[baseline, winner]. Without a kept candidate there is nothing to compare."""
    best = winner(ws)
    if best == BASELINE:
        raise SystemExit("the loop kept no candidate (the incumbent is still the baseline); nothing to compare")
    return [BASELINE, best]


def path_for(ws: Workspace, name: str, split: str) -> Path:
    return ws.final_dir / f"{name}__{split}.jsonl"


def run(ws: Workspace, splits: list[str], workers: int) -> None:
    for split in splits:
        for name in arms(ws):
            c = ws.candidate(name)
            records = evaluate.run(ws, c, split, REPEATS[split], path_for(ws, name, split), workers)
            s = evaluate.summarize(records)
            cost = "n/a" if s["cost_usd"] is None else f"${s['cost_usd']:.2f}"
            print(f"{name} {split}: success {s['success_rate'] * 100:.1f}%  EX {s['ex_rate'] * 100:.1f}%  "
                  f"unresolved {s['unresolved_verdicts']}  cost {cost}", flush=True)


def cluster_means(records: list[dict], clusters: dict[str, str], field: str) -> dict[str, float]:
    by: dict[str, list[float]] = {}
    for r in records:
        by.setdefault(clusters[r["id"]], []).append(float(bool(r[field])))
    return {k: sum(v) / len(v) for k, v in by.items()}


def boot_ci(per_cluster: list[list[float]], rng: random.Random) -> tuple[float, float, float]:
    """per_cluster[i] = values of one or more arms for cluster i; returns (point, lo, hi) of the mean of arm 0
    (one arm) or of arm 0 minus arm 1 (two arms)."""
    def stat(rows: list[list[float]]) -> float:
        return sum(r[0] - (r[1] if len(r) > 1 else 0.0) for r in rows) / len(rows)

    n = len(per_cluster)
    draws = sorted(stat([per_cluster[rng.randrange(n)] for _ in range(n)]) for _ in range(N_BOOT))
    return stat(per_cluster), draws[int(0.025 * N_BOOT)], draws[int(0.975 * N_BOOT)]


def print_comparison(names: list[str], means: dict[str, dict[str, float]], rng: random.Random) -> None:
    """One metric: each arm with its CI, then the winner minus the baseline (paired by cluster)."""
    clusters = sorted(set.intersection(*(set(m) for m in means.values())))
    for n in names:
        p, lo, hi = boot_ci([[means[n][c]] for c in clusters], rng)
        print(f"    {n:<22} {p * 100:5.1f}%  [{lo * 100:5.1f}, {hi * 100:5.1f}]")
    a, b = names[1], names[0]
    p, lo, hi = boot_ci([[means[a][c], means[b][c]] for c in clusters], rng)
    print(f"    {a} - {b}: {p * 100:+5.1f} points  [{lo * 100:+5.1f}, {hi * 100:+5.1f}]")


def report(ws: Workspace) -> None:
    names = arms(ws)
    rng = random.Random(SEED)
    for split in REPEATS:
        data = {n: load_jsonl(path_for(ws, n, split)) for n in names}
        missing = [n for n in names if not data[n]]
        if missing:
            print(f"\n== {split}: no results for {missing}; run `final_eval run` first")
            continue
        clusters = ws.clusters(split)
        expected = len(clusters) * REPEATS[split]
        print(f"\n== {split} ({REPEATS[split]} repeats; clusters resampled by "
              f"{'family' if split == 'test' else 'question'})")
        for n in names:
            if len(data[n]) != expected:
                print(f"  WARNING {n}: {len(data[n])} attempts, expected {expected}")
        for field, label in (("success", "Judge-based success"), ("ex_match", "strict EX")):
            print(f"  {label}")
            print_comparison(names, {n: cluster_means(data[n], clusters, field) for n in names}, rng)
        cost = sum(records_cost(data[n]) for n in names)
        print(f"  cost of these attempts: ${cost:.2f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("plan", "run", "report"))
    add_workspace_args(ap)
    ap.add_argument("--splits", nargs="+", choices=tuple(REPEATS), default=list(REPEATS))
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    ws = from_args(args)
    if args.command == "plan":
        for split in REPEATS:
            n_q = len(ws.cases(split))
            print(f"{split}: {n_q} questions x {REPEATS[split]} repeats x {len(arms(ws))} arms = "
                  f"{n_q * REPEATS[split] * len(arms(ws))} attempts")
        print("arms:", ", ".join(arms(ws)))
    elif args.command == "run":
        run(ws, args.splits, args.workers)
    else:
        report(ws)


if __name__ == "__main__":
    main()
