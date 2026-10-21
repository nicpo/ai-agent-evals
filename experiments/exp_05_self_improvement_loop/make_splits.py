"""Deterministic dev/validation split of a profile's questions.

Questions are stratified so both sides get a similar mix. With `--attempts` (saved attempt records
of the baseline, for example from `evaluate run --candidate c000_baseline --split regression
--repeats 3`) they are first grouped by the baseline's pass count and then ordered by difficulty
within each group: both sides get a similar number of questions the baseline fails. Without it
they are stratified by difficulty alone. Seed 20261015. Families of related questions are not
handled here: the held-out test set is a separate profile.

    python -m experiments.exp_05_self_improvement_loop.make_splits [--profile toy] [--attempts FILE] [--force]
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

from experiments.exp_05_self_improvement_loop.workspace import Workspace

SEED = 20261015
DIFF_RANK = {"easy": 0, "medium": 1, "hard": 2, "extra-hard": 3}


def make_splits(cases: list[dict], attempts: list[dict] | None = None, seed: int = SEED) -> dict:
    """Alternate questions between dev and validation within each stratum."""
    if len(cases) < 2:
        raise ValueError("need at least two questions to split")
    difficulty = {c["id"]: c.get("difficulty", "") for c in cases}
    groups: dict[int, list[str]] = defaultdict(list)
    passes: dict[str, list[int]] = defaultdict(list)
    for r in attempts or []:
        passes[r["id"]].append(r["success"])
    if attempts:
        if set(passes) != set(difficulty) or len({len(v) for v in passes.values()}) != 1:
            raise ValueError("attempts must cover every question with the same number of repeats")
        for qid in sorted(difficulty):
            groups[sum(passes[qid])].append(qid)
    else:
        groups[0] = sorted(difficulty)
    rng = random.Random(seed)
    dev: list[str] = []
    val: list[str] = []
    side = rng.random() < 0.5  # which side gets the first question of each group alternates
    for k in sorted(groups):
        ids = sorted(groups[k])
        rng.shuffle(ids)
        ids.sort(key=lambda q: DIFF_RANK.get(difficulty[q], len(DIFF_RANK)))  # stable: random within difficulty
        for q in ids:
            (dev if side else val).append(q)
            side = not side
    out = {"version": 2, "seed": seed, "dev": sorted(dev), "validation": sorted(val)}
    if attempts:
        out["baseline_passes"] = {q: sum(v) for q, v in sorted(passes.items())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--profile", default="toy")
    ap.add_argument("--attempts", type=Path, help="baseline attempt records (JSONL) to balance by pass count")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--force", action="store_true", help="overwrite an existing splits.json")
    args = ap.parse_args()
    ws = Workspace(args.profile)
    if ws.splits_path.exists() and not args.force:
        raise SystemExit(f"{ws.splits_path} exists; splits are fixed once the loop starts (use --force to overwrite)")
    attempts = None
    if args.attempts:
        attempts = [json.loads(l) for l in args.attempts.read_text(encoding="utf-8").splitlines() if l.strip()]
    out = make_splits(ws.profile_cases(ws.profile), attempts, args.seed)
    ws.splits_path.parent.mkdir(parents=True, exist_ok=True)
    ws.splits_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{ws.splits_path}: {len(out['dev'])} dev, {len(out['validation'])} validation")


if __name__ == "__main__":
    main()
