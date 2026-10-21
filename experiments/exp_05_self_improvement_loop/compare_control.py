"""Apply the keep rule to an unchanged baseline re-run versus the loop's recorded baseline.

The two arms are the same candidate, so any "keep" is a false accept caused by noise alone. Run the
control first (from the repo root), then compare:

    python -m experiments.exp_05_self_improvement_loop.evaluate run --candidate c000_baseline \
        --split dev --repeats 3 --run-name control_dev
    python -m experiments.exp_05_self_improvement_loop.evaluate run --candidate c000_baseline \
        --split validation --repeats 3 --run-name control_validation
    python -m experiments.exp_05_self_improvement_loop.compare_control
"""

from __future__ import annotations

import argparse

from experiments.exp_05_self_improvement_loop.loop import (
    BASELINE, SPLITS, VAL_MARGIN, decide, load_jsonl, mean_success, records_cost,
)
from experiments.exp_05_self_improvement_loop.workspace import add_workspace_args, from_args


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_workspace_args(ap)
    ws = from_args(ap.parse_args())
    inc = {s: load_jsonl(ws.loop_dir / f"{BASELINE}__{s}.jsonl") for s in SPLITS}
    ctl = {s: load_jsonl(ws.root / "runs" / f"control_{s}.jsonl") for s in SPLITS}
    for s in SPLITS:
        if not ctl[s]:
            raise SystemExit(f"runs/control_{s}.jsonl not found or empty; run the control first")

    result = decide(inc, ctl)

    print("Control (baseline re-run) vs recorded baseline")
    for s in SPLITS:
        print(f"  {s:<10} baseline {mean_success(inc[s]) * 100:5.1f}%   control {mean_success(ctl[s]) * 100:5.1f}%   "
              f"n={len(ctl[s])}")
    print(f"  validation gain {result['val_gain'] * 100:+.1f} points (keep threshold +{VAL_MARGIN * 100:.0f})")
    print(f"  dev change      {result['dev_gain'] * 100:+.1f} points")
    print(f"  always-passing questions now always failing: {result['broken'] or 'none'}")
    print(f"  keep rule: {'KEEP (false accept)' if result['keep'] else 'reject'}")
    if result["reasons"]:
        print("  reasons: " + "; ".join(result["reasons"]))
    print(f"  control cost: ${records_cost(ctl['dev'] + ctl['validation']):.2f}")


if __name__ == "__main__":
    main()
