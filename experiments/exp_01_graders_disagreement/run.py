"""Run the agent + grader suite for the graders-disagreement experiment.

Thin wrapper around ``eval.runner`` that defaults to the ``toy`` experiment
profile instead of the repo default, so this experiment is safe to run out of
the box against the toy eval set. Point it at a real eval set with
``--profile <your-profile>`` (see ``config/experiment_profiles.yaml``).

Usage:

    python -m experiments.exp_01_graders_disagreement.run
    python -m experiments.exp_01_graders_disagreement.run --profile v1 --limit 20

Requires the relevant provider API keys (agent + judge) in ``.env`` -- this
runs real LLM calls. Writes a results JSONL under ``output/results/`` exactly
like ``python -m eval.runner`` does; feed that file to ``analyze.py``.
"""

from __future__ import annotations

import argparse

from eval.runner import load_questions, run_suite
from experiment_profiles import load_profile


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default="toy", help="Experiment profile (default: toy).")
    p.add_argument("--difficulty", help="Filter to one difficulty tier.")
    p.add_argument("--limit", type=int, help="Cap the number of questions.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = _parse_args(argv)
    try:
        profile = load_profile(args.profile)
    except ValueError as exc:
        raise SystemExit(f"Invalid experiment profile: {exc}") from exc

    cases = load_questions(glob=profile.eval_glob)
    if args.difficulty:
        cases = [c for c in cases if c.get("difficulty") == args.difficulty]
    if args.limit is not None:
        cases = cases[: args.limit]
    if not cases:
        raise SystemExit("No eval cases matched -- check --profile / --difficulty / --limit.")

    # The judge must run: the whole point of this experiment is the EX-vs-judge
    # disagreement matrix, which needs Tier 5 verdicts.
    run_suite(cases, profile=profile, run_judge=True)


if __name__ == "__main__":
    main()
