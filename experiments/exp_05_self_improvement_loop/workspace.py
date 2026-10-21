"""Where an exp 05 run keeps its files, and which questions each split holds.

A workspace is one eval profile (default `toy`): its questions come from the profile's
`eval_glob` (config/experiment_profiles.yaml), its judge prompts from the profile's judge
bundle, and everything the experiment writes goes to output/exp_05/<profile>/ (git-ignored):

    splits.json        dev / validation split of the profile's questions (make_splits)
    manifest.json      run inputs and the test-set hash, recorded by `loop init`
    candidates/        candidates the loop proposes (the baseline is the profile's agent prompt)
    loop/              loop attempt records, <candidate>__<dev|validation>.jsonl
    loop_log.jsonl     one event per proposal and decision
    control_*.jsonl    unchanged-baseline control run (compare_control)
    final/             final arms on the test set and the regression suite
    independent/       calibration and re-grade files of the independent grader

Splits: `dev` and `validation` partition the profile's questions. `regression` is all of them
(not held out: the loop trained and selected on them). `test` is a separate, held-out profile
(`--test-profile`) whose files are hashed at `loop init`; later runs refuse a changed test set.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from config import EVALS_DIR, OUTPUT_DIR
from experiments.exp_05_self_improvement_loop import candidates as cand
from eval.runner import load_questions
from experiment_profiles import load_profile

SPLITS = ("dev", "validation", "regression", "test")


class Workspace:
    def __init__(self, profile: str = "toy", test_profile: str | None = None, root: Path | None = None):
        self.profile, self.test_profile = profile, test_profile
        self.root = root or OUTPUT_DIR / "exp_05" / profile
        self.splits_path = self.root / "splits.json"
        self.manifest_path = self.root / "manifest.json"
        self.log_path = self.root / "loop_log.jsonl"
        self.candidates_dir = self.root / "candidates"
        self.loop_dir = self.root / "loop"
        self.final_dir = self.root / "final"
        self.independent_dir = self.root / "independent"
        self.sandbox_dir = self.root / "sandbox"

    # -- profile data
    def profile_cases(self, profile: str) -> list[dict]:
        cases = load_questions(EVALS_DIR, load_profile(profile).eval_glob)
        ids = [c["id"] for c in cases]
        if not cases or len(ids) != len(set(ids)):
            raise SystemExit(f"profile {profile!r}: no questions, or duplicate ids")
        return sorted(cases, key=lambda c: c["id"])

    def test_sha256(self) -> str:
        if not self.test_profile:
            raise SystemExit("this step needs --test-profile (a held-out profile of fresh questions)")
        h = hashlib.sha256()
        for path in sorted(EVALS_DIR.glob(load_profile(self.test_profile).eval_glob)):
            h.update(path.read_bytes())
        return h.hexdigest()

    def judge_templates(self) -> dict:
        return load_profile(self.profile).judge_prompts.templates

    @property
    def max_tool_calls(self) -> int:
        return load_profile(self.profile).max_tool_calls

    def candidate(self, name: str) -> cand.Candidate:
        """The baseline (the profile's agent prompt, original tool descriptions) or a saved proposal."""
        if name == cand.BASELINE:
            return cand.Candidate(name, load_profile(self.profile).agent_prompt)
        return cand.load(name, self.candidates_dir)

    # -- splits
    def splits(self) -> dict:
        if not self.splits_path.exists():
            raise SystemExit(f"{self.splits_path} not found: run make_splits --profile {self.profile} first")
        return json.loads(self.splits_path.read_text(encoding="utf-8"))

    def cases(self, split: str) -> list[dict]:
        if split == "test":
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8")) if self.manifest_path.exists() else {}
            if manifest.get("test_set_sha256") != self.test_sha256():
                raise SystemExit("the test set is not locked or has changed since `loop init --test-profile ...`; "
                                 "refusing to evaluate")
            return self.profile_cases(self.test_profile)
        base = {c["id"]: c for c in self.profile_cases(self.profile)}
        if split == "regression":
            return [base[k] for k in sorted(base)]
        return [base[i] for i in self.splits()[split]]

    def clusters(self, split: str) -> dict[str, str]:
        """question id -> cluster for bootstrap resampling: the test set's families, else the question itself."""
        return {c["id"]: (c.get("family_id") or c["id"]) if split == "test" else c["id"] for c in self.cases(split)}


def add_workspace_args(ap) -> None:
    ap.add_argument("--profile", default="toy", help="eval profile whose questions are used (default: toy)")
    ap.add_argument("--test-profile", help="held-out profile of fresh questions (for example toy_test)")


def from_args(args) -> Workspace:
    return Workspace(args.profile, args.test_profile)
