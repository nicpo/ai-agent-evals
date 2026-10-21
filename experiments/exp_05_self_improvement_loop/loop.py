"""The bounded self-improvement loop: select dev failures, critique, propose, lint, evaluate, keep or reject.

Commands (run from the repo root; all take --profile, default toy):
    python -m experiments.exp_05_self_improvement_loop.loop init [--test-profile P]  # manifest + baseline on dev and validation
    python -m experiments.exp_05_self_improvement_loop.loop lint NAME                # lint a candidate folder (no API)
    python -m experiments.exp_05_self_improvement_loop.loop verify                   # check the recorded run inputs
    python -m experiments.exp_05_self_improvement_loop.loop run [--dry-run] [--max-proposals 10] [--cap 20]
    python -m experiments.exp_05_self_improvement_loop.loop status

Every candidate is evaluated on both dev and validation (3 repeats). Keep rule (pre-registered):
validation success >= incumbent + 5 points, dev >= incumbent - 5 points, and at most one question
that passed every repeat for the incumbent now fails every repeat. Spend is capped; the loop stops
at the proposal limit or when the next proposal would not fit in the budget. The run is resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from config import DB_PATH, EVALS_DIR
from experiment_profiles import load_profile
from experiments.exp_05_self_improvement_loop import candidates as cand, models
from experiments.exp_05_self_improvement_loop.lint import lint
from experiments.exp_05_self_improvement_loop.packet import render_packet, select_failures
from experiments.exp_05_self_improvement_loop.proposer import Proposal
from experiments.exp_05_self_improvement_loop.workspace import Workspace, add_workspace_args, from_args

HERE = Path(__file__).resolve().parent
BASELINE = cand.BASELINE
SEED = 20261015
VAL_MARGIN = 0.05
DEV_FLOOR = -0.05
SPLITS = ("dev", "validation")


# ---------------------------------------------------------------- records, cost, scoring
def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def attempt_cost(rec: dict) -> float | None:
    a = models.cost(rec["trace"]["model"], rec["trace"]["total_input_tokens"], rec["trace"]["total_output_tokens"])
    ju = rec.get("judge_usage") or {}
    j = models.cost(models.JUDGE["id"], ju.get("input_tokens", 0), ju.get("output_tokens", 0))
    return None if a is None or j is None else a + j


def records_cost(records: list[dict]) -> float:
    costs = [attempt_cost(r) for r in records]
    if any(c is None for c in costs):
        raise SystemExit("unknown model price: fill in models.PRICES before running the loop")
    return sum(costs)


def _key(r: dict) -> tuple:
    return (r["id"], r["repeat"])


def paired(inc: list[dict], new: list[dict]) -> tuple[list[dict], list[dict]]:
    """Compare the same cells; never silently discard an incomplete attempt."""
    a, b = [_key(r) for r in inc], [_key(r) for r in new]
    if len(a) != len(set(a)) or len(b) != len(set(b)) or set(a) != set(b):
        raise ValueError("incumbent and candidate must have one attempt for each matching question/repeat")
    return inc, new


def mean_success(records: list[dict]) -> float:
    return sum(r["success"] for r in records) / len(records) if records else 0.0


def decide(inc: dict[str, list[dict]], new: dict[str, list[dict]]) -> dict:
    """Pre-registered keep rule. inc/new map split name -> attempt records."""
    gains, n = {}, {}
    inc_all, new_all = [], []
    for split in SPLITS:
        a, b = paired(inc[split], new[split])
        n[split] = len(a)
        gains[split] = mean_success(b) - mean_success(a) if a else 0.0
        inc_all += a
        new_all += b
    by_inc: dict[str, list[int]] = {}
    by_new: dict[str, list[int]] = {}
    for r in inc_all:
        by_inc.setdefault(r["id"], []).append(r["success"])
    for r in new_all:
        by_new.setdefault(r["id"], []).append(r["success"])
    broken = sorted(q for q, v in by_inc.items() if all(v) and q in by_new and not any(by_new[q]))
    reasons = []
    if gains["validation"] < VAL_MARGIN - 1e-9:
        reasons.append(f"validation gain {gains['validation'] * 100:+.1f} < +{VAL_MARGIN * 100:.0f} points")
    if gains["dev"] < DEV_FLOOR - 1e-9:
        reasons.append(f"dev change {gains['dev'] * 100:+.1f} < {DEV_FLOOR * 100:.0f} points")
    if len(broken) > 1:
        reasons.append(f"{len(broken)} always-passing questions now always fail")
    return {"keep": not reasons, "reasons": reasons, "val_gain": gains["validation"], "dev_gain": gains["dev"],
            "broken": broken, "n_attempts": n}


# ---------------------------------------------------------------- run manifest
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest_snapshot(ws: Workspace, repeats: int = 3) -> dict:
    """Protocol inputs: the data, judge prompts, splits and models a run's attempts depend on."""
    profile = load_profile(ws.profile)
    inputs = [ws.splits_path, profile.judge_prompts.path, HERE / "prompts/critique.md", HERE / "prompts/propose.md",
              DB_PATH, *sorted(EVALS_DIR.glob(profile.eval_glob))]
    snap = {
        "baseline": {"name": BASELINE, "sha256": ws.candidate(BASELINE).sha256},
        "models": {"agent": models.AGENT, "judge": models.JUDGE, "proposer": models.PROPOSER},
        "inputs_sha256": {p.name if p.parent == ws.root else p.relative_to(HERE.parents[1]).as_posix(): _sha256(p)
                          for p in inputs},
        "repeats": repeats,
    }
    if ws.test_profile:
        snap["test_set_sha256"] = ws.test_sha256()
    return snap


def write_manifest(ws: Workspace, repeats: int = 3) -> None:
    now = manifest_snapshot(ws, repeats)
    if ws.manifest_path.exists():
        saved = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
        if any(saved.get(k) != v for k, v in now.items()):
            raise SystemExit(f"{ws.manifest_path} exists and differs from the current inputs; "
                             "keep it with this run's records, or use a fresh profile/output folder")
        print("manifest already recorded and matches")
        return
    ws.root.mkdir(parents=True, exist_ok=True)
    data = {"recorded_at_utc": datetime.now(UTC).isoformat(), **now}
    ws.manifest_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"recorded manifest for {len(data['inputs_sha256'])} data and prompt inputs"
          + (", test set locked" if ws.test_profile else ""))


def verify_manifest(ws: Workspace, repeats: int = 3) -> None:
    if not ws.manifest_path.exists():
        raise SystemExit("manifest.json missing: run `loop init` first")
    saved = json.loads(ws.manifest_path.read_text(encoding="utf-8"))
    now = manifest_snapshot(ws, repeats)
    drift = [key for key in now if saved.get(key) != now[key]]
    if drift:
        raise SystemExit("run inputs changed, refusing to mix attempts:\n  " + "\n  ".join(drift))


# ---------------------------------------------------------------- the loop
class Loop:
    def __init__(self, ws: Workspace, proposer, evaluate_fn, *, results_dir: Path | None = None,
                 log_path: Path | None = None, candidates_root: Path | None = None, cap: float = 20.0,
                 max_proposals: int = 10, repeats: int = 3, limit: int | None = None, seed: int = SEED, out=print):
        self.ws, self.proposer, self.evaluate_fn = ws, proposer, evaluate_fn
        self.results_dir = results_dir or ws.loop_dir
        self.log_path = log_path or ws.log_path
        self.candidates_root = candidates_root or ws.candidates_dir
        self.cap, self.max_proposals, self.repeats, self.limit, self.seed = cap, max_proposals, repeats, limit, seed
        self.out = out
        self.baseline = ws.candidate(BASELINE)
        self._seen_cases: list[dict] | None = None

    # -- log
    def _append(self, event: dict) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({**event, "at_utc": datetime.now(UTC).isoformat()}, default=str) + "\n")

    def _events(self) -> list[dict]:
        return load_jsonl(self.log_path)

    def _records(self, name: str, split: str) -> list[dict]:
        return load_jsonl(self.results_dir / f"{name}__{split}.jsonl")

    def _baseline_records(self) -> dict[str, list[dict]]:
        return {s: self._records(BASELINE, s) for s in SPLITS}

    def _load(self, name: str) -> cand.Candidate:
        return self.baseline if name == BASELINE else cand.load(name, self.candidates_root)

    # -- state
    def state(self) -> dict:
        events = self._events()
        proposals = {e["k"]: e for e in events if e["event"] == "proposal"}
        decisions = {e["k"]: e for e in events if e["event"] == "decision"}
        incumbent = BASELINE
        for k in sorted(decisions):
            if decisions[k]["keep"]:
                incumbent = decisions[k]["name"]
        base = self._baseline_records()
        base_cost = records_cost(base["dev"] + base["validation"])
        spent = base_cost + sum(e.get("proposer_cost") or 0 for e in proposals.values()) \
            + sum(e.get("eval_cost") or 0 for e in decisions.values())
        return {"proposals": proposals, "decisions": decisions, "incumbent": incumbent, "spent": spent,
                "baseline_cost": base_cost}

    def history_text(self, state: dict) -> str:
        lines = []
        for k in sorted(state["proposals"]):
            p, d = state["proposals"][k], state["decisions"].get(k)
            if d is None:
                continue
            if "lint" in d["reasons"] or d.get("lint_failed"):
                outcome = "rejected before evaluation (broke the rules)"
            else:
                outcome = (f"{'kept' if d['keep'] else 'rejected'}: validation {d['val_gain'] * 100:+.1f} points, "
                           f"dev {d['dev_gain'] * 100:+.1f} points")
            lines.append(f"- proposal {k}: {p['summary']} -> {outcome}")
        return "\n".join(lines) or "(none yet)"

    # -- one proposal
    def _evaluate(self, c: cand.Candidate) -> dict[str, list[dict]]:
        out = {}
        for split in SPLITS:
            path = self.results_dir / f"{c.name}__{split}.jsonl"
            out[split] = self.evaluate_fn(c, split, self.repeats, path, self.limit)
        return out

    def _complete_records(self, records: list[dict], split: str, candidate: cand.Candidate,
                          *, baseline: bool = False) -> list[dict]:
        ids = self.ws.splits()[split]
        if self.limit is not None:
            ids = ids[:self.limit]
        expected = {(qid, repeat) for qid in ids for repeat in range(1, self.repeats + 1)}
        selected = [r for r in records if _key(r) in expected]
        keys = [_key(r) for r in selected]
        missing = expected - set(keys)
        duplicates = len(keys) - len(set(keys))
        wrong_candidate = sum(r.get("candidate") != candidate.name or
                              r.get("candidate_sha256") != candidate.sha256 for r in selected)
        unexpected = 0 if baseline and self.limit is not None else len(records) - len(selected)
        if missing or duplicates or wrong_candidate or unexpected:
            raise RuntimeError(f"{candidate.name} {split}: incomplete or mixed evaluation "
                               f"(missing={len(missing)}, duplicates={duplicates}, "
                               f"wrong_candidate={wrong_candidate}, unexpected={unexpected}); "
                               "resume the run before deciding")
        return selected

    def _decide_and_log(self, k: int, c: cand.Candidate, parent: str) -> bool:
        new = self._evaluate(c)
        parent_candidate = self._load(parent)
        inc = {s: self._complete_records(self._records(parent, s), s, parent_candidate, baseline=parent == BASELINE)
               for s in SPLITS}
        new = {s: self._complete_records(new[s], s, c) for s in SPLITS}
        verdict = decide(inc, new)
        eval_cost = records_cost(new["dev"] + new["validation"])
        self._append({"event": "decision", "k": k, "name": c.name, "parent": parent, "eval_cost": eval_cost,
                      **verdict})
        self.out(f"  {c.name}: val {verdict['val_gain'] * 100:+.1f}, dev {verdict['dev_gain'] * 100:+.1f} points, "
                 f"{'KEPT' if verdict['keep'] else 'rejected'} {verdict['reasons'] or ''} (eval ${eval_cost:.2f})")
        return verdict["keep"]

    def _lint(self, c: cand.Candidate):
        if self._seen_cases is None:
            self._seen_cases = self.ws.cases("regression")  # the test set is never read here
        return lint(c, self._seen_cases)

    def _rejected_before_eval(self, k: int, name: str, parent: str) -> None:
        self._append({"event": "decision", "k": k, "name": name, "parent": parent, "keep": False,
                      "reasons": ["lint"], "lint_failed": True, "val_gain": 0.0, "dev_gain": 0.0,
                      "broken": [], "eval_cost": 0.0})

    def _propose(self, k: int, incumbent: cand.Candidate, state: dict) -> tuple[cand.Candidate | None, dict]:
        records = self._records(incumbent.name, "dev")
        chosen = select_failures(records, self.seed + k)
        if not chosen:
            return None, {"stop": "no failed dev attempts left for the incumbent"}
        packet = render_packet(chosen)
        history = self.history_text(state)
        name = f"c{k + 1:03d}"
        info = {"event": "proposal", "k": k, "name": name, "parent": incumbent.name,
                "selected_questions": sorted({r["id"] for r in chosen})}
        try:
            prop: Proposal = self.proposer.propose(incumbent, packet, history)
            c = cand.Candidate(name, prop.prompt, prop.descriptions)
            result = self._lint(c)
            if not result.ok:
                prop = self.proposer.propose(incumbent, packet, history, lint_feedback=result.violations, previous=prop)
                c = cand.Candidate(name, prop.prompt, prop.descriptions)
                result = self._lint(c)
        except (ValueError, KeyError) as exc:  # unusable proposer output counts as a spent proposal
            info.update(summary=f"proposer error: {exc}", lint={"ok": False, "violations": [str(exc)]},
                        proposer_cost=0.0)
            self._append(info)
            self._rejected_before_eval(k, name, incumbent.name)
            return None, {}
        info.update(summary=prop.summary, critique=prop.critique, proposer_attempts=prop.attempts,
                    proposer_cost=prop.cost or 0.0, lint={"ok": result.ok, "violations": result.violations})
        c.save(self.candidates_root, meta={"parent": incumbent.name, "k": k, "summary": prop.summary,
                                           "critique": prop.critique, "lint": info["lint"]})
        self._append(info)
        if not result.ok:
            self._rejected_before_eval(k, name, incumbent.name)
            self.out(f"  {name}: rejected by lint: {result.violations}")
            return None, {}
        return c, {}

    # -- main loop
    def run(self) -> None:
        state = self.state()
        # finish a proposal that was interrupted between proposing and deciding
        for k, p in sorted(state["proposals"].items()):
            if k not in state["decisions"] and p["lint"]["ok"]:
                self.out(f"resuming evaluation of {p['name']}")
                c = self._load(p["name"])
                self._decide_and_log(k, c, p["parent"])
        last_cost = None
        while True:
            state = self.state()
            k = len(state["proposals"]) + 1
            if k > self.max_proposals:
                self.out(f"stopping: {self.max_proposals} proposals used")
                break
            est = max(1.5, (last_cost or 0) * 1.2)
            if state["spent"] + est > self.cap:
                self.out(f"stopping: spent ${state['spent']:.2f} of ${self.cap:.2f}; next proposal needs about ${est:.2f}")
                break
            incumbent = self._load(state["incumbent"])
            self.out(f"proposal {k}/{self.max_proposals} (incumbent {incumbent.name}, spent ${state['spent']:.2f})")
            c, stop = self._propose(k, incumbent, state)
            if stop:
                self.out(f"stopping: {stop['stop']}")
                break
            before = self.state()["spent"]
            if c is not None:
                self._decide_and_log(k, c, incumbent.name)
            last_cost = self.state()["spent"] - before
        self.out(self.status_text())

    def status_text(self) -> str:
        st = self.state()
        rows = ["k  name   result    val     dev     cost   summary"]
        for k in sorted(st["proposals"]):
            p, d = st["proposals"][k], st["decisions"].get(k)
            if d is None:
                rows.append(f"{k:<2} {p['name']}  pending")
                continue
            res = "lint" if d.get("lint_failed") else ("KEPT" if d["keep"] else "rejected")
            cost = (p.get("proposer_cost") or 0) + (d.get("eval_cost") or 0)
            rows.append(f"{k:<2} {p['name']}  {res:<8} {d['val_gain'] * 100:+6.1f}  {d['dev_gain'] * 100:+6.1f}  "
                        f"${cost:5.2f}  {p['summary'][:70]}")
        rows.append(f"incumbent: {st['incumbent']}; spent ${st['spent']:.2f} of ${self.cap:.2f} "
                    f"(baseline ${st['baseline_cost']:.2f})")
        return "\n".join(rows)


def winner(ws: Workspace) -> str:
    """The incumbent after the loop: the last kept candidate, else the baseline."""
    return Loop(ws, None, None).state()["incumbent"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=("init", "lint", "verify", "run", "status"))
    ap.add_argument("name", nargs="?")
    add_workspace_args(ap)
    ap.add_argument("--max-proposals", type=int, default=10)
    ap.add_argument("--cap", type=float, default=20.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--dry-run", action="store_true",
                    help="stub proposer, 2 questions per split, 1 repeat, separate sandbox folder")
    args = ap.parse_args()
    ws = from_args(args)

    if args.command == "init":
        from experiments.exp_05_self_improvement_loop import evaluate

        write_manifest(ws, args.repeats)
        baseline = ws.candidate(BASELINE)
        for split in SPLITS:
            evaluate.run(ws, baseline, split, args.repeats, ws.loop_dir / f"{BASELINE}__{split}.jsonl", args.workers)
        print(f"baseline {BASELINE} recorded; loop `run` starts from here")
    elif args.command == "verify":
        verify_manifest(ws, args.repeats)
        print("manifest inputs match")
    elif args.command == "lint":
        if not args.name:
            ap.error("lint needs a candidate name")
        result = lint(ws.candidate(args.name), ws.cases("regression"))
        print("OK" if result.ok else "VIOLATIONS:\n  " + "\n  ".join(result.violations))
        sys.exit(0 if result.ok else 1)
    elif args.command == "status":
        print(Loop(ws, None, None).status_text())
    else:
        from experiments.exp_05_self_improvement_loop import evaluate

        baseline = ws.candidate(BASELINE)
        if args.dry_run:
            from experiments.exp_05_self_improvement_loop.proposer import StubProposer

            sandbox = ws.sandbox_dir
            shutil.rmtree(sandbox, ignore_errors=True)
            for split in SPLITS:  # a tiny baseline so the sandbox is self-contained
                evaluate.run(ws, baseline, split, 1, sandbox / "loop" / f"{BASELINE}__{split}.jsonl", args.workers, 2)
            loop = Loop(ws, StubProposer(), lambda c, s, r, p, lim: evaluate.run(ws, c, s, r, p, args.workers, lim),
                        results_dir=sandbox / "loop", log_path=sandbox / "loop_log.jsonl",
                        candidates_root=sandbox / "candidates", cap=args.cap, max_proposals=1, repeats=1, limit=2)
        else:
            verify_manifest(ws, args.repeats)
            if any(v == (None, None) for v in models.PRICES.values()):
                raise SystemExit("models.PRICES has unknown entries")
            from experiments.exp_05_self_improvement_loop.proposer import SonnetProposer

            loop = Loop(ws, SonnetProposer(), lambda c, s, r, p, lim: evaluate.run(ws, c, s, r, p, args.workers, lim),
                        cap=args.cap, max_proposals=args.max_proposals, repeats=args.repeats)
        loop.run()


if __name__ == "__main__":
    main()
