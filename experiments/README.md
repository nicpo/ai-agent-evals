# Experiments

Standalone code for the methodology described in a few blog posts about this
harness. Each experiment is built entirely on top of the existing harness
modules (`agent/`, `graders/`, `eval/`, `experiment_profiles.py`) -- nothing
here reimplements agent-calling or grading logic, and nothing here embeds real
eval-set questions, gold SQL, or answers. Every `run.py` defaults to the
`toy` experiment profile (`data/evals/toy/`, already safe to read in full) and
takes `--profile` to point at your own eval set instead.

| Experiment | Post | Method |
| --- | --- | --- |
| [`exp_01_graders_disagreement`](exp_01_graders_disagreement/) | Lessons from the mismatch between your deterministic grader and the LLM judge | EX x judge disagreement matrix; inspect the off-diagonal cases |
| [`exp_02_graders_agreement`](exp_02_graders_agreement/) | When the graders agree, they may not have seen everything | Probe dual-pass cases against schema-derived database perturbations |
| [`exp_03_repeatability_costs`](exp_03_repeatability_costs/) | Measuring agent reliability with repeated evals; Fewer agent steps do not mean proportionally lower cost | Repeated-run reliability (pass@k / pass^k), agent/judge variance, and token/dollar cost accounting |

See each experiment's own README for the method detail and reproduce commands.
Output (results JSONL, reports) is written under the harness's existing
`output/results/` directory, which is already git-ignored.
