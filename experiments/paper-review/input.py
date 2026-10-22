from experiments.repo_paths import repo_path

additional_instructions = """
#### Task: ML Paper Review (Accept/Reject prediction)
Given the full text of an ML-venue submission, the program writes a structured
review with an LLM and returns a final 'accept' or 'reject' decision. Metric =
per-paper binary accuracy against the real venue outcome.

#### What generalizes:
Papers differ each eval; only the REVIEW PROCEDURE transfers. Improve how the
program reads, critiques, and weighs evidence (rubric wording, review structure,
decision calibration, aggregation of per-dimension scores). Never key logic to
specific paper topics, titles, or ids, and calibrate the accept/reject threshold
against the balanced 50/50 prior — not against one subsample draw.

#### Performance budget:
Papers run up to ~50k tokens (prefill-dominated) and the solver reasons at
`reasoning_effort="high"`, so each review call can take several minutes and a
50-row subsample eval walls at its slowest row — it must fit its ~120-min
timeout. Extra passes per paper multiply BOTH time and cost by full paper
length — add one only if single-pass has clearly plateaued, and re-check the
timeout.

#### Reading results (the engine drives the eval/reflect protocol; make ONE self-contained change per iteration):
- Treat swings under ~0.08 as noise (50-row binary metric: one paper = 0.02).
- Failed rows or empty predictions mean your change broke the pipeline — fix
  that before reasoning about review quality. Malformed decisions score 0 and
  are never excluded from the denominator.

#### Constraints:
- NEVER read or use `tests/testdata/testset.json` — it is a blind hold-out.
- Do not change the language model. It must be DeepSeek v4 Flash.
- Every LLM call, including any new one you add, must go through
  `ReviewPipeline._complete` → `self.lm.complete(...)`. Never call `litellm`,
  `openai`, `requests` or any HTTP client directly: a direct call loses the
  hang guard and the provider fallback, and one hung provider request then
  stalls the entire subsample eval.
- Import `src/` modules by full package path
  (e.g. `from src.program.review_pipeline import ...`).
- Rework the review form, prompts, output contract, and decision logic in
  `src/program/review_pipeline.py` freely (new modules under `src/program/` are
  fine). Python may orchestrate — parse, validate, retry malformed JSON,
  combine structured signals — but the judgment about paper quality must come
  from the LLM.
- `langProBe/lm_provider.py`: Do not edit, delete, replace or bypass it: it owns the pinned model, the provider routing table, `reasoning_effort`, and the cross-provider fallback. Build task LMs with `build_task_lm()` and run your modules under `dspy.context(lm=self.lm, rm=self.rm)`

"""

OPTIMIZE_CONFIG = {}

# Engine 3 optimization config. Schema of record:
# src/engine/optimizer/v3/optimization_config.py (wire schema:
# src/api/schemas/requests.py::OptimizeRequest). Engine 3 removed the
# validation set: valset_path / valset_json / valset_bar / max_valset_size are
# rejected outright, and the architect compares branches on a fixed comparison
# set drawn from the trainset instead.
#
# Split providers (same wiring as arc-agi-2): the optimizer (opencode coding
# agent + reflection LM) runs on DeepInfra GLM-5.2 -- the vetted harness
# provider, the only one both opencode and the runner image can resolve without
# a custom provider block -- while the solver stays on DeepSeek-V4-Flash.
#
# The solver's provider wiring lives in `src/lm_provider.py`
# (the `docs/provider_fallback.md` C1-C8 contract), NOT in review_pipeline.py:
# one pinned model served by a ranked list of providers, gmi -> deepseek ->
# deepinfra, with a per-call divert to the next provider on any API error and a
# process-wide circuit breaker. `DEFAULT_PROVIDER` is `PROVIDER_PREFERENCE[0]`
# (gmi), the same as the other four seed programs. $LM_PROVIDER / $LM_FALLBACK repoint either per run without editing
# code OR rebuilding the seed bundle; changing the source needs both rebuilt.
#
# So a run needs GMI_API_KEY (or its alternative spelling GMI_CLOUD_API_KEY),
# DEEPINFRA_API_KEY and DEEPSEEK_API_KEY. All three hosts -- api.gmi-serving.com,
# api.deepinfra.com, api.deepseek.com -- are in runner/preflight.py and
# runner/squid.conf.
OPTIMIZE_CONFIG_LOCAL = {
    "repo_url": repo_path("PaperReview"),
    "program": "src.program.review_pipeline.ReviewPipeline",
    "metric": "src.metric.metric.review_decision_accuracy",
    # Engine 3: the old 100/100 train/val split folded into one pool, built by
    # scripts/build_trainval.py in the client repo (ordered concat, asserted
    # disjoint from the hold-out). NOT fenced by additional_deny_paths — under
    # engine 3 trainval IS the trainset, so fencing it would fence the agent's
    # own training data.
    "trainset_path": "data_splits/trainval.json",  # 200 rows = train(100) + val(100)
    "testset_path": "tests/testdata/testset.json",
    "input_keys": ["paper_text"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2",
    "coding_harness": "opencode",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2",
    # Engine 3 budget accounting: every architect eval row counts here --
    # subsamples AND comparison-set runs. Only the seed step's sanity rows and
    # the finalization test-set eval are free.
    #
    # 9000 targets 20 iterations. Comparison-set runs are paced by the credit
    # ledger (specs/archive-public/comparison_credits.md), which grants
    # `comparison_credits_per_iteration=1` by default, so an iteration costs
    # 5 subsample evals (5 x 50 = 250) + 1 comparison eval (100) =
    # 350 rows. Do not assume more than one comparison eval per iteration --
    # the ledger refuses the second.
    "max_metric_calls": 9000,
    "num_threads": 50,
    # Fixed comparison set drawn from the 200 trainset rows with seed 42.
    # Overlaps the training pool by design (engine 3 has no valset); matches
    # the 100-row valset the engine-2 runs admitted candidates on. Omitting
    # this falls back to the server default of 50 (src/config.py).
    "comparison_set_size": 100,
    "seed": 42,
    "additional_instructions": additional_instructions,
    # This names a branch in the CONTAINER WORKSPACE, not in the client
    # checkout. `build.py bundles` archives the tree of the branch named by
    # benchmarks.yaml::seed_branch (seed-program-deepseek-v4flash) with `git
    # archive`, which carries no refs at all, re-inits it as a one-commit repo
    # on a branch literally called `seed`, and bundles that; entrypoint.py then
    # clones `seed` and does `checkout -B main`. So the client branch NAME never
    # reaches the container -- only `main` and `seed` exist there. Naming the
    # seed branch here would not crash (seed.py warns and continues from HEAD)
    # but would never be what got checked out. Direct non-container runs against
    # a local checkout are the one case where it matters: pass
    # --set initial_branch=seed-program-deepseek-v4flash (this repo has no main).
    "initial_branch": "main",
    "subsample_size": 50,
    # Hard backstop (seconds) on ONE subsample eval subprocess — engine 3
    # redefines this as a hung-program killer, not a performance bound. Papers
    # run up to ~50k tokens at reasoning_effort=high and rows run parallel, so
    # the eval walls at its slowest row; the solver's hang guard caps a row at
    # TOTAL_BUDGET_S=2400s, so 7200 leaves headroom.
    "subsample_eval_timeout": 7200,
    "comparison_eval_timeout": 14400,  # comparison-set + finalization test evals
    # Post-run meta-reflection (advisory, non-fatal). Explicit because the
    # default differs by entry point: the API defaults it True, while
    # OptimizationConfig / local_launcher default it False.
    "enable_meta_reflection": True,
}
