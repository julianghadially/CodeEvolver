from experiments.repo_paths import repo_path

additional_instructions = """
#### Task: ARC-AGI-2 (abstraction and reasoning)
Each task gives 2-5 demonstration input->output grid pairs. Deduce the abstract
transformation rule and apply it to the test input grid (you must also choose
the output dimensions, which can differ from the input). Scoring is strict
exact-match per test grid — progress is discrete: a task counts only when the
whole grid is exactly right.

ARC-AGI-2 is deliberately built so that memorized rules do not transfer:
superficially similar tasks often have different underlying rules. It is
substantially harder than ARC-AGI-1 — expect a low absolute score and judge
changes by mechanism, not by a single noisy subsample.

#### Constraints (violating any of these fails the run):
- The solver MUST call the LLM at inference time on EVERY task and infer the
  rule dynamically from the demonstrations. A deterministic code-only solver,
  a fixed transform catalog, or rule enumeration keyed to grid shape/pattern
  is an automatic failure — such programs overfit the subsample and collapse
  on held-out data. Python may orchestrate (build prompts, parse output,
  verify candidates against train pairs, retry) but must NOT be the source of
  the transformation rule.
- NEVER read or use `tests/testdata/testset.json` — it is a blind hold-out.
- Do not change the language model used for the task. It must remain DeepSeek v4 Flash
- `src/infra/lm_provider.py`: Do not edit, delete, replace or bypass it. It owns
  the pinned model, the provider routing table and keys, `reasoning_effort`,
  the streaming hang guard and the cross-provider fallback. Reading it is fine
  and encouraged — that is where `build_task_lm()` is defined.
- Every LLM call, including any new one you add, must go through
  `ARCPipeline._call_llm` → `self.lm.completion(...)`. Never call `litellm`,
  `openai`, `requests` or any HTTP client directly: a direct call loses the
  hang guard and the provider fallback, and one hung provider request then
  stalls the entire parallel eval.
- Import `src/` modules by full package path (e.g. `from src.program.my_module import helper`).

#### Design guidance:
- Each task carries its own demonstration pairs, and whether a candidate rule
  reproduces them is checkable in Python without spending an eval row. How you
  use that is yours to decide.
- The generality must live in a per-task, inference-time procedure — NOT in a
  fixed, pre-written library of transforms selected by grid shape or pattern.
  That is the failure mode this benchmark is designed to punish.
- A subsample eval must fit its timeout; extra per-task LLM round-trips have
  repeatedly blown it on arc-agi. If a design will not fit, reduce per-task
  passes/retries rather than letting the eval time out (a timeout yields no
  usable score).
- Iterate on the AI system, including the prompt, reasoning, reasoning about how to reason, candidate generation, and
  verification logic in `src/program/arc_pipeline.py`; new modules under
  `src/program/` are fine.

#### Reading results (the engine drives the eval/reflect protocol for you):
- `failed_rows` / `row_errors` / `broken_program_warning` mean your change
  crashed the program (the 0.0 is from crashes, not wrong answers). Fix that
  and re-run before any strategy reasoning.
"""

OPTIMIZE_CONFIG = {}

# Engine 3 optimization config. Schema of record:
# src/engine/optimizer/v3/optimization_config.py (wire schema:
# src/api/schemas/requests.py::OptimizeRequest). Engine 3 removed the
# validation set: valset_path / valset_json / valset_bar / max_valset_size are
# rejected outright, and the architect compares branches on a fixed comparison
# set drawn from the trainset instead.
#
# Split providers: the optimizer (opencode coding agent + reflection LM) runs on
# DeepInfra GLM-5.2 -- the vetted harness provider, the only one both opencode
# and the runner image can resolve without a custom provider block -- while the
# solver stays on DeepSeek-V4-Flash.
#
# The solver's provider wiring lives in
# `src/infra/lm_provider.py` (the `docs/provider_fallback.md` C1-C8 contract),
# NOT in arc_pipeline.py: one pinned model served by a ranked list of providers,
# gmi -> deepseek -> deepinfra, with a per-call divert to the next provider on
# any API error and a process-wide circuit breaker. `DEFAULT_PROVIDER` is
# `PROVIDER_PREFERENCE[0]` -- gmi primary, deepseek covering. $LM_PROVIDER
# / $LM_FALLBACK repoint it per run without editing code OR rebuilding the seed
# bundle; changing DEFAULT_PROVIDER needs both rebuilt.
#
# So a run needs GMI_API_KEY (or its alternative spelling GMI_CLOUD_API_KEY),
# DEEPSEEK_API_KEY and DEEPINFRA_API_KEY. All three hosts --
# api.gmi-serving.com, api.deepseek.com, api.deepinfra.com -- are in
# runner/preflight.py and runner/squid.conf.
OPTIMIZE_CONFIG_LOCAL = {
    "repo_url": repo_path("ARC-AGI-2"),
    "program": "src.program.arc_pipeline.ARCPipeline",
    "metric": "src.metric.metric.arc_grid_accuracy",
    # Engine 3: the old 800/200 train/val split folded into one pool, built by
    # scripts/build_trainval.py in the client repo (ordered concat, asserted
    # disjoint from the hold-out). NOT fenced by additional_deny_paths -- under
    # engine 3 trainval IS the trainset, so fencing it would fence the agent's
    # own training data.
    "trainset_path": "data_splits/trainval.json",  # 1000 rows = train(800) + val(200)
    "testset_path": "tests/testdata/testset.json",  # 120-row blind hold-out
    "input_keys": ["train", "test"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2",
    "coding_harness": "opencode",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2",
    # ARC-AGI-2 is the slowest benchmark of the suite, so its budget is cut
    # down. Every architect eval row counts here -- subsamples AND comparison-set runs; only the seed step's
    # sanity rows and the finalization test-set eval are free.
    "max_metric_calls": 5600,
    # Rows run fully parallel, so an eval walls at its SLOWEST row. Matches
    # arc-agi: below ~25 the extra waves cost more than the per-request speedup
    # (the GMI solver endpoint is throughput-saturated, so each in-flight
    # request just streams slower).
    "num_threads": 50,
    # Fixed comparison set drawn from the trainset with seed 42. Overlaps the
    # training pool by design (engine 3 has no valset). Omitting this falls back
    # to the server default of 50 (src/config.py).
    #
    # 150, not 100: this benchmark's scores sit near p=0.5 where binomial
    # variance peaks, so it is noisier at a given N than a benchmark like hover
    # (p~0.1). Measured on the arc-agi-2 seed (50 rows x 5 repeats,
    # within_item_variance 0.162): SE is 4.0% at N=100 and 2.8%
    # at N=200, versus the ~2-3% the architect needs for the promotion gate to
    # mean anything.
    "comparison_set_size": 150,
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
    # --set initial_branch=seed-program-deepseek-v4flash (this repo's `main` is
    # an older solver lineage with no trainval pool).
    "initial_branch": "main",
    "subsample_size": 50,
    # Hard backstop (seconds) on ONE subsample eval subprocess -- engine 3
    # redefines this as a hung-program killer, not a performance bound. A
    # subprocess timeout raises before any score comes back, so a tight cap
    # throws away the whole subsample's spend and hands the architect a tool
    # error instead of a number. DeepSeek at reasoning_effort=high has produced
    # single rows at 40 min; the solver's own hang guard caps a row at
    # TOTAL_BUDGET_S=2400s, so 7200 leaves real headroom.
    "subsample_eval_timeout": 7200,
    "comparison_eval_timeout": 14400,  # comparison-set + finalization test evals
    # Post-run meta-reflection (advisory, non-fatal). Explicit because the
    # default differs by entry point: the API defaults it True, while
    # OptimizationConfig / local_launcher default it False.
    "enable_meta_reflection": True,
    # NOTE on the seed gate: engine 3 dropped `allow_zero_seed_score`, and
    # validate_sandbox_environment ABORTS when the primary metric is 0.0 across
    # every seed sanity row (src/engine/sandbox/verify_environment.py). That
    # gate is not a draw you can re-roll -- seed.py pins n_rows=5 and the check
    # takes `trainset_ids[:n_rows]`, so it is always the SAME first five rows.
    #
    # Treat a 0.0 seed here as a broken metric/program until proven otherwise --
    # check the traces' feedback strings before blaming ARC-AGI-2's difficulty.
    # Note the hold-out is the 120 public EVALUATION tasks while trainval is the
    # 1000 public TRAINING tasks (client repo's split_data.py), so seed scores on
    # trainval rows are expected to be far higher than the final test score.
}
