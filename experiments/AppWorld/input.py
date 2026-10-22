from experiments.repo_paths import repo_path

additional_instructions = """
#### Task Type: Interactive API-Based Task Completion
AppWorld is an interactive coding-agent benchmark set in a simulated world of
everyday applications. The program must understand a user's request, discover
the relevant APIs, execute code against persistent application state, and
explicitly indicate when the task is complete. The metric measures whether the
requested state changes or answer are correct while also checking for
unintended collateral changes.

The core difficulty is reliable multi-step execution. Tasks may require API
discovery, authentication, pagination, coordination across multiple apps,
tracking intermediate state, recovering from failed calls, and deciding when
enough evidence has been gathered to complete the task.

#### What's Allowed
- Change the program's prompts, memory, tool use, reasoning structure, API
  discovery process, error handling, and control flow.
- Add generic planning, verification, state tracking, and recovery mechanisms.
- Use the documented AppWorld APIs exposed through the existing environment.

#### Constraints
- Preserve the configured task model, provider, reasoning effort, evaluator,
  dataset splits, and program entry point. The task model must remain
  DeepSeek V4 Flash as configured.
- Do not access held-out data, evaluation internals, ground-truth solutions, or
  fixed runtime configuration.
- Do not add task-specific solutions, hardcoded task IDs, or hardcoded API
  sequences for individual examples.

#### Ideas
- Improve how the program discovers and reads API documentation before acting.
- Track authentication, intermediate objects, pagination, and completed
  subgoals explicitly across steps.
- Verify identifiers and current application state before irreversible actions.
- Add recovery behavior for malformed code, failed API calls, missing fields,
  and incomplete results.
- Distinguish information-seeking tasks from state-changing tasks and handle
  completion answers carefully.
- Use concise persistent memory so long trajectories do not lose earlier facts.

### Available Services
- AppWorld application APIs through the persistent `apis` object.
- `api_docs` for application descriptions and API specifications.
- `supervisor` for the acting user's account and profile information.

### Additional Notes
- Different tasks require different numbers of steps and different apps.
- APIs should be inspected before use, and paginated results may require
  multiple calls.
- Successful execution requires calling the supervisor completion API, with an
  answer only when the task requests one.
"""

# The runner overrides repo_url with the container workspace; this path only
# matters for a local, non-containerized run against the clean checkout.
_REPO = repo_path("appworld-codeevolver")

# Engine 3 optimization config. Schema of record:
# src/engine/optimizer/v3/optimization_config.py. Engine 3 removed the
# validation set: valset_path / valset_bar / max_valset_size are rejected;
# train+val are folded into data/appworld_trainval.json (315 rows, from the
# full-pool challenge split) and the architect compares branches on a fixed
# comparison set drawn from it instead. Held-out test: 417 rows.
_BASE = {
    "repo_url": _REPO,
    "program": "codeevolver_benchmarks.appworld_program.AppWorldProgram",
    "metric": "codeevolver_benchmarks.appworld_metric.appworld_metric",
    "trainset_path": "data/appworld_trainval.json",
    "testset_path": "data/appworld_test.json",
    "input_keys": ["task_id"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2",
    "coding_harness": "opencode",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2",
    # 6000 metric calls, sized from the measured seed
    # variance (run-to-run std 0.16 at subsample 50 — dominated by a
    # correlated provider-level component that a bigger subsample cannot
    # shrink, so the extra calls go to the comparison set instead):
    # 20 iterations x (2x50 subsample + 150 comparison) + the one-time
    # variance tool (5x50) = 5250, with headroom for hungrier iterations.
    "max_metric_calls": 6000,
    # Matches the AppWorld HTTP server pool's capacity (APPWORLD_SERVER_COUNT
    # servers, sessions multiplexed).
    "num_threads": 16,
    "subsample_size": 50,
    # ~half the 315-row trainval pool. The measured seed noise is high and
    # correlated, so a bigger FIXED comparison set is the one knob that helps
    # branch selection when subsample readings re-roll.
    "comparison_set_size": 150,
    # Optimizer iteration cap.
    "debug_max_iterations": 20,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "main",
    # Hung-program backstops, not performance bounds (see hover's rationale).
    # AppWorld rows are 40-step agent trajectories against a slow reasoning
    # model, so both sit above the engine defaults.
    "subsample_eval_timeout": 14400,  # 4 hours
    "comparison_eval_timeout": 28800,  # 8 hours - comparison + finalization test evals
    "enable_meta_reflection": True,
    # The simulator runtime, the metric, and the fixed task-model config must
    # stay out of the architect's reach: additional_instructions forbids
    # editing them, this enforces it. data/ carries the split definitions
    # (task-id lists); the holdout task ids themselves are additionally fenced
    # automatically via testset_path.
    "additional_deny_paths": [
        "codeevolver_benchmarks/_tracing.py",
        "codeevolver_benchmarks/common.py",
        "codeevolver_benchmarks/runtime_config.json",
        "codeevolver_benchmarks/appworld_metric.py",
        "codeevolver_benchmarks/appworld_runtime.py",
        "codeevolver_benchmarks/alfworld_vision_program.py",
        "codeevolver_benchmarks/alfworld_vision_metric.py",
        "codeevolver_benchmarks/alfworld_runtime.py",
        "requirements.txt",
        "alfworld-runtime-requirements.txt",
        ".env",
        ".env.example",
        "configs/",
        "data/",
        "experiment_data/",
        "langProBe/",
        ".runtime/",
        "scripts/",
    ],
}

OPTIMIZE_CONFIG_LOCAL = {
    **_BASE,
}

# One-row wiring check for the containerized runner (matrix.local.yaml passes
# CE_CONFIG=SMOKE_CONFIG_LOCAL through). Proves the simulator venv, the server
# pool, egress, and the metric end to end without committing provider spend.
SMOKE_CONFIG_LOCAL = {
    **_BASE,
    # The REAL program and metric on one row -- a genuine task-model call and a
    # trusted evaluation, not just an init check. (The *SmokeProgram stubs only
    # prove initialization; they never call the model or score anything.)
    "trainset_path": "data/appworld_smoke.json",
    # Distinct from the trainset: the entrypoint SEQUESTERS the testset file
    # out of the workspace, so pointing both at one file would break the
    # trainset read. One row keeps the finalization eval trivial.
    "testset_path": "data/appworld_smoke_test.json",
    "max_metric_calls": 10,
    "num_threads": 1,
    "subsample_size": 1,
    "comparison_set_size": 1,
    "debug_max_iterations": 1,
    "enable_meta_reflection": False,
}
