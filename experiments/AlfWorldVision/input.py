from experiments.repo_paths import repo_path

additional_instructions = """
#### Task Type: Vision-Based Embodied Task Completion
ALFWorld (vision) is an embodied household-task benchmark. Each episode the
program receives the current AI2-THOR camera frame and must choose the next
text action (go to, take, put, open, heat, cool, clean, examine, ...) until
the goal state is reached or the step budget runs out. The metric is binary
task success from the ALFWorld evaluator.

Improve task success and reliability. You may change the program's prompts,
visual reasoning, memory, action selection, and control flow.

#### Constraints
- Preserve the configured task model, provider, reasoning effort, evaluator,
  dataset splits, and program entry point. The task model must remain
  Qwen 3.6 35B-A3B as configured.
- Do not access held-out data, evaluation internals, or fixed runtime
  configuration.
- Do not add task-specific solutions or hardcode per-scene action sequences.

#### Ideas
- Track visited locations, objects seen per receptacle, and completed subgoals
  explicitly across steps instead of relying on the frame alone.
- Detect and break repeated-action loops (same action, unchanged observation).
- Distinguish the task's required object/receptacle early and verify state
  (open/closed, heated/cooled, clean) before declaring subgoals done.
- Keep prompts compact: the model sees an image every step, so text budget is
  precious on long trajectories.
"""

# The runner overrides repo_url with the container workspace; this path only
# matters for a local, non-containerized run against the clean checkout.
_REPO = repo_path("alfworld-codeevolver")

# Engine 3 optimization config. Schema of record:
# src/engine/optimizer/v3/optimization_config.py. Engine 3 removed the
# validation set: train+val are folded into data/alfworld_vision_trainval.json
# (4839 rows = 4639 train + 200 val) and the architect compares branches on a
# fixed comparison set drawn from it instead.
_BASE = {
    "repo_url": _REPO,
    "program": "codeevolver_benchmarks.alfworld_vision_program.AlfWorldVisionProgram",
    "metric": "codeevolver_benchmarks.alfworld_vision_metric.alfworld_vision_metric",
    "trainset_path": "data/alfworld_vision_trainval.json",
    "testset_path": "data/alfworld_vision_test.json",
    "input_keys": ["game_file"],
    "reflection_lm": "deepinfra/zai-org/GLM-5.2",
    "coding_harness": "opencode",
    "coding_harness_model": "deepinfra/zai-org/GLM-5.2",
    # Shared budget and subsample: 5000 metric calls, 50-row subsamples
    # across every benchmark. alfworld rows are full Unity
    # episodes (~10s/op CPU-rendered vs ~2s on the qualified GPU hosts), so
    # run this benchmark on a GPU droplet.
    "max_metric_calls": 5000,
    # Under real load the workers idle-wait on the vision model (~18% CPU each, host
    # load ~1.5/8), and RAM (~0.9 GiB/worker of 32) and VRAM (~165 MiB of
    # 20 GiB) have multiples of headroom, so 15 oversubscribed workers raise
    # throughput rather than thrash. MUST equal ALFWORLD_SERVER_COUNT on the
    # host running the fleet -- set that to 15 in the controller env; a
    # higher thread count than workers just queues rows behind the renderers.
    "num_threads": 15,
    "subsample_size": 50,
    # 100 is the largest comparison-set size that still guarantees 20 iterations
    # inside the 5000-call budget: ~2 subsamples (100 rows) + 1 credit-paced
    # comparison run (100) per iteration x 20, plus the one-time variance
    # tool (5 x 50 = 250), totals ~4250 with slack for hungrier iterations.
    # On a GPU host a 100-episode comparison takes tens of minutes.
    "comparison_set_size": 100,
    # Optimizer iteration cap, matching the qualified full profile (20).
    "debug_max_iterations": 20,
    "seed": 42,
    "additional_instructions": additional_instructions,
    "initial_branch": "main",
    # Hung-program backstops, not performance bounds (see hover's rationale).
    # Episodes render frame-by-frame in Unity; CPU-rendered hosts run ~5x
    # slower than GPU hosts, and both must clear the backstop.
    "subsample_eval_timeout": 14400,  # 4 hours
    "comparison_eval_timeout": 28800,  # 8 hours - comparison + finalization test evals
    "enable_meta_reflection": True,
    # The simulator runtime, the metric, and the fixed task-model config must
    # stay out of the architect's reach: additional_instructions forbids
    # editing them, this enforces it.
    "additional_deny_paths": [
        "codeevolver_benchmarks/_tracing.py",
        "codeevolver_benchmarks/common.py",
        "codeevolver_benchmarks/runtime_config.json",
        "codeevolver_benchmarks/alfworld_vision_metric.py",
        "codeevolver_benchmarks/alfworld_runtime.py",
        "codeevolver_benchmarks/appworld_program.py",
        "codeevolver_benchmarks/appworld_metric.py",
        "codeevolver_benchmarks/appworld_runtime.py",
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
# CE_CONFIG=SMOKE_CONFIG_LOCAL through). Proves the Xvfb/Xorg display path,
# the 3.9 runtime venv, the Unity renderer, egress, and the metric end to end.
SMOKE_CONFIG_LOCAL = {
    **_BASE,
    # The REAL program and metric on one row: reset, RGB frame to the model,
    # action parsing, /step, and a scored episode -- not just an init check.
    "trainset_path": "data/alfworld_vision_smoke.json",
    # Distinct from the trainset: the entrypoint SEQUESTERS the testset file
    # out of the workspace, so pointing both at one file would break the
    # trainset read. One row keeps the finalization eval trivial.
    "testset_path": "data/alfworld_vision_smoke_test.json",
    "max_metric_calls": 10,
    "num_threads": 1,
    "subsample_size": 1,
    "comparison_set_size": 1,
    "debug_max_iterations": 1,
    "enable_meta_reflection": False,
}
