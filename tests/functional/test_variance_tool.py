"""Copyright © 2026 440 Labs LLC

`run_variance_estimate` — a real architect session calls the real tool.

The offline suite (`tests/unit/test_variance_estimate.py`) proves the tool's
mechanics over a stubbed evaluator. What it cannot prove is the live path:
that a real model on each coding harness actually sees and can invoke a
ZERO-argument MCP / SDK tool (the one schema shape no other live tool
exercises), and that the real evaluator round-trip writes distinct
`iteration_{N}_variance[_k]/` trace directories and lands the estimate on the
run state. This test does exactly that, and nothing more:

- a real FactChecker clone (`LocalClientRepoEnv`) on a fresh `codeevolver-`
  branch, a real `CEState`, and the REAL tool registry
  (`IterationArchitect.build_tool_registry` via `super()`), with only
  `run_variance_estimate` wrapped in a call counter;
- the system prompt's additional instructions say "this is a test: call
  `run_variance_estimate` once, then stop" — no coding, no close tool
  (`agent_managed_git=False`), no skills, and the closing-protocol watchdogs
  are disabled so a deliberately change-free session is not re-prompted;
- the eval is as cheap as the tool allows: `subsample_size=2` and
  `variance_estimate_samples=2` → four rows of the fixture program's LM.

Asserted: exactly one successful tool call (a second call is the
once-per-run refusal and spends nothing); the run state and context carry
the estimate; two `eval_sample="variance"` records and four counted rows;
`iteration_1_variance/` + `iteration_1_variance_2/` each hold two rows and no
third directory exists; the next iteration's system prompt renders the
numbers. The fixture repo leaves DSPy's cache on, so an
`identical_outputs_warning` is a plausible, reported outcome — that is the
detector working, not a failure.

To run::

    python local_test_app.py variance_tool --coding-harness claude
    python local_test_app.py variance_tool --coding-harness opencode \\
        --coding-harness-model deepinfra/zai-org/GLM-5.2
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from tests.functional.test_e2e import DEFAULT_SEED_BRANCH

SUBSAMPLE_SIZE = 2
VARIANCE_SAMPLES = 2
ITERATION_NUM = 1
VARIANCE_MAX_ATTEMPTS = 2

# What the architect is told, on top of the production system prompt. The
# production prompt already describes the tool (the "not yet run" paragraph
# of the Sampling Guidance section); this only narrows the session to it.
ADDITIONAL_INSTRUCTIONS = """\
THIS IS A TEST RUN of the `run_variance_estimate` tool, not a real
optimization. Your only task this iteration:

1. Call the `run_variance_estimate` tool exactly once (it takes no
   arguments). It runs the current program a few times on a tiny fixed
   sample and returns the per-run scores and the noise estimate.
2. Briefly report the numbers it returned, then end your turn.

Do NOT edit any files, do NOT call `coding_agent`, do NOT run `run_eval`,
do NOT create branches or commit anything, and do NOT update memory.md.
There is no close_iteration step in this test — just stop after reporting.
"""


def _git(workspace: str, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=workspace, check=True, capture_output=True, text=True,
    ).stdout.strip()


def run_variance_tool(
    env,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    repo_url: str = "https://github.com/julianghadially/FactChecker",
    program: str = "src.factchecker.simple.factchecker_program:FactCheckerProgram",
    metric: str = "src.codeevolver.metric:metric",
    trainset_path: str = "data/FacTool_QA_train_10examples_normalized.jsonl",
    initial_branch: str = DEFAULT_SEED_BRANCH,
    primary_metric: str = "accuracy",
    eval_timeout: int = 1800,
    max_turns: int = 30,
) -> dict:
    """Run one architect session in-process and return the check results."""
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.evaluator.dataset_config import DatasetConfig
    from src.engine.optimizer.v3.ce_state import CEState
    from src.engine.optimizer.v3.eval_log import VARIANCE_EVAL_SAMPLE, variance_suffix
    from src.engine.optimizer.v3.iteration_architect import (
        ArchitectConfig,
        IterationArchitect,
    )
    from src.engine.optimizer.v3.iteration_context import IterationContext
    from src.engine.optimizer.v3.prompts import (
        build_architect_system_prompt,
        build_memory_md_template,
    )
    from src.engine.paths import memory_dir
    from src.engine.telemetry.trace_layout import iter_rows

    # The Claude Agent SDK refuses --dangerously-skip-permissions as root
    # unless IS_SANDBOX=1; every driver sets it, so do the same in-process.
    os.environ.setdefault("IS_SANDBOX", "1")

    env.prepare_workspace(repo_url=repo_url, branch=initial_branch)
    workspace = env.workspace_dir
    result: dict = {
        "success": False,
        "checks": {},
        "coding_harness": coding_harness,
        "coding_harness_model": coding_harness_model,
        "subsample_size": SUBSAMPLE_SIZE,
        "variance_samples": VARIANCE_SAMPLES,
    }
    checks = result["checks"]
    try:
        datasets = DatasetConfig.prepare(
            workspace, trainset_json=None, trainset_path=trainset_path,
            comparison_set_size=None, testset_json=None,
        )
        evaluator_config = EvaluatorConfig(
            program_factory=program, metric_fns=[metric],
            metric_names=[primary_metric], primary_metric=primary_metric,
            trainset_path=datasets.trainset_rel_path,
        )

        ts = time.strftime("%Y%m%d%H%M%S")
        seed_branch = f"codeevolver-{ts}-main"
        _git(workspace, "checkout", "-q", "-b", seed_branch)
        seed_sha = _git(workspace, "rev-parse", "HEAD")
        state = CEState(seed_branch=seed_branch, seed_sha=seed_sha)

        # The production prompt tells the architect to read memory.md first;
        # give it the seed-round template so that read succeeds.
        Path(memory_dir()).mkdir(parents=True, exist_ok=True)
        (Path(memory_dir()) / "memory.md").write_text(
            build_memory_md_template(
                seed_branch=seed_branch, architecture_title="FactChecker (seed)",
            )
        )

        class _TestArchitect(IterationArchitect):
            """Real registry; `run_variance_estimate` wrapped with a counter.
            No watchdogs — the session is change-free by instruction."""

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.tool_calls: list[dict] = []

            def build_tool_registry(self, *, agent_harness):
                registry = super().build_tool_registry(agent_harness=agent_harness)
                real = registry["run_variance_estimate"]
                tester = self

                def run_variance_estimate() -> dict:
                    try:
                        payload = real()
                    except Exception as exc:  # noqa: BLE001
                        tester.tool_calls.append({"error": f"{type(exc).__name__}: {exc}"})
                        raise
                    tester.tool_calls.append(dict(payload))
                    return payload

                registry["run_variance_estimate"] = run_variance_estimate
                return registry

            def _run_watchdog(self, watchdog, system_prompt, tool_registry) -> None:
                return None

        ctx = IterationContext(
            iteration_idx=ITERATION_NUM,
            parent_locked=True,
            subsample_size=SUBSAMPLE_SIZE,
            variance_estimate_samples=VARIANCE_SAMPLES,
            start_branch=seed_branch,
            start_sha=seed_sha,
            primary_branch_at_start=seed_branch,
        )
        cfg = ArchitectConfig(
            run_timestamp=ts,
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=program,
            evaluator=evaluator_config,
            trainset_ids=list(datasets.trainset_ids),
            subsample_timeout=eval_timeout,
            subsample_size=SUBSAMPLE_SIZE,
            variance_estimate_samples=VARIANCE_SAMPLES,
            max_turns=max_turns,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
            # No close tool and no skills: the session's only job is the call.
            agent_managed_git=False,
            stage_project_skills=False,
        )
        architect = _TestArchitect(
            context=ctx, state=state, workspace=workspace, config=cfg,
        )

        run_error = None
        try:
            architect.run()
        except Exception as exc:  # noqa: BLE001
            run_error = f"{type(exc).__name__}: {exc}"
        result["run_error"] = run_error

        ok_calls = [c for c in architect.tool_calls if c.get("ok") is True]
        refused = [c for c in architect.tool_calls if c.get("refused")]
        errored = [c for c in architect.tool_calls if c.get("error")]
        result["tool_call_count"] = len(architect.tool_calls)
        result["refused_calls"] = len(refused)
        result["tool_errors"] = [c["error"] for c in errored]
        est = state.variance_estimate
        result["variance_estimate"] = est

        checks["tool_called"] = len(architect.tool_calls) >= 1
        checks["exactly_one_successful_call"] = len(ok_calls) == 1
        checks["extra_calls_were_refused"] = len(architect.tool_calls) - len(ok_calls) == len(refused)
        checks["state_recorded"] = bool(est) and est.get("n_runs") == VARIANCE_SAMPLES
        checks["ctx_flags"] = ctx.variance_estimate_ran is True and ctx.variance_estimate == est
        checks["sample_size"] = bool(est) and len(est.get("row_ids") or []) == SUBSAMPLE_SIZE
        checks["scores_per_run"] = bool(est) and len(est.get("aggregate_scores") or []) == VARIANCE_SAMPLES
        checks["std_consistent"] = bool(est) and (
            abs(est["run_to_run_std"] ** 2 * est["n_rows"] - est["within_item_variance"]) < 1e-9
        )
        checks["measured_sha_is_head"] = bool(est) and est.get("sha") == _git(workspace, "rev-parse", "HEAD")
        variance_records = [
            r for r in ctx.eval_records if r.get("eval_sample") == VARIANCE_EVAL_SAMPLE
        ]
        checks["two_variance_eval_records"] = len(variance_records) == VARIANCE_SAMPLES
        checks["no_train_eval_recorded"] = ctx.eval_history == [] and ctx.initial_subsample_score is None
        checks["rows_counted"] = state.total_num_evals == SUBSAMPLE_SIZE * VARIANCE_SAMPLES

        traces = Path(env.traces_dir)
        rows_per_dir = {}
        for k in range(VARIANCE_SAMPLES):
            name = f"iteration_{ITERATION_NUM}_{variance_suffix(k)}"
            rows_per_dir[name] = len(list(iter_rows(str(traces / name))))
        result["trace_rows"] = rows_per_dir
        checks["trace_dirs_have_two_rows_each"] = all(
            n == SUBSAMPLE_SIZE for n in rows_per_dir.values()
        )
        checks["no_extra_trace_dir"] = not (
            traces / f"iteration_{ITERATION_NUM}_{variance_suffix(VARIANCE_SAMPLES)}"
        ).exists()
        evals_log = traces / f"iteration_{ITERATION_NUM}_evals.jsonl"
        logged = [
            json.loads(l) for l in evals_log.read_text().splitlines() if l.strip()
        ] if evals_log.exists() else []
        checks["evals_jsonl_has_variance_lines"] = (
            sum(1 for l in logged if l.get("eval_sample") == VARIANCE_EVAL_SAMPLE) == VARIANCE_SAMPLES
        )

        # The next iteration's prompt: the loop stamps state.variance_estimate
        # onto a fresh context, and the Sampling Guidance paragraph switches
        # from "call the tool" to the measured numbers.
        next_ctx = IterationContext(
            iteration_idx=ITERATION_NUM + 1, parent_locked=True,
            subsample_size=SUBSAMPLE_SIZE,
            variance_estimate=state.variance_estimate,
            variance_estimate_samples=VARIANCE_SAMPLES,
        )
        next_prompt = build_architect_system_prompt(
            additional_instructions=None, program_path=program,
            workspace_path=workspace, context=next_ctx,
        )
        checks["next_prompt_renders_estimate"] = bool(est) and (
            f"{est['run_to_run_std']:.4f}" in next_prompt
            and "run_variance_estimate" not in next_prompt
        )
        result["identical_outputs_warning"] = bool(est and est.get("identical_outputs_warning"))
        result["success"] = run_error is None and all(bool(v) for v in checks.values())
    except Exception as exc:  # noqa: BLE001
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            _git(workspace, "checkout", "-q", initial_branch)
        except Exception:  # noqa: BLE001
            pass
        env.cleanup()
    return result


def variance_tool_failures(result: dict) -> list[str]:
    """Hard failures for the entrypoint's pass/fail line."""
    failures: list[str] = []
    if result.get("error"):
        failures.append(result["error"])
    if result.get("run_error"):
        failures.append(f"architect session raised: {result['run_error']}")
    for name, ok in (result.get("checks") or {}).items():
        if not ok:
            failures.append(f"check failed: {name}")
    return failures


__all__ = [
    "ADDITIONAL_INSTRUCTIONS",
    "SUBSAMPLE_SIZE",
    "VARIANCE_MAX_ATTEMPTS",
    "VARIANCE_SAMPLES",
    "run_variance_tool",
    "variance_tool_failures",
]
