"""Copyright © 2026 440 Labs LLC

Per-iteration analysis agent (engine 3 change summary).

The architect writes memory.md itself, so this agent ONLY writes
`iteration_{N}_change_summary.md` (and, on landing iterations, rewrites
`codeevolver.md` in place — the loop commits that, see
`close_effects.commit_codeevolver_md_if_modified`). Best-effort — a flaky
summary never fails an already-recorded iteration.

Runs for non-landing iterations too. Those are where the run's negative
results live ("we tried X, it regressed via Y, that lever is exhausted") and
they are often the majority of a run; gating on a landed commit would leave
them with no written record at all.
"""

from __future__ import annotations

import logging
import os

from src.engine.agent.token_usage import TokenUsage
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.paths import traces_dir
from src.engine.telemetry.trace_layout import HOLDOUT_DENY_FRAGMENT, eval_trace_dir
from src.engine.services.subprocess_service import run_inspection_agent

from .clean_slate import CLEAN_SLATE_REPORT_DENY_FRAGMENT
from .close_effects import write_attempted_diff
from .coding_harness_config import CodingHarnessConfig
from .iteration_context import IterationContext
from .prompts import build_analysis_agent_prompt

logger = logging.getLogger(__name__)


def analysis_deny_paths(
    *, testset_rel: str | None, additional_deny_paths: list[str],
) -> list[str]:
    """Holdout fence for the analysis agent: the test set (by rel path and
    basename), the `_holdout` trace-directory fragment, the clean-slate
    report, and any client-supplied extras."""
    deny_paths: list[str] = []
    if testset_rel:
        deny_paths.append(testset_rel)
        deny_paths.append(os.path.basename(testset_rel))
    deny_paths.append(HOLDOUT_DENY_FRAGMENT)
    deny_paths.append(CLEAN_SLATE_REPORT_DENY_FRAGMENT)
    deny_paths.extend(p for p in additional_deny_paths if p)
    return deny_paths


def run_analysis_agent(
    ctx: IterationContext,
    *,
    workspace: str,
    evaluator_config: EvaluatorConfig,
    subsample_size: int,
    additional_deny_paths: list[str],
    agent_model: str | None,
    coding_harness_config: CodingHarnessConfig,
) -> None:
    """Run the analysis agent for `ctx`'s iteration. Never raises."""
    child_trace_dir = eval_trace_dir(traces_dir(), ctx.iteration_idx, "child")
    parent_trace_dir = eval_trace_dir(traces_dir(), ctx.iteration_idx, "parent")
    # The inspection agent has no Bash tool, so it cannot run `git diff`
    # itself — and for a reverted iteration there is no commit to diff
    # anyway. Materialize the diff of what was ATTEMPTED (parent tree vs
    # the best-scoring tree evaluated this iteration) into a file it can
    # Read. `run_eval` wrote that tree object even though the content was
    # never committed, and `anchor_iteration_state` keeps it alive.
    attempted_diff_path = write_attempted_diff(ctx, workspace=workspace)
    prompt = build_analysis_agent_prompt(
        iteration_num=ctx.iteration_idx,
        workspace_path=workspace,
        branch=ctx.child_branch,
        parent_sha=ctx.parent_sha,
        close_sha=ctx.close_sha,
        change_request=(
            "\n---\n".join(ctx.change_requests) if ctx.change_requests else ""
        ),
        subsample_baseline=ctx.initial_subsample_score,
        child_trace_dir=child_trace_dir,
        parent_trace_dir=parent_trace_dir,
        subsample_size=subsample_size,
        changes_landed=ctx.changes_landed,
        subsample_best=ctx.best_subsample_score,
        eval_series=[
            (rec.get("eval_sample"), rec.get("score"))
            for rec in ctx.eval_records
        ],
        attempted_diff_path=attempted_diff_path,
    )

    deny_paths = analysis_deny_paths(
        testset_rel=evaluator_config.testset_path,
        additional_deny_paths=additional_deny_paths,
    )

    try:
        result = run_inspection_agent(
            workspace=workspace,
            prompt=prompt,
            max_turns=25,
            iteration_num=ctx.iteration_idx,
            writable=True,
            deny_paths=deny_paths,
            model=agent_model,
            coding_harness=coding_harness_config.coding_harness,
            coding_harness_model=coding_harness_config.coding_harness_model,
            # Best-effort per this function's contract, but a hung session
            # blocks the whole iteration loop behind it — 10 min is generous
            # for a change-summary write, well under the 25-turn/120s-per-turn
            # (50 min) default it overrides. See the external_directory rule
            # in harnesses/opencode/permissions.py and the hard_kill path in
            # harnesses/opencode/harness.py for why this override actually
            # terminates the process.
            timeout_seconds=600,
        )
        usage_dict = result.get("usage") or {}
        ctx.record_token_usage(
            usage_dict.get("model") or agent_model,
            TokenUsage.from_dict(usage_dict),
        )
        if not result.get("success"):
            logger.warning(
                "iteration %d: analysis agent did not write a change "
                "summary (%s)",
                ctx.iteration_idx, result.get("error"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: analysis agent raised (non-fatal): %s",
            ctx.iteration_idx, exc,
        )


__all__ = ["analysis_deny_paths", "run_analysis_agent"]
