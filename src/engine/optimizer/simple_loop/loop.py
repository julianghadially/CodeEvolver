"""Copyright © 2026 440 Labs LLC

The simple loop: one linear branch, one fresh agent session per iteration.

Stages, in order (each a function below):

    initialize_repository   run branch + sandbox validation + seed record
    run_session             fresh OpenCode session with `run_eval` only
    commit_session_end      auto-commit whatever the agent left uncommitted
    check_session_constraints   LM judge on the session diff; revert on fail
    record_session          history record + progress callback
    check_stalled           halt after N consecutive empty/failed sessions
    finalize_on_holdout     seed + final program on the held-out test set
    post-run auditors       meta-reflection and run-integrity (advisory)

Budget: eval rows only, enforced inside `run_eval`. Safety nets: wall clock,
cancellation, and the stall counter. No score is attributed to the closing
commit; the final tree is whatever HEAD the last session left.
"""

from __future__ import annotations

import logging
import os
import random
import time
from typing import Any, Mapping

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DatasetConfig
from src.engine.evaluator.holdout import mean_or_none, sample_test_scores
from src.engine.evaluator.sandbox_client import run_eval
from src.engine.optimizer.v3.coding_harness_config import CodingHarnessConfig
from src.engine.schemas.agent_config import BaseAgentConfig
from src.engine.schemas.iteration_context import BaseIterationContext
from src.engine.schemas.iteration_history import IterationHistory
from src.engine.optimizer.v3.optimization_config import OptimizationConfig
from src.engine.paths import history_dir
from src.engine.sandbox.verify_environment import validate_sandbox_environment
from src.engine.services.callbacks import JobCallbacks
from src.engine.services.job_context import JobContext
from src.engine.services.subprocess_service import (
    clean_python_bytecode,
    git_add_and_commit,
    git_checkout,
    git_create_branch_simple,
    git_fetch,
    git_get_sha,
    git_push_authenticated,
    git_set_identity,
)

from .constraint_check import build_constraint_checker, check_session_constraints
from .prompt import build_simple_loop_prompt
from .records import session_record
from .session import run_session
from .state import SimpleLoopState

logger = logging.getLogger(__name__)

METHOD = "simple_loop"

# Own copies (not imported from the EE loop): a soft-deadline buffer matching
# CodeEvolver's, and the consecutive-empty-session halt threshold.
# KEEP IN SYNC with optimizer_loop.TIMEOUT_BUFFER_SECONDS, which carries the
# rationale for the value (it must cover the whole finalization pass).
TIMEOUT_BUFFER_SECONDS = 36000  # 10 hours
MAX_CONSECUTIVE_ARCHITECT_FAILURES = 5


def _utc_run_timestamp() -> str:
    """UTC stamp for the run branch name: YYYYMMDD-HHMMSS."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


class SimpleLoopStalledError(RuntimeError):
    """Raised when consecutive sessions neither evaluated nor completed."""


def initialize_repository(
    *,
    workspace: str,
    job: JobContext,
    run_timestamp: str,
    evaluator_config: EvaluatorConfig,
    datasets: DatasetConfig,
    validation_timeout: int,
) -> tuple[str, str]:
    """Create the run branch off `initial_branch`; return (branch, seed_sha)."""
    branch = f"codeevolver-{run_timestamp}-main"
    try:
        git_fetch(workspace)
    except RuntimeError as exc:
        logger.warning("simple loop seed: git fetch failed (continuing): %s", exc)
    try:
        git_checkout(workspace, job.initial_branch)
    except RuntimeError as exc:
        logger.warning(
            "simple loop seed: checkout of %s failed (continuing from HEAD): %s",
            job.initial_branch, exc,
        )
    git_create_branch_simple(workspace, branch)
    try:
        # Repo-local identity so the agent's own `git commit` works on a
        # fresh container (same identity the loop's helpers use).
        git_set_identity(workspace)
    except RuntimeError as exc:
        logger.warning(
            "simple loop seed: git identity setup failed (continuing): %s", exc,
        )
    validate_sandbox_environment(
        workspace=workspace,
        evaluator_config=evaluator_config,
        datasets=datasets,
        n_rows=5,
        timeout=validation_timeout,
    )
    return branch, git_get_sha(workspace)


def commit_session_end(workspace: str, iteration_idx: int) -> str:
    """Commit anything the agent left uncommitted; return the resulting HEAD."""
    try:
        git_add_and_commit(
            workspace, f"iter {iteration_idx}: session end (auto-committed)"
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "simple loop session %d: session-end commit failed (non-fatal): %s",
            iteration_idx, exc,
        )
    return git_get_sha(workspace)


def check_stalled(consecutive_empty: int, iteration_idx: int) -> None:
    """Halt the run once too many sessions in a row spent no rows or raised."""
    if consecutive_empty >= MAX_CONSECUTIVE_ARCHITECT_FAILURES:
        raise SimpleLoopStalledError(
            f"simple loop halting: {consecutive_empty} consecutive sessions "
            f"evaluated nothing or failed (last: session {iteration_idx})"
        )


def finalize_on_holdout(
    *,
    workspace: str,
    evaluator_config: EvaluatorConfig,
    datasets: DatasetConfig,
    branch: str,
    seed_sha: str,
    final_sha: str,
    timeout: int,
    test_samples: int = 1,
    seed_test_samples: int = 1,
) -> dict[str, Any]:
    """Score the seed and final programs on the test and comparison sets.

    The test set is drawn `seed_test_samples` times for the seed and
    `test_samples` times for the final program; every reading is
    returned (`seed_test_scores` / `final_test_scores`) and the `*_test_score`
    keys are their means. Outside the eval budget by construction: nothing
    here touches the state counter. The final program is only scored when it
    differs from the seed.
    """

    def _score(
        sha: str, iteration_idx: int, num_test_samples: int
    ) -> tuple[list[float], float | None]:
        clean_python_bytecode(workspace)
        git_checkout(workspace, sha, force=True)
        test_scores: list[float] = []
        comparison_score = None
        if datasets.has_testset:
            test_scores = sample_test_scores(
                workspace=workspace, config=evaluator_config, datasets=datasets,
                iteration_idx=iteration_idx, num_samples=num_test_samples,
                timeout=timeout, run_eval_fn=run_eval,
            )
        if datasets.comparison_ids:
            comparison_score = run_eval(
                workspace=workspace, config=evaluator_config,
                row_ids=datasets.comparison_ids, iteration_idx=iteration_idx,
                suffix="final_comparison", capture_traces=True, timeout=timeout,
            ).primary_score
        return test_scores, comparison_score

    scores: dict[str, Any] = {
        "seed_test_score": None, "final_test_score": None,
        "seed_train_score": None, "final_train_score": None,
        "seed_test_scores": [], "final_test_scores": [],
        "finalization_error": None,
    }
    if not (datasets.has_testset or datasets.comparison_ids):
        return scores
    try:
        scores["seed_test_scores"], scores["seed_train_score"] = _score(
            seed_sha, 0, seed_test_samples
        )
        if final_sha == seed_sha and test_samples <= seed_test_samples:
            scores["final_test_scores"] = list(scores["seed_test_scores"])
            scores["final_train_score"] = scores["seed_train_score"]
        else:
            scores["final_test_scores"], scores["final_train_score"] = _score(
                final_sha, 1, test_samples
            )
        scores["seed_test_score"] = mean_or_none(scores["seed_test_scores"])
        scores["final_test_score"] = mean_or_none(scores["final_test_scores"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple loop finalization: holdout eval failed (non-fatal): %s", exc)
        scores["finalization_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            git_checkout(workspace, branch, force=True)
        except Exception as exc:  # noqa: BLE001
            logger.warning("simple loop finalization: checkout of %s failed: %s", branch, exc)
    return scores


def run_simple_loop(
    *,
    config: OptimizationConfig,
    workspace: str,
    datasets: DatasetConfig,
    evaluator_config: EvaluatorConfig,
    callbacks: JobCallbacks,
    coding_harness_config: CodingHarnessConfig,
    agent_model: str | None,
) -> dict:
    """Run the whole simple loop; return the result dict."""
    max_eval_rows = config.max_metric_calls
    job = config.job
    rng = random.Random(config.seed)
    run_timestamp = _utc_run_timestamp()
    start_time = time.time()
    soft_deadline = (
        max(config.job_timeout_seconds - TIMEOUT_BUFFER_SECONDS, 0)
        if config.job_timeout_seconds is not None else None
    )
    state = SimpleLoopState(
        comparison_history_path=os.path.join(
            history_dir(), "comparison_set_history.jsonl"
        )
    )
    history = IterationHistory()
    constraint_lm = build_constraint_checker(
        additional_instructions=config.additional_instructions,
        reflection_lm=config.reflection_lm,
    )

    branch, seed_sha = initialize_repository(
        workspace=workspace, job=job, run_timestamp=run_timestamp,
        evaluator_config=evaluator_config, datasets=datasets,
        validation_timeout=config.comparison_eval_timeout,
    )
    _push(workspace, branch, job)
    history.create_entry(0).update(
        {"git_branch": branch, "close_sha": seed_sha, "skipped": False}
    )
    session_config = BaseAgentConfig(
        additional_instructions=config.additional_instructions,
        program_path=config.program,
        evaluator=evaluator_config,
        trainset_ids=datasets.trainset_ids,
        comparison_ids=datasets.comparison_ids,
        subsample_timeout=config.subsample_eval_timeout,
        comparison_timeout=config.comparison_eval_timeout,
        max_score=config.max_score,
        subsample_size=config.subsample_size,
        coding_harness=coding_harness_config.coding_harness,
        coding_harness_model=coding_harness_config.coding_harness_model,
        agent_managed_git=False,
        stage_project_skills=False,
        system_prompt_builder=build_simple_loop_prompt,
        agent_model=agent_model,
        capture_architect_traces=(
            config.enable_meta_reflection or config.enable_run_integrity_agent
        ),
        additional_deny_paths=list(config.additional_deny_paths or []),
    )

    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): same schedule as
    # every other arm — initial grant before iteration 1, +N per iteration.
    credits_enabled = (
        config.comparison_credits_per_iteration is not None
        and bool(datasets.comparison_ids)
    )
    if credits_enabled:
        state.grant_comparison_credits(config.comparison_credits_initial)

    iteration_idx = 0
    consecutive_empty = 0
    while not _budget_exhausted(state, max_eval_rows):
        if soft_deadline is not None and time.time() - start_time >= soft_deadline:
            logger.info("simple loop: wall-clock deadline reached — ending")
            break
        if callbacks.check_cancelled():
            logger.info("simple loop: job cancelled — ending")
            break
        iteration_idx += 1
        start_sha = git_get_sha(workspace)
        rows_before = state.total_num_evals
        if credits_enabled:
            state.grant_comparison_credits(config.comparison_credits_per_iteration)
        previous = history.get(iteration_idx - 1) or {}
        ctx = BaseIterationContext(
            iteration_idx=iteration_idx,
            parent_locked=True,
            subsample_size=config.subsample_size,
            comparison_set_size=len(datasets.comparison_ids),
            max_eval_rows=max_eval_rows,
            evals_used=rows_before,
            comparison_ack_code=f"{rng.randint(10, 99)}" if credits_enabled else "",
            previous_iteration_constraint_reverted=bool(
                previous.get("constraint_violation")
            ),
        )
        history.create_entry(iteration_idx)
        failed = False
        try:
            run_session(
                ctx=ctx, state=state, workspace=workspace,
                config=session_config, rng=rng,
            )
        except Exception as exc:  # noqa: BLE001
            failed = True
            history[iteration_idx]["subsample_error"] = f"{type(exc).__name__}: {exc}"
            logger.error("simple loop session %d failed: %s", iteration_idx, exc)
        end_sha = commit_session_end(workspace, iteration_idx)
        violation = None
        if end_sha != start_sha:
            violation = check_session_constraints(
                workspace=workspace, start_sha=start_sha, end_sha=end_sha,
                additional_instructions=config.additional_instructions,
                lm_callable=constraint_lm, iteration_idx=iteration_idx,
            )
            if violation is not None:
                # HEAD is the revert marker on top of `start_sha` (same
                # content); record it so `close_sha` is the next session's
                # `parent_sha`.
                end_sha = git_get_sha(workspace)
        rows_used = state.total_num_evals - rows_before
        history[iteration_idx].update(
            session_record(
                ctx, branch=branch, start_sha=start_sha, end_sha=end_sha,
                rows_used=rows_used, constraint_violation=violation, skipped=failed,
            )
        )
        if session_config.capture_architect_traces:
            _render_architect_markdown(
                iteration_idx, ctx, history[iteration_idx],
                harness=coding_harness_config.coding_harness,
            )
        _emit_progress(
            callbacks, state, history, iteration_idx, branch, end_sha, max_eval_rows,
        )
        _push(workspace, branch, job)
        consecutive_empty = consecutive_empty + 1 if (failed or rows_used == 0) else 0
        check_stalled(consecutive_empty, iteration_idx)

    final_sha = git_get_sha(workspace)
    scores = finalize_on_holdout(
        workspace=workspace, evaluator_config=evaluator_config, datasets=datasets,
        branch=branch, seed_sha=seed_sha, final_sha=final_sha,
        timeout=config.comparison_eval_timeout,
        test_samples=config.test_samples,
        seed_test_samples=config.seed_test_samples,
    )
    history[0]["test_score"] = scores["seed_test_score"]
    history[0]["test_scores"] = list(scores.get("seed_test_scores") or [])
    history[max(history)]["test_score"] = scores["final_test_score"]
    history[max(history)]["test_scores"] = list(scores.get("final_test_scores") or [])

    final_candidate = {"git_branch": branch, "git_sha": final_sha}
    result = {
        "success": True,
        "method": METHOD,
        "seed_branch": branch,
        "seed_sha": seed_sha,
        "iterations": len(history),
        "history": dict(history),
        "final_branch": branch,
        "final_sha": final_sha,
        "total_metric_calls": state.total_num_evals,
        **scores,
    }
    callbacks.set_completed(
        final_candidate=final_candidate,
        total_metric_calls=state.total_num_evals,
        num_branches=1,
        ce_state=_ce_state_payload(state, history, branch, final_sha),
        **scores,
    )
    _run_auditors(
        config=config, workspace=workspace, history=history, branch=branch,
        final_sha=final_sha, scores=scores, state=state, evaluator_config=evaluator_config,
        coding_harness_config=coding_harness_config, agent_model=agent_model,
        callbacks=callbacks, final_candidate=final_candidate,
    )
    result["history"] = dict(history)
    return result


def _run_auditors(
    *,
    config: OptimizationConfig,
    workspace: str,
    history: IterationHistory,
    branch: str,
    final_sha: str,
    scores: dict[str, float | None],
    state: SimpleLoopState,
    evaluator_config: EvaluatorConfig,
    coding_harness_config: CodingHarnessConfig,
    agent_model: str | None,
    callbacks: JobCallbacks,
    final_candidate: dict,
) -> None:
    """Post-run analysis agents (advisory, non-fatal, after set_completed)."""
    harness = dict(
        agent_model=agent_model,
        coding_harness=coding_harness_config.coding_harness,
        coding_harness_model=coding_harness_config.coding_harness_model,
    )
    if config.enable_meta_reflection:
        try:
            from src.engine.optimizer.meta.simple_meta_reflection import (
                run_simple_loop_meta_reflection,
            )

            run_simple_loop_meta_reflection(
                workspace=workspace, history=history, branch=branch,
                program_path=config.program,
                seed_test_score=scores["seed_test_score"],
                final_test_score=scores["final_test_score"],
                subsample_size=config.subsample_size,
                max_eval_rows=config.max_metric_calls,
                total_evals=state.total_num_evals,
                **harness,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("simple loop meta-reflection failed (non-fatal): %s", exc)
    if config.enable_run_integrity_agent:
        try:
            from src.engine.optimizer.meta.run_integrity import (
                run_integrity_inspection,
                stamp_integrity_verdict,
            )

            verdict = run_integrity_inspection(
                workspace=workspace, method=METHOD, num_iterations=len(history),
                best_valset_score=scores["final_train_score"],
                seed_test_score=scores["seed_test_score"],
                best_candidate_test_score=scores["final_test_score"],
                iteration_indices=[i for i in sorted(history) if i >= 1],
                testset_path=evaluator_config.testset_path,
                safe_mode=False,
                **harness,
            )
            if verdict is None:
                # Same loud failure as v3/finalization.run_post_run_integrity.
                logger.error(
                    "simple loop run-integrity: NO VERDICT -- the closing "
                    "integrity auditor produced no output (see the preceding "
                    "'meta-reflection: ... (method=run-integrity ...)' lines); "
                    "integrity_* stays None for this run"
                )
            elif stamp_integrity_verdict(history, verdict) is not None:
                callbacks.update_progress(
                    current_iteration=max(history),
                    final_candidate=final_candidate,
                    num_branches=1,
                    total_metric_calls=state.total_num_evals,
                    ce_state=_ce_state_payload(state, history, branch, final_sha),
                )
        except Exception as exc:  # noqa: BLE001
            logger.error("simple loop run-integrity failed (non-fatal): %s", exc)


def _budget_exhausted(state: SimpleLoopState, max_eval_rows: int | None) -> bool:
    return max_eval_rows is not None and state.total_num_evals >= max_eval_rows


def _ce_state_payload(
    state: SimpleLoopState, history: IterationHistory, branch: str, sha: str
) -> dict[str, Any]:
    """ce_state-shaped payload so the history mirror and UI keep working."""
    return {
        "program_candidates": [{"git_branch": branch, "git_sha": sha, "change_request": ""}],
        "candidate_scores": [
            history[i]["score"] for i in sorted(history) if history[i].get("score") is not None
        ],
        "parent_programs": [],
        "num_iterations": len(history),
        "total_evals": state.total_num_evals,
        "iteration_history": {str(k): v for k, v in history.items()},
    }



def _render_architect_markdown(
    iteration_idx: int, ctx: Any, record: Mapping[str, Any], **extra: Any
) -> None:
    """Best-effort `digest.md` / `transcript.md` for the session's architect
    trace; the header takes the session record's close facts. Never raises."""
    try:
        from src.engine.optimizer.v3.architect_trace_markdown import (
            build_trace_header,
            render_architect_trace_markdown,
        )

        header = build_trace_header(
            ctx,
            branch=record.get("git_branch"),
            close_sha=record.get("close_sha"),
            changes_landed=record.get("changes_landed"),
            best_train_score=record.get("best_subsample_score"),
            comparison_best_score=record.get("comparison_best_score"),
            skipped=record.get("skipped"),
            error=record.get("subsample_error"),
            constraint_violation=record.get("constraint_violation"),
            **extra,
        )
        render_architect_trace_markdown(iteration_idx, header)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "simple loop session %d: architect trace markdown render failed "
            "(non-fatal): %s", iteration_idx, exc,
        )

def _emit_progress(
    callbacks: JobCallbacks,
    state: SimpleLoopState,
    history: IterationHistory,
    iteration_idx: int,
    branch: str,
    sha: str,
    max_eval_rows: int | None = None,
) -> None:
    logger.info(
        "simple loop session %d: evals %d/%s rows (this session: %s), score=%s",
        iteration_idx, state.total_num_evals,
        max_eval_rows if max_eval_rows is not None else "unbounded",
        history[iteration_idx].get("eval_rows_used"), history[iteration_idx].get("score"),
    )
    callbacks.update_progress(
        current_iteration=iteration_idx,
        final_candidate={"git_branch": branch, "git_sha": sha},
        num_branches=1,
        total_metric_calls=state.total_num_evals,
        ce_state=_ce_state_payload(state, history, branch, sha),
        memory_md=None,
    )


def _push(workspace: str, branch: str, job: JobContext) -> None:
    """Push the run branch; non-fatal, no-op in local mode."""
    if not job.repo_url:
        return
    try:
        result = git_push_authenticated(
            workspace=workspace, branch=branch, **job.git_creds(), set_upstream=True
        )
        if not result.get("success"):
            logger.warning("simple loop: push of %s failed: %s", branch, result.get("stderr"))
    except Exception as exc:  # noqa: BLE001
        logger.warning("simple loop: push of %s raised (non-fatal): %s", branch, exc)


__all__ = [
    "SimpleLoopStalledError",
    "check_stalled",
    "commit_session_end",
    "finalize_on_holdout",
    "initialize_repository",
    "run_simple_loop",
]
