"""Copyright © 2026 440 Labs LLC

Run reporting for the engine 3 loop: the progress / completion payload the
API expects, the per-iteration stamping of a close record onto the
`IterationHistory` entry, and the end-of-iteration progress callback.

Nothing here touches git or the eval budget; it only reads `CEState` /
`IterationContext` and writes the history entry + callbacks.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.engine.schemas.iteration_history import IterationHistory
from src.engine.services.subprocess_service import read_file

from .ce_state import CEState
from .change_summary import read_change_summary, write_combined_change_summaries
from .codeevolver_md import (
    extract_architecture_summary,
    extract_architecture_title,
)
from .iteration_architect import memory_md_path
from .iteration_context import IterationContext
from .records import CloseRecord
from .run_metrics import derive_run_metrics

logger = logging.getLogger(__name__)


# Close-record keys copied verbatim onto the per-iteration history entry when
# present. Strategy-specific keys (pareto_evolution admission / comparison
# state, asa requirements) are absent on other arms' records.
CLOSE_RECORD_HISTORY_KEYS: tuple[str, ...] = (
    "was_primary",
    "is_primary",
    "candidate_admitted",
    "admission_reason",
    "selected_parent_iteration",
    "selection_policy",
    "comparison_set_score",
    "comparison_state",
    "comparison_eval_id_at_close",
    "comparison_score_at_close",
    "comparison_credits_after",
    # run_variance_estimate result (specs/archive-public/sampling_variance.md); non-null only
    # on the iteration that took it
    "variance_estimate",
    # asa (specs/archive-public/v3_asa.md); absent on other arms' records
    "asa_requirements",
    "asa_satisfied",
    "asa_violation",
    "asa_eligible_at_start",
    "asa_parent_is_eligible",
    "asa_parent_positive_contribution_to_primary_normalized",
    "asa_allowance_left",
    "asa_free_tail",
    "asa_phase1",
    "asa_loop_comparison_reason",
)


def build_ce_state_payload(
    state: CEState,
    history: IterationHistory,
    num_iterations: int,
) -> dict[str, Any]:
    """Shape the API expects for progress/completion updates (engine 3).

    - `branches` — the run's branch registry (branch -> last_sha/is_primary/…).
    - `primary_branch` — the current primary (close_iteration's source of truth).
    - `num_iterations` / `total_evals` — top-line counters for the UI.
    - `iteration_history` — full per-iteration record dict, merged on the
      API side via dotted MongoDB keys so out-of-order updates are safe.
    - `run_metrics` — derived exploration metrics (spec §17).
    """
    return {
        "branches": dict(state.branches),
        "primary_branch": state.primary_branch,
        "num_iterations": num_iterations,
        "total_evals": int(state.total_num_evals or 0),
        "iteration_history": {str(k): v for k, v in history.items()},
        "run_metrics": derive_run_metrics(state.iterations),
    }


def final_candidate(state: CEState) -> dict[str, Any]:
    """The run's final program: the last-declared primary branch + its sha."""
    primary = state.primary_branch
    entry = state.branches.get(primary) or {}
    return {
        "git_branch": primary,
        "git_sha": entry.get("last_sha"),
        "is_primary": True,
    }


def stamp_close_record_onto_history(
    entry: dict[str, Any],
    ctx: IterationContext,
    close_record: CloseRecord,
) -> None:
    """Copy the iteration's closing facts onto its `IterationHistory` entry.

    `score` is the iteration's progress reading (best train subsample tried)
    — NOT a score of the closing commit (see `records.CloseRecord`).
    """
    entry.update(ctx.history_fields())
    entry["close_sha"] = ctx.close_sha
    entry["skipped"] = ctx.skipped
    entry["score"] = ctx.best_subsample_score
    entry["primary_branch_after"] = close_record["primary_branch_after"]
    entry["primary_fallback"] = close_record.get("primary_fallback")
    entry["comparison_scores"] = close_record.get("comparison_scores")
    for key in CLOSE_RECORD_HISTORY_KEYS:
        if key in close_record:
            entry[key] = close_record[key]


def log_iteration_close(ctx: IterationContext, close_record: CloseRecord) -> None:
    """One INFO line summarising how the iteration closed."""
    iteration_idx = ctx.iteration_idx
    if not ctx.changes_landed:
        logger.info(
            "iteration %d: closed with no program change (branch=%s, "
            "close_sha=%s, baseline %s, best tried %s, "
            "coding_agent_called=%s, closed_by_fallback=%s)",
            iteration_idx,
            ctx.child_branch,
            (ctx.close_sha or "?")[:8],
            ctx.initial_subsample_score,
            ctx.best_subsample_score,
            ctx.coding_agent_called,
            ctx.closed_by_fallback,
        )
    else:
        logger.info(
            "iteration %d: closed (branch=%s, close_sha=%s, "
            "subsample baseline %s -> best %s [%s], comparison best %s, "
            "mode=%s, primary=%s)",
            iteration_idx,
            ctx.child_branch,
            (ctx.close_sha or "?")[:8],
            ctx.initial_subsample_score,
            ctx.best_subsample_score,
            (ctx.best_subsample_sha or "?")[:12],
            ctx.comparison_best_score,
            ctx.claimed_improvement_mode,
            close_record["primary_branch_after"],
        )


def stamp_and_report_iteration(
    iteration_idx: int,
    hist: IterationHistory,
    *,
    state: CEState,
    workspace: str,
    callbacks: Any,
    run_start_time: float,
    soft_deadline_seconds: int | None,
    max_eval_rows: int | None,
) -> None:
    """End-of-iteration bookkeeping + progress callback.

    Stamps the architecture title/summary (from codeevolver.md as it stands
    AFTER the iteration) and the analysis agent's change summary onto the
    record, regenerates the combined change-summaries file, logs the budget
    position, and posts progress to the API. Fail-open on the summary steps.
    """
    # Read codeevolver.md at end of iteration so the per-iteration record
    # reflects the architecture *after* the iteration's change.
    codeevolver_md_body = read_file(os.path.join(workspace, "codeevolver.md"))
    if codeevolver_md_body:
        arch_title = extract_architecture_title(codeevolver_md_body)
        arch_summary = extract_architecture_summary(codeevolver_md_body)
        if arch_title is not None:
            hist[iteration_idx]["architecture_title"] = arch_title
        if arch_summary is not None:
            hist[iteration_idx]["architecture_summary"] = arch_summary

    # Stamp this iteration's own concise change summary (written by the
    # analysis agent, which has already run by this point). Skipped
    # iterations have no summary file → left None. Then regenerate the
    # combined change-summaries file. Fail-open.
    try:
        this_summary = read_change_summary(iteration_idx)
        if this_summary is not None:
            hist[iteration_idx]["change_summary"] = this_summary
        write_combined_change_summaries(hist)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: change-summary update failed (non-fatal): %s",
            iteration_idx, exc,
        )

    ce_state_payload = build_ce_state_payload(
        state, hist, num_iterations=iteration_idx + 1,
    )
    total_evals = int(state.total_num_evals or 0)
    elapsed = time.time() - run_start_time
    if max_eval_rows is not None:
        eval_budget_str = f"{total_evals}/{max_eval_rows} rows ({total_evals / max_eval_rows:.1%})"
    else:
        eval_budget_str = f"{total_evals} rows (no max)"
    if soft_deadline_seconds is not None and soft_deadline_seconds > 0:
        time_budget_str = (
            f"{elapsed:.0f}s/{soft_deadline_seconds}s "
            f"({elapsed / soft_deadline_seconds:.1%})"
        )
    else:
        time_budget_str = f"{elapsed:.0f}s (no soft timeout)"
    logger.info(
        "iteration %d: progress — evals %s, time %s, branches=%d, primary=%s",
        iteration_idx,
        eval_budget_str,
        time_budget_str,
        len(state.branches),
        state.primary_branch,
    )
    memory_md_snapshot = read_file(memory_md_path())
    callbacks.update_progress(
        current_iteration=iteration_idx,
        final_candidate=final_candidate(state),
        num_branches=len(state.branches),
        total_metric_calls=total_evals,
        ce_state=ce_state_payload,
        memory_md=memory_md_snapshot,
    )


__all__ = [
    "CLOSE_RECORD_HISTORY_KEYS",
    "build_ce_state_payload",
    "final_candidate",
    "log_iteration_close",
    "stamp_and_report_iteration",
    "stamp_close_record_onto_history",
]
