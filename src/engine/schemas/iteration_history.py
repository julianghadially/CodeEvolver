"""Copyright © 2026 440 Labs LLC

Per-iteration history record schema — arm-agnostic, no I/O.

`IterationHistory` is a dict subclass mapping iteration number to a plain dict
record whose keys are `ITERATION_RECORD_FIELDS`. It tracks ALL iterations —
including rejected and skipped ones — unlike a candidate pool, which only holds
admitted candidates. Every optimizer arm (EE, freeform, pareto_evolution, the
simple loop) uses it.

The optimizer uses plain dicts with known keys (no Pydantic, no DB imports).
The API / orchestrator layer maps these to the canonical `IterationHistoryRecord`
Pydantic model at the DB boundary (`src/db/db_schemas/db_schemas.py`); this
list must stay in sync with that model.
"""

from __future__ import annotations

# Fields that each iteration record can contain. Matches the canonical
# IterationHistoryRecord schema in src/db/db_schemas/db_schemas.py.
ITERATION_RECORD_FIELDS: list[str] = [
    "git_branch",
    "selected_component",
    "change_request",
    "architecture_title",
    "architecture_summary",
    "change_summary",
    # `score` is the iteration's progress reading: the best train eval on the
    # baseline's subsample (== best_subsample_score). It is NOT a score of the
    # closing commit — see records.CloseRecord.
    "score",
    "test_score",
    "test_scores",
    "parent_subsample_score",
    "subsample_error",
    "skipped",
    "changes_landed",
    "best_subsample_score",
    "best_subsample_sha",
    "comparison_best_score",
    "comparison_best_sha",
    # pareto_evolution: how the closing content's comparison fitness was
    # obtained (None on EE records)
    "comparison_state",
    "comparison_eval_id_at_close",
    "comparison_score_at_close",
    "baseline_ran_on_modified_tree",
    "initial_subsample_score",
    "metrics_by_name",
    "watchdog_fired",
    "constraint_violation",
    "eval_rows_used",
    # Comparison-credit ledger balance at close (specs/archive-public/comparison_credits.md)
    "comparison_credits_after",
    # run_variance_estimate result (specs/archive-public/sampling_variance.md): non-null only
    # on the iteration that took the run's one estimate
    "variance_estimate",
    # Engine 3 closing metadata
    "close_sha",
    "close_tree",
    "parent_branch",
    "parent_sha",
    "parent_iteration",
    "parent_is_primary",
    "parent_derivation",
    "new_lineage",
    "improvement_mode",
    "memory_updated",
    "primary_branch_at_start",
    "primary_branch_after",
    "primary_fallback",
    "promotion_evidence",
    "closed_by_fallback",
    "comparison_scores",
    "was_primary",
    "is_primary",
    # pareto_evolution (loop-owned population); None on EE records
    "candidate_admitted",
    "admission_reason",
    "selected_parent_iteration",
    "selection_policy",
    "comparison_set_score",
    "comparison_score",
    # asa (specs/archive-public/v3_asa.md); None on other arms' records
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
    # Run-integrity inspector verdict (closing-loop invocation stamps these on
    # the LATEST record; see run_integrity.INTEGRITY_RECORD_KEYS)
    "integrity_cheating_category",
    "integrity_cheating_rationale",
    "integrity_functioning_iterations",
    "integrity_pct",
    "integrity_budget_bypass",
    "integrity_budget_bypass_estimated_rows",
    "integrity_budget_bypass_rationale",
    # Clean-slate inspector verdict (seed step stamps this on record 0)
    "clean_slate_status",
]


class IterationHistory(dict):
    """Dict subclass: iteration number -> dict record."""

    def create_entry(self, iteration: int) -> dict:
        """Create a new skeleton entry with every field set to None.

        Fields are filled in over the life of an iteration:
        - `on_iteration_start`                → entry created
        - `coding_agent` tool                 → change_request, selected_component
        - evaluator on child                  → score / best_subsample_score, metrics_by_name
        - deterministic close                 → closing metadata
        """
        record: dict = {field: None for field in ITERATION_RECORD_FIELDS}
        self[iteration] = record
        return record


__all__ = ["IterationHistory", "ITERATION_RECORD_FIELDS"]
