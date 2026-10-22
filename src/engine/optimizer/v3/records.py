"""Copyright © 2026 440 Labs LLC

Typed record shapes for the engine 3 run state.

These are the data concepts that flow between the close machinery
(`closing.py`), the run state (`ce_state.py`), the iteration context, and the
loop. They are `TypedDict`s on purpose — at runtime they are plain dicts, so
JSON persistence (`CEState.save/load`), the comparison-history JSONL, and
partial stub records (a skipped/errored iteration records only a few keys)
all keep working with zero conversion — while readers and type checkers get
the schema in one place.

The DB-boundary schema (`IterationHistoryRecord` in `src/db/db_schemas/`)
stays Pydantic; `CloseRecord` overlaps it but is the engine-internal shape
(the loop maps close records onto history records field-by-field).
"""

from __future__ import annotations

from typing import NotRequired, TypedDict

from src.engine.schemas.iteration_context import EvalRecord


class RegisteredBranch(TypedDict, total=False):
    """A `close_iteration` registration — the branch the iteration submitted.

    Produced (normalized) by `agent_tools.close_iteration_tool`; stored on
    `IterationContext.registered_branch`. Exactly one branch per iteration,
    which is what makes duplicate primaries structurally impossible.
    """

    branch: str
    was_primary: bool
    is_primary: bool
    promotion_evidence: str | None


class ComparisonEvalEntry(TypedDict):
    """One comparison-set eval.

    Appended (with a `ts` field) as a JSONL line to
    `{history_dir}/comparison_set_history.jsonl` at eval time, and embedded
    (without `ts`) in `CloseRecord.comparison_scores`.
    """

    branch: str | None
    sha: str | None
    score: float | None
    n_rows: int | None
    ts: NotRequired[float]
    eval_id: NotRequired[str]
    scores_by_metric: NotRequired[dict[str, float]]
    # Per-row scores (JSONL line only; see EvalRecord.row_scores).
    row_scores: NotRequired[dict[str, float]]
    # Tree signature of the scored content (JSONL line only).
    tree: NotRequired[str]


class BranchEntry(TypedDict):
    """One branch in `CEState.branches` (the run's branch registry)."""

    last_sha: str | None
    is_primary: bool
    created_iteration: int | None
    last_iteration: int | None


class CloseRecord(TypedDict, total=False):
    """One iteration's closing record — `CEState.iterations` is a list of
    these.

    Built by `closing.build_close_record` after the deterministic close
    (fallback close → parent derivation → finalize → constraint gate) and
    applied by `CEState.record_iteration`, which upserts the branch registry
    and executes the `primary_branch_after` transition.

    `total=False` because two degenerate shapes are also recorded: an
    architect-error stub ({iteration, skipped, error}) and constraint-gate
    overrides. `iteration` is always present.
    """

    iteration: int
    # True only when the iteration produced NOTHING: the architect raised, or
    # no branch was ever registered. An iteration that tested a change,
    # measured a regression and reverted is NOT skipped — it closed, it cost
    # budget, it is a valid parent, and it earns a change summary. Use
    # `changes_landed` to ask whether the program moved.
    skipped: bool
    # Did the PROGRAM change? close_tree differs from parent_tree by at least
    # one program file (see program_paths.py).
    changes_landed: bool
    error: str

    # What the architect registered / where it left the branch.
    branch: str | None
    # Registered-branch HEAD at close, INCLUDING any commit the loop makes
    # afterwards (the codeevolver.md change-summary append). This is the
    # iteration's end state: what the next iteration physically inherits, what
    # `refs/codeevolver/iterations/<N>` anchors, and what parent derivation
    # keys on.
    close_sha: str | None
    close_tree: str | None
    # Every codeevolver-* branch in the workspace at close, {branch: tip_sha} —
    # including experiment branches the architect created and abandoned, which
    # `registered_branch` never sees.
    branches_seen: dict[str, str]
    registered_branch: RegisteredBranch | None

    # Deterministic parent derivation (closing.derive_iteration_parent).
    parent_sha: str | None
    parent_tree: str | None
    parent_branch: str | None
    parent_iteration: int | None     # 0 = seed
    parent_is_primary: bool | None
    # "ancestor" | "merge_base" | "seed" (agent-managed git, derived post-
    # session) | "preselected" (loop-owned parent selection: pareto_evolution
    # stamps the parent at iteration start and skips derivation).
    parent_derivation: str | None
    new_lineage: bool | None

    # Scores. There is deliberately NO "score of the closing content" on the
    # record: the architect may close on content it never re-measured, and
    # never-committed evaluated states cannot be ordered against the closing
    # commit (see `IterationContext.finalize`). What is recorded:
    #   subsample_baseline — the iteration's first train eval (locked rows)
    #   subsample_best     — the best train eval on those SAME rows, regardless
    #                        of what landed; the iteration's progress reading.
    #                        `subsample_best_sha` is a checkout-able pointer at
    #                        that content (commit, or refs/codeevolver/evals
    #                        anchor when it was never committed).
    #   comparison_best    — best comparison-set eval on the registered branch
    #                        this iteration (+ `comparison_best_sha`).
    # pareto_evolution additionally records how the CLOSING content's
    # comparison fitness was obtained — `comparison_state` ("EXACT" |
    # "PROGRAM_EQUIVALENT" | "REVERTED_TO_COMPARISON_POINT" |
    # "RERUN_BY_LOOP"), `comparison_eval_id_at_close`,
    # `comparison_score_at_close` (== `comparison_set_score`). None on EE.
    subsample_baseline: float | None
    subsample_best: float | None
    subsample_best_sha: str | None
    comparison_best: float | None
    comparison_best_sha: str | None
    comparison_state: str | None
    comparison_eval_id_at_close: str | None
    comparison_score_at_close: float | None
    # Comparison-credit ledger balance when the iteration's record was
    # written (specs/archive-public/comparison_credits.md). None when the ledger is off.
    comparison_credits_after: int | None
    # True when the iteration's first training eval ran on an already-modified
    # working tree, so `subsample_baseline` measures the parent PLUS those
    # edits. Not a violation — permitted by design — but it changes how the
    # iteration's delta should be read.
    baseline_ran_on_modified_tree: bool | None
    baseline_dirty_paths: list[str]
    comparison_scores: list[ComparisonEvalEntry]

    # Architect claims (close_iteration tool).
    claimed_improvement_mode: str | None  # "exploration" | "exploitation" | None
    memory_updated: bool

    # Primary transition applied by CEState.record_iteration.
    primary_branch_at_start: str | None
    primary_branch_after: str | None
    primary_fallback: bool
    promotion_evidence: str | None
    # Was the iteration's parent the primary at iteration start, and is the
    # iteration's branch the primary after the transition? EE: from the
    # registered entry / `primary_branch_after`; pareto_evolution: stamped by
    # the loop (the agent never sets primary there).
    was_primary: bool | None
    is_primary: bool | None

    # The run's `run_variance_estimate` result (specs/archive-public/sampling_variance.md),
    # present only on the iteration that took it; None elsewhere. The
    # run-level copy lives on `CEState.variance_estimate`.
    variance_estimate: dict | None

    # Protocol outcomes.
    closed_by_fallback: bool
    watchdog_fired: bool
    constraint_violation: str | None

    # pareto_evolution (loop-owned population) — absent on EE records.
    # Admission: the closing content HAS a comparison-set eval.
    # Train regression is not a barrier.
    candidate_admitted: bool
    # Admitted: "agent_comparison" (EXACT / PROGRAM_EQUIVALENT) |
    # "reverted_to_comparison_point" | "loop_comparison" (RERUN_BY_LOOP —
    # agent ran none, train strictly progressed). Not admitted:
    # "no_comparison" | "no_change" | "constraint_violation" | "skipped"
    admission_reason: str | None
    # The population candidate (iteration number) the loop forked this
    # iteration from.
    selected_parent_iteration: int | None
    # How the loop chose the parent: "best" (exploit-and-merge draw fired;
    # merge note shown) | "pareto_sample". The LOOP's draw — distinct from the
    # architect's own `claimed_improvement_mode`.
    selection_policy: str | None
    # Aggregate comparison-set score attributed to the closing content.
    comparison_set_score: float | None

    # asa (specs/archive-public/v3_asa.md) — absent on other arms' records. Stamped by
    # `asa_on_iteration_closed`; every value is loop-derived.
    # Requirements fired at iteration start: subset of
    # ["new_lineage", "eligible_nonprimary"].
    asa_requirements: list[str]
    # Every fired requirement was met by the derived closing facts.
    asa_satisfied: bool
    # A requirement fired and was not met — the durable count the violation
    # allowance is charged against.
    asa_violation: bool
    # Eligible non-primary lineages at iteration start (frozen).
    asa_eligible_at_start: list[str]
    # The derived parent branch was in `asa_eligible_at_start`.
    asa_parent_is_eligible: bool
    # The parent lineage's normalized positive behavioral contribution vs the
    # primary at iteration start; None when the parent was the primary.
    asa_parent_positive_contribution_to_primary_normalized: float | None
    asa_allowance_left: int
    asa_free_tail: bool
    asa_phase1: bool
    # Why the loop ran a counted comparison eval at close:
    # "promotion" | "stale_primary" | None.
    asa_loop_comparison_reason: str | None


__all__ = [
    "BranchEntry",
    "CloseRecord",
    "ComparisonEvalEntry",
    "EvalRecord",
    "RegisteredBranch",
]
