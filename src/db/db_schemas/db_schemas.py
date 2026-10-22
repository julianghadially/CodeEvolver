from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any, Literal
from src.shared.enums import ProgramStatus, JobStatus

class IterationHistoryRecord(BaseModel):
    """Schema for a single iteration's tracking data.

    Keyed by iteration number in the iteration_history dict.
    Tracks ALL iterations including rejected candidates.
    """

    git_branch: str | None = None               # Branch the iteration's work was registered on
    selected_component: str | None = None       # Component mutated (e.g., "_code", "predict")
    change_request: str | None = None           # Description of the mutation
    architecture_title: str | None = None       # Architecture title from codeevolver.md
    architecture_summary: str | None = None     # Architecture summary from codeevolver.md
    change_summary: str | None = None           # Concise core-change summary for this iteration (analysis agent; None if skipped)
    score: float | None = None                  # The iteration's progress reading: best train eval on the baseline's subsample (== best_subsample_score). NOT a score of the closing commit
    test_score: float | None = None              # Held-out test score (mean over test_samples draws); set only on the seed and final-primary records at finalization
    test_scores: list[float] | None = None       # Every held-out draw behind test_score (same records)
    parent_subsample_score: float | None = None # Parent's (baseline) subsample score
    subsample_error: str | None = None          # Error string if evaluation failed
    skipped: bool | None = None                  # True ONLY when the iteration produced nothing (architect raised, or no branch registered). A tested-and-reverted iteration is NOT skipped — see changes_landed
    changes_landed: bool | None = None           # Did the PROGRAM change? close_tree differs from parent_tree by a program file (see v3/program_paths.py)
    best_subsample_score: float | None = None    # Best train eval of the iteration on the baseline's subsample, regardless of what landed
    best_subsample_sha: str | None = None        # Checkout-able pointer at that content: its commit, or refs/codeevolver/evals/<N>/<i> when it was never committed
    comparison_best_score: float | None = None   # Best comparison-set eval on the registered branch this iteration
    comparison_best_sha: str | None = None       # Checkout-able pointer at that content
    comparison_state: str | None = None          # pareto_evolution: how the CLOSING content's comparison fitness was obtained — "EXACT" | "PROGRAM_EQUIVALENT" | "REVERTED_TO_COMPARISON_POINT" | "RERUN_BY_LOOP"; None on EE
    comparison_eval_id_at_close: str | None = None  # pareto_evolution: eval_id of that comparison eval
    comparison_score_at_close: float | None = None  # pareto_evolution: its score (== comparison_set_score)
    comparison_credits_after: int | None = None  # Comparison-credit ledger balance when the record was written (specs/archive-public/comparison_credits.md); None when the ledger is off
    variance_estimate: dict[str, Any] | None = None  # The run's one run_variance_estimate result (specs/archive-public/sampling_variance.md) — scores per run, within_item_variance, run_to_run_std, sha/tree measured; non-null only on the iteration that took it
    baseline_ran_on_modified_tree: bool | None = None  # The iteration's first training eval ran on an already-modified working tree, so the baseline measures the parent PLUS those edits. Permitted (agent-directed search may need to repair a branch first) — recorded so the delta can be read correctly
    initial_subsample_score: float | None = None # Subsample score before the iteration's changes
    metrics_by_name: dict[str, float] | None = None  # Full multi-metric scores
    constraint_violation: str | None = None      # Violated constraint text when the post-change diff failed the constraint gate (force-reverted)
    eval_rows_used: int | None = None            # Eval rows the iteration's agent-initiated run_eval calls spent (simple_loop stamps it; None where not tracked)
    watchdog_fired: bool | None = None           # A closing-protocol watchdog re-prompt fired

    # Engine 3 closing metadata (deterministic close + close_iteration tool)
    close_sha: str | None = None                 # Registered-branch HEAD at close, including the loop's own post-close codeevolver.md commit. Anchored at refs/codeevolver/iterations/<N> so it stays checkout-able
    close_tree: str | None = None                # Tree of close_sha — the content identity of the end state
    parent_branch: str | None = None             # Derived parent branch
    parent_sha: str | None = None                # Derived parent sha
    parent_iteration: int | None = None          # Iteration the parent came from (0 = seed)
    parent_is_primary: bool | None = None        # Was the parent branch the primary at iteration start?
    parent_derivation: str | None = None         # "ancestor" | "merge_base" | "seed"
    new_lineage: bool | None = None              # Registered branch was new to the run
    improvement_mode: str | None = None          # Architect-claimed: "exploration" | "exploitation"
    memory_updated: bool | None = None           # Architect-claimed memory.md update
    primary_branch_at_start: str | None = None
    primary_branch_after: str | None = None
    primary_fallback: bool | None = None         # Close left no primary; prior primary kept
    promotion_evidence: str | None = None        # Evidence supplied when promoting a new primary
    closed_by_fallback: bool | None = None       # Loop fallback-closed (agent never called close_iteration)
    comparison_scores: list[dict[str, Any]] | None = None  # Comparison-set evals run this iteration
    was_primary: bool | None = None              # The iteration's parent was the primary at iteration start
    is_primary: bool | None = None               # The iteration's branch is the primary after the transition

    # pareto_evolution (loop-owned population) — None on EE records
    candidate_admitted: bool | None = None       # Closing content joined the population (it has a comparison-set eval)
    admission_reason: str | None = None          # "agent_comparison" | "reverted_to_comparison_point" | "loop_comparison" | "no_comparison" | "no_change" | "constraint_violation" | "skipped"
    selected_parent_iteration: int | None = None # Population candidate (iteration) the loop forked this iteration from
    selection_policy: str | None = None          # Loop's parent draw: "best" (exploit-and-merge) | "pareto_sample"
    comparison_set_score: float | None = None    # Aggregate comparison-set score of the closing content
    comparison_score: float | None = None        # Seed record only: the seed's comparison-set score

    # asa (specs/archive-public/v3_asa.md) — None on other arms' records
    asa_requirements: list[str] | None = None    # Requirements fired at iteration start: subset of ["new_lineage", "eligible_nonprimary"]
    asa_satisfied: bool | None = None            # Every fired requirement was met by the derived closing facts
    asa_violation: bool | None = None            # A requirement fired and was not met (charged against the allowance)
    asa_eligible_at_start: list[str] | None = None  # Eligible non-primary lineages at iteration start
    asa_parent_is_eligible: bool | None = None   # The derived parent branch was eligible at iteration start
    asa_parent_positive_contribution_to_primary_normalized: float | None = None  # Parent lineage's normalized positive contribution vs the primary; None when the parent was the primary
    asa_allowance_left: int | None = None        # Violation allowance remaining at iteration start (ceil(closed/10) - violations)
    asa_free_tail: bool | None = None            # Iteration ran in the free tail (no rules)
    asa_phase1: bool | None = None               # Phase-1 window was active (rule 1 live)
    asa_loop_comparison_reason: str | None = None  # Loop ran a counted comparison eval at close: "promotion" | "stale_primary" | "skipped_agent_ran_comparison" (loop run skipped — the agent already ran one this iteration)

    # Run-integrity inspector verdict (stamped on the latest record at close)
    integrity_cheating_category: str | None = None   # "no_cheating" | "non_impactful_cheating" | "impactful_cheating" | "unparsed"
    integrity_cheating_rationale: str | None = None  # Evidence paragraph from the inspector
    integrity_functioning_iterations: str | None = None  # "X/N"
    integrity_pct: float | None = None               # % of functioning iterations (0-100)
    integrity_budget_bypass: bool | None = None      # Training/comparison rows were run outside run_eval (flag only; all arms)
    integrity_budget_bypass_estimated_rows: int | None = None  # Inspector's estimate of rows run outside run_eval
    integrity_budget_bypass_rationale: str | None = None  # Evidence paragraph
    # Clean-slate inspector verdict (stamped on the seed record)
    clean_slate_status: str | None = None            # "clean" | "uncertain" | "unparsed" | "error" ("contaminated" aborts the run before stamping)


class ProgramRecord(BaseModel):
    """Database record for a program version."""

    repo_id: str
    program_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProgramStatus = ProgramStatus.PENDING


class RepoRecord(BaseModel):
    """Database record for a connected repository."""

    repo_id: str
    clerk_id: str | None = None  # May be None if repo connected before Clerk auth (two-stage flow)
    repo_url: str
    workspace_path: str
    installation_id: int | None = None  # GitHub App installation ID
    github_owner: str | None = None  # e.g., "octocat"
    github_repo_name: str | None = None  # e.g., "hello-world"
    github_repo_id: int | None = None  # GitHub's numeric repo ID (natural key)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: Literal["connected", "not_connected"] = "connected"

class JobRecord(BaseModel):
    """MongoDB document for an optimization job."""

    job_id: str
    clerk_id: str  # Clerk user ID of the job creator
    repo_id: str
    repo_url: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    current_iteration: int | None = None
    total_metric_calls: int | None = None
    num_branches: int | None = None
    # Final program (engine 3): the last-declared primary branch + its sha.
    # dict[str, Any], not dict[str, str]: `final_candidate` (v3/run_reporting.py)
    # carries `is_primary: True`, a bool. Typing the values as str made
    # GET /job/{id} raise a 500 the moment a candidate was registered --
    # which is the endpoint runner/entrypoint.py polls for terminal status,
    # so a run could never be seen to finish and never exported artifacts.
    final_candidate: dict[str, Any] | None = None
    # Comparison-set (trainset subsample) scores, measured once at
    # finalization on the seed and the final primary — metrics only, never
    # budget-counted.
    seed_train_score: float | None = None
    final_train_score: float | None = None
    # Held-out test-set scores, set once at finalization — the only numbers
    # measured on the locked test set (the overfitting check).
    seed_test_score: float | None = None
    final_test_score: float | None = None
    # Every test-set draw behind the two means above (`test_samples` each).
    seed_test_scores: list[float] | None = None
    final_test_scores: list[float] | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # CE State (stored on completion). Default is an empty dict rather than
    # None so the progress handler can safely $set dotted paths
    # (`ce_state.iteration_history`) — MongoDB rejects dotted $set into a
    # null parent with "cannot create field ... in element {ce_state: null}".
    ce_state: dict[str, Any] = Field(default_factory=dict)  # Full CEState serialized
    # Engine 3 branch registry: branch -> {last_sha, is_primary, ...}.
    branches: dict[str, Any] | None = None
    primary_branch: str | None = None

    # Per-iteration history: dict keyed by candidate index (string for MongoDB)
    ce_state_history: dict[str, Any] = Field(default_factory=dict)

    # Latest snapshot of the cross-iteration `/memory/memory.md` file from the
    # sandbox. Overwritten each iteration via the progress callback so the API
    # always holds the freshest rolled-up architect memory. None until the first
    # progress update lands.
    memory_md: str | None = None

    # Optional display name (e.g. a research run id); the portal shows it in
    # place of `job_id` when set. Currently only set via /import-job-history.
    name: str | None = None
    # Post-run reports, set via /import-job-history: the meta-reflection
    # agent's `meta_reflection.md` and the run-integrity agent's
    # `run_integrity.md`.
    meta_reflection_md: str | None = None
    integrity_audit_md: str | None = None
