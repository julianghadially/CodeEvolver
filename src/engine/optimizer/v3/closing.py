"""Copyright © 2026 440 Labs LLC

Deterministic iteration-closing logic for engine 3 (agent-managed git).

The architect owns branch choice during its session; after the session the
loop needs objective facts about what happened. This module computes them
from git state + the run's past closing records, with the side-effecting git
predicates injected as callables so every function here is pure and testable
with dict-based fakes (the same pattern as `IterationContext.finalize`).

- `derive_iteration_parent`: which past iteration is this iteration's parent?
  Not "where did the branch stem from of all time" — just the most recent
  past final sha found in the registered branch's current history. Rebases or
  deleted branches fall back to the nearest common ancestor; if there is no
  common ancestor at all, the seed is the parent.
- `resolve_primary`: apply a close_iteration registration to the run's
  current primary, with the never-primary-less fallback rule.
- `build_close_record`: assemble the iteration's closing record for
  `CEState.record_iteration`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .records import CloseRecord, ComparisonEvalEntry, RegisteredBranch

if TYPE_CHECKING:  # import for annotations only — closing stays cycle-free
    from .ce_state import CEState
    from .iteration_context import IterationContext

logger = logging.getLogger(__name__)

# (ancestor_sha, descendant_ref) -> bool
IsAncestorFn = Callable[[str, str], bool]
# (a, b) -> merge-base sha or None
MergeBaseFn = Callable[[str, str], "str | None"]


@dataclass
class DerivedParent:
    parent_sha: str | None
    parent_branch: str | None
    parent_iteration: int | None  # 0 = seed
    via: str  # "ancestor" | "merge_base" | "seed"


def derive_iteration_parent(
    *,
    close_sha: str,
    past_iterations: Sequence[CloseRecord],
    seed_sha: str,
    seed_branch: str,
    is_ancestor: IsAncestorFn,
    merge_base: MergeBaseFn,
) -> DerivedParent:
    """Derive this iteration's parent from past final shas + git history.

    Args:
        close_sha: HEAD of the branch the architect registered at close.
        past_iterations: prior closing records, oldest→newest. Each record
            needs `iteration`, `close_sha`, `branch`; records without a
            `close_sha` (architect-error stubs) are ignored.
        seed_sha / seed_branch: the run's seed commit (ultimate fallback).
        is_ancestor / merge_base: injected git predicates.

    Rule: walk past iterations newest→oldest; the first whose `close_sha` is
    an ancestor of this `close_sha` is the parent (`via="ancestor"`). If
    history was rewritten (rebase, deleted branch) so no past close sha is an
    ancestor, the first non-null `merge_base(close_sha, past_close_sha)` wins —
    the parent sha is the merge-base, attributed to that iteration
    (`via="merge_base"`). Otherwise the seed is the parent (`via="seed"`).

    Two things this deliberately does NOT do:

    - It does not filter out iterations that landed no change. An iteration
      that tested a change, measured a regression and reverted still leaves a
      real branch tip, and that tip is what the next iteration starts from —
      it is the parent whether or not the program moved. Excluding those
      records would skip the true parent and attribute the iteration to
      whatever landed two or three iterations back.
    - It keys on `close_sha` — the branch tip at close, including any commit
      the loop itself makes afterwards (the analysis agent's `codeevolver.md`
      update) — because that tip, not the last evaluated commit, is what the
      next iteration inherits.

    Only records with no `close_sha` at all are excluded: those are the error
    stubs written when the architect raised, which never touched git.
    """
    candidates = [rec for rec in past_iterations if rec.get("close_sha")]

    for rec in reversed(candidates):
        past_close = rec["close_sha"]
        if past_close == close_sha or is_ancestor(past_close, close_sha):
            return DerivedParent(
                parent_sha=past_close,
                parent_branch=rec.get("branch"),
                parent_iteration=rec.get("iteration"),
                via="ancestor",
            )

    for rec in reversed(candidates):
        base = merge_base(close_sha, rec["close_sha"])
        if base:
            return DerivedParent(
                parent_sha=base,
                parent_branch=rec.get("branch"),
                parent_iteration=rec.get("iteration"),
                via="merge_base",
            )

    return DerivedParent(
        parent_sha=seed_sha,
        parent_branch=seed_branch,
        parent_iteration=0,
        via="seed",
    )


def resolve_primary(
    entry: RegisteredBranch | None,
    prior_primary: str,
) -> tuple[str, bool]:
    """Apply a close_iteration registration to the run's primary branch.

    Returns `(primary_branch_after, primary_fallback)`.

    If the registration leaves no primary — the entry demotes the prior
    primary without promoting anything — the prior primary stays effective
    and `primary_fallback=True` (never reject: the run must always have a
    primary).
    """
    if not entry:
        return prior_primary, False

    branch = entry.get("branch")
    is_primary = _coerce_bool(entry.get("is_primary"))
    was_primary = _coerce_bool(entry.get("was_primary"))

    if is_primary and branch:
        return branch, False
    # Entry demotes the branch it registered. If that branch WAS the primary
    # and nothing was promoted in its place, the run would be primary-less —
    # fall back to the prior primary and warn.
    if was_primary and branch == prior_primary and not is_primary:
        logger.warning(
            "close_iteration demoted primary %r without promoting a "
            "replacement; keeping %r as primary (fallback)",
            branch, prior_primary,
        )
        return prior_primary, True
    return prior_primary, False


def _coerce_bool(value: Any) -> bool:
    """Coerce close_iteration's bool-or-\"True\"/\"False\"-string fields."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("true", "1", "yes")
    return bool(value)


def build_close_record(ctx: "IterationContext", state: "CEState") -> CloseRecord:
    """Assemble the closing record (`records.CloseRecord`) for
    `CEState.record_iteration`.

    `ctx` is the iteration's `IterationContext` after the deterministic close
    (parent derivation + `finalize()`) has run; `state` is the `CEState`.
    """
    entry = ctx.registered_branch
    primary_after, primary_fallback = resolve_primary(
        entry,
        ctx.primary_branch_at_start or state.primary_branch,
    )
    comparison_scores = comparison_entries(ctx)
    return {
        "iteration": ctx.iteration_idx,
        # `skipped` means the iteration produced NOTHING (architect raised, or
        # no branch was ever registered). An iteration that tested a change
        # and reverted is not skipped — see `changes_landed`.
        "skipped": ctx.skipped,
        "changes_landed": ctx.changes_landed,
        "branch": ctx.child_branch,
        "close_sha": ctx.close_sha,
        "close_tree": ctx.close_tree,
        "branches_seen": ctx.branches_seen,
        "registered_branch": ctx.registered_branch,
        "parent_sha": ctx.parent_sha,
        "parent_tree": ctx.parent_tree,
        "parent_branch": ctx.parent_branch,
        "parent_iteration": ctx.parent_iteration,
        "parent_is_primary": ctx.parent_is_primary,
        "parent_derivation": ctx.parent_derivation,
        "new_lineage": ctx.new_lineage,
        "subsample_baseline": ctx.initial_subsample_score,
        # Progress readings, not head certificates — see records.CloseRecord.
        "subsample_best": ctx.best_subsample_score,
        "subsample_best_sha": ctx.best_subsample_sha,
        "comparison_best": ctx.comparison_best_score,
        "comparison_best_sha": ctx.comparison_best_sha,
        "comparison_state": ctx.comparison_state,
        "comparison_eval_id_at_close": ctx.comparison_eval_id_at_close,
        "comparison_score_at_close": ctx.comparison_score_at_close,
        # Did the iteration's first training eval measure the parent, or the
        # parent plus edits the architect had already made? Observability —
        # editing before the baseline is permitted (see research_plan.md), but
        # a baseline taken on a modified tree makes the iteration's
        # baseline->final delta understate the change.
        "baseline_ran_on_modified_tree": ctx.baseline_tree_dirty,
        "baseline_dirty_paths": list(ctx.baseline_dirty_paths),
        "claimed_improvement_mode": ctx.claimed_improvement_mode,
        "memory_updated": ctx.memory_updated_claimed,
        "primary_branch_at_start": ctx.primary_branch_at_start,
        "primary_branch_after": primary_after,
        "primary_fallback": primary_fallback,
        "comparison_scores": comparison_scores,
        "promotion_evidence": (entry or {}).get("promotion_evidence"),
        # Parent-was-primary / branch-is-primary, as plain booleans. The
        # registered entry's `was_primary` is what the architect claimed;
        # `is_primary` is the OUTCOME of the transition (so a fallback-
        # resolved primary is reflected, not the claim).
        "was_primary": (
            _coerce_bool(entry.get("was_primary")) if entry
            else ctx.parent_is_primary
        ),
        "is_primary": (
            ctx.child_branch is not None and ctx.child_branch == primary_after
        ),
        "closed_by_fallback": ctx.closed_by_fallback,
        "watchdog_fired": ctx.watchdog_fired,
        # Only the iteration that ran the tool carries the estimate; later
        # iterations see it on the context (for the prompt) but not here.
        "variance_estimate": (
            dict(ctx.variance_estimate)
            if ctx.variance_estimate_ran and ctx.variance_estimate
            else None
        ),
    }


def comparison_entries(ctx: "IterationContext") -> list[ComparisonEvalEntry]:
    """Project the iteration's comparison evals for the close record
    (row scores deliberately excluded — they live in the population state and
    the comparison-history JSONL)."""
    out: list[ComparisonEvalEntry] = []
    for rec in ctx.eval_records:
        if rec.get("eval_sample") != "comparison":
            continue
        entry: ComparisonEvalEntry = {
            "branch": rec.get("branch"),
            "sha": rec.get("sha"),
            "score": rec.get("score"),
            "n_rows": rec.get("n_rows"),
        }
        if rec.get("eval_id"):
            entry["eval_id"] = rec["eval_id"]
        out.append(entry)
    return out


__all__ = [
    "DerivedParent",
    "derive_iteration_parent",
    "resolve_primary",
    "build_close_record",
    "comparison_entries",
]
