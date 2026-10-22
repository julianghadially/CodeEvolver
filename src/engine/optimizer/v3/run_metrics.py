"""Copyright © 2026 440 Labs LLC

Run-level (meta) metrics + the explore-frequency guard (engine 3).

The architect self-declares
`improvement_mode` (exploration / exploitation) at close; these pure helpers
derive the run-level exploration metrics from the closing records and decide
when to force an exploration iteration.

Derived-vs-claimed (spec §17): an iteration that merely *branches off* the
primary onto a new branch is still exploiting the primary lineage — for an
iteration to be derived-exploratory its PARENT program must be a non-primary
branch (`parent_is_primary is False`).
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

EXPLORE_GUARD_MIN_ITERATIONS = 4
EXPLORE_GUARD_THRESHOLD = 0.30


def _closed(iterations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Iterations that actually ran — every record but the architect-error stubs.

    Includes iterations that tested a change and reverted. Those are real
    attempts: they picked a parent, spent eval budget, declared an improvement
    mode, and left a branch tip. Excluding them would compute the run's
    exploration metrics over only part of the run and shrink the explore
    guard's denominator so it never fires.
    """
    return [rec for rec in iterations if not rec.get("skipped")]


def _landed(iterations: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Closed iterations whose program actually changed."""
    return [rec for rec in _closed(iterations) if rec.get("changes_landed")]


def derive_run_metrics(iterations: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute the spec §17 exploration metrics from closing records.

    - branches_explored: distinct branches that appeared in closing records.
    - iterations_closed / iterations_landed / land_rate: how many iterations
      ran, and how many of those moved the program.
    - iterations_on_non_primary: closes whose branch was not the primary at
      the iteration's start (a non-primary branch was worked on, even if it
      was forked from the primary).
    - efforts_on_non_primary_lineages: closes whose PARENT program was a
      non-primary branch.
    - claimed_explore_fraction: architect-declared exploration share.
    - derived_explore_fraction: share of closes with `parent_is_primary is
      False`.
    """
    closed = _closed(iterations)
    n = len(closed)
    branches = {rec.get("branch") for rec in closed if rec.get("branch")}
    on_non_primary = sum(
        1
        for rec in closed
        if rec.get("branch")
        and rec.get("primary_branch_at_start")
        and rec["branch"] != rec["primary_branch_at_start"]
    )
    non_primary_lineage = sum(
        1 for rec in closed if rec.get("parent_is_primary") is False
    )
    claimed = sum(
        1 for rec in closed if rec.get("claimed_improvement_mode") == "exploration"
    )
    landed = len(_landed(iterations))
    return {
        "branches_explored": len(branches),
        "iterations_closed": n,
        # How many closed iterations actually moved the program. The gap
        # between this and `iterations_closed` is the run's rejection rate —
        # iterations that built something, measured a regression and backed
        # out. That is signal about the search, not noise to be filtered.
        "iterations_landed": landed,
        "land_rate": (landed / n) if n else None,
        "iterations_on_non_primary": on_non_primary,
        "efforts_on_non_primary_lineages": non_primary_lineage,
        "claimed_explore_fraction": (claimed / n) if n else None,
        "derived_explore_fraction": (non_primary_lineage / n) if n else None,
    }


def explore_guard_should_fire(
    iterations: Sequence[Mapping[str, Any]],
    *,
    min_iterations: int = EXPLORE_GUARD_MIN_ITERATIONS,
    threshold: float = EXPLORE_GUARD_THRESHOLD,
) -> bool:
    """True when the run's claimed exploration share has fallen below quota.

    Checked at the start of each iteration; once at least `min_iterations`
    iterations have closed (including ones that tested a change and reverted
    — they consumed an iteration's worth of exploration budget either way)
    and the claimed explore fraction is below `threshold`, the prompt carries a
    "you must explore this iteration" block (`IterationContext.must_explore`).
    """
    closed = _closed(iterations)
    if len(closed) < min_iterations:
        return False
    fraction = derive_run_metrics(iterations)["claimed_explore_fraction"]
    return fraction is not None and fraction < threshold


__all__ = [
    "derive_run_metrics",
    "explore_guard_should_fire",
    "EXPLORE_GUARD_MIN_ITERATIONS",
    "EXPLORE_GUARD_THRESHOLD",
]
