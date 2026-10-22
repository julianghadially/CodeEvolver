"""Copyright © 2026 440 Labs LLC

ASA population-management rules (specs/archive-public/v3_asa.md §Rule set / §Window /
§Progress units). Pure: closed iteration records + budget numbers in, an
`AsaPlan` out. No I/O, no git.

Rule 1 (new lineages): during the phase-1 window (until budget used > 50%
AND >= 10 closed iterations), once >= 3 iterations have closed, fire when
the share of closed iterations that created a new registered lineage is
below 40%. Off entirely after phase 1.

Rule 2 (eligible non-primary parent): while at least one eligible
non-primary lineage exists, count the closed iterations that started with a
non-empty eligible set (`n_iter_in_window`); once >= 2 such iterations have
closed, fire when the share whose derived parent was eligible is <= 30%.

Free tail: no rule fires once `evals_remaining <= max(10 * subsample_size,
0.10 * max_eval_rows)` (two estimated iterations or 10% of the budget).

Allowance: `ceil(n_closed / 10) - violations_so_far` — 1 for iterations
1-10, 2 for 11-20, ... — granted at iteration start and held for the whole
iteration (the run_eval gate warns instead of refusing while it is > 0).

Rule 3 (must explore, claimed 30%) is EE's explore guard, unchanged.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..run_metrics import _closed
from .prompt_sections import REQ_ELIGIBLE_NONPRIMARY, REQ_NEW_LINEAGE

RULE1_RATE = 0.40
RULE1_MIN_CLOSED = 3          # prompt injects / enforcement begin on iteration 4
PHASE1_MIN_ITERATIONS = 10
PHASE1_BUDGET_FRACTION = 0.50
RULE2_RATE = 0.30
RULE2_MIN_WINDOW = 2          # begins on the 3rd eligible iteration
FREE_TAIL_MIN_ITERATIONS = 2  # x (5 * subsample_size) rows per estimated iteration
FREE_TAIL_BUDGET_FRACTION = 0.10
ROWS_PER_ITERATION_ESTIMATE = 5
ALLOWANCE_PER_ITERATIONS = 10


@dataclass(frozen=True)
class AsaPlan:
    """What the loop decided at iteration start."""

    phase1: bool
    free_tail: bool
    requirements: list[str] = field(default_factory=list)
    allowance_left: int = 0
    n_closed: int = 0
    n_new_lineages: int = 0
    rule1_rate: float | None = None
    n_eligible_window: int = 0
    n_eligible_parent: int = 0
    rule2_rate: float | None = None
    violations_so_far: int = 0


def is_free_tail(
    *, evals_used: int, max_eval_rows: int | None, subsample_size: int
) -> bool:
    if not max_eval_rows:
        return False
    remaining = max_eval_rows - evals_used
    threshold = max(
        FREE_TAIL_MIN_ITERATIONS * ROWS_PER_ITERATION_ESTIMATE * subsample_size,
        FREE_TAIL_BUDGET_FRACTION * max_eval_rows,
    )
    return remaining <= threshold


def is_phase1(*, n_closed: int, evals_used: int, max_eval_rows: int | None) -> bool:
    """Phase 1 lasts until BOTH budget used > 50% and >= 10 closed."""
    if not max_eval_rows:
        return True
    budget_past_half = (evals_used / max_eval_rows) > PHASE1_BUDGET_FRACTION
    return not (budget_past_half and n_closed >= PHASE1_MIN_ITERATIONS)


def allowance(n_closed: int) -> int:
    return math.ceil(n_closed / ALLOWANCE_PER_ITERATIONS)


def plan_iteration(
    iterations: Sequence[Mapping[str, Any]],
    *,
    eligible: Sequence[str],
    evals_used: int,
    max_eval_rows: int | None,
    subsample_size: int,
) -> AsaPlan:
    closed = _closed(iterations)
    n = len(closed)
    violations = sum(1 for rec in closed if rec.get("asa_violation"))
    allowance_left = max(0, allowance(n) - violations)
    free_tail = is_free_tail(
        evals_used=evals_used, max_eval_rows=max_eval_rows, subsample_size=subsample_size
    )
    phase1 = is_phase1(n_closed=n, evals_used=evals_used, max_eval_rows=max_eval_rows)

    n_new = sum(1 for rec in closed if rec.get("new_lineage"))
    rule1_rate = (n_new / n) if n else None

    window = [rec for rec in closed if rec.get("asa_eligible_at_start")]
    n_w = len(window)
    n_sat = sum(1 for rec in window if rec.get("asa_parent_is_eligible"))
    rule2_rate = (n_sat / n_w) if n_w else None

    requirements: list[str] = []
    if not free_tail:
        if phase1 and n >= RULE1_MIN_CLOSED and rule1_rate is not None and rule1_rate < RULE1_RATE:
            requirements.append(REQ_NEW_LINEAGE)
        if eligible and n_w >= RULE2_MIN_WINDOW and rule2_rate is not None and rule2_rate <= RULE2_RATE:
            requirements.append(REQ_ELIGIBLE_NONPRIMARY)

    return AsaPlan(
        phase1=phase1,
        free_tail=free_tail,
        requirements=requirements,
        allowance_left=allowance_left,
        n_closed=n,
        n_new_lineages=n_new,
        rule1_rate=rule1_rate,
        n_eligible_window=n_w,
        n_eligible_parent=n_sat,
        rule2_rate=rule2_rate,
        violations_so_far=violations,
    )


def primary_record_is_stale(
    iterations: Sequence[Mapping[str, Any]],
    *,
    primary: str,
    record_iteration: int | None,
    stale_after_landed: int,
) -> bool:
    """The primary's comparison record predates more than `stale_after_landed`
    LANDED iterations on the primary branch (iteration-based, not commits)."""
    if record_iteration is None:
        return False
    landed_since = sum(
        1
        for rec in _closed(iterations)
        if rec.get("branch") == primary
        and rec.get("changes_landed")
        and int(rec.get("iteration") or 0) > record_iteration
    )
    return landed_since > stale_after_landed


__all__ = [
    "AsaPlan",
    "allowance",
    "is_free_tail",
    "is_phase1",
    "plan_iteration",
    "primary_record_is_stale",
    "RULE1_RATE",
    "RULE2_RATE",
]
