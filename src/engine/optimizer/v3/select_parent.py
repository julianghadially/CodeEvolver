"""Copyright © 2026 440 Labs LLC

Preselect-parent git mechanic.

Engine 3 EE does not use this — the architect owns branch choice and
there is no `select_candidate` tool. `select_parent` locks the context and
forks the iteration's child branch from a preselected parent;
`fork_child_from` serves the pareto_evolution arm. It also hosts `utc_run_timestamp`, used
by the seed step and the loop.
"""

from __future__ import annotations

import datetime as _dt
import logging
import uuid
from typing import Any, Protocol

from .errors import SelectCandidateError
from src.engine.services.subprocess_service import git_checkout, git_create_branch, git_get_sha
from .iteration_context import IterationContext

logger = logging.getLogger(__name__)


class _CandidatePool(Protocol):
    """Minimal view over the candidate pool needed by `select_parent`.

    `program_candidates` is an *attribute* (a list), not a method. Typing it
    as a Protocol attribute makes a future accidental
    `state.program_candidates()` call a type-checker error rather than a
    runtime `TypeError: 'list' object is not callable`.
    """

    program_candidates: list[dict[str, Any]]


def select_parent(
    *,
    ctx: IterationContext,
    candidate_idx: int,
    state: _CandidatePool,
    workspace: str,
    run_timestamp: str,
) -> dict[str, Any]:
    """Check out parent `candidate_idx` and fork this iteration's child branch.

    Policy-free: the caller has already decided `candidate_idx` (the
    architect for EE, the optimizer loop for pareto_evolution). This only
    performs the git mechanic and mutates `ctx` in place.

    Args:
        ctx: IterationContext for the current iteration. Mutated in place.
        candidate_idx: Index of the parent candidate in
            `state.program_candidates`.
        state: Candidate-pool state (or a test double).
        workspace: Path to the client repo workspace.
        run_timestamp: The run's timestamp prefix used in branch names, e.g.
            `"20260414120000"`. Reused for every iteration of a single run so
            branches sort together.

    Returns:
        A JSON-serializable dict suitable for the Agent SDK to deliver back to
        the model (also used by the optimizer-driven callers, which ignore it).

    Raises:
        SelectCandidateError: On contract violations (already called this
            iteration, invalid index, or missing branch).
    """
    candidate_idx = int(candidate_idx)
    if ctx.parent_locked:
        raise SelectCandidateError(
            f"select_parent already called for iteration {ctx.iteration_idx}. "
            "It is one-shot per iteration."
        )

    candidates = state.program_candidates
    if not 0 <= candidate_idx < len(candidates):
        raise SelectCandidateError(
            f"candidate_idx={candidate_idx} is out of range "
            f"(pool has {len(candidates)} candidates)"
        )

    parent = candidates[candidate_idx]
    parent_branch = parent.get("git_branch")
    if not parent_branch:
        raise SelectCandidateError(
            f"candidate_idx={candidate_idx} has no `git_branch` field"
        )

    child_branch = f"codeevolver-{run_timestamp}-{uuid.uuid4().hex[:6]}"
    git_checkout(workspace, parent_branch)
    parent_sha = git_get_sha(workspace, parent_branch)
    git_create_branch(workspace, child_branch, parent_branch)

    ctx.parent_sha = parent_sha
    ctx.parent_branch = parent_branch
    ctx.child_branch = child_branch
    ctx.parent_locked = True

    logger.info(
        "iteration %d: selected candidate %d (branch=%s, sha=%s), child=%s",
        ctx.iteration_idx,
        candidate_idx,
        parent_branch,
        parent_sha[:8],
        child_branch,
    )

    return {
        "ok": True,
        "candidate_idx": candidate_idx,
        "parent_branch": parent_branch,
        "parent_sha": parent_sha,
        "child_branch": child_branch,
        "parent_architecture_title": parent.get("architecture_title"),
        "parent_architecture_summary": parent.get("architecture_summary"),
    }


def fork_child_from(
    *,
    ctx: IterationContext,
    parent_branch: str,
    parent_sha: str,
    parent_iteration: int | None,
    workspace: str,
    run_timestamp: str,
) -> str:
    """Check out `parent_sha` and fork this iteration's child branch from it.

    The loop-owned-population mechanic (pareto_evolution): the caller has
    already chosen the parent candidate as (branch, sha) — the sha is the
    candidate's recorded `close_sha`, NOT the branch tip, so a later iteration
    moving that branch can never change which program a candidate denotes.

    Child branch: `codeevolver-{run_timestamp}-iter{N}` — deterministic and
    unique per run. If it already exists (a crashed attempt of this same
    iteration) it is reset to the parent sha.

    Mutates `ctx`: parent_sha / parent_branch / parent_iteration /
    parent_tree / child_branch / parent_locked, and `parent_is_primary`
    relative to `ctx.primary_branch_at_start`.
    """
    from src.engine.services.subprocess_service import (
        git_branch_exists,
        git_tree_signature,
    )

    child_branch = f"codeevolver-{run_timestamp}-iter{ctx.iteration_idx}"
    git_checkout(workspace, parent_sha)
    if git_branch_exists(workspace, child_branch):
        logger.warning(
            "iteration %d: child branch %s already exists (crashed attempt?); "
            "resetting it to parent %s",
            ctx.iteration_idx, child_branch, parent_sha[:8],
        )
        from src.engine.services.subprocess_service import exec_bash

        exec_bash(workspace, f"git branch -D {child_branch}")
    git_create_branch(workspace, child_branch, parent_sha)

    ctx.parent_sha = parent_sha
    ctx.parent_branch = parent_branch
    ctx.parent_iteration = parent_iteration
    try:
        ctx.parent_tree = git_tree_signature(workspace, parent_sha)
    except Exception:  # noqa: BLE001
        ctx.parent_tree = None
    ctx.child_branch = child_branch
    ctx.parent_locked = True
    ctx.parent_is_primary = parent_branch == ctx.primary_branch_at_start

    logger.info(
        "iteration %d: forked %s from candidate iteration=%s (branch=%s, sha=%s)",
        ctx.iteration_idx, child_branch, parent_iteration, parent_branch,
        parent_sha[:8],
    )
    return child_branch


def utc_run_timestamp(now: _dt.datetime | None = None) -> str:
    """Build the `YYYYMMDDHHmmss` prefix used in branch names."""
    now = now or _dt.datetime.now(tz=_dt.timezone.utc)
    return now.strftime("%Y%m%d%H%M%S")


__all__ = ["fork_child_from", "select_parent", "SelectCandidateError", "utc_run_timestamp"]
