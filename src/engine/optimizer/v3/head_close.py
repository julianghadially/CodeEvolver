"""Copyright © 2026 440 Labs LLC

HEAD-registering close (`close_iteration_mode="head"`, the freeform arm).

The freeform architect is never told about branches, primaries or a
registration payload: whatever commit is checked out when the iteration
closes is its submitted state. `register_head` turns that git fact into the
same `RegisteredBranch` the EE close tool receives from the agent, so the
rest of the closing pipeline (parent derivation, finalize, constraint gate,
`build_close_record`, `resolve_primary`, push) runs unchanged:

- a dirty tree is auto-committed (the spec asks for a clean tree, not for a
  rejection — the agent was told the loop auto-commits);
- a branch without the `codeevolver-` prefix is renamed to carry it (the
  push guard and every branch listing key on the prefix);
- a detached HEAD gets a new branch, `codeevolver-<parent>-detached-<sha8>`,
  where `<parent>` is the branch the iteration started on;
- the entry is `is_primary=True` — the primary follows HEAD, so the run's
  final program is the HEAD of the last closed iteration.

Called from the `close_iteration` tool (so the close checks run on a real
branch) and again post-session by the loop (HEAD at session end wins, and
it is the fallback when the tool was never called). Idempotent.
"""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from src.engine.services.subprocess_service import (
    git_add_and_commit,
    git_branch_exists,
    git_checkout,
    git_create_branch,
    git_get_current_branch,
    git_get_sha,
    git_is_dirty,
    git_rename_branch,
)

from .records import RegisteredBranch

if TYPE_CHECKING:
    from .iteration_context import IterationContext

logger = logging.getLogger(__name__)

BRANCH_PREFIX = "codeevolver-"


def _resolve_head_branch(ctx: "IterationContext", workspace: str) -> str:
    """Return the `codeevolver-` branch HEAD is on, creating/renaming as needed."""
    sha = git_get_sha(workspace, "HEAD")
    branch = git_get_current_branch(workspace)

    if branch is None:
        parent = ctx.start_branch or ctx.primary_branch_at_start or "main"
        parent = parent[len(BRANCH_PREFIX):] if parent.startswith(BRANCH_PREFIX) else parent
        new = f"{BRANCH_PREFIX}{parent}-detached-{sha[:8]}"
        if git_branch_exists(workspace, new) and git_get_sha(workspace, new) != sha:
            new = f"{new}-{uuid.uuid4().hex[:6]}"
        if not git_branch_exists(workspace, new):
            git_create_branch(workspace, new, "HEAD")
        else:
            git_checkout(workspace, new)
        logger.warning(
            "iteration %d: HEAD was detached at %s; created branch %s",
            ctx.iteration_idx, sha[:12], new,
        )
        return new

    if not branch.startswith(BRANCH_PREFIX):
        new = f"{BRANCH_PREFIX}{branch}"
        if git_branch_exists(workspace, new):
            new = f"{new}-{sha[:8]}"
        git_rename_branch(workspace, branch, new)
        logger.warning(
            "iteration %d: branch %r lacked the %r prefix; renamed to %s",
            ctx.iteration_idx, branch, BRANCH_PREFIX, new,
        )
        return new

    return branch


def register_head(ctx: "IterationContext", workspace: str) -> RegisteredBranch:
    """Register HEAD as the iteration's submitted state (see module doc)."""
    if git_is_dirty(workspace):
        committed = git_add_and_commit(
            workspace,
            f"iter {ctx.iteration_idx}: uncommitted work at close (auto-committed)",
        )
        if committed:
            logger.info(
                "iteration %d: auto-committed dirty tree at close (sha=%s)",
                ctx.iteration_idx, git_get_sha(workspace, "HEAD")[:12],
            )

    branch = _resolve_head_branch(ctx, workspace)
    entry = RegisteredBranch(
        branch=branch,
        was_primary=branch == ctx.primary_branch_at_start,
        is_primary=True,
        promotion_evidence=None,
    )
    ctx.child_branch = branch
    ctx.registered_branch = entry
    ctx.claimed_improvement_mode = None
    return entry


__all__ = ["BRANCH_PREFIX", "register_head"]
