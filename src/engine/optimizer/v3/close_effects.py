"""Copyright © 2026 440 Labs LLC

Git side-effects of closing an iteration — the deterministic steps the loop
(`v3.optimizer_loop`) runs on the client repo after the architect session
ends.

Everything here is best-effort and never control flow: a failure is logged
and the loop carries on with what it has. Grouped by when the loop calls it:

- pre-session: `cleanup_dirty_workspace_if_needed`
- close: `head_close` (freeform), `fallback_close` (session ended without
  `close_iteration`), `force_revert_to_parent` (constraint gate fired)
- post-close, before the record: `checkout_registered_branch`,
  `write_attempted_diff`, `commit_codeevolver_md_if_modified`
- anchoring: `anchor_eval_trees`, `anchor_iteration_state` (durable
  `refs/codeevolver/*` so recorded shas / evaluated trees survive `git gc`)
- `tree_of` — tree hash of a ref, None when there is no ref
"""

from __future__ import annotations

import logging
import os

from src.engine.paths import history_dir
from src.engine.services.job_context import JobContext
from src.engine.services.subprocess_service import (
    exec_bash,
    git_add_and_commit,
    git_checkout,
    git_commit_tree,
    git_dirty_paths,
    git_get_current_branch,
    git_get_sha,
    git_push_authenticated,
    git_tree_signature,
    git_update_ref,
)

from .ce_state import CEState
from .constraint_gate import (
    build_constraint_revert_message,
    commit_constraint_revert_marker,
)
from .head_close import register_head
from .iteration_architect import hash_file, memory_md_path
from .iteration_context import IterationContext
from .program_paths import diff_is_non_program_only

logger = logging.getLogger(__name__)


def cleanup_dirty_workspace_if_needed(
    *,
    workspace: str,
    iteration_idx: int,
    job: JobContext,
) -> None:
    """Pre-iteration safety net — fires when the previous iteration's
    workstation-committed check failed to clean the workspace.

    `git_add_and_commit` is a no-op when the tree is already clean — the
    common case — so this guard is cheap to call unconditionally each
    iteration.
    """
    current_branch = git_get_current_branch(workspace)
    try:
        committed = git_add_and_commit(
            workspace,
            f"iter {iteration_idx - 1}: pre-iteration safety-net wip "
            f"(untested)",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: pre-iteration safety-net commit failed "
            "(non-fatal): %s",
            iteration_idx, exc,
        )
        return

    if not committed:
        return  # workspace was already clean — nothing to do

    try:
        safety_sha = git_get_sha(workspace, "HEAD")
    except Exception:  # noqa: BLE001
        safety_sha = "<unknown>"
    logger.warning(
        "iteration %d: pre-iteration safety net fired — committed "
        "uncommitted drift as %s on branch %s. The end-of-iteration "
        "workstation-committed check should have handled this.",
        iteration_idx,
        safety_sha[:8] if safety_sha != "<unknown>" else safety_sha,
        current_branch or "<detached HEAD>",
    )

    if not (job.repo_url and current_branch):
        return
    try:
        push_result = git_push_authenticated(
            workspace=workspace,
            branch=current_branch,
            **job.git_creds(),
            set_upstream=True,
        )
        if not push_result.get("success"):
            logger.warning(
                "iteration %d: pre-iteration safety-net push of %s "
                "failed (non-fatal): %s",
                iteration_idx, current_branch, push_result.get("stderr"),
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: pre-iteration safety-net push raised "
            "(non-fatal): %s",
            iteration_idx, exc,
        )


def head_close(
    ctx: IterationContext,
    *,
    workspace: str,
    state: CEState,
) -> None:
    """HEAD-registering close (`close_iteration_mode="head"`, freeform).

    Re-runs `register_head` after the session so the submitted state is
    HEAD at session end — whether or not the tool was called, and even if
    the agent moved HEAD after calling it. When the tool was never called
    this doubles as the fallback close (`closed_by_fallback=True`, memory
    claim derived from the memory.md hash). A git failure degrades to the
    generic `fallback_close`.
    """
    was_fallback = not ctx.close_called
    try:
        register_head(ctx, workspace)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: register_head failed (%s); using generic fallback close",
            ctx.iteration_idx, exc,
        )
        if was_fallback:
            fallback_close(ctx, workspace=workspace, state=state)
        return
    if was_fallback:
        current_hash = hash_file(memory_md_path())
        ctx.memory_updated_claimed = (
            current_hash is not None and current_hash != ctx.memory_md_hash_at_start
        )
        ctx.close_called = True
        ctx.closed_by_fallback = True
        logger.warning(
            "iteration %d: session ended without close_iteration; registered "
            "HEAD on branch %s (memory_updated=%s)",
            ctx.iteration_idx, ctx.child_branch, ctx.memory_updated_claimed,
        )


def fallback_close(
    ctx: IterationContext,
    *,
    workspace: str,
    state: CEState,
) -> None:
    """Deterministic fallback close when the architect never called
    `close_iteration` (crash, watchdog miss). The half-finished branch is
    inherited by the next iteration (spec §20): register the current
    `codeevolver-` branch, carry the primary state forward, leave
    improvement_mode unclaimed.
    """
    branch = None
    try:
        current = git_get_current_branch(workspace)
    except Exception:  # noqa: BLE001
        current = None
    if ctx.child_branch:
        # Loop-owned branch (pareto_evolution): the iteration's branch is a
        # fact, whatever HEAD the architect wandered to.
        branch = ctx.child_branch
    elif current and current.startswith("codeevolver-"):
        branch = current
    elif ctx.start_branch and ctx.start_branch.startswith("codeevolver-"):
        branch = ctx.start_branch
    if branch is None:
        branch = state.primary_branch

    was_primary = branch == (ctx.primary_branch_at_start or state.primary_branch)
    ctx.child_branch = branch
    ctx.registered_branch = {
        "branch": branch,
        "was_primary": was_primary,
        "is_primary": was_primary,
        "promotion_evidence": None,
    }
    ctx.claimed_improvement_mode = None
    current_hash = hash_file(memory_md_path())
    ctx.memory_updated_claimed = (
        current_hash is not None and current_hash != ctx.memory_md_hash_at_start
    )
    ctx.close_called = True
    ctx.closed_by_fallback = True
    logger.warning(
        "iteration %d: fallback-closed on branch %s (was_primary=%s, "
        "memory_updated=%s)",
        ctx.iteration_idx, branch, was_primary, ctx.memory_updated_claimed,
    )


def checkout_registered_branch(ctx: IterationContext, *, workspace: str) -> None:
    """Put the workspace on the branch the architect REGISTERED before any
    post-session agent writes into it.

    These differ whenever the architect leaves HEAD on a different branch
    than it submits; without this, that iteration's codeevolver.md commit
    would land on — and become the start state of — an unrelated lineage.
    Best-effort.
    """
    if not ctx.child_branch:
        return
    try:
        current = git_get_current_branch(workspace)
        if current != ctx.child_branch:
            logger.warning(
                "iteration %d: workspace is on %s but the registered branch is "
                "%s — checking out the registered branch before the analysis "
                "agent runs",
                ctx.iteration_idx, current, ctx.child_branch,
            )
            git_checkout(workspace, ctx.child_branch)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: could not check out registered branch (non-fatal): %s",
            ctx.iteration_idx, exc,
        )


def commit_codeevolver_md_if_modified(
    ctx: IterationContext,
    *,
    workspace: str,
    iteration_idx: int,
) -> bool:
    """Commit the analysis agent's `codeevolver.md` update on the registered
    branch. Returns True when a commit was made. Best-effort.

    The analysis agent owns `codeevolver.md`: after each landing
    iteration it rewrites the architecture description in place. The loop
    does not append to the file itself (appended blocks would accumulate
    forever). It commits what the agent changed; warns if it changed nothing on
    a landing iteration; discard anything it changed OUTSIDE codeevolver.md
    (it is told not to, and an unreviewed program edit here would silently
    become the next iteration's start state).
    """
    try:
        dirty = git_dirty_paths(workspace, max_paths=200)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: codeevolver.md status check failed (non-fatal): %s",
            iteration_idx, exc,
        )
        return False
    stray = [p for p in dirty if p != "codeevolver.md"]
    if stray:
        logger.warning(
            "iteration %d: analysis agent modified files other than "
            "codeevolver.md (%s) — discarding those edits",
            iteration_idx, ", ".join(stray[:8]),
        )
        try:
            exec_bash(workspace, "git checkout -- " + " ".join(
                f"'{p}'" for p in stray
            ))
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "iteration %d: could not discard stray analysis-agent edits "
                "(non-fatal): %s",
                iteration_idx, exc,
            )
    if "codeevolver.md" not in dirty:
        if ctx.changes_landed:
            logger.warning(
                "iteration %d: program changed but the analysis agent left "
                "codeevolver.md untouched — the architecture description may "
                "be stale",
                iteration_idx,
            )
        return False
    try:
        committed = git_add_and_commit(
            workspace,
            f"iter {iteration_idx}: update codeevolver.md",
            paths=["codeevolver.md"],
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: codeevolver.md commit failed (non-fatal): %s",
            iteration_idx, exc,
        )
        return False
    if committed:
        logger.info(
            "iteration %d: committed the analysis agent's codeevolver.md update",
            iteration_idx,
        )
    return bool(committed)


MAX_ATTEMPTED_DIFF_BYTES = 120_000


def write_attempted_diff(ctx: IterationContext, *, workspace: str) -> str | None:
    """Write the diff of what this iteration TRIED into the history dir.

    Parent tree vs. the best-scoring tree evaluated this iteration. For a
    landing iteration that is (near enough) the diff that shipped; for a
    non-landing one it is the only surviving description of the change the
    architect built, measured and backed out — the commit never existed, but
    the tree object does.

    Returns the path, or None when there is nothing to diff (no evals, or the
    best tree is the parent's). Best-effort; never raises.
    """
    tree = ctx.best_subsample_tree
    if not tree or not ctx.parent_tree or tree == ctx.parent_tree:
        return None
    try:
        result = exec_bash(workspace, f"git diff {ctx.parent_tree} {tree}")
        if result.get("returncode") != 0:
            return None
        body = result.get("stdout") or ""
        if not body.strip():
            return None
        if len(body) > MAX_ATTEMPTED_DIFF_BYTES:
            body = (
                body[:MAX_ATTEMPTED_DIFF_BYTES]
                + "\n\n... [diff truncated by CodeEvolver] ...\n"
            )
        path = os.path.join(
            history_dir(), f"iteration_{ctx.iteration_idx}_attempted.diff"
        )
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(body)
        return path
    except Exception as exc:  # noqa: BLE001 - context for the summary, not control flow
        logger.warning(
            "iteration %d: could not write the attempted diff (non-fatal): %s",
            ctx.iteration_idx, exc,
        )
        return None


def tree_of(workspace: str, ref: str | None) -> str | None:
    """Tree of `ref`, or None when there is no ref.

    `git_tree_signature(ws, None)` deliberately means "hash the working tree" —
    the mode `run_eval` uses. At the closing call sites a missing ref means
    UNKNOWN, and silently substituting the working tree would compute
    `changes_landed` against the wrong baseline.
    """
    return git_tree_signature(workspace, ref) if ref else None


def anchor_iteration_state(ctx: IterationContext, *, workspace: str) -> None:
    """Pin this iteration's end state (and the programs it tried) with refs.

    `close_sha` is recorded faithfully, but a recorded sha is not a reachable
    one: a later iteration that resets the same branch orphans it, and `git gc`
    prunes orphans. Every recorded end state must stay checkout-able for
    post-run analysis, so point a ref at it.

    The trees the iteration evaluated get the same treatment. `run_eval` writes
    a real tree object for every eval, including evals of an uncommitted
    working tree that was then reverted — that is what makes "what did this
    iteration try, and how did it differ from the parent?" answerable for a
    non-landing iteration. Unreferenced, those trees are prunable too.
    `git commit-tree` wraps each one so `git diff` and `git checkout` work on
    it directly.

    Entirely best-effort: refs are analysis infrastructure, never control flow.
    `refs/codeevolver/*` is outside `refs/heads/*`, so none of this is pushed
    to origin or visible to the architect's branch listing.
    """
    n = ctx.iteration_idx
    if ctx.close_sha:
        git_update_ref(workspace, f"refs/codeevolver/iterations/{n}", ctx.close_sha)
    anchor_eval_trees(ctx, workspace=workspace)


def anchor_eval_trees(ctx: IterationContext, *, workspace: str) -> None:
    """Wrap every evaluated tree in a `refs/codeevolver/evals/<N>/<i>` commit
    and stamp the ref on the record as `anchor_ref`. Idempotent — records that
    already carry an anchor are skipped, so this can run before the close
    record is built (so `subsample_best_sha` can point at a never-committed
    tree) and again at the end."""
    n = ctx.iteration_idx
    anchored: dict[str, str] = {
        rec["tree"]: rec["anchor_ref"]
        for rec in ctx.eval_records
        if rec.get("tree") and rec.get("anchor_ref")
    }
    for i, rec in enumerate(ctx.eval_records):
        tree = rec.get("tree")
        if not tree:
            continue
        if tree in anchored:
            rec.setdefault("anchor_ref", anchored[tree])
            continue
        snapshot = git_commit_tree(
            workspace,
            tree,
            parent=ctx.parent_sha,
            message=(
                f"codeevolver iter {n} eval snapshot "
                f"({rec.get('eval_sample')}, score={rec.get('score')})"
            ),
        )
        if snapshot:
            ref = f"refs/codeevolver/evals/{n}/{i}"
            if git_update_ref(workspace, ref, snapshot):
                rec["anchor_ref"] = ref
                anchored[tree] = ref
    ctx.refresh_durable_refs()


def revert_to_comparison_point(
    ctx: IterationContext, *, workspace: str, target_sha: str
) -> bool:
    """pareto_evolution close contract (specs/archive-public/comparison_credits.md): the
    comparison run is the close step, so program changes made AFTER it are
    reverted back to the auto-committed comparison-run commit. Returns True
    when the reset succeeded.

    Mirrors `force_revert_to_parent`: reset the child branch, then re-run
    `finalize()` so the score attribution and `changes_landed` are derived
    from the tree actually on disk. Best-effort — a failed reset leaves the
    close as-is (the caller then falls back to a loop rerun).
    """
    iteration_idx = ctx.iteration_idx
    try:
        if ctx.child_branch:
            git_checkout(workspace, ctx.child_branch)
        reset = exec_bash(workspace, f"git reset --hard {target_sha}")
        if reset.get("returncode") != 0:
            logger.warning(
                "iteration %d: revert to comparison point failed (non-fatal): %s",
                iteration_idx, reset.get("stderr"),
            )
            return False
        ctx.close_sha = target_sha
        ctx.close_tree = tree_of(workspace, target_sha)
        ctx.finalize(lambda a, b: diff_is_non_program_only(workspace, a, b))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: revert to comparison point raised (non-fatal): %s",
            iteration_idx, exc,
        )
        return False


def force_revert_to_parent(
    ctx: IterationContext,
    *,
    workspace: str,
    violated_constraint: str | None = None,
) -> bool:
    """Constraint-gate force-revert: put the registered branch back on the
    parent commit, leave an empty `Revert iteration N` marker commit on top
    (the fact + the quoted constraint, never the judge's reason) and re-close.
    Returns True when the reset succeeded (the marker is best-effort).

    Re-closes via `finalize()` rather than hand-writing the attribution.
    `initial_subsample_score` is the session's first train eval, which is NOT
    necessarily an eval of the parent's content — branch navigation is
    ungated, so eval-on-A then checkout-B then close-on-B is a permitted
    sequence. Re-running finalize derives the score, its source, and
    `changes_landed` from the tree that is actually on disk. Best-effort.
    """
    iteration_idx = ctx.iteration_idx
    try:
        if ctx.child_branch:
            git_checkout(workspace, ctx.child_branch)
        reset = exec_bash(workspace, f"git reset --hard {ctx.parent_sha}")
        if reset.get("returncode") != 0:
            logger.warning(
                "iteration %d: constraint revert failed (non-fatal): %s",
                iteration_idx, reset.get("stderr"),
            )
            return False
        ctx.close_sha = ctx.parent_sha
        ctx.close_tree = ctx.parent_tree
        marker = commit_constraint_revert_marker(
            workspace,
            build_constraint_revert_message(iteration_idx, violated_constraint),
        )
        if marker:
            # Same tree as the parent, so `finalize` still reports no change.
            ctx.close_sha = marker
        ctx.finalize(lambda a, b: diff_is_non_program_only(workspace, a, b))
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: constraint revert raised (non-fatal): %s",
            iteration_idx, exc,
        )
        return False


__all__ = [
    "MAX_ATTEMPTED_DIFF_BYTES",
    "anchor_eval_trees",
    "anchor_iteration_state",
    "checkout_registered_branch",
    "cleanup_dirty_workspace_if_needed",
    "commit_codeevolver_md_if_modified",
    "fallback_close",
    "force_revert_to_parent",
    "head_close",
    "revert_to_comparison_point",
    "tree_of",
    "write_attempted_diff",
]
