"""Copyright © 2026 440 Labs LLC

Post-session closing-protocol watchdogs for the `IterationArchitect`.

Each watchdog is one-shot: a `decide` function
reads the host's state (context, config, git) and either returns `None`
(stay quiet) or a `Firing` carrying the exact resume prompt and the
"firing …" log line; `run_watchdog` then does the shared tail — log, resume
the session with the prompt, set `ctx.watchdog_fired`, and swallow + warn
on failure. Order is owned by `IterationArchitect.run()`.

The git helpers are imported into THIS namespace; tests that pin a
watchdog's git reads patch them here (`architect_watchdogs.git_get_sha`,
…), while the drift commit / reconciliation keep theirs on
`iteration_architect`.
"""

from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from src.engine.paths import memory_dir, traces_dir
from src.engine.services.subprocess_service import (
    git_get_sha,
    git_is_dirty,
    git_tree_signature,
)

from .iteration_context import TRAIN_EVAL_SAMPLES
from .program_paths import diff_is_non_program_only, program_diff_paths

logger = logging.getLogger(__name__)


def hash_file(path: str) -> str | None:
    """sha256 of a file's bytes, or None when it does not exist / can't read."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def memory_md_path() -> str:
    return f"{memory_dir()}/memory.md"


def _close_skill_invoked(tool_calls: list[dict[str, Any]]) -> bool:
    """True when a skill-loading tool call (`skill` on OpenCode, `Skill` on
    Claude) referenced `close_iteration` — or named no skill at all, in
    which case any skill load is accepted (fail-open on unreadable input)."""
    for call in tool_calls:
        name = str(call.get("name") or "")
        if name.lower() != "skill":
            continue
        inp = call.get("input")
        if not isinstance(inp, dict) or not inp:
            return True
        if "close_iteration" in " ".join(str(v) for v in inp.values()):
            return True
    return False


class WatchdogHost(Protocol):
    """What a watchdog reads off the architect. `IterationArchitect` is one."""

    context: Any
    config: Any
    workspace: str
    _last_session_id: str | None
    _session_tool_calls: list[dict[str, Any]] | None


def registered_head_sha(host: WatchdogHost) -> str | None:
    """HEAD of the branch the architect registered (falls back to HEAD)."""
    ref = host.context.child_branch or "HEAD"
    try:
        return git_get_sha(host.workspace, ref)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: could not read sha of %r (non-fatal): %s",
            host.context.iteration_idx, ref, exc,
        )
        return None


@dataclass(frozen=True)
class Firing:
    """A watchdog's decision to resume the session."""

    prompt: str
    log_msg: str
    log_args: tuple[Any, ...] = ()


@dataclass(frozen=True)
class Watchdog:
    name: str  # used only in the "<name> re-prompt failed (non-fatal)" warning
    decide: Callable[[WatchdogHost], Firing | None]


LaunchFn = Callable[..., None]


def run_watchdog(
    host: WatchdogHost,
    watchdog: Watchdog,
    system_prompt: str,
    tool_registry: dict[str, Callable[..., dict]],
    launch: LaunchFn,
) -> None:
    """Shared one-shot tail: decide → warn → resume → `watchdog_fired`."""
    firing = watchdog.decide(host)
    if firing is None:
        return
    logger.warning(firing.log_msg, *firing.log_args)
    try:
        launch(
            system_prompt,
            tool_registry,
            resume_session_id=host._last_session_id,
            initial_prompt_override=firing.prompt,
        )
        host.context.watchdog_fired = True
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: %s re-prompt failed (non-fatal): %s",
            host.context.iteration_idx, watchdog.name, exc,
        )


# ----- decisions -----


def decide_no_change(host: WatchdogHost) -> Firing | None:
    """Re-prompt the architect when it ended its session having made no
    code change at all and never closed the iteration.

    Fires when there is no `close_iteration` call, no `coding_agent`
    call, a clean working tree, and a HEAD still at the iteration-start
    sha (e.g. the architect ran the baseline eval, analyzed, then ended
    its turn on a narrative message). Fail-open on git errors.
    """
    iteration_idx = host.context.iteration_idx
    workspace = host.workspace

    if host.context.close_called:
        return None  # the architect deliberately concluded the iteration.
    if host.context.coding_agent_called:
        return None  # a change was attempted; the close-missing watchdog owns the rest.

    try:
        if git_is_dirty(workspace):
            return None  # uncommitted edits — later checks handle them.
        if host.context.start_sha is None:
            return None  # cannot compare — don't fire spuriously.
        if git_get_sha(workspace, "HEAD") != host.context.start_sha:
            return None  # a commit landed — a change exists.
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: no-change watchdog git check raised "
            "(allowing exit): %s",
            iteration_idx, exc,
        )
        return None

    if not host._last_session_id:
        logger.warning(
            "iteration %d: architect made no code change but no session_id "
            "captured; skipping no-change watchdog",
            iteration_idx,
        )
        return None

    return Firing(
        prompt=(
            f"You are about to end iteration {iteration_idx} WITHOUT having "
            "made any code change or closed the iteration. Do ONE of the "
            "following before exiting:\n"
            "  (a) Make your planned changes — call `coding_agent` with a "
            "concrete change_request, or make a direct `Edit`/`Write` and "
            "commit it — then call `run_eval` to test it, update memory.md, "
            "and call the `close_iteration` tool.\n"
            "  (b) If you have genuinely decided NOT to change anything this "
            "iteration (e.g. no actionable hypothesis), record that decision "
            "in memory.md and call the `close_iteration` tool to explicitly "
            "close the iteration. This prompt will not repeat."
        ),
        log_msg=(
            "iteration %d: architect ended its turn with NO code change and "
            "no close_iteration call; firing no-change watchdog"
        ),
        log_args=(iteration_idx,),
    )


def decide_close_missing(host: WatchdogHost) -> Firing | None:
    """One-shot re-prompt when the session ended without `close_iteration`.

    If the resume still ends without a close, the loop performs a
    deterministic fallback close (registering the current branch) so a
    crashed session's half-finished branch is inherited by the next
    iteration (spec §20).
    """
    iteration_idx = host.context.iteration_idx
    if host.context.close_called:
        return None
    if not host._last_session_id:
        logger.warning(
            "iteration %d: close_iteration missing but no session_id "
            "captured; loop will fallback-close",
            iteration_idx,
        )
        return None

    return Firing(
        prompt=(
            f"The session is about to exit but you have not called the "
            f"`close_iteration` tool for iteration {iteration_idx}. Invoke "
            "the `close_iteration` skill now and follow it: make sure your "
            "work is committed, update memory.md with your outcome "
            "reflection, then call the `close_iteration` tool with the fields "
            "its schema requires. Then exit."
        ),
        log_msg=(
            "iteration %d: session ended without close_iteration; firing "
            "close-missing watchdog"
        ),
        log_args=(iteration_idx,),
    )


def decide_close_skill(host: WatchdogHost) -> Firing | None:
    """One-shot re-prompt when `close_iteration` was called without the skill.

    The close_iteration skill carries the closing protocol (commit,
    memory reflection, per-arm notes); an agent that jumps straight to
    the tool skipped it. Fires only when the harness reported the
    session's tool calls (`raw_provider_metadata["tool_calls"]`) and none
    of them is a skill load naming `close_iteration`. Fail-open otherwise.
    All arms.
    """
    iteration_idx = host.context.iteration_idx
    if not host.context.close_called or not host.config.stage_project_skills:
        return None
    if not host._last_session_id or host._session_tool_calls is None:
        return None
    if _close_skill_invoked(host._session_tool_calls):
        return None

    return Firing(
        prompt=(
            f"You called the `close_iteration` tool for iteration "
            f"{iteration_idx} without loading the `close_iteration` skill. "
            "Load the `close_iteration` skill now and follow its steps "
            "(make sure your work is committed, update memory.md with your "
            "reflection, and any other notes it carries), then call the "
            "`close_iteration` tool again — the last call wins. Then exit."
        ),
        log_msg=(
            "iteration %d: close_iteration called without loading the "
            "close_iteration skill; firing close-skill watchdog"
        ),
        log_args=(iteration_idx,),
    )


def decide_revert_check(host: WatchdogHost) -> Firing | None:
    """Revert check (spec §2): warn when the closing sha scored lower on
    the training subsample than an earlier sha this iteration.

    One-shot resume asking the architect to leave the branch in the best
    state it judges — reverting in git if it wants. There is no forced
    revert: the architect may deliberately keep a lower-scoring state
    (e.g. a generalizable change it believes in). Once the run has a
    `run_variance_estimate`, a deficit within one run-to-run standard
    deviation does not fire (specs/archive-public/sampling_variance.md).

    Fires only when the iteration has NO comparison-set reading. The
    comparison set is the promotion basis on every arm, so once one ran,
    a train-subsample deficit is not evidence worth a resume — such a
    firing would argue against comparison evidence the architect already
    holds.
    """
    context = host.context
    iteration_idx = context.iteration_idx
    head_sha = registered_head_sha(host)
    if head_sha is None:
        return None

    # Comparison evidence supersedes the train subsample; loop-initiated
    # (`by_loop`) comparisons count — they are evidence all the same.
    if any(
        rec.get("eval_sample") == "comparison" for rec in context.eval_records
    ):
        logger.info(
            "iteration %d: revert check suppressed — a comparison eval ran "
            "this iteration",
            iteration_idx,
        )
        return None

    # `single` (1-row smoke checks) are excluded: they are in the train
    # family for budget/identity bookkeeping but are not a comparable
    # reading against a full-subsample closing score. Mirrors `finalize()`.
    train_records = [
        rec for rec in context.eval_records
        if rec.get("eval_sample") in TRAIN_EVAL_SAMPLES
        and rec.get("eval_sample") != "single"
    ]
    if len(train_records) < 2:
        return None
    # Score HEAD by CONTENT. A sha-keyed lookup answers the wrong
    # question: dirty-tree evals never carry HEAD's sha, and a closing
    # commit that only adds the change summary to codeevolver.md would
    # score as a *different* program, so the check would fire comparing an
    # unrelated re-eval against the best.
    head_tree = git_tree_signature(host.workspace, head_sha)
    head_scores = [
        r["score"] for r in train_records
        if r.get("sha") == head_sha
        or (
            r.get("tree")
            and diff_is_non_program_only(host.workspace, r["tree"], head_tree)
        )
    ]
    if not head_scores:
        return None  # closing content never scored — nothing to compare.
    head_score = head_scores[-1]
    best_rec = max(train_records, key=lambda r: r["score"])
    # Tolerance: one run-to-run standard deviation once the run has taken
    # its `run_variance_estimate` (specs/archive-public/sampling_variance.md §Revert-Check);
    # strict until then. A deficit within the noise floor is not a regression.
    sigma = float(
        ((getattr(context, "variance_estimate", None) or {}).get("run_to_run_std")) or 0.0
    )
    if best_rec["score"] <= head_score + sigma:
        return None
    # The best eval IS the closing program (recorded under a different sha
    # or a doc-only-different tree) — nothing to revert to.
    if best_rec.get("sha") == head_sha or (
        best_rec.get("tree")
        and diff_is_non_program_only(host.workspace, best_rec["tree"], head_tree)
    ):
        return None

    if not host._last_session_id:
        return None

    return Firing(
        prompt=(
            f"Warning: the current git sha (`{head_sha}`) corresponds to a "
            f"training subsample score of {head_score} — lower than a prior "
            f"subsample score of {best_rec['score']} at sha "
            f"`{best_rec.get('sha')}` this iteration"
            + (
                f" (by more than the run-to-run standard deviation of "
                f"{sigma:.4f})"
                if sigma > 0
                else ""
            )
            + ". You should leave the "
            "branch in the best state possible. Please pick the sha that you "
            "feel is best. If you want to revert, do so in git (e.g. "
            f"`git reset --hard {best_rec.get('sha')}`), then call the "
            "close_iteration tool again. If you are keeping the current "
            "state deliberately, just call close_iteration again."
        ),
        log_msg=(
            "iteration %d: closing sha %s scored %.4f < best %.4f (sha %s); "
            "firing revert-check watchdog"
        ),
        log_args=(
            iteration_idx, head_sha[:8], head_score,
            best_rec["score"], (best_rec.get("sha") or "?")[:8],
        ),
    )


def decide_memory_diff(host: WatchdogHost) -> Firing | None:
    """Memory-diff check: real work happened but memory.md has no diff.

    Engine 3's outcome reflection lives in memory.md (written directly by
    the architect, per the close_iteration skill). If the file is
    unchanged since iteration start while the iteration did real work
    (>= 2 train evals, or a sha moved), resume once and ask for the
    reflection.
    """
    context = host.context
    iteration_idx = context.iteration_idx
    if context.memory_md_hash_at_start is None and not os.path.exists(
        memory_md_path()
    ):
        return None  # no memory file at all (e.g. tests) — nothing to diff.

    current_hash = hash_file(memory_md_path())
    if current_hash != context.memory_md_hash_at_start:
        return None  # memory.md changed — reflection happened.

    train_evals = [
        rec for rec in context.eval_records
        if rec.get("eval_sample") in ("train", "subsample")
    ]
    head_sha = registered_head_sha(host)
    moved = head_sha is not None and head_sha != context.start_sha
    if len(train_evals) < 2 and not moved:
        return None  # nothing substantive to reflect on.

    if not host._last_session_id:
        return None

    return Firing(
        prompt=(
            f"It appears that you did real work in iteration {iteration_idx} but "
            f"`{memory_md_path()}` has not been updated. Write your outcome "
            "reflection directly into memory.md now (see the "
            "`close_iteration` skill for guidance. Update the memory.md file then call "
            "the `close_iteration` tool again with memory_updated=true."
        ),
        log_msg=(
            "iteration %d: memory.md has no diff after real work; firing "
            "memory-diff watchdog"
        ),
        log_args=(iteration_idx,),
    )


# ----- registry (order is owned by IterationArchitect.run) -----

NO_CHANGE = Watchdog("no-change watchdog", decide_no_change)
CLOSE_MISSING = Watchdog("close-missing watchdog", decide_close_missing)
CLOSE_SKILL = Watchdog("close-skill watchdog", decide_close_skill)
REVERT_CHECK = Watchdog("revert-check", decide_revert_check)
MEMORY_DIFF = Watchdog("memory-diff watchdog", decide_memory_diff)

# Before the drift commit: these decide on the session's own end state.
PRE_COMMIT_WATCHDOGS: tuple[Watchdog, ...] = (NO_CHANGE, CLOSE_MISSING, CLOSE_SKILL)
# After the drift commit + eval-sha reconciliation: these are sha/tree-keyed.
# (The evo close REVERTS to the comparison-run point rather than warning
# ahead of a loop rerun — specs/archive-public/comparison_credits.md.)
POST_RECONCILE_WATCHDOGS: tuple[Watchdog, ...] = (REVERT_CHECK, MEMORY_DIFF)


__all__ = [
    "CLOSE_MISSING",
    "CLOSE_SKILL",
    "Firing",
    "MEMORY_DIFF",
    "NO_CHANGE",
    "POST_RECONCILE_WATCHDOGS",
    "PRE_COMMIT_WATCHDOGS",
    "REVERT_CHECK",
    "Watchdog",
    "WatchdogHost",
    "_close_skill_invoked",
    "hash_file",
    "memory_md_path",
    "registered_head_sha",
    "run_watchdog",
]
