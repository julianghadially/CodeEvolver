"""Copyright © 2026 440 Labs LLC

Post-change constraint gate, shared by every optimizer loop.

After an iteration commits its change, this gate judges the iteration's actual
git diff (`parent_sha..final_sha`) against the client's `additional_instructions`
constraints. Reading the diff — rather than the architect's change-request
string — is deliberate: it catches both the `coding_agent` path and direct
architect `Edit`/`Write` edits, so a constraint violation cannot slip through by
bypassing the coding agent.

The gate is **opt-in by `additional_instructions`**: with no constraints
supplied it is a no-op, so default loop behavior is unchanged. It is also
**fail-open** — any infra error returns `None` so a flaky checker call never
blocks optimization. What each loop *does* with a violation differs:

  - EE / freeform / pareto (admission-gated): record the iteration as rejected
    and never admit the child to the pool.
  - simple_loop (no admission): `simple_loop/constraint_check.py`
    `git reset --hard`'s HEAD back to the session start so the
    violating code does not carry forward. This module is on the
    simple_loop v3 import allowlist because the gate is a shared
    anti-cheating control, not simple loop logic.

"""

from __future__ import annotations

import logging
import subprocess
from typing import Callable

from src.engine.agent import check_diff_constraints
from src.engine.schemas.lm_output_schemas import ConstraintCheckResult
from src.engine.services.subprocess_service import exec_bash, git_get_sha

logger = logging.getLogger(__name__)


def evaluate_change_constraints(
    *,
    workspace: str,
    parent_sha: str | None,
    final_sha: str | None,
    additional_instructions: str | None,
    lm_callable: Callable[[str], str] | None,
    change_requests: list[str] | None = None,
    outcome_reflection: str | None = None,
) -> ConstraintCheckResult | None:
    """Judge an iteration's diff against client constraints.

    Returns the :class:`ConstraintCheckResult` (inspect ``.status`` for
    ``"fail"``) when a check actually ran, or ``None`` when the gate is
    disabled / there is nothing to check / an error occurred (fail-open):

      - no `additional_instructions` or no `lm_callable` → gate disabled,
      - `parent_sha`/`final_sha` missing or equal → no change to judge,
      - empty diff → nothing changed,
      - any exception computing the diff → ``None`` (never block the loop).
    """
    if not additional_instructions or not additional_instructions.strip():
        return None
    if lm_callable is None:
        return None
    if not parent_sha or not final_sha or parent_sha == final_sha:
        return None

    try:
        result = exec_bash(workspace, f"git diff {parent_sha} {final_sha}")
        if result.get("returncode") != 0:
            logger.warning(
                "constraint gate: `git diff %s %s` failed (returncode=%s) — "
                "skipping check: %s",
                parent_sha[:8], final_sha[:8],
                result.get("returncode"), result.get("stderr"),
            )
            return None
        diff = result.get("stdout") or ""
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "constraint gate: computing diff raised (non-fatal) — skipping "
            "check: %s",
            exc,
        )
        return None

    if not diff.strip():
        logger.info(
            "constraint gate: empty diff %s..%s — nothing to judge",
            parent_sha[:8], final_sha[:8],
        )
        return None

    result = check_diff_constraints(
        additional_instructions=additional_instructions,
        diff=diff,
        lm_callable=lm_callable,
        change_requests=change_requests,
        outcome_reflection=outcome_reflection,
    )
    # The verdict is the loop's only anti-cheating evidence, so every judged
    # diff leaves one line, including a pass.
    if result is None:
        logger.warning(
            "constraint gate: no verdict for %s..%s (judge returned None)",
            parent_sha[:8], final_sha[:8],
        )
    elif result.status == "fail":
        logger.warning(
            "constraint gate: FAIL for %s..%s — violated %r: %s",
            parent_sha[:8], final_sha[:8], result.violated_constraint, result.reason,
        )
    else:
        logger.info(
            "constraint gate: pass for %s..%s — %s",
            parent_sha[:8], final_sha[:8], result.reason,
        )
    return result


# --- Revert visibility (all arms) --------------------------------------------
#
# Every arm reverts a violating iteration with `git reset --hard`, which leaves
# no trace in git and nothing in the next iteration's prompt, so the agent can
# rebuild the same violating tree session after session. The marker
# commit and the kickoff note share the FACT of the revert and the quoted
# constraint (text the agent already holds in its prompt) — never the judge's
# reason, which would teach the agent how to argue past the checker. The note
# deliberately does not point the agent at git: the user message is the
# signal; the commit on the reverted branch only makes it stickier.

PREVIOUS_ITERATION_REVERTED_NOTE = (
    "The previous iteration's changes were reverted by the constraint checker "
    "because they violated the constraints in the additional instructions."
)


def build_constraint_revert_message(
    iteration_idx: int, violated_constraint: str | None
) -> str:
    """Commit message for the revert marker: the notice plus the verbatim
    violated constraint. Takes `verdict.violated_constraint`, never
    `verdict.reason`."""
    quoted = (violated_constraint or "").strip() or "(not quoted by the checker)"
    return (
        f"Revert iteration {iteration_idx}: the constraint checker found this "
        "change violated the constraints in the additional instructions.\n\n"
        f"Violated constraint: {quoted}"
    )


def commit_constraint_revert_marker(workspace: str, message: str) -> str | None:
    """Empty commit carrying `message` on the current HEAD (the reset point).

    Called AFTER the loop's `git reset --hard` so the branch history reads
    `<parent> -> Revert iteration N`. Best-effort: returns the new HEAD sha,
    or None (logged) when the commit fails — a failed marker never undoes
    the reset.
    """
    try:
        result = subprocess.run(
            [
                "git", "-c", "user.email=ce@codeevolver.ai",
                "-c", "user.name=CodeEvolver",
                "commit", "--allow-empty", "-q", "-m", message,
            ],
            cwd=workspace, capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.warning(
                "constraint gate: revert marker commit failed (non-fatal): %s",
                result.stderr,
            )
            return None
        return git_get_sha(workspace)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "constraint gate: revert marker commit raised (non-fatal): %s", exc
        )
        return None


__all__ = [
    "PREVIOUS_ITERATION_REVERTED_NOTE",
    "build_constraint_revert_message",
    "commit_constraint_revert_marker",
    "evaluate_change_constraints",
]
