"""Copyright © 2026 440 Labs LLC

Post-session constraint gate — the loop's anti-cheating check.

Judges the session's actual diff (`start_sha..end_sha`) against the client's
`additional_instructions` with the shared LM judge (`v3/constraint_gate.py`,
pinned to Claude Sonnet 5). With no admission gate, a violation must not
carry forward, so the branch is reset to the session's start commit, and an
empty marker commit (`Revert iteration N: ...` + the quoted constraint) is
left on top so the revert is visible in `git log`.
"""

from __future__ import annotations

import logging
from typing import Callable

from src.engine.optimizer.v3.constraint_gate import (
    build_constraint_revert_message,
    commit_constraint_revert_marker,
    evaluate_change_constraints,
)
from src.engine.optimizer.v3.lm_config import build_constraint_lm_callable
from src.engine.services.subprocess_service import exec_bash

logger = logging.getLogger(__name__)


def build_constraint_checker(
    *, additional_instructions: str | None, reflection_lm: str | None
) -> Callable[[str], str] | None:
    """The judge callable, or None (gate disabled) when no constraints exist."""
    if additional_instructions and additional_instructions.strip() and reflection_lm:
        return build_constraint_lm_callable(reflection_lm)
    logger.warning(
        "simple loop: no additional_instructions supplied — the constraint gate "
        "is DISABLED for this run"
    )
    return None


def check_session_constraints(
    *,
    workspace: str,
    start_sha: str,
    end_sha: str,
    additional_instructions: str | None,
    lm_callable: Callable[[str], str] | None,
    iteration_idx: int,
) -> str | None:
    """Judge the session diff; on a violation reset HEAD to `start_sha` and
    leave the revert marker commit on top (HEAD moves; content does not).

    Returns the violated constraint text when the session was reverted, else
    None. Fail-open on judge/infra errors (the shared gate returns None).
    """
    verdict = evaluate_change_constraints(
        workspace=workspace,
        parent_sha=start_sha,
        final_sha=end_sha,
        additional_instructions=additional_instructions,
        lm_callable=lm_callable,
    )
    if verdict is None or verdict.status != "fail":
        return None
    reason = verdict.violated_constraint or verdict.reason or "unspecified"
    logger.warning(
        "simple loop: constraint violation (%s) — reverting %s -> %s",
        reason, end_sha[:8], start_sha[:8],
    )
    reset = exec_bash(workspace, f"git reset --hard {start_sha}")
    if reset.get("returncode") != 0:
        raise RuntimeError(
            f"constraint revert to {start_sha[:8]} failed: {reset.get('stderr')}"
        )
    commit_constraint_revert_marker(
        workspace,
        build_constraint_revert_message(iteration_idx, verdict.violated_constraint),
    )
    return reason


__all__ = ["build_constraint_checker", "check_session_constraints"]
