"""Copyright © 2026 440 Labs LLC

Pure-function helpers for the IterationArchitect session.

Factored out of `iteration_architect.py` so the class file stays focused on
session orchestration. Nothing here closes over class instance state.

  - `SKILLS_DIR` + `_stage_skills_into_workspace` — copy project skills
    into `{workspace}/.claude/skills/` so the Claude Agent SDK can find
    them via `setting_sources=["project"]`.
  - `_log_assistant_message` / `_log_result_message` — per-turn and
    session-exit logging so long architect sessions aren't opaque.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping

logger = logging.getLogger(__name__)


SKILLS_DIR = Path(__file__).parent / "skills"


def _log_assistant_message(
    iteration_idx: int, message: Any, TextBlock: type, ToolUseBlock: type
) -> None:
    """Log a summary of each assistant turn so long sessions aren't opaque.

    Emits one line per text block (truncated) and one per tool-use block
    (tool name + truncated input). Gives per-turn visibility when the
    session hangs or errors after minutes of work.
    """
    try:
        blocks = getattr(message, "content", None) or []
        for block in blocks:
            if isinstance(block, TextBlock):
                text = getattr(block, "text", "") or ""
                text = text.strip().replace("\n", " ")
                if text:
                    logger.info(
                        "iteration %d: architect text | %s",
                        iteration_idx,
                        text[:500] + ("…" if len(text) > 500 else ""),
                    )
            elif isinstance(block, ToolUseBlock):
                name = getattr(block, "name", "?")
                inp = getattr(block, "input", None)
                preview = repr(inp)
                if len(preview) > 500:
                    preview = preview[:500] + "…"
                logger.info(
                    "iteration %d: architect tool_use | %s(%s)",
                    iteration_idx, name, preview,
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("iteration %d: assistant log failure: %s", iteration_idx, exc)


def _log_result_message(iteration_idx: int, message: Any) -> None:
    """Dump the full ResultMessage on session exit — good or bad.

    ResultMessage fields that matter when diagnosing a failure:
    `subtype` ("success"|"error_max_turns"|"error_during_execution"),
    `stop_reason` (per-turn Anthropic stop reason), `num_turns`,
    `duration_ms`, and the optional `result` text.
    """
    try:
        logger.info(
            "iteration %d: architect ResultMessage | is_error=%s subtype=%r "
            "stop_reason=%r num_turns=%s duration_ms=%s duration_api_ms=%s "
            "total_cost_usd=%s result=%r",
            iteration_idx,
            getattr(message, "is_error", None),
            getattr(message, "subtype", None),
            getattr(message, "stop_reason", None),
            getattr(message, "num_turns", None),
            getattr(message, "duration_ms", None),
            getattr(message, "duration_api_ms", None),
            getattr(message, "total_cost_usd", None),
            (getattr(message, "result", None) or "")[:500],
        )
        usage = getattr(message, "usage", None)
        if usage:
            logger.info("iteration %d: architect usage | %r", iteration_idx, usage)
    except Exception as exc:  # noqa: BLE001
        logger.warning("iteration %d: ResultMessage log failure: %s", iteration_idx, exc)


def _stage_skills_into_workspace(
    workspace: str,
    skills: tuple[str, ...] | None = None,
    overrides: Mapping[str, Path] | None = None,
    extra_values: Mapping[str, str] | None = None,
) -> None:
    """Skill staging for Claude harness. Copy SKILLS_DIR into `{workspace}/.claude/skills/`.

    The Claude Agent SDK discovers project skills from `.claude/skills/`
    relative to `cwd` (with `setting_sources=["project"]`). SKILLS_DIR lives
    under the mounted engine tree at /app/src/engine/optimizer/v3/skills/,
    so we materialize a copy inside the session's cwd on each iteration.

    `.claude/` is ignored by git (see `LocalEnvironment._ensure_gitignore_entries`)
    so the coding agent's `git add -A` does not sweep these staged files into
    candidate commits.
    """
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills

    stage_codeevolver_skills(
        workspace,
        harness="claude",
        skills=skills,
        overrides=overrides,
        extra_values=extra_values,
    )


__all__ = [
    "SKILLS_DIR",
    "_log_assistant_message",
    "_log_result_message",
    "_stage_skills_into_workspace",
]
