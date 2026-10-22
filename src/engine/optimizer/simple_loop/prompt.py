"""Copyright © 2026 440 Labs LLC

System-prompt builder for the simple loop.

Loads `prompt.md` and substitutes run-level values. Wired as
`ArchitectConfig.system_prompt_builder`, so the signature matches the v3
builders. The budget note is deliberately bare -- percent used and rows
remaining, nothing else -- because Baseline 0 gets no planning guidance
(specs/v3_simple_loop.md: "the sample size comments are not something to keep").
"""

from __future__ import annotations

from pathlib import Path

from src.engine.optimizer.v3.constraint_gate import PREVIOUS_ITERATION_REVERTED_NOTE
from src.engine.schemas.iteration_context import BaseIterationContext
from src.engine.paths import traces_dir

_PROMPT_PATH = Path(__file__).parent / "prompt.md"


def _budget_note(max_eval_rows: int | None, evals_used: int) -> str:
    """One line of eval-budget awareness for the system prompt.

    Empty when no budget is configured. Own copy (not the CE prompt module) so
    the simple loop shares no prompt code with the treatment arms; the number
    it reports is the start-of-session snapshot, matching when the prompt is
    built. States ONLY the percent used and the rows remaining -- no per-eval
    costs, no endgame advice: the simple loop is the no-guidance baseline.
    """
    if not max_eval_rows:
        return ""
    remaining = max(max_eval_rows - evals_used, 0)
    pct = evals_used / max_eval_rows * 100
    return (
        f"You have used {pct:.0f}% of your evaluation budget; "
        f"{remaining} of {max_eval_rows} rows remain.\n\n"
    )


def _additional_section(additional_instructions: str | None) -> str:
    if not additional_instructions or not additional_instructions.strip():
        return ""
    return "\n## Additional instructions\n" + additional_instructions.strip() + "\n"


def _program_section(program_path: str | None) -> str:
    if not program_path:
        return ""
    return f"\nThe program entry point is `{program_path}`.\n"


def build_simple_loop_prompt(
    *,
    additional_instructions: str | None,
    program_path: str | None,
    workspace_path: str,
    context: BaseIterationContext,
) -> str:
    """Render the simple-loop system prompt for one session."""
    body = _PROMPT_PATH.read_text(encoding="utf-8")
    return body.format(
        traces_dir=traces_dir(),
        workspace_path=workspace_path,
        program_section=_program_section(program_path),
        budget_note=_budget_note(context.max_eval_rows, context.evals_used),
        additional_section=_additional_section(additional_instructions),
    )


def build_kickoff_message(context: BaseIterationContext) -> str:
    """The per-session user message that starts the agent.

    Carries the constraint-revert note when the previous session was reverted
    by the gate — the fact only, same wording as every other arm.
    """
    message = (
        f"You are on session {context.iteration_idx}. Improve the program on the current branch."
    )
    if context.previous_iteration_constraint_reverted:
        message += " " + PREVIOUS_ITERATION_REVERTED_NOTE
    return message


__all__ = ["build_simple_loop_prompt", "build_kickoff_message"]
