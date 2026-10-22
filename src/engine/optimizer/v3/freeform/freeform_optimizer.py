"""Copyright © 2026 440 Labs LLC

Freeform optimizer — research Baseline 1 (`specs/v3_freeform.md`).

What differs from EE (everything else is a control):
- Prompt: `prompts.FREEFORM_ARCHITECT_SYSTEM_PROMPT` — no explore/exploit
  guidance, no git/candidate-pool section, no comparison-set sentence.
- Skills: `exploration` is not staged; `close_iteration` is replaced by
  `skills/close_iteration_replacement.md` (one-field tool, HEAD is the
  submitted state) which renders `{final_iteration_note}` when budget
  exhaustion approaches.
- Explore guard off; `improvement_mode` never requested (None on records).
- Close: `close_iteration_mode="head"` — `head_close.register_head` derives
  the branch from HEAD. The primary follows HEAD, including after a
  constraint force-revert (the branch was reset to the parent's content, and
  the loop continues from HEAD either way), so the run's final program is
  the HEAD of the last closed iteration.
- Memory template: `build_memory_md_template(method="freeform")` → the
  macro-free template (no Macro Approaches, no Candidate Pool).
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

from ..optimizer_loop import run_optimization
from ..prompts import build_freeform_architect_system_prompt
from ..strategy import LoopServices, OptimizerStrategy

if TYPE_CHECKING:
    from ..ce_state import CEState
    from ..iteration_context import IterationContext
    from ..records import CloseRecord

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Rendered into the close skill's `{final_iteration_note}` slot when budget
# exhaustion approaches (HEAD at the last close is the run's final program).
FINAL_ITERATION_NOTE = (
    "**Budget exhaustion is approaching.** If this is your final iteration, "
    "take extra care to submit the correct head as the final program. "
    "Whatever head you submit will be final once the budget is complete."
)
FINAL_ITERATION_SUBSAMPLE_MULTIPLE = 5
FINAL_ITERATION_BUDGET_FRACTION = 0.9


def final_iteration_note(ctx: "IterationContext") -> str:
    """The note, or "" while the budget is comfortably unspent.

    Fires when the remaining eval rows are at most
    `FINAL_ITERATION_SUBSAMPLE_MULTIPLE` training subsamples; when the
    subsample size is unknown, at `FINAL_ITERATION_BUDGET_FRACTION` of the
    budget; never on an unbudgeted run.
    """
    max_rows = ctx.max_eval_rows
    if not max_rows:
        return ""
    used = ctx.evals_used or 0
    n = ctx.subsample_size
    if n:
        near = (max_rows - used) <= FINAL_ITERATION_SUBSAMPLE_MULTIPLE * n
    else:
        near = used / max_rows >= FINAL_ITERATION_BUDGET_FRACTION
    return FINAL_ITERATION_NOTE if near else ""


def freeform_skill_template_values(ctx: "IterationContext") -> Mapping[str, str]:
    return {"final_iteration_note": final_iteration_note(ctx)}


def freeform_on_iteration_closed(
    ctx: "IterationContext",
    state: "CEState",
    close_record: "CloseRecord",
    services: LoopServices,
) -> None:
    """The primary follows HEAD: the closed branch is always the primary."""
    branch = close_record.get("branch")
    if branch:
        close_record["primary_branch_after"] = branch
        close_record["primary_fallback"] = False


FREEFORM_STRATEGY = OptimizerStrategy(
    method="freeform",
    prompt_builder=build_freeform_architect_system_prompt,
    skills_to_stage=("close_iteration",),
    skill_overrides={
        "close_iteration": SKILLS_DIR / "close_iteration_replacement.md",
    },
    skill_template_values=freeform_skill_template_values,
    enable_explore_guard=False,
    close_iteration_mode="head",
    on_iteration_closed=freeform_on_iteration_closed,
)


def run_freeform_optimization(**kwargs) -> dict:
    """Entry point: `run_entry(run_freeform_optimization, method="freeform")`."""
    return run_optimization(strategy=FREEFORM_STRATEGY, **kwargs)


__all__ = [
    "FINAL_ITERATION_NOTE",
    "FREEFORM_STRATEGY",
    "final_iteration_note",
    "freeform_on_iteration_closed",
    "freeform_skill_template_values",
    "run_freeform_optimization",
]
