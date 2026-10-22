"""Copyright © 2026 440 Labs LLC

ASA rule injections (specs/archive-public/v3_asa.md §Rule 1 / §Rule 2).

Pure text. Imported lazily by `prompts.make_architect_prompt_builder` and
`IterationArchitect._build_initial_prompt`, which append these to the EE
system prompt / user message ONLY when `ctx.asa_requirements` is non-empty —
so every other arm's prompt is byte-identical to before. Both rules may fire
in the same iteration (one iteration can satisfy both: a new lineage forked
from an eligible non-primary parent).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..iteration_context import IterationContext

REQ_NEW_LINEAGE = "new_lineage"
REQ_ELIGIBLE_NONPRIMARY = "eligible_nonprimary"

MUST_CREATE_NEW_LINEAGE_SECTION = """
## THIS ITERATION MUST EXPLORE AND ADD A NEW LINEAGE
The run has created fewer new lineages than its pacing floor. Build this iteration's change on a new `codeevolver-` branch and register it at close. You choose the parent. Create the new branch FIRST (`git checkout -b codeevolver-<approach>`), then run your baseline — `run_eval` refuses a baseline taken on an existing branch while this requirement is active. See the `exploration` skill.
"""

MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY_SECTION = """
## THIS ITERATION MUST SELECT AN ELIGIBLE NON-PRIMARY BRANCH AS ITS PARENT
A non-primary branch has emerged that contains meaningfully positive behavioral contribution relative to the primary branch, and the share of iterations that have improved off of it (either in a new-branch exploration or an exploitation) has fallen below the 30% quota. See the `exploration` skill for context and use the `branch_comparison` tool to identify eligible non-primary branches. Check out the eligible branch (or fork a new `codeevolver-` branch from it) BEFORE running your baseline.
Eligible non-primary branches at the start of this iteration (normalized positive contribution vs the primary):
{eligible_lines}
"""

NEW_LINEAGE_NOTE = (
    "This iteration MUST add a new lineage and explore — create the new "
    "`codeevolver-` branch before your baseline; see the `exploration` skill. "
)
ELIGIBLE_NONPRIMARY_NOTE = (
    "This iteration MUST select an eligible non-primary branch as its parent — "
    "see the candidate pool section of the `exploration` skill. You may exploit "
    "or explore here. "
)

# run_eval refusal / warning notes (spec §Rule 1 / §Rule 2).
GATE_NOTE_NEW_LINEAGE = (
    "You must create a new branch / lineage for this iteration. Please select "
    "that branch before running the baseline and then call the run_eval tool "
    "again."
)
GATE_NOTE_ELIGIBLE_NONPRIMARY = (
    "You must use an eligible non-primary branch as the parent for this "
    "iteration. Please select that branch before running the baseline and "
    "then call the run_eval tool again."
)


def _eligible_lines(ctx: "IterationContext") -> str:
    contributions = getattr(ctx, "asa_eligible_contributions_at_start", {}) or {}
    names = list(getattr(ctx, "asa_eligible_at_start", []) or [])
    if not names:
        return "- (none recorded)"
    return "\n".join(
        f"- `{name}`: {contributions[name]:.4f}" if name in contributions else f"- `{name}`"
        for name in names
    )


def system_prompt_sections(ctx: "IterationContext") -> str:
    reqs = list(getattr(ctx, "asa_requirements", []) or [])
    out = ""
    if REQ_NEW_LINEAGE in reqs:
        out += MUST_CREATE_NEW_LINEAGE_SECTION
    if REQ_ELIGIBLE_NONPRIMARY in reqs:
        out += MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY_SECTION.format(
            eligible_lines=_eligible_lines(ctx)
        )
    return out


def user_message_notes(ctx: "IterationContext") -> str:
    reqs = list(getattr(ctx, "asa_requirements", []) or [])
    out = ""
    if REQ_NEW_LINEAGE in reqs:
        out += NEW_LINEAGE_NOTE
    if REQ_ELIGIBLE_NONPRIMARY in reqs:
        out += ELIGIBLE_NONPRIMARY_NOTE
    return out


__all__ = [
    "ELIGIBLE_NONPRIMARY_NOTE",
    "GATE_NOTE_ELIGIBLE_NONPRIMARY",
    "GATE_NOTE_NEW_LINEAGE",
    "MUST_BRANCH_FROM_ELIGIBLE_NONPRIMARY_SECTION",
    "MUST_CREATE_NEW_LINEAGE_SECTION",
    "NEW_LINEAGE_NOTE",
    "REQ_ELIGIBLE_NONPRIMARY",
    "REQ_NEW_LINEAGE",
    "system_prompt_sections",
    "user_message_notes",
]
