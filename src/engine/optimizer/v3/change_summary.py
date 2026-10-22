"""Copyright © 2026 440 Labs LLC

Change-summary artifacts for the EE loop (analysis agent + architect Phase 1).

The per-iteration record schema (`IterationHistory` / `ITERATION_RECORD_FIELDS`)
lives in `src/engine/schemas/iteration_history.py`; this module holds only the
EE-specific change-summary I/O built on top of it — reading each iteration's
`iteration_{N}_change_summary.md` and assembling the combined
`iteration_change_summaries.md` the architect reads. It imports `memory_dir` +
the subprocess file helpers, so it is not schema and stays in the v3 package.
"""

from __future__ import annotations

from src.engine.schemas.iteration_history import IterationHistory
from src.engine.services.subprocess_service import read_file, write_file

def seed_change_summary_section(architecture_title: str | None) -> str:
    """Synthesized combined-file entry for the seed program (iteration 0).

    The analysis agent never runs on the seed, so there is no
    `iteration_0_change_summary.md` to read. Both the seed step (initial write)
    and `build_combined_change_summaries` use this so the seed entry stays
    identical in both places.
    """
    title = architecture_title or "Seed program"
    return f"# Change Description Iteration 0 (seed)\nSeed program: {title}"


def change_summary_path(iteration: int) -> str:
    """Path to the per-iteration change summary file the analysis agent
    writes (`iteration_{N}_change_summary.md`)."""
    from src.engine.paths import memory_dir
    return f"{memory_dir()}/iteration_{iteration}_change_summary.md"


def combined_change_summaries_path() -> str:
    """Path to the combined change-summaries file the architect reads in
    Phase 1 (`iteration_change_summaries.md`)."""
    from src.engine.paths import memory_dir
    return f"{memory_dir()}/iteration_change_summaries.md"


def read_change_summary(iteration: int) -> str | None:
    """Read `iteration_{N}_change_summary.md` (the concise core-change summary).

    Returns the stripped content, or None when the file is absent or empty —
    e.g. for skipped iterations, where the analysis agent never ran.
    """
    content = read_file(change_summary_path(iteration))
    return content.strip() if content and content.strip() else None


def build_combined_change_summaries(history: "IterationHistory") -> str:
    """Concatenate every iteration's change summary into one document.

    Each section is headed by its iteration and branch (engine 3 candidate
    identity = branch + sha). A synthesized seed entry (iteration 0) is always
    placed first.

    Iterations that landed nothing ARE included, tagged `[REVERTED]`. They are
    where the run's negative results live — "this lever regresses via burial,
    stop trying it" — and excluding them would leave much of a run with no
    written record. Only
    genuinely skipped iterations (the architect raised) and iterations whose
    summary file is missing/empty are omitted.
    """
    seed_record = history.get(0) or {}
    sections: list[str] = [
        seed_change_summary_section(seed_record.get("architecture_title"))
    ]

    for iteration in sorted(n for n in history if isinstance(n, int) and n > 0):
        record = history[iteration]
        if record.get("skipped"):
            continue
        summary = read_change_summary(iteration)
        if not summary:
            continue
        branch = record.get("git_branch")
        sha = record.get("close_sha")
        short_sha = f" @ {str(sha)[:8]}" if sha else ""
        branch_part = f" ({branch}{short_sha})" if branch else ""
        # `changes_landed` may be absent on records written before the field
        # existed; treat missing as "landed" so old histories render as before.
        landed = record.get("changes_landed", True)
        reverted_tag = "" if landed else " [REVERTED — nothing landed]"
        header = (
            f"# Change Description Iteration {iteration}{branch_part}{reverted_tag}"
        )
        # The analysis agent is asked to lead a non-landing summary with a
        # `[REVERTED]` line; the header already says so, so drop the duplicate.
        if not landed and summary.startswith("[REVERTED]"):
            summary = summary[len("[REVERTED]"):].lstrip()
        sections.append(f"{header}\n{summary}")
    return "\n\n".join(sections) + "\n"


def write_combined_change_summaries(history: "IterationHistory") -> None:
    """Overwrite `iteration_change_summaries.md` with the combined document.

    Regenerated every iteration so the architect always reads a fresh copy.
    """
    write_file(combined_change_summaries_path(), build_combined_change_summaries(history))


__all__ = [
    "seed_change_summary_section",
    "change_summary_path",
    "combined_change_summaries_path",
    "read_change_summary",
    "build_combined_change_summaries",
    "write_combined_change_summaries",
]
