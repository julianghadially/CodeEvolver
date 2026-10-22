"""Copyright © 2026 440 Labs LLC

Unit tests for the combined iteration change-summaries feature (offline).

Covers:
- `build_combined_change_summaries` / `write_combined_change_summaries` —
  header format (engine 3: branch @ shortsha identity), seed entry, and
  omission of skipped / missing-summary iterations.
- The `IterationHistoryRecord.change_summary` schema field round-trips.
- Prompt wiring — architect prompts read `iteration_change_summaries.md`; the
  per-iteration analysis agent writes `iteration_<N>_change_summary.md` (no
  stale `iteration_description` names) and never touches memory.md.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.engine.schemas.iteration_history import IterationHistory
from src.engine.optimizer.v3.change_summary import (
    build_combined_change_summaries,
    change_summary_path,
    combined_change_summaries_path,
    read_change_summary,
    seed_change_summary_section,
    write_combined_change_summaries,
)


@pytest.fixture
def memory_env(tmp_path, monkeypatch):
    """Point memory_dir() at a clean temp dir with no job-id suffix."""
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    return tmp_path


def _write_summary(iteration: int, text: str) -> None:
    Path(change_summary_path(iteration)).write_text(text, encoding="utf-8")


def _history() -> IterationHistory:
    """Seed (0) + closed-on-branch (1) + closed-without-branch (2) +
    skipped (3) + closed-but-missing-file (4)."""
    hist = IterationHistory()
    hist[0] = {"architecture_title": "DSPy ReAct seed"}
    hist[1] = {
        "git_branch": "codeevolver-20260414-main",
        "close_sha": "abcdef0123456789",
        "skipped": False,
    }
    hist[2] = {"git_branch": None, "skipped": False}
    hist[3] = {"git_branch": None, "skipped": True}
    hist[4] = {
        "git_branch": "codeevolver-20260414-explore",
        "close_sha": "feedbeef01234567",
        "skipped": False,
    }
    return hist


def test_combined_headers_mapping_and_omissions(memory_env):
    hist = _history()
    _write_summary(1, "Added re-ranking to retrieval.")
    _write_summary(2, "Tried two-phase split; regressed.")
    _write_summary(3, "should be ignored (skipped)")  # skipped → omitted
    # iteration 4 deliberately has no file → omitted

    combined = build_combined_change_summaries(hist)

    # Seed entry first, synthesized from architecture_title.
    assert combined.startswith(
        "# Change Description Iteration 0 (seed)\nSeed program: DSPy ReAct seed"
    )
    # Closed iteration is identified by its branch @ short sha.
    assert (
        "# Change Description Iteration 1 (codeevolver-20260414-main @ abcdef01)\n"
        "Added re-ranking to retrieval." in combined
    )
    # Iteration without a recorded branch gets a bare header.
    assert (
        "# Change Description Iteration 2\nTried two-phase split; regressed."
        in combined
    )
    # Skipped iteration is omitted entirely (file content not surfaced).
    assert "should be ignored" not in combined
    assert "Iteration 3" not in combined
    # Closed-but-missing-file iteration is omitted (no summary to show).
    assert "Iteration 4" not in combined
    # No trailing "Iterations without candidates..." line.
    assert "Iterations without candidates" not in combined


def test_seed_entry_falls_back_when_no_title(memory_env):
    hist = IterationHistory()
    hist[0] = {"architecture_title": None}
    combined = build_combined_change_summaries(hist)
    assert combined.startswith(
        "# Change Description Iteration 0 (seed)\nSeed program: Seed program"
    )


def test_read_change_summary_missing_and_empty(memory_env):
    assert read_change_summary(7) is None  # no file
    _write_summary(7, "   \n  ")  # whitespace only
    assert read_change_summary(7) is None
    _write_summary(7, "  real content \n")
    assert read_change_summary(7) == "real content"


def test_write_combined_overwrites(memory_env):
    hist = _history()
    _write_summary(1, "first")
    write_combined_change_summaries(hist)
    out = Path(combined_change_summaries_path()).read_text(encoding="utf-8")
    assert "Iteration 1 (codeevolver-20260414-main @ abcdef01)\nfirst" in out
    # Regenerate after a new summary lands; file is overwritten, not appended.
    _write_summary(1, "second")
    write_combined_change_summaries(hist)
    out2 = Path(combined_change_summaries_path()).read_text(encoding="utf-8")
    assert "second" in out2
    assert "first" not in out2


def test_seed_section_helper_shared_format():
    assert seed_change_summary_section("My Arch") == (
        "# Change Description Iteration 0 (seed)\nSeed program: My Arch"
    )
    assert seed_change_summary_section(None).endswith("Seed program: Seed program")


def test_schema_field_roundtrips():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord

    rec = IterationHistoryRecord(change_summary="Added a retrieval tool.")
    dumped = rec.model_dump()
    assert dumped["change_summary"] == "Added a retrieval tool."
    # Defaults to None when omitted.
    assert IterationHistoryRecord().model_dump()["change_summary"] is None


def test_architect_prompts_read_combined_file():
    from src.engine.optimizer.v3.iteration_context import IterationContext
    from src.engine.optimizer.v3.prompts import (
        ARCHITECT_SYSTEM_PROMPT,
        FREEFORM_ARCHITECT_SYSTEM_PROMPT,
        build_evo_architect_system_prompt,
    )

    evo = build_evo_architect_system_prompt(
        additional_instructions=None,
        program_path=None,
        workspace_path="/workspace",
        context=IterationContext(iteration_idx=1),
    )
    for prompt in (ARCHITECT_SYSTEM_PROMPT, evo, FREEFORM_ARCHITECT_SYSTEM_PROMPT):
        assert "iteration_change_summaries.md" in prompt


def test_analysis_agent_prompt_writes_change_summary_only():
    from src.engine.optimizer.v3.prompts import build_analysis_agent_prompt

    prompt = build_analysis_agent_prompt(
        iteration_num=3,
        workspace_path="/workspace/repo",
        branch="codeevolver-20260414-main",
        change_request="add tool",
        subsample_baseline=0.5,
        subsample_best=0.6,
        child_trace_dir="/traces/iteration_3_child",
        parent_trace_dir="/traces/iteration_2_child",
        subsample_size=20,
    )
    # Writes the canonical per-iteration change summary file...
    assert "iteration_3_change_summary.md" in prompt
    assert "iteration_description" not in prompt
    assert "iteration_change_summaries.md" in prompt
    # ...and is explicitly forbidden from touching memory.md (the architect
    # owns its own outcome reflection in engine 3).
    assert "Do NOT edit" in prompt and "memory.md" in prompt
    # It OWNS codeevolver.md: on a landing iteration it rewrites the
    # architecture description in place, keeping the parsed headers and
    # never turning the file into a changelog.
    assert "Update `/workspace/repo/codeevolver.md` IN PLACE" in prompt
    assert "## ARCHITECTURE TITLE:" in prompt
    assert "NOT a changelog" in prompt
    assert "Please remove change summaries" not in prompt
    # Context: branch identity plus the baseline/best subsample scores — the
    # closing commit is never claimed to have been measured.
    assert "codeevolver-20260414-main" in prompt
    assert "0.5000" in prompt and "0.6000" in prompt
    assert "closing commit itself may not have been re-measured" in prompt
    assert "/traces/iteration_3_child/" in prompt
    assert "/traces/iteration_2_child/" in prompt


def test_reverted_iteration_is_included_and_tagged(memory_env):
    """An iteration that tested a change and reverted still gets a section.

    Omitting reverted iterations would drop a run's negative results ("this
    lever regresses, stop trying it") before the next architect reads them.
    Only a genuinely skipped iteration (the architect raised) is omitted.
    """
    hist = IterationHistory()
    hist[0] = {"architecture_title": "DSPy ReAct seed"}
    hist[1] = {
        "git_branch": "codeevolver-hover-react-gap",
        "close_sha": "a4c123b1612dd272",
        "skipped": False,
        "changes_landed": False,
    }
    _write_summary(1, "[REVERTED]\nAdded coverage verification; regressed 0.83 -> 0.79 via burial.")

    combined = build_combined_change_summaries(hist)

    assert (
        "# Change Description Iteration 1 "
        "(codeevolver-hover-react-gap @ a4c123b1) [REVERTED — nothing landed]"
        in combined
    )
    assert "regressed 0.83 -> 0.79 via burial." in combined
    # The header already says REVERTED; the agent's leading tag is not repeated.
    assert combined.count("[REVERTED]") == 0


def test_records_without_changes_landed_render_as_before(memory_env):
    """Histories written before the field existed must not all read REVERTED."""
    hist = IterationHistory()
    hist[0] = {"architecture_title": "seed"}
    hist[1] = {
        "git_branch": "codeevolver-old",
        "close_sha": "abcdef0123456789",
        "skipped": False,
    }
    _write_summary(1, "Added re-ranking.")

    combined = build_combined_change_summaries(hist)

    assert "REVERTED" not in combined
    assert "# Change Description Iteration 1 (codeevolver-old @ abcdef01)" in combined
