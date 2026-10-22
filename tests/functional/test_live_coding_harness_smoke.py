"""Opt-in live smoke tests for real coding harness processes.

These tests are skipped by default so normal unit tests do not require a live
OpenCode or Claude installation, provider credentials, or model access.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest


OPENCODE_MODEL = "openai/gpt-5.3-codex-spark"


def _write_smoke_workspace(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "README.md").write_text(
        "# CodeEvolver harness smoke\n\nThis disposable repo is safe to edit.\n"
    )


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("CODEEVOLVER_LIVE_OPENCODE_SMOKE") != "1",
    reason="set CODEEVOLVER_LIVE_OPENCODE_SMOKE=1 to launch a real OpenCode run",
)
def test_live_opencode_coding_agent_smoke(tmp_path: Path) -> None:
    from src.engine.services.subprocess_service import run_coding_agent

    _write_smoke_workspace(tmp_path)

    result = run_coding_agent(
        str(tmp_path),
        change_request=(
            "Read README.md, append one short line confirming the OpenCode "
            "coding harness smoke ran, and stop."
        ),
        commit_changes=False,
        max_turns=6,
        coding_harness="opencode",
        coding_harness_model=os.environ.get(
            "CODEEVOLVER_LIVE_OPENCODE_MODEL", OPENCODE_MODEL
        ),
    )

    assert result["success"] is True, result
    assert "OpenCode" in (tmp_path / "README.md").read_text()


@pytest.mark.integration
@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("CODEEVOLVER_LIVE_CLAUDE_SMOKE") != "1",
    reason="set CODEEVOLVER_LIVE_CLAUDE_SMOKE=1 to launch a real Claude run",
)
def test_live_claude_coding_agent_smoke(tmp_path: Path) -> None:
    from src.engine.services.subprocess_service import run_coding_agent

    _write_smoke_workspace(tmp_path)

    result = run_coding_agent(
        str(tmp_path),
        change_request=(
            "Read README.md, append one short line confirming the Claude "
            "coding harness smoke ran, and stop."
        ),
        commit_changes=False,
        max_turns=6,
        coding_harness="claude",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result["success"] is True, result
    assert "Claude" in (tmp_path / "README.md").read_text()
