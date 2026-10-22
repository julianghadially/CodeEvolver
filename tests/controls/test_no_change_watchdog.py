"""Copyright © 2026 440 Labs LLC

No-change watchdog (deterministic, no real SDK).

When the architect ends its session having made no code change at all — no
`close_iteration` call, no `coding_agent` call, a clean working tree, and
HEAD still at the iteration-start sha — the loop resumes the session once and
asks it to either make the change it analyzed or explicitly close the
iteration. Without this, such an iteration is silently fallback-closed,
wasting the session's time and cost.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from src.engine.optimizer.v3 import iteration_architect as ia_module
from src.engine.optimizer.v3 import architect_watchdogs as wd_module
from src.engine.services.subprocess_service import git_is_dirty

from tests.controls._arch_helpers import make_architect


def _capture_launch(monkeypatch, arch):
    calls: list[dict] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda sp, tr, *, resume_session_id=None, initial_prompt_override=None: calls.append(
            {"resume": resume_session_id, "prompt": initial_prompt_override}
        ),
    )
    return calls


def _no_change_setup(monkeypatch):
    """Architect that produced nothing: no close, no coding_agent call, clean
    tree, HEAD still at the iteration-start sha."""
    arch, ctx, _ = make_architect(iteration_idx=3)
    assert ctx.start_sha == "seedsha"
    ctx.coding_agent_called = False
    arch._last_session_id = "sess-1"
    monkeypatch.setattr(wd_module, "git_is_dirty", lambda ws: False)
    monkeypatch.setattr(
        wd_module, "git_get_sha", lambda ws, ref="HEAD": "seedsha"
    )
    return arch, ctx


def test_watchdog_fires_when_no_change_made(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert ctx.watchdog_fired is True
    assert len(calls) == 1
    assert calls[0]["resume"] == "sess-1"
    assert "coding_agent" in calls[0]["prompt"]
    assert "close_iteration" in calls[0]["prompt"]


def test_watchdog_does_not_fire_when_close_called(monkeypatch):
    # Informed abandonment: the architect deliberately concluded the
    # iteration through the close_iteration tool. Do not re-prompt.
    arch, ctx = _no_change_setup(monkeypatch)
    ctx.close_called = True
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_when_tree_is_dirty(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)
    # Uncommitted edits exist — later watchdogs handle them.
    monkeypatch.setattr(wd_module, "git_is_dirty", lambda ws: True)
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_when_head_advanced(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)
    # A commit landed past the iteration-start sha → a change exists.
    monkeypatch.setattr(
        wd_module, "git_get_sha", lambda ws, ref="HEAD": "child_sha"
    )
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_when_coding_agent_called(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)
    ctx.coding_agent_called = True
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_without_start_sha(monkeypatch):
    # No iteration-start snapshot — cannot compare, so
    # never fire spuriously.
    arch, ctx = _no_change_setup(monkeypatch)
    ctx.start_sha = None
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_does_not_fire_without_session_id(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)
    arch._last_session_id = None
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_watchdog_fails_open_on_git_error(monkeypatch):
    arch, ctx = _no_change_setup(monkeypatch)

    def _boom(ws):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(wd_module, "git_is_dirty", _boom)
    calls = _capture_launch(monkeypatch, arch)

    arch._run_no_change_watchdog_if_needed("SYS", {})

    assert calls == []  # fail open — never trap the architect on a git error
    assert ctx.watchdog_fired is False


# --- git_is_dirty helper, against a real temp repo ---

def _init_git(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"], cwd=path, check=True
    )
    subprocess.run(["git", "config", "user.name", "Test"], cwd=path, check=True)


def test_git_is_dirty_detects_clean_untracked_and_committed(tmp_path):
    _init_git(tmp_path)
    # Fresh repo with no commits and no files is clean.
    assert git_is_dirty(str(tmp_path)) is False

    # Untracked file → dirty.
    (tmp_path / "f.txt").write_text("hello")
    assert git_is_dirty(str(tmp_path)) is True

    # Committed → clean again.
    subprocess.run(["git", "add", "-A"], cwd=tmp_path, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=tmp_path, check=True)
    assert git_is_dirty(str(tmp_path)) is False
