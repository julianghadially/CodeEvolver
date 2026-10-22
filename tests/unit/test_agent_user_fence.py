"""Copyright © 2026 440 Labs LLC

Unit tests for the Docker-runner holdout fence (unprivileged agent user).

The engine spawns agent subprocesses with ``user=$CODEEVOLVER_AGENT_USER``
when that env var is set (only the runner image sets it), so the kernel — not
tool-level deny paths — fences the root-owned mode-600 testset that
``runner/entrypoint.py::sequester_holdout`` moved outside the workspace. These
tests pin three properties:

1. Inertness: with the env var unset (every local / non-Docker deploy) no
   spawn site gains a ``user=`` kwarg — behavior is byte-identical to before.
2. Activation: with the env var set, the OpenCode/Claude agent spawn paths
   pass ``user=<name>``, EXCEPT the run-integrity inspector's privileged
   exception (``run_as_agent_user=False`` — it must read the testset).
3. Sequestration: the testset file leaves both the working tree and the git
   object store (``git show``/``cat-file`` find nothing), and the run is
   submitted with the relocated absolute ``testset_path``.
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "runner"))

from src.engine import paths  # noqa: E402
from src.engine.agent.harnesses.claude import ClaudeCodeHarness  # noqa: E402
from src.engine.agent.harnesses.base import (  # noqa: E402
    AgentRole,
    HarnessPermissionPolicy,
    HarnessRequest,
)
from src.engine.agent.harnesses.opencode.harness import OpenCodeHarness  # noqa: E402
from src.engine.services import subprocess_service  # noqa: E402


AGENT_USER_ENV = "CODEEVOLVER_AGENT_USER"


@pytest.fixture
def agent_user_set(monkeypatch):
    monkeypatch.setenv(AGENT_USER_ENV, "ce-agent")


@pytest.fixture
def agent_user_unset(monkeypatch):
    monkeypatch.delenv(AGENT_USER_ENV, raising=False)


# ---------------------------------------------------------------------------
# paths accessor
# ---------------------------------------------------------------------------

def test_agent_user_none_when_unset(agent_user_unset):
    assert paths.agent_user() is None


def test_agent_user_none_when_blank(monkeypatch):
    monkeypatch.setenv(AGENT_USER_ENV, "   ")
    assert paths.agent_user() is None


def test_agent_user_returns_name(agent_user_set):
    assert paths.agent_user() == "ce-agent"


# ---------------------------------------------------------------------------
# OpenCode harness spawns
# ---------------------------------------------------------------------------

class _RecordingRunner:
    def __init__(self):
        self.kwargs = None

    def __call__(self, cmd, **kwargs):
        self.kwargs = kwargs
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")


def _run_command(**call_kwargs):
    runner = _RecordingRunner()
    harness = OpenCodeHarness(runner=runner)
    harness._run_command(["opencode", "--version"], cwd=".", timeout=5, **call_kwargs)
    return runner.kwargs


def test_opencode_run_command_inert_when_unset(agent_user_unset):
    assert "user" not in _run_command()


def test_opencode_run_command_passes_user_when_set(agent_user_set):
    assert _run_command()["user"] == "ce-agent"


def test_opencode_run_command_privileged_exception(agent_user_set):
    assert "user" not in _run_command(as_agent_user=False)


def test_opencode_inspection_respects_metadata_flag(agent_user_set, monkeypatch, tmp_path):
    """run_inspection forwards raw_provider_metadata.run_as_agent_user=False."""
    import types

    captured = {}

    def fake_run_command(
        cmd, *, cwd, timeout, env_overrides=None, as_agent_user=True, hard_kill=False,
    ):
        captured["as_agent_user"] = as_agent_user
        return subprocess.CompletedProcess(cmd, 0, stdout="{}", stderr="")

    harness = OpenCodeHarness(
        runner=_RecordingRunner(),
        architect_config=types.SimpleNamespace(stage_project_skills=False),
    )
    monkeypatch.setattr(harness, "_run_command", fake_run_command)
    monkeypatch.setattr(harness, "_verify_binary_if_needed", lambda: None)
    harness.run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=str(tmp_path),
            prompt="inspect",
            permissions=HarnessPermissionPolicy.autonomous_policy(),
            max_turns=1,
            raw_provider_metadata={"run_as_agent_user": False},
        )
    )
    assert captured["as_agent_user"] is False


# ---------------------------------------------------------------------------
# subprocess_service (Claude subprocess runners + inspection plumbing)
# ---------------------------------------------------------------------------

def test_service_agent_user_kwargs(agent_user_set):
    assert subprocess_service._agent_user_kwargs() == {"user": "ce-agent"}


def test_service_agent_user_kwargs_empty(agent_user_unset):
    assert subprocess_service._agent_user_kwargs() == {}


def test_run_inspection_agent_threads_privileged_flag(monkeypatch):
    """run_inspection_agent(run_as_agent_user=False) reaches the runner."""
    seen = {}

    def fake_runner(workspace, **kwargs):
        seen.update(kwargs)
        return {"success": True, "proposed_change": "ok", "error": None}

    monkeypatch.setattr(
        subprocess_service,
        "_run_claude_inspection_agent_subprocess",
        fake_runner,
    )
    result = subprocess_service.run_inspection_agent(
        workspace="/tmp/ws",
        prompt="audit",
        coding_harness="claude",
        run_as_agent_user=False,
    )
    assert result["success"] is True
    assert seen["run_as_agent_user"] is False


def test_claude_harness_defaults_run_as_agent_user_true():
    seen = {}

    def fake_runner(workspace, **kwargs):
        seen.update(kwargs)
        return {"success": True, "proposed_change": "ok", "error": None}

    harness = ClaudeCodeHarness(inspection_runner=fake_runner)
    harness.run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace="/tmp/ws",
            prompt="inspect",
            permissions=HarnessPermissionPolicy.autonomous_policy(),
            max_turns=1,
            raw_provider_metadata={},
        )
    )
    assert seen["run_as_agent_user"] is True


def test_run_integrity_passes_privileged_exception(monkeypatch):
    """The run-integrity inspector spawns with run_as_agent_user=False."""
    from src.engine.optimizer.meta import meta_reflection, run_integrity

    seen = {}

    def fake_reflection_agent(**kwargs):
        seen.update(kwargs)
        return None  # short-circuits the rest of the inspection

    monkeypatch.setattr(run_integrity, "run_reflection_agent", fake_reflection_agent)
    run_integrity.run_integrity_inspection(
        workspace="/tmp/ws",
        method="ee",
        num_iterations=1,
        best_valset_score=None,
        seed_test_score=None,
        best_candidate_test_score=None,
        iteration_indices=[1],
        testset_path="/opt/holdout/x_test.json",
    )
    assert seen["run_as_agent_user"] is False
    # ...and the shared driver forwards it to run_inspection_agent.
    forwarded = {}

    def fake_inspection_agent(**kwargs):
        forwarded.update(kwargs)
        return {"success": True, "proposed_change": "report", "error": None}

    monkeypatch.setattr(
        subprocess_service, "run_inspection_agent", fake_inspection_agent
    )
    monkeypatch.setattr(subprocess_service, "write_file", lambda *a, **k: None)
    meta_reflection.run_reflection_agent(
        workspace="/tmp/ws",
        prompt="p",
        label="test",
        num_iterations=1,
        run_as_agent_user=False,
        out_path="/tmp/out.md",
    )
    assert forwarded["run_as_agent_user"] is False


# ---------------------------------------------------------------------------
# runner entrypoint: holdout sequestration
# ---------------------------------------------------------------------------

@pytest.fixture
def seed_workspace(tmp_path, monkeypatch):
    """A squashed one-commit repo carrying a tracked testset, like the runner's."""
    import entrypoint

    ws = tmp_path / "workspace"
    (ws / "data").mkdir(parents=True)
    (ws / "data" / "bench_test.json").write_text('[{"q": "secret"}]')
    (ws / "program.py").write_text("print('hi')\n")
    git = ["git", "-C", str(ws)]
    subprocess.run(["git", "init", "-q", "-b", "main", str(ws)], check=True)
    subprocess.run([*git, "-c", "user.email=t@t", "-c", "user.name=t", "add", "-A"], check=True)
    subprocess.run(
        [*git, "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
        check=True,
    )
    subprocess.run([*git, "branch", "seed"], check=True)

    monkeypatch.setattr(entrypoint, "WORKSPACE", ws)
    monkeypatch.setattr(entrypoint, "HOLDOUT_DIR", tmp_path / "holdout")
    return entrypoint, ws


def test_sequester_holdout_evicts_file_and_blob(seed_workspace):
    entrypoint, ws = seed_workspace
    git = ["git", "-C", str(ws)]
    blob = subprocess.run(
        [*git, "rev-parse", "HEAD:data/bench_test.json"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()

    overrides = entrypoint.sequester_holdout(
        {"testset_path": "data/bench_test.json"}
    )

    dst = entrypoint.HOLDOUT_DIR / "bench_test.json"
    assert overrides == [f"testset_path={dst}"]
    assert not (ws / "data" / "bench_test.json").exists()
    assert dst.read_text() == '[{"q": "secret"}]'
    assert stat.S_IMODE(dst.stat().st_mode) == 0o600
    # The blob is unreachable AND gone from the object store entirely.
    assert subprocess.run([*git, "cat-file", "-e", blob], capture_output=True).returncode != 0
    show = subprocess.run(
        [*git, "show", "HEAD:data/bench_test.json"], capture_output=True, text=True
    )
    assert show.returncode != 0
    # Working tree stays clean for the seed step's clean-tree expectations.
    status = subprocess.run(
        [*git, "status", "--porcelain"], capture_output=True, text=True, check=True
    ).stdout.strip()
    assert status == ""


def test_sequester_holdout_no_testset_is_noop(seed_workspace):
    """A config with no testset_path leaves the workspace untouched.

    (`sequester_holdout` now takes the resolved config directly rather than
    re-deriving it from --set pairs; `effective_config` is the single place
    that turns input.py + CE_SET into the submitted dict.)
    """
    entrypoint, ws = seed_workspace
    assert entrypoint.sequester_holdout({}) == []
    assert (ws / "data" / "bench_test.json").exists()


def test_sequester_holdout_missing_file_aborts(seed_workspace):
    entrypoint, _ = seed_workspace
    with pytest.raises(SystemExit):
        entrypoint.sequester_holdout({"testset_path": "data/nope.json"})


# ---------------------------------------------------------------------------
# runner entrypoint: git trust for the agent user (safe.directory)
# ---------------------------------------------------------------------------

def test_trust_workspace_writes_the_system_gitconfig(seed_workspace, tmp_path, monkeypatch):
    """The workspace is root-owned and only chmod'ed open; git's
    `safe.directory` check is about OWNERSHIP, so without trust the agent user
    hits `dubious ownership` on every git command. The trust goes into the
    SYSTEM config — root-owned, read for every uid, not the agent-writable
    `$HOME/.gitconfig`."""
    entrypoint, ws = seed_workspace
    system_cfg = tmp_path / "etc-gitconfig"
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", str(system_cfg))  # git >= 2.32
    entrypoint.trust_workspace_for_agent()
    assert f"directory = {ws}" in system_cfg.read_text()
    listed = subprocess.run(
        ["git", "config", "--system", "--get-all", "safe.directory"],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    assert str(ws) in listed
    # Nothing leaks into the per-user config the agent could edit.
    assert not (tmp_path / ".gitconfig").exists()


def test_trust_workspace_precedes_sequestration_in_main_order():
    """The holdout fence is a separate, later stage: trusting the workspace
    repo must not touch `/opt/holdout` (not a repo; root-only modes)."""
    import inspect

    import entrypoint

    src = inspect.getsource(entrypoint.main)
    assert src.index("open_workspace_permissions()") < src.index("trust_workspace_for_agent()")
    assert src.index("trust_workspace_for_agent()") < src.index("sequester_holdout(")
    assert src.index("sequester_holdout(") < src.index("run_preflight()")


def test_preflight_agent_git_inert_when_unset(agent_user_unset):
    import preflight

    assert preflight._check_agent_git("/nonexistent")["ok"] is True


def test_preflight_agent_git_reports_dubious_ownership(agent_user_set, seed_workspace, monkeypatch):
    import preflight

    _, ws = seed_workspace

    def fake_run(cmd, **kwargs):
        assert kwargs.get("user") == "ce-agent" and cmd[:3] == ["git", "-C", str(ws)]
        return subprocess.CompletedProcess(
            cmd, 128, stdout="",
            stderr=f"fatal: detected dubious ownership in repository at '{ws}'",
        )

    monkeypatch.setattr(preflight.subprocess, "run", fake_run)
    report = preflight._check_agent_git(str(ws))
    assert report["ok"] is False and "dubious ownership" in report["detail"]


def test_preflight_agent_git_ok_when_head_resolves(agent_user_set, seed_workspace, monkeypatch):
    import preflight

    _, ws = seed_workspace
    monkeypatch.setattr(
        preflight.subprocess, "run",
        lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="abc123def456\n", stderr=""),
    )
    report = preflight._check_agent_git(str(ws))
    assert report == {"configured": True, "ok": True, "detail": "git as 'ce-agent' sees HEAD abc123def456"}
