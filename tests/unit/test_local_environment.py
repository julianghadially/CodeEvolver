"""Tests for LocalEnvironment lifecycle on the host.

Validates the lifecycle operations (repo lock acquired on start and released
on stop, pre-push hook installed, .gitignore entries appended, refusal to
start when the path isn't a git repo), plus the uv-based dependency install
path (gated on `uv` being on PATH; these reach the network).
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from src.local.local_environment import LocalEnvironment
from src.local.repo_lock import RepoLockedError

# Tiny, pure-Python, dependency-free package used to exercise real installs.
_TINY_PKG = "iniconfig==2.0.0"
uv_required = pytest.mark.skipif(
    shutil.which("uv") is None, reason="uv not on PATH"
)


def _init_git(path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=path, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=path, check=True,
    )


def test_start_refuses_when_path_missing(tmp_path):
    env = LocalEnvironment(repo_path=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        env.start(use_venv=False)


def test_start_refuses_when_not_a_git_repo(tmp_path):
    env = LocalEnvironment(repo_path=tmp_path)
    with pytest.raises(RuntimeError, match="not a git repository"):
        env.start(use_venv=False)


def test_start_installs_hook_and_gitignore(tmp_path):
    _init_git(tmp_path)
    env = LocalEnvironment(repo_path=tmp_path)
    env.start(use_venv=False)
    try:
        hook = tmp_path / ".git" / "hooks" / "pre-push"
        assert hook.exists()
        assert hook.stat().st_mode & 0o111  # executable
        assert "BRANCH GUARD" in hook.read_text()

        gitignore = (tmp_path / ".gitignore").read_text().splitlines()
        for required in [".venv", "env", ".env", ".claude/", ".opencode/"]:
            assert required in gitignore
    finally:
        env.stop()


def test_concurrent_start_on_same_repo_raises(tmp_path):
    _init_git(tmp_path)
    first = LocalEnvironment(repo_path=tmp_path)
    second = LocalEnvironment(repo_path=tmp_path)
    first.start(use_venv=False)
    try:
        with pytest.raises(RepoLockedError):
            second.start(use_venv=False)
    finally:
        first.stop()


def test_stop_releases_lock(tmp_path):
    _init_git(tmp_path)
    first = LocalEnvironment(repo_path=tmp_path)
    first.start(use_venv=False)
    first.stop()
    second = LocalEnvironment(repo_path=tmp_path)
    second.start(use_venv=False)  # must not raise
    second.stop()


@uv_required
def test_uv_creates_venv_and_installs_requirements(tmp_path):
    _init_git(tmp_path)
    (tmp_path / "requirements.txt").write_text(_TINY_PKG + "\n")
    env = LocalEnvironment(repo_path=tmp_path)
    env.start(use_venv=True)
    try:
        venv_python = tmp_path / "env" / "bin" / "python"
        assert venv_python.exists()
        # The requirement is importable from the freshly built venv.
        result = subprocess.run(
            [str(venv_python), "-c", "import iniconfig"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0, result.stderr
    finally:
        env.stop()


@uv_required
def test_uv_installs_into_pipless_uv_venv(tmp_path):
    """Reproduces the phantom-wiki failure mode: a venv created by `uv venv`
    ships no pip, so the old `<venv>/bin/pip install` path errored with "pip
    missing." uv installs into that same venv fine because it targets the
    interpreter, not an in-venv pip.
    """
    (tmp_path / "requirements.txt").write_text(_TINY_PKG + "\n")
    venv_path = tmp_path / "env"
    subprocess.run(["uv", "venv", str(venv_path)], check=True, capture_output=True)
    # Precondition: the uv-created venv has no pip — the original bug's trigger.
    assert not (venv_path / "bin" / "pip").exists()

    env = LocalEnvironment(repo_path=tmp_path)
    env._install_client_requirements()  # must succeed despite no pip in venv

    result = subprocess.run(
        [str(venv_path / "bin" / "python"), "-c", "import iniconfig"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr


def test_gitignore_entries_are_not_duplicated(tmp_path):
    _init_git(tmp_path)
    (tmp_path / ".gitignore").write_text(".venv\nenv\n")
    env = LocalEnvironment(repo_path=tmp_path)
    env.start(use_venv=False)
    try:
        lines = (tmp_path / ".gitignore").read_text().splitlines()
        # Two original entries + only the new staging/runtime entries appended.
        assert lines.count(".venv") == 1
        assert lines.count("env") == 1
        assert ".env" in lines
        assert ".claude/" in lines
        assert ".opencode/" in lines
    finally:
        env.stop()
