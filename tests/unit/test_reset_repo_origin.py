"""Unit tests for reset_repo's origin-removal contamination guard.

`reset_repo.py -D` removes the `origin` remote so the in-place research agent
can no longer fetch the `codeevolver-*` branches prior `-S` runs pushed to
GitHub. To keep saves working across cycles, `-D` stashes the origin URL into
`git config codeevolver.origin-url` and `-S` (`push_top_branches`) falls back to
that stashed URL when `origin` is absent.

These tests exercise that with a local bare repo standing in for GitHub — no
network, no GitHub App.
"""
from __future__ import annotations

import os
import subprocess

import pytest

import scripts.reset_repo as reset_repo
from scripts.reset_repo import (
    STASHED_ORIGIN_KEY,
    _git_config_get,
    _resolve_push_target,
    push_top_branches,
    remove_origin,
)


def _git(workspace, *args, check=True):
    result = subprocess.run(
        ["git", *args], cwd=workspace, capture_output=True, text=True
    )
    if check and result.returncode != 0:
        raise AssertionError(f"git {' '.join(args)} failed: {result.stderr}")
    return result


@pytest.fixture
def repo_with_origin(tmp_path):
    """A working clone whose `origin` is a local bare repo (stands in for GitHub).

    Returns (workspace_path, bare_remote_path).
    """
    bare = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", str(bare))

    work = tmp_path / "work"
    work.mkdir()
    _git(work, "init")
    _git(work, "config", "user.email", "t@t.test")
    _git(work, "config", "user.name", "Test")
    (work / "README.md").write_text("hi\n")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")
    _git(work, "remote", "add", "origin", str(bare))
    # An admitted-candidate branch, as a run would produce.
    _git(work, "branch", "codeevolver-20260101-aaa")
    return str(work), str(bare)


def test_resolve_push_target_prefers_origin(repo_with_origin):
    work, _ = repo_with_origin
    assert _resolve_push_target(work) == "origin"


def test_remove_origin_stashes_url_and_removes_remote(repo_with_origin):
    work, bare = repo_with_origin
    assert _git_config_get(work, "remote.origin.url") == bare

    remove_origin(work, dry_run=False)

    # Remote gone, URL stashed.
    assert _git_config_get(work, "remote.origin.url") is None
    assert _git_config_get(work, STASHED_ORIGIN_KEY) == bare


def test_resolve_push_target_falls_back_to_stash(repo_with_origin):
    work, bare = repo_with_origin
    remove_origin(work, dry_run=False)
    # origin is gone; the saved URL becomes the push target.
    assert _resolve_push_target(work) == bare


def test_push_top_branches_uses_stashed_url_after_removal(repo_with_origin):
    work, bare = repo_with_origin
    remove_origin(work, dry_run=False)

    push_top_branches(work, ["codeevolver-20260101-aaa"], dry_run=False)

    # The branch reached the bare remote even though `origin` was removed.
    refs = _git(bare, "branch", "--list").stdout
    assert "codeevolver-20260101-aaa" in refs


def test_remove_origin_is_idempotent(repo_with_origin):
    work, bare = repo_with_origin
    remove_origin(work, dry_run=False)
    # Second call: nothing to remove, stash preserved.
    remove_origin(work, dry_run=False)
    assert _git_config_get(work, STASHED_ORIGIN_KEY) == bare
    assert _git_config_get(work, "remote.origin.url") is None


def test_remove_origin_dry_run_does_not_mutate(repo_with_origin):
    work, bare = repo_with_origin
    remove_origin(work, dry_run=True)
    # Remote still present, nothing stashed.
    assert _git_config_get(work, "remote.origin.url") == bare
    assert _git_config_get(work, STASHED_ORIGIN_KEY) is None


def test_push_top_branches_warns_when_no_target(tmp_path, capsys):
    """No origin and no stash → cannot push, but does not raise."""
    work = tmp_path / "bare_work"
    work.mkdir()
    _git(work, "init")
    _git(work, "config", "user.email", "t@t.test")
    _git(work, "config", "user.name", "Test")
    (work / "f").write_text("x")
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "init")

    push_top_branches(str(work), ["codeevolver-x"], dry_run=False)
    out = capsys.readouterr().out
    assert "cannot push" in out.lower()
