"""Tests for `subprocess_service.git_tree_signature` / `git_commit_trees`.

Replaces `test_git_worktree_signature.py`. The old helper folded HEAD into
its hash, so committing unchanged content changed the signature — it answered
"did anything move?". These helpers answer "is this the same CODE?", which is
what an eval record needs: `run_eval` stamps HEAD's sha, but the evaluator
imports from the working tree, so an eval of uncommitted edits is recorded
under the parent commit. Commit-invariance is what lets the close checks
recognise the architect's commit as the code they already scored.
"""
from __future__ import annotations

import subprocess

from src.engine.services.subprocess_service import (
    git_commit_trees,
    git_tree_signature,
)


def _git(workspace, *args):
    subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)


def _sha(workspace, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=workspace, capture_output=True, text=True
    ).stdout.strip()


def _init_repo(workspace):
    _git(workspace, "init", "-q")
    _git(workspace, "config", "user.email", "t@t")
    _git(workspace, "config", "user.name", "t")
    (workspace / "a.py").write_text("x = 1\n")
    _git(workspace, "add", "-A")
    _git(workspace, "commit", "-q", "-m", "init")


def test_tree_is_commit_invariant(tmp_path):
    """THE load-bearing property: committing does not change the tree hash.

    Reproduces the live-run shape — edit, eval (capture tree), commit — and
    asserts the eval's tree still identifies the new commit even though the
    sha moved.
    """
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")  # edit, uncommitted

    at_eval_time = git_tree_signature(tmp_path)
    sha_before = _sha(tmp_path)

    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "architect commits after evaluating")

    assert _sha(tmp_path) != sha_before, "commit should have moved HEAD"
    assert git_tree_signature(tmp_path) == at_eval_time
    assert git_tree_signature(tmp_path, "HEAD") == at_eval_time


def test_tree_stable_on_clean_tree(tmp_path):
    _init_repo(tmp_path)
    assert git_tree_signature(tmp_path) == git_tree_signature(tmp_path)


def test_tree_changes_on_uncommitted_edit(tmp_path):
    _init_repo(tmp_path)
    before = git_tree_signature(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    assert git_tree_signature(tmp_path) != before


def test_tree_changes_on_untracked_file(tmp_path):
    _init_repo(tmp_path)
    before = git_tree_signature(tmp_path)
    (tmp_path / "b.py").write_text("y = 1\n")
    assert git_tree_signature(tmp_path) != before


def test_tree_ignores_gitignored_file(tmp_path):
    """`add -A` honours .gitignore, so ignored files are not part of identity.

    Matters because the synthetic test fixtures gitignore `.env` and the
    holdout testset — churn there must not invalidate an eval.
    """
    _init_repo(tmp_path)
    (tmp_path / ".gitignore").write_text(".env\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "ignore .env")

    before = git_tree_signature(tmp_path)
    (tmp_path / ".env").write_text("SECRET=1\n")
    assert git_tree_signature(tmp_path) == before


def test_tree_signature_is_non_mutating(tmp_path):
    """Guards the throwaway-index trick: the real repo must be untouched."""
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 9\n")
    (tmp_path / "untracked.py").write_text("z = 1\n")

    def _status():
        return subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=tmp_path, capture_output=True, text=True,
        ).stdout

    status_before, head_before = _status(), _sha(tmp_path)

    # Read the index mtime immediately either side of the call and assert
    # before running any other git command: `git status` opportunistically
    # rewrites the real index to refresh its stat cache, which would make
    # this assertion about the test rather than the helper.
    index = tmp_path / ".git" / "index"
    index_mtime_before = index.stat().st_mtime_ns
    git_tree_signature(tmp_path)
    assert index.stat().st_mtime_ns == index_mtime_before

    assert _status() == status_before
    assert _sha(tmp_path) == head_before


def test_tree_of_ref(tmp_path):
    _init_repo(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c2")

    assert git_tree_signature(tmp_path, "HEAD~1") != git_tree_signature(tmp_path, "HEAD")


def test_tree_of_unknown_ref_is_none(tmp_path):
    _init_repo(tmp_path)
    assert git_tree_signature(tmp_path, "no-such-branch") is None


def test_returns_none_on_non_repo(tmp_path):
    # None, not "?": a self-equal sentinel would make two failed reads compare
    # equal and let an unevaluated HEAD pass the close check.
    assert git_tree_signature(str(tmp_path / "does_not_exist")) is None
    assert git_commit_trees(str(tmp_path / "does_not_exist")) == []


def test_commit_trees_newest_first_and_since_range(tmp_path):
    _init_repo(tmp_path)
    c1 = _sha(tmp_path)
    (tmp_path / "a.py").write_text("x = 2\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c2")
    c2 = _sha(tmp_path)
    (tmp_path / "a.py").write_text("x = 3\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "c3")
    c3 = _sha(tmp_path)

    assert [sha for sha, _ in git_commit_trees(tmp_path)] == [c3, c2, c1]
    assert [sha for sha, _ in git_commit_trees(tmp_path, since=c1)] == [c3, c2]

    # The tree paired with each commit is that commit's tree.
    trees = dict(git_commit_trees(tmp_path))
    assert trees[c3] == git_tree_signature(tmp_path, c3)


def test_commit_trees_falls_back_when_since_is_unreachable(tmp_path):
    """A rewritten/deleted `since` must not lose reconciliation entirely."""
    _init_repo(tmp_path)
    trees = git_commit_trees(tmp_path, since="0" * 40)
    assert [sha for sha, _ in trees] == [_sha(tmp_path)]
