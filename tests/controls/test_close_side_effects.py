"""Copyright © 2026 440 Labs LLC

The loop's own post-close side effects, against a real git repo.

Helpers in `close_effects` that touch the client repo after the architect
exits: `checkout_registered_branch` + `commit_codeevolver_md_if_modified`
(the analysis agent owns codeevolver.md; the loop commits its update on the
REGISTERED branch) and `write_attempted_diff` (which materializes the change
an iteration tried, so the analysis agent — which has no Bash tool — can
describe it).

Two properties only show up against real git:

- the summary commits onto the REGISTERED branch, not whatever branch happens
  to be checked out (otherwise it lands on another lineage and becomes that
  lineage's start state);
- it runs BEFORE `state.record_iteration` and the push, so its commit is in
  both the record and origin, which is what the next iteration inherits.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.close_effects import (
    checkout_registered_branch,
    commit_codeevolver_md_if_modified,
    write_attempted_diff,
)
from src.engine.services.subprocess_service import git_tree_signature


def _git(ws, *args) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ws, capture_output=True, text=True)


def _sha(ws, ref="HEAD") -> str:
    return _git(ws, "rev-parse", ref).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    ws = tmp_path / "repo"
    ws.mkdir()
    _git(ws, "init", "-q")
    _git(ws, "config", "user.email", "t@t")
    _git(ws, "config", "user.name", "t")
    (ws / "program.py").write_text("K = 7\n")
    (ws / "codeevolver.md").write_text("# Architecture\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "seed")
    return ws


@pytest.fixture()
def memory_env(tmp_path, monkeypatch):
    mem = tmp_path / "memory"
    mem.mkdir()
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(mem))
    monkeypatch.setenv("CODEEVOLVER_HISTORY_DIR", str(tmp_path / "history"))
    return mem


def _ctx(repo, *, branch, parent_sha, iteration=5) -> IterationContext:
    ctx = IterationContext(iteration_idx=iteration, parent_locked=True)
    ctx.child_branch = branch
    ctx.parent_sha = parent_sha
    ctx.parent_tree = git_tree_signature(str(repo), parent_sha)
    ctx.close_sha = _sha(repo, branch)
    ctx.close_tree = git_tree_signature(str(repo), ctx.close_sha)
    ctx.changes_landed = True
    return ctx


# --- codeevolver.md: analysis-agent update, committed by the loop -----------


def test_update_commits_onto_the_registered_branch_not_the_checked_out_one(
    repo, memory_env
):
    """The iteration-5 defect: two branches, HEAD on the wrong one.

    The architect explored on `lm-rerank`, registered it, but left HEAD on
    `react-gap`. The loop now puts the workspace on the REGISTERED branch
    before the analysis agent writes, so its codeevolver.md update lands
    where the program lives.
    """
    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-lm-rerank")
    (repo / "program.py").write_text("K = 20\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rerank")
    registered_tip = _sha(repo)

    # The architect leaves HEAD somewhere else entirely.
    _git(repo, "checkout", "-q", "-b", "codeevolver-react-gap", parent_sha)
    other_tip_before = _sha(repo)

    ctx = _ctx(repo, branch="codeevolver-lm-rerank", parent_sha=parent_sha)
    ctx.close_sha, ctx.close_tree = registered_tip, git_tree_signature(
        str(repo), registered_tip
    )

    checkout_registered_branch(ctx, workspace=str(repo))
    # ... the analysis agent rewrites the description in place ...
    (repo / "codeevolver.md").write_text("# Architecture\n\nNow with a reranker.\n")
    committed = commit_codeevolver_md_if_modified(
        ctx, workspace=str(repo), iteration_idx=5
    )

    assert committed is True
    assert _sha(repo, "codeevolver-react-gap") == other_tip_before, (
        "the unrelated branch must not move"
    )
    new_tip = _sha(repo, "codeevolver-lm-rerank")
    assert new_tip != registered_tip, "the registered branch got the commit"
    body = _git(repo, "show", f"{new_tip}:codeevolver.md").stdout
    assert "Now with a reranker." in body
    assert "Change summary (iteration" not in body
    assert "Please remove change summaries" not in body


def test_no_commit_when_the_agent_left_the_doc_alone(repo, memory_env, caplog):
    import logging

    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    (repo / "program.py").write_text("K = 20\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rerank")
    tip = _sha(repo)

    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)
    with caplog.at_level(logging.WARNING):
        committed = commit_codeevolver_md_if_modified(
            ctx, workspace=str(repo), iteration_idx=5
        )

    assert committed is False
    assert _sha(repo, "codeevolver-x") == tip, "nothing to commit, no commit"
    assert "left codeevolver.md untouched" in caplog.text


def test_no_warning_when_nothing_landed(repo, memory_env, caplog):
    """codeevolver.md documents the program; a reverted iteration changed none."""
    import logging

    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    tip = _sha(repo)

    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)
    ctx.changes_landed = False
    with caplog.at_level(logging.WARNING):
        committed = commit_codeevolver_md_if_modified(
            ctx, workspace=str(repo), iteration_idx=5
        )

    assert committed is False
    assert _sha(repo, "codeevolver-x") == tip
    assert "left codeevolver.md untouched" not in caplog.text


def test_stray_program_edits_by_the_analysis_agent_are_discarded(repo, memory_env):
    """The agent is told to touch only codeevolver.md; an unreviewed program
    edit must never become the next iteration's start state."""
    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    (repo / "program.py").write_text("K = 20\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "rerank")

    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)
    (repo / "codeevolver.md").write_text("# Architecture\n\nK is 20 now.\n")
    (repo / "program.py").write_text("K = 99  # stray\n")

    committed = commit_codeevolver_md_if_modified(
        ctx, workspace=str(repo), iteration_idx=5
    )

    assert committed is True
    assert (repo / "program.py").read_text() == "K = 20\n"
    assert _git(repo, "status", "--porcelain").stdout.strip() == ""
    tip = _sha(repo, "codeevolver-x")
    assert "K is 20 now." in _git(repo, "show", f"{tip}:codeevolver.md").stdout


# --- attempted-change diff ---------------------------------------------------


def test_attempted_diff_captures_a_change_that_was_never_committed(
    repo, memory_env
):
    """The evidence a non-landing iteration's change summary is built from.

    `run_eval` writes a real tree object for an eval of the working tree, so
    the reverted code is still diffable even though no commit ever held it.
    """
    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)

    (repo / "program.py").write_text("K = 20  # tried, then reverted\n")
    ctx.best_subsample_tree = git_tree_signature(str(repo))
    _git(repo, "checkout", "-q", "--", "program.py")

    path = write_attempted_diff(ctx, workspace=str(repo))

    assert path is not None
    body = open(path).read()
    assert "tried, then reverted" in body
    assert "program.py" in body


def test_attempted_diff_is_none_when_nothing_was_tried(repo, memory_env):
    parent_sha = _sha(repo)
    ctx = _ctx(repo, branch="master", parent_sha=parent_sha)
    ctx.best_subsample_tree = ctx.parent_tree  # only the baseline was evaluated

    assert write_attempted_diff(ctx, workspace=str(repo)) is None


def test_attempted_diff_is_truncated_not_unbounded(repo, memory_env, monkeypatch):
    """A whole-file rewrite must not put megabytes into the agent's prompt."""
    monkeypatch.setattr(
        "src.engine.optimizer.v3.close_effects.MAX_ATTEMPTED_DIFF_BYTES", 500
    )
    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)

    (repo / "program.py").write_text("\n".join(f"LINE_{i} = {i}" for i in range(2000)))
    ctx.best_subsample_tree = git_tree_signature(str(repo))
    _git(repo, "checkout", "-q", "--", "program.py")

    path = write_attempted_diff(ctx, workspace=str(repo))

    body = open(path).read()
    assert "[diff truncated by CodeEvolver]" in body
    assert len(body) < 1000


def test_attempted_diff_survives_a_git_failure(repo, memory_env):
    ctx = _ctx(repo, branch="master", parent_sha=_sha(repo))
    ctx.best_subsample_tree = "0" * 40  # object does not exist

    assert write_attempted_diff(ctx, workspace=str(repo)) is None


# --- the record must not claim an attribution finalize() would not make ------


def test_reclosing_after_the_doc_commit_keeps_the_record_honest(repo, memory_env):
    """After the codeevolver.md commit the closing tree differs from every
    evaluated tree by that one non-program file. The record never claimed the
    closing commit was measured, so nothing degrades: `subsample_best` is
    unchanged, `changes_landed` is unchanged, and a by-content lookup of the
    new tip still finds the program's score as program-equivalent."""
    from src.engine.optimizer.v3.program_paths import diff_is_non_program_only

    parent_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-x")
    (repo / "program.py").write_text("K = 20\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "raise k")

    ctx = _ctx(repo, branch="codeevolver-x", parent_sha=parent_sha)
    ctx.initial_subsample_score = 0.50
    ctx.record_eval(parent_sha, 0.50, tree=ctx.parent_tree)
    ctx.record_eval(parent_sha, 0.72, tree=ctx.close_tree)
    equivalent = lambda a, b: diff_is_non_program_only(str(repo), a, b)
    ctx.finalize(equivalent)
    assert ctx.score_for_content(ctx.close_tree, equivalent) == (0.72, "exact_tree")

    (repo / "codeevolver.md").write_text("# Architecture\n\nK = 20.\n")
    assert commit_codeevolver_md_if_modified(ctx, workspace=str(repo), iteration_idx=5)

    # What the loop does after the commit.
    new_close = _sha(repo, "codeevolver-x")
    assert new_close != ctx.close_sha
    ctx.close_sha = new_close
    ctx.close_tree = git_tree_signature(str(repo), new_close)
    ctx.finalize(equivalent)

    assert ctx.best_subsample_score == 0.72
    assert ctx.changes_landed is True
    assert ctx.score_for_content(ctx.close_tree, equivalent) == (0.72, "program_equivalent")


def test_constraint_revert_rescores_instead_of_reusing_the_baseline(repo):
    """`initial_subsample_score` is not necessarily the parent's score.

    Branch navigation is ungated, so `run_eval(train)` on branch A → checkout B
    → edit → close on B is permitted. The derived parent is then B's tip while
    the session's first train eval belongs to A. Hand-writing
    `final_subsample_score = initial_subsample_score` with
    `final_score_source = "exact_tree"` asserts that eval ran on the parent's
    content when it did not; re-closing derives it from the tree on disk.
    """
    from src.engine.optimizer.v3.close_effects import force_revert_to_parent
    from src.engine.optimizer.v3.program_paths import diff_is_non_program_only

    branch_a_sha = _sha(repo)
    _git(repo, "checkout", "-q", "-b", "branch-b")
    (repo / "program.py").write_text("K = 99\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "different lineage")
    parent_sha = _sha(repo)

    ctx = _ctx(repo, branch="branch-b", parent_sha=parent_sha)
    # The baseline was measured on branch A's content, not on the parent's.
    ctx.initial_subsample_score = 0.50
    ctx.record_eval(
        branch_a_sha, 0.50, tree=git_tree_signature(str(repo), branch_a_sha)
    )

    # The violating change is committed on the child branch; the constraint
    # gate force-reverts to the parent and re-closes.
    (repo / "program.py").write_text("ANSWERS = {'q1': 'a1'}\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "iter 5: hard-code answers")
    violating_sha = _sha(repo)
    ctx.close_sha = violating_sha
    assert force_revert_to_parent(
        ctx, workspace=str(repo), violated_constraint="never hardcode answers"
    ) is True
    equivalent = lambda a, b: diff_is_non_program_only(str(repo), a, b)

    # The branch sits on an empty marker commit above the parent: same
    # content, visible `Revert iteration 5` line in `git log`, no reason.
    assert _sha(repo, "branch-b") == ctx.close_sha
    assert ctx.close_sha not in (parent_sha, violating_sha)
    assert _sha(repo, "HEAD~1") == parent_sha
    assert git_tree_signature(str(repo), ctx.close_sha) == ctx.parent_tree
    body = _git(repo, "log", "-1", "--format=%B").stdout
    assert "Revert iteration 5" in body
    assert "Violated constraint: never hardcode answers" in body
    assert (repo / "program.py").read_text() == "K = 99\n"

    assert ctx.changes_landed is False
    assert ctx.score_for_content(ctx.close_tree, equivalent) == (None, "none"), (
        "the parent's content was never evaluated — say so rather than "
        "borrowing branch A's number"
    )
    # The record carries the iteration's best reading regardless — branch
    # A's number is a fact about what was measured, not about the parent.
    assert ctx.best_subsample_score == 0.50


def test_revert_to_comparison_point_resets_the_branch_and_recloses(repo):
    """specs/archive-public/comparison_credits.md: evo's close reverts post-comparison
    program edits back to the auto-committed comparison-run commit."""
    from src.engine.optimizer.v3.close_effects import revert_to_comparison_point
    from src.engine.optimizer.v3.program_paths import diff_is_non_program_only

    ws = str(repo)
    _git(repo, "checkout", "-q", "-b", "codeevolver-child")
    (repo / "program.py").write_text("K = 8\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "iter 1: auto-commit before comparison eval")
    cmp_sha = _sha(repo)
    cmp_tree = git_tree_signature(ws, cmp_sha)
    # Post-comparison drift: another program edit, committed.
    (repo / "program.py").write_text("K = 9\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "post-comparison drift")

    ctx = IterationContext(
        iteration_idx=1,
        parent_locked=True,
        child_branch="codeevolver-child",
        parent_sha=_sha(repo, "HEAD~2"),
        parent_tree=git_tree_signature(ws, _sha(repo, "HEAD~2")),
        close_sha=_sha(repo),
        close_tree=git_tree_signature(ws),
    )
    ctx.initial_subsample_score = 0.4
    ctx.record_eval(cmp_sha, 0.9, eval_sample="comparison", n_rows=3, tree=cmp_tree)

    assert revert_to_comparison_point(ctx, workspace=ws, target_sha=cmp_sha) is True
    assert _sha(repo) == cmp_sha
    assert (repo / "program.py").read_text() == "K = 8\n"
    assert ctx.close_sha == cmp_sha
    assert ctx.close_tree == cmp_tree
    assert ctx.changes_landed is True  # K=8 still differs from the parent


def test_revert_to_comparison_point_fails_open_on_a_bad_target(repo):
    from src.engine.optimizer.v3.close_effects import revert_to_comparison_point

    ctx = IterationContext(
        iteration_idx=1, parent_locked=True, child_branch="codeevolver-missing"
    )
    assert (
        revert_to_comparison_point(
            ctx, workspace=str(repo), target_sha="0000000000000000"
        )
        is False
    )
