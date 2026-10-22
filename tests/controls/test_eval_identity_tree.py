"""Content identity for eval records: the edit -> eval -> commit -> close flow.

`run_eval` stamps each eval with `git rev-parse HEAD`, but the evaluator
imports client code from the WORKING TREE, so an eval of uncommitted edits
carries the PARENT commit's sha. Closing therefore cannot ask "which evaluated
SHA is reachable from the branch?": a dirty-tree eval the architect ran and
then reverted keeps the parent's sha, which is trivially reachable, and would
mis-score or mis-categorise the iteration.

Engine 3 asks a content question instead. The iteration ended on `close_sha`;
its score is the eval whose TREE is program-equivalent to `close_tree`; and
"did anything change?" compares the parent tree to the close tree. Program
equivalence (`v3/program_paths.py`) is what lets the architect record its change
summary in codeevolver.md and close on that commit without a redundant eval.

`reconcile_eval_shas` remains, because the watchdogs and the audit trail
read record shas; it is not load-bearing for closing.
"""
from __future__ import annotations

import subprocess

import pytest

from src.engine.optimizer.v3.agent_tools import close_iteration_tool
from src.engine.optimizer.v3.errors import CloseIterationError
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.close_effects import anchor_iteration_state
from src.engine.optimizer.v3.program_paths import diff_is_non_program_only
from src.engine.services.subprocess_service import (
    git_commit_trees,
    git_is_ancestor,
    git_tree_signature,
)

_BRANCH = "codeevolver-20260101130000-main"
_REPO = "myrepo"


def _git(ws, *args):
    subprocess.run(["git", *args], cwd=ws, check=True, capture_output=True)


def _git_out(ws, *args) -> subprocess.CompletedProcess:
    """Run git without raising, returning the CompletedProcess.

    The gc/reachability tests need to assert on a git command's exit code and
    output, which `_git`'s `check=True` and discarded result cannot express.
    """
    return subprocess.run(
        ["git", *args], cwd=ws, capture_output=True, text=True
    )


def _sha(ws, ref="HEAD"):
    return subprocess.run(
        ["git", "rev-parse", ref], cwd=ws, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture()
def repo(tmp_path):
    """A repo on `_BRANCH` with one commit — the iteration's starting point."""
    _git(tmp_path, "init", "-q")
    _git(tmp_path, "config", "user.email", "t@t")
    _git(tmp_path, "config", "user.name", "t")
    (tmp_path / "judge.py").write_text("VERDICT = 'hedge'\n")
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "seed")
    _git(tmp_path, "checkout", "-q", "-b", _BRANCH)
    return tmp_path


def _ctx(repo) -> IterationContext:
    return IterationContext(
        iteration_idx=1,
        parent_locked=True,
        start_branch=_BRANCH,
        start_sha=_sha(repo),
        primary_branch_at_start=_BRANCH,
    )


def _live_run_shape(repo, ctx) -> str:
    """Parent eval, edit, child eval (same sha, new tree), commit.

    Returns the child commit's sha. Both evals are recorded against the
    parent sha, exactly as `run_eval` does on a dirty tree.
    """
    parent_sha = _sha(repo)
    ctx.record_eval(
        parent_sha, 0.825, eval_sample="train",
        branch=_BRANCH, tree=git_tree_signature(repo),
    )
    ctx.initial_subsample_score = 0.825

    (repo / "judge.py").write_text("VERDICT = 'decisive'\n")
    ctx.record_eval(
        parent_sha, 0.925, eval_sample="train",
        branch=_BRANCH, tree=git_tree_signature(repo),
    )

    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "iter 1: strengthen judge")
    return _sha(repo)


def _close_kwargs(ctx, repo, **overrides):
    kwargs = {
        "ctx": ctx,
        "workspace": str(repo),
        "register_branch": {
            "branch": _BRANCH, "was_primary": True, "is_primary": True,
        },
        "improvement_mode": "exploitation",
        "memory_updated": True,
    }
    kwargs.update(overrides)
    return kwargs


def test_edit_eval_commit_close_needs_no_second_eval(repo):
    """The control test for the whole bug: close accepts, no third eval."""
    ctx = _ctx(repo)
    child_sha = _live_run_shape(repo, ctx)
    assert {r["sha"] for r in ctx.eval_records} == {ctx.start_sha}, (
        "both evals should carry the PARENT sha — that is the bug's shape"
    )
    assert child_sha not in {r["sha"] for r in ctx.eval_records}

    result = close_iteration_tool(**_close_kwargs(ctx, repo))

    assert result["ok"] is True
    assert ctx.close_called is True
    # The point: still exactly two evals. No re-run was needed to close.
    assert len(ctx.eval_records) == 2


def _finalize(ctx, repo, close_sha):
    """Close the iteration the way the loop does: stamp the end state, then
    attribute a score to it by content."""
    ctx.close_sha = close_sha
    ctx.close_tree = git_tree_signature(str(repo), close_sha)
    ctx.parent_tree = git_tree_signature(str(repo), ctx.parent_sha)
    return ctx.finalize(lambda a, b: diff_is_non_program_only(str(repo), a, b))


def _head_score(ctx, repo):
    return ctx.score_for_content(
        ctx.close_tree, lambda a, b: diff_is_non_program_only(str(repo), a, b)
    )


def test_finalize_attributes_the_eval_to_the_commit_that_holds_it(repo):
    """A dirty-tree eval still identifies the commit its content became.

    The old sha-only walk needed `reconcile_eval_shas` to have run first or it
    would report a fall-back to the parent. Content matching does not: the
    eval's tree IS the commit's tree.
    """
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    child_sha = _live_run_shape(repo, ctx)
    ctx.child_branch = _BRANCH
    ctx.parent_sha = parent_sha

    result = _finalize(ctx, repo, child_sha)

    assert ctx.close_sha == child_sha
    assert result.best_subsample_score == 0.925
    assert _head_score(ctx, repo) == (0.925, "exact_tree")
    assert result.changes_landed is True


def test_finalize_needs_no_reconciliation_to_see_the_change(repo):
    """The regression this whole change exists to prevent.

    Previously, skipping reconciliation made `finalize()` walk back to the
    parent sha (which a dirty-tree eval carries and which is trivially
    reachable), so the iteration was recorded as skipped with a score from
    discarded code. Nothing here calls `reconcile_eval_shas`.
    """
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    child_sha = _live_run_shape(repo, ctx)
    ctx.child_branch = _BRANCH
    ctx.parent_sha = parent_sha

    result = _finalize(ctx, repo, child_sha)

    assert result.changes_landed is True
    assert result.best_subsample_score == 0.925
    assert _head_score(ctx, repo)[0] == 0.925


def test_doc_only_commit_closes_without_a_second_eval(repo):
    """The close_iteration skill tells the architect to record its change
    summary in codeevolver.md before closing. That commit changes the tree
    but not the program, so it must close without a re-eval of byte-identical
    code.
    """
    ctx = _ctx(repo)
    _live_run_shape(repo, ctx)
    (repo / "codeevolver.md").write_text("## Change summary (iteration 1)\nX\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "iter 1: append change summary")
    doc_sha = _sha(repo)

    result = close_iteration_tool(**_close_kwargs(ctx, repo))

    assert result["ok"] is True
    assert len(ctx.eval_records) == 2, "no re-eval was needed"

    ctx.parent_sha = ctx.start_sha
    finalized = _finalize(ctx, repo, doc_sha)
    # The score survives the doc commit rather than degrading to None.
    assert finalized.best_subsample_score == 0.925
    assert _head_score(ctx, repo) == (0.925, "program_equivalent")


def test_unevaluated_program_change_closes_without_a_gate(repo):
    """A post-eval program edit does not force a redundant 50-row eval. The record carries the iteration's
    best reading and, by design, no score of the closing commit."""
    ctx = _ctx(repo)
    _live_run_shape(repo, ctx)
    (repo / "judge.py").write_text("VERDICT = 'never evaluated'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "untested program change")
    head = _sha(repo)

    result = close_iteration_tool(**_close_kwargs(ctx, repo))

    assert result["ok"] is True
    assert len(ctx.eval_records) == 2, "no re-eval was demanded"
    ctx.parent_sha = ctx.start_sha
    finalized = _finalize(ctx, repo, head)
    assert finalized.changes_landed is True
    assert finalized.best_subsample_score == 0.925
    assert _head_score(ctx, repo) == (None, "none")


def test_reverted_iteration_closes_as_no_change_not_skipped(repo):
    """Tested, regressed, reverted — a closed iteration, not a skipped one."""
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    ctx.parent_sha = parent_sha
    ctx.child_branch = _BRANCH
    ctx.record_eval(
        parent_sha, 0.825, eval_sample="train", branch=_BRANCH,
        tree=git_tree_signature(str(repo)),
    )
    ctx.initial_subsample_score = 0.825
    (repo / "judge.py").write_text("VERDICT = 'worse'\n")
    ctx.record_eval(
        parent_sha, 0.611, eval_sample="train", branch=_BRANCH,
        tree=git_tree_signature(str(repo)),
    )
    _git(repo, "checkout", "-q", "--", "judge.py")  # the architect reverts

    result = _finalize(ctx, repo, _sha(repo))

    assert result.changes_landed is False
    assert _head_score(ctx, repo)[0] == 0.825
    assert ctx.best_subsample_score == 0.825
    # The regression that was backed out is still on the record.
    assert [r["score"] for r in ctx.eval_records] == [0.825, 0.611]


def test_iteration_end_state_survives_a_later_reset_and_gc(repo):
    """A recorded sha is not a reachable one.

    A later iteration can reset the branch a recorded sha was on, leaving it
    reachable from no branch, and `git gc` prunes it. Analysis has to be able to check out every iteration's end
    state long after the run.
    """
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    child_sha = _live_run_shape(repo, ctx)
    ctx.child_branch = _BRANCH
    ctx.parent_sha = parent_sha
    _finalize(ctx, repo, child_sha)

    anchor_iteration_state(ctx, workspace=str(repo))

    # A later iteration abandons this work and prunes.
    _git(repo, "reset", "-q", "--hard", parent_sha)
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "gc", "-q", "--prune=now")

    assert _git_out(repo, "cat-file", "-e", f"{child_sha}^{{commit}}").returncode == 0
    resolved = _git_out(repo, "rev-parse", "refs/codeevolver/iterations/1").stdout.strip()
    assert resolved == child_sha
    # And the program it evaluated is still checkout-able.
    show = _git_out(repo, "show", f"{child_sha}:judge.py")
    assert "decisive" in show.stdout


def test_evaluated_but_never_committed_content_survives_gc(repo):
    """The reverted program stays diffable, which is what lets the analysis
    agent write a real "here is what we tried" summary for a non-landing
    iteration."""
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    ctx.parent_sha = parent_sha
    ctx.child_branch = _BRANCH
    ctx.record_eval(
        parent_sha, 0.825, eval_sample="train", tree=git_tree_signature(str(repo)),
    )
    (repo / "judge.py").write_text("VERDICT = 'tried and reverted'\n")
    tried_tree = git_tree_signature(str(repo))
    ctx.record_eval(parent_sha, 0.611, eval_sample="train", tree=tried_tree)
    _git(repo, "checkout", "-q", "--", "judge.py")
    _finalize(ctx, repo, _sha(repo))

    anchor_iteration_state(ctx, workspace=str(repo))
    _git(repo, "reflog", "expire", "--expire=now", "--all")
    _git(repo, "gc", "-q", "--prune=now")

    assert _git_out(repo, "cat-file", "-t", tried_tree).stdout.strip() == "tree"
    diff = _git_out(repo, "diff", ctx.parent_tree, tried_tree).stdout
    assert "tried and reverted" in diff


def test_reconcile_is_idempotent(repo):
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    _live_run_shape(repo, ctx)

    trees = git_commit_trees(repo, _BRANCH, since=parent_sha)
    assert ctx.reconcile_eval_shas(trees) == 1
    before = [dict(r) for r in ctx.eval_records]
    assert ctx.reconcile_eval_shas(trees) == 0
    assert [dict(r) for r in ctx.eval_records] == before


def test_reconcile_noop_when_git_fails(repo):
    ctx = _ctx(repo)
    _live_run_shape(repo, ctx)
    before = [dict(r) for r in ctx.eval_records]

    assert ctx.reconcile_eval_shas([]) == 0
    assert [dict(r) for r in ctx.eval_records] == before


def test_reconcile_newest_commit_wins_for_a_repeated_tree(repo):
    """An in-iteration revert reproduces an earlier tree; newest commit wins."""
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    original = (repo / "judge.py").read_text()

    (repo / "judge.py").write_text("VERDICT = 'experiment'\n")
    ctx.record_eval(
        parent_sha, 0.4, eval_sample="train", tree=git_tree_signature(repo)
    )
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "experiment")

    # Revert to the seed content and commit that as a new commit.
    (repo / "judge.py").write_text(original)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "revert")
    revert_sha = _sha(repo)

    ctx.record_eval(
        revert_sha, 0.825, eval_sample="train", tree=git_tree_signature(repo)
    )
    ctx.reconcile_eval_shas(git_commit_trees(repo, _BRANCH, since=parent_sha))

    # The reverted tree also exists at the seed commit; the newest wins so a
    # record is never dragged backwards in history.
    assert ctx.eval_records[1]["sha"] == revert_sha


def test_genuinely_untested_head_closes_but_is_never_scored(repo):
    """Reconciliation is not a blanket claim either: an unevaluated commit
    closes (no gate), but nothing attributes a score to it."""
    ctx = _ctx(repo)
    _live_run_shape(repo, ctx)

    # A further edit that was never evaluated, then committed.
    (repo / "judge.py").write_text("VERDICT = 'never evaluated'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "unevaluated change")

    result = close_iteration_tool(**_close_kwargs(ctx, repo))
    assert result["ok"] is True and ctx.close_called is True
    ctx.close_tree = git_tree_signature(str(repo), _sha(repo))
    assert _head_score(ctx, repo) == (None, "none")


def test_close_still_rejects_dirty_tree(repo):
    ctx = _ctx(repo)
    _live_run_shape(repo, ctx)
    (repo / "judge.py").write_text("VERDICT = 'uncommitted'\n")

    with pytest.raises(CloseIterationError, match="uncommitted changes"):
        close_iteration_tool(**_close_kwargs(ctx, repo))


def test_close_backward_compat_with_tree_less_records(repo):
    """Legacy/persisted records carry no `tree`; sha matching still closes."""
    ctx = _ctx(repo)
    (repo / "judge.py").write_text("VERDICT = 'decisive'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "committed before evaluating")
    ctx.record_eval(_sha(repo), 0.9, eval_sample="train", branch=_BRANCH)
    assert ctx.eval_records[0].get("tree") is None

    assert close_iteration_tool(**_close_kwargs(ctx, repo))["ok"] is True


def test_reconcile_records_the_pre_reconciliation_sha(repo):
    """Audit trail: `comparison_set_history.jsonl` is written synchronously
    and cannot be retro-edited, so the two can legitimately disagree."""
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    child_sha = _live_run_shape(repo, ctx)

    ctx.reconcile_eval_shas(git_commit_trees(repo, _BRANCH, since=parent_sha))

    moved = ctx.eval_records[1]
    assert moved["sha"] == child_sha
    assert moved["sha_at_eval"] == parent_sha
    # The untouched record gains no audit key.
    assert "sha_at_eval" not in ctx.eval_records[0]


def test_reconcile_skips_records_with_no_tree(repo):
    """Legacy/persisted records carry no `tree` and must be left alone."""
    ctx = _ctx(repo)
    parent_sha = ctx.start_sha
    ctx.record_eval(parent_sha, 0.5, eval_sample="train")  # no tree
    (repo / "judge.py").write_text("VERDICT = 'decisive'\n")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "c2")

    assert ctx.reconcile_eval_shas(
        git_commit_trees(repo, _BRANCH, since=parent_sha)
    ) == 0
    assert ctx.eval_records[0]["sha"] == parent_sha
