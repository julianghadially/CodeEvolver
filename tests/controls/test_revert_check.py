"""Copyright © 2026 440 Labs LLC

Revert check + eval-sha reconciliation (deterministic, no real SDK).

After the architect session exits, the loop commits any uncommitted drift and
reconciles dirty-tree eval records onto the commits that hold their content.
The revert check then warns when the closing sha scored LOWER on the training
subsample than an earlier sha this iteration, asking the architect to leave
the branch in the best state (no forced revert).

There is no untested-HEAD watchdog: the close record never claims the
closing commit was measured, so re-prompting for a re-eval would only cost
budget. `test_no_untested_head_watchdog_exists` pins that.
"""

from __future__ import annotations

from src.engine.optimizer.v3 import iteration_architect as ia_module
from src.engine.optimizer.v3 import architect_watchdogs as wd_module

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


def _patch_head(monkeypatch, sha: str, seen_refs: list[str] | None = None):
    def _get_sha(ws, ref="HEAD"):
        if seen_refs is not None:
            seen_refs.append(ref)
        return sha

    monkeypatch.setattr(wd_module, "git_get_sha", _get_sha)


def _patch_commit_trees(monkeypatch, pairs=()):
    """Pin `git_commit_trees`. Empty = reconciliation is a no-op."""
    monkeypatch.setattr(ia_module, "git_commit_trees", lambda *a, **k: list(pairs))


# --- revert check ------------------------------------------------------------


def test_revert_check_fires_when_closing_sha_scored_below_earlier_best(monkeypatch):
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.55, eval_sample="train", n_rows=5)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert ctx.watchdog_fired is True
    assert len(calls) == 1
    assert calls[0]["resume"] == "sess-3"
    # The prompt names the better sha and offers a git revert path.
    assert "best_sha" in calls[0]["prompt"]
    assert "reset" in calls[0]["prompt"]
    assert "close_iteration" in calls[0]["prompt"]


def test_revert_check_does_not_fire_when_head_is_best(monkeypatch):
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.record_eval("baseline_sha", 0.55, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.80, eval_sample="train", n_rows=5)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_revert_check_tolerates_a_deficit_within_one_run_to_run_std(monkeypatch):
    """specs/archive-public/sampling_variance.md §Revert-Check: once the run has taken its
    `run_variance_estimate`, the threshold is best − σ, so a closing sha
    within the noise floor is not a regression."""
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.variance_estimate = {"run_to_run_std": 0.10}
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.75, eval_sample="train", n_rows=5)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


def test_revert_check_still_fires_beyond_one_run_to_run_std(monkeypatch):
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.variance_estimate = {"run_to_run_std": 0.02}
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.75, eval_sample="train", n_rows=5)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert len(calls) == 1
    assert "0.0200" in calls[0]["prompt"]
    assert ctx.watchdog_fired is True


def test_revert_check_does_not_fire_for_untested_head(monkeypatch):
    # HEAD never evaluated at all → nothing to compare; no re-prompt either.
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.record_eval("sha_a", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("sha_b", 0.55, eval_sample="train", n_rows=5)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "untested_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert calls == []
    assert ctx.watchdog_fired is False


# --- reconciliation: the commit-after-eval flow -----------------------------


def test_reconciliation_repoints_a_dirty_eval_onto_its_commit(monkeypatch):
    """The flow the close_iteration skill prescribes: edit -> eval -> commit.

    `run_eval` stamps HEAD (the PARENT commit, since the tree was dirty).
    Reconciliation re-points that record onto the commit whose tree matches
    and marks it `committed`, which is what `subsample_best_sha` and the
    revert check read.
    """
    arch, ctx, _ = make_architect(iteration_idx=4)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.start_sha = "parent_sha"
    ctx.record_eval(
        "parent_sha", 0.50, eval_sample="train", n_rows=5, tree="tree_abc"
    )
    arch._last_session_id = "sess-2"

    _patch_head(monkeypatch, "committed_sha")
    _patch_commit_trees(monkeypatch, [("committed_sha", "tree_abc")])
    calls = _capture_launch(monkeypatch, arch)

    arch._reconcile_eval_shas()
    assert ctx.eval_records[0]["sha"] == "committed_sha"
    assert ctx.eval_records[0]["sha_at_eval"] == "parent_sha"
    assert ctx.eval_records[0]["committed"] is True
    assert calls == []


def test_no_untested_head_watchdog_exists():
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect

    assert not hasattr(IterationArchitect, "_run_untested_head_watchdog_if_needed")


def test_reconcile_is_a_noop_when_git_fails(monkeypatch):
    """Degrade to today's sha-only behaviour, never crash the close path."""
    arch, ctx, _ = make_architect(iteration_idx=4)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.record_eval("parent_sha", 0.5, eval_sample="train", tree="tree_abc")

    def _boom(*a, **k):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(ia_module, "git_commit_trees", _boom)
    arch._reconcile_eval_shas()  # must not raise

    assert ctx.eval_records[0]["sha"] == "parent_sha"


def test_revert_check_fires_once_records_carry_the_committed_sha(monkeypatch):
    """The revert check was dead in the normal flow before reconciliation.

    `head_scores` filters eval records by `sha == head_sha`; a dirty-tree eval
    never carried HEAD's sha, so the list was always empty and the check
    returned early every time.
    """
    arch, ctx, _ = make_architect(iteration_idx=4)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.start_sha = "parent_sha"
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5, tree="tree_best")
    ctx.record_eval("parent_sha", 0.40, eval_sample="train", n_rows=5, tree="tree_head")
    arch._last_session_id = "sess-2"

    _patch_head(monkeypatch, "committed_sha")
    _patch_commit_trees(monkeypatch, [("committed_sha", "tree_head")])
    calls = _capture_launch(monkeypatch, arch)

    arch._reconcile_eval_shas()  # HEAD now holds the LOWER-scoring code
    arch._run_revert_check_if_needed("SYS", {})

    assert len(calls) == 1
    assert "best_sha" in calls[0]["prompt"]


def test_revert_check_silent_when_head_is_already_the_best(monkeypatch):
    arch, ctx, _ = make_architect(iteration_idx=4)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.start_sha = "parent_sha"
    ctx.record_eval("old_sha", 0.40, eval_sample="train", n_rows=5, tree="tree_old")
    ctx.record_eval("parent_sha", 0.80, eval_sample="train", n_rows=5, tree="tree_head")
    arch._last_session_id = "sess-2"

    _patch_head(monkeypatch, "committed_sha")
    _patch_commit_trees(monkeypatch, [("committed_sha", "tree_head")])
    calls = _capture_launch(monkeypatch, arch)

    arch._reconcile_eval_shas()
    arch._run_revert_check_if_needed("SYS", {})

    assert calls == []


# --- a doc-only closing commit is the same PROGRAM ---------------------------


def _patch_program_equivalence(monkeypatch, head_tree: str, equivalent_to: set[str]):
    """Make `head_tree` program-equivalent to the named eval trees.

    Models the real case: the architect appended its change summary to
    codeevolver.md and committed, so the closing tree differs from the tree it
    evaluated by a documentation file alone.
    """
    monkeypatch.setattr(
        wd_module, "git_tree_signature", lambda ws, ref=None: head_tree
    )
    monkeypatch.setattr(
        wd_module,
        "diff_is_non_program_only",
        lambda ws, a, b: a == b or {a, b} <= (equivalent_to | {head_tree}),
    )


def test_revert_check_silent_when_the_best_eval_is_the_closing_program(monkeypatch):
    """The other half of the same false positive.

    A redundant re-eval of program-identical code that scores lower than the
    earlier reading must not make the revert check warn that the closing sha
    is worse than "an earlier sha".
    """
    arch, ctx, _ = make_architect(iteration_idx=7)
    ctx.child_branch = "codeevolver-hover-react-gap"
    ctx.start_sha = "parent_sha"
    ctx.record_eval("code_sha", 0.9308, eval_sample="train", n_rows=50, tree="tree_code")
    ctx.record_eval("doc_sha", 0.9114, eval_sample="train", n_rows=50, tree="tree_with_docs")
    arch._last_session_id = "sess-7"

    _patch_head(monkeypatch, "doc_sha")
    _patch_program_equivalence(monkeypatch, "tree_with_docs", {"tree_code"})
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert calls == [], "the best eval and the closing program are the same code"


def test_revert_check_ignores_single_row_smoke_evals(monkeypatch):
    """A 1-row `single` smoke check must not fire the check against a 50-row
    closing score. `single` is in the train family for bookkeeping but is not
    a comparable reading."""
    arch, ctx, _ = make_architect(iteration_idx=10)
    ctx.child_branch = "codeevolver-20260601130000-orthomerge"
    ctx.record_eval("base_sha", 0.8134, eval_sample="train", n_rows=50)
    ctx.record_eval("head_sha", 0.992, eval_sample="single", n_rows=1)
    ctx.record_eval("head_sha", 0.8730, eval_sample="train", n_rows=50)
    arch._last_session_id = "sess-10"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert ctx.watchdog_fired is False
    assert calls == []


def test_revert_check_needs_two_real_train_readings(monkeypatch):
    """One train eval plus any number of `single` smoke checks is not enough to
    compare against — the check stays silent rather than treating a 1-row
    reading as the earlier best."""
    arch, ctx, _ = make_architect(iteration_idx=3)
    ctx.child_branch = "codeevolver-20260601130000-x"
    ctx.record_eval("head_sha", 0.99, eval_sample="single", n_rows=1)
    ctx.record_eval("head_sha", 0.70, eval_sample="train", n_rows=50)
    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)
    arch._run_revert_check_if_needed("SYS", {})
    assert calls == [] and ctx.watchdog_fired is False


def test_revert_check_suppressed_when_comparison_eval_ran(monkeypatch):
    """The comparison set is the promotion basis on every arm — once the
    iteration holds a comparison reading, a train-subsample deficit is not
    evidence worth a resume."""
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.55, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.60, eval_sample="comparison", n_rows=20)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert ctx.watchdog_fired is False
    assert calls == []


def test_revert_check_suppressed_by_loop_initiated_comparison(monkeypatch):
    """A `by_loop` comparison (pareto's fitness-measurement run) is
    comparison evidence all the same."""
    arch, ctx, _ = make_architect(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260601130000-rerank"
    ctx.record_eval("best_sha", 0.80, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.55, eval_sample="train", n_rows=5)
    ctx.record_eval("head_sha", 0.60, eval_sample="comparison", n_rows=20, by_loop=True)
    arch._last_session_id = "sess-3"

    _patch_head(monkeypatch, "head_sha")
    calls = _capture_launch(monkeypatch, arch)

    arch._run_revert_check_if_needed("SYS", {})

    assert ctx.watchdog_fired is False
    assert calls == []
