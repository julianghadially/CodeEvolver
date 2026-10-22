"""Copyright © 2026 440 Labs LLC

Pure-function tests for the engine 3 deterministic closing machinery:

  - `closing.derive_iteration_parent` — which past iteration is this
    iteration's parent (ancestor walk, merge-base fallback, seed fallback),
    with dict/lambda git fakes (the same pattern as
    `IterationContext.finalize`).
  - `closing.resolve_primary` — applying a close_iteration registration to
    the run's primary, with the never-primary-less fallback.
  - `run_metrics.derive_run_metrics` + `explore_guard_should_fire` —
    the exploration metrics and the explore-frequency guard.
  - `CEState` — closing-record upserts, primary transitions, comparison
    history JSONL, and JSON save/load round-trip.
"""

from __future__ import annotations

import json

import pytest

from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.closing import (
    derive_iteration_parent,
    resolve_primary,
)
from src.engine.optimizer.v3.run_metrics import (
    EXPLORE_GUARD_MIN_ITERATIONS,
    derive_run_metrics,
    explore_guard_should_fire,
)


# --- derive_iteration_parent -------------------------------------------------

_SEED_SHA = "seed_sha"
_SEED_BRANCH = "codeevolver-20260601120000-main"


def _past(*records):
    """Past closing records, oldest → newest."""
    return list(records)


def _rec(iteration, close_sha, branch, **extra):
    """A past closing record. Parent derivation keys on `close_sha` — the
    branch tip the next iteration physically inherits — not on `final_sha`."""
    return {
        "iteration": iteration,
        "close_sha": close_sha,
        "final_sha": close_sha,
        "branch": branch,
        **extra,
    }


def _ancestor_fake(edges: dict[str, set[str]]):
    """is_ancestor(a, b): a is an ancestor of b iff a in edges[b]."""
    return lambda a, b: a in edges.get(b, set())


def test_derive_parent_ancestor_path_picks_newest():
    past = _past(
        _rec(1, "sha_a", "codeevolver-b1"),
        _rec(2, "sha_b", "codeevolver-b2"),
    )
    # BOTH past final shas are ancestors of the close sha — the newest wins.
    is_ancestor = _ancestor_fake({"close_sha": {"sha_a", "sha_b"}})

    parent = derive_iteration_parent(
        close_sha="close_sha",
        past_iterations=past,
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=is_ancestor,
        merge_base=lambda a, b: pytest.fail("merge_base must not be consulted"),
    )

    assert parent.via == "ancestor"
    assert parent.parent_sha == "sha_b"
    assert parent.parent_branch == "codeevolver-b2"
    assert parent.parent_iteration == 2


def test_derive_parent_close_sha_equal_to_past_final_sha():
    # A no-change iteration closing exactly at a past final sha is that
    # iteration's child (no is_ancestor call needed).
    past = _past(_rec(1, "sha_a", "codeevolver-b1"))

    parent = derive_iteration_parent(
        close_sha="sha_a",
        past_iterations=past,
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=lambda a, b: False,
        merge_base=lambda a, b: None,
    )

    assert parent.via == "ancestor"
    assert parent.parent_sha == "sha_a"
    assert parent.parent_iteration == 1


def test_derive_parent_uses_iterations_that_landed_nothing():
    """A tested-and-reverted iteration is still the next iteration's parent.

    It picked a parent, spent budget, and left a real branch tip — that tip is
    what the next iteration starts from whether or not the program moved.
    Excluding those records would skip over the true parent.
    """
    past = _past(
        _rec(1, "sha_a", "codeevolver-b1"),
        _rec(2, "sha_b", "codeevolver-b2", changes_landed=False),
    )
    is_ancestor = _ancestor_fake({"close_sha": {"sha_a", "sha_b"}})

    parent = derive_iteration_parent(
        close_sha="close_sha",
        past_iterations=past,
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=is_ancestor,
        merge_base=lambda a, b: None,
    )

    assert parent.parent_sha == "sha_b"
    assert parent.parent_iteration == 2


def test_derive_parent_ignores_architect_error_stubs():
    """The one shape with no git state: the architect raised."""
    past = _past(
        _rec(1, "sha_a", "codeevolver-b1"),
        {"iteration": 2, "skipped": True, "error": "boom"},
    )
    is_ancestor = _ancestor_fake({"close_sha": {"sha_a"}})

    parent = derive_iteration_parent(
        close_sha="close_sha",
        past_iterations=past,
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=is_ancestor,
        merge_base=lambda a, b: None,
    )

    assert parent.parent_sha == "sha_a"
    assert parent.parent_iteration == 1


def test_derive_parent_merge_base_fallback_on_rewritten_history():
    # History was rebased: no past final sha is an ancestor of the close sha,
    # but a common ancestor exists — attribute the parent to that iteration
    # at the merge-base sha.
    past = _past(
        _rec(1, "sha_a", "codeevolver-b1"),
        _rec(2, "sha_b", "codeevolver-b2"),
    )
    merge_bases = {("close_sha", "sha_b"): "base_sha"}

    parent = derive_iteration_parent(
        close_sha="close_sha",
        past_iterations=past,
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=lambda a, b: False,
        merge_base=lambda a, b: merge_bases.get((a, b)),
    )

    assert parent.via == "merge_base"
    assert parent.parent_sha == "base_sha"
    assert parent.parent_branch == "codeevolver-b2"
    assert parent.parent_iteration == 2


def test_derive_parent_seed_fallback():
    parent = derive_iteration_parent(
        close_sha="orphan_sha",
        past_iterations=_past(_rec(1, "sha_a", "codeevolver-b1")),
        seed_sha=_SEED_SHA,
        seed_branch=_SEED_BRANCH,
        is_ancestor=lambda a, b: False,
        merge_base=lambda a, b: None,
    )

    assert parent.via == "seed"
    assert parent.parent_sha == _SEED_SHA
    assert parent.parent_branch == _SEED_BRANCH
    assert parent.parent_iteration == 0


# --- resolve_primary ---------------------------------------------------------

_PRIOR = "codeevolver-prior-primary"


def test_resolve_primary_promotion():
    entry = {"branch": "codeevolver-new", "was_primary": False, "is_primary": True}

    assert resolve_primary(entry, _PRIOR) == ("codeevolver-new", False)


def test_resolve_primary_demotion_without_replacement_falls_back():
    # The entry demotes the prior primary and promotes nothing — the run must
    # never be primary-less, so the prior primary stays with fallback=True.
    entry = {"branch": _PRIOR, "was_primary": True, "is_primary": False}

    assert resolve_primary(entry, _PRIOR) == (_PRIOR, True)


def test_resolve_primary_untouched_by_non_primary_registration():
    entry = {"branch": "codeevolver-side", "was_primary": False, "is_primary": False}

    assert resolve_primary(entry, _PRIOR) == (_PRIOR, False)


def test_resolve_primary_empty_registration_keeps_prior():
    assert resolve_primary(None, _PRIOR) == (_PRIOR, False)
    assert resolve_primary({}, _PRIOR) == (_PRIOR, False)


# --- derive_run_metrics + explore_guard_should_fire --------------------------


def _closed_rec(
    branch,
    *,
    mode="exploitation",
    parent_is_primary=True,
    primary_at_start=_PRIOR,
    skipped=False,
):
    return {
        "branch": branch,
        "skipped": skipped,
        "claimed_improvement_mode": mode,
        "parent_is_primary": parent_is_primary,
        "primary_branch_at_start": primary_at_start,
    }


def test_derive_run_metrics():
    iterations = [
        _closed_rec(_PRIOR),  # exploit on the primary itself
        _closed_rec("codeevolver-b2", mode="exploration", parent_is_primary=False),
        _closed_rec("codeevolver-b3"),  # forked off primary but exploiting
        _closed_rec("codeevolver-b2", skipped=True),  # skipped — not counted
    ]

    metrics = derive_run_metrics(iterations)

    assert metrics["branches_explored"] == 3
    assert metrics["iterations_on_non_primary"] == 2
    assert metrics["efforts_on_non_primary_lineages"] == 1
    assert metrics["claimed_explore_fraction"] == pytest.approx(1 / 3)
    assert metrics["derived_explore_fraction"] == pytest.approx(1 / 3)


def test_derive_run_metrics_empty():
    metrics = derive_run_metrics([])
    assert metrics["branches_explored"] == 0
    assert metrics["claimed_explore_fraction"] is None
    assert metrics["derived_explore_fraction"] is None


def test_explore_guard_needs_four_closed_iterations():
    assert EXPLORE_GUARD_MIN_ITERATIONS == 4
    all_exploit = [_closed_rec(f"codeevolver-b{i}") for i in range(3)]
    # Only 3 real closes — never fires, whatever the fraction.
    assert explore_guard_should_fire(all_exploit) is False
    # Skipped records do not count toward the minimum.
    assert (
        explore_guard_should_fire(all_exploit + [_closed_rec("codeevolver-bx", skipped=True)])
        is False
    )


def test_explore_guard_fires_below_threshold():
    # 4 closes, 1 claimed exploration → 0.25 < 0.30 → fire.
    iterations = [
        _closed_rec("codeevolver-b1", mode="exploration"),
        _closed_rec("codeevolver-b2"),
        _closed_rec("codeevolver-b3"),
        _closed_rec("codeevolver-b4"),
    ]
    assert explore_guard_should_fire(iterations) is True


def test_explore_guard_quiet_at_or_above_threshold():
    # 4 closes, 2 claimed explorations → 0.5 >= 0.30 → no fire.
    iterations = [
        _closed_rec("codeevolver-b1", mode="exploration"),
        _closed_rec("codeevolver-b2", mode="exploration"),
        _closed_rec("codeevolver-b3"),
        _closed_rec("codeevolver-b4"),
    ]
    assert explore_guard_should_fire(iterations) is False


# --- CEState -----------------------------------------------------------------

_SEED = "codeevolver-20260601120000-main"


def _make_state(tmp_path=None) -> CEState:
    history = str(tmp_path / "history" / "comparison_set_history.jsonl") if tmp_path else None
    return CEState(
        seed_branch=_SEED, seed_sha="seed_sha", comparison_history_path=history
    )


def test_ce_state_starts_with_seed_primary_and_zero_evals():
    state = _make_state()
    assert state.primary_branch == _SEED
    assert state.total_num_evals == 0
    assert state.known_branches() == {_SEED}
    assert state.branches[_SEED]["is_primary"] is True

    state.increment_evals(7)
    state.increment_evals(3)
    assert state.total_num_evals == 10


def test_ce_state_record_iteration_upserts_branch_and_moves_primary():
    state = _make_state()

    state.record_iteration(
        {
            "iteration": 1,
            "skipped": False,
            "branch": "codeevolver-b1",
            "close_sha": "b1_sha",
            "primary_branch_after": _SEED,  # not promoted yet
        }
    )
    assert state.primary_branch == _SEED
    assert state.branches["codeevolver-b1"]["last_sha"] == "b1_sha"
    assert state.branches["codeevolver-b1"]["is_primary"] is False
    assert state.branches["codeevolver-b1"]["created_iteration"] == 1

    # Second close on the same branch upserts (no duplicate entry) and
    # promotes it — exactly one primary flag remains set.
    state.record_iteration(
        {
            "iteration": 2,
            "skipped": False,
            "branch": "codeevolver-b1",
            "close_sha": "b1_sha2",
            "primary_branch_after": "codeevolver-b1",
        }
    )
    assert state.primary_branch == "codeevolver-b1"
    assert state.branches["codeevolver-b1"]["last_sha"] == "b1_sha2"
    assert state.branches["codeevolver-b1"]["last_iteration"] == 2
    assert state.branches["codeevolver-b1"]["is_primary"] is True
    assert state.branches[_SEED]["is_primary"] is False
    assert state.num_iterations == 2

    # A branchless stub (skipped/errored iteration) records without touching
    # the registry or the primary.
    state.record_iteration({"iteration": 3, "skipped": True})
    assert state.num_iterations == 3
    assert state.known_branches() == {_SEED, "codeevolver-b1"}
    assert state.primary_branch == "codeevolver-b1"


def test_ce_state_record_comparison_eval_appends_jsonl(tmp_path):
    state = _make_state(tmp_path)

    state.record_comparison_eval(
        iteration=1, branch="codeevolver-b1", sha="sha1", score=0.5, n_rows=20
    )
    state.record_comparison_eval(
        iteration=2,
        branch="codeevolver-b2",
        sha="sha2",
        score=0.6,
        n_rows=20,
        scores_by_metric={"accuracy": 0.6},
    )

    lines = [
        json.loads(line)
        for line in open(state.comparison_history_path).read().splitlines()
    ]
    assert len(lines) == 2
    assert lines[0]["branch"] == "codeevolver-b1"
    assert lines[0]["sha"] == "sha1"
    assert lines[0]["score"] == 0.5
    assert lines[1]["scores_by_metric"] == {"accuracy": 0.6}
    assert all("ts" in line for line in lines)


def test_ce_state_save_load_round_trip(tmp_path):
    state = _make_state(tmp_path)
    state.increment_evals(42)
    state.record_iteration(
        {
            "iteration": 1,
            "skipped": False,
            "branch": "codeevolver-b1",
            "close_sha": "b1_sha",
            "primary_branch_after": "codeevolver-b1",
        }
    )

    path = str(tmp_path / "state" / "ce_state.json")
    state.save(path)
    loaded = CEState.load(path)

    assert loaded.seed_branch == state.seed_branch
    assert loaded.seed_sha == state.seed_sha
    assert loaded.primary_branch == "codeevolver-b1"
    assert loaded.branches == state.branches
    assert loaded.iterations == state.iterations
    assert loaded.total_num_evals == 42
    assert loaded.comparison_history_path == state.comparison_history_path


def test_ce_state_load_rejects_foreign_schema(tmp_path):
    path = tmp_path / "bad_state.json"
    path.write_text(json.dumps({"schema_version": 1, "seed_branch": "x"}))

    with pytest.raises(ValueError, match="schema_version"):
        CEState.load(str(path))

# ---------------------------------------------------------------------------
# build_close_record: the closing scores and the end-state fields
# ---------------------------------------------------------------------------

from src.engine.optimizer.v3.closing import build_close_record
from src.engine.optimizer.v3.iteration_context import IterationContext
from tests.controls._arch_helpers import FakeState


def _equivalence_fn():
    return lambda a, b: bool(a) and bool(b) and a == b


def _closed_ctx(**overrides):
    ctx = IterationContext(
        iteration_idx=3,
        parent_locked=True,
        start_branch="codeevolver-x-main",
        start_sha="seed",
        primary_branch_at_start="codeevolver-x-main",
    )
    ctx.child_branch = "codeevolver-x-main"
    ctx.registered_branch = {
        "branch": "codeevolver-x-main",
        "was_primary": True,
        "is_primary": True,
        "promotion_evidence": None,
    }
    ctx.close_called = True
    ctx.initial_subsample_score = 0.40
    ctx.parent_sha = "sha_parent"
    ctx.parent_tree = "tree_parent"
    for key, value in overrides.items():
        setattr(ctx, key, value)
    return ctx


def test_close_record_scores_the_content_that_closed():
    ctx = _closed_ctx()
    ctx.record_eval("sha_a", 0.40, tree="tree_parent")
    ctx.record_eval("sha_a", 0.55, tree="tree_b")
    ctx.close_sha = "sha_b"
    ctx.close_tree = "tree_b"
    ctx.finalize(_equivalence_fn())

    rec = build_close_record(ctx, FakeState())

    assert rec["subsample_best"] == 0.55
    assert rec["subsample_best_sha"] == "sha_a"
    assert rec["close_sha"] == "sha_b"
    assert rec["changes_landed"] is True
    assert rec["skipped"] is False
    # No score of the closing content, under any name.
    assert "subsample_final" not in rec
    assert "final_sha" not in rec
    assert "final_score_source" not in rec
    # EE never stamps the evolution-only comparison mapping.
    assert rec["comparison_state"] is None
    assert rec["comparison_best"] is None


def test_close_record_has_no_score_when_closing_content_was_untested():
    """The architect committed program changes it never evaluated."""
    ctx = _closed_ctx()
    ctx.record_eval("sha_a", 0.40, tree="tree_parent")
    ctx.record_eval("sha_a", 0.55, tree="tree_b")
    ctx.close_sha = "sha_untested"
    ctx.close_tree = "tree_untested"
    ctx.finalize(_equivalence_fn())

    rec = build_close_record(ctx, FakeState())

    assert rec["close_sha"] == "sha_untested"
    # The strongest version tried is the iteration's reading; the record
    # never claimed the closing content was measured.
    assert rec["subsample_best"] == 0.55
    assert rec["subsample_baseline"] == 0.40


def test_close_record_ignores_comparison_evals_when_scoring():
    ctx = _closed_ctx()
    ctx.record_eval("sha_a", 0.40, tree="tree_parent")
    ctx.record_eval("sha_c", 0.90, tree="tree_c", eval_sample="comparison")
    ctx.close_sha = "sha_c"
    ctx.close_tree = "tree_c"
    ctx.finalize(_equivalence_fn())

    rec = build_close_record(ctx, FakeState())

    assert rec["subsample_best"] == 0.40  # only train evals score the subsample
    assert rec["comparison_best"] == 0.90
    assert rec["comparison_best_sha"] == "sha_c"
    assert rec["comparison_scores"][0]["score"] == 0.90


def test_close_record_marks_a_reverted_iteration_as_closed_not_skipped():
    """The distinction the whole record turns on.

    An iteration that tested a change, measured a regression and reverted is
    `closed`. Recording it as `skipped` would drop it from run_metrics, from
    parent derivation and from the change summaries — the same treatment as
    an architect crash.
    """
    ctx = _closed_ctx()
    ctx.record_eval("sha_parent", 0.40, tree="tree_parent")
    ctx.record_eval("sha_parent", 0.31, tree="tree_tried")
    ctx.close_sha = "sha_parent"
    ctx.close_tree = "tree_parent"
    ctx.finalize(_equivalence_fn())

    rec = build_close_record(ctx, FakeState())

    assert rec["skipped"] is False
    assert rec["changes_landed"] is False
    assert rec["subsample_best"] == 0.40


def test_close_record_carries_branches_seen():
    ctx = _closed_ctx()
    ctx.close_sha = "sha_b"
    ctx.close_tree = "tree_b"
    ctx.branches_seen = {
        "codeevolver-x-main": "sha_b",
        "codeevolver-abandoned-experiment": "sha_z",
    }
    ctx.finalize(_equivalence_fn())

    rec = build_close_record(ctx, FakeState())

    # The abandoned branch never appears in registered_branch, so without
    # this sweep the run record has no trace it ever existed.
    assert rec["branches_seen"]["codeevolver-abandoned-experiment"] == "sha_z"
