"""Copyright © 2026 440 Labs LLC

Unit tests for IterationContext — best-of-iteration progress readings,
same-subsample rule, durable refs, content lookup, changes_landed.
"""

from __future__ import annotations

import pytest

from src.engine.agent.token_usage import TokenUsage
from src.engine.optimizer.v3.iteration_context import IterationContext


def _equivalence_fn(equivalent_pairs: set[tuple[str, str]] | None = None):
    """Program-equivalence stand-in for `program_paths.diff_is_non_program_only`.

    Identical trees are equivalent; anything else only if the test declares the
    pair equivalent (mirroring a doc-only diff).
    """
    pairs = equivalent_pairs or set()

    def fn(a: str | None, b: str | None) -> bool:
        if not a or not b:
            return False
        if a == b:
            return True
        return (a, b) in pairs or (b, a) in pairs

    return fn


def _closed_context(
    *,
    parent_tree: str = "tree_parent",
    close_tree: str = "tree_close",
    close_sha: str = "close_sha",
) -> IterationContext:
    ctx = IterationContext(iteration_idx=3)
    ctx.parent_sha = "parent_sha"
    ctx.parent_tree = parent_tree
    ctx.parent_branch = "codeevolver-20260414000000-main"
    ctx.child_branch = "codeevolver-20260414000000-child"
    ctx.parent_locked = True
    ctx.close_sha = close_sha
    ctx.close_tree = close_tree
    return ctx


# --- finalize: progress readings, never a score of the closing commit --------


def test_finalize_reports_best_on_the_baselines_subsample_not_the_close():
    """A recorded iteration, restated.

    The architect evaluated three variants, kept the second (0.8285) and
    committed it, and the trailing eval of the discarded third variant carried
    the parent's sha. The iteration's reading is its BEST train eval; nothing
    is attributed to the closing commit, so a reverted trailing eval can never
    masquerade as the outcome.
    """
    ctx = _closed_context(close_tree="tree_kept", close_sha="commit_kept")
    ctx.initial_subsample_score = 0.3992
    ctx.record_eval("seed_sha", 0.3992, tree="tree_parent")
    ctx.record_eval("seed_sha", 0.7300, tree="tree_v1")
    ctx.record_eval("seed_sha", 0.8285, tree="tree_kept", committed=False)
    ctx.record_eval("seed_sha", 0.7694, tree="tree_discarded")

    result = ctx.finalize(_equivalence_fn())

    assert result.best_subsample_score == 0.8285
    assert ctx.best_subsample_tree == "tree_kept"
    assert result.changes_landed is True
    assert not hasattr(result, "final_subsample_score")


def test_finalize_reports_no_change_when_the_program_did_not_move():
    ctx = _closed_context(parent_tree="tree_same", close_tree="tree_same")
    ctx.initial_subsample_score = 0.7730
    ctx.record_eval("sha_parent", 0.7730, tree="tree_same")
    ctx.record_eval("sha_parent", 0.7878, tree="tree_tried")

    result = ctx.finalize(_equivalence_fn())

    assert result.changes_landed is False
    assert ctx.best_subsample_score == 0.7878, "the tried version is still recorded"


def test_finalize_accepts_a_doc_only_commit_as_the_same_program():
    ctx = _closed_context(parent_tree="tree_parent", close_tree="tree_parent_with_docs")
    ctx.record_eval("sha_p", 0.8715, tree="tree_parent")

    result = ctx.finalize(_equivalence_fn({("tree_parent", "tree_parent_with_docs")}))

    assert result.changes_landed is False


def test_finalize_does_not_score_an_unevaluated_closing_program():
    """No "nearest before" fallback, no head certificate: the closing content
    was never measured, and the record does not pretend otherwise."""
    ctx = _closed_context(close_tree="tree_never_run")
    ctx.initial_subsample_score = 0.60
    ctx.record_eval("sha_p", 0.60, tree="tree_parent")
    ctx.record_eval("sha_p", 0.71, tree="tree_something_else")

    result = ctx.finalize(_equivalence_fn())

    assert result.best_subsample_score == 0.71
    assert result.changes_landed is True
    assert ctx.score_for_content("tree_never_run", _equivalence_fn()) == (None, "none")


def test_best_ignores_single_rows_and_other_subsamples():
    """`subsample_best` vs `subsample_baseline` must be readings of the SAME
    locked rows: `single` evals and train evals on a redrawn sample are out."""
    ctx = _closed_context()
    ctx.initial_subsample_score = 0.50
    ctx.baseline_sample_key = "50:abc"
    ctx.record_eval("sha", 0.50, tree="tree_parent", sample_key="50:abc")
    ctx.record_eval("sha", 0.99, tree="tree_x", eval_sample="single", sample_key="1:zzz")
    ctx.record_eval("sha", 0.95, tree="tree_y", sample_key="50:other")
    ctx.record_eval("sha", 0.62, tree="tree_close", sample_key="50:abc")

    ctx.finalize(_equivalence_fn())

    assert ctx.best_subsample_score == 0.62
    assert ctx.best_subsample_tree == "tree_close"


def test_comparison_best_is_restricted_to_the_registered_branch():
    ctx = _closed_context()
    ctx.record_eval("sha", 0.50, tree="tree_close")
    ctx.record_eval(
        "sha_inc", 0.91, tree="tree_incumbent", eval_sample="comparison",
        branch="codeevolver-20260414000000-main",
    )
    ctx.record_eval(
        "sha", 0.77, tree="tree_close", eval_sample="comparison",
        branch=ctx.child_branch,
    )

    result = ctx.finalize(_equivalence_fn())

    assert result.comparison_best_score == 0.77
    assert ctx.comparison_best_tree == "tree_close"
    assert ctx.best_subsample_score == 0.50, "comparison evals never score the subsample"


def test_durable_refs_prefer_the_commit_then_the_anchor():
    ctx = _closed_context()
    ctx.record_eval("parent_sha", 0.50, tree="tree_parent", committed=True)
    ctx.record_eval("parent_sha", 0.80, tree="tree_dirty", committed=False)
    ctx.finalize(_equivalence_fn())
    # Before anchoring, the best (never committed) can only point at the sha
    # it was evaluated under.
    assert ctx.best_subsample_sha == "parent_sha"

    ctx.eval_records[1]["anchor_ref"] = "refs/codeevolver/evals/3/1"
    ctx.refresh_durable_refs()
    assert ctx.best_subsample_sha == "refs/codeevolver/evals/3/1"

    ctx.eval_records[1]["committed"] = True
    ctx.eval_records[1]["sha"] = "real_commit"
    ctx.refresh_durable_refs()
    assert ctx.best_subsample_sha == "real_commit"


def test_score_for_content_exact_then_equivalent_newest_first():
    ctx = _closed_context()
    ctx.record_eval("sha", 0.50, tree="tree_close")
    ctx.record_eval("sha", 0.58, tree="tree_close")
    ctx.record_eval("sha", 0.91, tree="tree_close", eval_sample="comparison")
    eq = _equivalence_fn({("tree_close", "tree_close_docs")})

    assert ctx.score_for_content("tree_close", eq) == (0.58, "exact_tree")
    assert ctx.score_for_content("tree_close_docs", eq) == (0.58, "program_equivalent")
    assert ctx.score_for_content("tree_other", eq) == (None, "none")
    assert ctx.score_for_content(None, eq) == (None, "none")


def test_finalize_noop_when_no_branch_was_registered():
    ctx = IterationContext(iteration_idx=3)
    ctx.parent_sha = "parent_sha"
    ctx.record_eval("sha_a", 0.65, tree="tree_a")

    result = ctx.finalize(_equivalence_fn())

    assert result.best_subsample_score == 0.65
    assert result.changes_landed is False


def test_finalize_degrades_to_sha_comparison_when_trees_unreadable():
    """A git failure must never silently report "nothing changed"."""
    ctx = _closed_context(close_sha="advanced_sha")
    ctx.parent_tree = None
    ctx.close_tree = None
    ctx.record_eval("advanced_sha", 0.7, tree=None)

    result = ctx.finalize(_equivalence_fn())

    assert result.changes_landed is True


def test_best_subsample_tracks_the_strongest_version_tried():
    ctx = _closed_context(parent_tree="tree_same", close_tree="tree_same")
    ctx.record_eval("sha_p", 0.60, tree="tree_same")
    ctx.record_eval("sha_p", 0.81, tree="tree_best")
    ctx.record_eval("sha_p", 0.55, tree="tree_worse")

    ctx.finalize(_equivalence_fn())

    assert ctx.best_subsample_score == 0.81
    assert ctx.best_subsample_tree == "tree_best"


def test_changes_landed_false_before_finalize():
    ctx = _closed_context()
    assert ctx.changes_landed is False


def test_history_fields_selected_component_direct_edit():
    ctx = _closed_context(close_tree="tree_edited")
    ctx.initial_subsample_score = 0.60
    ctx.record_eval("parent_sha", 0.60, tree="tree_parent")
    ctx.record_eval("parent_sha", 0.68, tree="tree_edited")
    ctx.finalize(_equivalence_fn())

    # No coding_agent call, but a direct edit landed — still surface as _code.
    assert ctx.history_fields()["selected_component"] == "_code"


def test_history_fields_selected_component_none_when_no_change():
    ctx = _closed_context(parent_tree="tree_same", close_tree="tree_same")
    ctx.initial_subsample_score = 0.60
    ctx.record_eval("parent_sha", 0.60, tree="tree_same")
    ctx.finalize(_equivalence_fn())

    assert ctx.history_fields()["selected_component"] is None


def test_history_fields_shape():
    ctx = IterationContext(iteration_idx=5)
    ctx.child_branch = "codeevolver-20260414000000-abc"
    ctx.parent_branch = "codeevolver-20260414000000-main"
    ctx.parent_sha = "parent_sha"
    ctx.parent_iteration = 2
    ctx.parent_is_primary = True
    ctx.parent_derivation = "ancestor"
    ctx.new_lineage = False
    ctx.claimed_improvement_mode = "exploitation"
    ctx.memory_updated_claimed = True
    ctx.parent_tree = "tree_parent"
    ctx.close_sha = "close_sha_abc"
    ctx.close_tree = "tree_close"
    ctx.initial_subsample_score = 0.40
    ctx.record_eval("parent_sha", 0.40, tree="tree_parent")
    ctx.record_eval("parent_sha", 0.60, tree="tree_close")
    ctx.record_coding_agent("Add re-ranking")
    ctx.record_coding_agent("Tighten prompt on re-ranker")
    ctx.finalize(_equivalence_fn())

    fields = ctx.history_fields()

    assert fields["git_branch"] == "codeevolver-20260414000000-abc"
    assert fields["selected_component"] == "_code"
    assert "Add re-ranking" in fields["change_request"]
    assert "Tighten prompt" in fields["change_request"]
    assert fields["parent_branch"] == "codeevolver-20260414000000-main"
    assert fields["parent_sha"] == "parent_sha"
    assert fields["parent_iteration"] == 2
    assert fields["parent_is_primary"] is True
    assert fields["parent_derivation"] == "ancestor"
    assert fields["new_lineage"] is False
    assert fields["improvement_mode"] == "exploitation"
    assert fields["memory_updated"] is True
    assert fields["closed_by_fallback"] is False
    assert fields["parent_subsample_score"] == 0.40
    assert fields["close_tree"] == "tree_close"
    assert fields["changes_landed"] is True
    assert fields["best_subsample_score"] == 0.60
    assert fields["comparison_best_score"] is None
    assert fields["comparison_state"] is None
    # No score of the closing commit, under any name.
    assert "subsample_score" not in fields
    assert "final_sha" not in fields
    assert "final_score_source" not in fields
    # Engine 2 candidate-index / narrative fields are gone.
    assert "parent_program" not in fields
    assert "selected_was_best" not in fields
    assert "subsample_analysis" not in fields


def test_record_token_usage_accumulates_by_model_and_normalizes():
    ctx = IterationContext(iteration_idx=3)
    # Same model under both the bare and anthropic/ forms must aggregate.
    ctx.record_token_usage("claude-opus-4-8", TokenUsage(input_tokens=10, output_tokens=2))
    ctx.record_token_usage(
        "anthropic/claude-opus-4-8", TokenUsage(input_tokens=5, cache_read_input_tokens=3)
    )
    ctx.record_token_usage(None, TokenUsage(input_tokens=1))

    by_model = ctx.token_usage_by_model
    assert by_model["claude-opus-4-8"]["input_tokens"] == 15
    assert by_model["claude-opus-4-8"]["output_tokens"] == 2
    assert by_model["claude-opus-4-8"]["cache_read_input_tokens"] == 3
    assert by_model["claude-opus-4-8"]["total"] == 20
    assert by_model["unknown"]["input_tokens"] == 1


def test_token_usage_by_model_is_json_serializable():
    import json

    ctx = IterationContext(iteration_idx=3)
    ctx.record_token_usage("claude-opus-4-8", TokenUsage(input_tokens=10))
    # Must survive the JSON round-trip used for SDK session resumption — i.e.
    # stay a plain dict of ints, not a TokenUsage/ledger object.
    restored = json.loads(json.dumps(ctx.token_usage_by_model))
    assert restored == ctx.token_usage_by_model
