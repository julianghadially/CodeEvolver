"""Copyright © 2026 440 Labs LLC

Controls for the pareto_evolution fixed rules (`pareto_optimizer`):

- admission is the EXISTENCE of a comparison-set eval of the
  closing content — agent-run (EXACT / PROGRAM_EQUIVALENT), reverted-to, or
  the loop fallback. A train tie or regression is NOT a barrier: an agent
  that ran the comparison on a non-improved iteration is admitted;
- the loop's fallback comparison (RERUN_BY_LOOP, counted) fires only when the
  agent ran none AND the train subsample strictly progressed; an iteration
  with neither is `no_comparison` and spends nothing;
- constraint-gate reverts and no-change closes never spend a comparison eval
  and never join the population;
- the primary after every iteration is the population's best comparison-set
  score; `was_primary` / `is_primary` / `selection_policy` are stamped;
- the seed hook measures the seed on the comparison set, COUNTED against the
  budget, and seeds candidate 0.
"""

from __future__ import annotations

import random
from typing import Any
from unittest.mock import patch

import pytest

from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.pareto_evolution import pareto_optimizer as po
from src.engine.optimizer.v3.pareto_evolution.population import Population
from src.engine.optimizer.v3.strategy import LoopServices

_MAIN = "codeevolver-20260601120000-main"
_PO = "src.engine.optimizer.v3.pareto_evolution.pareto_optimizer"


@pytest.fixture
def history_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_HISTORY_DIR", str(tmp_path / "history"))
    return tmp_path / "history"


def _services(rng=None, comparison_ids=(0, 1, 2)) -> LoopServices:
    return LoopServices(
        workspace="/ws",
        evaluator_config=None,  # type: ignore[arg-type]
        trainset_ids=list(range(10)),
        comparison_ids=list(comparison_ids),
        comparison_timeout=None,
        run_timestamp="20260601120000",
        rng=rng or random.Random(0),
        max_eval_rows=100,
    )


def _seed_population(rows=None) -> Population:
    pop = Population()
    pop.add(
        {
            "iteration": 0, "branch": _MAIN, "sha": "seedsha", "tree": "seedtree",
            "parent_iteration": None,
            "row_scores": rows or {"0": 1.0, "1": 0.0, "2": 0.0},
            "comparison_set_score": 1 / 3,
            "train_baseline": None, "train_best": None,
        }
    )
    pop.save()
    return pop


def _ctx(it=1, *, baseline=0.5, final=0.7, landed=True, child=None) -> IterationContext:
    child = child or f"codeevolver-20260601120000-iter{it}"
    ctx = IterationContext(
        iteration_idx=it, parent_locked=True,
        child_branch=child, parent_branch=_MAIN, parent_sha="seedsha",
        parent_tree="seedtree", parent_iteration=0, parent_is_primary=True,
        primary_branch_at_start=_MAIN, close_sha=f"sha{it}", close_tree=f"tree{it}",
    )
    ctx.initial_subsample_score = baseline
    ctx.best_subsample_score = final
    ctx.changes_landed = landed
    return ctx


def _fake_loop_eval(rows: dict[str, float], score: float):
    calls: list[dict[str, Any]] = []

    def _run(*, ctx, state, services, ref):
        calls.append({"ref": ref})
        state.increment_evals(len(rows))
        ctx.record_eval(
            ctx.close_sha, score, eval_sample="comparison", branch=ctx.child_branch,
            n_rows=len(rows), tree=ctx.close_tree, row_scores=rows,
            eval_id="loop-eval", by_loop=True,
        )
        from src.engine.evaluator.sandbox_client import EvalResult

        return EvalResult(
            scores_by_metric={"accuracy": score}, primary_score=score,
            trace_meta=[{"example_id": k, "score": v} for k, v in rows.items()],
            trace_dir="/t", trace_index_file="/ti", row_ids=list(range(len(rows))), raw={},
        )

    return _run, calls


def _close(ctx, state, rec=None, *, loop_rows=None, loop_score=0.5):
    rec = dict(rec or {"iteration": ctx.iteration_idx, "primary_branch_after": _MAIN})
    run, calls = _fake_loop_eval(loop_rows or {"0": 1.0, "1": 1.0, "2": 0.0}, loop_score)
    with patch(f"{_PO}.run_loop_comparison_eval", run), patch(
        f"{_PO}.diff_is_non_program_only", lambda ws, a, b: a == b
    ):
        po.evo_on_iteration_closed(ctx, state, rec, _services())
    return rec, calls


def test_agent_measured_closing_content_costs_the_loop_nothing(history_dir):
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    ctx.record_eval(
        "sha1", 0.6, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="tree1", row_scores={"0": 1.0, "1": 1.0, "2": 0.0}, eval_id="agent-eval",
    )
    rec, calls = _close(ctx, state)
    assert calls == []
    assert rec["candidate_admitted"] is True
    assert rec["admission_reason"] == "agent_comparison"
    assert rec["comparison_set_score"] == 0.6
    assert rec["comparison_state"] == "EXACT"
    assert rec["comparison_eval_id_at_close"] == "agent-eval"
    assert rec["comparison_score_at_close"] == 0.6
    assert ctx.comparison_state == "EXACT"
    assert state.loop_comparison_reruns == 0
    assert rec["selected_parent_iteration"] == 0
    assert rec["selection_policy"] == "pareto_sample"
    assert rec["primary_branch_after"] == ctx.child_branch
    assert rec["was_primary"] is True and rec["is_primary"] is True
    pop = Population.load_or_none()
    assert 1 in pop and pop.get(1)["row_scores"] == {"0": 1.0, "1": 1.0, "2": 0.0}
    assert pop.get(1)["train_baseline"] == 0.5 and pop.get(1)["train_best"] == 0.7
    assert state.total_num_evals == 0


def test_missing_comparison_eval_is_run_once_by_the_loop_and_counted(history_dir):
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    rec, calls = _close(ctx, state, loop_score=0.66)
    assert calls == [{"ref": ctx.child_branch}]
    assert state.total_num_evals == 3
    assert rec["candidate_admitted"] is True
    assert rec["admission_reason"] == "loop_comparison"
    assert rec["comparison_set_score"] == 0.66
    assert rec["comparison_state"] == "RERUN_BY_LOOP"
    assert rec["comparison_eval_id_at_close"] == "loop-eval"
    assert rec["comparison_scores"][-1]["branch"] == ctx.child_branch
    assert rec["comparison_scores"][-1]["eval_id"] == "loop-eval"
    assert state.loop_comparison_reruns == 1 and state.loop_comparison_rows == 3
    pop = Population.load_or_none()
    assert pop.best()["iteration"] == 1
    assert pop.get(1)["train_best"] > pop.get(1)["train_baseline"]


def test_post_comparison_edits_are_reverted_to_the_comparison_point(history_dir):
    """specs/archive-public/comparison_credits.md: the comparison run is the close step —
    program changes after it are reverted, not re-measured."""
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    ctx.record_eval(
        "cmp_sha", 0.9, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="tree_before_more_edits", row_scores={"0": 1.0, "1": 1.0, "2": 1.0},
        eval_id="agent-eval",
    )
    rec = {"iteration": 1, "primary_branch_after": _MAIN}
    run, calls = _fake_loop_eval({"0": 1.0}, 0.1)

    def _fake_revert(c, *, workspace, target_sha):
        assert target_sha == "cmp_sha"
        c.close_sha = target_sha
        c.close_tree = "tree_before_more_edits"
        return True

    with patch(f"{_PO}.run_loop_comparison_eval", run), patch(
        f"{_PO}.diff_is_non_program_only", lambda ws, a, b: a == b
    ), patch(f"{_PO}.revert_to_comparison_point", _fake_revert):
        po.evo_on_iteration_closed(ctx, state, rec, _services())
    assert calls == []  # no loop rerun — the agent's reading stands
    assert rec["candidate_admitted"] is True
    assert rec["admission_reason"] == "reverted_to_comparison_point"
    assert rec["comparison_state"] == "REVERTED_TO_COMPARISON_POINT"
    assert rec["comparison_set_score"] == 0.9
    assert rec["comparison_eval_id_at_close"] == "agent-eval"
    assert rec["close_sha"] == "cmp_sha"
    assert rec["close_tree"] == "tree_before_more_edits"
    assert state.loop_comparison_reruns == 0
    pop = Population.load_or_none()
    assert pop.get(1)["row_scores"] == {"0": 1.0, "1": 1.0, "2": 1.0}


def test_failed_revert_falls_back_to_a_loop_rerun(history_dir):
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    ctx.record_eval(
        "cmp_sha", 0.9, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="tree_before_more_edits", row_scores={"0": 1.0, "1": 1.0, "2": 1.0},
    )
    rec = {"iteration": 1, "primary_branch_after": _MAIN}
    run, calls = _fake_loop_eval({"0": 1.0, "1": 1.0, "2": 0.0}, 0.4)
    with patch(f"{_PO}.run_loop_comparison_eval", run), patch(
        f"{_PO}.diff_is_non_program_only", lambda ws, a, b: a == b
    ), patch(f"{_PO}.revert_to_comparison_point", lambda *a, **k: False):
        po.evo_on_iteration_closed(ctx, state, rec, _services())
    assert len(calls) == 1
    assert rec["comparison_set_score"] == 0.4
    assert rec["comparison_state"] == "RERUN_BY_LOOP"
    assert rec["admission_reason"] == "loop_comparison"


def test_doc_only_difference_maps_as_program_equivalent(history_dir):
    """A codeevolver.md-only difference is the same program: no re-run."""
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    ctx.record_eval(
        "sha1", 0.8, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="tree1_before_docs", row_scores={"0": 1.0, "1": 1.0, "2": 0.0},
    )
    rec = {"iteration": 1, "primary_branch_after": _MAIN}
    run, calls = _fake_loop_eval({"0": 1.0}, 0.1)
    with patch(f"{_PO}.run_loop_comparison_eval", run), patch(
        f"{_PO}.diff_is_non_program_only",
        lambda ws, a, b: a == b or {a, b} == {"tree1_before_docs", "tree1"},
    ):
        po.evo_on_iteration_closed(ctx, state, rec, _services())
    assert calls == []
    assert rec["comparison_state"] == "PROGRAM_EQUIVALENT"
    assert rec["comparison_set_score"] == 0.8


@pytest.mark.parametrize(
    "ctx_kw, rec_extra, reason",
    [
        ({"final": 0.5}, {}, "no_comparison"),   # tie, agent ran no comparison
        ({"final": 0.3}, {}, "no_comparison"),   # regression, agent ran no comparison
        ({"final": None}, {}, "no_comparison"),  # never train-evaluated, no comparison
        ({"landed": False}, {}, "no_change"),
        ({}, {"constraint_violation": "hardcoded answers"}, "constraint_violation"),
    ],
)
def test_non_admitted_iterations_spend_nothing_and_keep_best_primary(
    history_dir, ctx_kw, rec_extra, reason
):
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx(**ctx_kw)
    rec, calls = _close(ctx, state, rec={"iteration": 1, **rec_extra})
    assert calls == []
    assert rec["candidate_admitted"] is False
    assert rec["admission_reason"] == reason
    assert rec["comparison_set_score"] is None
    assert rec["comparison_state"] is None
    assert rec["primary_branch_after"] == _MAIN
    assert rec["is_primary"] is False
    assert len(Population.load_or_none()) == 1
    assert state.total_num_evals == 0
    assert state.loop_comparison_reruns == 0


@pytest.mark.parametrize("final", [0.5, 0.3, None])
def test_agent_comparison_admits_non_improved_iteration(history_dir, final):
    """A train tie with an agent-run comparison eval is admitted: the
    comparison eval is the admission criterion; train evidence is
    informational."""
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx(final=final)
    ctx.record_eval(
        "sha1", 0.812, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="tree1", row_scores={"0": 1.0, "1": 1.0, "2": 0.0}, eval_id="agent-eval",
    )
    rec, calls = _close(ctx, state)
    assert calls == [] and state.total_num_evals == 0
    assert rec["candidate_admitted"] is True
    assert rec["admission_reason"] == "agent_comparison"
    assert rec["comparison_state"] == "EXACT"
    assert rec["comparison_set_score"] == 0.812
    pop = Population.load_or_none()
    assert 1 in pop
    assert pop.get(1)["train_baseline"] == 0.5 and pop.get(1)["train_best"] == final
    # It beats the seed (1/3) on the comparison set, so it IS the primary now.
    assert rec["primary_branch_after"] == ctx.child_branch and rec["is_primary"] is True


def test_reverted_comparison_point_equal_to_parent_is_no_change(history_dir):
    """The agent compared the unchanged parent, then edited: the revert lands on
    parent content, which the population already holds — not a candidate, and
    no loop rerun is paid for."""
    _seed_population()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx()
    ctx.record_eval(
        "seedsha", 0.4, eval_sample="comparison", branch=ctx.child_branch, n_rows=3,
        tree="seedtree", row_scores={"0": 1.0, "1": 0.0, "2": 0.0}, eval_id="agent-eval",
    )
    rec = {"iteration": 1, "primary_branch_after": _MAIN}
    run, calls = _fake_loop_eval({"0": 1.0}, 0.1)

    def _fake_revert(c, *, workspace, target_sha):
        assert target_sha == "seedsha"
        c.close_sha, c.close_tree, c.changes_landed = target_sha, "seedtree", False
        return True

    with patch(f"{_PO}.run_loop_comparison_eval", run), patch(
        f"{_PO}.diff_is_non_program_only", lambda ws, a, b: a == b
    ), patch(f"{_PO}.revert_to_comparison_point", _fake_revert):
        po.evo_on_iteration_closed(ctx, state, rec, _services())
    assert calls == []
    assert rec["candidate_admitted"] is False
    assert rec["admission_reason"] == "no_change"
    assert rec["comparison_state"] is None
    assert rec["changes_landed"] is False
    assert len(Population.load_or_none()) == 1


def test_primary_is_best_comparison_score_not_latest(history_dir):
    pop = _seed_population()
    pop.add(
        {
            "iteration": 1, "branch": "codeevolver-20260601120000-iter1", "sha": "sha1",
            "tree": "tree1", "parent_iteration": 0,
            "row_scores": {"0": 1.0, "1": 1.0, "2": 1.0}, "comparison_set_score": 1.0,
            "train_baseline": 0.5, "train_best": 0.7,
        }
    )
    pop.save()
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    ctx = _ctx(it=2)
    ctx.exploit_phase = True
    rec, calls = _close(ctx, state, loop_rows={"0": 1.0, "1": 0.0, "2": 0.0}, loop_score=1 / 3)
    assert rec["candidate_admitted"] is True
    assert rec["selection_policy"] == "best"
    assert rec["primary_branch_after"] == "codeevolver-20260601120000-iter1"
    assert rec["is_primary"] is False
    assert len(Population.load_or_none()) == 3


def test_seed_hook_counts_comparison_eval_and_seeds_candidate_zero(history_dir):
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    history = {0: {}}
    run, calls = _fake_loop_eval({"0": 1.0, "1": 0.0, "2": 1.0}, 2 / 3)

    def _seed_run(*, ctx, state, services, ref):
        ctx.close_sha, ctx.close_tree, ctx.child_branch = "seedsha", "seedtree", ref
        return run(ctx=ctx, state=state, services=services, ref=ref)

    with patch(f"{_PO}.run_loop_comparison_eval", _seed_run):
        po.evo_on_seed_complete(state, history, _services())  # type: ignore[arg-type]
    assert calls == [{"ref": _MAIN}]
    assert state.total_num_evals == 3
    pop = Population.load_or_none()
    assert list(pop.candidates) == [0]
    assert pop.get(0)["branch"] == _MAIN and pop.get(0)["sha"] == "seedsha"
    assert pop.get(0)["row_scores"] == {"0": 1.0, "1": 0.0, "2": 1.0}
    assert history[0]["comparison_score"] == pytest.approx(2 / 3)


def test_seed_hook_requires_comparison_set(history_dir):
    state = CEState(seed_branch=_MAIN, seed_sha="seedsha")
    with pytest.raises(RuntimeError, match="comparison set"):
        po.evo_on_seed_complete(state, {}, _services(comparison_ids=()))  # type: ignore[arg-type]
