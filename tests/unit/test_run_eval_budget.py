"""Copyright © 2026 440 Labs LLC

Unit tests for the `run_eval` eval-row budget: the per-response
`budget_note` / `eval_rows_remaining` fields, and the gate that refuses
training evals once the budget is reached while still allowing one final
comparison-set run for wrap-up.

The gate reads the run-level `ctx.max_eval_rows` and the live
`state.total_num_evals`, so the tests drive both directly and stub the
subprocess eval away. There is no environment switch: SaaS and research runs
get identical behavior.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding, RunEvalError


_TRAINSET = [{"id": i, "category": "metals"} for i in range(20)]


def _make_eval_result(row_ids: list[int]) -> EvalResult:
    return EvalResult(
        scores_by_metric={"accuracy": 0.5},
        primary_score=0.5,
        # One trace_meta line per row: this is what increment_evals counts.
        trace_meta=[{"row_id": r} for r in row_ids],
        trace_dir="/traces/iteration_0_parent",
        trace_index_file="/traces/iteration_0_parent/index.jsonl",
        row_ids=list(row_ids),
        raw={},
    )


class _FakeRunEval:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> EvalResult:
        self.calls.append(kwargs)
        return _make_eval_result(kwargs.get("row_ids") or [])


def _stub_subprocess():
    fake = _FakeRunEval()
    return fake, patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=fake,
        git_get_sha=lambda _ws: "fake_sha",
        git_tree_signature=lambda _ws: "fake_tree",
        git_get_current_branch=lambda _ws: "codeevolver-test-child",
        git_dirty_paths=lambda _ws: [],
    )


def _make_state(used: int, tmp_path: Path) -> CEState:
    state = CEState(
        seed_branch="seed",
        seed_sha="sha",
        comparison_history_path=str(tmp_path / "comparison_set_history.jsonl"),
    )
    state.increment_evals(used)
    return state


def _make_binding(
    tmp_path: Path,
    *,
    used: int,
    max_eval_rows: int | None = 100,
    state: CEState | None = None,
    comparison_ids: list[int] | None = None,
    subsample_size: int = 10,
    comparison_closes_iteration: bool = False,
) -> tuple[RunEvalBinding, CEState]:
    (tmp_path / "train.json").write_text(json.dumps(_TRAINSET))
    ctx = IterationContext(
        iteration_idx=0,
        parent_locked=True,
        max_eval_rows=max_eval_rows,
        evals_used=used,
        subsample_size=subsample_size,
    )
    st = state if state is not None else _make_state(used, tmp_path)
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=str(tmp_path),
        config=EvaluatorConfig(
            program_factory="my_pkg.program:make_program",
            metric_fns=["my_pkg.metrics:accuracy"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path="train.json",
        ),
        trainset_ids=list(range(len(_TRAINSET))),
        comparison_ids=[0, 1, 2] if comparison_ids is None else comparison_ids,
        rng=random.Random(0),
        subsample_size=subsample_size,
        state=st,
        comparison_closes_iteration=comparison_closes_iteration,
    )
    return binding, st


# --- budget note ------------------------------------------------------------


def test_under_budget_note_reports_rows_remaining(tmp_path):
    binding, state = _make_binding(tmp_path, used=50, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    # 10 rows evaluated → 50 + 10 = 60 used, 40 left.
    assert state.total_num_evals == 60
    assert resp["eval_rows_remaining"] == 40
    assert resp["budget_note"] == "40 rows remaining."
    # No mention of the bonus while budget remains.
    assert "comparison set run" not in resp["budget_note"]


def test_overshoot_note_clamps_to_zero_and_offers_final_comparison(tmp_path):
    binding, state = _make_binding(tmp_path, used=95, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert state.total_num_evals == 105  # overshoot is allowed
    assert resp["eval_rows_remaining"] == 0
    assert resp["budget_note"].startswith(
        "0 rows remaining. 1 final comparison set run remaining"
    )


def test_no_budget_configured_emits_no_budget_fields(tmp_path):
    binding, _ = _make_binding(tmp_path, used=0, max_eval_rows=None)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert "budget_note" not in resp
    assert "eval_rows_remaining" not in resp


def test_note_is_independent_of_ce_research_mode(tmp_path, monkeypatch):
    """One rule everywhere — the env var must not change what the agent reads."""
    notes = []
    for value in ("true", "false", ""):
        monkeypatch.setenv("CE_RESEARCH_MODE", value)
        binding, _ = _make_binding(tmp_path, used=95, max_eval_rows=100)
        fake, ctxm = _stub_subprocess()
        with ctxm:
            notes.append(binding("train")["budget_note"])
    assert len(set(notes)) == 1
    assert notes[0].startswith(
        "0 rows remaining. 1 final comparison set run remaining"
    )


def test_note_omits_the_allowance_when_there_is_no_comparison_set(tmp_path):
    binding, _ = _make_binding(
        tmp_path, used=95, max_eval_rows=100, comparison_ids=[]
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert resp["budget_note"] == "0 rows remaining."
    assert "final_comparison_runs_remaining" not in resp


# --- the gate ---------------------------------------------------------------


def test_train_allowed_at_one_row_under_budget(tmp_path):
    """990/1000 may still run a subsample of any size."""
    binding, state = _make_binding(
        tmp_path, used=99, max_eval_rows=100, subsample_size=20
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert resp["ok"] is True
    assert state.total_num_evals == 119


def test_train_refused_when_budget_reached(tmp_path):
    binding, state = _make_binding(tmp_path, used=100, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="Eval budget reached"):
        binding("train")
    assert fake.calls == []  # refused BEFORE running
    assert state.total_num_evals == 100  # and without spending rows


def test_single_refused_when_budget_reached(tmp_path):
    binding, _ = _make_binding(tmp_path, used=120, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="no further training evals"):
        binding("single")
    assert fake.calls == []


def test_legacy_subsample_alias_is_refused_too(tmp_path):
    binding, _ = _make_binding(tmp_path, used=100, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="Eval budget reached"):
        binding("subsample")
    assert fake.calls == []


def test_one_final_comparison_allowed_then_refused(tmp_path):
    binding, state = _make_binding(tmp_path, used=100, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison")
    assert resp["ok"] is True
    assert state.final_comparison_evals == 1
    assert resp["eval_rows_remaining"] == 0
    assert resp["final_comparison_runs_remaining"] == 0
    assert "0 final comparison set runs remaining" in resp["budget_note"]

    with ctxm, pytest.raises(RunEvalError, match="already been run"):
        binding("comparison")
    assert len(fake.calls) == 1
    assert state.final_comparison_evals == 1


def test_train_message_drops_the_offer_once_the_final_comparison_is_spent(
    tmp_path,
):
    binding, state = _make_binding(tmp_path, used=100, max_eval_rows=100)
    state.note_final_comparison_eval()
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="No further evals of any kind"):
        binding("train")
    assert fake.calls == []


def test_under_budget_comparison_does_not_spend_the_allowance(tmp_path):
    binding, state = _make_binding(tmp_path, used=50, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        binding("comparison")
    assert state.final_comparison_evals == 0


def test_gate_is_independent_of_ce_research_mode(tmp_path, monkeypatch):
    monkeypatch.setenv("CE_RESEARCH_MODE", "")
    binding, _ = _make_binding(tmp_path, used=500, max_eval_rows=100)
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="Eval budget reached"):
        binding("train")
    assert fake.calls == []


def test_run_without_comparison_set_reports_the_missing_set_not_the_budget(
    tmp_path,
):
    """A run with no comparison set configured — the error must say so."""
    binding, _ = _make_binding(
        tmp_path, used=100, max_eval_rows=100, comparison_ids=[]
    )
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="no comparison set is configured"):
        binding("comparison")
    with ctxm, pytest.raises(RunEvalError, match="No further evals of any kind"):
        binding("train")


# --- state persistence ------------------------------------------------------


def test_final_comparison_counter_round_trips(tmp_path):
    state = _make_state(100, tmp_path)
    state.note_final_comparison_eval()
    path = str(tmp_path / "ce_state.json")
    state.save(path)
    assert CEState.load(path).final_comparison_evals == 1


def test_state_file_written_before_the_counter_existed_loads_as_zero(tmp_path):
    state = _make_state(100, tmp_path)
    path = tmp_path / "ce_state.json"
    payload = state.to_json_dict()
    payload.pop("final_comparison_evals")
    path.write_text(json.dumps(payload))
    assert CEState.load(str(path)).final_comparison_evals == 0


# --- close_requires_train_eval: the 5c affordability predicate ---------------
#
# `close_iteration` check 5c demands a train baseline. A baseline is only
# meaningful as one leg of a before/after PAIR, so the gate is waived once a
# pair is unaffordable — otherwise the architect is told to run a train eval
# that `run_eval` refuses, and the iteration cannot close at all.


def test_close_requires_train_eval_while_a_train_pair_is_affordable(tmp_path):
    # 20 rows left, subsample 10 → exactly one before/after pair. Enforced.
    binding, _ = _make_binding(tmp_path, used=80, max_eval_rows=100)
    assert binding.close_requires_train_eval() is True


def test_close_waives_the_train_eval_one_row_below_a_full_pair(tmp_path):
    # 19 rows left → the pair no longer fits. This is the exact boundary
    # `prompts._budget_note` uses to tell the architect it has "no room for a
    # before/after train pair", so the prompt and the gate agree.
    binding, _ = _make_binding(tmp_path, used=81, max_eval_rows=100)
    assert binding.close_requires_train_eval() is False


def test_close_waives_the_train_eval_when_the_budget_is_spent(tmp_path):
    # Budget exhaustion needs no separate branch: 0 remaining is already below
    # 2x subsample.
    binding, _ = _make_binding(tmp_path, used=100, max_eval_rows=100)
    assert binding.close_requires_train_eval() is False


def test_close_requires_train_eval_when_no_budget_is_configured(tmp_path):
    # No budget → there is always room for a pair; never waive.
    binding, _ = _make_binding(tmp_path, used=10_000, max_eval_rows=None)
    assert binding.close_requires_train_eval() is True


def test_close_waives_the_train_eval_once_the_comparison_latch_is_engaged(tmp_path):
    """The comparison-latch deadlock.

    On `comparison_closes_iteration` arms an agent-run comparison refuses every
    later eval, so no train eval is reachable — at ANY budget level. The waiver
    must not depend on the budget.
    """
    binding, _ = _make_binding(
        tmp_path, used=0, max_eval_rows=100, comparison_closes_iteration=True
    )
    assert binding.close_requires_train_eval() is True  # full budget, no comparison yet

    binding.ctx.record_eval("sha", 0.9, eval_sample="comparison", tree="tree")
    assert binding.close_requires_train_eval() is False


def test_a_loop_run_comparison_does_not_waive_the_train_eval(tmp_path):
    """`by_loop` comparisons do not engage the latch, so they must not waive 5c.

    Mirrors the refusal in `__call__`, which also excludes `by_loop` records —
    the gate and the refusal read the same condition.
    """
    binding, _ = _make_binding(
        tmp_path, used=0, max_eval_rows=100, comparison_closes_iteration=True
    )
    binding.ctx.record_eval(
        "sha", 0.9, eval_sample="comparison", tree="tree", by_loop=True
    )
    assert binding.close_requires_train_eval() is True


def test_the_comparison_latch_is_inert_on_arms_that_do_not_set_it(tmp_path):
    binding, _ = _make_binding(
        tmp_path, used=0, max_eval_rows=100, comparison_closes_iteration=False
    )
    binding.ctx.record_eval("sha", 0.9, eval_sample="comparison", tree="tree")
    assert binding.close_requires_train_eval() is True
