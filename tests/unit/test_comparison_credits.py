"""Copyright © 2026 440 Labs LLC

Unit tests for the comparison-credit ledger (specs/archive-public/comparison_credits.md):
the acknowledgement round-trip (per-iteration verification code), the
credit gate on agent-initiated comparison evals, the loop-run overdraft,
the final-allowance exemption, the disabled-ledger path, and CEState
persistence (a resume cannot re-earn credits).

Fixture pattern mirrors `test_run_eval_budget.py`: real `CEState`, real
`RunEvalBinding`, subprocess eval stubbed away.
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
from src.engine.optimizer.v3.run_eval_tool import (
    RunEvalBinding,
    RunEvalError,
    STANDARD_COMPARISON_ACK_NOTE,
)

_TRAINSET = [{"id": i} for i in range(20)]
_CODE = "42"


def _make_eval_result(row_ids: list[int]) -> EvalResult:
    return EvalResult(
        scores_by_metric={"accuracy": 0.5},
        primary_score=0.5,
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


def _make_binding(
    tmp_path: Path,
    *,
    credits: int = 1,
    ack_code: str = _CODE,
    used: int = 0,
    max_eval_rows: int | None = 100,
    comparison_closes_iteration: bool = False,
    comparison_ack_note: str | None = None,
) -> tuple[RunEvalBinding, CEState, IterationContext]:
    (tmp_path / "train.json").write_text(json.dumps(_TRAINSET))
    ctx = IterationContext(
        iteration_idx=1,
        parent_locked=True,
        max_eval_rows=max_eval_rows,
        evals_used=used,
        subsample_size=5,
        comparison_ack_code=ack_code,
    )
    state = CEState(
        seed_branch="seed",
        seed_sha="sha",
        comparison_history_path=str(tmp_path / "comparison_set_history.jsonl"),
    )
    state.increment_evals(used)
    state.grant_comparison_credits(credits)
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
        comparison_ids=[0, 1, 2],
        rng=random.Random(0),
        subsample_size=5,
        state=state,
        comparison_ack_note=comparison_ack_note,
        comparison_closes_iteration=comparison_closes_iteration,
    )
    return binding, state, ctx


# --- acknowledgement --------------------------------------------------------


def test_first_comparison_call_returns_note_and_code_and_spends_nothing(tmp_path):
    binding, state, ctx = _make_binding(tmp_path)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison")
    assert resp["ok"] is False
    assert resp["refused"] == "acknowledgement_required"
    assert resp["verification_code"] == _CODE
    assert _CODE in resp["next_step"]
    # Live values templated into the note.
    assert "3 rows per run" in resp["note"]
    assert "Remaining credits: 1" in resp["note"]
    assert fake.calls == []
    assert state.total_num_evals == 0
    assert state.comparison_credits_balance == 1
    assert ctx.comparison_ack_given is False


def test_correct_code_runs_debits_and_reports_balance(tmp_path):
    binding, state, ctx = _make_binding(tmp_path, credits=2)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison", acknowledge_message=_CODE)
    assert resp["ok"] is True
    assert len(fake.calls) == 1
    assert state.comparison_credits_balance == 1
    assert resp["comparison_credits_remaining"] == 1
    assert ctx.comparison_ack_given is True


def test_ack_lasts_the_full_iteration(tmp_path):
    binding, state, _ = _make_binding(tmp_path, credits=2)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        binding("comparison", acknowledge_message=_CODE)
        resp = binding("comparison")  # no code needed the second time
    assert resp["ok"] is True
    assert len(fake.calls) == 2
    assert state.comparison_credits_balance == 0


def test_wrong_code_is_refused_without_spending(tmp_path):
    binding, state, _ = _make_binding(tmp_path)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison", acknowledge_message="99")
    assert resp["ok"] is False and resp["refused"] == "acknowledgement_required"
    assert fake.calls == [] and state.comparison_credits_balance == 1


def test_arm_specific_note_is_used(tmp_path):
    note = "ARM NOTE {comparison_set_size}/{subsample_size} c={credits} r={rows_remaining}"
    binding, _, _ = _make_binding(tmp_path, comparison_ack_note=note)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison")
    assert resp["note"] == "ARM NOTE 3/5 c=1 r=100"


# --- the credit gate --------------------------------------------------------


def test_comparison_refused_at_zero_balance_before_the_ack(tmp_path):
    binding, state, _ = _make_binding(tmp_path, credits=0)
    fake, ctxm = _stub_subprocess()
    with ctxm, pytest.raises(RunEvalError, match="0 comparison credit"):
        binding("comparison")
    assert fake.calls == [] and state.total_num_evals == 0


def test_train_evals_never_touch_the_ledger(tmp_path):
    binding, state, _ = _make_binding(tmp_path, credits=0)
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert resp["ok"] is True
    assert state.comparison_credits_balance == 0
    assert "comparison_credits_remaining" not in resp


def test_disabled_ledger_runs_comparisons_without_ack_or_debit(tmp_path):
    binding, state, _ = _make_binding(tmp_path, credits=0, ack_code="")
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison")
    assert resp["ok"] is True
    assert len(fake.calls) == 1
    assert state.comparison_credits_balance == 0
    assert "comparison_credits_remaining" not in resp


def test_final_allowance_comparison_needs_ack_but_no_credits(tmp_path):
    """Uniform ack contract: the post-budget final comparison requires
    the acknowledgement code like every other comparison — only the credit
    balance check and the debit stay exempt. A refusal must NOT burn the
    one-shot allowance."""
    binding, state, ctx = _make_binding(
        tmp_path, credits=0, used=100, max_eval_rows=100
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        refusal = binding("comparison")  # no code — refused, nothing spent
    assert refusal["ok"] is False
    assert refusal["refused"] == "acknowledgement_required"
    assert "reserved final" in refusal["note"]
    assert state.final_comparison_evals == 0  # allowance untouched
    assert len(fake.calls) == 0
    with ctxm:
        resp = binding("comparison", acknowledge_message=ctx.comparison_ack_code)
    assert resp["ok"] is True
    assert state.final_comparison_evals == 1
    assert state.comparison_credits_balance == 0  # not debited


def test_loop_run_overdraws_instead_of_refusing(tmp_path):
    from src.engine.optimizer.v3 import loop_eval
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig
    from src.engine.optimizer.v3.strategy import LoopServices

    binding, state, ctx = _make_binding(tmp_path, credits=0)
    services = LoopServices(
        workspace=str(tmp_path),
        evaluator_config=binding.config,
        trainset_ids=binding.trainset_ids,
        comparison_ids=[0, 1, 2],
        comparison_timeout=None,
        run_timestamp="20260601120000",
        rng=random.Random(0),
        max_eval_rows=100,
        config=OptimizationConfig(program="p", metric="m"),
    )
    fake, ctxm = _stub_subprocess()
    with ctxm, patch.object(loop_eval, "clean_python_bytecode"), patch.object(
        loop_eval, "git_checkout"
    ):
        result = loop_eval.run_loop_comparison_eval(
            ctx=ctx, state=state, services=services, ref="some-branch"
        )
    assert result.primary_score == 0.5
    assert state.comparison_credits_balance == -1


def test_loop_run_does_not_debit_when_ledger_disabled(tmp_path):
    from src.engine.optimizer.v3 import loop_eval
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig
    from src.engine.optimizer.v3.strategy import LoopServices

    binding, state, ctx = _make_binding(tmp_path, credits=0, ack_code="")
    cfg = OptimizationConfig(program="p", metric="m")
    cfg.comparison_credits_per_iteration = None
    services = LoopServices(
        workspace=str(tmp_path),
        evaluator_config=binding.config,
        trainset_ids=binding.trainset_ids,
        comparison_ids=[0, 1, 2],
        comparison_timeout=None,
        run_timestamp="20260601120000",
        rng=random.Random(0),
        max_eval_rows=100,
        config=cfg,
    )
    fake, ctxm = _stub_subprocess()
    with ctxm, patch.object(loop_eval, "clean_python_bytecode"), patch.object(
        loop_eval, "git_checkout"
    ):
        loop_eval.run_loop_comparison_eval(
            ctx=ctx, state=state, services=services, ref="some-branch"
        )
    assert state.comparison_credits_balance == 0


# --- evo stop latch ---------------------------------------------------------


def test_comparison_closes_iteration_refuses_all_evals_after_the_run(tmp_path):
    binding, state, ctx = _make_binding(
        tmp_path, credits=2, comparison_closes_iteration=True
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison", acknowledge_message=_CODE)
    assert resp["ok"] is True
    for sample in ("train", "single", "comparison"):
        with ctxm, pytest.raises(RunEvalError, match="close step"):
            binding(sample)
    assert len(fake.calls) == 1
    assert state.comparison_credits_balance == 1


def test_loop_comparison_record_does_not_trip_the_stop_latch(tmp_path):
    binding, _, ctx = _make_binding(
        tmp_path, credits=2, comparison_closes_iteration=True
    )
    ctx.record_eval(
        "sha", 0.5, eval_sample="comparison", n_rows=3, by_loop=True
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("train")
    assert resp["ok"] is True


# --- balance schedule + persistence -----------------------------------------


def test_initial_balance_math_matches_the_spec(tmp_path):
    """evo/ASA spend the up-front credit on the seed eval and start iteration
    1 at balance 1; arms with no seed eval start at 2."""
    seedful = CEState(seed_branch="s", seed_sha="x")
    seedful.grant_comparison_credits(1)   # initial grant
    seedful.debit_comparison_credit()     # seed hook comparison
    seedful.grant_comparison_credits(1)   # iteration 1 start
    assert seedful.comparison_credits_balance == 1

    seedless = CEState(seed_branch="s", seed_sha="x")
    seedless.grant_comparison_credits(1)
    seedless.grant_comparison_credits(1)
    assert seedless.comparison_credits_balance == 2


def test_balance_round_trips_so_a_resume_cannot_re_earn(tmp_path):
    state = CEState(seed_branch="s", seed_sha="x")
    state.grant_comparison_credits(3)
    state.debit_comparison_credit()
    path = str(tmp_path / "ce_state.json")
    state.save(path)
    assert CEState.load(path).comparison_credits_balance == 2


def test_config_knobs_round_trip_and_validate():
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    cfg = OptimizationConfig(program="p", metric="m")
    assert cfg.comparison_credits_initial == 1
    assert cfg.comparison_credits_per_iteration == 1
    d = cfg.to_dict()
    assert d["comparison_credits_initial"] == 1
    assert d["comparison_credits_per_iteration"] == 1
    d["repo_url"] = None
    cfg2 = OptimizationConfig.from_dict({**d, "program": "p", "metric": "m"})
    assert cfg2.comparison_credits_per_iteration == 1

    with pytest.raises(ValueError, match="comparison_credits_initial"):
        OptimizationConfig(program="p", metric="m", comparison_credits_initial=-1)
    with pytest.raises(ValueError, match="comparison_credits_per_iteration"):
        OptimizationConfig(
            program="p", metric="m", comparison_credits_per_iteration=-2
        )
    disabled = OptimizationConfig(
        program="p", metric="m", comparison_credits_per_iteration=None
    )
    assert disabled.comparison_credits_per_iteration is None


def test_standard_note_placeholders_all_resolve():
    text = STANDARD_COMPARISON_ACK_NOTE.format(
        comparison_set_size=200, subsample_size=50, credits=1, rows_remaining=100
    )
    assert "{" not in text


# --- precedence + contract pins ---------------------------------------------


def test_final_allowance_wins_over_an_available_credit(tmp_path):
    """Budget exhausted with a credit in hand: the wrap-up run takes the
    allowance path (ack code required, no debit) — the credit is left
    untouched."""
    binding, state, ctx = _make_binding(
        tmp_path, credits=1, used=100, max_eval_rows=100
    )
    fake, ctxm = _stub_subprocess()
    with ctxm:
        resp = binding("comparison", acknowledge_message=ctx.comparison_ack_code)
    assert resp["ok"] is True
    assert state.final_comparison_evals == 1
    assert state.comparison_credits_balance == 1
    # The balance is still reported — just not spent.
    assert resp["comparison_credits_remaining"] == 1


def test_record_field_is_whitelisted_end_to_end():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord
    from src.engine.optimizer.v3.run_reporting import CLOSE_RECORD_HISTORY_KEYS
    from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS

    assert "comparison_credits_after" in ITERATION_RECORD_FIELDS
    assert "comparison_credits_after" in CLOSE_RECORD_HISTORY_KEYS
    assert "comparison_credits_after" in IterationHistoryRecord.model_fields


def test_run_eval_tool_schema_declares_the_optional_ack_parameter():
    from src.engine.agent.harnesses.architect_tools import architect_tool_spec

    spec = architect_tool_spec("run_eval")
    props = spec.input_schema["properties"]
    assert "acknowledge_message" in props and props["acknowledge_message"]["type"] == "string"
    assert "acknowledge_message" not in spec.input_schema["required"]
    assert "credit" in spec.description and "acknowledge_message" in spec.description
