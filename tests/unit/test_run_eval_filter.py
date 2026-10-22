"""Copyright © 2026 440 Labs LLC

Unit tests for the `filter` kwarg on the `run_eval` tool binding.

The binding applies the filter *in-process* before sampling, then writes the
concrete row_ids to the subprocess command. These tests exercise the
filter/sampling path by stubbing the subprocess call with a fake that records
the row_ids it receives. Trainset rows are loaded lazily from the path in
`EvaluatorConfig.trainset_path` — tests write a real file under `tmp_path`.
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
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_eval_tool import (
    RunEvalBinding,
    RunEvalError,
    _normalize_filter,
    _row_matches,
)


# --- Fixtures ---------------------------------------------------------------


def _make_evaluator_config(trainset_path: str = "train.json") -> EvaluatorConfig:
    return EvaluatorConfig(
        program_factory="my_pkg.program:make_program",
        metric_fns=["my_pkg.metrics:accuracy"],
        metric_names=["accuracy"],
        primary_metric="accuracy",
        trainset_path=trainset_path,
    )


def _make_eval_result(primary: float, row_ids: list[int]) -> EvalResult:
    return EvalResult(
        scores_by_metric={"accuracy": primary},
        primary_score=primary,
        trace_meta=[],
        trace_dir="/traces/iteration_0_parent",
        trace_index_file="/traces/iteration_0_parent/index.jsonl",
        row_ids=list(row_ids),
        raw={},
    )


class _FakeRunEval:
    """Records kwargs each time the binding calls into _run_eval; returns
    scripted EvalResults (default 0.5 so the ceiling-resample branch isn't
    exercised unless a test wants it)."""

    def __init__(self, primary: float = 0.5):
        self.primary = primary
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> EvalResult:
        self.calls.append(kwargs)
        return _make_eval_result(self.primary, kwargs.get("row_ids") or [])


def _locked_ctx() -> IterationContext:
    ctx = IterationContext(iteration_idx=0, parent_locked=True)
    ctx.parent_sha = "parent"
    ctx.child_branch = "codeevolver-test-child"
    return ctx


def _write_trainset(tmp_path: Path, rows: list[dict[str, Any]]) -> str:
    p = tmp_path / "train.json"
    p.write_text(json.dumps(rows))
    return "train.json"


def _make_binding(
    ctx: IterationContext,
    workspace: str,
    trainset_rel_path: str,
    trainset_len: int,
    *,
    seed: int = 0,
    subsample_size: int = 10,
) -> RunEvalBinding:
    return RunEvalBinding(
        ctx=ctx,
        workspace=workspace,
        config=_make_evaluator_config(trainset_rel_path),
        trainset_ids=list(range(trainset_len)),
        comparison_ids=[0, 1, 2],
        rng=random.Random(seed),
        subsample_size=subsample_size,
    )


_TRAINSET = [
    {"id": 0, "category": "metals", "desc": "steel rod"},
    {"id": 1, "category": "metals", "desc": "copper pipe"},
    {"id": 2, "category": "textiles", "desc": "cotton fabric"},
    {"id": 3, "category": "textiles", "desc": "wool yarn"},
    {"id": 4, "category": "metals", "desc": "aluminum foil"},
    {"id": 5, "category": "electronics", "desc": "circuit board"},
]


def _stub_subprocess():
    """Patch the real subprocess-dispatching _run_eval and git_get_sha on the
    binding module so tests never hit the filesystem or git."""
    fake = _FakeRunEval(primary=0.5)
    return fake, patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=fake,
        git_get_sha=lambda _ws: "fake_sha",
    )


# --- filter normalization ---------------------------------------------------


def test_normalize_filter_none_and_empty():
    assert _normalize_filter(None) is None
    assert _normalize_filter({}) is None


def test_normalize_filter_scalar_becomes_eq():
    assert _normalize_filter({"category": "metals"}) == {
        "category": {"eq": "metals"}
    }


def test_normalize_filter_preserves_op_map():
    out = _normalize_filter({"desc": {"contains": "steel"}})
    assert out == {"desc": {"contains": "steel"}}


def test_normalize_filter_in_canonicalizes_order():
    a = _normalize_filter({"cat": {"in": ["b", "a"]}})
    b = _normalize_filter({"cat": {"in": ["a", "b"]}})
    assert a == b


def test_normalize_filter_rejects_unknown_op():
    with pytest.raises(RunEvalError, match="unknown operators"):
        _normalize_filter({"x": {"gt": 5}})


def test_normalize_filter_rejects_bad_shapes():
    with pytest.raises(RunEvalError):
        _normalize_filter("not-a-dict")  # type: ignore[arg-type]
    with pytest.raises(RunEvalError, match="empty op map"):
        _normalize_filter({"x": {}})
    with pytest.raises(RunEvalError, match=r"must be a list"):
        _normalize_filter({"x": {"in": "metals"}})
    with pytest.raises(RunEvalError, match=r"must be a string"):
        _normalize_filter({"x": {"contains": 5}})


# --- row matching -----------------------------------------------------------


def test_row_matches_eq():
    flt = _normalize_filter({"category": "metals"})
    assert _row_matches(_TRAINSET[0], flt)
    assert not _row_matches(_TRAINSET[2], flt)


def test_row_matches_neq_and_in_and_contains():
    neq = _normalize_filter({"category": {"neq": "textiles"}})
    assert _row_matches(_TRAINSET[0], neq)
    assert not _row_matches(_TRAINSET[2], neq)

    inf = _normalize_filter({"category": {"in": ["metals", "electronics"]}})
    assert _row_matches(_TRAINSET[5], inf)
    assert not _row_matches(_TRAINSET[2], inf)

    contains = _normalize_filter({"desc": {"contains": "Steel"}})  # case-insensitive
    assert _row_matches(_TRAINSET[0], contains)
    assert not _row_matches(_TRAINSET[2], contains)


def test_row_matches_multi_column_ands():
    flt = _normalize_filter(
        {"category": "metals", "desc": {"contains": "steel"}}
    )
    assert _row_matches(_TRAINSET[0], flt)
    # Right category, wrong desc:
    assert not _row_matches(_TRAINSET[1], flt)


# --- binding: filter narrows the pool ---------------------------------------


def test_filter_narrows_subsample_pool(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        result = binding(
            eval_sample="train",
            filter={"category": "metals"},
        )

    # All drawn ids must come from the metals subset {0, 1, 4}.
    assert set(result["row_ids"]).issubset({0, 1, 4})
    assert set(ctx.subsample_ids).issubset({0, 1, 4})
    assert ctx.subsample_filter == {"category": {"eq": "metals"}}


def test_second_call_with_same_filter_reuses_locked_ids(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        first = binding(
            eval_sample="train",
            filter={"category": "metals"},
        )
        second = binding(
            eval_sample="train",
            filter={"category": "metals"},
        )

    assert first["row_ids"] == second["row_ids"]


def test_second_call_omitting_filter_reuses_locked_filter(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        binding(
            eval_sample="train",
            filter={"category": "metals"},
        )
        # No filter on second call — should reuse locked ids, not error.
        second = binding(eval_sample="train")

    assert set(second["row_ids"]).issubset({0, 1, 4})


def test_second_call_with_different_filter_errors(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        binding(
            eval_sample="train",
            filter={"category": "metals"},
        )
        with pytest.raises(RunEvalError, match="filter is locked"):
            binding(
                eval_sample="train",
                filter={"category": "textiles"},
            )


def test_empty_filter_match_raises(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        with pytest.raises(RunEvalError, match="matches 0 of"):
            binding(
                eval_sample="train",
                filter={"category": "nonexistent"},
            )


def test_filter_without_trainset_path_errors_clearly(tmp_path):
    ctx = _locked_ctx()
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=str(tmp_path),
        config=EvaluatorConfig(
            program_factory="my_pkg.program:make_program",
            metric_fns=["my_pkg.metrics:accuracy"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path=None,  # no path configured
        ),
        trainset_ids=[0, 1, 2],
        comparison_ids=[0],
        rng=random.Random(0),
        subsample_size=3,
    )
    _, stub = _stub_subprocess()

    with stub:
        with pytest.raises(RunEvalError, match="trainset_path"):
            binding(
                eval_sample="train",
                filter={"category": "metals"},
            )


def test_filter_with_missing_trainset_file_errors(tmp_path):
    ctx = _locked_ctx()
    binding = _make_binding(
        ctx, str(tmp_path), "nope.json", 3, seed=0, subsample_size=3,
    )
    _, stub = _stub_subprocess()

    with stub:
        with pytest.raises(RunEvalError, match="trainset not found"):
            binding(
                eval_sample="train",
                filter={"category": "metals"},
            )


def test_no_filter_skips_file_load(tmp_path):
    """Regression: omitting `filter` should not require reading trainset_path.

    Set a bogus trainset_path and confirm run_eval without a filter still works.
    """
    ctx = _locked_ctx()
    binding = _make_binding(
        ctx, str(tmp_path), "does-not-exist.json", 6, seed=42, subsample_size=3,
    )
    _, stub = _stub_subprocess()

    with stub:
        result = binding(eval_sample="train")

    # Unfiltered evals do not echo row_ids inline; the count is
    # reported as n_rows.
    assert "row_ids" not in result
    assert result["n_rows"] == 3
    assert ctx.subsample_filter is None


def test_filter_load_is_cached_across_calls(tmp_path):
    """The lazy loader should read the file exactly once per binding."""
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = _make_binding(ctx, str(tmp_path), rel, len(_TRAINSET), seed=42)
    _, stub = _stub_subprocess()

    with stub:
        binding(
            eval_sample="train",
            filter={"category": "metals"},
        )

        # Corrupt the file after first load — second call must use the cache.
        (tmp_path / rel).write_text("!!not json!!")
        binding(eval_sample="train")

    assert binding._trainset_rows_cache is not None
    assert len(binding._trainset_rows_cache) == len(_TRAINSET)


# --- binding: comparison set --------------------------------------------------


def test_comparison_eval_uses_fixed_ids_and_records_out_of_history(tmp_path):
    """eval_sample="comparison" runs the run's FIXED comparison ids with the
    "comparison" suffix, reports through state.record_comparison_eval, and
    stays out of ctx.eval_history (only ctx.eval_records)."""

    class _FakeState:
        def __init__(self):
            self.comparison_calls: list[dict[str, Any]] = []
            self.total_num_evals = 0

        def increment_evals(self, n: int) -> None:
            self.total_num_evals += n

        def record_comparison_eval(self, **kwargs: Any) -> None:
            self.comparison_calls.append(kwargs)

    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    state = _FakeState()
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=str(tmp_path),
        config=_make_evaluator_config(rel),
        trainset_ids=list(range(len(_TRAINSET))),
        comparison_ids=[0, 2, 5],
        rng=random.Random(42),
        subsample_size=3,
        state=state,
    )
    fake, stub = _stub_subprocess()

    with stub:
        result = binding(eval_sample="comparison")

    assert "row_ids" not in result  # summary-only response
    assert result["n_rows"] == 3
    assert result["suffix"] == "comparison"
    assert fake.calls[-1]["row_ids"] == [0, 2, 5]
    assert fake.calls[-1]["suffix"] == "comparison"
    # Recorded via state.record_comparison_eval with the evaluated sha.
    assert len(state.comparison_calls) == 1
    rec = state.comparison_calls[0]
    assert rec["iteration"] == 0
    assert rec["sha"] == "fake_sha"
    assert rec["score"] == pytest.approx(0.5)
    assert rec["n_rows"] == 3
    # Comparison evals count against the budget...
    assert state.total_num_evals == 3
    # ...but never enter eval_history (finalize() must not see them);
    # they DO land in the full eval_records log.
    assert ctx.eval_history == []
    assert len(ctx.eval_records) == 1
    assert ctx.eval_records[0]["eval_sample"] == "comparison"


def test_comparison_eval_errors_when_no_comparison_set(tmp_path):
    rel = _write_trainset(tmp_path, _TRAINSET)
    ctx = _locked_ctx()
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=str(tmp_path),
        config=_make_evaluator_config(rel),
        trainset_ids=list(range(len(_TRAINSET))),
        comparison_ids=[],
        rng=random.Random(42),
        subsample_size=3,
    )
    _, stub = _stub_subprocess()

    with stub:
        with pytest.raises(RunEvalError, match="no comparison set"):
            binding(eval_sample="comparison")
