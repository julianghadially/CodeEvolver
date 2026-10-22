"""Copyright © 2026 440 Labs LLC

The summary-only `run_eval` response: per-row feedback lives on disk
(`index.jsonl` / `index_pretty_print.txt`), the inline dict carries counts and
file pointers instead. A 100+ KB inline blob would be truncated by both
harnesses and force score recovery via grep on every eval.

Fixture pattern mirrors `test_comparison_credits.py`: real `CEState`, real
`RunEvalBinding`, subprocess eval stubbed away.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any
from unittest.mock import patch

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.eval_log import evals_log_path
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding

_TRAINSET = [{"id": i, "category": "a" if i % 2 else "b"} for i in range(20)]


def _meta_row(i: int, *, failed: bool = False, error: str | None = None, score: float = 1.0):
    return {
        "example_id": i,
        "score": 0.0 if failed else score,
        "failed": failed,
        "feedback": "row feedback " + "x" * 500,
        "error_summary": error,
        "row_file": f"rows/{i}/row.json",
    }


class _FakeRunEval:
    def __init__(self, meta_fn) -> None:
        self.meta_fn = meta_fn
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> EvalResult:
        self.calls.append(kwargs)
        row_ids = kwargs.get("row_ids") or []
        # meta_fn keys on POSITION in the sample, not the row id —
        # the subsample draw is rng-dependent.
        meta = [self.meta_fn(pos) for pos in range(len(row_ids))]
        return EvalResult(
            scores_by_metric={"accuracy": 0.5},
            primary_score=0.5,
            trace_meta=meta,
            trace_dir="/traces/iteration_1_parent",
            trace_index_file="/traces/iteration_1_parent/index.jsonl",
            row_ids=list(row_ids),
            raw={},
        )


def _stub(meta_fn=lambda i: _meta_row(i)):
    fake = _FakeRunEval(meta_fn)
    return fake, patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=fake,
        git_get_sha=lambda _ws: "fake_sha",
        git_tree_signature=lambda _ws: "fake_tree",
        git_get_current_branch=lambda _ws: "codeevolver-test-child",
        git_dirty_paths=lambda _ws: [],
    )


def _make_binding(tmp_path: Path, *, ack_code: str = "") -> tuple[RunEvalBinding, CEState, IterationContext]:
    (tmp_path / "train.json").write_text(json.dumps(_TRAINSET))
    ctx = IterationContext(
        iteration_idx=1, parent_locked=True, max_eval_rows=1000,
        evals_used=0, subsample_size=5, comparison_ack_code=ack_code,
    )
    state = CEState(
        seed_branch="seed", seed_sha="sha",
        comparison_history_path=str(tmp_path / "comparison_set_history.jsonl"),
    )
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
    )
    return binding, state, ctx


def test_train_response_is_summary_only(tmp_path):
    binding, _, _ = _make_binding(tmp_path)
    _, stub = _stub()
    with stub:
        r = binding(eval_sample="train")
    for dropped in ("trace_meta", "trace_meta_truncated", "row_ids"):
        assert dropped not in r
    assert r["ok"] is True
    assert r["primary_score"] == 0.5
    assert r["n_rows"] == 5
    assert r["zero_or_failed_count"] == 0
    assert r["infra_failed"] == 0
    assert r["score_reliable"] is True
    assert r["trace_pretty_file"] == "/traces/iteration_1_parent/index_pretty_print.txt"
    assert r["evals_log_file"] == evals_log_path(1)


def test_row_ids_kept_only_when_a_filter_is_in_effect(tmp_path):
    binding, _, ctx = _make_binding(tmp_path)
    _, stub = _stub()
    with stub:
        r = binding(eval_sample="train", filter={"category": "a"})
    assert ctx.subsample_filter is not None
    assert "row_ids" in r and len(r["row_ids"]) == 5


def test_infra_failures_are_split_from_real_failures(tmp_path):
    """A provider-fault zero (matches the retry classifier) is
    `infra_failed` and flips `score_reliable`; a plain metric failure is a
    real program failure and leaves the score reliable."""
    infra_error = "litellm.RateLimitError: connection reset by peer"

    def meta(i):
        if i == 0:
            return _meta_row(i, failed=True, error=infra_error)
        if i == 1:
            return _meta_row(i, failed=True, error="AssertionError: wrong answer")
        return _meta_row(i)

    binding, _, _ = _make_binding(tmp_path)
    _, stub = _stub(meta)
    with stub:
        r = binding(eval_sample="train")
    assert r["zero_or_failed_count"] == 2
    assert r["infra_failed"] == 1
    assert r["score_reliable"] is False


def test_comparison_response_carries_history_path(tmp_path):
    binding, state, _ = _make_binding(tmp_path)
    _, stub = _stub()
    with stub:
        r = binding(eval_sample="comparison")
    assert r["comparison_history_path"] == state.comparison_history_path
    assert r["n_rows"] == 3


def test_ack_note_states_overshoot_when_rows_run_short(tmp_path):
    """Under budget but with fewer rows left than a comparison costs, the
    refusal note must say the eval may overshoot, so the agent does not
    abandon its validation over the ambiguity."""
    binding, state, ctx = _make_binding(tmp_path, ack_code="37")
    state.grant_comparison_credits(1)
    ctx.max_eval_rows = 100
    ctx.evals_used = 98
    state.increment_evals(98)  # 2 rows left < comparison_set_size (3)
    _, stub = _stub()
    with stub:
        refusal = binding(eval_sample="comparison")
    assert refusal["ok"] is False
    assert refusal["refused"] == "acknowledgement_required"
    assert "allowed to overshoot" in refusal["note"]
