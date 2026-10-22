"""Copyright © 2026 440 Labs LLC

`run_variance_estimate` (specs/archive-public/sampling_variance.md): the pure statistics,
the tool driver over a stubbed evaluator, the once-per-run contract, the
run-state / record / config plumbing, the tool surface on both harnesses,
and the system-prompt rendering before and after the estimate exists.

Fixture pattern mirrors `test_run_eval_budget.py` (stub `_run_eval` + the
git reads on `run_eval_tool`, real `CEState` + `RunEvalBinding`).
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.engine.agent.harnesses import FakeCodingHarness
from src.engine.agent.harnesses.architect_tools import (
    ARCHITECT_TOOL_SPECS,
    mcp_architect_tool_definitions,
    ordered_architect_tool_names,
)
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3 import variance_estimate as ve_module
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.closing import build_close_record
from src.engine.optimizer.v3.eval_budget import enforce_eval_budget
from src.engine.optimizer.v3.eval_log import VARIANCE_EVAL_SAMPLE, variance_suffix
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.prompts import (
    build_architect_system_prompt,
    build_evo_architect_system_prompt,
    build_freeform_architect_system_prompt,
)
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding, RunEvalError
from src.engine.optimizer.v3.variance_estimate import (
    ALREADY_RUN_NOTE,
    IDENTICAL_OUTPUTS_NOTE,
    estimate_variance,
    run_variance_estimate_tool,
    variance_note,
)

from tests.controls._arch_helpers import FakeState, make_architect


# --- pure statistics ---------------------------------------------------------


def test_estimate_variance_matches_hand_computation():
    # 3 rows x 5 runs. Row a: constant; row b: [1,0,1,0,1]; row c: [0.5,0.5,0.5,0.5,1].
    runs = [
        {"a": 1.0, "b": 1.0, "c": 0.5},
        {"a": 1.0, "b": 0.0, "c": 0.5},
        {"a": 1.0, "b": 1.0, "c": 0.5},
        {"a": 1.0, "b": 0.0, "c": 0.5},
        {"a": 1.0, "b": 1.0, "c": 1.0},
    ]
    aggregates = [sum(r.values()) / 3 for r in runs]
    stats = estimate_variance(runs, aggregates)
    # s_b² = Σ(y-0.6)²/4 = (3*0.16 + 2*0.36)/4 = 0.3 ; s_c² = (4*0.01 + 0.16)/4 = 0.05
    expected_wiv = (0.0 + 0.3 + 0.05) / 3
    assert math.isclose(stats.within_item_variance, expected_wiv)
    assert math.isclose(stats.run_to_run_std, math.sqrt(expected_wiv / 3))
    assert stats.n_rows == 3 and stats.n_runs == 5
    # Empirical std of the aggregates is ddof=1.
    mean = sum(aggregates) / 5
    emp = math.sqrt(sum((a - mean) ** 2 for a in aggregates) / 4)
    assert math.isclose(stats.aggregate_std_empirical, emp)


def test_identical_runs_have_zero_variance():
    runs = [{"a": 0.5, "b": 1.0}] * 4
    stats = estimate_variance(runs, [0.75] * 4)
    assert stats.within_item_variance == 0.0
    assert stats.run_to_run_std == 0.0
    assert stats.aggregate_std_empirical == 0.0


def test_rows_missing_from_any_run_are_excluded():
    runs = [{"a": 1.0, "b": 0.0}, {"a": 0.0}, {"a": 1.0, "b": 1.0}]
    stats = estimate_variance(runs, [0.5, 0.0, 1.0])
    assert stats.n_rows == 1  # only "a" was scored in every run
    assert math.isclose(stats.within_item_variance, 1 / 3)


def test_estimate_variance_rejects_too_few_runs_or_no_common_rows():
    with pytest.raises(ValueError, match="at least 2 runs"):
        estimate_variance([{"a": 1.0}], [1.0])
    with pytest.raises(ValueError, match="no row"):
        estimate_variance([{"a": 1.0}, {"b": 1.0}], [1.0, 1.0])
    with pytest.raises(ValueError, match="one entry per run"):
        estimate_variance([{"a": 1.0}, {"a": 1.0}], [1.0])


def test_variance_suffixes_are_distinct_and_not_holdout():
    from src.engine.telemetry.trace_layout import is_holdout_suffix

    names = [variance_suffix(k) for k in range(5)]
    assert names == ["variance", "variance_2", "variance_3", "variance_4", "variance_5"]
    assert len(set(names)) == 5
    assert not any(is_holdout_suffix(n) for n in names)


# --- prompt text -------------------------------------------------------------


def _estimate(**overrides: Any) -> dict[str, Any]:
    base = {
        "iteration": 2,
        "sha": "abcdef1234567890",
        "n_runs": 5,
        "subsample_size": 50,
        "n_rows": 50,
        "within_item_variance": 0.0123,
        "run_to_run_std": 0.0157,
        "aggregate_scores": [0.8, 0.82, 0.79, 0.81, 0.8],
        "identical_outputs_warning": False,
    }
    base.update(overrides)
    return base


def test_variance_note_before_and_after_the_estimate():
    before = variance_note(None, 50, 5)
    assert "run_variance_estimate" in before
    assert "5 times with a fixed sample of 50" in before
    assert "iterations 1-3" in before
    assert "{" not in before

    after = variance_note(_estimate(), 50, 5)
    assert "run_variance_estimate" not in after
    assert "iteration 2 and sha abcdef12 was run 5 times" in after
    assert "within-item variance was 0.0123" in after
    assert "50-item evaluation metric was 0.0157" in after
    assert "[0.8000, 0.8200, 0.7900, 0.8100, 0.8000]" in after
    assert IDENTICAL_OUTPUTS_NOTE not in after
    assert "{" not in after

    warned = variance_note(_estimate(identical_outputs_warning=True), 50, 5)
    assert IDENTICAL_OUTPUTS_NOTE in warned


@pytest.mark.parametrize(
    "builder",
    [
        build_architect_system_prompt,
        build_evo_architect_system_prompt,
        build_freeform_architect_system_prompt,
    ],
)
def test_every_arm_prompt_renders_both_variants(builder):
    def _build(**kw: Any) -> str:
        ctx = IterationContext(
            iteration_idx=3, subsample_size=20, max_eval_rows=1000, evals_used=100,
            child_branch="codeevolver-x-iter3", **kw,
        )
        return builder(
            additional_instructions=None,
            program_path="my_pkg.program:make_program",
            workspace_path="/ws/repo",
            context=ctx,
        )

    unset = _build()
    assert "run_variance_estimate" in unset
    assert "5 times with a fixed sample of 20" in unset
    assert not re.search(r"\{[a-z_]+\}", unset)

    taken = _build(variance_estimate=_estimate(), variance_estimate_samples=3)
    assert "run_variance_estimate" not in taken
    assert "0.0157" in taken and "[0.8000, 0.8200, 0.7900, 0.8100, 0.8000]" in taken
    assert not re.search(r"\{[a-z_]+\}", taken)


# --- tool driver over a stubbed evaluator -----------------------------------


_TRAINSET = [{"id": i} for i in range(40)]


class _FakeRunEval:
    """Deterministic per-run scores: row r scores 1.0 on odd runs when r is
    odd, so rows differ across runs and the estimate is non-zero."""

    def __init__(self, fail_from_call: int | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.fail_from_call = fail_from_call

    def __call__(self, **kwargs: Any) -> EvalResult:
        self.calls.append(kwargs)
        k = len(self.calls)
        if self.fail_from_call is not None and k >= self.fail_from_call:
            raise RuntimeError("evaluator exploded")
        row_ids = list(kwargs.get("row_ids") or [])
        meta = [
            {"example_id": r, "score": 1.0 if (r + k) % 2 == 0 else 0.0}
            for r in row_ids
        ]
        score = sum(m["score"] for m in meta) / max(1, len(meta))
        suffix = kwargs["suffix"]
        return EvalResult(
            scores_by_metric={"accuracy": score},
            primary_score=score,
            trace_meta=meta,
            trace_dir=f"/traces/iteration_1_{suffix}",
            trace_index_file=f"/traces/iteration_1_{suffix}/index.jsonl",
            row_ids=row_ids,
            raw={},
        )


def _stubs(fake: _FakeRunEval, *, dirty: bool = False, branch: str | None = "codeevolver-t"):
    commits: list[str] = []

    def _commit(_ws: str, msg: str, **_kw: Any) -> bool:
        commits.append(msg)
        return True

    p1 = patch.multiple(
        "src.engine.optimizer.v3.run_eval_tool",
        _run_eval=fake,
        git_get_sha=lambda _ws, *a: "head_sha",
        git_tree_signature=lambda _ws, *a: "tree_sig",
        git_get_current_branch=lambda _ws: branch,
        git_dirty_paths=lambda _ws: ["x.py"] if dirty else [],
        git_is_dirty=lambda _ws: dirty,
        git_add_and_commit=_commit,
    )
    p2 = patch.multiple(
        ve_module,
        git_get_sha=lambda _ws, *a: "head_sha",
        git_tree_signature=lambda _ws, *a: "tree_sig",
        git_get_current_branch=lambda _ws: branch,
    )
    return commits, p1, p2


def _binding(
    tmp_path: Path,
    *,
    used: int = 0,
    max_eval_rows: int | None = 1000,
    subsample_size: int = 10,
    comparison_closes_iteration: bool = False,
    state: CEState | None = None,
) -> tuple[RunEvalBinding, IterationContext, CEState]:
    (tmp_path / "train.json").write_text(json.dumps(_TRAINSET))
    ctx = IterationContext(
        iteration_idx=1,
        parent_locked=True,
        max_eval_rows=max_eval_rows,
        evals_used=used,
        subsample_size=subsample_size,
    )
    st = state or CEState(
        seed_branch="seed",
        seed_sha="sha",
        comparison_history_path=str(tmp_path / "comparison_set_history.jsonl"),
    )
    st.increment_evals(used - st.total_num_evals)
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
        subsample_size=subsample_size,
        state=st,
        comparison_closes_iteration=comparison_closes_iteration,
    )
    return binding, ctx, st


def _run(binding: RunEvalBinding, ctx: IterationContext, state: CEState, n_runs: int = 5) -> dict:
    return run_variance_estimate_tool(
        ctx=ctx, state=state, binding=binding, workspace=binding.workspace, n_runs=n_runs,
    )


def test_tool_runs_n_times_on_one_fresh_sample_and_records_everything(tmp_path):
    binding, ctx, state = _binding(tmp_path, used=7)
    # A locked baseline that the variance runs must leave alone.
    ctx.subsample_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    ctx.initial_subsample_score = 0.3
    ctx.baseline_sample_key = "baseline-key"
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2:
        resp = _run(binding, ctx, state)

    assert resp["ok"] is True
    assert [c["suffix"] for c in fake.calls] == [
        "variance", "variance_2", "variance_3", "variance_4", "variance_5",
    ]
    rows = fake.calls[0]["row_ids"]
    assert all(c["row_ids"] == rows for c in fake.calls)
    assert len(rows) == 10 and rows == sorted(rows)
    # Fresh draw — not the iteration's locked subsample; the lock is untouched.
    assert rows != ctx.subsample_ids
    assert ctx.subsample_ids == list(range(10))
    assert ctx.initial_subsample_score == 0.3
    assert ctx.baseline_sample_key == "baseline-key"
    # Counted, recorded as variance evals, never train evals.
    assert state.total_num_evals == 7 + 5 * 10
    assert [r["eval_sample"] for r in ctx.eval_records] == [VARIANCE_EVAL_SAMPLE] * 5
    assert all(r.get("row_scores") for r in ctx.eval_records)
    assert ctx.eval_history == []
    # The estimate itself.
    assert resp["n_runs"] == 5 and resp["n_rows"] == 10
    assert resp["sha"] == "head_sha" and resp["tree"] == "tree_sig"
    assert resp["branch"] == "codeevolver-t" and resp["iteration"] == 1
    assert len(resp["aggregate_scores"]) == 5
    assert resp["within_item_variance"] > 0
    assert math.isclose(
        resp["run_to_run_std"], math.sqrt(resp["within_item_variance"] / 10)
    )
    assert resp["truncated"] is False and resp["identical_outputs_warning"] is False
    assert len(resp["eval_ids"]) == 5 and all(resp["eval_ids"])
    assert resp["eval_rows_remaining"] == 1000 - 57
    assert "auto_committed" not in resp
    # Run-level + iteration-level state.
    assert state.variance_estimate == {k: v for k, v in resp.items() if k in state.variance_estimate}
    assert ctx.variance_estimate is state.variance_estimate or ctx.variance_estimate == state.variance_estimate
    assert ctx.variance_estimate_ran is True


def test_second_call_is_refused_without_spending(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2:
        first = _run(binding, ctx, state)
        used_after_first = state.total_num_evals
        second = _run(binding, ctx, state)
    assert second["ok"] is False and second["refused"] == "already_run"
    assert second["note"] == ALREADY_RUN_NOTE
    assert second["variance_estimate"]["run_to_run_std"] == first["run_to_run_std"]
    assert state.total_num_evals == used_after_first
    assert len(fake.calls) == 5


def test_state_round_trips_and_refuses_a_second_record(tmp_path):
    state = CEState(seed_branch="s", seed_sha="x")
    state.record_variance_estimate({"run_to_run_std": 0.02, "n_runs": 5})
    path = str(tmp_path / "ce_state.json")
    state.save(path)
    loaded = CEState.load(path)
    assert loaded.variance_estimate == {"run_to_run_std": 0.02, "n_runs": 5}
    with pytest.raises(ValueError, match="already recorded"):
        loaded.record_variance_estimate({"run_to_run_std": 0.5})
    # A fresh state has none.
    assert CEState(seed_branch="s", seed_sha="x").variance_estimate is None


def test_at_budget_is_refused_before_any_spend(tmp_path):
    binding, ctx, state = _binding(tmp_path, used=1000, max_eval_rows=1000)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2, pytest.raises(RunEvalError, match="run_variance_estimate"):
        _run(binding, ctx, state)
    assert fake.calls == []
    assert state.variance_estimate is None and ctx.variance_estimate_ran is False


def test_budget_exhaustion_midway_keeps_the_completed_runs(tmp_path):
    # 25 rows left: run 1 (10) -> 10 used, run 2 -> 20, run 3 -> 30 >= 25 stop.
    binding, ctx, state = _binding(tmp_path, used=975, max_eval_rows=1000)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2:
        resp = _run(binding, ctx, state)
    assert len(fake.calls) == 3
    assert resp["ok"] is True and resp["truncated"] is True and resp["n_runs"] == 3
    assert "3 of 5 runs" in resp["note"]
    assert state.total_num_evals == 975 + 30


def test_one_completed_run_is_not_an_estimate(tmp_path):
    # 5 rows left: run 1 spends 10 -> at budget -> refused; only one reading.
    binding, ctx, state = _binding(tmp_path, used=995, max_eval_rows=1000)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2, pytest.raises(RunEvalError, match="at least two"):
        _run(binding, ctx, state)
    assert len(fake.calls) == 1
    assert state.variance_estimate is None


def test_evaluator_failure_after_the_first_run_truncates(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    fake = _FakeRunEval(fail_from_call=4)
    _, p1, p2 = _stubs(fake)
    with p1, p2:
        resp = _run(binding, ctx, state)
    assert resp["n_runs"] == 3 and resp["truncated"] is True


def test_dirty_tree_is_auto_committed_and_reported(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    fake = _FakeRunEval()
    commits, p1, p2 = _stubs(fake, dirty=True)
    with p1, p2:
        resp = _run(binding, ctx, state)
    assert commits == ["iter 1: auto-commit before variance eval"]
    assert resp["auto_committed"] is True and "auto-committed" in resp["auto_commit_note"]


def test_detached_head_is_not_auto_committed(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    fake = _FakeRunEval()
    commits, p1, p2 = _stubs(fake, dirty=True, branch=None)
    with p1, p2:
        resp = _run(binding, ctx, state)
    assert commits == [] and "auto_committed" not in resp and resp["branch"] is None


def test_identical_outputs_are_flagged(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2, patch("src.engine.evaluator.holdout.samples_are_identical", lambda dirs: True):
        resp = _run(binding, ctx, state)
    assert resp["identical_outputs_warning"] is True
    assert IDENTICAL_OUTPUTS_NOTE in resp["note"]


def test_refused_after_the_comparison_on_a_comparison_closes_arm(tmp_path):
    binding, ctx, state = _binding(tmp_path, comparison_closes_iteration=True)
    ctx.record_eval("sha", 0.5, eval_sample="comparison", n_rows=3)
    fake = _FakeRunEval()
    _, p1, p2 = _stubs(fake)
    with p1, p2, pytest.raises(RunEvalError, match="close step"):
        _run(binding, ctx, state)
    assert fake.calls == []


def test_unlocked_context_and_bad_run_count_are_refused(tmp_path):
    binding, ctx, state = _binding(tmp_path)
    ctx.parent_locked = False
    with pytest.raises(RunEvalError, match="not been initialized"):
        _run(binding, ctx, state)
    ctx.parent_locked = True
    with pytest.raises(RunEvalError, match=">= 2"):
        _run(binding, ctx, state, n_runs=1)


def test_budget_policy_names_the_tool_when_refusing_a_variance_run():
    with pytest.raises(RunEvalError, match="run_variance_estimate"):
        enforce_eval_budget(
            "variance", budget=(100, 100), has_comparison_set=True, final_left=1
        )
    assert enforce_eval_budget(
        "variance", budget=(99, 100), has_comparison_set=True, final_left=1
    ) is False


# --- tool surface + registry --------------------------------------------------


def test_tool_spec_exists_and_is_ordered_on_both_harnesses():
    spec = ARCHITECT_TOOL_SPECS["run_variance_estimate"]
    assert spec.input_schema["required"] == [] and spec.input_schema["properties"] == {}
    assert "ONCE per run" in spec.description
    assert "run_variance_estimate" in ordered_architect_tool_names()
    ee = {"run_eval", "coding_agent", "inspect_traces", "close_iteration"}
    exposed = {d["name"] for d in mcp_architect_tool_definitions(registered_names=ee)}
    assert "run_variance_estimate" not in exposed
    exposed = {
        d["name"]
        for d in mcp_architect_tool_definitions(registered_names=ee | {"run_variance_estimate"})
    }
    assert "run_variance_estimate" in exposed


def test_registry_exposes_the_tool_until_the_run_has_an_estimate():
    arch, _ctx, _state = make_architect()
    registry = arch.build_tool_registry(agent_harness=FakeCodingHarness())
    assert "run_variance_estimate" in registry
    assert callable(registry["run_variance_estimate"])

    taken = FakeState()
    taken.variance_estimate = {"run_to_run_std": 0.01}
    arch, _ctx, _state = make_architect(state=taken)
    registry = arch.build_tool_registry(agent_harness=FakeCodingHarness())
    assert "run_variance_estimate" not in registry
    # The rest of the surface is unchanged.
    assert {"run_eval", "coding_agent", "inspect_traces", "close_iteration"} <= set(registry)


def test_simple_loop_never_exposes_the_tool():
    from src.engine.optimizer.simple_loop import session as sl_session

    text = Path(sl_session.__file__).read_text()
    assert "run_variance_estimate" not in text
    assert '"run_eval"' in text


# --- records / config plumbing ------------------------------------------------


def test_record_field_is_whitelisted_everywhere():
    from src.db.db_schemas.db_schemas import IterationHistoryRecord
    from src.engine.optimizer.v3.run_reporting import CLOSE_RECORD_HISTORY_KEYS
    from src.engine.schemas.iteration_history import ITERATION_RECORD_FIELDS

    assert "variance_estimate" in ITERATION_RECORD_FIELDS
    assert "variance_estimate" in IterationHistoryRecord.model_fields
    assert "variance_estimate" in CLOSE_RECORD_HISTORY_KEYS


def test_close_record_carries_the_estimate_only_on_the_iteration_that_ran_it():
    state = CEState(seed_branch="codeevolver-x-main", seed_sha="s")
    ran = IterationContext(
        iteration_idx=2, parent_locked=True, child_branch="codeevolver-x-main",
        variance_estimate=_estimate(), variance_estimate_ran=True,
    )
    assert build_close_record(ran, state)["variance_estimate"]["run_to_run_std"] == 0.0157
    later = IterationContext(
        iteration_idx=3, parent_locked=True, child_branch="codeevolver-x-main",
        variance_estimate=_estimate(), variance_estimate_ran=False,
    )
    assert build_close_record(later, state)["variance_estimate"] is None


def test_config_knob_round_trips_and_validates():
    from src.api.schemas.requests import OptimizeRequest
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig
    from src.engine.schemas.agent_config import BaseAgentConfig

    cfg = OptimizationConfig(program="p", metric="m")
    assert cfg.variance_estimate_samples == 5
    d = cfg.to_dict()
    assert d["variance_estimate_samples"] == 5
    d["repo_url"] = None
    assert OptimizationConfig.from_dict({**d, "variance_estimate_samples": 2}).variance_estimate_samples == 2
    with pytest.raises(ValueError, match="variance_estimate_samples"):
        OptimizationConfig(program="p", metric="m", variance_estimate_samples=1)
    assert OptimizeRequest.model_fields["variance_estimate_samples"].default == 5
    assert BaseAgentConfig.__dataclass_fields__["variance_estimate_samples"].default == 5
