"""Copyright © 2026 440 Labs LLC

Loop-level wiring of the comparison-credit ledger (specs/archive-public/comparison_credits.md).

`tests/unit/test_comparison_credits.py` proves the gate given a context and
state the test built by hand. This file proves the LOOP builds them: that
`_run_optimizer_loop` grants +N at iteration start, stamps a 2-digit
acknowledgement code on the context, threads the production tool registry
so a comparison call through it debits the shared state, stamps
`comparison_credits_after` on the close record, and does none of that when
the ledger is disabled.

Drives the real `_run_optimizer_loop` for one iteration against a real git
workspace with a stub architect (no model) in place of `IterationArchitect`;
the stub uses the PRODUCTION `build_tool_registry` so `run_eval` is the real
binding, with only the evaluator subprocess patched.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3 import optimizer_loop, run_eval_tool
from src.engine.optimizer.v3.architect_tool_registry import build_tool_registry
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.ee.ee_optimizer import EE_STRATEGY
from src.engine.optimizer.v3.optimization_config import OptimizationConfig
from src.engine.optimizer.v3.optimizer_loop import OptimizerHooks, _run_optimizer_loop
from src.engine.schemas.iteration_history import IterationHistory

MAIN = "codeevolver-20260601120000-main"


def _git(ws: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=ws, capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.fixture()
def env(tmp_path, monkeypatch):
    for name in ("MEMORY", "HISTORY", "TRACES", "ARCHITECT_TRACES", "LOGS"):
        d = tmp_path / name.lower()
        d.mkdir()
        monkeypatch.setenv(f"CODEEVOLVER_{name}_DIR", str(d))
    ws = tmp_path / "repo"
    ws.mkdir()
    _git(ws, "init", "-q", "-b", MAIN)
    _git(ws, "config", "user.email", "t@t")
    _git(ws, "config", "user.name", "t")
    (ws / "program.py").write_text("K = 7\n")
    (ws / "codeevolver.md").write_text("# Architecture\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-q", "-m", "seed")
    return ws


def _fake_run_eval(**kwargs: Any) -> EvalResult:
    rows = list(kwargs.get("row_ids") or [])
    return EvalResult(
        scores_by_metric={"accuracy": 0.5},
        primary_score=0.5,
        trace_meta=[{"example_id": r, "score": 0.5} for r in rows],
        trace_dir="/t",
        trace_index_file="/ti",
        row_ids=rows,
        raw={},
    )


def _run_one_iteration(
    ws: Path,
    monkeypatch,
    *,
    per_iteration: int | None,
    initial_balance: int,
    agent_actions,
) -> tuple[CEState, IterationHistory, list[Any]]:
    """Run exactly one loop iteration with a stub architect whose `run()`
    calls `agent_actions(context, registry, state)` against the production
    tool registry."""
    seen: list[Any] = []

    class _StubArchitect:
        def __init__(self, *, context, state, workspace, config, rng):
            self.context, self.state, self.workspace, self.config, self.rng = (
                context, state, workspace, config, rng,
            )

        def run(self) -> None:
            seen.append(self.context)
            registry = build_tool_registry(
                context=self.context, state=self.state, workspace=self.workspace,
                cfg=self.config, rng=self.rng, agent_harness=None,
            )
            agent_actions(self.context, registry, self.state)

    monkeypatch.setattr(optimizer_loop, "IterationArchitect", _StubArchitect)
    monkeypatch.setattr(run_eval_tool, "_run_eval", _fake_run_eval)

    config = OptimizationConfig(
        program="program:make", metric="metrics:acc", workspace=str(ws),
        max_metric_calls=100, subsample_size=3, comparison_set_size=3,
        comparison_credits_initial=1, comparison_credits_per_iteration=per_iteration,
        seed=7,
    )
    state = CEState(
        seed_branch=MAIN, seed_sha=_git(ws, "rev-parse", "HEAD"),
        comparison_history_path=str(ws.parent / "history" / "comparison_set_history.jsonl"),
    )
    state.grant_comparison_credits(initial_balance)
    history = IterationHistory()
    history.create_entry(0).update({"git_branch": MAIN, "skipped": False})
    hooks = OptimizerHooks(budget_exhausted=lambda i: i >= 2)
    evaluator = EvaluatorConfig(
        program_factory="program:make", metric_fns=["metrics:acc"],
        metric_names=["accuracy"], primary_metric="accuracy",
        trainset_path=str(ws / "train.json"),
    )
    _run_optimizer_loop(
        config=config, strategy=EE_STRATEGY, state=state, workspace=str(ws),
        evaluator_config=evaluator, trainset_ids=[0, 1, 2, 3, 4, 5],
        comparison_ids=[0, 1, 2], hooks=hooks, history=history, start_iteration=1,
    )
    return state, history, seen


def test_loop_grants_stamps_code_and_records_balance(env, monkeypatch):
    def idle(ctx, registry, state):
        pass

    state, history, seen = _run_one_iteration(
        env, monkeypatch, per_iteration=1, initial_balance=1, agent_actions=idle,
    )
    ctx = seen[0]
    assert len(ctx.comparison_ack_code) == 2 and ctx.comparison_ack_code.isdigit()
    assert state.comparison_credits_balance == 2  # 1 carried in + 1 at iteration start
    assert history[1]["comparison_credits_after"] == 2
    assert state.iterations[-1]["comparison_credits_after"] == 2


def test_loop_threads_the_gate_through_the_production_registry(env, monkeypatch):
    """A comparison call through the loop-built registry is refused until
    acknowledged with the code the LOOP stamped, then debits the loop's state."""
    trace: list[Any] = []

    def spend(ctx, registry, state):
        first = registry["run_eval"]("comparison")
        trace.append(first)
        second = registry["run_eval"](
            "comparison", acknowledge_message=ctx.comparison_ack_code
        )
        trace.append(second)

    state, history, seen = _run_one_iteration(
        env, monkeypatch, per_iteration=1, initial_balance=0, agent_actions=spend,
    )
    assert trace[0]["ok"] is False and trace[0]["refused"] == "acknowledgement_required"
    assert trace[0]["verification_code"] == seen[0].comparison_ack_code
    assert trace[1]["ok"] is True and trace[1]["comparison_credits_remaining"] == 0
    assert state.comparison_credits_balance == 0
    assert history[1]["comparison_credits_after"] == 0
    assert state.total_num_evals == 3


def test_loop_disabled_ledger_grants_nothing_and_records_none(env, monkeypatch):
    def compare_freely(ctx, registry, state):
        resp = registry["run_eval"]("comparison")  # no code needed
        assert resp["ok"] is True and "comparison_credits_remaining" not in resp

    state, history, seen = _run_one_iteration(
        env, monkeypatch, per_iteration=None, initial_balance=0,
        agent_actions=compare_freely,
    )
    assert seen[0].comparison_ack_code == ""
    assert state.comparison_credits_balance == 0
    assert history[1]["comparison_credits_after"] is None


def test_ack_code_is_reproducible_from_the_seed(env, monkeypatch):
    codes = []
    for _ in range(2):
        _, _, seen = _run_one_iteration(
            env, monkeypatch, per_iteration=1, initial_balance=0,
            agent_actions=lambda c, r, s: None,
        )
        codes.append(seen[0].comparison_ack_code)
        _git(env, "reset", "-q", "--hard", "HEAD")
    assert codes[0] == codes[1]
