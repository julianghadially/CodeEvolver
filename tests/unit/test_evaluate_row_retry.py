"""Copyright © 2026 440 Labs LLC

Row-level retry of PROVIDER faults in the mounted evaluator.

GMI intermittently answers a single row with `Error code: 402 - {'error':
'Insufficient balance', 'reason': 'model_access_denied'}` and then serves the
next row normally, so it is a provider fault rather than a billing state.
litellm never retries 402 (`utils._should_retry` covers 408/409/429/5xx only),
so before this the row scored 0.0 and quietly depressed the aggregate.

The tests that matter most here are the NEGATIVE ones: a retry must never fire
for the evolved program's own exceptions, and a retried row must never be
counted twice against the eval-row budget.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

FIXTURE_MODULE_NAME = "_test_evaluate_row_retry_fixtures"
N_ROWS = 4

# Verbatim provider error text.
GMI_402 = (
    "litellm.APIError: APIError: OpenAIException - Error code: 402 - "
    "{'error': 'Insufficient balance', 'reason': 'model_access_denied'}"
)


class APIError(Exception):
    """Named to match litellm's, since the classifier reads the type name."""


def _install_fixture(fail_times: int, exc_factory) -> dict:
    """Program that raises `exc_factory()` the first `fail_times` calls per row."""
    mod = types.ModuleType(FIXTURE_MODULE_NAME)
    calls: dict[int, int] = {}

    def program(idx: int):
        seen = calls.get(idx, 0)
        calls[idx] = seen + 1
        if seen < fail_times:
            raise exc_factory()
        return {"value": idx}

    def accuracy(output, example):
        return 1.0 if output.get("value") == example["idx"] else 0.0

    mod.program = program
    mod.accuracy = accuracy
    sys.modules[FIXTURE_MODULE_NAME] = mod
    return calls


def _cmd(num_threads: int = 1) -> dict:
    return {
        "program_factory": f"{FIXTURE_MODULE_NAME}:program",
        "metric_fns": [f"{FIXTURE_MODULE_NAME}:accuracy"],
        "metric_names": ["accuracy"],
        "primary_metric": "accuracy",
        "input_keys": ["idx"],
        "row_ids": None,
        "iteration_idx": 0,
        "suffix": "child",
        "capture_traces": False,
        "num_threads": num_threads,
        "extra": {"trainset_path": "trainset.json"},
    }


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    (tmp_path / "trainset.json").write_text(
        json.dumps([{"idx": i} for i in range(N_ROWS)])
    )
    return str(tmp_path)


@pytest.fixture(autouse=True)
def _fast_backoff(monkeypatch):
    """Keep the real schedule's SHAPE (two attempts) without its 63s wall time.

    The per-eval row cap is lifted here so these tests exercise the retry
    mechanism itself; `test_the_row_cap_bounds_a_sustained_outage` pins the cap
    separately rather than letting it confound every other assertion.
    """
    from src.engine.evaluator.mounted import evaluate

    monkeypatch.setattr(evaluate, "RETRY_BACKOFF_SECONDS", (0.0, 0.0))
    monkeypatch.setattr(evaluate, "RETRY_ROW_FRACTION_CAP", 1.0)


def test_gmi_402_is_retried_and_the_row_recovers(workspace):
    """The canonical case: two 402s, then the provider behaves."""
    _install_fixture(2, lambda: APIError(GMI_402))
    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_cmd(), workspace=workspace)
    assert result["success"], result.get("error")
    # Recovered rows score for real rather than 0.0.
    assert result["scores_by_metric"]["accuracy"] == 1.0
    assert all(not m.get("failed") for m in result["trace_meta"])

    summary = result["retry_summary"]
    assert summary["rows"] == N_ROWS
    assert summary["recovered"] == N_ROWS
    assert summary["exhausted"] == 0
    assert summary["attempts"] == 2 * N_ROWS


def test_a_program_bug_is_never_retried(workspace):
    """The benchmark-protecting case.

    Re-running the evolved program's OWN crash would score the agent on code
    that does not run, so an unrecognised exception must fail on attempt one.
    """
    calls = _install_fixture(1, lambda: ValueError("program is broken"))
    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_cmd(), workspace=workspace)
    assert result["success"], result.get("error")
    assert result["scores_by_metric"]["accuracy"] == 0.0
    assert all(m.get("failed") for m in result["trace_meta"])
    # One attempt per row: the failure was never re-run.
    assert calls == {i: 1 for i in range(N_ROWS)}
    assert result["retry_summary"] is None


def test_a_402_that_never_clears_still_terminates(workspace):
    """Exhausting the schedule falls back to today's behaviour, never hangs."""
    calls = _install_fixture(99, lambda: APIError(GMI_402))
    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_cmd(), workspace=workspace)
    assert result["success"], result.get("error")
    assert result["scores_by_metric"]["accuracy"] == 0.0
    assert all(m.get("failed") for m in result["trace_meta"])
    # Each row stops after the schedule is spent: 1 initial + 2 retries.
    assert calls == {i: 3 for i in range(N_ROWS)}
    assert result["retry_summary"]["exhausted"] == N_ROWS


def test_the_row_cap_bounds_a_sustained_outage(workspace, monkeypatch):
    """A provider outage must not multiply the wall-clock of every row."""
    from src.engine.evaluator.mounted import evaluate

    # 25% of 4 rows -> 1 row may retry; the other 3 fail on attempt one.
    monkeypatch.setattr(evaluate, "RETRY_ROW_FRACTION_CAP", 0.0)
    calls = _install_fixture(99, lambda: APIError(GMI_402))

    result = evaluate.handle(_cmd(), workspace=workspace)
    assert result["success"], result.get("error")
    # budget = int(4 * 0.0) + 1 = exactly one row gets a slot.
    assert result["retry_summary"]["rows"] == 1
    assert sum(calls.values()) == (N_ROWS - 1) + 3


def test_retries_are_not_charged_to_the_eval_row_budget(workspace):
    """`run_eval_tool` increments the budget by len(trace_meta).

    One entry per ROW however many attempts it took is what keeps a provider
    fault from spending the architect's budget.
    """
    _install_fixture(2, lambda: APIError(GMI_402))
    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_cmd(), workspace=workspace)
    assert len(result["trace_meta"]) == N_ROWS
    assert len(result["row_ids"]) == N_ROWS


def test_retry_works_on_the_parallel_path(workspace):
    """The allowance and tally are shared across pool workers."""
    _install_fixture(1, lambda: APIError(GMI_402))
    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_cmd(num_threads=N_ROWS), workspace=workspace)
    assert result["success"], result.get("error")
    assert result["scores_by_metric"]["accuracy"] == 1.0
    assert len(result["trace_meta"]) == N_ROWS
    assert result["retry_summary"]["recovered"] == N_ROWS
