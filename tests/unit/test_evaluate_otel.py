"""Copyright © 2026 440 Labs LLC

End-to-end test for OpenTelemetry trace capture inside the mounted evaluator.

Verifies that a client program decorating one inner function with
`tracer.start_as_current_span(...)` produces a properly-shaped `trace[]`
in the JSONL the evaluator writes, with parent/child wiring resolved.
"""

from __future__ import annotations

import json
import sys
import tempfile

import pytest

from src.engine.telemetry import OTEL_AVAILABLE

pytestmark = pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")

if OTEL_AVAILABLE:
    from opentelemetry import trace


FIXTURE_MODULE_NAME = "_test_evaluate_otel_fixtures"


def _install_fixture_module() -> None:
    """Register the fake program + metric in sys.modules so _resolve can find them."""
    import types

    mod = types.ModuleType(FIXTURE_MODULE_NAME)

    def fake_program(statement: str):
        tracer = trace.get_tracer("fake.client")
        with tracer.start_as_current_span("function_name") as s:
            s.set_attribute("ce.span_kind", "tool")
            s.set_attribute("ce.inputs.statement", statement)
            s.set_attribute("ce.output", "claim_1")
            return {"verdict": "false"}

    def failing_program(statement: str):
        raise RuntimeError("boom")

    def accuracy(output, example):
        return 1.0

    mod.fake_program = fake_program
    mod.failing_program = failing_program
    mod.accuracy = accuracy
    sys.modules[FIXTURE_MODULE_NAME] = mod


def _build_cmd(program_name: str, capture_traces: bool = True) -> dict:
    return {
        "program_factory": f"{FIXTURE_MODULE_NAME}:{program_name}",
        "metric_fns": [f"{FIXTURE_MODULE_NAME}:accuracy"],
        "metric_names": ["accuracy"],
        "primary_metric": "accuracy",
        "row_ids": None,
        "iteration_idx": 0,
        "suffix": "child",
        "capture_traces": capture_traces,
        "extra": {"trainset_path": "trainset.json"},
    }


def _write_trainset(workspace: str, rows: list[dict]) -> None:
    import os

    os.makedirs(workspace, exist_ok=True)
    with open(f"{workspace}/trainset.json", "w") as f:
        json.dump(rows, f)


@pytest.fixture(autouse=True)
def _install_fixtures_and_redirect_traces(monkeypatch):
    _install_fixture_module()
    # Point /traces to a writable temp dir so the evaluator's os.makedirs works.
    # `evaluate._traces_dir()` resolves CODEEVOLVER_TRACES_DIR first, so setting
    # the env var is the supported override.
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", tmp)
    yield tmp


def _read_single_entry(traces_dir: str, iteration_idx: int, suffix: str) -> dict:
    from src.engine.telemetry.trace_layout import eval_trace_dir, iter_rows

    rows = list(iter_rows(eval_trace_dir(traces_dir, iteration_idx, suffix), with_spans=True))
    assert len(rows) == 1, f"expected 1 trace row, got {len(rows)}"
    return rows[0]


def test_captures_client_spans_with_parent_child_wiring(tmp_path, _install_fixtures_and_redirect_traces):
    traces_dir = _install_fixtures_and_redirect_traces
    workspace = str(tmp_path)
    _write_trainset(workspace, [{"statement": "x"}])

    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_build_cmd("fake_program"), workspace=workspace)
    assert result["success"], result.get("error")

    entry = _read_single_entry(traces_dir, 0, "child")
    trace_items = entry["trace"]
    assert [t["signature_key"] for t in trace_items] == ["pipeline", "function_name"]

    root, inner = trace_items
    assert root["parent_key"] is None
    assert root["span_kind"] == "chain"
    assert root["inputs"]["statement"] == "x"

    assert inner["parent_key"] == "pipeline"
    assert inner["span_kind"] == "tool"
    assert inner["inputs"]["statement"] == "x"
    assert inner["output"] == "claim_1"


def test_program_exception_populates_error_on_root_span(tmp_path, _install_fixtures_and_redirect_traces):
    traces_dir = _install_fixtures_and_redirect_traces
    workspace = str(tmp_path)
    _write_trainset(workspace, [{"statement": "oops"}])

    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_build_cmd("failing_program"), workspace=workspace)
    assert result["success"], result.get("error")

    entry = _read_single_entry(traces_dir, 0, "child")
    assert entry["failed"] is True
    assert entry["error_summary"] is not None

    trace_items = entry["trace"]
    assert len(trace_items) == 1
    root = trace_items[0]
    assert root["signature_key"] == "pipeline"
    assert "boom" in (root["error"] or "")


def test_capture_traces_false_produces_empty_trace(tmp_path, _install_fixtures_and_redirect_traces):
    traces_dir = _install_fixtures_and_redirect_traces
    workspace = str(tmp_path)
    _write_trainset(workspace, [{"statement": "x"}])

    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_build_cmd("fake_program", capture_traces=False), workspace=workspace)
    assert result["success"], result.get("error")

    entry = _read_single_entry(traces_dir, 0, "child")
    assert entry["trace"] == []
