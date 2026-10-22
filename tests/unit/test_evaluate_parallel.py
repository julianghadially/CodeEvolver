"""Copyright © 2026 440 Labs LLC

Parallel-correctness tests for the mounted evaluator.

When `num_threads > 1`, rows run concurrently on a ThreadPoolExecutor inside the
evaluator subprocess. These tests prove that:
  - rows are emitted in stable row order regardless of completion order,
  - each row's `trace[]` contains ONLY its own spans (no cross-row leakage from
    the shared OTel provider) — the core correctness guarantee of the
    contextvar-scoped span collection,
  - parallel aggregate scores equal the sequential run's,
  - output is deterministic across repeated parallel runs,
  - execution is genuinely parallel (a Barrier wired into the fixture program
    forces all concurrent workers to rendezvous; serialized execution would
    time out and fail every row).
"""

from __future__ import annotations

import json
import sys
import tempfile
import threading

import pytest

from src.engine.telemetry import OTEL_AVAILABLE

pytestmark = pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")

if OTEL_AVAILABLE:
    from opentelemetry import trace


FIXTURE_MODULE_NAME = "_test_evaluate_parallel_fixtures"
N_ROWS = 8
NUM_THREADS = 4


def _install_fixture_module(barrier: threading.Barrier | None) -> None:
    """Register a program + metric whose inner span carries a per-row marker.

    The program waits on `barrier` (when set) so that real parallelism is
    required for it to proceed — under serialized execution the barrier times
    out and the program raises, failing the row.
    """
    import types

    mod = types.ModuleType(FIXTURE_MODULE_NAME)
    mod._barrier = barrier  # type: ignore[attr-defined]

    def parallel_program(idx: int):
        b = getattr(mod, "_barrier", None)
        if b is not None:
            # Times out (raising BrokenBarrierError) if workers don't run
            # concurrently — turns "silently serialized" into a hard failure.
            b.wait(timeout=10.0)
        tracer = trace.get_tracer("fake.client")
        with tracer.start_as_current_span("inner") as s:
            s.set_attribute("ce.span_kind", "tool")
            s.set_attribute("ce.inputs.idx", idx)
            s.set_attribute("ce.output", f"out_{idx}")
            return {"value": idx}

    def accuracy(output, example):
        # Deterministic, row-dependent so a mis-partition would change the mean.
        return 1.0 if output.get("value") == example["idx"] else 0.0

    mod.parallel_program = parallel_program
    mod.accuracy = accuracy
    sys.modules[FIXTURE_MODULE_NAME] = mod


def _build_cmd(num_threads: int, suffix: str = "child") -> dict:
    return {
        "program_factory": f"{FIXTURE_MODULE_NAME}:parallel_program",
        "metric_fns": [f"{FIXTURE_MODULE_NAME}:accuracy"],
        "metric_names": ["accuracy"],
        "primary_metric": "accuracy",
        "input_keys": ["idx"],
        "row_ids": None,
        "iteration_idx": 0,
        "suffix": suffix,
        "capture_traces": True,
        "num_threads": num_threads,
        "extra": {"trainset_path": "trainset.json"},
    }


def _write_trainset(workspace: str) -> None:
    import os

    os.makedirs(workspace, exist_ok=True)
    with open(f"{workspace}/trainset.json", "w") as f:
        json.dump([{"idx": i} for i in range(N_ROWS)], f)


def _read_trace_lines(traces_dir: str, suffix: str) -> list[dict]:
    from src.engine.telemetry.trace_layout import eval_trace_dir, iter_rows

    return list(iter_rows(eval_trace_dir(traces_dir, 0, suffix), with_spans=True))


def _read_index_text(traces_dir: str, suffix: str) -> str:
    from src.engine.telemetry.trace_layout import eval_trace_dir, index_path

    with open(index_path(eval_trace_dir(traces_dir, 0, suffix))) as f:
        return f.read()


@pytest.fixture
def traces_dir(monkeypatch):
    tmp = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", tmp)
    return tmp


def test_parallel_rows_are_partitioned_and_ordered(tmp_path, traces_dir):
    """Each row's trace holds only its own spans; rows stay in row order."""
    _install_fixture_module(threading.Barrier(min(N_ROWS, NUM_THREADS)))
    workspace = str(tmp_path)
    _write_trainset(workspace)

    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_build_cmd(NUM_THREADS), workspace=workspace)
    assert result["success"], result.get("error")
    assert result["row_ids"] == list(range(N_ROWS))

    lines = _read_trace_lines(traces_dir, "child")
    assert len(lines) == N_ROWS

    for i, entry in enumerate(lines):
        # Row order preserved despite nondeterministic completion order.
        assert entry["example_id"] == i, f"row {i} out of order"
        assert entry["failed"] is False, entry.get("error_summary")

        # Partition: exactly this row's two spans, no leakage from other rows.
        sig_keys = [t["signature_key"] for t in entry["trace"]]
        assert sig_keys == ["pipeline", "inner"], f"row {i}: {sig_keys}"

        # Compare as strings: the root's inputs are stringified by the
        # evaluator (`str(v)`), the inner span keeps the int — both must point
        # at THIS row only.
        root, inner = entry["trace"]
        assert str(root["inputs"]["idx"]) == str(i)
        assert inner["parent_key"] == "pipeline"
        assert str(inner["inputs"]["idx"]) == str(i)  # marker matches THIS row only


def test_parallel_matches_sequential(tmp_path, monkeypatch):
    """Aggregate scores are identical whether run with 1 or many threads."""
    # Sequential run (no barrier — a barrier would deadlock single-threaded).
    seq_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", seq_dir)
    _install_fixture_module(None)
    ws_seq = str(tmp_path / "seq")
    _write_trainset(ws_seq)

    from src.engine.evaluator.mounted.evaluate import handle

    seq = handle(_build_cmd(1), workspace=ws_seq)
    assert seq["success"], seq.get("error")

    # Parallel run.
    par_dir = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", par_dir)
    _install_fixture_module(threading.Barrier(min(N_ROWS, NUM_THREADS)))
    ws_par = str(tmp_path / "par")
    _write_trainset(ws_par)

    par = handle(_build_cmd(NUM_THREADS), workspace=ws_par)
    assert par["success"], par.get("error")

    assert par["scores_by_metric"] == seq["scores_by_metric"]
    assert seq["scores_by_metric"]["accuracy"] == 1.0  # all rows correct
    # Index files (example_id + scores, no timestamps) match across modes.
    assert _read_index_text(par_dir, "child") == _read_index_text(seq_dir, "child")


def test_parallel_is_deterministic_across_runs(tmp_path, monkeypatch):
    """Two parallel runs produce identical index files (order + scores)."""
    from src.engine.evaluator.mounted.evaluate import handle

    dir_a = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", dir_a)
    _install_fixture_module(threading.Barrier(min(N_ROWS, NUM_THREADS)))
    ws_a = str(tmp_path / "a")
    _write_trainset(ws_a)
    a = handle(_build_cmd(NUM_THREADS), workspace=ws_a)
    assert a["success"], a.get("error")

    dir_b = tempfile.mkdtemp()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", dir_b)
    _install_fixture_module(threading.Barrier(min(N_ROWS, NUM_THREADS)))
    ws_b = str(tmp_path / "b")
    _write_trainset(ws_b)
    b = handle(_build_cmd(NUM_THREADS), workspace=ws_b)
    assert b["success"], b.get("error")

    assert _read_index_text(dir_a, "child") == _read_index_text(dir_b, "child")


# ---------------------------------------------------------------------------
# Re-parenting across a dropped DSPy wrapper, end-to-end through handle() under
# threads. Exercises the real dspy filter + the per-row drop-link skip-map.
# ---------------------------------------------------------------------------

REPARENT_MODULE_NAME = "_test_evaluate_reparent_fixtures"


def _install_reparent_module() -> None:
    import types

    mod = types.ModuleType(REPARENT_MODULE_NAME)

    def reparent_program(idx: int):
        tracer = trace.get_tracer("fake.client")
        # `ChatAdapter.__call__` is in the DSPy filter's DROP_NAMES, so it is
        # filtered out; `LM.__call__` (its child) must re-parent to `pipeline`.
        with tracer.start_as_current_span("ChatAdapter.__call__"):
            with tracer.start_as_current_span("LM.__call__") as leaf:
                leaf.set_attribute("ce.span_kind", "llm")
                leaf.set_attribute("ce.inputs.idx", idx)
                return {"value": idx}

    def accuracy(output, example):
        return 1.0 if output.get("value") == example["idx"] else 0.0

    mod.reparent_program = reparent_program
    mod.accuracy = accuracy
    sys.modules[REPARENT_MODULE_NAME] = mod


def _build_reparent_cmd(num_threads: int) -> dict:
    cmd = _build_cmd(num_threads)
    cmd["program_factory"] = f"{REPARENT_MODULE_NAME}:reparent_program"
    cmd["metric_fns"] = [f"{REPARENT_MODULE_NAME}:accuracy"]
    return cmd


def test_dropped_wrapper_reparents_to_pipeline_under_threads(tmp_path, traces_dir):
    """LM.__call__ whose ChatAdapter parent was filtered re-parents to pipeline,
    per row, with no cross-row drop-link contamination under threads."""
    _install_reparent_module()
    workspace = str(tmp_path)
    _write_trainset(workspace)

    from src.engine.evaluator.mounted.evaluate import handle

    result = handle(_build_reparent_cmd(NUM_THREADS), workspace=workspace)
    assert result["success"], result.get("error")

    lines = _read_trace_lines(traces_dir, "child")
    assert len(lines) == N_ROWS
    for i, entry in enumerate(lines):
        assert entry["example_id"] == i
        sig_keys = [t["signature_key"] for t in entry["trace"]]
        # ChatAdapter dropped; LM.__call__ collapsed up to pipeline.
        assert sig_keys == ["pipeline", "LM.__call__"], f"row {i}: {sig_keys}"
        pipeline_entry, lm = entry["trace"]
        assert pipeline_entry["parent_key"] is None
        assert lm["parent_key"] == "pipeline"  # re-parented across the drop
        assert str(lm["inputs"]["idx"]) == str(i)  # correct row, no bleed
