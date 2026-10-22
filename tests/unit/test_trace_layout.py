"""Copyright © 2026 440 Labs LLC

The per-eval trace directory (`src/engine/telemetry/trace_layout.py`) and the
fences that cover it and the architect trace directory.
"""

from __future__ import annotations

import json
import os

import pytest

from src.engine.telemetry import trace_layout as tl


def _rows(n: int = 2, *, big: int = 0) -> list[dict]:
    rows = []
    for i in range(n):
        rows.append({
            "example_id": i,
            "score": 1.0 if i else 0.0,
            "scores_by_metric": {"acc": 1.0 if i else 0.0},
            "failed": False,
            "feedback": f"row {i} feedback",
            "feedback_by_metric": None,
            "error_summary": None,
            "input": {"q": f"question {i}"},
            "output": {"a": "x" * big} if big else {"a": f"answer {i}"},
            "trace": [
                {"signature_key": "pipeline", "span_kind": "chain", "parent_key": None,
                 "inputs": {"q": f"question {i}"}, "output": "o", "error": None},
                {"signature_key": "LM.__call__", "span_kind": "llm", "parent_key": "pipeline",
                 "inputs": {"messages": [{"role": "user", "content": "p" * big}]},
                 "output": "o", "error": "boom" if i == 0 else None,
                 "metadata": {"llm.model_name": "m"}},
            ],
        })
    return rows


def test_directory_shape_index_and_span_files(tmp_path):
    d = str(tmp_path / "iteration_3_child")
    index_path, meta = tl.write_eval_trace(d, _rows(), max_line_chars=None)
    assert index_path == os.path.join(d, "index.jsonl")
    assert sorted(os.listdir(d)) == ["index.jsonl", "index_pretty_print.txt", "rows"]
    assert sorted(os.listdir(os.path.join(d, "rows", "0"))) == [
        "row.json", "span_000.json", "span_001.json",
    ]
    # index == trace_meta, and it carries the row_file pointer.
    with open(index_path) as f:
        index = [json.loads(l) for l in f if l.strip()]
    assert index == meta
    assert index[0] == {
        "example_id": 0, "score": 0.0, "failed": False,
        "feedback": "row 0 feedback", "error_summary": None,
        "row_file": "rows/0/row.json",
    }
    # row.json holds everything but the span bodies + a spans table.
    row = json.load(open(os.path.join(d, "rows", "0", "row.json")))
    assert "trace" not in row
    assert [s["file"] for s in row["spans"]] == ["span_000.json", "span_001.json"]
    assert row["spans"][1]["signature_key"] == "LM.__call__"
    assert row["spans"][1]["error"] == "boom"
    assert row["spans"][1]["parent_key"] == "pipeline"
    span = json.load(open(os.path.join(d, "rows", "0", "span_001.json")))
    assert span["inputs"]["messages"][0]["role"] == "user"
    # Pretty-printed: many short lines, not one long one.
    assert open(os.path.join(d, "rows", "0", "row.json")).read().count("\n") > 5


def test_roundtrip_readers(tmp_path):
    d = str(tmp_path / "iteration_3_child")
    rows = _rows(3)
    tl.write_eval_trace(d, rows, max_line_chars=None)
    assert [r["example_id"] for r in tl.read_index(d)] == [0, 1, 2]
    back = list(tl.iter_rows(d, with_spans=True))
    for b in back:
        b.pop("spans")
    assert back == rows
    assert tl.read_index(str(tmp_path / "missing")) == []


def test_chunking_keeps_every_line_under_the_cap_and_rejoins(tmp_path):
    d = str(tmp_path / "iteration_3_child")
    rows = _rows(1, big=30_000)
    tl.write_eval_trace(d, rows, max_line_chars=1900)
    for root, _, files in os.walk(d):
        for name in files:
            text = open(os.path.join(root, name), encoding="utf-8").read()
            assert max(len(l) for l in text.splitlines()) < 2000, name
    back = list(tl.iter_rows(d, with_spans=True))
    back[0].pop("spans")
    assert back == rows


def test_rewrite_is_atomic_and_removes_stale_rows(tmp_path):
    d = str(tmp_path / "iteration_3_child")
    tl.write_eval_trace(d, _rows(3), max_line_chars=None)
    assert os.path.isdir(os.path.join(d, "rows", "2"))
    tl.write_eval_trace(d, _rows(1), max_line_chars=None)
    assert sorted(os.listdir(os.path.join(d, "rows"))) == ["0"]
    assert not os.path.exists(d + ".tmp")


def test_provenance_stamp_roundtrip_and_rewrite_contract(tmp_path):
    """`provenance.json` is written by the host AFTER `write_eval_trace`; the
    next eval's rewrite removes it, so the host must re-stamp every time."""
    d = str(tmp_path / "iteration_3_child")
    assert tl.read_provenance(d) is None
    assert tl.write_provenance(d, {"eval_id": "x"}) is False  # no directory yet
    tl.write_eval_trace(d, _rows(1), max_line_chars=None)
    stamp = {"eval_id": "abc123", "sha_at_eval": "deadbeef", "tree": "cafe", "score": 0.5}
    assert tl.write_provenance(d, stamp) is True
    assert tl.read_provenance(d) == stamp
    assert sorted(os.listdir(d)) == [
        "index.jsonl", "index_pretty_print.txt", "provenance.json", "rows",
    ]
    assert not os.path.exists(tl.provenance_path(d) + ".tmp")
    # Readers of index/rows are unaffected by the sidecar.
    assert [r["example_id"] for r in tl.read_index(d)] == [0]
    tl.write_eval_trace(d, _rows(1), max_line_chars=None)
    assert tl.read_provenance(d) is None


def test_none_example_id_falls_back_to_line_position(tmp_path):
    d = str(tmp_path / "iteration_3_child")
    rows = _rows(1)
    rows[0]["example_id"] = None
    _, meta = tl.write_eval_trace(d, rows, max_line_chars=None)
    assert meta[0]["row_file"] == "rows/line_0/row.json"
    assert list(tl.iter_rows(d))[0]["input"] == {"q": "question 0"}


def test_holdout_suffixes_carry_the_fence_fragment():
    assert tl.holdout_suffix(0) == "holdout"
    assert tl.holdout_suffix(1) == "holdout_2"
    with pytest.raises(ValueError):
        tl.holdout_suffix(-1)
    for k in range(4):
        assert tl.HOLDOUT_DENY_FRAGMENT in tl.eval_trace_dir_name(7, tl.holdout_suffix(k))
        assert tl.is_holdout_suffix(tl.holdout_suffix(k))
    for s in ("test", "test_2", "holdouts", "child", "parent", "comparison"):
        assert not tl.is_holdout_suffix(s)
    # A client path like `src/my_test/` must NOT be caught by the fragment.
    assert tl.HOLDOUT_DENY_FRAGMENT not in "src/my_test/data.json"


# --- fences over the new directories ---------------------------------------


def _pre_tool_use(tool_name: str, **tool_input):
    return {"tool_name": tool_name, "tool_input": tool_input}


def test_sub_agent_deny_hook_covers_holdout_and_architect_dirs():
    import asyncio

    from src.engine.agent.mounted.environment_setup import create_deny_paths_hook
    from src.engine.optimizer.v3.architect_tool_registry import (
        ARCHITECT_TRACE_DENY_FRAGMENT,
        _build_holdout_deny_paths,
    )

    deny = _build_holdout_deny_paths("data/bench_test.json") + [ARCHITECT_TRACE_DENY_FRAGMENT]
    hook = create_deny_paths_hook(deny, log_prefix="TEST")

    def run(call):
        return asyncio.run(hook(call, "id", None))

    denied = [
        _pre_tool_use("Read", file_path="/traces/job/iteration_4_holdout/rows/0/row.json"),
        _pre_tool_use("Read", file_path="/traces/job/iteration_4_holdout_2/index.jsonl"),
        _pre_tool_use("Bash", command="cat /traces/job/iteration_4_holdout/rows/0/span_001.json"),
        _pre_tool_use("Grep", path="/traces/job", pattern="iteration_4_holdout"),
        _pre_tool_use("Read", file_path="/architect_traces/job/iteration_4_architect_trace/turn_0003.json"),
        _pre_tool_use("Glob", path="/architect_traces/job/iteration_4_architect_trace", pattern="*.json"),
    ]
    for call in denied:
        out = run(call)
        assert out, call
        assert out["hookSpecificOutput"]["permissionDecision"] == "deny", call
    allowed = [
        _pre_tool_use("Read", file_path="/traces/job/iteration_4_child/rows/0/row.json"),
        _pre_tool_use("Read", file_path="/traces/job/iteration_4_parent/index.jsonl"),
        _pre_tool_use("Read", file_path="/workspace/src/my_test/data.json"),
    ]
    for call in allowed:
        assert run(call) == {}, call


def test_architect_guard_covers_holdout_and_architect_dirs():
    import asyncio

    from src.engine.optimizer.v3.architect_hooks import _make_holdout_guard_hook

    guard = _make_holdout_guard_hook(2, "data/bench_test.json", ["extra_secret"])

    def run(call):
        return asyncio.run(guard(call, "id", None))

    for call in (
        _pre_tool_use("Read", file_path="/architect_traces/job/iteration_2_architect_trace/index.jsonl"),
        _pre_tool_use("Read", file_path="/traces/job/iteration_2_holdout/rows/0/row.json"),
        _pre_tool_use("Bash", command="ls /traces/job/iteration_2_holdout_2/rows"),
    ):
        assert run(call)["hookSpecificOutput"]["permissionDecision"] == "deny", call
    assert run(_pre_tool_use("Read", file_path="/traces/job/iteration_2_child/rows/1/row.json")) == {}
    assert run(_pre_tool_use("Read", file_path="/workspace/src/my_test/data.json")) == {}


def test_pretty_index_file_written_capped_and_readable(tmp_path):
    """`index_pretty_print.txt`: the run_eval response carries no inline
    trace_meta, so every trace dir carries a readable rendering of
    the index — first PRETTY_MAX_ROWS rows, every line capped at
    PRETTY_MAX_LINE_CHARS chars."""
    from src.engine.telemetry.trace_layout import (
        PRETTY_MAX_LINE_CHARS,
        PRETTY_MAX_ROWS,
        pretty_index_path,
        write_eval_trace,
    )

    td = str(tmp_path / "iteration_3_child")
    entries = [
        {
            "example_id": i,
            "score": 0.0 if i == 1 else 1.0,
            "failed": i == 1,
            "feedback": f"row {i}: " + "long feedback " * 400,
            "error_summary": "RateLimitError: 503" if i == 1 else None,
            "trace": [],
        }
        for i in range(PRETTY_MAX_ROWS + 5)
    ]
    write_eval_trace(td, entries, max_line_chars=None)
    text = open(pretty_index_path(td), encoding="utf-8").read()
    assert f"{PRETTY_MAX_ROWS} of {PRETTY_MAX_ROWS + 5} rows" in text
    assert "example_id=1" in text and "[FAILED]" in text
    assert "RateLimitError: 503" in text
    assert max(len(line) for line in text.splitlines()) <= PRETTY_MAX_LINE_CHARS
    # Rewritten atomically with the rest of the directory.
    write_eval_trace(td, entries[:2], max_line_chars=None)
    assert "2 of 2 rows" in open(pretty_index_path(td), encoding="utf-8").read()
