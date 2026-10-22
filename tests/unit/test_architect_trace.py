"""Copyright © 2026 440 Labs LLC

Unit tests for architect (meta) session trace capture.

These run fully offline — no Claude Agent SDK. They feed duck-typed message
blocks (matching the SDK's TextBlock / ThinkingBlock / ToolUseBlock /
ToolResultBlock surface) to the serializers + writer and assert the JSONL
schema, truncation, the non-persistence of thinking signatures, and the
strictly-non-fatal contract.
"""

from __future__ import annotations

import json
import os

import pytest

from src.engine.optimizer.v3 import architect_trace as at


# --- Duck-typed stand-ins for the Claude Agent SDK content blocks -----------


class FakeText:
    def __init__(self, text):
        self.text = text


class FakeThinking:
    def __init__(self, thinking, signature=None):
        self.thinking = thinking
        self.signature = signature


class FakeToolUse:
    def __init__(self, name, id, input):
        self.name = name
        self.id = id
        self.input = input


class FakeToolResult:
    def __init__(self, tool_use_id, content, is_error=False):
        self.tool_use_id = tool_use_id
        self.content = content
        self.is_error = is_error


class FakeMsg:
    def __init__(self, content):
        self.content = content


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    return tmp_path


def _read_lines(iteration_idx):
    return at.read_architect_trace(iteration_idx)


def test_serialize_assistant_captures_thinking_text_and_tool_use(trace_dir):
    msg = FakeMsg(
        [
            FakeThinking("reconsider retrieval", signature="opaque-sig"),
            FakeText("running a baseline"),
            FakeToolUse("mcp__ce__run_eval", "toolu_1", {"eval_sample": "train"}),
        ]
    )
    blocks = at.serialize_assistant_blocks(msg, FakeText, FakeThinking, FakeToolUse)
    kinds = [b["kind"] for b in blocks]
    assert kinds == ["thinking", "text", "tool_use"]
    # Thinking text is captured untruncated; signature is NOT persisted.
    assert blocks[0]["text"] == "reconsider retrieval"
    assert blocks[0]["signature_present"] is True
    assert "signature" not in blocks[0]
    assert blocks[2]["name"] == "mcp__ce__run_eval"
    assert blocks[2]["input"] == {"eval_sample": "train"}


def test_serialize_user_caps_large_tool_result_and_flags_errors(trace_dir):
    big = "x" * 30_000
    msg = FakeMsg(
        [
            FakeToolResult("toolu_1", big),
            FakeToolResult("toolu_2", "secret missing", is_error=True),
        ]
    )
    blocks = at.serialize_user_blocks(msg, FakeToolResult)
    assert blocks[0]["truncated"] is True
    assert blocks[0]["content"].endswith("[truncated]")
    assert len(blocks[0]["content"]) < len(big)
    assert blocks[1]["is_error"] is True
    assert blocks[1]["truncated"] is False


def test_append_writes_jsonl_in_order(trace_dir):
    at.append_architect_turn(7, role="assistant", blocks=[{"kind": "text", "text": "a"}])
    at.append_architect_turn(
        7, role="user", blocks=[{"kind": "tool_result", "content": "r"}]
    )
    at.append_architect_turn(
        7, role="result", blocks=[], usage={"output_tokens": 5}, meta={"num_turns": 2}
    )
    lines = _read_lines(7)
    assert [ln["role"] for ln in lines] == ["assistant", "user", "result"]
    assert all(ln["iteration"] == 7 and "ts" in ln for ln in lines)
    assert lines[2]["usage"] == {"output_tokens": 5}
    assert lines[2]["meta"] == {"num_turns": 2}
    # assistant/user lines carry no usage/meta keys
    assert "usage" not in lines[0] and "meta" not in lines[0]


def test_opencode_stub_marks_not_captured(trace_dir):
    at.append_opencode_stub(8, stdout_tail="opencode said things")
    line = _read_lines(8)[0]
    assert line["role"] == "result"
    assert line["meta"]["captured"] is False
    assert line["meta"]["harness"] == "opencode"
    assert "opencode said things" in line["meta"]["stdout_tail"]


def test_append_is_non_fatal_on_bad_input(trace_dir):
    # A non-JSON-serializable block must not raise — default=str handles it,
    # and any residual error is swallowed by the writer's guard.
    at.append_architect_turn(9, role="assistant", blocks=[object()])  # must not raise
    # File may or may not exist depending on serialization; the contract is
    # only that no exception escapes.


def test_cap_tool_result_text_caps_and_coerces():
    text, truncated = at.cap_tool_result_text("x" * 30_000)
    assert truncated is True
    assert text.endswith("[truncated]")
    text, truncated = at.cap_tool_result_text({"ok": True})
    assert truncated is False
    assert json.loads(text) == {"ok": True}


def test_write_opencode_trace_writes_schema_valid_lines(trace_dir):
    turns = [
        {"role": "assistant", "blocks": [{"kind": "text", "text": "a"}]},
        {
            "role": "user",
            "blocks": [
                {
                    "kind": "tool_result",
                    "tool_use_id": "call_1",
                    "is_error": False,
                    "content": "r",
                    "truncated": False,
                }
            ],
        },
        {
            "role": "result",
            "blocks": [],
            "usage": {"output_tokens": 5},
            "meta": {"harness": "opencode", "captured": True},
        },
    ]
    assert at.write_opencode_trace(11, turns) == 3
    lines = _read_lines(11)
    assert [ln["role"] for ln in lines] == ["assistant", "user", "result"]
    assert all(ln["iteration"] == 11 and "ts" in ln for ln in lines)
    assert lines[2]["usage"] == {"output_tokens": 5}
    assert lines[2]["meta"]["captured"] is True
    assert "usage" not in lines[0] and "meta" not in lines[0]


def test_write_opencode_trace_rewrites_not_appends(trace_dir):
    at.write_opencode_trace(12, [{"role": "assistant", "blocks": []}])
    # Second capture is a fuller snapshot of the same session — it replaces.
    at.write_opencode_trace(
        12,
        [
            {"role": "assistant", "blocks": [{"kind": "text", "text": "full"}]},
            {"role": "result", "blocks": [], "meta": {"captured": True}},
        ],
    )
    lines = _read_lines(12)
    assert len(lines) == 2
    assert lines[0]["blocks"] == [{"kind": "text", "text": "full"}]


def test_write_opencode_trace_empty_returns_zero(trace_dir):
    assert at.write_opencode_trace(13, []) == 0
    assert not os.path.exists(at.architect_trace_path(13))


def test_trace_path_uses_dedicated_dir(trace_dir):
    path = at.architect_trace_path(3)
    assert path.endswith("iteration_3_architect_trace")
    assert at.ARCHITECT_TRACE_DIR_SUFFIX in os.path.basename(path)
    assert str(trace_dir) in path


def test_directory_layout_has_index_and_turn_files(trace_dir):
    at.append_architect_turn(
        21, role="assistant",
        blocks=[{"kind": "thinking", "text": "t"}, {"kind": "tool_use", "name": "run_eval", "input": {}}],
    )
    at.append_architect_turn(
        21, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "x", "is_error": True, "content": "boom"}],
    )
    d = at.architect_trace_path(21)
    assert sorted(os.listdir(d)) == ["index.jsonl", "turn_0000.json", "turn_0001.json"]
    with open(os.path.join(d, "index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]
    assert index[0]["kinds"] == ["thinking", "tool_use"]
    assert index[0]["tools"] == ["run_eval"]
    assert index[0]["is_error"] is False
    assert index[1]["is_error"] is True
    assert index[1]["file"] == "turn_0001.json"
    with open(os.path.join(d, "turn_0000.json")) as f:
        assert json.load(f)["blocks"][0]["text"] == "t"


def test_turn_files_chunk_long_strings_when_gated(trace_dir, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", "1900")
    at.append_architect_turn(22, role="assistant", blocks=[{"kind": "thinking", "text": "x" * 6000}])
    d = at.architect_trace_path(22)
    with open(os.path.join(d, "turn_0000.json")) as f:
        text = f.read()
    assert max(len(ln) for ln in text.splitlines()) < 2000
    assert at.read_architect_trace(22)[0]["blocks"][0]["text"] == "x" * 6000


def test_index_summary_digests_tool_calls_results_and_text(trace_dir):
    """`summary` is the navigation column: without it the index only says
    that turns happened, and a reader would have to open every turn file of
    an iteration to find anything."""
    at.append_architect_turn(
        31, role="assistant",
        blocks=[
            {"kind": "thinking", "text": "baseline first"},
            {"kind": "text", "text": "Now let me run the   baseline evaluation."},
            {"kind": "tool_use", "name": "ce_run_eval", "input": {"eval_sample": "train"}},
            {"kind": "tool_use", "name": "bash", "input": {"command": "git status", "workdir": "/w"}},
        ],
    )
    at.append_architect_turn(
        31, role="user",
        blocks=[{
            "kind": "tool_result", "tool_use_id": "x", "is_error": False,
            "content": json.dumps({
                "ok": True, "eval_id": "2aabfe228f21", "eval_sample": "train",
                "suffix": "parent", "sha": "d1371c17149d4395", "tree": "729fae923d5a4fd1",
                "row_ids": [1, 5, 24], "primary_score": 0.83036,
                "trace_meta": [
                    {"example_id": 1, "score": 0.99, "failed": False},
                    {"example_id": 5, "score": 0.0, "failed": False},
                    {"example_id": 24, "score": 0.5, "failed": True},
                ],
            }),
        }],
    )
    at.append_architect_turn(
        31, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "y", "is_error": True, "content": "boom " * 100}],
    )
    at.append_architect_turn(
        31, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "z", "is_error": False,
                 "content": json.dumps({"ok": True, "summary": "Row 28 failed because…"})}],
    )
    at.append_architect_turn(31, role="user", blocks=[{"kind": "text", "text": "You are about to end iteration 31 WITHOUT closing."}])
    at.append_architect_turn(31, role="result", blocks=[], meta={"captured": True, "num_messages": 12})
    at.append_architect_turn(31, role="result", blocks=[], meta={"captured": False, "note": "export unavailable"})
    with open(os.path.join(at.architect_trace_path(31), "index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]
    s = [r["summary"] for r in index]
    assert s[0] == "ce_run_eval(eval_sample=train); bash(command=git status) — Now let me run the baseline evaluation."
    assert s[1] == "run_eval train/parent score=0.8304 rows=3 zero/failed=2 eval_id=2aabfe228f21 sha=d1371c17 tree=729fae92"
    assert s[2].startswith("ERROR: boom boom") and len(s[2]) <= 200 and s[2].endswith("…")
    assert s[3] == "inspect_traces: Row 28 failed because…"
    assert s[4] == "prompt: You are about to end iteration 31 WITHOUT closing."
    assert s[5] == "session result num_messages=12"
    assert s[6] == "NOT CAPTURED: export unavailable"
    assert all(len(x) <= 200 for x in s)


def test_opencode_rewrite_keeps_export_timestamps(trace_dir):
    at.write_opencode_trace(
        32,
        [
            {"role": "user", "blocks": [{"kind": "text", "text": "go"}], "ts": "2026-08-30T16:00:00+00:00"},
            {"role": "assistant", "blocks": [{"kind": "text", "text": "ok"}]},
        ],
    )
    turns = at.read_architect_trace(32)
    assert turns[0]["ts"] == "2026-08-30T16:00:00+00:00"
    assert turns[1]["ts"] != "2026-08-30T16:00:00+00:00"  # stamped now, not copied


def test_append_session_prompt_turn_schema(trace_dir):
    at.append_session_prompt_turn(5, "go", resume=True)
    turns = at.read_architect_trace(5)
    assert turns[0]["role"] == "user"
    assert turns[0]["blocks"] == [{"kind": "text", "text": "go"}]
    assert turns[0]["meta"] == {"prompt_kind": "resume"}
    with open(os.path.join(at.architect_trace_path(5), "index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]
    assert index[0]["summary"].startswith("prompt: go")
    at.append_session_prompt_turn(5, "start", resume=False)
    assert at.read_architect_trace(5)[1]["meta"] == {"prompt_kind": "initial"}


def test_run_eval_digest_reads_summary_only_response(trace_dir):
    """The summary-only run_eval response (no row_ids / trace_meta;
    n_rows / zero_or_failed_count / infra_failed / score_reliable instead) must keep
    the digest line readable — plus the infra flags when they matter."""
    line = at.run_eval_result_line({
        "ok": True, "eval_id": "abc123", "eval_sample": "train",
        "suffix": "child", "sha": "d1371c17149d4395", "tree": "729fae923d5a4fd1",
        "primary_score": 0.75, "n_rows": 50, "zero_or_failed_count": 6,
        "infra_failed": 4, "score_reliable": False,
    })
    assert line == (
        "run_eval train/child score=0.7500 rows=50 zero/failed=6 "
        "eval_id=abc123 sha=d1371c17 tree=729fae92 infra_failed=4 UNRELIABLE"
    )
    # A reliable eval adds neither flag.
    clean = at.run_eval_result_line({
        "eval_sample": "train", "suffix": "parent", "primary_score": 0.9,
        "n_rows": 50, "zero_or_failed_count": 0, "infra_failed": 0,
        "score_reliable": True,
    })
    assert "infra_failed" not in clean and "UNRELIABLE" not in clean
    assert "rows=50 zero/failed=0" in clean
    # Older traces carry the legacy `failed_count` key; the digest line must
    # still render them.
    legacy = at.run_eval_result_line({
        "eval_sample": "train", "suffix": "parent", "primary_score": 0.9,
        "n_rows": 50, "failed_count": 3,
    })
    assert "rows=50 zero/failed=3" in legacy
