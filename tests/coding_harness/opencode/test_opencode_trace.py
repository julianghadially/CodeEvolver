"""Copyright © 2026 440 Labs LLC

Unit tests for `serialize_opencode_export` — the mapper from an
`opencode export <sessionID>` session document to architect-trace turn dicts.

The fixture shapes are pinned from a real export (opencode 1.17.x):
`{"info": {...session...}, "messages": [{"info": {...}, "parts": [...]}]}`
with `reasoning` / `text` / `tool` / `step-start` / `step-finish` parts, tool
`state` carrying status/input/output, and OpenAI-style encrypted reasoning
metadata.
"""

from __future__ import annotations

from src.engine.agent.harnesses.opencode.output import serialize_opencode_export


def _export_doc(messages):
    return {"info": {"id": "ses_abc123", "title": "iteration 5"}, "messages": messages}


def _assistant_msg(parts, *, model_id="gpt-5.4-mini"):
    return {
        "info": {"role": "assistant", "modelID": model_id, "id": "msg_1"},
        "parts": parts,
    }


def _tool_part(
    *,
    name="ce_run_eval",
    call_id="call_1",
    status="completed",
    input=None,
    output='{"ok": true}',
    include_output=True,
    error=None,
):
    """Build a tool part in the shape OpenCode actually exports.

    `error` matters: a real errored part carries `state` keys
    ['status', 'input', 'error', 'time'] and NO 'output' key. The previous
    fixture put the message in `output` even for `status="error"` — a shape
    OpenCode never emits — so `test_error_status_sets_is_error` passed while
    production silently dropped every tool error message.
    """
    state = {"status": status, "input": input or {"eval_sample": "subsample"}}
    if error is not None:
        state["error"] = error
        state["time"] = {"start": 0, "end": 1}
    elif include_output:
        state["output"] = output
    return {"type": "tool", "tool": name, "callID": call_id, "state": state}


def test_reasoning_and_text_parts_map_to_thinking_and_text_blocks():
    doc = _export_doc(
        [
            _assistant_msg(
                [
                    {
                        "type": "reasoning",
                        "text": "the retriever misses multi-hop rows",
                        "metadata": {
                            "openai": {"reasoningEncryptedContent": "gAAA..."}
                        },
                    },
                    {"type": "text", "text": "Running a baseline eval."},
                ]
            )
        ]
    )
    turns = serialize_opencode_export(doc)
    assert [t["role"] for t in turns] == ["assistant", "result"]
    blocks = turns[0]["blocks"]
    assert [b["kind"] for b in blocks] == ["thinking", "text"]
    assert blocks[0]["text"] == "the retriever misses multi-hop rows"
    assert blocks[0]["signature_present"] is True
    assert "reasoningEncryptedContent" not in str(blocks)
    assert blocks[1]["text"] == "Running a baseline eval."


def test_encrypted_only_reasoning_with_empty_text_is_skipped():
    # OpenAI reasoning often exports with empty text + encrypted content only —
    # no legible thinking exists, so no thinking block is emitted (best-effort).
    doc = _export_doc(
        [
            _assistant_msg(
                [
                    {
                        "type": "reasoning",
                        "text": "",
                        "metadata": {"openai": {"reasoningEncryptedContent": "gA..."}},
                    },
                    {"type": "text", "text": "done"},
                ]
            )
        ]
    )
    turns = serialize_opencode_export(doc)
    assert [b["kind"] for b in turns[0]["blocks"]] == ["text"]


def test_completed_tool_part_yields_tool_use_and_tool_result_pair():
    doc = _export_doc(
        [_assistant_msg([_tool_part(output='{"primary_score": 0.75}')])]
    )
    turns = serialize_opencode_export(doc)
    assert [t["role"] for t in turns] == ["assistant", "user", "result"]
    tool_use = turns[0]["blocks"][0]
    assert tool_use == {
        "kind": "tool_use",
        "name": "ce_run_eval",
        "id": "call_1",
        "input": {"eval_sample": "subsample"},
    }
    tool_result = turns[1]["blocks"][0]
    assert tool_result["kind"] == "tool_result"
    assert tool_result["tool_use_id"] == "call_1"
    assert tool_result["is_error"] is False
    assert tool_result["content"] == '{"primary_score": 0.75}'
    assert tool_result["truncated"] is False


_LIVE_ERROR = (
    '{"ok": false, "error": "CloseIterationError: The registered branch HEAD '
    "(ff9243a8) was never evaluated with run_eval. Run "
    "run_eval(eval_sample='train') on it, or git reset --hard to an "
    'evaluated sha, then call close_iteration again."}'
)


def test_error_status_captures_the_error_message():
    """The message must survive, not just the `is_error` flag.

    `meta_reflection` greps '"is_error": true' to find tool failures; an
    empty `content` gives it the marker and nothing to read. Uses a
    verbatim provider error.
    """
    doc = _export_doc([_assistant_msg([_tool_part(status="error", error=_LIVE_ERROR)])])
    turns = serialize_opencode_export(doc)
    result_block = turns[1]["blocks"][0]
    assert result_block["is_error"] is True
    assert "never evaluated with run_eval" in result_block["content"]


def test_dict_error_extracts_message():
    doc = _export_doc(
        [
            _assistant_msg(
                [_tool_part(status="error", error={"message": "boom", "data": {"x": 1}})]
            )
        ]
    )
    result_block = serialize_opencode_export(doc)[1]["blocks"][0]
    assert result_block["is_error"] is True
    assert "boom" in result_block["content"]


def test_output_wins_over_error_when_both_present():
    part = _tool_part(status="error", error="the error")
    part["state"]["output"] = "the output"
    result_block = serialize_opencode_export(_export_doc([_assistant_msg([part])]))[1][
        "blocks"
    ][0]
    assert result_block["content"] == "the output"


def test_blank_output_falls_back_to_error():
    part = _tool_part(status="error", error="the error")
    part["state"]["output"] = "   "
    result_block = serialize_opencode_export(_export_doc([_assistant_msg([part])]))[1][
        "blocks"
    ][0]
    assert result_block["content"] == "the error"


def test_error_without_status_still_emits_an_error_result():
    part = {
        "type": "tool",
        "tool": "ce_run_eval",
        "callID": "call_1",
        "state": {"input": {}, "error": "no status field"},
    }
    result_block = serialize_opencode_export(_export_doc([_assistant_msg([part])]))[1][
        "blocks"
    ][0]
    assert result_block["is_error"] is True
    assert "no status field" in result_block["content"]


def test_patch_part_captures_touched_files():
    doc = _export_doc(
        [
            _assistant_msg(
                [{"type": "patch", "hash": "abc", "files": ["src/judge.py"]}]
            )
        ]
    )
    blocks = serialize_opencode_export(doc)[0]["blocks"]
    assert any(
        b["kind"] == "text" and "src/judge.py" in b["text"] for b in blocks
    )


def test_pending_tool_part_yields_tool_use_without_tool_result():
    doc = _export_doc(
        [
            _assistant_msg(
                [_tool_part(status="pending", include_output=False)]
            )
        ]
    )
    turns = serialize_opencode_export(doc)
    assert [t["role"] for t in turns] == ["assistant", "result"]
    assert turns[0]["blocks"][0]["kind"] == "tool_use"


def test_terminal_tool_without_output_gets_empty_content():
    doc = _export_doc(
        [_assistant_msg([_tool_part(status="completed", include_output=False)])]
    )
    turns = serialize_opencode_export(doc)
    assert turns[1]["blocks"][0]["content"] == ""


def test_large_tool_output_is_truncated():
    doc = _export_doc([_assistant_msg([_tool_part(output="x" * 30_000)])])
    turns = serialize_opencode_export(doc)
    result_block = turns[1]["blocks"][0]
    assert result_block["truncated"] is True
    assert result_block["content"].endswith("[truncated]")
    assert len(result_block["content"]) < 30_000


def test_step_parts_are_skipped_and_tokens_feed_result_usage():
    doc = _export_doc(
        [
            _assistant_msg(
                [
                    {"type": "step-start", "id": "prt_1"},
                    {"type": "text", "text": "done"},
                    {
                        "type": "step-finish",
                        "tokens": {
                            "input": 500,
                            "output": 100,
                            "reasoning": 25,
                            "cache": {"write": 10, "read": 300},
                        },
                    },
                ]
            )
        ]
    )
    turns = serialize_opencode_export(doc)
    assert [b["kind"] for b in turns[0]["blocks"]] == ["text"]
    usage = turns[-1]["usage"]
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 125  # output + reasoning
    assert usage["cache_creation_input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 300


def test_result_line_meta_carries_session_model_and_count():
    doc = _export_doc(
        [
            {"info": {"role": "user"}, "parts": [{"type": "text", "text": "prompt"}]},
            _assistant_msg([{"type": "text", "text": "done"}]),
        ]
    )
    turns = serialize_opencode_export(doc)
    meta = turns[-1]["meta"]
    assert meta["harness"] == "opencode"
    assert meta["captured"] is True
    assert meta["session_id"] == "ses_abc123"  # from the export's session info
    assert meta["model"] == "gpt-5.4-mini"
    assert meta["num_messages"] == 2


def test_user_prompt_messages_become_user_text_lines():
    # The initial prompt and watchdog re-prompts are user messages — capturing
    # them shows the meta-reflection agent when/why the loop re-prompted.
    doc = _export_doc(
        [
            {
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "close_iteration is required"}],
            },
            _assistant_msg([{"type": "text", "text": "closing now"}]),
        ]
    )
    turns = serialize_opencode_export(doc)
    assert turns[0]["role"] == "user"
    assert turns[0]["blocks"] == [
        {"kind": "text", "text": "close_iteration is required"}
    ]


def test_unknown_part_type_with_text_degrades_to_text_block():
    doc = _export_doc(
        [_assistant_msg([{"type": "novel-part", "text": "ZEPHYR_QUASAR_42"}])]
    )
    turns = serialize_opencode_export(doc)
    assert turns[0]["blocks"] == [{"kind": "text", "text": "ZEPHYR_QUASAR_42"}]


def test_empty_or_malformed_documents_return_empty():
    assert serialize_opencode_export(None) == []
    assert serialize_opencode_export("not a dict") == []
    assert serialize_opencode_export({}) == []
    assert serialize_opencode_export({"info": {}, "messages": []}) == []
    # Messages with nothing serializable -> [] (caller falls back to the stub).
    assert (
        serialize_opencode_export(
            _export_doc([{"info": {"role": "assistant"}, "parts": []}])
        )
        == []
    )


def test_explicit_session_and_model_kwargs_win_over_document():
    doc = _export_doc([_assistant_msg([{"type": "text", "text": "done"}])])
    turns = serialize_opencode_export(
        doc, model="openai/gpt-5.3-codex-spark", session_id="ses_override"
    )
    meta = turns[-1]["meta"]
    assert meta["session_id"] == "ses_override"
    assert meta["model"] == "openai/gpt-5.3-codex-spark"


def test_merge_result_turns_collapses_multi_session_results():
    """A watchdog resume must not strand a result line mid-file.

    `_capture_architect_trace` concatenates one serialization per session id,
    and each ends with its own `role="result"` turn carrying only that
    session's usage.
    """
    from src.engine.agent.harnesses.opencode.output import _merge_result_turns

    turns = [
        {"role": "assistant", "blocks": [{"kind": "text", "text": "first pass"}]},
        {
            "role": "result",
            "blocks": [],
            "meta": {"session_id": "ses_a", "num_messages": 2},
            "usage": {"input_tokens": 10, "output_tokens": 5, "model": "m"},
        },
        {"role": "assistant", "blocks": [{"kind": "text", "text": "resumed"}]},
        {
            "role": "result",
            "blocks": [],
            "meta": {"session_id": "ses_b", "num_messages": 3},
            "usage": {"input_tokens": 7, "output_tokens": 1, "model": "m"},
        },
    ]

    merged = _merge_result_turns(turns, ["ses_a", "ses_b"])

    assert [t["role"] for t in merged] == ["assistant", "assistant", "result"]
    result = merged[-1]
    assert result["usage"]["input_tokens"] == 17
    assert result["usage"]["output_tokens"] == 6
    assert result["usage"]["model"] == "m"
    assert result["meta"]["num_messages"] == 5
    assert result["meta"]["session_ids"] == ["ses_a", "ses_b"]


def test_merge_result_turns_is_a_noop_for_one_session():
    from src.engine.agent.harnesses.opencode.output import _merge_result_turns

    turns = [
        {"role": "assistant", "blocks": []},
        {"role": "result", "blocks": [], "usage": {"input_tokens": 3}},
    ]
    assert _merge_result_turns(turns, ["ses_a"]) == turns


def test_export_message_time_becomes_turn_ts():
    """`info.time.created` (epoch ms) → per-turn ISO `ts`; a post-session
    export otherwise stamps every turn with one export time."""
    from src.engine.agent.harnesses.opencode.output import serialize_opencode_export

    doc = {
        "info": {"id": "ses_1"},
        "messages": [
            {"info": {"role": "user", "time": {"created": 1788132000000}},
             "parts": [{"type": "text", "text": "go"}]},
            {"info": {"role": "assistant", "modelID": "m", "time": {"created": 1788132060000}},
             "parts": [
                 {"type": "text", "text": "ok"},
                 {"type": "tool", "tool": "bash", "callID": "c1",
                  "state": {"status": "completed", "input": {"command": "ls"}, "output": "a"}},
             ]},
            {"info": {"role": "assistant"}, "parts": [{"type": "text", "text": "no time"}]},
        ],
    }
    turns = serialize_opencode_export(doc)
    assert turns[0]["ts"] == "2026-08-30T23:20:00+00:00"
    assert turns[1]["ts"] == "2026-08-30T23:21:00+00:00"      # assistant blocks
    assert turns[2]["ts"] == "2026-08-30T23:21:00+00:00"      # its tool results
    assert "ts" not in turns[3]                               # no time → stamped at write
    assert turns[-1]["role"] == "result"
