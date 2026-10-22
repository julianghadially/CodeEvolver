from __future__ import annotations

import json
import logging


def test_parse_opencode_json_output_extracts_text_session_and_tool_calls():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    stdout = "\n".join(
        [
            json.dumps({"type": "session.updated", "sessionID": "session-1"}),
            json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}}),
            json.dumps({"type": "tool.call", "name": "write", "input": {"file": "x.py"}}),
        ]
    )

    output, session_id, metadata = parse_opencode_output(stdout)

    assert output == "done"
    assert session_id == "session-1"
    assert metadata["json_events"] == 3
    assert metadata["tool_calls"] == [{"name": "write", "input": {"file": "x.py"}}]


def test_parse_opencode_tool_use_events_from_remote_mcp_shape():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    stdout = "\n".join(
        [
            json.dumps(
                {
                    "type": "tool_use",
                    "sessionID": "session-1",
                    "part": {
                        "type": "tool",
                        "tool": "ce_run_eval",
                        "callID": "call-1",
                        "state": {
                            "status": "completed",
                            "input": {"eval_sample": "subsample"},
                            "output": '{"ok": true, "primary_score": 0.9}',
                        },
                    },
                }
            ),
            json.dumps(
                {
                    "type": "text",
                    "part": {
                        "type": "text",
                        "text": "scores improved",
                    },
                }
            ),
        ]
    )

    output, session_id, metadata = parse_opencode_output(stdout)

    assert output == "scores improved"
    assert session_id == "session-1"
    assert metadata["tool_calls"] == [
        {
            "name": "ce_run_eval",
            "input": {"eval_sample": "subsample"},
            "call_id": "call-1",
            "status": "completed",
            "output": '{"ok": true, "primary_score": 0.9}',
        }
    ]


def test_parse_opencode_output_returns_plain_stdout_for_non_json():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    output, session_id, metadata = parse_opencode_output("plain output")

    assert output == "plain output"
    assert session_id is None
    assert metadata == {"json_events": 0}


def test_parse_opencode_output_sums_step_finish_tokens_and_model():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    stdout = "\n".join(
        [
            json.dumps(
                {"info": {"role": "assistant", "id": "m1", "modelID": "gpt-5.4-mini"}}
            ),
            json.dumps(
                {
                    "type": "step-finish",
                    "tokens": {
                        "input": 200,
                        "output": 50,
                        "reasoning": 10,
                        "cache": {"write": 5, "read": 100},
                    },
                }
            ),
            json.dumps(
                {
                    "part": {
                        "type": "step-finish",
                        "tokens": {
                            "input": 100,
                            "output": 20,
                            "reasoning": 0,
                            "cache": {"write": 0, "read": 40},
                        },
                    }
                }
            ),
        ]
    )

    _, _, metadata = parse_opencode_output(stdout)

    usage = metadata["usage"]
    assert usage["input_tokens"] == 300
    # output folds reasoning in: (50+10) + (20+0)
    assert usage["output_tokens"] == 80
    assert usage["cache_creation_input_tokens"] == 5
    assert usage["cache_read_input_tokens"] == 140
    assert usage["total"] == 300 + 80 + 5 + 140
    assert usage["model"] == "gpt-5.4-mini"
    assert metadata["model"] == "gpt-5.4-mini"


def test_parse_opencode_output_falls_back_to_info_tokens_deduped_by_id():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    # No step-finish parts (OpenCode #26855): fall back to per-message info
    # tokens, keeping the last (cumulative) value per message id.
    stdout = "\n".join(
        [
            json.dumps(
                {
                    "info": {
                        "role": "assistant",
                        "id": "m1",
                        "modelID": "gpt-5.4-mini",
                        "tokens": {
                            "input": 100,
                            "output": 10,
                            "reasoning": 0,
                            "cache": {"write": 0, "read": 0},
                        },
                    }
                }
            ),
            json.dumps(
                {
                    "info": {
                        "role": "assistant",
                        "id": "m1",
                        "modelID": "gpt-5.4-mini",
                        "tokens": {
                            "input": 100,
                            "output": 40,
                            "reasoning": 0,
                            "cache": {"write": 0, "read": 0},
                        },
                    }
                }
            ),
        ]
    )

    _, _, metadata = parse_opencode_output(stdout)

    usage = metadata["usage"]
    # Last-write-wins for message m1: input=100, output=40 (not summed to 80).
    assert usage["input_tokens"] == 100
    assert usage["output_tokens"] == 40
    assert usage["total"] == 140
    assert usage["model"] == "gpt-5.4-mini"


def test_parse_opencode_output_omits_usage_when_no_tokens_present():
    from src.engine.agent.harnesses.opencode.output import parse_opencode_output

    stdout = "\n".join(
        [
            json.dumps({"type": "session.updated", "sessionID": "session-1"}),
            json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}}),
        ]
    )

    _, _, metadata = parse_opencode_output(stdout)

    assert "usage" not in metadata
    assert "model" not in metadata


class _ListLogger:
    """Minimal logger stand-in capturing formatted messages."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def info(self, fmt: str, *args) -> None:
        self.lines.append(fmt % args)


def test_log_opencode_events_orders_tools_and_text_and_dedupes():
    from src.engine.agent.harnesses.opencode.output import log_opencode_events

    stdout = "\n".join(
        [
            json.dumps({"part": {"type": "text", "text": "run baseline"}}),
            json.dumps(
                {
                    "part": {
                        "type": "tool",
                        "tool": "run_eval",
                        "callID": "c1",
                        "state": {"input": {"eval_sample": "subsample"}, "status": "completed"},
                    }
                }
            ),
            # duplicate callID (status transition) — should not log twice
            json.dumps(
                {
                    "part": {
                        "type": "tool",
                        "tool": "run_eval",
                        "callID": "c1",
                        "state": {"input": {"eval_sample": "subsample"}, "status": "running"},
                    }
                }
            ),
            json.dumps(
                {
                    "part": {
                        "type": "tool",
                        "tool": "edit",
                        "callID": "c2",
                        "state": {"input": {"filePath": "arc.py"}, "status": "completed"},
                    }
                }
            ),
            json.dumps(
                {
                    "part": {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "c3",
                        "state": {"input": {"command": "git commit"}, "status": "completed"},
                    }
                }
            ),
            # duplicate text — should not log twice
            json.dumps({"part": {"type": "text", "text": "run baseline"}}),
        ]
    )

    log = _ListLogger()
    emitted = log_opencode_events(7, stdout, log)

    # 1 text + 3 distinct tools (run_eval logged once, edit, bash)
    assert emitted == 4
    assert log.lines == [
        "iteration 7: architect text | run baseline",
        "iteration 7: architect tool_use | run_eval({'eval_sample': 'subsample'})",
        "iteration 7: architect tool_use | edit({'filePath': 'arc.py'})",
        "iteration 7: architect tool_use | bash({'command': 'git commit'})",
    ]
    # The ordered transcript shows editing happened between/after the eval —
    # exactly the visibility the detailed agent log exists to provide.
    eval_idx = next(i for i, l in enumerate(log.lines) if "run_eval(" in l)
    edit_idx = next(i for i, l in enumerate(log.lines) if "edit(" in l)
    assert eval_idx < edit_idx


def test_log_opencode_events_truncates_long_input_and_text():
    from src.engine.agent.harnesses.opencode.output import log_opencode_events

    long_cmd = "x" * 1000
    stdout = "\n".join(
        [
            json.dumps({"part": {"type": "text", "text": "y" * 1000}}),
            json.dumps(
                {
                    "part": {
                        "type": "tool",
                        "tool": "bash",
                        "callID": "c1",
                        "state": {"input": {"command": long_cmd}, "status": "completed"},
                    }
                }
            ),
        ]
    )

    log = _ListLogger()
    log_opencode_events(1, stdout, log)

    # Both lines were truncated (text ends with the ellipsis; the tool_use
    # ellipsis sits inside the `name(...)` wrapper, so it ends with "…)").
    assert log.lines[0].endswith("…")
    assert log.lines[1].endswith("…)")
    # text truncated to ~500 chars + ellipsis, not the full 1000
    assert len(log.lines[0]) < 600


def test_log_opencode_events_handles_non_json_and_empty():
    from src.engine.agent.harnesses.opencode.output import log_opencode_events

    log = _ListLogger()
    assert log_opencode_events(1, "not json\n\n[]", log) == 0
    assert log.lines == []


def test_get_detailed_agent_logger_is_file_only_and_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("CODEEVOLVER_JOB_ID", "job_xyz")
    # Reset any handler installed by an earlier test in this process.
    logging.getLogger("codeevolver.detailed_agent").handlers.clear()

    from src.engine.paths import detailed_agent_log_path
    from src.engine.services.logger import get_detailed_agent_logger

    expected = tmp_path / "job_xyz_detailed_agent_log.log"
    assert detailed_agent_log_path() == str(expected)

    lg = get_detailed_agent_logger()
    lg2 = get_detailed_agent_logger()
    assert lg is lg2
    assert len(lg.handlers) == 1  # idempotent — no duplicate handlers
    assert lg.propagate is False

    lg.info("iteration 3: architect tool_use | edit({'f': 'a.py'})")
    for h in lg.handlers:
        h.flush()
    assert "architect tool_use | edit" in expected.read_text()
