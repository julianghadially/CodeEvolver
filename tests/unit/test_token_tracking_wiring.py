"""Copyright © 2026 440 Labs LLC

Wiring tests for token tracking across process boundaries:
- the subprocess-service stdout marker parser (`AGENT_USAGE:`/`INSPECT_USAGE:`)
- the litellm constraint-checker callable's per-model usage accumulation
"""

from __future__ import annotations

import sys
import types

from src.engine.services.subprocess_service import _parse_usage_marker


def test_parse_usage_marker_extracts_json_payload():
    stdout = (
        "[AGENT] Starting code mutation...\n"
        "[AGENT] Completed in 7 turns\n"
        'AGENT_USAGE:{"input_tokens":120,"output_tokens":30,'
        '"cache_creation_input_tokens":0,"cache_read_input_tokens":5,'
        '"total":155,"model":"claude-opus-4-8"}\n'
        "AGENT_SUCCESS\n"
    )
    usage = _parse_usage_marker(stdout, "AGENT_USAGE:")
    assert usage["input_tokens"] == 120
    assert usage["output_tokens"] == 30
    assert usage["model"] == "claude-opus-4-8"


def test_parse_usage_marker_absent_returns_empty():
    assert _parse_usage_marker("no markers here\nAGENT_SUCCESS\n", "AGENT_USAGE:") == {}
    assert _parse_usage_marker("", "INSPECT_USAGE:") == {}


def test_parse_usage_marker_inspect_prefix_and_bad_json():
    stdout = (
        "INSPECT_USAGE:not-json\n"  # malformed, skipped
        'INSPECT_USAGE:{"input_tokens":9,"model":"claude-opus-4-8"}\n'
        "INSPECT_NO_OUTPUT\n"
    )
    usage = _parse_usage_marker(stdout, "INSPECT_USAGE:")
    assert usage == {"input_tokens": 9, "model": "claude-opus-4-8"}


def test_reflection_lm_callable_accumulates_token_usage(monkeypatch):
    """The constraint-checker callable captures litellm usage per model while
    keeping its `Callable[[str], str]` contract intact."""
    from src.engine.optimizer.v3 import lm_config

    # Stub the `litellm` module imported inside the callable.
    fake_litellm = types.ModuleType("litellm")

    def _completion(*, model, messages):
        return {
            "model": model,
            "choices": [{"message": {"content": "pass"}}],
            "usage": {
                "prompt_tokens": 50,
                "completion_tokens": 8,
                "prompt_tokens_details": {"cached_tokens": 10},
            },
        }

    fake_litellm.completion = _completion
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    call = lm_config.build_reflection_lm_callable("anthropic/claude-opus-4-8")
    assert call("first prompt") == "pass"
    assert call("second prompt") == "pass"

    totals = call.token_ledger.to_dict()  # type: ignore[attr-defined]
    # litellm reported the model verbatim ("anthropic/..."); the ledger
    # normalizes it to the bare id so it lines up with the SDK agents.
    assert totals["claude-opus-4-8"]["input_tokens"] == 100  # 50 * 2
    assert totals["claude-opus-4-8"]["output_tokens"] == 16  # 8 * 2
    assert totals["claude-opus-4-8"]["cache_read_input_tokens"] == 20  # 10 * 2
