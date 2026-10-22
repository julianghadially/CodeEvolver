"""Copyright © 2026 440 Labs LLC

Unit tests for the token-usage accounting module
(`src/engine/agent/token_usage.py`): parsing the two usage shapes,
arithmetic, model-id normalization, and the per-model ledger.
"""

from __future__ import annotations

from src.engine.agent.token_usage import (
    ModelTokenLedger,
    TokenUsage,
    normalize_model_id,
)


def test_from_sdk_usage_maps_keys_and_tolerates_missing():
    u = TokenUsage.from_sdk_usage(
        {
            "input_tokens": 100,
            "output_tokens": 20,
            "cache_creation_input_tokens": 5,
            "cache_read_input_tokens": 7,
        }
    )
    assert u.input_tokens == 100
    assert u.output_tokens == 20
    assert u.cache_creation_input_tokens == 5
    assert u.cache_read_input_tokens == 7
    assert u.total == 132

    # None / empty / partial mappings degrade to zeros, never raise.
    assert TokenUsage.from_sdk_usage(None) == TokenUsage.zero()
    assert TokenUsage.from_sdk_usage({}) == TokenUsage.zero()
    partial = TokenUsage.from_sdk_usage({"input_tokens": 3})
    assert partial.input_tokens == 3
    assert partial.output_tokens == 0


def test_from_litellm_usage_maps_openai_shaped_names():
    # dict form with nested cached-token details (newer litellm)
    u = TokenUsage.from_litellm_usage(
        {
            "prompt_tokens": 200,
            "completion_tokens": 40,
            "prompt_tokens_details": {"cached_tokens": 50},
        }
    )
    assert u.input_tokens == 200
    assert u.output_tokens == 40
    assert u.cache_read_input_tokens == 50

    # object form with the older flat cache field
    class _Usage:
        prompt_tokens = 10
        completion_tokens = 2
        cache_read_input_tokens = 4

    o = TokenUsage.from_litellm_usage(_Usage())
    assert o.input_tokens == 10
    assert o.output_tokens == 2
    assert o.cache_read_input_tokens == 4

    assert TokenUsage.from_litellm_usage(None) == TokenUsage.zero()


def test_add_sub_and_dict_roundtrip():
    a = TokenUsage(input_tokens=10, output_tokens=4, cache_read_input_tokens=2)
    b = TokenUsage(input_tokens=5, output_tokens=1, cache_creation_input_tokens=3)
    s = a + b
    assert s.input_tokens == 15
    assert s.output_tokens == 5
    assert s.cache_creation_input_tokens == 3
    assert s.cache_read_input_tokens == 2

    # subtraction is field-wise and clamped at zero
    d = a - b
    assert d.input_tokens == 5
    assert d.output_tokens == 3
    assert d.cache_creation_input_tokens == 0  # 0 - 3 clamps

    assert TokenUsage.from_dict(a.to_dict()) == a
    assert TokenUsage.from_dict(None) == TokenUsage.zero()


def test_normalize_model_id_collapses_anthropic_prefix():
    assert normalize_model_id("anthropic/claude-opus-4-8") == "claude-opus-4-8"
    assert normalize_model_id("claude-opus-4-8") == "claude-opus-4-8"
    assert normalize_model_id(None) == "unknown"
    assert normalize_model_id("") == "unknown"


def test_ledger_aggregates_by_model_and_collapses_prefix():
    led = ModelTokenLedger()
    led.add("anthropic/claude-opus-4-8", TokenUsage(input_tokens=10, output_tokens=2))
    led.add("claude-opus-4-8", TokenUsage(input_tokens=5, output_tokens=1))
    led.add("gpt-5", TokenUsage(input_tokens=7))

    d = led.to_dict()
    # the two opus forms aggregate under one key
    assert d["claude-opus-4-8"]["input_tokens"] == 15
    assert d["claude-opus-4-8"]["output_tokens"] == 3
    assert d["gpt-5"]["input_tokens"] == 7
    assert d["__total__"]["input_tokens"] == 22
    assert d["__total__"]["total"] == 25

    gt = led.grand_total()
    assert gt.input_tokens == 22
    assert gt.output_tokens == 3


def test_ledger_merge_accepts_ledger_and_mapping():
    led = ModelTokenLedger()
    led.add("claude-opus-4-8", TokenUsage(input_tokens=10))

    other = ModelTokenLedger()
    other.add("claude-opus-4-8", TokenUsage(input_tokens=5))
    led.merge(other)

    # mapping shape (e.g. IterationContext.token_usage_by_model)
    led.merge({"claude-opus-4-8": {"input_tokens": 2, "output_tokens": 1, "total": 3}})

    assert led.grand_total().input_tokens == 17
    assert led.grand_total().output_tokens == 1


def test_ledger_delta_since_snapshot():
    led = ModelTokenLedger()
    led.add("claude-opus-4-8", TokenUsage(input_tokens=10))
    before = led.snapshot()
    led.add("claude-opus-4-8", TokenUsage(input_tokens=4, output_tokens=2))
    led.add("gpt-5", TokenUsage(input_tokens=1))

    delta = led.delta_since(before)
    assert delta["claude-opus-4-8"].input_tokens == 4
    assert delta["claude-opus-4-8"].output_tokens == 2
    assert delta["gpt-5"].input_tokens == 1
    # models with no change since the snapshot are dropped
    assert before == {"claude-opus-4-8": TokenUsage(input_tokens=10)}
