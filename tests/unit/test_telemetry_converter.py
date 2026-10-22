"""Copyright © 2026 440 Labs LLC

Unit tests for src.engine.telemetry.converter.
"""

from __future__ import annotations

from src.engine.telemetry.converter import (
    span_to_trace_entry,
    spans_to_trace_entries,
)


class _Status:
    def __init__(self, code_name: str = "UNSET", description: str | None = None):
        self.status_code = type("Code", (), {"name": code_name})()
        self.description = description


class _Ctx:
    def __init__(self, span_id: int):
        self.span_id = span_id


class FakeSpan:
    """Minimal stand-in for opentelemetry.sdk.trace.ReadableSpan."""

    def __init__(
        self,
        name: str,
        span_id: int,
        attributes: dict | None = None,
        parent_id: int | None = None,
        start_time: int = 0,
        status: _Status | None = None,
    ):
        self.name = name
        self.attributes = attributes or {}
        self.context = _Ctx(span_id)
        self.parent = _Ctx(parent_id) if parent_id is not None else None
        self.start_time = start_time
        self.status = status or _Status()


def test_span_to_trace_entry_extracts_ce_attributes():
    span = FakeSpan(
        name="function_name",
        span_id=1,
        attributes={
            "ce.span_kind": "tool",
            "ce.inputs.statement": "tomatoes are vegetables",
            "ce.inputs.user_id": "u42",
            "ce.output": "claim_1",
        },
    )
    entry = span_to_trace_entry(span)
    assert entry["signature_key"] == "function_name"
    assert entry["span_kind"] == "tool"
    assert entry["parent_key"] is None
    assert entry["inputs"] == {
        "statement": "tomatoes are vegetables",
        "user_id": "u42",
    }
    assert entry["output"] == "claim_1"
    assert entry["error"] is None
    assert "metadata" not in entry  # all attrs were ce.* — nothing left over


def test_span_to_trace_entry_keeps_unknown_attributes_in_metadata():
    span = FakeSpan(
        name="llm.call",
        span_id=2,
        attributes={
            "ce.span_kind": "llm",
            "ce.inputs.prompt": "hi",
            "ce.output": "hello",
            "model": "gpt-5",
            "tokens.input": 12,
        },
    )
    entry = span_to_trace_entry(span)
    assert entry["metadata"] == {"model": "gpt-5", "tokens.input": 12}


def test_span_to_trace_entry_promotes_framework_attributes():
    """A span without ce.inputs.* but with gen_ai.prompt should fall back."""
    span = FakeSpan(
        name="openai.chat",
        span_id=3,
        attributes={
            "gen_ai.prompt": "Translate this",
            "gen_ai.completion": "Traduis ceci",
        },
    )
    entry = span_to_trace_entry(span)
    assert entry["inputs"] == {"prompt": "Translate this"}
    assert entry["output"] == "Traduis ceci"
    # The presence of any gen_ai.* attribute should upgrade kind to llm.
    assert entry["span_kind"] == "llm"


def test_span_to_trace_entry_pulls_error_from_status():
    span = FakeSpan(
        name="failing",
        span_id=4,
        attributes={"ce.span_kind": "tool"},
        status=_Status("ERROR", description="boom"),
    )
    entry = span_to_trace_entry(span)
    assert entry["error"] == "boom"


def test_spans_to_trace_entries_resolves_parent_by_name():
    root = FakeSpan(name="pipeline", span_id=10, start_time=1)
    child = FakeSpan(
        name="function_name",
        span_id=11,
        parent_id=10,
        start_time=2,
        attributes={"ce.span_kind": "tool"},
    )
    grandchild = FakeSpan(
        name="llm.judge",
        span_id=12,
        parent_id=11,
        start_time=3,
        attributes={"ce.span_kind": "llm"},
    )

    entries = spans_to_trace_entries([grandchild, root, child])  # unsorted
    assert [e["signature_key"] for e in entries] == [
        "pipeline",
        "function_name",
        "llm.judge",
    ]
    assert entries[0]["parent_key"] is None
    assert entries[1]["parent_key"] == "pipeline"
    assert entries[2]["parent_key"] == "function_name"


def test_spans_to_trace_entries_handles_orphan_parent_gracefully():
    """If the parent span isn't in the batch, parent_key resolves to None."""
    orphan_child = FakeSpan(
        name="orphan",
        span_id=20,
        parent_id=999,  # not present
        start_time=1,
    )
    entries = spans_to_trace_entries([orphan_child])
    assert entries[0]["parent_key"] is None


def _long(n: int) -> str:
    return "x" * n


def test_dedupe_replaces_long_input_when_parent_has_same_value():
    big = _long(6_000)
    parent = FakeSpan(
        name="pipeline",
        span_id=30,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": big},
    )
    child = FakeSpan(
        name="wrapper",
        span_id=31,
        parent_id=30,
        start_time=2,
        attributes={"ce.span_kind": "tool", "ce.inputs.prompt": big},
    )
    entries = spans_to_trace_entries([parent, child])
    assert entries[0]["inputs"]["prompt"] == big  # parent untouched
    assert entries[1]["inputs"]["prompt"] == "<inherited from parent span 'pipeline'>"


def test_dedupe_skips_short_inputs():
    """Below DEDUPE_THRESHOLD, equal values are kept verbatim."""
    parent = FakeSpan(
        name="pipeline",
        span_id=40,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.text": "hi"},
    )
    child = FakeSpan(
        name="wrapper",
        span_id=41,
        parent_id=40,
        start_time=2,
        attributes={"ce.span_kind": "tool", "ce.inputs.text": "hi"},
    )
    entries = spans_to_trace_entries([parent, child])
    assert entries[1]["inputs"]["text"] == "hi"


def test_dedupe_keeps_value_when_parent_differs():
    parent_value = _long(6_000) + "PARENT"
    child_value = _long(6_000) + "CHILD"
    parent = FakeSpan(
        name="pipeline",
        span_id=50,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": parent_value},
    )
    child = FakeSpan(
        name="wrapper",
        span_id=51,
        parent_id=50,
        start_time=2,
        attributes={"ce.span_kind": "tool", "ce.inputs.prompt": child_value},
    )
    entries = spans_to_trace_entries([parent, child])
    assert entries[1]["inputs"]["prompt"] == child_value


def test_dedupe_resolves_parent_by_id_not_by_name():
    """Two siblings with the same name but different parents must be
    deduped against their own parent, not the other sibling's parent."""
    big_a = _long(6_000) + "A"
    big_b = _long(6_000) + "B"
    parent_a = FakeSpan(
        name="parent",
        span_id=60,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": big_a},
    )
    parent_b = FakeSpan(
        name="parent",  # same display name as parent_a
        span_id=61,
        start_time=2,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": big_b},
    )
    child_a = FakeSpan(
        name="child",
        span_id=62,
        parent_id=60,
        start_time=3,
        attributes={"ce.span_kind": "tool", "ce.inputs.prompt": big_a},
    )
    child_b = FakeSpan(
        name="child",
        span_id=63,
        parent_id=61,
        start_time=4,
        attributes={"ce.span_kind": "tool", "ce.inputs.prompt": big_b},
    )
    entries = spans_to_trace_entries([parent_a, parent_b, child_a, child_b])
    by_id = {e["signature_key"] + "_" + str(e.get("start_time")): e for e in entries}
    # child_a (start=3) inherits from parent_a (start=1, big_a)
    # child_b (start=4) inherits from parent_b (start=2, big_b)
    child_a_entry = entries[2]
    child_b_entry = entries[3]
    assert child_a_entry["inputs"]["prompt"].startswith("<inherited from parent")
    assert child_b_entry["inputs"]["prompt"].startswith("<inherited from parent")
    # And the parent entries themselves are untouched (different big values)
    assert entries[0]["inputs"]["prompt"] == big_a
    assert entries[1]["inputs"]["prompt"] == big_b


def test_dedupe_disabled_when_threshold_zero():
    big = _long(6_000)
    parent = FakeSpan(
        name="pipeline",
        span_id=70,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": big},
    )
    child = FakeSpan(
        name="wrapper",
        span_id=71,
        parent_id=70,
        start_time=2,
        attributes={"ce.span_kind": "tool", "ce.inputs.prompt": big},
    )
    entries = spans_to_trace_entries([parent, child], dedupe_threshold=0)
    assert entries[1]["inputs"]["prompt"] == big


# ---------------------------------------------------------------------------
# Re-parenting kept spans across filtered-out (dropped) ancestors.
# drop_links is the per-row {dropped_span_id: parent_span_id} skip-map.
# ---------------------------------------------------------------------------


def test_reparent_to_nearest_kept_ancestor_via_drop_links():
    """The exact DSPy bug: leaf's direct parent was dropped; it must re-parent
    to the surviving grandparent, not become a detached root."""
    grandparent = FakeSpan(name="JudgeModule.forward", span_id=1, start_time=1)
    # span_id=2 (e.g. ChatAdapter.__call__) was filtered out — only its link
    # survives in drop_links: 2's parent is the kept grandparent (1).
    leaf = FakeSpan(name="LM.__call__", span_id=3, parent_id=2, start_time=2)
    entries = spans_to_trace_entries(
        [grandparent, leaf], drop_links={2: 1}
    )
    by_name = {e["signature_key"]: e for e in entries}
    assert by_name["JudgeModule.forward"]["parent_key"] is None
    assert by_name["LM.__call__"]["parent_key"] == "JudgeModule.forward"


def test_reparent_through_chain_of_dropped_ancestors():
    """Multiple dropped wrappers between leaf and the surviving root."""
    root = FakeSpan(name="JudgeModule.forward", span_id=10, start_time=1)
    leaf = FakeSpan(name="LM.__call__", span_id=14, parent_id=13, start_time=2)
    # 13 -> 12 -> 11 -> 10 all dropped except the kept root (10).
    entries = spans_to_trace_entries(
        [root, leaf], drop_links={13: 12, 12: 11, 11: 10}
    )
    by_name = {e["signature_key"]: e for e in entries}
    assert by_name["LM.__call__"]["parent_key"] == "JudgeModule.forward"


def test_reparent_cycle_in_drop_links_is_safe():
    """A cycle among dropped nodes resolves to a detached root, no hang."""
    leaf = FakeSpan(name="LM.__call__", span_id=3, parent_id=2, start_time=1)
    # 2 and 4 are both dropped (not in the kept set) and point at each other.
    entries = spans_to_trace_entries([leaf], drop_links={2: 4, 4: 2})
    assert entries[0]["parent_key"] is None


def test_dropped_ancestor_with_no_surviving_parent_is_root():
    """If the dropped chain bottoms out at a node with no parent, the kept
    descendant is a true root."""
    leaf = FakeSpan(name="LM.__call__", span_id=3, parent_id=2, start_time=1)
    entries = spans_to_trace_entries([leaf], drop_links={2: None})
    assert entries[0]["parent_key"] is None


def test_drop_links_none_is_backward_compatible():
    """Without a skip-map, behavior is identical to the pre-change converter:
    direct parent if kept, else None."""
    parent = FakeSpan(name="pipeline", span_id=1, start_time=1)
    child = FakeSpan(name="inner", span_id=2, parent_id=1, start_time=2)
    orphan = FakeSpan(name="stray", span_id=3, parent_id=999, start_time=3)
    entries = spans_to_trace_entries([parent, child, orphan])
    by_name = {e["signature_key"]: e for e in entries}
    assert by_name["pipeline"]["parent_key"] is None
    assert by_name["inner"]["parent_key"] == "pipeline"
    assert by_name["stray"]["parent_key"] is None  # 999 not kept -> detached


def test_dedupe_fires_against_resolved_ancestor_across_dropped_wrapper():
    """Dedupe now reaches across a dropped wrapper: the leaf's big prompt,
    equal to the surviving ancestor's, collapses to the inherited marker —
    which would NOT happen pre-change (direct parent was dropped -> skipped)."""
    big = _long(6_000)
    grandparent = FakeSpan(
        name="JudgeModule.forward",
        span_id=1,
        start_time=1,
        attributes={"ce.span_kind": "chain", "ce.inputs.prompt": big},
    )
    leaf = FakeSpan(
        name="LM.__call__",
        span_id=3,
        parent_id=2,  # dropped wrapper
        start_time=2,
        attributes={"ce.span_kind": "llm", "ce.inputs.prompt": big},
    )
    entries = spans_to_trace_entries([grandparent, leaf], drop_links={2: 1})
    by_name = {e["signature_key"]: e for e in entries}
    assert by_name["LM.__call__"]["inputs"]["prompt"] == (
        "<inherited from parent span 'JudgeModule.forward'>"
    )


# ---------------------------------------------------------------------------
# Framework-attribute dedupe: promoted payloads leave `metadata`, OpenInference
# flattened messages are rebuilt, ce.* spans are untouched.
# ---------------------------------------------------------------------------


def test_openinference_flat_messages_rebuilt_and_deduped():
    span = FakeSpan(
        name="LM.__call__",
        span_id=80,
        attributes={
            "input.value": '{"messages": [...]}',
            "llm.input_messages.0.message.role": "system",
            "llm.input_messages.0.message.content": "be brief",
            "llm.input_messages.1.message.role": "user",
            "llm.input_messages.1.message.content": "hi",
            "llm.output_messages.0.message.role": "assistant",
            "llm.output_messages.0.message.content": "hello",
            "output.value": '[{"text": "hello"}]',
            "llm.model_name": "gpt-5.4-nano",
            "llm.token_count.total": 12,
            "openinference.span.kind": "LLM",
        },
    )
    entry = span_to_trace_entry(span)
    assert entry["inputs"] == {
        "messages": [
            {"role": "system", "content": "be brief"},
            {"role": "user", "content": "hi"},
        ]
    }
    assert entry["output"] == [{"role": "assistant", "content": "hello"}]
    assert entry["span_kind"] == "llm"
    # Everything promoted is gone from metadata; the rest is kept.
    assert entry["metadata"] == {
        "llm.model_name": "gpt-5.4-nano",
        "llm.token_count.total": 12,
        "openinference.span.kind": "LLM",
    }


def test_input_value_fallback_is_removed_from_metadata():
    span = FakeSpan(
        name="openai.chat",
        span_id=81,
        attributes={"gen_ai.prompt": "p", "gen_ai.completion": "c", "keep": 1},
    )
    entry = span_to_trace_entry(span)
    assert entry["inputs"] == {"prompt": "p"}
    assert entry["output"] == "c"
    assert entry["metadata"] == {"keep": 1}


def test_ce_spans_keep_framework_attributes_in_metadata():
    """A client-decorated span is byte-identical to before: nothing popped."""
    span = FakeSpan(
        name="decorated",
        span_id=82,
        attributes={
            "ce.inputs.q": "x",
            "ce.output": "y",
            "input.value": "raw",
            "llm.input_messages.0.message.content": "m",
        },
    )
    entry = span_to_trace_entry(span)
    assert entry["inputs"] == {"q": "x"}
    assert entry["output"] == "y"
    assert entry["metadata"] == {
        "input.value": "raw",
        "llm.input_messages.0.message.content": "m",
    }
