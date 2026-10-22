"""Copyright © 2026 440 Labs LLC

Validates the per-row drop-link skip-map wiring end to end with real OTel spans:
`FilteredSpanProcessor.on_drop` -> `_record_drop_link` -> `_row_drop_links`,
then `spans_to_trace_entries(drop_links=...)` re-parents a kept span whose
direct parent was filtered out to its nearest surviving ancestor.
"""

from __future__ import annotations

import pytest

from src.engine.telemetry import (
    OTEL_AVAILABLE,
    collect_row_spans,
    setup_tracer,
    shutdown_tracer,
)
from src.engine.telemetry.converter import spans_to_trace_entries
from src.engine.telemetry.filters import make_keep_predicate

pytestmark = pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")

if OTEL_AVAILABLE:
    from opentelemetry import trace


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    yield
    shutdown_tracer()


def _drop_named(name: str):
    """Drop predicate matching one span name (mirrors dspy.should_drop shape)."""
    return lambda span: span.name == name


def test_dropped_span_records_link_and_leaf_reparents():
    # Drop the middle span ("DROPME"); keep root + leaf.
    setup_tracer(span_filter=make_keep_predicate([_drop_named("DROPME")]))
    tracer = trace.get_tracer("test")

    ids: dict[str, int] = {}
    with collect_row_spans() as (buf, drop_links):
        with tracer.start_as_current_span("root") as root:
            ids["root"] = root.get_span_context().span_id
            with tracer.start_as_current_span("DROPME") as mid:
                ids["mid"] = mid.get_span_context().span_id
                with tracer.start_as_current_span("leaf") as leaf:
                    ids["leaf"] = leaf.get_span_context().span_id
        row_spans = list(buf)
        row_links = dict(drop_links)

    # Kept buffer holds root + leaf only; DROPME was filtered out.
    assert sorted(s.name for s in row_spans) == ["leaf", "root"]
    # The dropped span's (id -> parent_id) link was recorded for this row.
    assert row_links == {ids["mid"]: ids["root"]}

    # Conversion re-parents leaf across the dropped middle to root.
    entries = spans_to_trace_entries(row_spans, drop_links=row_links)
    by_name = {e["signature_key"]: e for e in entries}
    assert by_name["root"]["parent_key"] is None
    assert by_name["leaf"]["parent_key"] == "root"


def test_no_drops_yields_empty_skip_map():
    setup_tracer(span_filter=make_keep_predicate([_drop_named("NOMATCH")]))
    tracer = trace.get_tracer("test")
    with collect_row_spans() as (buf, drop_links):
        with tracer.start_as_current_span("root"):
            with tracer.start_as_current_span("child"):
                pass
        assert dict(drop_links) == {}
        assert sorted(s.name for s in buf) == ["child", "root"]
