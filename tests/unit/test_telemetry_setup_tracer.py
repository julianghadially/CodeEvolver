"""Copyright © 2026 440 Labs LLC

Tests for setup_tracer / drain_spans / shutdown_tracer.

Validates the critical "don't replace the client provider" behavior: if a real
TracerProvider is already installed when setup_tracer() is called, we attach
our SimpleSpanProcessor to it; we do NOT overwrite the global.

Note on OTel quirks: the OTel global TracerProvider is set-once per process.
Tests that share a pytest process therefore share whatever TracerProvider was
installed first. setup_tracer's probe-and-attach pattern handles this
gracefully — every test below either creates the provider on first run or
attaches to it on subsequent runs, and our exporter still collects spans.
"""

from __future__ import annotations

import pytest

from src.engine.telemetry import (
    OTEL_AVAILABLE,
    drain_spans,
    setup_tracer,
    shutdown_tracer,
)

pytestmark = pytest.mark.skipif(not OTEL_AVAILABLE, reason="opentelemetry not installed")

if OTEL_AVAILABLE:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )


@pytest.fixture(autouse=True)
def _reset_telemetry_state():
    """Detach our processor between tests so each starts from a clean slate."""
    yield
    shutdown_tracer()


def _emit_one_span(name: str = "test.span") -> None:
    tracer = trace.get_tracer("test")
    with tracer.start_as_current_span(name):
        pass


def test_setup_lets_our_exporter_collect_spans():
    setup_tracer()
    _emit_one_span("only_span")
    spans = drain_spans()
    assert [s.name for s in spans] == ["only_span"]


def test_setup_attaches_to_existing_provider_without_replacing_it():
    """If a real TracerProvider is already global, attach to it; don't replace."""
    provider_before = trace.get_tracer_provider()

    # If the global is still a Proxy on first run, install one ourselves so
    # we have a known real provider to attach to.
    if not hasattr(provider_before, "add_span_processor"):
        first_provider = TracerProvider()
        trace.set_tracer_provider(first_provider)
        provider_before = trace.get_tracer_provider()

    sentinel_exporter = InMemorySpanExporter()
    provider_before.add_span_processor(SimpleSpanProcessor(sentinel_exporter))

    setup_tracer()

    # Critical: the global provider object is unchanged.
    assert trace.get_tracer_provider() is provider_before

    _emit_one_span("shared_span")

    ours = drain_spans()
    theirs = sentinel_exporter.get_finished_spans()
    assert [s.name for s in ours] == ["shared_span"]
    assert [s.name for s in theirs] == ["shared_span"]


def test_setup_is_idempotent_across_calls():
    setup_tracer()
    _emit_one_span("first")
    drain_spans()  # consume
    setup_tracer()  # second call should tear down + reinstall cleanly
    _emit_one_span("second")
    spans = drain_spans()
    assert [s.name for s in spans] == ["second"]


def test_drain_clears_buffer():
    setup_tracer()
    _emit_one_span("a")
    first = drain_spans()
    second = drain_spans()
    assert [s.name for s in first] == ["a"]
    assert second == []


def test_shutdown_stops_forwarding_spans_to_our_exporter():
    setup_tracer()
    _emit_one_span("alive")
    assert len(drain_spans()) == 1

    shutdown_tracer()
    _emit_one_span("after_shutdown")
    # drain returns [] because our exporter ref was released, not because the
    # provider died.
    assert drain_spans() == []
