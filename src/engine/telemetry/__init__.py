"""Copyright © 2026 440 Labs LLC

OpenTelemetry trace collection for the CodeEvolver evaluator.

This module exposes a small lifecycle API used by the mounted evaluator to
collect spans emitted by client code (and any AI framework that emits OTel
spans) into a process-local in-memory exporter, so per-row traces can be
written to /traces/iteration_{N}_{suffix}/ (one row.json + span files per example).

Design notes
------------
1. We never replace an existing TracerProvider. Many client codebases (and
   AI frameworks) call `trace.set_tracer_provider(...)` at import time. If
   we set our own provider first, the client's later set-call wins and we
   collect zero spans. If we set our own provider last, we silently break
   the client's existing telemetry shipping (LangFuse, Honeycomb, etc.).
   `setup_tracer()` therefore probes the current global; if it's a real
   SDK `TracerProvider`, we attach our span processor to it instead
   of replacing it. Both processors then receive every span independently.

2. `setup_tracer()` must be called AFTER the client module is imported and
   any client classes are instantiated, so the client gets the chance to
   wire its own provider before we probe.

3. `OTEL_AVAILABLE` is False when `opentelemetry` isn't importable (e.g.
   host-side unit tests on a clean machine). All four functions degrade
   to no-ops; `drain_spans()` returns []. Callers don't need to guard.
"""

from __future__ import annotations

import contextlib
import contextvars
from typing import Callable, Optional

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    SpanProcessor = object  # type: ignore[assignment,misc]

from src.engine.telemetry.filters import FilteredSpanProcessor


_exporter = None  # type: ignore[var-annotated]
_processor = None  # type: ignore[var-annotated]
_owns_provider = False

# Per-row span buffer selected via a ContextVar. When a worker enters
# `collect_row_spans()` it sets its own list here; the routing processor's
# `on_end` appends each finished span to whatever buffer is current in the
# calling context. Because every row task runs inside its own
# `contextvars.copy_context()`, concurrent rows each see their own buffer and
# spans never cross rows. When no buffer is set (None), spans fall back to the
# in-memory exporter so the legacy `drain_spans()` path keeps working.
_row_buffer: contextvars.ContextVar = contextvars.ContextVar(
    "ce_row_span_buffer", default=None
)

# Per-row ancestry skip-map: `{dropped_span_id: parent_span_id}`. Populated by
# `_record_drop_link` (wired as the filter's `on_drop` hook) so the converter
# can re-parent a kept span whose direct parent was filtered out to its nearest
# surviving ancestor. Ids only — the dropped spans' (often large, duplicated)
# payloads are still discarded at `on_end`. Per-row by the same
# `copy_context()` isolation as `_row_buffer`, so ids never collide across rows.
_row_drop_links: contextvars.ContextVar = contextvars.ContextVar(
    "ce_row_drop_links", default=None
)


def _record_drop_link(span) -> None:
    """Record a dropped span's (id -> parent_id) into the current row's map.

    No-op when no row collection is active (e.g. the legacy `drain_spans`
    path). The filter's `on_drop` calls this synchronously in the
    span-ending context, so it lands in the right row's map.
    """
    links = _row_drop_links.get()
    if links is None:
        return
    ctx = getattr(span, "context", None) or getattr(span, "get_span_context", lambda: None)()
    sid = getattr(ctx, "span_id", None)
    if sid is None:
        return
    parent = getattr(span, "parent", None)
    links[sid] = getattr(parent, "span_id", None) if parent is not None else None


class _RoutingSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Leaf processor that routes each finished span to the current row buffer.

    Captures its own exporter and an `_active` flag so that
    `shutdown_tracer()` makes it inert without detaching it from the global
    provider (OTel has no remove-processor API, and the global provider is
    set-once per process). This mirrors how `SimpleSpanProcessor.shutdown()`
    renders a processor inert — keeping `setup_tracer()`
    idempotent even though stale processors remain attached.
    """

    def __init__(self, exporter):
        self._exporter = exporter
        self._active = True

    def on_start(self, span, parent_context=None):  # type: ignore[override]
        pass

    def on_end(self, span):  # type: ignore[override]
        if not self._active:
            return
        buf = _row_buffer.get()
        if buf is not None:
            buf.append(span)
        elif self._exporter is not None:
            self._exporter.export((span,))

    def shutdown(self):  # type: ignore[override]
        self._active = False

    def force_flush(self, timeout_millis=30_000):  # type: ignore[override]
        return True


@contextlib.contextmanager
def collect_row_spans():
    """Scope a per-row span buffer + ancestry skip-map for the `with` block.

    Yields `(spans, drop_links)`. Every kept span whose `on_end` fires within
    this context is appended to `spans`; every dropped span's
    `(id -> parent_id)` link is recorded into `drop_links`. Nest the row's
    `pipeline` root span *inside* this block and read both after the root span
    has ended (so its own `on_end` has run):

        with collect_row_spans() as (spans, drop_links):
            with tracer.start_as_current_span("pipeline"):
                ...
            row_spans, row_links = list(spans), dict(drop_links)

    Pass `drop_links` to `converter.spans_to_trace_entries(...)` so kept spans
    re-parent to their nearest surviving ancestor. No-op-safe: yields empties
    even when OTel is unavailable.
    """
    buf: list = []
    drop_links: dict = {}
    buf_token = _row_buffer.set(buf)
    links_token = _row_drop_links.set(drop_links)
    try:
        yield buf, drop_links
    finally:
        _row_drop_links.reset(links_token)
        _row_buffer.reset(buf_token)


def setup_tracer(
    service_name: str = "codeevolver-eval",
    span_filter: Optional[Callable[[object], bool]] = None,
) -> None:
    """Install an in-memory span collector on the global TracerProvider.

    If a real SDK `TracerProvider` is already set as the global, attach our
    span processor to it (sharing with whatever the client wired up).
    Otherwise create our own provider and install it as the global.

    When `span_filter` is provided, the attached processor is wrapped with
    `FilteredSpanProcessor` so only spans for which `span_filter(span)` is
    True reach the in-memory exporter. Use
    `telemetry.filters.make_keep_predicate([...])` to compose per-framework
    drop rules (see `telemetry/filters/ai_frameworks/`).

    Idempotent: a second call tears down the previous setup before re-entering.
    """
    global _exporter, _processor, _owns_provider

    if not OTEL_AVAILABLE:
        return

    if _processor is not None:
        shutdown_tracer()

    _exporter = InMemorySpanExporter()
    base_processor = _RoutingSpanProcessor(_exporter)
    _processor = (
        FilteredSpanProcessor(base_processor, span_filter, on_drop=_record_drop_link)
        if span_filter is not None
        else base_processor
    )

    existing = trace.get_tracer_provider()
    if hasattr(existing, "add_span_processor"):
        existing.add_span_processor(_processor)
        _owns_provider = False
    else:
        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name})
        )
        provider.add_span_processor(_processor)
        trace.set_tracer_provider(provider)
        _owns_provider = True


def drain_spans() -> list:
    """Return finished spans collected since the last drain, then clear."""
    if not OTEL_AVAILABLE or _exporter is None:
        return []
    spans = list(_exporter.get_finished_spans())
    _exporter.clear()
    return spans


def shutdown_tracer() -> None:
    """Stop forwarding spans to our exporter and release our state.

    We never call `provider.shutdown()` on the global TracerProvider:
    OTel only allows the global provider to be set once per process, so a
    teardown-and-reset pattern would leave any subsequent code (including
    other tests in the same pytest process) talking to a dead provider.
    Instead we just shut down our own SimpleSpanProcessor, which detaches
    it from forwarding without touching the provider. The processor stays
    attached to the provider but inert — and the OS will reclaim it when
    the process exits.
    """
    global _exporter, _processor, _owns_provider

    if not OTEL_AVAILABLE:
        return

    if _processor is not None:
        try:
            _processor.force_flush()
        except Exception:
            pass
        try:
            _processor.shutdown()
        except Exception:
            pass

    _exporter = None
    _processor = None
    _owns_provider = False


__all__ = [
    "OTEL_AVAILABLE",
    "setup_tracer",
    "collect_row_spans",
    "drain_spans",
    "shutdown_tracer",
]
