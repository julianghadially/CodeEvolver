"""Copyright © 2026 440 Labs LLC

Span filters for the CodeEvolver telemetry pipeline.

A "filter" here is a `should_keep(span) -> bool` predicate. Filters drop
noisy framework-internal spans before they reach the in-memory exporter,
so the per-row `trace[]` arrays written to /traces/ stay small and
readable for the reflection LM.

Framework-specific drop rules live under `ai_frameworks/` (one module per
framework, e.g. `ai_frameworks/dspy.py`). This package exposes only the
generic pieces:

- `FilteredSpanProcessor` — wraps any OTel `SpanProcessor` and delegates
  `on_end` only when the predicate returns True.
- `KEEP_KINDS` — OpenInference span kinds that carry signal (LLM, TOOL,
  RETRIEVER, RERANKER, EMBEDDING, AGENT). These are always kept regardless
  of framework drop rules.
- `make_keep_predicate(drop_predicates)` — compose `KEEP_KINDS` with any
  number of framework `should_drop` predicates into a single `should_keep`.
"""

from __future__ import annotations

from typing import Callable, Iterable

try:
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor

    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False
    ReadableSpan = object  # type: ignore[assignment,misc]
    SpanProcessor = object  # type: ignore[assignment,misc]


KEEP_KINDS = frozenset(
    {"LLM", "TOOL", "RETRIEVER", "RERANKER", "EMBEDDING", "AGENT"}
)


DropPredicate = Callable[["ReadableSpan"], bool]
KeepPredicate = Callable[["ReadableSpan"], bool]


def make_keep_predicate(
    drop_predicates: Iterable[DropPredicate] = (),
) -> KeepPredicate:
    """Compose OpenInference keep-kinds with per-framework drop rules.

    A span is kept if its `openinference.span.kind` is in `KEEP_KINDS`,
    OR if no drop predicate claims it. Any drop predicate returning True
    causes the span to be filtered out.
    """
    drops = tuple(drop_predicates)

    def should_keep(span: "ReadableSpan") -> bool:
        attrs = span.attributes or {}
        if attrs.get("openinference.span.kind") in KEEP_KINDS:
            return True
        for drop in drops:
            if drop(span):
                return False
        return True

    return should_keep


class FilteredSpanProcessor(SpanProcessor):  # type: ignore[misc]
    """Forward spans to `wrapped` only when `should_keep(span)` is True.

    `on_drop`, when given, is called with each span that fails `should_keep`
    instead of the span being silently discarded. The telemetry layer uses
    this to record the dropped span's (id -> parent_id) ancestry link so the
    converter can re-parent kept descendants to their nearest surviving
    ancestor. Relies on `on_end` running synchronously in the span-ending
    thread/context (true for the SimpleSpanProcessor-style processors we wrap);
    a `BatchSpanProcessor` on our own path would break that contextvar
    assumption — the same assumption the row-span buffer already makes.
    """

    def __init__(
        self,
        wrapped: "SpanProcessor",
        should_keep: KeepPredicate,
        on_drop: Callable[["ReadableSpan"], None] | None = None,
    ):
        self._wrapped = wrapped
        self._should_keep = should_keep
        self._on_drop = on_drop

    def on_start(self, span, parent_context=None):  # type: ignore[override]
        self._wrapped.on_start(span, parent_context)

    def on_end(self, span: "ReadableSpan") -> None:  # type: ignore[override]
        if self._should_keep(span):
            self._wrapped.on_end(span)
        elif self._on_drop is not None:
            self._on_drop(span)

    def shutdown(self) -> None:  # type: ignore[override]
        self._wrapped.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:  # type: ignore[override]
        return self._wrapped.force_flush(timeout_millis)


__all__ = [
    "OTEL_AVAILABLE",
    "KEEP_KINDS",
    "DropPredicate",
    "KeepPredicate",
    "make_keep_predicate",
    "FilteredSpanProcessor",
]
