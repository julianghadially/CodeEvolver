"""Copyright © 2026 440 Labs LLC

Local trace-dump SpanProcessor for inspecting `@traceable` output during
development. Optional companion to `traceable.py`.

What it does
------------
Buffers spans by ``trace_id`` and emits **one JSON line per completed
trace** to a file (default ``traces/traces.jsonl`` in the repo root).
Each line is a self-contained example: the root span plus every
descendant, sorted by start time, with ``parent_id`` preserved so the
reader can reconstruct the tree.

Output format::

    {
      "trace_id": "...",
      "root_name": "pipeline",
      "duration_ms": 4521.3,
      "spans": [
        {"name": "pipeline", "span_id": "...", "parent_id": null,
         "attributes": {"ce.span_kind": "chain", "ce.inputs.x": "...", "ce.output": "..."},
         "start_time_ns": ..., "end_time_ns": ..., "status": "OK"},
        {"name": "parse_claim", "span_id": "...", "parent_id": "<root span_id>", ...},
        ...
      ]
    }

Each line is one example; spans inside the line are flat with
``parent_id`` references — the same shape CodeEvolver's converter
produces in ``trace[]`` entries, so this is portable across repos.

Usage
-----
Copy this file into the client repo (e.g. as
``<your_pkg>/_tracing_export.py``) and call ``install_file_exporter()``
once at process startup, after any framework-specific tracer setup
(``configure_azure_monitor``, DSPy instrumentor, etc.):

    from your_pkg._tracing_export import install_file_exporter
    install_file_exporter()                       # default path
    install_file_exporter("traces/my_run.jsonl")  # custom path

Constraints
-----------
- Additive: does NOT replace any existing TracerProvider. CodeEvolver's
  in-memory exporter and any framework auto-instrumentation keep working
  alongside.
- Idempotent: first call truncates the file (so each fresh process
  starts with an empty file); subsequent calls in the same process are
  no-ops.
- Local inspection only. Do NOT enable in production — the file grows
  per request and contains full prompts.
- Standard ``opentelemetry-api`` + ``opentelemetry-sdk`` only. No
  CodeEvolver imports.
"""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor
from opentelemetry.sdk.trace import TracerProvider as SDKTracerProvider

_DEFAULT_PATH = Path("traces/traces.jsonl")

_INSTALL_LOCK = threading.Lock()
_INSTALLED = False


class _GroupedFileSpanProcessor(SpanProcessor):
    """Collects spans per ``trace_id``; flushes each trace as one JSON
    line when its root span ends."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("", encoding="utf-8")  # truncate per process
        self._buffer: dict[int, list[ReadableSpan]] = defaultdict(list)
        self._lock = threading.Lock()

    def on_start(self, span, parent_context=None) -> None:  # noqa: D401
        return

    def on_end(self, span: ReadableSpan) -> None:
        trace_id = span.context.trace_id
        is_root = span.parent is None or span.parent.trace_id != trace_id

        with self._lock:
            self._buffer[trace_id].append(span)
            if is_root:
                spans = self._buffer.pop(trace_id, [])
                if spans:
                    self._write(spans)

    def shutdown(self) -> None:
        # Flush any traces whose root never ended (process exit, errors).
        with self._lock:
            for spans in self._buffer.values():
                if spans:
                    self._write(spans)
            self._buffer.clear()

    def force_flush(self, timeout_millis: int | None = None) -> bool:  # noqa: ARG002
        return True

    def _write(self, spans: list[ReadableSpan]) -> None:
        spans.sort(key=lambda s: s.start_time or 0)
        root = spans[0]
        duration_ms: float | None = None
        if root.end_time and root.start_time:
            duration_ms = (root.end_time - root.start_time) / 1_000_000

        bundle: dict[str, Any] = {
            "trace_id": format(root.context.trace_id, "032x"),
            "root_name": root.name,
            "duration_ms": duration_ms,
            "spans": [_span_dict(s) for s in spans],
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(bundle, ensure_ascii=False, default=str) + "\n")


def _span_dict(span: ReadableSpan) -> dict[str, Any]:
    parent_id = (
        format(span.parent.span_id, "016x") if span.parent is not None else None
    )
    status_code = span.status.status_code.name if span.status else None
    return {
        "name": span.name,
        "span_id": format(span.context.span_id, "016x"),
        "parent_id": parent_id,
        "start_time_ns": span.start_time,
        "end_time_ns": span.end_time,
        "attributes": dict(span.attributes or {}),
        "status": status_code,
    }


def install_file_exporter(path: Path | str = _DEFAULT_PATH) -> Path:
    """Idempotently install the grouped trace-dump processor. First call
    truncates the file; subsequent calls in the same process are no-ops.
    Returns the resolved path so callers can log it."""
    global _INSTALLED
    resolved = Path(path)

    with _INSTALL_LOCK:
        if _INSTALLED:
            return resolved

        provider = trace.get_tracer_provider()
        if not isinstance(provider, SDKTracerProvider):
            provider = SDKTracerProvider()
            trace.set_tracer_provider(provider)

        provider.add_span_processor(_GroupedFileSpanProcessor(resolved))
        _INSTALLED = True

    return resolved


__all__ = ["install_file_exporter"]
