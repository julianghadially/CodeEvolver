"""Copyright © 2026 440 Labs LLC

Drop-in `@traceable` decorator for instrumenting client code with
OpenTelemetry spans that CodeEvolver's IterationArchitect can read.

Usage
-----
Copy this file into the client repository (e.g. as `<your_pkg>/_tracing.py`)
and decorate functions you want surfaced in the trace:

    from your_pkg._tracing import traceable

    @traceable("tool")
    def parse_claim(text: str) -> list[str]:
        ...

    @traceable("llm")
    async def call_judge(prompt: str) -> str:
        ...

What gets recorded
------------------
For each call, the decorator opens an OTel span and sets:

  - `ce.span_kind`       — one of "llm" / "tool" / "chain" / "retriever" / "function"
  - `ce.inputs.<name>`   — one attribute per declared parameter
  - `ce.output`          — string form of the return value
  - `ce.error`           — exception type and message (raised exceptions still propagate)

CodeEvolver's `span_to_trace_entry` (`src/engine/telemetry/converter.py`)
reads these attributes back into trace JSONL. Setting any other attributes
is fine — they land in `metadata` on the trace entry.

Constraints
-----------
- Standard `opentelemetry-api` only. No CodeEvolver imports.
- Sync and async functions both supported.
- Inputs and output are stored in full by default — the architect is expected
  to grep / parse them itself. Pass `max_attr_chars=N` per-decoration to opt
  into per-attribute truncation if a specific call site genuinely needs to
  be capped (e.g. PII, binary blobs).
- If `opentelemetry` is not importable in the runtime, the decorator falls
  back to a no-op so client code keeps working in non-CodeEvolver contexts.
"""

from __future__ import annotations

import functools
import inspect
from typing import Any, Callable, TypeVar

try:
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import StatusCode

    _OTEL_AVAILABLE = True
except Exception:  # pragma: no cover — exercised only when OTel isn't installed
    _otel_trace = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment]
    _OTEL_AVAILABLE = False


F = TypeVar("F", bound=Callable[..., Any])

_DEFAULT_MAX_ATTR_CHARS: int | None = None
_VALID_SPAN_KINDS = {"llm", "tool", "chain", "retriever", "function"}


def _truncate(value: Any, limit: int | None) -> str:
    text = str(value)
    if limit is None or len(text) <= limit:
        return text
    return text[:limit]


_IMPLICIT_PARAMS = {"self", "cls"}


def _bind_inputs(fn: Callable[..., Any], args: tuple, kwargs: dict) -> dict[str, Any]:
    try:
        bound = inspect.signature(fn).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: v for k, v in bound.arguments.items() if k not in _IMPLICIT_PARAMS}
    except TypeError:
        return dict(kwargs)


def traceable(
    span_kind: str = "function",
    name: str | None = None,
    max_attr_chars: int | None = _DEFAULT_MAX_ATTR_CHARS,
) -> Callable[[F], F]:
    """Wrap a function in an OTel span using CodeEvolver's `ce.*` convention.

    Parameters
    ----------
    span_kind:
        One of `"llm"`, `"tool"`, `"chain"`, `"retriever"`, `"function"`.
        Drives the architect's filtering — pick the most specific kind that
        applies. `"function"` is fine for generic deterministic code.
    name:
        Override the span name. Defaults to `fn.__qualname__`, which renders
        nicely in trace JSONL (e.g. `"FactChecker.parse_claim"`).
    max_attr_chars:
        Per-attribute character budget. Default `None` means no truncation —
        the architect reads the full input / output and greps it. Pass a
        positive int to opt into truncation at a specific call site (e.g.
        a function that handles secrets or binary blobs).
    """
    if span_kind not in _VALID_SPAN_KINDS:
        raise ValueError(
            f"span_kind must be one of {sorted(_VALID_SPAN_KINDS)}; got {span_kind!r}"
        )

    def decorator(fn: F) -> F:
        if not _OTEL_AVAILABLE:
            return fn

        span_name = name or fn.__qualname__
        tracer = _otel_trace.get_tracer(fn.__module__ or "codeevolver.client")

        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                with tracer.start_as_current_span(span_name) as span:
                    span.set_attribute("ce.span_kind", span_kind)
                    for k, v in _bind_inputs(fn, args, kwargs).items():
                        span.set_attribute(f"ce.inputs.{k}", _truncate(v, max_attr_chars))
                    try:
                        result = await fn(*args, **kwargs)
                    except Exception as exc:
                        span.set_attribute(
                            "ce.error", _truncate(f"{type(exc).__name__}: {exc}", max_attr_chars)
                        )
                        span.set_status(StatusCode.ERROR, str(exc))
                        span.record_exception(exc)
                        raise
                    span.set_attribute("ce.output", _truncate(result, max_attr_chars))
                    return result

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            with tracer.start_as_current_span(span_name) as span:
                span.set_attribute("ce.span_kind", span_kind)
                for k, v in _bind_inputs(fn, args, kwargs).items():
                    span.set_attribute(f"ce.inputs.{k}", _truncate(v, max_attr_chars))
                try:
                    result = fn(*args, **kwargs)
                except Exception as exc:
                    span.set_attribute(
                        "ce.error", _truncate(f"{type(exc).__name__}: {exc}", max_attr_chars)
                    )
                    span.set_status(StatusCode.ERROR, str(exc))
                    span.record_exception(exc)
                    raise
                span.set_attribute("ce.output", _truncate(result, max_attr_chars))
                return result

        return sync_wrapper  # type: ignore[return-value]

    return decorator


__all__ = ["traceable"]
