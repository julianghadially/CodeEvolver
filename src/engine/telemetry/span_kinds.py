"""Copyright © 2026 440 Labs LLC

Span classification used by CodeEvolver's `ce.span_kind` attribute.

Distinct from OTel's built-in `SpanKind` (which encodes RPC semantics:
CLIENT/SERVER/INTERNAL). `CESpanKind` classifies AI pipeline operations
so the reflection agent can reason about an execution graph that mixes
LLM calls, deterministic tools, and orchestration nodes.
"""

from __future__ import annotations

from enum import Enum


class CESpanKind(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    CHAIN = "chain"
    RETRIEVER = "retriever"
    FUNCTION = "function"


__all__ = ["CESpanKind"]
