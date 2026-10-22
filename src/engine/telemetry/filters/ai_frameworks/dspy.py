"""Copyright © 2026 440 Labs LLC

DSPy-specific span drop rules.

DSPy + OpenInference emits a deep span tree for every LM call:

    pipeline -> JudgeModule.forward -> ChainOfThought.forward
        -> Predict.forward -> Predict(Signature).forward
        -> ChatAdapter.__call__ -> LM.__call__

The intermediate `Predict` / `ChainOfThought` / `ChatAdapter` nodes are
structural plumbing — same inputs and outputs as the `LM.__call__` child
they wrap, just repeated at each level. Keeping them bloats the trace
JSONL and burns architect / reflection-LM context on no-signal rows.

`LM.__call__`, TOOL / RETRIEVER spans, and the user's own
`dspy.Module.forward` spans all have `openinference.span.kind` set and
are retained by the generic `KEEP_KINDS` check in
`telemetry.filters.make_keep_predicate`. This module only adds DSPy's
structural drop rules — it does not reimplement that keep logic.
"""

from __future__ import annotations

from typing import Any


DROP_NAMES = frozenset(
    {
        "ChainOfThought.forward",
        "Predict.forward",
        "ChatAdapter.__call__",
        "ChatAdapter.format",
        "ChatAdapter.parse",
    }
)

DROP_NAME_PREFIXES = ("Predict(",)


def should_drop(span: Any) -> bool:
    """Return True if `span` is a DSPy internal plumbing span."""
    name = span.name or ""
    return name in DROP_NAMES or name.startswith(DROP_NAME_PREFIXES)


__all__ = ["DROP_NAMES", "DROP_NAME_PREFIXES", "should_drop"]
