"""Copyright © 2026 440 Labs LLC

Convert OpenTelemetry spans into the trace-entry dicts written to
/traces/iteration_{N}_{suffix}/rows/<example_id>/span_NNN.json.

Attribute conventions
---------------------
CodeEvolver reserves the `ce.` namespace on span attributes:
  - `ce.span_kind`     -> entry["span_kind"]
  - `ce.inputs.<name>` -> entry["inputs"][<name>]
  - `ce.output`        -> entry["output"]
  - `ce.error`         -> entry["error"]

Anything else becomes `entry["metadata"]`.

Framework fallback: when a span has no `ce.inputs.*` but DOES have
common AI-framework attributes, promote those into `inputs` / `output`.
This lets us collect spans from OpenInference / OpenLLMetry instrumentors
without forcing the client to re-decorate. OpenInference flattens chat
messages to `llm.input_messages.<i>.message.{role,content}`; those are
rebuilt into `inputs["messages"]` (a list of `{role, content}`), and only
when there are none does `input.value` / `gen_ai.prompt` become
`inputs["prompt"]`. Whatever was promoted is dropped from `metadata` so
an LM span does not carry the same prompt three times (`input.value`, the
flattened messages, and `inputs["prompt"]`).
Spans that set `ce.inputs.*` / `ce.output` keep their metadata untouched.

Parent-input deduplication: after entries are built, a child whose
`inputs[<key>]` is a long string (>= DEDUPE_THRESHOLD chars) and is
exactly equal to the parent span's value for the same key is replaced
with a short marker. This stops a 100k-token prompt being repeated up
the call stack from blowing out reflection-LM context. Resolved by
span_id, not span name, so sibling collisions are correctly handled.

Spec note: OTel attributes can only be primitives or sequences of
primitives. `ce.output` therefore always round-trips as a string even
though a client may set a dict. Clients
that want structured output must `json.dumps()` it themselves; this
converter does not attempt to JSON-parse.
"""

from __future__ import annotations

import re
from typing import Any

CE_PREFIX = "ce."
CE_INPUTS_PREFIX = "ce.inputs."

_FRAMEWORK_INPUT_KEYS = ("input.value", "gen_ai.prompt")
_FRAMEWORK_OUTPUT_KEYS = ("output.value", "gen_ai.completion")
# OpenInference flattens chat messages: `llm.input_messages.0.message.role`,
# `llm.input_messages.0.message.content`, ... (same for output_messages).
_FLAT_MESSAGE_RE = re.compile(
    r"^llm\.(input|output)_messages\.(\d+)\.message\.(role|content)$"
)


def _collect_flat_messages(attrs: dict[str, Any]) -> tuple[dict[str, list[dict]], set[str]]:
    """Rebuild `{"input": [...], "output": [...]}` message lists from the
    flattened OpenInference keys. Returns the lists plus the consumed keys."""
    by_side: dict[str, dict[int, dict[str, Any]]] = {"input": {}, "output": {}}
    consumed: set[str] = set()
    for k, v in attrs.items():
        m = _FLAT_MESSAGE_RE.match(k)
        if m is None:
            continue
        side, idx, field = m.group(1), int(m.group(2)), m.group(3)
        by_side[side].setdefault(idx, {})[field] = v
        consumed.add(k)
    messages = {
        side: [by_idx[i] for i in sorted(by_idx)]
        for side, by_idx in by_side.items()
        if by_idx
    }
    return messages, consumed

DEDUPE_THRESHOLD = 5_000

# Upper bound on the ancestry walk — guards against a pathologically deep or
# malformed chain. Returning None past this degrades to a detached root.
_MAX_ANCESTRY_DEPTH = 10_000


def span_to_trace_entry(span: Any) -> dict:
    """Convert a single OTel ReadableSpan to a trace entry dict.

    `parent_key` is left as None — the caller (`spans_to_trace_entries`)
    fills it in after building the row-local span_id -> name lookup.
    """
    attrs = dict(span.attributes or {})
    inputs: dict[str, Any] = {}
    output: Any = None
    error: str | None = None
    metadata: dict[str, Any] = {}
    kind = "function"

    for k, v in attrs.items():
        if k.startswith(CE_INPUTS_PREFIX):
            inputs[k[len(CE_INPUTS_PREFIX):]] = v
        elif k == "ce.output":
            output = v
        elif k == "ce.error":
            error = v
        elif k == "ce.span_kind":
            kind = str(v)
        else:
            metadata[k] = v

    # Framework fallback. Every attribute promoted into inputs/output is
    # removed from metadata so the payload is stored once, not three times.
    flat_messages, flat_keys = _collect_flat_messages(attrs)
    if not inputs:
        if "input" in flat_messages:
            inputs["messages"] = flat_messages["input"]
            for k in flat_keys:
                if k.startswith("llm.input_messages."):
                    metadata.pop(k, None)
            # `input.value` is the same request serialized as JSON.
            metadata.pop("input.value", None)
        else:
            for fk in _FRAMEWORK_INPUT_KEYS:
                if fk in attrs:
                    inputs["prompt"] = attrs[fk]
                    metadata.pop(fk, None)
                    break
    if output is None:
        if "output" in flat_messages:
            output = flat_messages["output"]
            for k in flat_keys:
                if k.startswith("llm.output_messages."):
                    metadata.pop(k, None)
            metadata.pop("output.value", None)
        else:
            for fk in _FRAMEWORK_OUTPUT_KEYS:
                if fk in attrs:
                    output = attrs[fk]
                    metadata.pop(fk, None)
                    break

    if kind == "function" and (
        flat_messages or any(k.startswith("gen_ai.") for k in attrs)
    ):
        kind = "llm"

    status = getattr(span, "status", None)
    if error is None and status is not None:
        code = getattr(status, "status_code", None)
        code_name = getattr(code, "name", None)
        if code_name == "ERROR":
            error = getattr(status, "description", None) or "unknown"

    entry: dict[str, Any] = {
        "signature_key": span.name,
        "span_kind": kind,
        "parent_key": None,
        "inputs": inputs,
        "output": output,
        "error": error,
    }
    if metadata:
        entry["metadata"] = metadata
    start_time_ns = getattr(span, "start_time", None)
    if start_time_ns is not None:
        entry["start_time"] = start_time_ns / 1e9
    return entry


def _resolve_kept_ancestor(
    start_parent_id: int | None,
    parent_of: dict[int, int | None],
    kept_ids: set[int],
) -> int | None:
    """Climb `parent_of` from `start_parent_id` to the nearest kept span id.

    Skips ids that are not in `kept_ids` (dropped/filtered spans), bridging
    over them via the drop-link entries merged into `parent_of`. Returns the
    nearest kept ancestor's id, or None if the chain reaches a true root or a
    dropped node with no recorded parent. Cycle/depth-safe: a loop or an
    over-deep chain returns None (detached root) rather than hanging.
    """
    cur = start_parent_id
    seen: set[int] = set()
    depth = 0
    while cur is not None and cur not in kept_ids:
        if cur in seen or depth > _MAX_ANCESTRY_DEPTH:
            return None
        seen.add(cur)
        depth += 1
        cur = parent_of.get(cur)
    return cur


def spans_to_trace_entries(
    spans: list,
    dedupe_threshold: int = DEDUPE_THRESHOLD,
    drop_links: dict[int, int | None] | None = None,
) -> list[dict]:
    """Convert and order spans, resolving parent_key to the nearest kept ancestor.

    The lookup is row-local because `evaluate.py` collects spans per row, so
    cross-row span_id collisions are impossible here.

    `drop_links` is the per-row `{dropped_span_id: parent_span_id}` ancestry
    skip-map produced by `telemetry.collect_row_spans`. When a kept span's
    direct parent was filtered out (absent from the kept set), the parent is
    resolved by walking `drop_links` up to the nearest surviving ancestor — so
    e.g. a DSPy `LM.__call__` whose `ChatAdapter`/`Predict` wrappers were
    dropped re-parents to the surviving `JudgeModule.forward`. When
    `drop_links` is None/empty the walk reduces to the direct parent (kept) or
    None, matching the pre-skip-map behavior exactly.

    `dedupe_threshold` controls the parent-input dedupe pass (run against the
    *resolved* ancestor). Set to 0 or a negative number to disable.
    """
    name_by_id: dict[int, str] = {}
    for s in spans:
        ctx = getattr(s, "context", None) or getattr(s, "get_span_context", lambda: None)()
        sid = getattr(ctx, "span_id", None)
        if sid is not None:
            name_by_id[sid] = s.name

    kept_ids = set(name_by_id)
    # Ancestry of every node we know about: kept spans' own parent links plus
    # the dropped spans' links, so the walk can bridge across dropped chains.
    parent_of: dict[int, int | None] = dict(drop_links or {})
    for s in spans:
        ctx = getattr(s, "context", None) or getattr(s, "get_span_context", lambda: None)()
        sid = getattr(ctx, "span_id", None)
        if sid is not None:
            parent = getattr(s, "parent", None)
            parent_of[sid] = getattr(parent, "span_id", None) if parent is not None else None

    sorted_spans = sorted(spans, key=lambda s: getattr(s, "start_time", 0) or 0)
    entries: list[dict] = []
    entry_by_id: dict[int, dict] = {}
    parent_id_by_idx: dict[int, int | None] = {}
    for idx, s in enumerate(sorted_spans):
        entry = span_to_trace_entry(s)
        parent = getattr(s, "parent", None)
        parent_id = getattr(parent, "span_id", None) if parent is not None else None
        resolved_parent_id = _resolve_kept_ancestor(parent_id, parent_of, kept_ids)
        if resolved_parent_id is not None:
            entry["parent_key"] = name_by_id.get(resolved_parent_id)

        ctx = getattr(s, "context", None) or getattr(s, "get_span_context", lambda: None)()
        sid = getattr(ctx, "span_id", None)
        if sid is not None:
            entry_by_id[sid] = entry
        parent_id_by_idx[idx] = resolved_parent_id
        entries.append(entry)

    if dedupe_threshold > 0:
        _dedupe_inputs_against_parent(
            entries=entries,
            parent_id_by_idx=parent_id_by_idx,
            entry_by_id=entry_by_id,
            threshold=dedupe_threshold,
        )
    return entries


def _dedupe_inputs_against_parent(
    *,
    entries: list[dict],
    parent_id_by_idx: dict[int, int | None],
    entry_by_id: dict[int, dict],
    threshold: int,
) -> None:
    """Replace child `inputs[<key>]` values >= threshold chars that are
    exactly equal to the parent's value for the same key with a marker.

    Mutates `entries` in place. Equality test is plain `==` on strings;
    CPython compares via memcmp, so the cost is dominated by string size
    and stays sub-millisecond per row even at million-character inputs.
    """
    for idx, entry in enumerate(entries):
        parent_id = parent_id_by_idx.get(idx)
        if parent_id is None:
            continue
        parent_entry = entry_by_id.get(parent_id)
        if parent_entry is None:
            continue
        parent_inputs = parent_entry.get("inputs") or {}
        inputs = entry.get("inputs") or {}
        if not parent_inputs or not inputs:
            continue
        parent_name = parent_entry.get("signature_key") or entry.get("parent_key") or "<parent>"
        for k, v in list(inputs.items()):
            if not isinstance(v, str) or len(v) < threshold:
                continue
            if parent_inputs.get(k) == v:
                inputs[k] = f"<inherited from parent span '{parent_name}'>"


__all__ = ["span_to_trace_entry", "spans_to_trace_entries", "DEDUPE_THRESHOLD"]
