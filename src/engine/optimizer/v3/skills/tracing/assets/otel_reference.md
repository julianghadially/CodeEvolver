# OpenTelemetry reference for CodeEvolver tracing

This is the self-contained reference for how OTel is collected, what attributes
matter, and what the architect sees per row. It is mirrored from the relevant
sections of `specs/engine2.md`; if the two ever drift, the spec wins for engine
internals and this skill wins for client-facing instrumentation guidance.

## How OTel collection works

CodeEvolver relies on the process-global `TracerProvider`. Any code that calls
`trace.get_tracer(...)` routes spans to whatever provider is currently set.
The optimizer exploits this:

1. **Before evaluation**: `setup_tracer()` (in `src/engine/telemetry/__init__.py`)
   probes the global provider. If a real SDK `TracerProvider` is already
   installed (e.g. the client's LangFuse/Honeycomb wiring), it adds our
   `SimpleSpanProcessor` + `InMemorySpanExporter` to it — without replacing
   the existing setup. If no real provider exists, it installs one.
2. **During evaluation**: the client program runs. Any OTel-decorated
   function (LLM calls, parsers, rules, etc.) emits spans that are
   automatically collected in the in-memory exporter.
3. **After each row**: `drain_spans()` returns the row's spans and clears the
   exporter buffer so the next row gets a fresh trace.
4. **On shutdown**: `shutdown_tracer()` detaches the processor without
   destroying the provider (OTel globals are set-once per process).

No CodeEvolver imports in client code. The client uses standard OpenTelemetry:

```python
from opentelemetry import trace
tracer = trace.get_tracer("my-service")

with tracer.start_as_current_span("function_name"):
    ...
```

The `@traceable` decorator in this skill is just a thin wrapper over that
exact pattern.

## What gets collected

Span attributes (set by whoever creates the span — client decorator, AI
framework instrumentor, or our `@traceable` decorator):

| Attribute             | Set by             | Purpose                                                                 |
| --------------------- | ------------------ | ----------------------------------------------------------------------- |
| Span name             | Client / decorator | Identifies the function (e.g. `"FactChecker.parse_claim"`, `"llm.generate"`). Defaults to `fn.__qualname__` for `@traceable`. |
| Parent span ID        | OTel context propagation | Automatic — nested spans get parent-child relationships.            |
| Start / end time      | OTel SDK           | Automatic.                                                              |
| `ce.span_kind`        | Client / decorator | Classifies the span: `"llm"`, `"tool"`, `"chain"`, `"retriever"`, `"function"`. |
| `ce.inputs.*`         | Client / decorator | Function inputs as span attributes (one key per parameter).             |
| `ce.output`           | Client / decorator | Function return value as a span attribute.                              |
| `ce.error`            | Client / decorator | Exception type + message if the function threw.                         |

The `ce.*` prefix is the CodeEvolver convention. Standard OTel attributes
(`gen_ai.request.model`, etc.) are preserved in `metadata` on the trace
entry but are not required.

## Trace entry fields

The converter (`src/engine/telemetry/converter.py`) turns each span into a
trace entry dict:

| Field           | Type            | Required | Description                                                                 |
| --------------- | --------------- | -------- | --------------------------------------------------------------------------- |
| `signature_key` | string          | yes      | Span name — identifies the function or module.                              |
| `span_kind`     | string          | no       | `"llm"`, `"tool"`, `"chain"`, `"retriever"`, `"function"`. Defaults to `"function"` if `ce.span_kind` is absent. Inferred to `"llm"` when any `gen_ai.*` attribute is present. |
| `parent_key`    | string or null  | no       | Name of the parent span. `null` for root-level steps. Resolved row-locally by `spans_to_trace_entries`. |
| `inputs`        | dict            | yes      | Extracted from `ce.inputs.*` span attributes. Falls back to `gen_ai.prompt` / `llm.input_messages` / `input.value` when no `ce.inputs.*` exist (so OpenInference spans still show inputs). |
| `output`        | string or dict  | yes      | From `ce.output`. OTel attributes are primitives only, so this round-trips as a string. JSON-encode structured output yourself if you need it. |
| `error`         | string or null  | no       | From `ce.error`, or from the span status if the span ended with `StatusCode.ERROR`. |
| `metadata`      | dict            | no       | Any non-`ce.*` span attributes worth preserving (e.g., `model`, `tokens.*`). |
| `start_time`    | float           | no       | Span start time in seconds since epoch (used for ordering).                 |

## Trace directory format

`/traces/iteration_{N}_{suffix}/` — one directory per eval:

```
index.jsonl                      one short line per example: example_id, score,
                                 failed, feedback, error_summary, row_file
rows/<example_id>/row.json       the example (below) plus a `spans` table
rows/<example_id>/span_NNN.json  one span entry each (the trace[] items below)
provenance.json                  which eval produced this directory: eval_id,
                                 sha_at_eval, tree, committed, branch,
                                 eval_sample, suffix, score, ts
```

A directory is rewritten by every eval with the same suffix; `provenance.json`
is the only way to tell which version of the program an older directory
describes.

`row.json`:

```json
{
  "example_id": 0,
  "score": 0.85,
  "feedback": "Partially correct — wrong jurisdiction",
  "failed": false,
  "input": {"statement": "tomatoes are vegetables"},
  "output": {"verdict": "false"},
  "spans": [
    {"k": 0, "signature_key": "pipeline", "span_kind": "chain", "parent_key": null, "error": null, "file": "span_000.json"},
    {"k": 1, "signature_key": "FactChecker.parse_claim", "span_kind": "tool", "parent_key": "pipeline", "error": null, "file": "span_001.json"},
    {"k": 2, "signature_key": "llm.judge", "span_kind": "llm", "parent_key": "pipeline", "error": null, "file": "span_002.json"}
  ]
}
```

and the span files, e.g. `span_002.json`:

```json
{
  "signature_key": "llm.judge",
  "span_kind": "llm",
  "parent_key": "pipeline",
  "inputs": {"prompt": "Classify whether this claim is true or false..."},
  "output": "False",
  "metadata": {"model": "gpt-4o", "tokens.input": 342, "tokens.output": 89}
}
```

Every file is pretty-printed so an agent can read it in one call. When the run's coding harness is OpenCode (whose file tools cap lines at 2,000 characters), strings longer than that are written as `{"__chunked_string__": ["...", "..."]}` — the pieces concatenate to the original string.

The suffix convention:

- `"parent"` — evaluation of the current candidate before changes
- `"child"` — evaluation after the architect's proposed changes
- `"comparison"` — the fixed comparison set
- `"holdout"`, `"holdout_2"`, … — finalization held-out test-set draws (seed + best candidate); fenced from every agent

`index.jsonl` and the `spans` table are engine-side artifacts — clients don't generate them and don't need to know their shape. See `specs/engine3.md` if you're curious.

## Wrapping an existing telemetry system

If a client has their own telemetry pipeline (e.g. a LangFuse SDK call) and wants those events to also show up in CodeEvolver traces, wrap the call in an OTel span and mirror the kwargs as attributes. Example pattern:

```python
def attach_attributes(span, data, prefix="lf"):
    for k, v in data.items():
        try:
            span.set_attribute(f"{prefix}.{k}", str(v))
        except Exception:
            span.set_attribute(f"{prefix}.{k}", "[unserializable]")

with tracer.start_as_current_span("langfuse.generation") as span:
    attach_attributes(span, kwargs)
    result = self.langfuse.generation(**kwargs)
    if isinstance(result, dict):
        attach_attributes(span, result, prefix="lf.response")
```

CodeEvolver's converter dumps non-`ce.*` attributes into the trace entry's `metadata` field, so all the mirrored LangFuse data lands there for the architect to read. The client's own LangFuse pipeline keeps shipping in parallel — `setup_tracer` does not replace it.

## Raw instrumentation pattern (without `@traceable`)

If you need something the decorator can't express (custom span name per call, conditional capture, partial output), use the raw OTel pattern directly:

```python
from opentelemetry import trace
tracer = trace.get_tracer(__name__)

def client_function(text: str):
    with tracer.start_as_current_span("function_name") as span:
        span.set_attribute("ce.span_kind", "tool")
        span.set_attribute("ce.inputs.text", str(text))
        try:
            result = ...
            span.set_attribute("ce.output", str(result))
            return result
        except Exception as exc:
            span.set_attribute("ce.error", f"{type(exc).__name__}: {exc}")
            raise
```

This is what `@traceable` expands to internally. Keep `opentelemetry-api` as the only import — no CodeEvolver imports.

## Operational notes

- **Span volume.** Programs with many small functions can produce large trace directories. The architect's prompts already see only the index (span files are opened on demand via `inspect_traces`), so volume affects disk + reflection-LM context only. If a single row's trace exceeds a few hundred entries, prune by raising the threshold for what counts as "instrumentation-worthy" (see SKILL.md §"Step 3").
- **Attribute size.** `@traceable` stores inputs and outputs in full by default — the architect is expected to grep / parse large payloads itself rather than work from a truncated slice. If a single attribute genuinely needs to be capped (PII, binary blobs), pass `max_attr_chars=N` per-decoration.
- **Async programs.** OTel uses `contextvars`-based context propagation, which works correctly with `async/await` on Python 3.7+. No special handling needed — the `@traceable` decorator detects coroutine functions automatically.
- **Parent-input dedupe.** When a child span's `ce.inputs[<key>]` is ≥ 5,000 chars and exactly equal to its parent's value for the same key, the converter replaces the child's value with `"<inherited from parent span '<name>'>"`. This stops a long prompt from being repeated up a wrapper-chain in the trace — the architect still sees the prompt once on the originating span. Likewise, framework attributes promoted into `inputs` / `output` (OpenInference's `input.value` and flattened `llm.input_messages.*`, which become `inputs.messages`) are removed from `metadata` rather than stored a second and third time. Resolved by span_id, so siblings with the same name don't accidentally trigger. Implemented in `src/engine/telemetry/converter.py::_dedupe_inputs_against_parent`; threshold defaults to `DEDUPE_THRESHOLD = 5000`. Cost: sub-millisecond per row even at million-character inputs (CPython `==` is memcmp at C speed).

## Assumptions about client code

For the architect's traces to be useful, the client program needs:

1. **LLM calls are decorated** — Each language model query has an OTel span
   with `ce.span_kind = "llm"` and input/output attributes. This may come
   from:
   - AI framework auto-instrumentation (DSPy + OpenInference, LangChain +
     callbacks, OpenAI SDK instrumentor, etc.)
   - Manual `@traceable("llm")` or `@tracer.start_as_current_span(...)`.
2. **Key deterministic functions are decorated** — Parsers, rules,
   transformers, validators, and other pipeline functions have OTel spans
   with appropriate `ce.span_kind` values. Either present in the client
   codebase already, or added by hand / by the future tracing agent.
3. **Parent-child relationships are natural** — OTel's context propagation
   handles this automatically. If `pipeline()` calls `parse()` which calls
   `validate()`, and all three are decorated, the span tree is
   `pipeline → parse → validate`. No manual wiring.

## DSPy + OpenInference

For DSPy clients, install `openinference-instrumentation-dspy` and call
`DSPyInstrumentor().instrument()` once at module import. That wraps DSPy's
`Predict` / `ChainOfThought` / `ReAct` etc. to emit OTel spans with
`llm.input_messages` / `input.value` / `output.value` attributes. The
converter's framework fallback promotes those into the standard
trace-entry shape automatically — no `@traceable` needed on DSPy modules
themselves, only on the surrounding deterministic code.

The DSPy span filter (`src/engine/telemetry/filters/ai_frameworks/dspy.py`)
drops the structural plumbing wrappers (`ChainOfThought.forward`,
`Predict.forward`, `ChatAdapter.__call__/format/parse`, `Predict(...)`)
because they carry duplicated inputs/outputs of the leaf `LM.__call__`.
LM calls, TOOL/RETRIEVER spans, and the user's own `dspy.Module.forward`
spans are retained.

**Re-parenting across dropped wrappers.** Dropping a span would otherwise
orphan its kept children, because OTel records only each span's *direct*
parent id. To avoid that, the filter records every dropped span's
`(id -> parent_id)` link into a per-row ancestry skip-map
(`telemetry.collect_row_spans` yields `(spans, drop_links)`), and
`converter.spans_to_trace_entries(..., drop_links=...)` walks that map to
re-parent each kept span to its *nearest surviving ancestor*. So a kept
`LM.__call__` whose `ChatAdapter`/`Predict` wrappers were dropped nests under
the surviving `JudgeModule.forward` rather than appearing as a detached root.
The skip-map holds ids only — the dropped spans' (duplicated) payloads are
still discarded. The parent-input dedupe pass runs against this resolved
ancestor too. The walk is cycle/depth-safe (malformed chains degrade to a
detached root).

## Span filters

`telemetry/filters/__init__.py` exposes `make_keep_predicate(drops)` which
composes the framework-agnostic `KEEP_KINDS` set (OpenInference standard:
`LLM`, `TOOL`, `RETRIEVER`, `RERANKER`, `EMBEDDING`, `AGENT`) with
per-framework `should_drop(span)` predicates. Anything in `KEEP_KINDS`
is always kept. Anything else is kept unless a drop predicate claims it.

The evaluator wires this in at `src/engine/evaluator/mounted/evaluate.py`:
`setup_tracer(span_filter=make_keep_predicate([dspy_filter.should_drop]))`.
New framework filters live alongside `dspy.py` and are appended to the
list without touching the core telemetry module.
