---
name: tracing
description: >
  Instrument a client repository with OpenTelemetry spans so CodeEvolver's
  IterationArchitect can inspect execution traces. Use when adding tracing
  to a new client project, when auditing existing tracing coverage, or
  when the architect's traces look thin (missing modules, no inputs/outputs).
compatibility: >
  Requires `opentelemetry-api` in the client repo. Standard OTel — no
  CodeEvolver imports leak into client code.
---

# Tracing skill

You are instrumenting a client codebase so CodeEvolver can read its execution
graph. The architect cannot improve what it cannot see, so missing or
unreadable traces directly hurt optimization quality.

This skill ships:

- `assets/traceable.py` — a copy-pasteable `@traceable` decorator that wraps
  a function in an OTel span and records inputs / output / errors using the
  `ce.*` attribute convention CodeEvolver's converter expects.
- `assets/file_exporter.py` — an **optional, request-only** local
  trace-dump processor. Buffers spans by `trace_id` and emits one JSON
  line per completed trace to `traces/traces.jsonl`. **Do not install
  this by default** — CodeEvolver already writes per-iteration trace
  directories via the orchestrator (`/traces/iteration_{N}_{suffix}/`). Only add it
  when the user explicitly asks for a local trace dump, e.g. when they
  want to inspect `@traceable` output running the program outside the
  orchestrator.
- `assets/otel_reference.md` — the OTel collection model, attribute table,
  and JSONL trace format. Read this when you need ground truth for what the
  trace entries look like or what the architect actually sees.
- `assets/examples/` — before/after examples for plain Python, an LM call,
  and a DSPy program.

## What CodeEvolver needs from your traces

Per execution row, the architect inspects a JSONL `trace[]` like this:

```json
{
  "signature_key": "FactChecker.parse_claim",
  "span_kind": "tool",
  "parent_key": "pipeline",
  "inputs": {"text": "tomatoes are vegetables"},
  "output": "[\"tomatoes are vegetables\"]"
}
```

Each entry comes from a single OTel span. The decorator's only job is to
emit those spans with the right attributes. You do **not** need to format
anything — the converter (`src/engine/telemetry/converter.py`) handles
the JSONL shape.

The architect leans hardest on:

1. **Span hierarchy.** Parent-child relationships expose the pipeline
   structure. OTel does this automatically via context propagation when
   one decorated function calls another.
2. **`ce.span_kind`.** Tells the architect what kind of step ran (LLM,
   tool, retriever, etc.). Drives the keep/drop filter — `LLM`, `TOOL`,
   `RETRIEVER`, `RERANKER`, `EMBEDDING`, `AGENT` are always kept.
3. **Inputs and output.** Without these the architect cannot reason about
   why a row failed. Stored in full by default — pass `max_attr_chars=N`
   per-decoration if a specific call site needs to be capped (e.g. PII or
   binary blobs). The architect is expected to grep / parse large payloads
   itself.

## Step 1: figure out if the repo is already instrumented

Before writing any decorators, check whether an existing instrumentation
path covers the LM calls. In order of preference:

| Situation                                | What to do                                                                                                                               |
| ---------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| Client uses **DSPy**                     | Add `openinference-instrumentation-dspy` to `requirements.txt`. Call `DSPyInstrumentor().instrument()` once at module import. LM calls are now spanned. |
| Client uses **LangChain / LlamaIndex / OpenAI SDK directly** | Install the matching `openinference-instrumentation-*` package. Same pattern.                                                           |
| Client already has its **own LangFuse/OTel pipeline** | Do nothing — `setup_tracer` probe-and-attaches to the existing global `TracerProvider`. Their pipeline keeps shipping; we mirror to our exporter. |
| None of the above                        | Move on to manual instrumentation with `@traceable`.                                                                                     |

Auto-instrumentation covers the LM calls but **not** the deterministic glue
around them (parsers, validators, rule engines, retrievers without a
supported framework). Even on a DSPy project you usually still want
`@traceable` on the deterministic functions that surround the DSPy modules.

## Step 2: drop in `@traceable` and decorate

1. Copy `assets/traceable.py` from this skill into the client repo. A
   reasonable home is `<your_pkg>/_tracing.py` or `<your_pkg>/tracing.py`.
   Do not rename the function.
2. Make sure `opentelemetry-api` is in `requirements.txt` (the sandbox
   already installs `opentelemetry-api` and `opentelemetry-sdk` into the
   client venv via `_install_telemetry_runtime()`, but listing it locally
   keeps the client repo runnable outside CodeEvolver).
3. Decorate the functions you want surfaced. Pass the right `span_kind`:

   ```python
   from your_pkg._tracing import traceable

   @traceable("tool")
   def parse_claim(text: str) -> list[str]:
       ...

   @traceable("retriever")
   def fetch_evidence(query: str) -> list[str]:
       ...

   @traceable("llm")
   def call_judge(prompt: str) -> str:
       ...
   ```

   Span kinds: `"llm"` for any call that hits a language model, `"tool"`
   for parsers / rules / transformations, `"retriever"` for data lookup,
   `"chain"` for top-level pipeline orchestrators, `"function"` for
   anything else worth seeing.

## Step 3: judgment — which functions to instrument

The goal is **pipeline-level visibility, not line-level profiling**. A
trace stack with 50 entries that each pass the same 100k-token prompt is
unreadable; a trace with 6–10 well-chosen spans tells the architect
exactly where signal lives.

**Instrument these:**

- Functions that call a language model (if not already auto-instrumented).
- Parsers, validators, rule engines, classifiers, post-processors.
- Retrieval / reranking / embedding calls.
- Any function whose correctness directly affects the metric.
- The top-level pipeline entry point — give it `@traceable("chain")` so
  the architect always sees a root span if the framework's auto-root
  doesn't cover it.

**Skip these:**

- Trivial helpers (`_normalize_whitespace`, `_to_lower`, formatters).
- Pass-through wrappers that immediately call another decorated function
  with the same args. They duplicate signal and bloat the trace.
- Tight inner loops (decorate the function that calls the loop, not the
  body).
- Anything CodeEvolver's existing framework filter already drops — see
  `src/engine/telemetry/filters/ai_frameworks/` for the current list.

When in doubt, instrument it. The architect can ignore noise; it cannot
recover silence. Keep the **pass-through** rule above firmly in mind so
you do not duplicate prompt payloads up the stack.

## Step 4: what NOT to do

- **No CodeEvolver imports.** The decorator uses only `opentelemetry`.
  Importing from `src.engine.*` or `codeevolver.*` couples the client
  repo to CodeEvolver internals and breaks portability.
- **Do not re-implement the converter.** Set the `ce.*` attributes
  through the decorator (or through raw `tracer.start_as_current_span(...)`
  if you need a custom case). The converter handles the rest.
- **Do not capture sensitive data without thinking.** Inputs/output land
  in trace JSONL on disk and are read back into the architect's prompts.
  If a function handles secrets, either skip it or override
  `max_attr_chars=0` so attributes are recorded as empty strings. Full
  capture is the default — opt out at the call site, not silently.
- **Do not replace the global `TracerProvider` with one that owns the
  lifecycle.** CodeEvolver's in-memory exporter has to attach to the
  global provider; if you replace it with one that ships spans elsewhere
  and tears down on its own (e.g. a remote OTLP exporter wired up at
  process start with no fallback), CodeEvolver can't see the spans. If
  the client already exports to LangFuse/Honeycomb, leave it —
  `setup_tracer` probe-and-attaches, it doesn't replace. **Additive
  span processors are fine** — that's exactly what
  `assets/file_exporter.py` is, and how `setup_tracer` itself works:
  attach to whatever provider exists, never own it.

## Step 4b: optional — local trace dump (only if requested)

**Do not add this unless the user explicitly asks for it.** Inside CodeEvolver, the orchestrator already collects spans via `setup_tracer` + `InMemorySpanExporter` and writes them to `/traces/iteration_{N}_{suffix}/rows/<example_id>/` (one `row.json` plus one `span_NNN.json` per span). The architect reads from there. Installing a second file exporter inside the client repo is redundant in that path.

Where it *is* useful: when the user wants to run their program **outside** the orchestrator (a local script, a FastAPI dev server, a notebook) and inspect what `@traceable` is producing without standing up a real observability backend. In a plain Python process with no tracer wired up, OTel installs a default no-op provider and spans evaporate — you see nothing.

If asked, copy `assets/file_exporter.py` into the client repo (e.g. as `<your_pkg>/_tracing_export.py`) and call `install_file_exporter()` once at process startup, after any framework-specific tracer setup (`configure_azure_monitor`, `DSPyInstrumentor().instrument()`, etc.):

```python
from your_pkg._tracing_export import install_file_exporter
install_file_exporter()                       # default: traces/traces.jsonl
install_file_exporter("traces/my_run.jsonl")  # custom path
```

The output is one JSON line per completed trace (one example per line), with all spans for that trace grouped, sorted by start time, and linked via `parent_id`:

```json
{"trace_id": "...", "root_name": "pipeline", "duration_ms": 4521.3, "spans": [
  {"name": "pipeline", "span_id": "abc", "parent_id": null, "attributes": {"ce.span_kind": "chain", ...}, ...},
  {"name": "parse_claim", "span_id": "def", "parent_id": "abc", "attributes": {"ce.span_kind": "tool", ...}, ...},
  ...
]}
```

This makes the example→spans grouping obvious and matches the flat-list-with-`parent_id` shape CodeEvolver's converter produces in `trace[]` entries (per-row), so the file is portable across repos.

Operational notes:

- The exporter truncates the file on the first `install_file_exporter()` call per process, so each fresh run overwrites the previous dump.
- Idempotent — calling it twice in the same process is a no-op.
- Add `traces/` to `.gitignore` in the client repo.
- **Never enable in production.** The file grows per request and contains full prompts.
- Additive only: it does not replace the global provider, so CodeEvolver's in-memory exporter and any framework auto-instrumentation keep working alongside.

## Step 5: verify

1. Run a single-row evaluation:
   ```bash
   modal run modal_test_app.py::<test_entrypoint>
   ```
   or, locally:
   ```bash
   python -m src.engine.evaluator.mounted.evaluate \
       --program <pkg.mod:Class> --metric <pkg.mod:fn> \
       --trainset <path> --rows 0
   ```
2. Open the trace directory: `/traces/iteration_0_child/` (or whatever
   suffix the eval used). Each `rows/<example_id>/row.json` lists the
   row's spans in its `spans` table, one `span_NNN.json` per decorated
   call; `provenance.json` names the eval (id / sha / tree) that wrote
   the directory — it is rewritten by every eval with that suffix.
3. Confirm:
   - Span names match `fn.__qualname__` (or your `name=` override).
   - `span_kind` is the value you passed.
   - `inputs` has one key per declared parameter.
   - `output` is set on success, `error` is set on failure.
   - `parent_key` reflects the call hierarchy (None for the root span).
4. If a span is missing, check that the function actually got called and
   that it isn't an inner pass-through that the framework filter drops.

## Reference

- `assets/otel_reference.md` — the OTel collection model, attribute table,
  example JSONL, and DSPy auto-instrumentation note.
- `assets/file_exporter.py` — optional grouped-by-trace-id file exporter
  for local inspection (see Step 4b). Only installed if the user asks.
- `assets/examples/plain_function.py` — minimal sync function example.
- `assets/examples/llm_call.py` — async LM call example.
- `assets/examples/dspy_program.py` — DSPy + OpenInference setup.
- Engine-side internals (the bits this skill doesn't touch):
  - `src/engine/telemetry/__init__.py` — `setup_tracer`, `drain_spans`.
  - `src/engine/telemetry/converter.py` — span → trace entry mapping.
  - `src/engine/telemetry/filters/` — keep/drop predicates.
