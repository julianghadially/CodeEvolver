"""Copyright © 2026 440 Labs LLC

Generic mounted evaluator.

This script runs inside the client sandbox (mounted at `/app/evaluator/mounted/`)
and is dispatched by `sandbox/mounted/master_script.py` via the `"evaluate"`
command. It is framework-agnostic: the client supplies a program factory
(`pkg.mod:factory_fn`) and a list of metric functions, and this script runs
each row through the program, scores it, writes trace files, and returns the
result.

Trace capture uses OpenTelemetry: each row's program call is wrapped in a
`pipeline` root span, and any spans the client (or a framework) emits underneath
are drained into that row's `trace[]`. The lifecycle helpers in
`src.engine.telemetry` attach our SimpleSpanProcessor to the existing global
TracerProvider when one is already set, so a client's own telemetry pipeline
(LangFuse, Honeycomb, etc.) keeps working alongside ours.
"""

from __future__ import annotations

import concurrent.futures
import contextvars
import csv
import importlib
import inspect
import json
import os
import random
import sys
import threading
import time
import traceback
from typing import Any, Callable

from src.engine.schemas.example import Example
from src.engine.telemetry import (
    OTEL_AVAILABLE,
    collect_row_spans,
    setup_tracer,
    shutdown_tracer,
)
from src.engine.paths import trace_max_line_chars
from src.engine.telemetry.converter import spans_to_trace_entries
from src.engine.telemetry.trace_layout import (
    eval_trace_dir,
    is_holdout_suffix,
    write_eval_trace,
)
from src.engine.telemetry.filters import make_keep_predicate
from src.engine.telemetry.filters.ai_frameworks import dspy as dspy_filter

if OTEL_AVAILABLE:
    from opentelemetry import trace as _otel_trace

# Resolved per-call by `_traces_dir()` so the launcher (local vs default) can
# point at `/traces` or `~/.codeevolver/traces` without code changes here.
from src.engine.paths import traces_dir as _traces_dir

# --- provider-glitch row retry ------------------------------------------
#
# A row that failed because the PROVIDER glitched is not a measurement. GMI
# intermittently answers one row with
#
#   APIError: litellm.APIError: APIError: OpenAIException - Error code: 402 -
#   {'error': 'Insufficient balance', 'reason': 'model_access_denied'}
#
# and then serves the very next row normally -- a real exhausted balance could
# not do that, so it is a provider-side fault, not a billing state. litellm
# already retries 408/409/429/5xx (`litellm.utils._should_retry`) but
# deliberately excludes 402 as a permanent client error, so these fall all the
# way through to a scored 0.0 and silently depress the aggregate.
#
# Retries are NOT charged to the eval-row budget, by construction rather than
# by a special case: the budget counts one entry per ROW
# (`run_eval_tool` increments by `len(trace_meta)`), and retrying inside
# `_run_row` still produces exactly one entry. An arm is therefore never
# penalised for the provider's flakiness.
#
# Backoff is 3s then 60s. Rows on this benchmark take ~6 minutes, so even the
# 60s wait is a small fraction of one row, and it is only ever paid on the
# ~0.2% of rows that fail. Jitter matters because ~25 rows run concurrently:
# a provider blip tends to hit several at once, and un-jittered retries would
# all land together and re-trigger the same condition.
RETRY_BACKOFF_SECONDS = (3.0, 60.0)
RETRY_JITTER = 0.25
# Ceiling on the FRACTION of an eval's rows that may spend retries (a row
# claims its slot once, then may use the whole schedule), so a sustained
# provider outage degrades to today's behaviour instead of multiplying the
# wall-clock of every row. Observed 402 rate is ~0.2%, so 5% is ~25x head-
# room while still bounding a real outage.
RETRY_ROW_FRACTION_CAP = 0.05

# Two gates, both required, and default-deny. A retry must never fire for the
# evolved program's OWN exceptions: the metric exists to measure whether that
# program works, and silently re-running its crashes would score the agent on
# code that does not run. The first gate demands the exception look like it
# came from a provider client, the second demands a known-transient signature.
_PROVIDER_ERROR_TYPES = (
    "litellm",
    "apierror",
    "apistatuserror",
    "apiconnectionerror",
    "serviceunavailable",
    "internalservererror",
    "ratelimiterror",
    "timeout",
)
_RETRYABLE_SIGNATURES = (
    "error code: 402",
    "insufficient balance",
    "model_access_denied",
    "connection reset",
    "connection aborted",
    "remote end closed connection",
    "temporarily unavailable",
)


def _is_retryable(error_summary: str | None) -> bool:
    """True only for a provider-side fault we expect to clear on a re-run.

    Conservative on purpose: an unrecognised error is NOT retried. If a
    provider changes its wording the failures stay visible as before (and the
    per-eval retry summary logs nothing), which is the safe direction to fail.
    """
    if not error_summary:
        return False
    s = error_summary.lower()
    if not any(t in s for t in _PROVIDER_ERROR_TYPES):
        return False
    return any(sig in s for sig in _RETRYABLE_SIGNATURES)


# Hard ceiling on row-evaluation concurrency. Guards a misconfigured
# `num_threads` from exhausting file descriptors / outbound connections.
# The /optimize request's `num_threads` is clamped to this in `handle`.
MAX_EVAL_THREADS = 128


def _resolve(dotted_path: str) -> Any:
    """Resolve a dotted `pkg.mod:attr` path to the attribute."""
    if ":" in dotted_path:
        module_name, _, attr = dotted_path.partition(":")
    else:
        module_name, _, attr = dotted_path.rpartition(".")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def _resolve_program(program_path: str) -> Callable[..., Any]:
    """Resolve a `pkg.mod:Name` path to the per-row callable.

    The contract is deliberately minimal — no sniffing, no fallbacks:
      - If the target is a class, instantiate it once (`Cls()`); the
        instance must define `__call__` with explicit named parameters
        matching dataset fields. All setup (framework config, clients,
        pipeline assembly) belongs in `__init__` with defaults.
      - Otherwise the target must already be callable (a plain function).
        It is used directly and is called per row with row-field kwargs.

    Thread-safety: when `num_threads > 1` this single resolved callable is
    invoked concurrently across rows. Its `__call__` (and the metric fns) must
    be thread-safe — keep per-request state local, not on `self`. DSPy modules
    satisfy this. Spans the client emits on threads it spawns itself are not
    captured per-row (the row-buffer ContextVar isn't propagated into them).
    """
    target = _resolve(program_path)
    if inspect.isclass(target):
        return target()
    if not callable(target):
        raise TypeError(
            f"program {program_path!r} resolved to {type(target).__name__}, "
            f"expected a class or callable"
        )
    return target


def _add_workspace_to_import_path(workspace: str) -> None:
    """Make client modules importable, including `src.*` packages.

    The engine itself is imported as a top-level `src` package. Client repos
    can also expose their code under `src.*`, so appending the workspace to
    `sys.path` is not enough after Python has already loaded engine `src`.
    Extending the loaded package path lets imports like `src.factchecker`
    resolve without moving the client workspace ahead of engine modules.
    """
    if workspace and workspace not in sys.path:
        sys.path.append(workspace)

    client_src = os.path.join(workspace, "src") if workspace else ""
    if not client_src or not os.path.isdir(client_src):
        return
    try:
        import src as src_package
    except ImportError:
        return
    package_path = getattr(src_package, "__path__", None)
    if package_path is not None and client_src not in package_path:
        package_path.append(client_src)


def _load_dataset(workspace: str, extra: dict, suffix: str) -> list[dict]:
    """Load the right dataset for the given eval suffix.

    The host sends both paths in `extra`; we pick one based on the suffix:

    - `"holdout"`, `"holdout_<k>"` → held-out test set file (finalization
      draws the test set several times; sample k>1 uses `holdout_<k>`)
    - `"comparison"`              → trainset file (the comparison set is a
      fixed subsample of the trainset; the host sends comparison row ids)
    - anything else (`"parent"`, `"child"`, `"single"`, ...) → trainset

    Empty dataset falls through as `[]` so a smoke-test caller can still
    invoke the evaluator with an explicit `row_ids=[]` batch.
    """
    if not isinstance(extra, dict):
        return []
    if is_holdout_suffix(suffix):
        rel_path = extra.get("testset_path")
    else:
        rel_path = extra.get("trainset_path")
    if not rel_path:
        return []
    full_path = os.path.join(workspace, rel_path)
    if full_path.endswith(".jsonl"):
        with open(full_path) as f:
            return [json.loads(line) for line in f if line.strip()]
    if full_path.endswith(".csv"):
        # Mirror `dataset_config._resolve_split`: each row is a dict keyed by
        # the CSV header, values left as strings.
        with open(full_path, newline="") as f:
            return list(csv.DictReader(f))
    with open(full_path) as f:
        loaded = json.load(f)
    if not isinstance(loaded, list):
        raise ValueError(f"dataset at {full_path} must be a list of dicts")
    return loaded


def _introspect_params(fn: Any) -> list[str] | None:
    """Return the callable's concrete named parameters.

    For a class instance, `inspect.signature` reads `__call__` naturally.
    We deliberately do not sniff framework-specific methods (`.forward`
    etc.) — if a client wraps a framework module, their wrapper's
    `__call__` must declare explicit named parameters.

    Returns `None` when the signature cannot be introspected or exposes
    no concrete named params (e.g. `def f(**kwargs)`); the caller falls
    through to passing every row field.
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return None

    names: list[str] = []
    for name, p in sig.parameters.items():
        if name == "self":
            continue
        if p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.append(name)
    return names or None


def _select_row_kwargs(
    row: dict,
    param_names: list[str] | None,
    exclude: tuple[str, ...] = (),
) -> dict:
    """Pick row fields that match a callable's named parameters.

    If `param_names` is None, the callable's signature was unreachable
    or exposed no concrete params — pass every row field (minus the
    excluded keys). Otherwise pass only the intersection of row and
    named params, which keeps non-input columns like `label` from
    leaking into the program call.
    """
    if param_names is None:
        return {k: v for k, v in row.items() if k not in exclude}
    return {k: row[k] for k in param_names if k in row and k not in exclude}


def _resolve_program_param_names(
    program: Any, input_keys: list[str] | None
) -> list[str] | None:
    """Decide which row keys gate the program call.

    `input_keys` is the contract that prevents non-input columns (id, label, answer) from
    leaking into the program call when the program's `__call__` accepts `**kwargs` and signature
    introspection returns None. Falls back to `_introspect_params` for
    clients that didn't declare them.
    """
    if input_keys:
        return list(input_keys)
    return _introspect_params(program)


def _parse_metric_result(result: Any) -> tuple[float, str | None]:
    """Normalize a metric return value into `(score, feedback)`.

    Accepted shapes:
      - scalar `float` / `int` / `bool`  → score, no feedback
      - dict with `"score"` / `"feedback"` keys
      - any object with `.score` / `.feedback` attributes (e.g. dspy.Prediction)

    A missing / unparseable score is treated as 0.0.
    """
    if isinstance(result, (int, float)):  # bool is an int subclass; fine
        return float(result), None
    if isinstance(result, dict):
        raw_score = result.get("score")
        feedback = result.get("feedback")
    else:
        raw_score = getattr(result, "score", None)
        feedback = getattr(result, "feedback", None)
    if raw_score is None:
        return 0.0, feedback if isinstance(feedback, str) else None
    try:
        score = float(raw_score)
    except (TypeError, ValueError):
        score = 0.0
    return score, feedback if isinstance(feedback, str) else None


def _truncate(s: str, n: int) -> str:
    """Cap a string at `n` chars, marking truncation."""
    if len(s) <= n:
        return s
    return s[:n] + "...(truncated)"


def _output_to_trace_string(value: Any) -> str:
    """Return a string representation for trace files."""
    safe_value = _to_jsonable(value)
    if isinstance(safe_value, str):
        return safe_value
    try:
        return json.dumps(safe_value)
    except (TypeError, ValueError):
        return repr(safe_value)


def _write_traces(
    iteration_idx: int,
    suffix: str,
    entries: list[dict],
) -> tuple[str, str, list[dict]]:
    """Write the eval's trace directory. Returns (trace_dir, index_path, meta).

    Layout (`telemetry/trace_layout.py`): `iteration_{N}_{suffix}/index.jsonl`
    plus one `rows/{example_id}/row.json` and `span_{k}.json` per span, all
    pretty-printed — and chunked to OpenCode's line cap when
    `paths.trace_max_line_chars()` is set for the run. Creates the traces
    root (configurable via `CODEEVOLVER_TRACES_DIR`) if it does not exist.
    """
    traces_root = _traces_dir()
    os.makedirs(traces_root, exist_ok=True)
    trace_dir = eval_trace_dir(traces_root, iteration_idx, suffix)
    index_path, meta = write_eval_trace(
        trace_dir, entries, max_line_chars=trace_max_line_chars()
    )
    return trace_dir, index_path, meta


def handle(cmd: dict, workspace: str) -> dict:
    """Master-script entry point.

    Input fields on `cmd`:
        program_factory, metric_fns, metric_names,
        primary_metric, row_ids, iteration_idx, suffix, capture_traces, extra

    Output:
        {
            "success": True/False,
            "error": str | None,
            "scores_by_metric": {name: aggregate_score},
            "row_ids": [...],
            "trace_dir": "/traces/iteration_{N}_{suffix}",
            "trace_index_file": "/traces/iteration_{N}_{suffix}/index.jsonl",
            "trace_meta": [...],
        }
    """
    try:
        program_factory = cmd["program_factory"]
        metric_fns_paths: list[str] = cmd["metric_fns"]
        metric_names: list[str] = cmd["metric_names"]
        primary_metric: str = cmd["primary_metric"]
        row_ids: list[int] | None = cmd.get("row_ids")
        iteration_idx: int = int(cmd.get("iteration_idx", 0))
        suffix: str = cmd.get("suffix", "child")
        extra: dict = cmd.get("extra") or {}
        capture_traces: bool = bool(cmd.get("capture_traces", True))
        input_keys: list[str] | None = cmd.get("input_keys")
        num_threads: int = int(cmd.get("num_threads", 1) or 1)

        # Keep engine imports ahead of client code, but explicitly merge the
        # client's `src/` tree into the already-loaded engine `src` package.
        _add_workspace_to_import_path(workspace)

        program = _resolve_program(program_factory)
        metric_fns = [_resolve(p) for p in metric_fns_paths]

        dataset = _load_dataset(workspace, extra, suffix)
        if row_ids is None:
            row_ids = list(range(len(dataset)))

        # At most one worker per row; never below 1; clamp to the hard ceiling.
        # `effective_threads <= 1` runs the sequential path (identical to the
        # pre-parallel evaluator), so default-1 configs are byte-for-byte
        # unchanged and OTel context stays on the main thread.
        effective_threads = max(1, min(num_threads, len(row_ids), MAX_EVAL_THREADS))

        # Auto-map row fields to the program signature. `input_keys`, when
        # provided, gates the program call authoritatively — this keeps
        # `id` / `label` / `answer` from leaking into programs whose
        # `__call__` accepts `**kwargs` (mirrors v1's contract). Metrics do
        # not auto-map: each metric is called `fn(output=..., example=...)`
        # with the full row wrapped as an `Example`.
        prog_param_names = _resolve_program_param_names(program, input_keys)

        per_row_scores: dict[str, list[float]] = {name: [] for name in metric_names}
        entries: list[dict] = []
        start = time.time()

        # Shared across pool workers: the retry allowance for this eval, and
        # the tally reported afterwards. `_retry_lock` guards both.
        retry_budget = int(len(row_ids) * RETRY_ROW_FRACTION_CAP) + 1
        retry_state = {"remaining": retry_budget, "rows": 0, "attempts": 0,
                       "recovered": 0, "exhausted": 0}
        _retry_lock = threading.Lock()

        def _claim_retry_slot() -> bool:
            """Reserve this ROW's place in the eval's shared retry allowance.

            Claimed once per row, not once per attempt: the cap is a fraction
            of ROWS (see RETRY_ROW_FRACTION_CAP), so a row that gets a slot may
            then use the whole backoff schedule.
            """
            with _retry_lock:
                if retry_state["remaining"] <= 0:
                    return False
                retry_state["remaining"] -= 1
                return True

        def _note_attempt() -> None:
            with _retry_lock:
                retry_state["attempts"] += 1

        def _note_row(retries: int, recovered: bool) -> None:
            with _retry_lock:
                retry_state["rows"] += 1
                if recovered:
                    retry_state["recovered"] += 1
                else:
                    retry_state["exhausted"] += 1

        # OTel setup happens AFTER program import + instantiation so any
        # provider the client wired at import time is the one our probe sees.
        tracing_on = capture_traces and OTEL_AVAILABLE
        tracer = None
        if tracing_on:
            setup_tracer(
                span_filter=make_keep_predicate([dspy_filter.should_drop])
            )
            tracer = _otel_trace.get_tracer("codeevolver.evaluator")

        # --- per-row work, factored so it runs identically on the main thread
        # (sequential path) or on a pool worker (parallel path). Each helper
        # touches only loop-invariant resolved objects + its own locals; no
        # shared mutable state, so it is safe to call concurrently. ---

        def _call_program(row: dict, root: Any) -> tuple[Any, bool, str | None]:
            """Run the client program for one row.

            Returns (output, failed, error_summary). Records inputs/output/error
            on the row's `pipeline` root span when tracing is on.
            """
            failed = False
            error_summary: str | None = None
            output: Any = None
            try:
                prog_kwargs = _select_row_kwargs(row, prog_param_names)
                if root is not None:
                    for k, v in prog_kwargs.items():
                        root.set_attribute(f"ce.inputs.{k}", _truncate(str(v), 2000))
                output = program(**prog_kwargs)
                if root is not None:
                    root.set_attribute(
                        "ce.output",
                        _truncate(_output_to_trace_string(output), 2000),
                    )
            except Exception as exc:  # noqa: BLE001
                failed = True
                error_summary = f"{type(exc).__name__}: {exc}"
                if root is not None:
                    root.set_attribute("ce.error", error_summary)
                    root.record_exception(exc)
            return output, failed, error_summary

        def _run_metrics(
            row: dict,
            output: Any,
            failed: bool,
            error_summary: str | None,
            row_scores: dict[str, float],
            row_feedback: dict[str, str],
        ) -> tuple[bool, str | None]:
            """Score one row across all metrics, filling row_scores/row_feedback.

            Returns the (possibly updated) (failed, error_summary).
            """
            for name, fn in zip(metric_names, metric_fns):
                if failed:
                    score = 0.0
                    feedback = None
                else:
                    try:
                        raw = fn(output=output, example=Example(row))
                        score, feedback = _parse_metric_result(raw)
                    except Exception as exc:  # noqa: BLE001
                        failed = True
                        error_summary = error_summary or f"metric {name}: {exc}"
                        score = 0.0
                        feedback = None
                row_scores[name] = score
                if feedback:
                    row_feedback[name] = feedback
            return failed, error_summary

        def _run_row_once(idx: int, rid: int) -> tuple[int, dict, dict[str, float]]:
            """Evaluate one row in isolation. Returns (idx, entry, row_scores).

            `collect_row_spans()` scopes a per-row span buffer in this context;
            the `pipeline` root span is nested inside it so its `on_end` fires
            before the buffer is read. Run under `copy_context().run` on a pool
            worker, this keeps each row's spans (and DSPy's settings) isolated.
            Never raises: an unexpected error synthesizes a failed entry so one
            bad row cannot abort the whole batch.
            """
            try:
                row = dataset[rid] if 0 <= rid < len(dataset) else {}

                if tracing_on:
                    with collect_row_spans() as (buf, drop_links):
                        with tracer.start_as_current_span("pipeline") as root:
                            root.set_attribute("ce.span_kind", "chain")
                            root.set_attribute("example_id", rid)
                            output, failed, error_summary = _call_program(row, root)
                        row_spans = list(buf)
                        row_drop_links = dict(drop_links)
                else:
                    output, failed, error_summary = _call_program(row, None)
                    row_spans = []
                    row_drop_links = {}

                trace_entries = spans_to_trace_entries(
                    row_spans, drop_links=row_drop_links
                )

                row_scores: dict[str, float] = {}
                row_feedback: dict[str, str] = {}
                failed, error_summary = _run_metrics(
                    row, output, failed, error_summary, row_scores, row_feedback
                )

                entry = {
                    "example_id": rid,
                    "score": row_scores.get(primary_metric, 0.0),
                    "scores_by_metric": row_scores,
                    "failed": failed,
                    "feedback": row_feedback.get(primary_metric),
                    "feedback_by_metric": row_feedback or None,
                    "error_summary": error_summary,
                    "input": row,
                    "output": None if failed else _to_jsonable(output),
                    "trace": trace_entries,
                }
                return idx, entry, row_scores
            except Exception as exc:  # noqa: BLE001
                row = dataset[rid] if 0 <= rid < len(dataset) else {}
                entry = {
                    "example_id": rid,
                    "score": 0.0,
                    "scores_by_metric": {},
                    "failed": True,
                    "feedback": None,
                    "feedback_by_metric": None,
                    "error_summary": f"{type(exc).__name__}: {exc}",
                    "input": row,
                    "output": None,
                    "trace": [],
                }
                return idx, entry, {}

        def _run_row(idx: int, rid: int) -> tuple[int, dict, dict[str, float]]:
            """`_run_row_once`, re-run when the PROVIDER (not the program) failed.

            Produces exactly one entry per row however many attempts it took,
            which is what keeps retries off the eval-row budget. `retry_count`
            is stamped on the entry so an eval that fought the provider is
            visible rather than silently slower.
            """
            attempt = 0
            holds_slot = False
            while True:
                idx_, entry, row_scores = _run_row_once(idx, rid)
                if not entry.get("failed"):
                    break
                if not _is_retryable(entry.get("error_summary")):
                    break
                if attempt >= len(RETRY_BACKOFF_SECONDS):
                    break
                if not holds_slot:
                    if not _claim_retry_slot():
                        break
                    holds_slot = True
                delay = RETRY_BACKOFF_SECONDS[attempt]
                time.sleep(delay * (1.0 + random.uniform(-RETRY_JITTER, RETRY_JITTER)))
                attempt += 1
                _note_attempt()
            if attempt:
                entry["retry_count"] = attempt
                _note_row(attempt, recovered=not entry.get("failed"))
            return idx_, entry, row_scores

        # Dispatch: results land positionally so completion order is irrelevant;
        # `entries` / `per_row_scores` are assembled in row order on the main
        # thread after all rows finish, keeping outputs fully deterministic.
        results: list[tuple[dict, dict[str, float]] | None] = [None] * len(row_ids)
        try:
            if effective_threads <= 1:
                for idx, rid in enumerate(row_ids):
                    _, entry, row_scores = _run_row(idx, rid)
                    results[idx] = (entry, row_scores)
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=effective_threads
                ) as pool:
                    # copy_context() is snapshotted on the MAIN thread so each
                    # worker inherits the OTel + DSPy contextvars set up here.
                    futures = [
                        pool.submit(
                            contextvars.copy_context().run, _run_row, idx, rid
                        )
                        for idx, rid in enumerate(row_ids)
                    ]
                    for fut in concurrent.futures.as_completed(futures):
                        idx, entry, row_scores = fut.result()
                        results[idx] = (entry, row_scores)
        finally:
            if tracing_on:
                shutdown_tracer()

        for idx, rid in enumerate(row_ids):
            entry, row_scores = results[idx]  # type: ignore[misc]
            entries.append(entry)
            for name in metric_names:
                per_row_scores[name].append(row_scores.get(name, 0.0))

        trace_dir, trace_index_file, trace_meta = _write_traces(
            iteration_idx, suffix, entries
        )

        # Aggregate: arithmetic mean of per-row scores. 0.0 when no rows were run.
        aggregate = {
            name: (sum(vals) / len(vals) if vals else 0.0)
            for name, vals in per_row_scores.items()
        }

        return {
            "success": True,
            "error": None,
            "scores_by_metric": aggregate,
            "row_ids": row_ids,
            "trace_dir": trace_dir,
            "trace_index_file": trace_index_file,
            "trace_meta": trace_meta,
            "elapsed_seconds": time.time() - start,
            # Absent-when-quiet by design: `None` means the provider behaved,
            # so a reader never has to distinguish "no retries" from "field not
            # written by an older evaluator".
            "retry_summary": (
                dict(retry_state) if retry_state["rows"] else None
            ),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }


def _to_jsonable(
    value: Any,
    *,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    """Recursively convert a value into a JSON-serializable structure.

    Extracts custom objects' fields (via ``toDict``/``to_dict``/``__dict__``)
    into nested dicts so traces carry structured output rather than an opaque
    ``repr`` string. Cycle-safe and depth-limited.
    """
    if seen is None:
        seen = set()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if depth > 12:
        return repr(value)

    obj_id = id(value)
    if obj_id in seen:
        return "<cycle>"
    seen.add(obj_id)

    if isinstance(value, dict):
        return {
            str(k): _to_jsonable(v, depth=depth + 1, seen=seen)
            for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [
            _to_jsonable(item, depth=depth + 1, seen=seen)
            for item in value
        ]
    if isinstance(value, (set, frozenset)):
        return [
            _to_jsonable(item, depth=depth + 1, seen=seen)
            for item in sorted(value, key=repr)
        ]

    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        pass

    mapping = _object_mapping(value)
    if mapping:
        return _to_jsonable(mapping, depth=depth + 1, seen=seen)
    return repr(value)


def _object_mapping(value: Any) -> dict[str, Any] | None:
    to_dict = getattr(value, "toDict", None) or getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            mapped = to_dict()
        except Exception:  # noqa: BLE001
            mapped = None
        if isinstance(mapped, dict):
            return mapped

    raw_attrs = getattr(value, "__dict__", None)
    if not isinstance(raw_attrs, dict):
        return None
    attrs = {
        key: attr_value
        for key, attr_value in raw_attrs.items()
        if isinstance(key, str) and not key.startswith("_")
    }
    return attrs or None
