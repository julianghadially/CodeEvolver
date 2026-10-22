"""Copyright © 2026 440 Labs LLC

On-disk layout of the per-iteration eval traces (one directory per eval)::

    {traces_dir}/iteration_{N}_{suffix}/
      index.jsonl                      one short line per example: example_id,
                                       score, failed, feedback, error_summary,
                                       row_file  (also returned as trace_meta)
      index_pretty_print.txt           the same per-row records as readable
                                       text: first PRETTY_MAX_ROWS rows,
                                       every line capped at
                                       PRETTY_MAX_LINE_CHARS chars
      rows/{example_id}/row.json       everything about the example EXCEPT the
                                       span bodies: scores, feedback, input,
                                       output, and a `spans` table
                                       (k, signature_key, span_kind,
                                       parent_key, error, file, ...)
      rows/{example_id}/span_{k:03d}.json   one converter entry per span
      provenance.json                  which eval produced this directory:
                                       eval_id, sha_at_eval, tree, committed,
                                       branch, eval_sample, suffix, score, ts
                                       (written by `run_eval` AFTER the eval —
                                       the evaluator subprocess has no git)

A directory is REWRITTEN by every eval with the same suffix (the `child`
directory after each post-change eval, `comparison` after each comparison
run), so a reader looking at an earlier iteration's directory must check
`provenance.json` to learn which tree it describes — the directory may hold
the traces of a change that was later reverted.

Every file is pretty-printed through `trace_json.dump_trace_json`, so a
reader can `Read` a whole row or span in one call on either coding harness
(see that module for the OpenCode line-cap chunking). The former single-line
`iteration_{N}_{suffix}.jsonl` — 165–525 KB per line — could not be read by
either harness's file tool.

Holdout evals use the suffixes `holdout`, `holdout_2`, … so every held-out
trace directory contains the fence fragment `_holdout` (see
`HOLDOUT_DENY_FRAGMENT`); the substring fences on both harnesses match it.
`_test` was rejected as a fragment because it would fence client paths such
as `src/my_test/`.

This module is stdlib-only and imported by the evaluator subprocess (client
venv, engine mounted) as well as the host.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Iterator
from typing import Any

from .trace_json import dump_trace_json, join_chunked_strings

INDEX_FILE = "index.jsonl"
PRETTY_INDEX_FILE = "index_pretty_print.txt"
# Caps for the pretty-printed index: same row cap the inline trace_meta used
# before it moved out of the run_eval response, plus a per-line char cap so
# one giant feedback blob cannot make the file unreadable.
PRETTY_MAX_ROWS = 40
PRETTY_MAX_LINE_CHARS = 2000
ROWS_DIR = "rows"
ROW_FILE = "row.json"
PROVENANCE_FILE = "provenance.json"

logger = logging.getLogger(__name__)

HOLDOUT_SUFFIX = "holdout"
# Substring fence fragment: every holdout trace directory name contains it.
HOLDOUT_DENY_FRAGMENT = "_holdout"


def holdout_suffix(sample_idx: int) -> str:
    """Trace suffix for the k-th held-out test-set draw: `holdout`, `holdout_2`, …"""
    if sample_idx < 0:
        raise ValueError(f"sample_idx must be >= 0, got {sample_idx}")
    return HOLDOUT_SUFFIX if sample_idx == 0 else f"{HOLDOUT_SUFFIX}_{sample_idx + 1}"


def is_holdout_suffix(suffix: str) -> bool:
    return suffix == HOLDOUT_SUFFIX or suffix.startswith(HOLDOUT_SUFFIX + "_")


def eval_trace_dir_name(iteration_idx: int, suffix: str) -> str:
    return f"iteration_{iteration_idx}_{suffix}"


def eval_trace_dir(traces_root: str, iteration_idx: int, suffix: str) -> str:
    return os.path.join(traces_root, eval_trace_dir_name(iteration_idx, suffix))


def index_path(trace_dir: str) -> str:
    return os.path.join(trace_dir, INDEX_FILE)


def row_dir_name(example_id: Any, position: int) -> str:
    """`rows/<example_id>`; falls back to the line position when the id is None."""
    return str(example_id) if example_id is not None else f"line_{position}"


def row_dir(trace_dir: str, example_id: Any, position: int = 0) -> str:
    return os.path.join(trace_dir, ROWS_DIR, row_dir_name(example_id, position))


def row_file(trace_dir: str, example_id: Any, position: int = 0) -> str:
    return os.path.join(row_dir(trace_dir, example_id, position), ROW_FILE)


def span_file_name(k: int) -> str:
    return f"span_{k:03d}.json"


def _index_record(entry: dict, position: int) -> dict[str, Any]:
    example_id = entry.get("example_id")
    return {
        "example_id": example_id,
        "score": entry.get("score"),
        "failed": entry.get("failed", False),
        "feedback": entry.get("feedback"),
        "error_summary": entry.get("error_summary"),
        "row_file": f"{ROWS_DIR}/{row_dir_name(example_id, position)}/{ROW_FILE}",
    }


def pretty_index_path(trace_dir: str) -> str:
    return os.path.join(trace_dir, PRETTY_INDEX_FILE)


def _pretty_index_text(records: list[dict]) -> str:
    """Human/agent-readable rendering of the index records."""
    shown = records[:PRETTY_MAX_ROWS]
    lines = [
        f"Per-row eval feedback: {len(shown)} of {len(records)} rows "
        f"(full data: {INDEX_FILE}; per-row detail: {ROWS_DIR}/)",
    ]
    for rec in shown:
        status = "FAILED" if rec.get("failed") else "ok"
        lines.append("")
        lines.append(
            f"--- example_id={rec.get('example_id')}  "
            f"score={rec.get('score')}  [{status}]"
        )
        for label in ("feedback", "error_summary"):
            value = rec.get(label)
            if value in (None, ""):
                continue
            lines.append(f"{label}:")
            for raw_line in str(value).splitlines():
                lines.append(f"  {raw_line}")
    capped = [
        line if len(line) <= PRETTY_MAX_LINE_CHARS
        else line[: PRETTY_MAX_LINE_CHARS - 1] + "\u2026"
        for line in lines
    ]
    return "\n".join(capped) + "\n"


def _span_summary(k: int, span: Any) -> dict[str, Any]:
    if not isinstance(span, dict):
        return {"k": k, "file": span_file_name(k)}
    return {
        "k": k,
        "signature_key": span.get("signature_key"),
        "span_kind": span.get("span_kind"),
        "parent_key": span.get("parent_key"),
        "error": span.get("error"),
        "file": span_file_name(k),
        "inputs_chars": len(json.dumps(span.get("inputs"), default=str)),
        "output_chars": len(json.dumps(span.get("output"), default=str)),
    }


def write_eval_trace(
    trace_dir: str,
    entries: list[dict],
    *,
    max_line_chars: int | None,
) -> tuple[str, list[dict]]:
    """Write `entries` (the evaluator's per-row dicts) as a trace directory.

    Builds `{trace_dir}.tmp` and atomically replaces `trace_dir`, so a
    concurrent reader never sees a half-written eval (the `child` directory
    is rewritten on every post-change eval). Returns
    `(index_path, index_records)`; the records are what `run_eval` returns
    inline as `trace_meta`.
    """
    tmp_dir = trace_dir.rstrip(os.sep) + ".tmp"
    shutil.rmtree(tmp_dir, ignore_errors=True)
    os.makedirs(os.path.join(tmp_dir, ROWS_DIR), exist_ok=True)

    meta: list[dict] = []
    with open(index_path(tmp_dir), "w", encoding="utf-8") as idx:
        for position, entry in enumerate(entries):
            record = _index_record(entry, position)
            idx.write(json.dumps(record, default=str) + "\n")
            meta.append(record)

            rdir = row_dir(tmp_dir, entry.get("example_id"), position)
            os.makedirs(rdir, exist_ok=True)
            spans = entry.get("trace") or []
            row = {k: v for k, v in entry.items() if k != "trace"}
            row["spans"] = [_span_summary(k, s) for k, s in enumerate(spans)]
            with open(os.path.join(rdir, ROW_FILE), "w", encoding="utf-8") as f:
                f.write(dump_trace_json(row, max_line_chars=max_line_chars))
            for k, span in enumerate(spans):
                with open(os.path.join(rdir, span_file_name(k)), "w", encoding="utf-8") as f:
                    f.write(dump_trace_json(span, max_line_chars=max_line_chars))

    with open(pretty_index_path(tmp_dir), "w", encoding="utf-8") as f:
        f.write(_pretty_index_text(meta))

    shutil.rmtree(trace_dir, ignore_errors=True)
    os.replace(tmp_dir, trace_dir)
    return index_path(trace_dir), meta


def provenance_path(trace_dir: str) -> str:
    return os.path.join(trace_dir, PROVENANCE_FILE)


def write_provenance(trace_dir: str, payload: dict[str, Any]) -> bool:
    """Stamp `payload` as `{trace_dir}/provenance.json` (atomic; best-effort).

    Called by the host side (`run_eval_tool._run_once`) after
    `write_eval_trace` has replaced the directory, because only the host knows
    the git state the eval measured. Returns False (and logs) instead of
    raising — provenance must never fail an eval.
    """
    if not trace_dir or not os.path.isdir(trace_dir):
        logger.warning("trace provenance: %r is not a directory — not stamped", trace_dir)
        return False
    path = provenance_path(trace_dir)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(json.dumps(payload, indent=2, default=str) + "\n")
        os.replace(tmp, path)
        return True
    except OSError as exc:
        logger.warning("trace provenance: could not write %s: %s", path, exc)
        return False


def read_provenance(trace_dir: str) -> dict[str, Any] | None:
    """The provenance stamp of a trace directory, or None when absent."""
    path = provenance_path(trace_dir)
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_index(trace_dir: str) -> list[dict]:
    """The index records of a trace directory (empty when missing)."""
    path = index_path(trace_dir)
    if not os.path.exists(path):
        return []
    out: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_json(path: str) -> Any:
    with open(path, encoding="utf-8") as f:
        return join_chunked_strings(json.load(f))


def load_row(trace_dir: str, row_file_rel: str, *, with_spans: bool = False) -> dict:
    """Load one row (chunked strings rejoined). With `with_spans`, the span
    bodies are loaded back into `row["trace"]` in `k` order."""
    path = os.path.join(trace_dir, row_file_rel)
    row = _load_json(path)
    if with_spans:
        rdir = os.path.dirname(path)
        row["trace"] = [
            _load_json(os.path.join(rdir, s["file"]))
            for s in row.get("spans") or []
        ]
    return row


def iter_rows(trace_dir: str, *, with_spans: bool = False) -> Iterator[dict]:
    """Yield every row of a trace directory in index order."""
    for record in read_index(trace_dir):
        yield load_row(trace_dir, record["row_file"], with_spans=with_spans)


__all__ = [
    "HOLDOUT_DENY_FRAGMENT",
    "HOLDOUT_SUFFIX",
    "INDEX_FILE",
    "PROVENANCE_FILE",
    "ROW_FILE",
    "ROWS_DIR",
    "eval_trace_dir",
    "eval_trace_dir_name",
    "holdout_suffix",
    "index_path",
    "is_holdout_suffix",
    "iter_rows",
    "load_row",
    "provenance_path",
    "read_index",
    "read_provenance",
    "row_dir",
    "row_file",
    "span_file_name",
    "write_eval_trace",
    "write_provenance",
]
