"""Copyright © 2026 440 Labs LLC

Capture of the IterationArchitect's own agent session ("meta-traces").

The architect runs in-process through the Claude Agent SDK; its message stream
carries the model's *thinking* (``ThinkingBlock``), assistant text, tool calls
(``ToolUseBlock``), and tool results (``ToolResultBlock`` inside a
``UserMessage``). This module serializes that stream to a per-iteration trace
DIRECTORY so a post-run meta-reflection agent can study how CodeEvolver's own
meta-agent reasoned and acted.

These meta-traces are written to ``paths.architect_traces_dir()`` — a dedicated
directory that is a *sibling* of ``traces_dir()`` (the client/Task-Agent OTel
traces). Keeping them physically separate means architect meta-traces can never
be confused with, or pollute, client trace data. The per-iteration
``inspect_traces`` tool, the coding agent, and the architect's own session are
all *fenced* from reading these files (see ``iteration_architect`` /
``architect_hooks``); only the end-of-run meta-reflection agent reads them.

Everything here is **best-effort**: capture must never sink an iteration, so
every public function swallows its own exceptions and logs a warning. The data
is diagnostic, not load-bearing.

On-disk layout — one directory per iteration::

    {architect_traces_dir}/iteration_{N}_architect_trace/
      index.jsonl        one short line per turn: {"turn", "role", "ts",
                         "kinds": [block kinds], "tools": [tool_use names],
                         "is_error": any tool_result error, "chars", "file",
                         "summary": a one-line human-readable digest — the
                         tool calls with their key argument, a run_eval
                         result's sample/score/sha, an inspect_traces
                         result's first sentence, the assistant text (or
                         thinking) preview, or the prompt text for user
                         turns — so a reader can pick the turns worth
                         opening WITHOUT reading every turn file}
      turn_{i:04d}.json  the turn, pretty-printed (`trace_json.dump_trace_json`
                         — chunked to OpenCode's line cap when the run's
                         harness is OpenCode)
      digest.md          rendered AFTER the session's last watchdog resume by
      transcript.md      `architect_trace_markdown.render_architect_trace_markdown`
      header.json        (never by the capture path — OpenCode rewrites the
                         directory on every export): the digest is the whole
                         session minus thinking with tool results capped at
                         1,000 chars; the transcript adds full thinking and
                         full tool inputs with results capped at 2,000; every
                         capped payload points at its turn file. `header.json`
                         feeds the run-level `{architect_traces_dir}/README.md`.

Turn schema — one object per stream message::

    {"ts": "<iso8601>", "iteration": 5, "role": "assistant",
     "blocks": [
       {"kind": "thinking", "text": "...", "signature_present": true},
       {"kind": "text", "text": "..."},
       {"kind": "tool_use", "name": "mcp__ce__run_eval", "id": "toolu_...",
        "input": {...}}]}
    {"ts": "...", "iteration": 5, "role": "user",
     "blocks": [
       {"kind": "tool_result", "tool_use_id": "toolu_...", "is_error": false,
        "content": "...", "truncated": false}]}
    {"ts": "...", "iteration": 5, "role": "result", "blocks": [],
     "usage": {...},
     "meta": {"subtype": "success", "num_turns": 12, "duration_ms": 84210,
              "total_cost_usd": 0.42, "session_id": "..."}}

OpenCode sessions are captured post-invocation via ``opencode export
<sessionID>`` — a consolidated session document with first-class ``reasoning``
parts, tool parts (terminal state + output), and per-step token metadata. The
harness serializes that document into the same schema
(``opencode/output.py::serialize_opencode_export``) and rewrites the
iteration's trace directory via ``write_opencode_trace`` after every invocation
(initial + watchdog resumes) — each export is an idempotent full-session
snapshot, so rewrite-on-capture converges to the complete session. When no
session id was captured or the export yields nothing, a single
``role="result"`` stub line with ``captured=false`` is written instead (see
``append_opencode_stub``) so the meta-reflection agent can degrade gracefully.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime, timezone
from typing import Any

from src.engine.paths import architect_traces_dir, trace_max_line_chars
from src.engine.telemetry.trace_json import dump_trace_json, load_trace_json

logger = logging.getLogger(__name__)

# Tool results can embed whole eval dumps; cap each so trace files stay bounded.
# Assistant text and thinking are NOT capped — they are the reflection signal.
_MAX_TOOL_RESULT_CHARS = 20_000


# Substring fence fragment: every architect trace directory name contains it
# (`architect_tool_registry.ARCHITECT_TRACE_DENY_FRAGMENT` is the same string).
ARCHITECT_TRACE_DIR_SUFFIX = "_architect_trace"
INDEX_FILE = "index.jsonl"


def architect_trace_path(iteration_idx: int) -> str:
    """Absolute path to an iteration's architect meta-trace DIRECTORY."""
    return os.path.join(
        architect_traces_dir(),
        f"iteration_{iteration_idx}{ARCHITECT_TRACE_DIR_SUFFIX}",
    )


def turn_file_name(turn_idx: int) -> str:
    return f"turn_{turn_idx:04d}.json"


# `summary` budget per index line. Long enough for "tool(arg) — what the
# architect said it was doing"; short enough that a 130-turn index is a
# few screens, not a novel.
_SUMMARY_CHARS = 200
_SUMMARY_ARG_CHARS = 70
# The argument that identifies a tool call, by preference. Falls back to the
# first argument when none of these is present. Shared with the markdown
# renderer (`architect_trace_markdown`).
TOOL_KEY_ARGS = (
    "eval_sample", "command", "filePath", "file_path", "path", "pattern",
    "query", "change_request", "name", "trace", "memory_updated",
)


def preview(text: Any, limit: int = _SUMMARY_CHARS) -> str:
    """Whitespace-collapsed, ellipsis-truncated preview of `text`."""
    if text is None:
        return ""
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def result_content_text(block: dict[str, Any]) -> str:
    """A tool_result block's content as one string (chunk lists rejoined)."""
    content = block.get("content")
    if isinstance(content, dict) and "__chunked_string__" in content:
        content = "".join(str(c) for c in content["__chunked_string__"])
    return content if isinstance(content, str) else json.dumps(content, default=str)


def parse_json_result(text: str) -> dict[str, Any] | None:
    """The JSON object a tool result carries, or None when it is not one."""
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        parsed = json.loads(stripped)
    except ValueError:
        return None
    return parsed if isinstance(parsed, dict) else None


def run_eval_result_line(parsed: dict[str, Any]) -> str:
    """One structured line for a `run_eval` tool result (the JSON carries
    `primary_score`). Shared by the index `summary` and the markdown
    renderer so both surfaces read identically:
    ``run_eval train/parent score=0.8304 rows=50 zero/failed=6 eval_id=… sha=… tree=… auto_committed``.
    """
    score = parsed.get("primary_score")
    score_s = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
    rows = parsed.get("row_ids")
    n_rows = len(rows) if isinstance(rows, list) else parsed.get("n_rows")
    meta = parsed.get("trace_meta")
    # Summary-only responses carry `zero_or_failed_count`; `failed_count` is
    # a legacy name, kept so older traces still render when replayed offline.
    failed = (
        sum(1 for m in meta if isinstance(m, dict) and (m.get("failed") or not m.get("score")))
        if isinstance(meta, list)
        else parsed.get("zero_or_failed_count", parsed.get("failed_count"))
    )
    bits = [
        f"run_eval {parsed.get('eval_sample')}/{parsed.get('suffix')}",
        f"score={score_s}",
    ]
    if n_rows is not None:
        bits.append(f"rows={n_rows}" + (f" zero/failed={failed}" if failed is not None else ""))
    if parsed.get("eval_id"):
        bits.append(f"eval_id={parsed['eval_id']}")
    if parsed.get("sha"):
        bits.append(f"sha={str(parsed['sha'])[:8]}")
    if parsed.get("tree"):
        bits.append(f"tree={str(parsed['tree'])[:8]}")
    infra = parsed.get("infra_failed")
    if isinstance(infra, int) and infra > 0:
        bits.append(f"infra_failed={infra}")
    if parsed.get("score_reliable") is False:
        bits.append("UNRELIABLE")
    if parsed.get("auto_committed"):
        bits.append("auto_committed")
    return " ".join(bits)


def _tool_use_summary(block: dict[str, Any]) -> str:
    name = str(block.get("name") or "?")
    inp = block.get("input")
    if not isinstance(inp, dict) or not inp:
        return f"{name}()"
    key = next((k for k in TOOL_KEY_ARGS if k in inp), next(iter(inp)))
    return f"{name}({key}={preview(inp[key], _SUMMARY_ARG_CHARS)})"


def _tool_result_summary(block: dict[str, Any]) -> str:
    text = result_content_text(block)
    prefix = "ERROR: " if block.get("is_error") else ""
    parsed = parse_json_result(text)
    if parsed is not None:
        if "primary_score" in parsed:
            return prefix + preview(run_eval_result_line(parsed))
        if parsed.get("ok") is False and parsed.get("error"):
            return "ERROR: " + preview(parsed["error"])
        if "summary" in parsed:
            return prefix + preview(f"inspect_traces: {parsed['summary']}")
        if "note" in parsed and "branch" in parsed:
            return prefix + preview(f"close_iteration branch={parsed['branch']}: {parsed['note']}")
        if "message" in parsed:
            return prefix + preview(parsed["message"])
    return prefix + preview(text)


def _turn_summary(entry: dict[str, Any]) -> str:
    """One line a reader can triage the turn by (see the module docstring)."""
    blocks = [b for b in (entry.get("blocks") or []) if isinstance(b, dict)]
    role = entry.get("role")
    if role == "result":
        meta = entry.get("meta") if isinstance(entry.get("meta"), dict) else {}
        if meta.get("captured") is False:
            return preview(f"NOT CAPTURED: {meta.get('note') or ''}")
        bits = [f"{k}={meta[k]}" for k in ("num_turns", "num_messages", "duration_ms", "total_cost_usd") if k in meta]
        return preview("session result " + " ".join(bits))
    tool_uses = [_tool_use_summary(b) for b in blocks if b.get("kind") == "tool_use"]
    results = [_tool_result_summary(b) for b in blocks if b.get("kind") == "tool_result"]
    texts = [b.get("text") for b in blocks if b.get("kind") == "text" and b.get("text")]
    thinking = [b.get("text") for b in blocks if b.get("kind") == "thinking" and b.get("text")]
    parts: list[str] = []
    if tool_uses:
        parts.append("; ".join(tool_uses))
    if results:
        parts.append(" | ".join(results))
    if texts:
        parts.append(("prompt: " if role == "user" else "") + preview(texts[0], 120))
    elif thinking and not results:
        parts.append("(thinking) " + preview(thinking[0], 120))
    return preview(" — ".join(p for p in parts if p))


def _index_record(turn_idx: int, entry: dict[str, Any]) -> dict[str, Any]:
    blocks = entry.get("blocks") or []
    record: dict[str, Any] = {
        "turn": turn_idx,
        "role": entry.get("role"),
        "ts": entry.get("ts"),
        "kinds": [b.get("kind") for b in blocks if isinstance(b, dict)],
        "tools": [
            b.get("name") for b in blocks
            if isinstance(b, dict) and b.get("kind") == "tool_use"
        ],
        "is_error": any(
            isinstance(b, dict) and b.get("kind") == "tool_result" and b.get("is_error")
            for b in blocks
        ),
        "chars": len(json.dumps(blocks, default=str)),
        "file": turn_file_name(turn_idx),
    }
    meta = entry.get("meta")
    if isinstance(meta, dict) and "captured" in meta:
        record["captured"] = bool(meta["captured"])
    try:
        record["summary"] = _turn_summary(entry)
    except Exception as exc:  # noqa: BLE001 — the digest is a convenience
        record["summary"] = f"(summary failed: {exc})"
    return record


def _write_turn(trace_dir: str, turn_idx: int, entry: dict[str, Any]) -> None:
    """Write one turn file and its index line. Caller ensures `trace_dir` exists."""
    with open(os.path.join(trace_dir, turn_file_name(turn_idx)), "w", encoding="utf-8") as f:
        f.write(dump_trace_json(entry, max_line_chars=trace_max_line_chars()))
    with open(os.path.join(trace_dir, INDEX_FILE), "a", encoding="utf-8") as f:
        f.write(json.dumps(_index_record(turn_idx, entry), default=str) + "\n")


def _next_turn_idx(trace_dir: str) -> int:
    path = os.path.join(trace_dir, INDEX_FILE)
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def read_architect_trace(iteration_idx: int) -> list[dict[str, Any]]:
    """Every turn of an iteration's trace, in order (chunked strings rejoined).

    Empty when no trace directory exists. Used by tests and tooling; the
    meta-reflection agents navigate the directory themselves.
    """
    trace_dir = architect_trace_path(iteration_idx)
    path = os.path.join(trace_dir, INDEX_FILE)
    if not os.path.exists(path):
        return []
    turns: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                turns.append(load_trace_json(os.path.join(trace_dir, record["file"])))
    return turns


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def cap_tool_result_text(content: Any) -> tuple[str, bool]:
    """Coerce a tool-result payload to text, capped at ``_MAX_TOOL_RESULT_CHARS``.

    Returns ``(text, truncated)``. Shared by both harness capture paths so a
    single large eval dump cannot blow up a trace file regardless of provider.
    """
    text = content if isinstance(content, str) else json.dumps(content, default=str)
    truncated = len(text) > _MAX_TOOL_RESULT_CHARS
    if truncated:
        text = text[:_MAX_TOOL_RESULT_CHARS] + "…[truncated]"
    return text, truncated


def serialize_assistant_blocks(
    message: Any,
    TextBlock: type,
    ThinkingBlock: type,
    ToolUseBlock: type,
) -> list[dict[str, Any]]:
    """Serialize an ``AssistantMessage``'s content blocks.

    Captures thinking text (the key reflection signal), assistant text, and
    tool calls. The opaque thinking ``signature`` is NOT persisted (large and
    useless to a reader) — only a ``signature_present`` flag is recorded.
    """
    out: list[dict[str, Any]] = []
    try:
        for block in getattr(message, "content", None) or []:
            if isinstance(block, ThinkingBlock):
                out.append(
                    {
                        "kind": "thinking",
                        "text": getattr(block, "thinking", "") or "",
                        "signature_present": bool(getattr(block, "signature", None)),
                    }
                )
            elif isinstance(block, TextBlock):
                out.append({"kind": "text", "text": getattr(block, "text", "") or ""})
            elif isinstance(block, ToolUseBlock):
                out.append(
                    {
                        "kind": "tool_use",
                        "name": getattr(block, "name", "?"),
                        "id": getattr(block, "id", None),
                        "input": getattr(block, "input", None),
                    }
                )
    except Exception as exc:  # noqa: BLE001
        logger.warning("architect_trace: assistant serialize failure: %s", exc)
    return out


def serialize_user_blocks(message: Any, ToolResultBlock: type) -> list[dict[str, Any]]:
    """Serialize the ``ToolResultBlock``s carried by a ``UserMessage``.

    Each result's content is coerced to text and capped at
    ``_MAX_TOOL_RESULT_CHARS`` (with a ``truncated`` flag) so a single large
    eval dump cannot blow up the trace file.
    """
    out: list[dict[str, Any]] = []
    try:
        for block in getattr(message, "content", None) or []:
            if not isinstance(block, ToolResultBlock):
                continue
            text, truncated = cap_tool_result_text(getattr(block, "content", None))
            out.append(
                {
                    "kind": "tool_result",
                    "tool_use_id": getattr(block, "tool_use_id", None),
                    "is_error": bool(getattr(block, "is_error", False)),
                    "content": text,
                    "truncated": truncated,
                }
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("architect_trace: user serialize failure: %s", exc)
    return out


def append_architect_turn(
    iteration_idx: int,
    *,
    role: str,
    blocks: list[dict[str, Any]],
    usage: dict[str, Any] | None = None,
    meta: dict[str, Any] | None = None,
) -> None:
    """Append one turn to the iteration's architect meta-trace directory.

    Best-effort: creates the directory on demand and swallows every error so a
    capture hiccup never affects the architect session.
    """
    try:
        entry: dict[str, Any] = {
            "ts": _now_iso(),
            "iteration": iteration_idx,
            "role": role,
            "blocks": blocks,
        }
        if usage is not None:
            entry["usage"] = usage
        if meta is not None:
            entry["meta"] = meta
        trace_dir = architect_trace_path(iteration_idx)
        os.makedirs(trace_dir, exist_ok=True)
        _write_turn(trace_dir, _next_turn_idx(trace_dir), entry)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: architect_trace append failure (non-fatal): %s",
            iteration_idx,
            exc,
        )


def append_session_prompt_turn(iteration_idx: int, prompt: str, *, resume: bool) -> None:
    """Record the prompt that opened a session as a user text turn.

    The Claude harness feeds the prompt to the SDK as a stream the capture
    loop never sees, so without this the initial prompt and every watchdog
    re-prompt were absent from Claude traces (OpenCode's export carries them
    natively). `meta.prompt_kind` lets the markdown renderer label the turn
    `Prompt` vs `Watchdog resume` without guessing from ordinals.
    """
    append_architect_turn(
        iteration_idx,
        role="user",
        blocks=[{"kind": "text", "text": prompt or ""}],
        meta={"prompt_kind": "resume" if resume else "initial"},
    )


def write_opencode_trace(iteration_idx: int, turns: list[dict[str, Any]]) -> int:
    """Rewrite the iteration's trace directory from serialized OpenCode turns.

    Each turn dict carries ``{"role", "blocks"}`` plus optional ``usage`` /
    ``meta`` / ``ts`` (the output of ``opencode/output.py::
    serialize_opencode_export``); ``iteration`` is stamped here. ``ts`` is
    the export's own message timestamp when it carried one — a post-session
    export otherwise stamps every turn with the SAME export time, which is
    no ordering signal at all — and falls back to now.

    This **clears and rewrites** the directory (unlike the Claude path, which
    appends per stream message) because every ``opencode export`` is an
    idempotent full-session snapshot — the capture after each invocation
    (initial + watchdog resumes) replaces the directory with the complete
    session so far. Returns the number of turns written; 0 (empty input or
    any failure) lets the caller fall back to ``append_opencode_stub``.
    """
    if not turns:
        return 0
    try:
        entries: list[dict[str, Any]] = []
        for turn in turns:
            entry: dict[str, Any] = {
                "ts": turn.get("ts") or _now_iso(),
                "iteration": iteration_idx,
                "role": turn.get("role", "assistant"),
                "blocks": turn.get("blocks") or [],
            }
            if turn.get("usage") is not None:
                entry["usage"] = turn["usage"]
            if turn.get("meta") is not None:
                entry["meta"] = turn["meta"]
            entries.append(entry)
        trace_dir = architect_trace_path(iteration_idx)
        tmp_dir = trace_dir + ".tmp"
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)
        for turn_idx, entry in enumerate(entries):
            _write_turn(tmp_dir, turn_idx, entry)
        shutil.rmtree(trace_dir, ignore_errors=True)
        os.replace(tmp_dir, trace_dir)
        return len(entries)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: architect_trace rewrite failure (non-fatal): %s",
            iteration_idx,
            exc,
        )
        return 0


def append_opencode_stub(iteration_idx: int, stdout_tail: str | None = None) -> None:
    """Write a 'captured:false' turn for an OpenCode architect session.

    Written when session export is unavailable — no session id was captured
    from the run's stdout, or ``opencode export`` failed or yielded nothing.
    This stub lets the meta-reflection agent detect that no rich architect
    trace exists for the iteration and degrade gracefully (it can still
    reflect on memory + iteration change summaries + the repo).
    """
    meta: dict[str, Any] = {
        "harness": "opencode",
        "captured": False,
        "note": "OpenCode session export unavailable; no rich architect trace",
    }
    if stdout_tail:
        meta["stdout_tail"] = stdout_tail[-2000:]
    append_architect_turn(iteration_idx, role="result", blocks=[], meta=meta)


__all__ = [
    "ARCHITECT_TRACE_DIR_SUFFIX",
    "architect_trace_path",
    "read_architect_trace",
    "turn_file_name",
    "cap_tool_result_text",
    "serialize_assistant_blocks",
    "serialize_user_blocks",
    "append_architect_turn",
    "write_opencode_trace",
    "append_opencode_stub",
]
