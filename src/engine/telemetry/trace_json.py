"""Copyright © 2026 440 Labs LLC

Pretty-printed JSON writer for the per-row / per-span / per-turn trace files.

Every trace file an agent is pointed at is written through `dump_trace_json`
so the two harnesses' read tools return it whole:

- Claude Code's `Read` has no per-line cap but a per-call token budget, which
  the per-row / per-span file split keeps each file under.
- OpenCode's `read` (and `grep`) truncate every LINE at 2,000 characters —
  hardcoded (`MAX_LINE_LENGTH` in `tool/read.ts`; the request to make it
  configurable was closed "not planned"). Pretty-printing alone does not help:
  `json.dumps(indent=2)` never breaks a string, so a 20 KB prompt is still one
  line. When `max_line_chars` is set, every string longer than it is written
  as a JSON **list of chunks** instead, each chunk on its own line. Nothing
  else is altered; readers rebuild the original with `"".join(chunks)`
  (`join_chunked_strings` does that for a whole document).

The gate is per run, not per file: `paths.trace_max_line_chars()` reads
`CODEEVOLVER_TRACE_MAX_LINE_CHARS`, which `optimizer/v3/_entry.py::run_entry`
sets only when the coding harness is OpenCode. Claude runs write plain JSON.
"""

from __future__ import annotations

import json
from typing import Any

# Sentinel key marking a string that was split so `join_chunked_strings` can
# tell a chunk list from a genuine list of strings.
CHUNKED_KEY = "__chunked_string__"


def _escaped_width(ch: str) -> int:
    """Width of `ch` once JSON-encoded with ensure_ascii=False."""
    o = ord(ch)
    if ch == '"' or ch == "\\" or ch in "\n\r\t\b\f":
        return 2
    if o < 0x20:
        return 6  # \u00XX
    return 1


def escaped_len(text: str) -> int:
    """Length of `text` as it appears inside a JSON string literal."""
    return sum(_escaped_width(ch) for ch in text)


def split_string(text: str, max_line_chars: int) -> list[str]:
    """Split `text` into pieces whose JSON-escaped width is <= `max_line_chars`.

    The limit is measured on the *escaped* text (a newline or quote costs two
    characters on disk) because that is what the read tool's line cap sees.
    Prefers to cut at the last whitespace in the second half of the window so
    words stay whole; hard-cuts when there is none. Never drops or reorders a
    character: `"".join(split_string(s, n)) == s`.
    """
    if max_line_chars <= 0:
        raise ValueError(f"max_line_chars must be positive, got {max_line_chars}")
    if escaped_len(text) <= max_line_chars:
        return [text]
    pieces: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        width = 0
        end = start
        last_ws = -1  # index just after the last whitespace in the second half
        while end < n:
            w = _escaped_width(text[end])
            if width + w > max_line_chars:
                break
            width += w
            end += 1
            if text[end - 1].isspace() and width >= max_line_chars // 2:
                last_ws = end
        if end >= n:
            pieces.append(text[start:])
            break
        cut = last_ws if last_ws > start else end
        pieces.append(text[start:cut])
        start = cut
    return pieces


def chunk_long_strings(value: Any, max_line_chars: int) -> Any:
    """Return a copy of `value` with every over-long string replaced by
    `{"__chunked_string__": [chunk, ...]}`. Other values pass through."""
    if isinstance(value, str):
        if len(value) > max_line_chars // 2 and escaped_len(value) > max_line_chars:
            return {CHUNKED_KEY: split_string(value, max_line_chars)}
        return value
    if isinstance(value, dict):
        return {
            str(k): chunk_long_strings(v, max_line_chars) for k, v in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [chunk_long_strings(v, max_line_chars) for v in value]
    return value


def join_chunked_strings(value: Any) -> Any:
    """Inverse of `chunk_long_strings`: rebuild every chunked string."""
    if isinstance(value, dict):
        if set(value) == {CHUNKED_KEY} and isinstance(value[CHUNKED_KEY], list):
            return "".join(str(p) for p in value[CHUNKED_KEY])
        return {k: join_chunked_strings(v) for k, v in value.items()}
    if isinstance(value, list):
        return [join_chunked_strings(v) for v in value]
    return value


def dump_trace_json(value: Any, *, max_line_chars: int | None) -> str:
    """Pretty-print `value` for an agent-facing trace file.

    `max_line_chars` None/0 → plain `json.dumps(indent=2)`. Otherwise every
    string longer than the limit is chunked first (see module docstring).
    Non-JSON values fall back to `str()` so a writer never raises on content.
    """
    if max_line_chars:
        value = chunk_long_strings(value, max_line_chars)
    return json.dumps(value, indent=2, ensure_ascii=False, default=str)


def load_trace_json(path: str) -> Any:
    """Read a file written by `dump_trace_json`, rejoining chunked strings."""
    with open(path, encoding="utf-8") as f:
        return join_chunked_strings(json.load(f))


__all__ = [
    "CHUNKED_KEY",
    "chunk_long_strings",
    "dump_trace_json",
    "escaped_len",
    "join_chunked_strings",
    "load_trace_json",
    "split_string",
]
