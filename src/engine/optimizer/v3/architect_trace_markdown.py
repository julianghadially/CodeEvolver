"""Copyright © 2026 440 Labs LLC

Rendered markdown surfaces for an architect session trace directory.

`architect_trace.py` captures a session as `index.jsonl` + `turn_NNNN.json`.
That layer is complete but unreadable at triage time: an iteration is 60–130
turns and 420–720K chars, ~95% of the architect's own words are thinking, and
~75% of tool-result bytes are payloads (run_eval trace_meta, file reads) that
exist elsewhere. This module renders, deterministically and without an LLM:

- ``digest.md`` — the whole session **minus thinking**: a header, then every
  turn in order with the assistant's text in full, each tool call as
  ``name(args)``, each tool result capped at `DIGEST_RESULT_CHARS`, run_eval
  results as one structured line, prompts / watchdog re-prompts verbatim,
  errors flagged. 40–110K chars on the measured hover run — small enough that
  an auditor reads EVERY iteration's digest.
- ``transcript.md`` — everything: full thinking, full tool inputs, results
  capped at `TRANSCRIPT_RESULT_CHARS`. 175–455K chars; read in pages for the
  one or two iterations that matter.
- ``header.json`` — the merged header (loop facts + trace stats) that feeds the
  run-level ``{architect_traces_dir}/README.md`` (one row per iteration).

Every capped payload ends with ``→ full: turn_NNNN.json``. Every emitted line
is under `MD_MAX_LINE_CHARS` (OpenCode's `read` truncates lines at 2,000).

**Render only after the session's last watchdog resume.** OpenCode's
`write_opencode_trace` rm-trees and rewrites the directory on every export, so
anything rendered earlier is deleted. The loops call
`render_architect_trace_markdown` after the iteration is closed, which is also
when the close facts for the header exist. Never called from the capture path.

Best-effort like the capture layer: every public function swallows its own
exceptions and logs a warning; a render failure never affects the iteration.
"""
from __future__ import annotations

import glob
import json
import logging
import os
import re
from collections import Counter
from datetime import datetime
from typing import Any, Mapping

from src.engine.optimizer.v3.architect_trace import (
    ARCHITECT_TRACE_DIR_SUFFIX,
    INDEX_FILE,
    TOOL_KEY_ARGS,
    architect_trace_path,
    parse_json_result,
    preview,
    result_content_text,
    run_eval_result_line,
)
from src.engine.paths import architect_traces_dir
from src.engine.telemetry.trace_json import load_trace_json, split_string

logger = logging.getLogger(__name__)

# Under OpenCode's 2,000-char `read` line cap, with headroom for list prefixes.
MD_MAX_LINE_CHARS = 1_800
DIGEST_RESULT_CHARS = 1_000
TRANSCRIPT_RESULT_CHARS = 2_000
INSPECT_SUMMARY_CHARS = 800
# Non-identifying tool args are flattened to this many chars in the digest.
DIGEST_OTHER_ARG_CHARS = 80
# The identifying arg is shown "in full" — but a 30 KB bash heredoc still
# gets a pointer to the turn file.
DIGEST_KEY_ARG_SOFT_CAP = 4_000
# Edits / writes: path + this much of the new content.
EDIT_NEW_CONTENT_HEAD = 200

DIGEST_FILE = "digest.md"
TRANSCRIPT_FILE = "transcript.md"
HEADER_FILE = "header.json"
README_FILE = "README.md"

_DIGEST = "digest"
_TRANSCRIPT = "transcript"

# Watchdog re-prompt openers (`architect_watchdogs.py`), after the `/s\n\n`
# skill prefix OpenCode prepends. Order matters only for readability.
_WATCHDOG_OPENERS: tuple[tuple[str, str], ...] = (
    ("You are about to end iteration", "no-change watchdog"),
    ("The session is about to exit but you have not called", "close-missing watchdog"),
    ("You called the `close_iteration` tool", "close-skill watchdog"),
    ("Warning: the current git sha", "revert-check watchdog"),
    ("It appears that you did real work", "memory-diff watchdog"),
)

_EDIT_TOOLS = {"edit", "write", "multiedit"}
_EDIT_CONTENT_KEYS = ("newString", "new_string", "content", "edits")
_READ_TOOLS = {"read"}
_MULTILINE_ARG_TOOLS = {"bash", "ce_coding_agent", "coding_agent"}


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------


def build_trace_header(ctx: Any, **overrides: Any) -> dict[str, Any]:
    """Loop-side facts for the rendered header, arm-agnostic.

    Reads what every `BaseIterationContext` has (eval records, baseline
    score, token usage) and `getattr`s the engine-3 `IterationContext`
    extras (best / comparison scores, branch, close sha, watchdog flag).
    `overrides` win — the loops pass what only the close record knows.
    """
    records = list(getattr(ctx, "eval_records", None) or [])
    by_sample: Counter[str] = Counter()
    rows = 0
    for rec in records:
        if not isinstance(rec, Mapping):
            continue
        by_sample[str(rec.get("eval_sample") or "?")] += 1
        try:
            rows += int(rec.get("n_rows") or 0)
        except (TypeError, ValueError):
            pass
    header: dict[str, Any] = {
        "iteration": getattr(ctx, "iteration_idx", None),
        "branch": getattr(ctx, "child_branch", None),
        "close_sha": getattr(ctx, "close_sha", None),
        "changes_landed": getattr(ctx, "changes_landed", None),
        "baseline_score": getattr(ctx, "initial_subsample_score", None),
        "best_train_score": getattr(ctx, "best_subsample_score", None),
        "comparison_best_score": getattr(ctx, "comparison_best_score", None),
        "improvement_mode": getattr(ctx, "claimed_improvement_mode", None),
        "eval_rows_spent": rows,
        "evals_by_sample": dict(by_sample),
        "watchdog_fired": getattr(ctx, "watchdog_fired", None),
        "skipped": getattr(ctx, "skipped", None),
        "token_usage_by_model": dict(getattr(ctx, "token_usage_by_model", None) or {}),
    }
    header.update(overrides)
    return header


def render_trace_dir(
    trace_dir: str, header: Mapping[str, Any] | None = None
) -> tuple[str, str] | None:
    """Render `digest.md`, `transcript.md` and `header.json` into `trace_dir`.

    Path-based core used by the loops (via `render_architect_trace_markdown`),
    the test seeder and tooling. Returns `(digest_path, transcript_path)`, or
    None when the directory / index does not exist or rendering failed
    (logged, never raised).
    """
    try:
        index_path = os.path.join(trace_dir, INDEX_FILE)
        if not os.path.isfile(index_path):
            return None
        turns = _load_turns(trace_dir)
        merged = dict(header or {})
        if merged.get("iteration") is None:
            merged["iteration"] = _iteration_from_dir(trace_dir)
        stats = _trace_stats(turns)
        merged["trace_stats"] = stats
        digest = _render_document(turns, merged, stats, _DIGEST)
        transcript = _render_document(turns, merged, stats, _TRANSCRIPT)
        digest_path = os.path.join(trace_dir, DIGEST_FILE)
        transcript_path = os.path.join(trace_dir, TRANSCRIPT_FILE)
        _atomic_write(digest_path, digest)
        _atomic_write(transcript_path, transcript)
        _atomic_write(
            os.path.join(trace_dir, HEADER_FILE),
            json.dumps(merged, indent=2, default=str, ensure_ascii=False) + "\n",
        )
        return digest_path, transcript_path
    except Exception as exc:  # noqa: BLE001 — diagnostic surface only
        logger.warning(
            "architect_trace_markdown: render failure for %s (non-fatal): %s",
            trace_dir, exc,
        )
        return None


def render_architect_trace_markdown(
    iteration_idx: int,
    header: Mapping[str, Any] | None = None,
    *,
    update_readme: bool = True,
) -> tuple[str, str] | None:
    """Render the iteration's trace directory, then refresh the run README."""
    out = render_trace_dir(architect_trace_path(iteration_idx), header)
    if update_readme:
        write_run_readme()
    return out


def write_run_readme(root: str | None = None) -> str | None:
    """Regenerate `{root}/README.md` from every rendered iteration's
    `header.json`. One row per iteration, numeric order, relative links.
    Non-fatal; returns the README path or None."""
    try:
        root = root or architect_traces_dir()
        pattern = os.path.join(root, f"iteration_*{ARCHITECT_TRACE_DIR_SUFFIX}", HEADER_FILE)
        rows: list[tuple[int, str, dict[str, Any]]] = []
        for path in glob.glob(pattern):
            dirname = os.path.basename(os.path.dirname(path))
            n = _iteration_from_dir(dirname)
            try:
                with open(path, encoding="utf-8") as f:
                    hdr = json.load(f)
            except (OSError, ValueError):
                continue
            rows.append((n if n is not None else -1, dirname, hdr))
        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        lines = [
            "# Architect sessions — run overview",
            "",
            "One row per iteration. Open `digest.md` for EVERY iteration first (the "
            "session minus thinking, tool results capped at "
            f"{DIGEST_RESULT_CHARS:,} chars); open `transcript.md` (full thinking and "
            f"tool inputs, results capped at {TRANSCRIPT_RESULT_CHARS:,}) for the one or "
            "two iterations that matter; open a `turn_NNNN.json` only when a payload "
            "was capped (`→ full: turn_NNNN.json`).",
            "",
            "| iter | branch | train baseline → best | comparison best | landed | "
            "watchdog | eval rows | turns | tool errors | elapsed | files |",
            "|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for _n, dirname, hdr in rows:
            st = hdr.get("trace_stats") or {}
            if st.get("captured") is False:
                turns_s = "not captured"
            else:
                turns_s = str(st.get("turns", "n/a"))
            lines.append(
                "| {it} | {br} | {base} → {best} | {comp} | {landed} | {wd} | {rows} | "
                "{turns} | {errs} | {el} | [digest]({d}/{df}) · [transcript]({d}/{tf}) |".format(
                    it=hdr.get("iteration", "?"),
                    br=_md_cell(hdr.get("branch")),
                    base=_score(hdr.get("baseline_score")),
                    best=_score(hdr.get("best_train_score")),
                    comp=_score(hdr.get("comparison_best_score")),
                    landed=_yes_no(hdr.get("changes_landed")),
                    wd=_yes_no(hdr.get("watchdog_fired")),
                    rows=_int(hdr.get("eval_rows_spent")),
                    turns=turns_s,
                    errs=st.get("tool_errors", "n/a"),
                    el=st.get("elapsed") or "n/a",
                    d=dirname, df=DIGEST_FILE, tf=TRANSCRIPT_FILE,
                )
            )
        lines.append("")
        path = os.path.join(root, README_FILE)
        _atomic_write(path, "\n".join(lines))
        return path
    except Exception as exc:  # noqa: BLE001
        logger.warning("architect_trace_markdown: README failure (non-fatal): %s", exc)
        return None


__all__ = [
    "DIGEST_FILE",
    "DIGEST_RESULT_CHARS",
    "HEADER_FILE",
    "MD_MAX_LINE_CHARS",
    "README_FILE",
    "TRANSCRIPT_FILE",
    "TRANSCRIPT_RESULT_CHARS",
    "build_trace_header",
    "render_architect_trace_markdown",
    "render_trace_dir",
    "write_run_readme",
]


# --------------------------------------------------------------------------
# Loading + stats
# --------------------------------------------------------------------------


def _iteration_from_dir(path: str) -> int | None:
    m = re.search(r"iteration_(\d+)" + re.escape(ARCHITECT_TRACE_DIR_SUFFIX), os.path.basename(path.rstrip("/")))
    return int(m.group(1)) if m else None


def _load_turns(trace_dir: str) -> list[tuple[int, str, dict[str, Any] | None, str | None]]:
    """`(turn_idx, file, turn_or_None, error_or_None)` in index order."""
    out: list[tuple[int, str, dict[str, Any] | None, str | None]] = []
    with open(os.path.join(trace_dir, INDEX_FILE), encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except ValueError as exc:
                out.append((len(out), "?", None, f"index line unreadable: {exc}"))
                continue
            turn_idx = int(rec.get("turn", len(out)))
            file = str(rec.get("file") or f"turn_{turn_idx:04d}.json")
            try:
                turn = load_trace_json(os.path.join(trace_dir, file))
                if not isinstance(turn, dict):
                    raise ValueError("turn is not an object")
                out.append((turn_idx, file, turn, None))
            except Exception as exc:  # noqa: BLE001
                out.append((turn_idx, file, None, f"{type(exc).__name__}: {exc}"))
    return out


def _blocks(turn: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not turn:
        return []
    return [b for b in (turn.get("blocks") or []) if isinstance(b, dict)]


def _is_prompt_turn(turn: dict[str, Any] | None) -> bool:
    if not turn or turn.get("role") != "user":
        return False
    blocks = _blocks(turn)
    return bool(blocks) and all(b.get("kind") == "text" for b in blocks)


def _parse_ts(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_seconds(total: float) -> str:
    total = max(0.0, total)
    if total < 60:
        return f"{total:.1f}s"
    minutes, seconds = divmod(int(round(total)), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {seconds:02d}s"
    return f"{minutes}m {seconds:02d}s"


# Stamps closer than this are export-time artefacts (a post-session export
# stamped every turn within microseconds), not timing.
_MIN_ELAPSED_SECONDS = 1.0


def _elapsed(prev: datetime | None, cur: datetime | None) -> str | None:
    if prev is None or cur is None:
        return None
    delta = (cur - prev).total_seconds()
    if delta < _MIN_ELAPSED_SECONDS:
        return None
    return "+" + _fmt_seconds(delta)


def _trace_stats(turns: list[tuple[int, str, dict[str, Any] | None, str | None]]) -> dict[str, Any]:
    roles: Counter[str] = Counter()
    tools: Counter[str] = Counter()
    tool_errors = 0
    thinking_chars = 0
    captured: bool | None = None
    harness = model = None
    usage_totals: Counter[str] = Counter()
    result_meta: dict[str, Any] = {}
    timestamps: list[datetime] = []
    for _idx, _file, turn, err in turns:
        if err or turn is None:
            roles["unreadable"] += 1
            continue
        ts = _parse_ts(turn.get("ts"))
        if ts is not None:
            timestamps.append(ts)
        role = turn.get("role")
        if role == "result":
            roles["result"] += 1
            meta = turn.get("meta") if isinstance(turn.get("meta"), dict) else {}
            if "captured" in meta:
                captured = bool(meta["captured"]) if captured is None else (captured or bool(meta["captured"]))
            harness = harness or meta.get("harness")
            model = model or meta.get("model")
            for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd", "session_id", "session_ids"):
                if k in meta and k not in result_meta:
                    result_meta[k] = meta[k]
            usage = turn.get("usage")
            if isinstance(usage, dict):
                for k, v in usage.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        usage_totals[k] += v
            continue
        if _is_prompt_turn(turn):
            roles["prompt"] += 1
            continue
        roles["assistant" if role == "assistant" else "tool-result"] += 1
        for b in _blocks(turn):
            kind = b.get("kind")
            if kind == "tool_use":
                tools[str(b.get("name") or "?")] += 1
            elif kind == "tool_result" and b.get("is_error"):
                tool_errors += 1
            elif kind == "thinking":
                thinking_chars += len(b.get("text") or "")
    distinct = sorted(set(timestamps))
    elapsed = None
    first = last = None
    if len(distinct) >= 2 and (distinct[-1] - distinct[0]).total_seconds() >= _MIN_ELAPSED_SECONDS:
        first, last = distinct[0], distinct[-1]
        elapsed = _fmt_seconds((last - first).total_seconds())
    return {
        "turns": len(turns),
        "roles": dict(roles),
        "tool_calls": dict(tools.most_common()),
        "tool_errors": tool_errors,
        "thinking_chars": thinking_chars,
        "captured": captured,
        "harness": harness,
        "model": model,
        "usage": dict(usage_totals),
        "result_meta": result_meta,
        "elapsed": elapsed,
        "first_ts": first.isoformat() if first else None,
        "last_ts": last.isoformat() if last else None,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _score(v: Any) -> str:
    return f"{v:.4f}" if isinstance(v, (int, float)) and not isinstance(v, bool) else "n/a"


def _int(v: Any) -> str:
    return f"{v:,}" if isinstance(v, int) and not isinstance(v, bool) else "n/a"


def _yes_no(v: Any) -> str:
    if v is None:
        return "n/a"
    return "yes" if v else "no"


def _md_cell(v: Any) -> str:
    return "n/a" if v in (None, "") else str(v).replace("|", "\\|").replace("\n", " ")


def _short_sha(v: Any) -> str:
    return str(v)[:8] if isinstance(v, str) and v else "n/a"


def _render_header(header: Mapping[str, Any], stats: Mapping[str, Any], mode: str) -> list[str]:
    it = header.get("iteration")
    lines = [f"# Iteration {it if it is not None else '?'} architect session — {mode}", ""]
    lines.append(
        f"- branch: {_md_cell(header.get('branch'))} · close_sha: {_short_sha(header.get('close_sha'))}"
        f" · changes_landed: {_yes_no(header.get('changes_landed'))}"
        f" · primary after: {_md_cell(header.get('primary_branch_after'))}"
    )
    lines.append(
        f"- train subsample: baseline {_score(header.get('baseline_score'))} → best "
        f"{_score(header.get('best_train_score'))} · comparison best: "
        f"{_score(header.get('comparison_best_score'))} · mode: {_md_cell(header.get('improvement_mode'))}"
    )
    by_sample = header.get("evals_by_sample") or {}
    by_sample_s = ", ".join(f"{k}×{v}" for k, v in sorted(by_sample.items())) if by_sample else "none"
    wd = header.get("watchdog_fired")
    lines.append(
        f"- eval rows spent: {_int(header.get('eval_rows_spent'))} ({by_sample_s})"
        f" · watchdog resumes: {'yes' if wd else ('no' if wd is not None else 'n/a')}"
        f" (prompt turns: {stats.get('roles', {}).get('prompt', 0)})"
    )
    roles = stats.get("roles") or {}
    role_s = ", ".join(f"{k} {v}" for k, v in roles.items()) or "none"
    tools = stats.get("tool_calls") or {}
    tools_s = ", ".join(f"{k} {v}" for k, v in tools.items()) or "none"
    lines.append(
        f"- turns: {stats.get('turns', 0)} ({role_s}) · tool calls: {tools_s}"
        f" · tool errors: {stats.get('tool_errors', 0)}"
    )
    if stats.get("elapsed"):
        lines.append(
            f"- elapsed: {stats['elapsed']} (first ts {stats.get('first_ts')} → last {stats.get('last_ts')})"
        )
    usage = stats.get("usage") or {}
    usage_s = ", ".join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k, v in usage.items()) or "n/a"
    rm = stats.get("result_meta") or {}
    rm_s = " ".join(f"{k}={rm[k]}" for k in ("subtype", "num_turns", "duration_ms", "total_cost_usd") if k in rm)
    lines.append(
        f"- harness: {_md_cell(stats.get('harness'))} · model: {_md_cell(stats.get('model'))}"
        f" · tokens: {usage_s}" + (f" · {rm_s}" if rm_s else "")
    )
    if mode == _DIGEST and stats.get("thinking_chars"):
        lines.append(
            f"- thinking omitted here: {stats['thinking_chars']:,} chars — see `{TRANSCRIPT_FILE}`"
        )
    extras = []
    if header.get("error"):
        extras.append(f"error: {preview(header['error'], 300)}")
    if header.get("skipped"):
        extras.append("skipped: yes")
    if header.get("constraint_violation"):
        extras.append(f"constraint_violation: {preview(header['constraint_violation'], 300)}")
    if extras:
        lines.append("- " + " · ".join(extras))
    other = TRANSCRIPT_FILE if mode == _DIGEST else DIGEST_FILE
    lines.append(f"- files: `{INDEX_FILE}` · `turn_NNNN.json` · `{other}`")
    lines.append("")
    return lines


def _fenced(payload: str, lang: str = "") -> list[str]:
    """Fence `payload` with more backticks than any run inside it."""
    longest = max((len(m) for m in re.findall(r"`+", payload)), default=0)
    fence = "`" * max(4, longest + 1)
    body = payload.splitlines() or [""]
    return [fence + lang, *body, fence]


def _cap(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    return text[:limit] + "…", True


def _pointer(file: str) -> str:
    return f"→ full: {file}"


def _watchdog_label(text: str) -> str:
    # OpenCode's export quotes the prompt and the skill prefix is `/s`.
    stripped = text.lstrip().lstrip("\"'").lstrip()
    if stripped.startswith("/s"):
        stripped = stripped[2:].lstrip()
    for opener, label in _WATCHDOG_OPENERS:
        if stripped.startswith(opener):
            return label
    return "(unrecognized)"


def _tool_name_by_id(turns: list[tuple[int, str, dict[str, Any] | None, str | None]]) -> dict[str, str]:
    names: dict[str, str] = {}
    for _i, _f, turn, _e in turns:
        for b in _blocks(turn):
            if b.get("kind") == "tool_use" and b.get("id"):
                names[str(b["id"])] = str(b.get("name") or "?")
    return names


def _base_tool_name(name: str) -> str:
    """`mcp__ce__run_eval` / `ce_run_eval` / `Bash` → `run_eval` / `bash`."""
    n = name.lower()
    if n.startswith("mcp__"):
        n = n.split("__")[-1]
    if n.startswith("ce_"):
        n = n[3:]
    return n


def _render_tool_use(block: dict[str, Any], mode: str, file: str) -> list[str]:
    name = str(block.get("name") or "?")
    base = _base_tool_name(name)
    inp = block.get("input")
    lines: list[str] = []
    if not isinstance(inp, dict):
        arg = "" if inp is None else preview(inp, DIGEST_OTHER_ARG_CHARS)
        lines.append(f"- **call** `{name}({arg})`")
    elif not inp:
        lines.append(f"- **call** `{name}()`")
    else:
        key = next((k for k in TOOL_KEY_ARGS if k in inp), next(iter(inp)))
        key_val = inp[key]
        key_text = key_val if isinstance(key_val, str) else json.dumps(key_val, default=str)
        block_body: str | None = None
        if base in _MULTILINE_ARG_TOOLS and "\n" in key_text.strip():
            block_body = key_text
            key_s = f"{key}=⤵"
        else:
            key_s = f"{key}={preview(key_text, DIGEST_KEY_ARG_SOFT_CAP)}"
        others: list[str] = []
        if base in _EDIT_TOOLS:
            for ck in _EDIT_CONTENT_KEYS:
                if ck in inp and ck != key:
                    cv = inp[ck]
                    cv_s = cv if isinstance(cv, str) else json.dumps(cv, default=str)
                    others.append(f"new: {preview(cv_s, EDIT_NEW_CONTENT_HEAD)}")
                    break
            for k, v in inp.items():
                if k == key or k in _EDIT_CONTENT_KEYS:
                    continue
                others.append(f"{k}={preview(v, DIGEST_OTHER_ARG_CHARS)}")
        else:
            for k, v in inp.items():
                if k == key:
                    continue
                others.append(f"{k}={preview(v, DIGEST_OTHER_ARG_CHARS)}")
        arg_s = ", ".join([key_s, *others])
        lines.append(f"- **call** `{name}({arg_s})`")
        if block_body is not None:
            body, capped = _cap(block_body, DIGEST_KEY_ARG_SOFT_CAP)
            lines.extend(_fenced(body))
            if capped:
                lines.append(_pointer(file))
    if mode == _TRANSCRIPT and isinstance(inp, dict) and inp:
        lines.append("  input:")
        lines.extend(_fenced(json.dumps(inp, indent=2, ensure_ascii=False, default=str), "json"))
    return lines


def _render_tool_result(
    block: dict[str, Any], tool_name: str | None, mode: str, file: str
) -> list[str]:
    text = result_content_text(block)
    limit = DIGEST_RESULT_CHARS if mode == _DIGEST else TRANSCRIPT_RESULT_CHARS
    is_error = bool(block.get("is_error"))
    parsed = parse_json_result(text)
    base = _base_tool_name(tool_name) if tool_name else None
    label = f" ({tool_name})" if tool_name else ""
    lines: list[str] = []
    trace_capped = " (trace capped at 20,000 chars)" if block.get("truncated") else ""

    def _generic(prefix: str) -> None:
        body, capped = _cap(text, limit)
        lines.append(f"- {prefix}{label}{trace_capped}")
        lines.extend(_fenced(body))
        if capped:
            lines.append(_pointer(file))

    if is_error:
        _generic("**ERROR** result")
        return lines
    if parsed is not None:
        if base == "run_eval" or "primary_score" in parsed:
            if parsed.get("ok") is False and parsed.get("error"):
                lines.append(f"- **ERROR** run_eval: {preview(parsed['error'], limit)}")
            elif "primary_score" in parsed:
                lines.append(f"- **result** {run_eval_result_line(parsed)}")
            else:
                lines.append(f"- **result** run_eval: {preview(text, limit)}")
            if mode == _TRANSCRIPT:
                body, capped = _cap(text, limit)
                lines.extend(_fenced(body, "json"))
                if capped:
                    lines.append(_pointer(file))
            elif len(text) > limit or "trace_meta" in parsed:
                lines.append(_pointer(file))
            return lines
        if base == "inspect_traces" or ("summary" in parsed and "ok" in parsed):
            summary = str(parsed.get("summary") or "")
            cap_n = INSPECT_SUMMARY_CHARS if mode == _DIGEST else limit
            body, capped = _cap(summary, cap_n)
            lines.append(f"- **result** inspect_traces{trace_capped}")
            lines.extend(_fenced(body))
            if capped:
                lines.append(_pointer(file))
            return lines
        if base == "close_iteration" or ("note" in parsed and "branch" in parsed):
            lines.append(
                f"- **result** close_iteration branch={parsed.get('branch')}: "
                f"{preview(parsed.get('note'), limit)}"
            )
            if mode == _TRANSCRIPT:
                lines.extend(_fenced(text, "json"))
            return lines
        if parsed.get("ok") is False and parsed.get("error"):
            lines.append(f"- **ERROR** result{label}: {preview(parsed['error'], limit)}")
            if mode == _TRANSCRIPT:
                body, capped = _cap(text, limit)
                lines.extend(_fenced(body, "json"))
                if capped:
                    lines.append(_pointer(file))
            return lines
    _generic("**result**")
    return lines


def _render_turn(
    turn_idx: int,
    file: str,
    turn: dict[str, Any] | None,
    err: str | None,
    prev_ts: datetime | None,
    names: Mapping[str, str],
    mode: str,
    prompt_ordinal: int,
) -> tuple[list[str], datetime | None]:
    if err or turn is None:
        return [f"### turn {turn_idx:04d} · (unreadable: {err})", ""], prev_ts
    ts = _parse_ts(turn.get("ts"))
    suffix = _elapsed(prev_ts, ts)
    suffix_s = f" ({suffix})" if suffix else ""
    role = turn.get("role")
    blocks = _blocks(turn)
    lines: list[str] = []

    if role == "result":
        meta = turn.get("meta") if isinstance(turn.get("meta"), dict) else {}
        lines.append(f"### turn {turn_idx:04d} · session result{suffix_s}")
        if meta.get("captured") is False:
            lines.append(f"**NOT CAPTURED**: {preview(meta.get('note'), 300)}")
            tail = meta.get("stdout_tail")
            if isinstance(tail, str) and tail.strip():
                body, _ = _cap(tail, DIGEST_RESULT_CHARS)
                lines.append("stdout tail:")
                lines.extend(_fenced(body))
        else:
            bits = [f"{k}={meta[k]}" for k in (
                "subtype", "num_turns", "num_messages", "duration_ms", "total_cost_usd",
                "model", "session_id", "session_ids",
            ) if k in meta]
            usage = turn.get("usage")
            if isinstance(usage, dict):
                bits.extend(
                    f"{k}={v:,}" if isinstance(v, int) and not isinstance(v, bool) else f"{k}={v}"
                    for k, v in usage.items()
                    if isinstance(v, (int, float))
                )
            lines.append("Session result: " + (" ".join(bits) or "(no metadata)"))
        lines.append("")
        return lines, ts or prev_ts

    if _is_prompt_turn(turn):
        meta = turn.get("meta") if isinstance(turn.get("meta"), dict) else {}
        kind = meta.get("prompt_kind")
        text = "\n\n".join(str(b.get("text") or "") for b in blocks)
        if kind == "initial" or (kind is None and prompt_ordinal == 0):
            lines.append(f"### turn {turn_idx:04d} · Prompt{suffix_s}")
        else:
            lines.append(f"### turn {turn_idx:04d} · Watchdog resume — {_watchdog_label(text)}{suffix_s}")
        lines.extend(_fenced(text))
        lines.append("")
        return lines, ts or prev_ts

    if role == "assistant":
        lines.append(f"### turn {turn_idx:04d} · assistant{suffix_s}")
        thinking_chars = 0
        for b in blocks:
            kind = b.get("kind")
            if kind == "thinking":
                text = str(b.get("text") or "")
                if mode == _TRANSCRIPT:
                    lines.append("**thinking**")
                    lines.append("")
                    lines.append(text)
                    lines.append("")
                else:
                    thinking_chars += len(text)
            elif kind == "text":
                lines.append(str(b.get("text") or ""))
                lines.append("")
            elif kind == "tool_use":
                lines.extend(_render_tool_use(b, mode, file))
        if thinking_chars and mode == _DIGEST:
            lines.insert(1, f"_(thinking omitted: {thinking_chars:,} chars — see {TRANSCRIPT_FILE})_")
        lines.append("")
        return lines, ts or prev_ts

    # user turn carrying tool results (possibly mixed with text)
    lines.append(f"### turn {turn_idx:04d} · tool results{suffix_s}")
    for b in blocks:
        kind = b.get("kind")
        if kind == "tool_result":
            tool_name = names.get(str(b.get("tool_use_id") or ""))
            lines.extend(_render_tool_result(b, tool_name, mode, file))
        elif kind == "text":
            lines.append(str(b.get("text") or ""))
            lines.append("")
    lines.append("")
    return lines, ts or prev_ts


def _wrap_lines(text: str) -> str:
    out: list[str] = []
    for line in text.split("\n"):
        if len(line) <= MD_MAX_LINE_CHARS // 2:
            out.append(line)
            continue
        out.extend(split_string(line, MD_MAX_LINE_CHARS))
    return "\n".join(out)


def _render_document(
    turns: list[tuple[int, str, dict[str, Any] | None, str | None]],
    header: Mapping[str, Any],
    stats: Mapping[str, Any],
    mode: str,
) -> str:
    lines = _render_header(header, stats, mode)
    names = _tool_name_by_id(turns)
    prev_ts: datetime | None = None
    prompt_ordinal = 0
    for turn_idx, file, turn, err in turns:
        turn_lines, prev_ts = _render_turn(
            turn_idx, file, turn, err, prev_ts, names, mode, prompt_ordinal
        )
        if _is_prompt_turn(turn):
            prompt_ordinal += 1
        lines.extend(turn_lines)
    doc = "\n".join(lines).rstrip("\n") + "\n"
    return _wrap_lines(doc)


def _atomic_write(path: str, text: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)
