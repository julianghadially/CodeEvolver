"""OpenCode JSONL output parsing."""

from __future__ import annotations

import json
import logging
from typing import Any


def parse_opencode_output(stdout: str) -> tuple[str, str | None, dict[str, Any]]:
    texts: list[str] = []
    session_id: str | None = None
    events = 0
    json_lines = 0
    tool_calls: list[dict[str, Any]] = []
    # Token accounting. OpenCode emits a `step-finish` part per LLM step whose
    # `tokens` are that step's usage (additive across steps). Assistant-message
    # `info` objects carry the `modelID`. When no step-finish is seen (OpenCode
    # issue #26855 — run can exit before the final step-finish), fall back to
    # per-message `info.tokens` deduped by message id. We never mix the two
    # sources, to avoid double counting.
    step_tokens: list[dict[str, Any]] = []
    info_tokens_by_id: dict[str, dict[str, Any]] = {}
    info_tokens_anon: list[dict[str, Any]] = []
    model_id: str | None = None
    # OpenCode's own failure events (`{"type":"error","error":{"name":…,
    # "data":{"message":…,"ref":…}}}`), kept apart from text so a session
    # that died in bootstrap is reported as an error, never as a "report".
    error_events: list[str] = []
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        json_lines += 1
        events += 1
        session_id = session_id or _extract_session_id(event)
        if event.get("type") == "error":
            error_events.append(_format_error_event(event))
            continue
        texts.extend(_extract_text_parts(event))
        tool_calls.extend(_extract_tool_calls(event))
        step_tokens.extend(_extract_step_tokens(event))
        _collect_info_tokens(event, info_tokens_by_id, info_tokens_anon)
        model_id = _extract_model_id(event) or model_id

    if not json_lines:
        return stdout, None, {"json_events": 0}

    if step_tokens:
        token_dicts: list[dict[str, Any]] = step_tokens
    else:
        # Fallback: deduped per-message info tokens (cumulative per message →
        # last-write-wins per id) plus any id-less info token blocks.
        token_dicts = list(info_tokens_by_id.values()) + info_tokens_anon

    metadata: dict[str, Any] = {"json_events": events, "tool_calls": tool_calls}
    if error_events:
        metadata["error_events"] = error_events
    if token_dicts:
        metadata["usage"] = _to_token_usage(token_dicts, model_id)
        metadata["model"] = model_id
    return ("\n".join(texts), session_id, metadata)


def log_opencode_events(
    iteration_idx: int, stdout: str, logger: logging.Logger
) -> int:
    """Log the architect's tool calls + reasoning text from OpenCode stdout.

    Emits Claude-parity lines to ``logger`` (the detailed, file-only agent
    logger), in the order they appear in the event stream so the transcript
    reads as an ordered sequence — e.g. ``run_eval`` → ``Edit`` → ``Bash`` →
    ``run_eval`` — making it visible that editing happened between evals
    (ordering, not timestamp, carries that since this runs post-session):

        iteration N: architect tool_use | Edit({...})
        iteration N: architect text | <reasoning>

    Tool calls are deduped by call id (OpenCode emits a part per status
    transition); text parts by exact string (to drop repeated/streamed copies).
    Mirrors the truncation/format of the Claude path
    (``architect_helpers._log_assistant_message``). Returns the number of lines
    emitted (0 → nothing parseable).
    """
    emitted = 0
    seen_calls: set[Any] = set()
    seen_text: set[str] = set()
    for line in stdout.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            event = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if not isinstance(event, dict):
            continue
        for text in _extract_text_parts(event):
            text = text.strip().replace("\n", " ")
            if not text or text in seen_text:
                continue
            seen_text.add(text)
            logger.info(
                "iteration %d: architect text | %s",
                iteration_idx,
                text[:500] + ("…" if len(text) > 500 else ""),
            )
            emitted += 1
        for call in _extract_tool_calls(event):
            key = call.get("call_id") or (call.get("name"), repr(call.get("input")))
            if key in seen_calls:
                continue
            seen_calls.add(key)
            preview = repr(call.get("input"))
            if len(preview) > 500:
                preview = preview[:500] + "…"
            logger.info(
                "iteration %d: architect tool_use | %s(%s)",
                iteration_idx,
                call.get("name", "?"),
                preview,
            )
            emitted += 1
    return emitted


def _extract_step_tokens(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return `tokens` dicts from any `step-finish` parts in the event.

    Handles both the bare event shape (`{"type": "step-finish", "tokens": …}`)
    and the wrapped shape (`{"part": {"type": "step-finish", "tokens": …}}`).
    """
    out: list[dict[str, Any]] = []
    for candidate in (event, event.get("part")):
        if not isinstance(candidate, dict):
            continue
        event_type = str(candidate.get("type") or "").lower()
        if event_type in ("step-finish", "step_finish"):
            tokens = candidate.get("tokens")
            if isinstance(tokens, dict):
                out.append(tokens)
    return out


def _collect_info_tokens(
    event: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    anon: list[dict[str, Any]],
) -> None:
    """Record assistant-message `info.tokens`, deduped by message id.

    Used only as a fallback when no `step-finish` parts are emitted. A message's
    `info.tokens` is cumulative for that message, so we keep the last value seen
    per message id rather than summing repeated updates.
    """
    for info in _info_candidates(event):
        if str(info.get("role") or "") != "assistant":
            continue
        tokens = info.get("tokens")
        if not isinstance(tokens, dict):
            continue
        msg_id = info.get("id")
        if isinstance(msg_id, str) and msg_id:
            by_id[msg_id] = tokens
        else:
            anon.append(tokens)


def _extract_model_id(event: dict[str, Any]) -> str | None:
    """Pull `modelID` from the event or any nested assistant-message info."""
    for candidate in (event, *_info_candidates(event)):
        model = candidate.get("modelID")
        if isinstance(model, str) and model:
            return model
    return None


def _info_candidates(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Dicts that may carry an assistant-message `info` (modelID + tokens)."""
    out: list[dict[str, Any]] = []
    info = event.get("info")
    if isinstance(info, dict):
        out.append(info)
    message = event.get("message")
    if isinstance(message, dict):
        out.append(message)
        nested = message.get("info")
        if isinstance(nested, dict):
            out.append(nested)
    return out


def _to_token_usage(
    token_dicts: list[dict[str, Any]], model_id: str | None
) -> dict[str, Any]:
    """Sum OpenCode token dicts into CodeEvolver's normalized usage shape.

    OpenCode tokens: `{input, output, reasoning, cache: {read, write}}`. We fold
    `reasoning` into `output_tokens` so our `total` matches OpenCode's reported
    total (input + output + reasoning + cache.read + cache.write).
    """
    input_tokens = output_tokens = cache_write = cache_read = 0
    for tokens in token_dicts:
        input_tokens += _as_int(tokens.get("input"))
        output_tokens += _as_int(tokens.get("output")) + _as_int(
            tokens.get("reasoning")
        )
        cache = tokens.get("cache")
        if isinstance(cache, dict):
            cache_write += _as_int(cache.get("write"))
            cache_read += _as_int(cache.get("read"))
    usage: dict[str, Any] = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cache_creation_input_tokens": cache_write,
        "cache_read_input_tokens": cache_read,
        "total": input_tokens + output_tokens + cache_write + cache_read,
        "model": model_id,
    }
    return usage


def _as_int(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _extract_session_id(event: dict[str, Any]) -> str | None:
    for key in ("session_id", "sessionId", "sessionID", "id"):
        value = event.get(key)
        if isinstance(value, str) and value:
            return value
    session = event.get("session")
    if isinstance(session, dict):
        for key in ("id", "session_id", "sessionId", "sessionID"):
            value = session.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _format_error_event(event: dict[str, Any]) -> str:
    """`NAME: message (ref)` for an OpenCode error event; tolerant of shape."""
    err = event.get("error")
    if not isinstance(err, dict):
        return str(err if err is not None else event)[:500]
    data = err.get("data") if isinstance(err.get("data"), dict) else {}
    name = err.get("name") or "error"
    message = data.get("message") or err.get("message") or ""
    ref = data.get("ref")
    text = f"{name}: {message}" if message else str(name)
    return f"{text} (ref {ref})" if ref else text


def _extract_text_parts(event: dict[str, Any]) -> list[str]:
    values: list[str] = []
    part = event.get("part")
    if isinstance(part, dict):
        values.extend(_texts_from_parts([part]))
    message = event.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            values.append(content)
        parts = message.get("parts")
        if isinstance(parts, list):
            values.extend(_texts_from_parts(parts))
    for key in ("text", "content", "output"):
        value = event.get(key)
        if isinstance(value, str):
            values.append(value)
    return values


def _extract_tool_calls(event: dict[str, Any]) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    candidates: list[Any] = [event]
    part = event.get("part")
    if isinstance(part, dict):
        candidates.append(part)
    message = event.get("message")
    if isinstance(message, dict):
        candidates.append(message)
        parts = message.get("parts")
        if isinstance(parts, list):
            candidates.extend(parts)

    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = (
            item.get("tool")
            or item.get("name")
            or item.get("toolName")
            or item.get("tool_name")
        )
        item_type = str(item.get("type") or "").lower()
        if not isinstance(name, str):
            continue
        if "tool" not in item_type and name.lower() not in {
            "edit",
            "write",
            "patch",
            "apply_patch",
            "read",
            "grep",
            "glob",
            "bash",
        }:
            continue
        state = item.get("state")
        if not isinstance(state, dict):
            state = {}
        call: dict[str, Any] = {
            "name": name,
            "input": state.get("input")
            or item.get("input")
            or item.get("args")
            or item.get("arguments")
            or {},
        }
        call_id = item.get("callID") or item.get("call_id") or item.get("id")
        if isinstance(call_id, str) and call_id:
            call["call_id"] = call_id
        status = state.get("status")
        if isinstance(status, str) and status:
            call["status"] = status
        if "output" in state:
            call["output"] = state.get("output")
        calls.append(call)
    return calls


def _merge_result_turns(
    turns: list[dict[str, Any]], session_ids: list[str] | None = None
) -> list[dict[str, Any]]:
    """Collapse per-session ``role="result"`` turns into one trailing turn.

    `_capture_architect_trace` concatenates one `serialize_opencode_export`
    result per session id, and each appends its own trailing result turn. On
    a watchdog resume that strands a result line mid-file carrying only that
    session's usage, while `meta_reflection` documents a single trailing
    result line for the whole iteration. Usage counters are summed; `meta`
    comes from the last result turn, with every session id recorded.

    A no-op for the single-session case.
    """
    results = [t for t in turns if t.get("role") == "result"]
    if len(results) <= 1:
        return turns

    body = [t for t in turns if t.get("role") != "result"]
    totals: dict[str, int] = {}
    model = None
    for turn in results:
        usage = turn.get("usage")
        if not isinstance(usage, dict):
            continue
        model = usage.get("model") or model
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            totals[key] = totals.get(key, 0) + value

    merged = dict(results[-1])
    meta = dict(merged.get("meta") or {})
    meta["num_messages"] = sum(
        int((t.get("meta") or {}).get("num_messages") or 0) for t in results
    )
    if session_ids:
        meta["session_ids"] = list(session_ids)
    merged["meta"] = meta
    if totals:
        if model is not None:
            totals["model"] = model  # type: ignore[assignment]
        merged["usage"] = totals
    body.append(merged)
    return body


def _tool_error_text(error: Any) -> Any:
    """Extract legible text from an OpenCode tool part's ``state["error"]``.

    Real errored tool parts carry ``state`` keys
    ``['status', 'input', 'error', 'time']`` with NO ``output`` key — the
    message is a plain string (occasionally a small dict) under ``error``.
    Returns a payload for ``cap_tool_result_text``, which json-dumps anything
    that is not a string.
    """
    if error is None:
        return ""
    if isinstance(error, str):
        return error
    if isinstance(error, dict):
        for key in ("message", "output", "error", "text", "stderr"):
            value = error.get(key)
            if isinstance(value, str) and value.strip():
                return value
        data = error.get("data")
        if isinstance(data, dict):
            value = data.get("message")
            if isinstance(value, str) and value.strip():
                return value
    return error  # cap_tool_result_text json-dumps it


def serialize_opencode_export(
    export_doc: Any,
    *,
    model: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Map an ``opencode export`` session document into architect-trace turns.

    The export is a consolidated end-of-session snapshot:
    ``{"info": {...session...}, "messages": [{"info": {...}, "parts": [...]}]}``
    with first-class ``reasoning`` parts, ``tool`` parts carrying terminal
    ``state`` (status/input/output), and per-step ``step-finish`` tokens.
    Unlike ``--format json`` stdout there are no streaming part-transition
    re-emissions, so no dedupe is needed — messages are walked in order.

    Returns turn dicts (``{"role", "blocks"}`` + optional ``usage``/``meta``)
    in the ``architect_trace`` JSONL schema, ready for
    ``architect_trace.write_opencode_trace``: one assistant line per assistant
    message (thinking / text / tool_use blocks), followed by a user line with
    that message's terminal tool_results; user prompt messages (the initial
    prompt and watchdog re-prompts) become user lines with text blocks; one
    trailing ``role="result"`` line carries summed token usage and session
    meta. Returns ``[]`` when nothing serializable (caller falls back to the
    ``captured:false`` stub).

    Reasoning caveat: some providers (e.g. OpenAI) return reasoning as
    encrypted content with an empty ``text`` — such parts carry no legible
    thinking and are skipped; thinking capture is best-effort by design.
    """
    from src.engine.optimizer.v3.architect_trace import cap_tool_result_text

    if not isinstance(export_doc, dict):
        return []
    session_info = export_doc.get("info")
    if isinstance(session_info, dict):
        sid = session_info.get("id")
        if not session_id and isinstance(sid, str) and sid:
            session_id = sid
    messages = export_doc.get("messages")
    if not isinstance(messages, list):
        return []

    turns: list[dict[str, Any]] = []
    step_tokens: list[dict[str, Any]] = []
    model_id: str | None = model
    num_messages = 0
    for message in messages:
        if not isinstance(message, dict):
            continue
        info = message.get("info")
        info = info if isinstance(info, dict) else {}
        role = str(info.get("role") or "")
        raw_parts = message.get("parts")
        raw_parts = raw_parts if isinstance(raw_parts, list) else []
        found_model = info.get("modelID")
        if not model_id and isinstance(found_model, str) and found_model:
            model_id = found_model

        assistant_blocks: list[dict[str, Any]] = []
        tool_results: list[dict[str, Any]] = []
        user_blocks: list[dict[str, Any]] = []
        for part in raw_parts:
            if not isinstance(part, dict):
                continue
            part_type = str(part.get("type") or "").lower()
            if part_type in ("step-start", "step_start"):
                continue
            if part_type in ("step-finish", "step_finish"):
                tokens = part.get("tokens")
                if isinstance(tokens, dict):
                    step_tokens.append(tokens)
                continue
            if part_type == "reasoning":
                text = part.get("text") or part.get("content")
                if isinstance(text, str) and text.strip():
                    assistant_blocks.append(
                        {
                            "kind": "thinking",
                            "text": text,
                            "signature_present": _reasoning_signature_present(part),
                        }
                    )
                continue
            if "tool" in part_type:
                state = part.get("state")
                state = state if isinstance(state, dict) else {}
                call_id = (
                    part.get("callID") or part.get("call_id") or part.get("id")
                )
                assistant_blocks.append(
                    {
                        "kind": "tool_use",
                        "name": part.get("tool") or part.get("name") or "?",
                        "id": call_id,
                        "input": state.get("input") or part.get("input") or {},
                    }
                )
                status = str(state.get("status") or "").lower()
                has_error = status == "error" or bool(state.get("error"))
                if (
                    status in ("completed", "error")
                    or "output" in state
                    or "error" in state
                ):
                    # `output` wins when it carries text, but real errored
                    # parts have NO `output` key at all — the message lives in
                    # `state["error"]`. Reading only `output` serialized every
                    # failed tool call as `is_error: true, content: ""`,
                    # blanking the most diagnostic event in a trace;
                    # meta_reflection greps '"is_error": true' to find exactly
                    # these. The whitespace check matters too: a part with
                    # `output: ""` alongside a real error must fall through.
                    payload = state.get("output")
                    if payload is None or (
                        isinstance(payload, str) and not payload.strip()
                    ):
                        payload = _tool_error_text(state.get("error"))
                    content, truncated = cap_tool_result_text(payload)
                    tool_results.append(
                        {
                            "kind": "tool_result",
                            "tool_use_id": call_id,
                            "is_error": has_error,
                            "content": content,
                            "truncated": truncated,
                        }
                    )
                continue
            if part_type == "patch":
                # `patch` parts name the files a write touched. Without this
                # branch they would fall through to the text fallback and
                # vanish, while `_collect_tool_calls` treats patches as a
                # first-class edit signal, so trace and collector would disagree
                # about whether edits happened. Emitted as a text block to
                # keep the schema meta_reflection is told about unchanged.
                files = part.get("files")
                if isinstance(files, dict):
                    files = sorted(files.keys())
                if isinstance(files, list) and files:
                    assistant_blocks.append(
                        {
                            "kind": "text",
                            "text": "[patch] " + ", ".join(
                                str(f) for f in files[:50]
                            ),
                        }
                    )
                continue
            # `text` parts and any unknown part type carrying legible text.
            text = part.get("text") or part.get("content")
            if isinstance(text, str) and text:
                target = assistant_blocks if role == "assistant" else user_blocks
                target.append({"kind": "text", "text": text})

        ts = _message_ts(info)
        if role == "assistant":
            if assistant_blocks:
                turns.append(_with_ts({"role": "assistant", "blocks": assistant_blocks}, ts))
                num_messages += 1
            if tool_results:
                turns.append(_with_ts({"role": "user", "blocks": tool_results}, ts))
        elif role == "user" and user_blocks:
            turns.append(_with_ts({"role": "user", "blocks": user_blocks}, ts))
            num_messages += 1

    if not turns:
        return []
    result_turn: dict[str, Any] = {
        "role": "result",
        "blocks": [],
        "meta": {
            "harness": "opencode",
            "captured": True,
            "session_id": session_id,
            "model": model_id,
            "num_messages": num_messages,
        },
    }
    if step_tokens:
        result_turn["usage"] = _to_token_usage(step_tokens, model_id)
    turns.append(result_turn)
    return turns


def _message_ts(info: dict[str, Any]) -> str | None:
    """ISO-8601 UTC timestamp of an export message (``info.time.created``,
    epoch milliseconds), or None when the export carries none. Without this
    every turn of a post-session export is stamped with the export time and
    the trace has no per-turn timing at all."""
    time_info = info.get("time")
    if not isinstance(time_info, dict):
        return None
    created = time_info.get("created") or time_info.get("completed")
    if not isinstance(created, (int, float)) or created <= 0:
        return None
    try:
        from datetime import datetime, timezone

        return datetime.fromtimestamp(created / 1000.0, tz=timezone.utc).isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _with_ts(turn: dict[str, Any], ts: str | None) -> dict[str, Any]:
    if ts:
        turn["ts"] = ts
    return turn


def _reasoning_signature_present(part: dict[str, Any]) -> bool:
    """True when a reasoning part carries opaque provider-side content
    (e.g. OpenAI's ``reasoningEncryptedContent``) — the OpenCode analog of the
    Claude thinking signature, recorded as a flag and never persisted."""
    metadata = part.get("metadata")
    if not isinstance(metadata, dict):
        return False
    for provider_meta in metadata.values():
        if isinstance(provider_meta, dict) and provider_meta.get(
            "reasoningEncryptedContent"
        ):
            return True
    return False


def _texts_from_parts(parts: list[Any]) -> list[str]:
    texts: list[str] = []
    for part in parts:
        if isinstance(part, str):
            texts.append(part)
            continue
        if not isinstance(part, dict):
            continue
        text = part.get("text") or part.get("content")
        if isinstance(text, str):
            texts.append(text)
    return texts
