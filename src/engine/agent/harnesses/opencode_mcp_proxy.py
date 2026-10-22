"""Local MCP stdio proxy for OpenCode architect tools.

The optimizer process owns tool execution and iteration state. OpenCode starts
this module as a local MCP server; each MCP tool call is forwarded to the
optimizer-owned localhost control server.
"""

from __future__ import annotations

import json
import socket
import sys
import urllib.error
import urllib.request
from typing import Any

from src.engine.paths import (
    architect_mcp_call_timeout_seconds,
    architect_mcp_proxy_token,
    architect_mcp_proxy_url,
)

# Tool discovery (GET /tools at startup) is a loopback call to an
# already-listening in-process control server, so it returns instantly; keep its
# timeout short so a misconfigured/missing server fails fast instead of hanging.
_DISCOVERY_TIMEOUT_SECONDS = 30


def main() -> None:
    proxy_url = architect_mcp_proxy_url()
    token = architect_mcp_proxy_token()
    # POST /call (tool execution) can block for a very long time (run_eval /
    # coding_agent / inspect_traces); use the long, harness-supplied timeout so
    # the loopback read never gives up before the tool finishes.
    call_timeout = architect_mcp_call_timeout_seconds()
    tools = _http_json(
        "GET", f"{proxy_url}/tools", token, timeout=_DISCOVERY_TIMEOUT_SECONDS
    ).get("tools", [])

    while True:
        message = _read_message()
        if message is None:
            return
        method = message.get("method")
        request_id = message.get("id")

        if method == "notifications/initialized":
            continue
        if method == "initialize":
            _write_result(
                request_id,
                {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "codeevolver-architect",
                        "version": "0.1.0",
                    },
                },
            )
            continue
        if method == "tools/list":
            _write_result(request_id, {"tools": tools})
            continue
        if method == "tools/call":
            params = message.get("params") or {}
            response = _http_json(
                "POST",
                f"{proxy_url}/call",
                token,
                {
                    "name": params.get("name"),
                    "arguments": params.get("arguments") or {},
                },
                timeout=call_timeout,
            )
            if response.get("ok"):
                _write_result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(response.get("result"), default=str),
                            }
                        ]
                    },
                )
            else:
                _write_result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "ok": False,
                                        "error": response.get("error", "tool failed"),
                                    }
                                ),
                            }
                        ],
                        "isError": True,
                    },
                )
            continue

        if request_id is not None:
            _write_error(request_id, -32601, f"Unsupported MCP method: {method}")


def _http_json(
    method: str,
    url: str,
    token: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float,
) -> dict[str, Any]:
    data = None
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read() or b"{}")
        except json.JSONDecodeError:
            return {"ok": False, "error": str(exc)}
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as exc:
        # Transport-level failure (timeout, connection refused, reset). Return a
        # structured error so the caller surfaces an MCP isError to OpenCode and
        # the stdio loop keeps running — an uncaught raise here would crash the
        # proxy and tear down the MCP server for the rest of the session.
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _read_message() -> dict[str, Any] | None:
    first = sys.stdin.buffer.readline()
    if not first:
        return None

    if first.lstrip().startswith(b"{"):
        return json.loads(first)

    headers: dict[str, str] = {}
    line = first
    while line and line.strip():
        raw = line.decode("ascii", errors="replace").strip()
        if ":" in raw:
            key, value = raw.split(":", 1)
            headers[key.lower()] = value.strip()
        line = sys.stdin.buffer.readline()

    content_length = int(headers.get("content-length", "0"))
    if content_length <= 0:
        return None
    return json.loads(sys.stdin.buffer.read(content_length))


def _write_result(request_id: Any, result: dict[str, Any]) -> None:
    _write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


def _write_error(request_id: Any, code: int, message: str) -> None:
    _write_message(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {"code": code, "message": message},
        }
    )


def _write_message(payload: dict[str, Any]) -> None:
    # MCP's stdio transport frames messages as newline-delimited JSON — one
    # JSON object per line, no embedded newlines, NO `Content-Length` headers
    # (that is LSP framing, which opencode's MCP stdio reader cannot parse — it
    # would choke on the header line and drop the whole server, leaving the
    # architect with zero `ce` tools). `json.dumps` escapes any newline inside
    # string values, so compact single-line output is always newline-safe.
    body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(body)
    sys.stdout.buffer.write(b"\n")
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
