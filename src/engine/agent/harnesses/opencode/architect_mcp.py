"""OpenCode architect MCP transport bridge."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from src.engine.agent.harnesses.architect_tools import (
    mcp_architect_tool_definitions,
)

logger = logging.getLogger(__name__)


class OpenCodeArchitectToolBridge:
    """Python-owned architect MCP tool surface.

    OpenCode talks to a local MCP process; that process forwards tool calls to
    this bridge through the in-process control server so optimizer state and
    guards stay owned by Python.
    """

    def __init__(
        self,
        tool_registry: dict[str, Callable[..., dict]],
        *,
        agent_managed_git: bool = True,
        close_iteration_mode: str = "register_branch",
    ) -> None:
        self.tool_registry = dict(tool_registry)
        self.agent_managed_git = agent_managed_git
        self.close_iteration_mode = close_iteration_mode

    def list_tools(self) -> list[dict[str, Any]]:
        return mcp_architect_tool_definitions(
            agent_managed_git=self.agent_managed_git,
            registered_names=set(self.tool_registry),
            close_iteration_mode=self.close_iteration_mode,  # type: ignore[arg-type]
        )

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        allowed = {tool["name"] for tool in self.list_tools()}
        if name not in allowed:
            raise KeyError(f"OpenCode architect MCP tool {name!r} is not registered")
        return self.tool_registry[name](**arguments)

    def handle_mcp_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        method = str(message.get("method") or "")
        request_id = message.get("id")

        if method == "notifications/initialized":
            return None
        if method == "initialize":
            return _mcp_result(
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
        if method == "tools/list":
            return _mcp_result(request_id, {"tools": self.list_tools()})
        if method == "tools/call":
            params = dict(message.get("params") or {})
            try:
                result = self.call_tool(
                    str(params.get("name") or ""),
                    dict(params.get("arguments") or {}),
                )
                return _mcp_result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, default=str),
                            }
                        ]
                    },
                )
            except Exception as exc:  # noqa: BLE001
                return _mcp_result(
                    request_id,
                    {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {
                                        "ok": False,
                                        "error": f"{type(exc).__name__}: {exc}",
                                    }
                                ),
                            }
                        ],
                        "isError": True,
                    },
                )
        return _mcp_error(request_id, -32601, f"Unsupported MCP method: {method}")


class OpenCodeArchitectControlServer:
    """Tiny localhost HTTP bridge used by the OpenCode MCP proxy."""

    def __init__(
        self,
        bridge: OpenCodeArchitectToolBridge,
        *,
        pre_edit_guard: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.bridge = bridge
        self.pre_edit_guard = pre_edit_guard
        self.token = uuid.uuid4().hex
        self._server = ThreadingHTTPServer(
            ("127.0.0.1", 0),
            self._make_handler(),
        )
        self.url = f"http://127.0.0.1:{self._server.server_port}"
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            name="opencode-architect-mcp-control",
            daemon=True,
        )

    def __enter__(self) -> "OpenCodeArchitectControlServer":
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=5)

    def _make_handler(self):
        bridge = self.bridge
        pre_edit_guard = self.pre_edit_guard
        token = self.token

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send_json({"ok": False, "error": "unauthorized"}, status=401)
                    return
                if self.path != "/tools":
                    self._send_json({"ok": False, "error": "not found"}, status=404)
                    return
                self._send_json({"ok": True, "tools": bridge.list_tools()})

            def do_POST(self) -> None:  # noqa: N802
                if not self._authorized():
                    self._send_json({"ok": False, "error": "unauthorized"}, status=401)
                    return
                path = self.path.split("?", 1)[0]
                if path not in {"/call", "/guard/pre-edit", "/mcp"}:
                    self._send_json({"ok": False, "error": "not found"}, status=404)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    payload = json.loads(self.rfile.read(length) or b"{}")
                    if path == "/mcp":
                        # Legacy JSON-RPC-over-HTTP path. The architect now uses
                        # a LOCAL stdio MCP server (opencode_mcp_proxy.py), which
                        # hits GET /tools + POST /call — not this endpoint. Kept
                        # (and unit-tested via the bridge) for direct callers.
                        response = bridge.handle_mcp_message(dict(payload))
                        if response is None:
                            self._send_json({})
                        else:
                            self._send_json(response)
                        return
                    if path == "/guard/pre-edit":
                        if pre_edit_guard is None:
                            result = {"allowed": True, "reason": None}
                        else:
                            result = pre_edit_guard(
                                str(payload.get("tool") or ""),
                                dict(payload.get("arguments") or {}),
                            )
                    else:
                        result = bridge.call_tool(
                            str(payload.get("name") or ""),
                            dict(payload.get("arguments") or {}),
                        )
                except Exception as exc:  # noqa: BLE001
                    self._send_json(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    return
                self._send_json({"ok": True, "result": result})

            def log_message(self, format: str, *args) -> None:  # noqa: A002
                return

            def _authorized(self) -> bool:
                auth = self.headers.get("Authorization", "")
                header_token = self.headers.get("X-CodeEvolver-Token", "")
                return auth == f"Bearer {token}" or header_token == token

            def _send_json(self, payload: dict[str, Any], *, status: int = 200) -> None:
                body = json.dumps(payload, default=str).encode("utf-8")
                try:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except (BrokenPipeError, ConnectionError) as exc:
                    # opencode aborted/closed the request before we finished the
                    # (possibly multi-minute) tool call and wrote the response —
                    # e.g. an MCP execution timeout (-32001). There is nothing to
                    # write to; don't let ThreadingHTTPServer dump a BrokenPipe
                    # traceback. The tool already ran; its result is reflected in
                    # the optimizer's own state/logs. Log a concise WARNING (no
                    # stack) — after the experimental.mcp_timeout fix this should
                    # be rare, so when it fires it's a real signal a tool overran
                    # opencode's timeout or the client disconnected unexpectedly.
                    logger.warning(
                        f"{type(exc).__name__} in architect MCP control server: client closed the "
                        "connection before the response was sent (%s %s) — the "
                        "tool ran but opencode likely timed out or aborted it: "
                        "%s: %s",
                        self.command,
                        self.path,
                        type(exc).__name__,
                        exc,
                    )
                    return

        return Handler


class InertOpenCodeArchitectControlServer:
    """Control-server stand-in for request-construction tests."""

    def __init__(
        self,
        bridge: OpenCodeArchitectToolBridge,
        *,
        pre_edit_guard: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
    ) -> None:
        self.bridge = bridge
        self.pre_edit_guard = pre_edit_guard
        self.token = uuid.uuid4().hex
        self.url = "http://127.0.0.1:0"

    def __enter__(self) -> "InertOpenCodeArchitectControlServer":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def _mcp_result(request_id: Any, result: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _mcp_error(request_id: Any, code: int, message: str) -> dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }
