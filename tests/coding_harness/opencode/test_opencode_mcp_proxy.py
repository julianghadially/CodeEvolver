"""Round-trip tests for the OpenCode architect MCP stdio proxy.

The architect's ee MCP server is a LOCAL stdio server: opencode spawns
`opencode_mcp_proxy.main()`, which forwards each MCP `tools/call` over loopback
HTTP to the in-process `OpenCodeArchitectControlServer`. These tests drive
`main()` in-process against a real control server (no opencode, no model calls)
to prove the stdio<->loopback bridge round-trips and survives a tool failure.
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
from typing import Any

from src.engine.agent.harnesses import opencode_mcp_proxy
from src.engine.agent.harnesses.opencode.architect_mcp import (
    OpenCodeArchitectControlServer,
    OpenCodeArchitectToolBridge,
)
from src.engine.agent.harnesses.opencode.harness import (
    OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS,
    OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
)
from src.engine.paths import engine_root


class _FakeStd:
    """Stand-in for sys.stdin / sys.stdout exposing a BytesIO `.buffer`."""

    def __init__(self, data: bytes = b"") -> None:
        self.buffer = io.BytesIO(data)


def _frame(messages: list[dict[str, Any]]) -> bytes:
    """Encode MCP messages as newline-delimited single-line JSON.

    `_read_message` treats a line starting with `{` as a whole JSON-RPC message,
    so newline framing is the simplest way to feed a sequence to the proxy.
    """
    return b"".join(
        json.dumps(m, separators=(",", ":")).encode("utf-8") + b"\n" for m in messages
    )


def _parse_framed(data: bytes) -> list[dict[str, Any]]:
    """Parse the proxy's stdout exactly as opencode's MCP stdio reader does:
    newline-delimited JSON, one message per line.

    The proxy MUST NOT use LSP `Content-Length` framing — opencode's reader
    splits on newlines and would choke on a header line, dropping the whole
    server (the architect would then see zero `ce` tools). We assert that here
    so the framing can never regress to Content-Length.
    """
    assert b"Content-Length" not in data, (
        "proxy emitted LSP Content-Length framing; MCP stdio requires "
        "newline-delimited JSON"
    )
    out: list[dict[str, Any]] = []
    for line in data.split(b"\n"):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _drive_proxy(
    bridge: OpenCodeArchitectToolBridge,
    messages: list[dict[str, Any]],
    monkeypatch,
) -> list[dict[str, Any]]:
    """Run `opencode_mcp_proxy.main()` against a live control server."""
    with OpenCodeArchitectControlServer(bridge) as control:
        monkeypatch.setenv("CODEEVOLVER_ARCHITECT_MCP_PROXY_URL", control.url)
        monkeypatch.setenv("CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN", control.token)
        monkeypatch.setenv("CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS", "3600")
        fake_out = _FakeStd()
        monkeypatch.setattr(opencode_mcp_proxy.sys, "stdin", _FakeStd(_frame(messages)))
        monkeypatch.setattr(opencode_mcp_proxy.sys, "stdout", fake_out)
        opencode_mcp_proxy.main()
    return _parse_framed(fake_out.buffer.getvalue())


def test_proxy_roundtrips_tools_list_and_run_eval(monkeypatch):
    eval_calls: list[dict[str, Any]] = []

    def run_eval(eval_sample: str, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        eval_calls.append({"eval_sample": eval_sample, "filter": filter})
        return {"ok": True, "primary_score": 0.42, "eval_sample": eval_sample}

    bridge = OpenCodeArchitectToolBridge(
        {"run_eval": run_eval}, agent_managed_git=False
    )

    responses = _drive_proxy(
        bridge,
        [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "run_eval", "arguments": {"eval_sample": "subsample"}},
            },
        ],
        monkeypatch,
    )

    by_id = {r["id"]: r for r in responses}

    # initialize handshake answered locally by the proxy.
    assert by_id[0]["result"]["serverInfo"]["name"] == "codeevolver-architect"

    # tools/list served from the list the proxy fetched via GET /tools at startup,
    # and it matches the control server's source-of-truth bridge.list_tools().
    listed = [t["name"] for t in by_id[1]["result"]["tools"]]
    assert listed == [t["name"] for t in bridge.list_tools()] == ["run_eval"]

    # tools/call round-trips through POST /call: the tool actually ran, and its
    # result is returned verbatim in the MCP content payload.
    assert eval_calls == [{"eval_sample": "subsample", "filter": None}]
    call_result = by_id[2]["result"]
    assert call_result.get("isError") is not True
    assert json.loads(call_result["content"][0]["text"]) == {
        "ok": True,
        "primary_score": 0.42,
        "eval_sample": "subsample",
    }


def test_proxy_returns_iserror_without_crashing_on_tool_failure(monkeypatch):
    later_calls: list[str] = []

    def boom(eval_sample: str, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        raise RuntimeError("eval blew up")

    def inspect_traces(query: str) -> dict[str, Any]:
        later_calls.append(query)
        return {"ok": True, "summary": "ok"}

    bridge = OpenCodeArchitectToolBridge(
        {"run_eval": boom, "inspect_traces": inspect_traces},
        agent_managed_git=False,
    )

    responses = _drive_proxy(
        bridge,
        [
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": "run_eval", "arguments": {"eval_sample": "single"}},
            },
            # A second call after the failure proves the stdio loop survived the
            # control-server error rather than crashing the whole MCP server.
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "inspect_traces", "arguments": {"query": "why"}},
            },
        ],
        monkeypatch,
    )

    by_id = {r["id"]: r for r in responses}

    failed = by_id[1]["result"]
    assert failed["isError"] is True
    assert "eval blew up" in json.loads(failed["content"][0]["text"])["error"]

    assert later_calls == ["why"]
    assert by_id[2]["result"].get("isError") is not True
    assert json.loads(by_id[2]["result"]["content"][0]["text"]) == {
        "ok": True,
        "summary": "ok",
    }


def test_proxy_subprocess_spawn_matches_opencode(tmp_path):
    """Spawn the proxy the way opencode does — `[sys.executable, -m, module]`
    with ONLY the MCP `environment` block (a clean env, no inherited PATH/etc.) —
    and parse its stdout as opencode's newline-delimited MCP stdio reader.

    This is the end-to-end shape the in-process tests can't cover: it exercises
    the real subprocess, the `python -m` import under `PYTHONPATH`, and — most
    importantly — the on-the-wire framing. (The framing bug that left the
    architect with zero `ce` tools passed the in-process tests because they
    drove `main()` directly; only a faithful newline parse of real subprocess
    output catches it.)
    """
    bridge = OpenCodeArchitectToolBridge(
        {"run_eval": lambda eval_sample, filter=None: {"ok": True, "primary_score": 0.5}},
        agent_managed_git=False,
    )

    # opencode runs the child from the CLIENT workspace, and real client repos
    # ship their own `src/` package, which can shadow CodeEvolver's `src`
    # and kill the proxy on startup. Reproduce that layout so this test exercises
    # the launch form that survives it.
    (tmp_path / "src" / "clientpkg").mkdir(parents=True)
    (tmp_path / "src" / "__init__.py").write_text("", encoding="utf-8")
    (tmp_path / "src" / "clientpkg" / "__init__.py").write_text("", encoding="utf-8")

    with OpenCodeArchitectControlServer(bridge) as control:
        child_env = {
            "PYTHONPATH": engine_root(),
            # Absolute script path + PYTHONSAFEPATH keep the client cwd off
            # sys.path so only PYTHONPATH resolves `src`.
            "PYTHONSAFEPATH": "1",
            "CODEEVOLVER_ARCHITECT_MCP_PROXY_URL": control.url,
            "CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN": control.token,
            "CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS": str(
                OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS
            ),
        }
        proc = subprocess.Popen(
            [sys.executable, OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT],
            cwd=str(tmp_path),
            env=child_env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        messages = [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "run_eval", "arguments": {"eval_sample": "subsample"}},
            },
        ]
        stdin_bytes = b"".join(
            json.dumps(m, separators=(",", ":")).encode("utf-8") + b"\n" for m in messages
        )
        out, err = proc.communicate(input=stdin_bytes, timeout=40)

    assert proc.returncode == 0, err.decode(errors="replace")
    responses = _parse_framed(out)  # asserts newline (not Content-Length) framing
    by_id = {r["id"]: r for r in responses}

    assert by_id[0]["result"]["serverInfo"]["name"] == "codeevolver-architect"
    assert [t["name"] for t in by_id[1]["result"]["tools"]] == ["run_eval"]
    assert json.loads(by_id[2]["result"]["content"][0]["text"]) == {
        "ok": True,
        "primary_score": 0.5,
    }


def test_proxy_forwards_acknowledge_message_and_renders_refusal_as_non_error(monkeypatch):
    """specs/archive-public/comparison_credits.md: the acknowledgement refusal is a NORMAL
    tool result (`ok: False` dict), never `isError` — the model must read it
    as instructions, not as a failure. And the new optional
    `acknowledge_message` argument must reach the tool unchanged."""
    calls: list[dict[str, Any]] = []

    def run_eval(eval_sample: str, filter: dict[str, Any] | None = None,
                 acknowledge_message: str | None = None) -> dict[str, Any]:
        calls.append({"eval_sample": eval_sample, "acknowledge_message": acknowledge_message})
        if acknowledge_message != "42":
            return {
                "ok": False,
                "refused": "acknowledgement_required",
                "verification_code": "42",
            }
        return {"ok": True, "primary_score": 0.7, "comparison_credits_remaining": 0}

    bridge = OpenCodeArchitectToolBridge({"run_eval": run_eval}, agent_managed_git=False)
    responses = _drive_proxy(
        bridge,
        [
            {"jsonrpc": "2.0", "id": 0, "method": "initialize"},
            {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
             "params": {"name": "run_eval", "arguments": {"eval_sample": "comparison"}}},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
             "params": {"name": "run_eval",
                        "arguments": {"eval_sample": "comparison", "acknowledge_message": "42"}}},
        ],
        monkeypatch,
    )
    by_id = {r["id"]: r for r in responses}

    schema = next(t for t in by_id[1]["result"]["tools"] if t["name"] == "run_eval")["inputSchema"]
    assert "acknowledge_message" in schema["properties"]
    assert "acknowledge_message" not in schema["required"]

    assert calls == [
        {"eval_sample": "comparison", "acknowledge_message": None},
        {"eval_sample": "comparison", "acknowledge_message": "42"},
    ]
    refusal = by_id[2]["result"]
    assert refusal.get("isError") is not True
    assert json.loads(refusal["content"][0]["text"])["refused"] == "acknowledgement_required"
    ok = json.loads(by_id[3]["result"]["content"][0]["text"])
    assert ok["ok"] is True and ok["comparison_credits_remaining"] == 0
