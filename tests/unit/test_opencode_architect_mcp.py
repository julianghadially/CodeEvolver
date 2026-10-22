"""Copyright © 2026 440 Labs LLC

Regression tests for the OpenCode architect's `ce` MCP tool server.

Background: OpenCode spawns the local MCP proxy with cwd set to the CLIENT
workspace. A client repo that ships its own `src/` package (the common case)
can shadow CodeEvolver's `src` on sys.path, so `python -m
src.engine.agent.harnesses.opencode_mcp_proxy` dies with
"No module named 'src.engine'". OpenCode logs only
`server unavailable key=ce status=failed` and runs the session anyway, so the
architect gets a whole iteration with no run_eval / close_iteration tools
while the optimizer sees a "successful" session.

These tests pin both halves of the fix: the proxy must start under a shadowing
cwd, and the preflight must refuse to invoke the model when it does not.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.engine.agent.harnesses.opencode.architect_mcp import (
    OpenCodeArchitectControlServer,
    OpenCodeArchitectToolBridge,
)
from src.engine.agent.harnesses.opencode.harness import (
    OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
    OpenCodeHarness,
)

REPO_ROOT = str(Path(__file__).resolve().parents[2])
ARCHITECT_TOOLS = ("run_eval", "coding_agent", "inspect_traces", "close_iteration")

_HANDSHAKE = "".join(
    json.dumps(message) + "\n"
    for message in (
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
)


def _bridge() -> OpenCodeArchitectToolBridge:
    return OpenCodeArchitectToolBridge(
        {name: (lambda **kwargs: {"ok": True}) for name in ARCHITECT_TOOLS}
    )


@pytest.fixture
def client_repo_with_src(tmp_path: Path) -> Path:
    """A client workspace that ships its own `src/` package (e.g. FactChecker)."""
    src = tmp_path / "src"
    (src / "factchecker").mkdir(parents=True)
    (src / "__init__.py").write_text("", encoding="utf-8")
    (src / "factchecker" / "__init__.py").write_text("", encoding="utf-8")
    return tmp_path


def _proxy_env(control: OpenCodeArchitectControlServer) -> dict[str, str]:
    return {
        "PYTHONPATH": REPO_ROOT,
        "PYTHONSAFEPATH": "1",
        "CODEEVOLVER_ARCHITECT_MCP_PROXY_URL": control.url,
        "CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN": control.token,
        "CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS": "600",
    }


def _listed_tools(stdout: bytes) -> set[str]:
    listed: set[str] = set()
    for line in stdout.decode("utf-8", errors="replace").splitlines():
        try:
            tools = json.loads(line).get("result", {}).get("tools")
        except json.JSONDecodeError:
            continue
        if isinstance(tools, list):
            listed = {tool["name"] for tool in tools}
    return listed


def test_proxy_lists_tools_when_client_cwd_ships_its_own_src(
    client_repo_with_src: Path,
) -> None:
    """The bug: a client `src/` shadowed CodeEvolver's and killed the server."""
    with OpenCodeArchitectControlServer(_bridge()) as control:
        completed = subprocess.run(
            [sys.executable, OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT],
            input=_HANDSHAKE.encode("utf-8"),
            capture_output=True,
            env=_proxy_env(control),
            cwd=str(client_repo_with_src),
            timeout=60,
        )
    assert completed.returncode == 0, completed.stderr.decode()
    assert _listed_tools(completed.stdout) == set(ARCHITECT_TOOLS)


def test_preflight_passes_with_the_real_proxy(client_repo_with_src: Path) -> None:
    bridge = _bridge()
    harness = OpenCodeHarness(architect_tool_registry=dict(bridge.tool_registry))
    with OpenCodeArchitectControlServer(bridge) as control:
        env_overrides = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(
                {
                    "mcp": {
                        "ce": {
                            "command": [
                                sys.executable,
                                OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
                            ],
                            "environment": _proxy_env(control),
                        }
                    }
                }
            )
        }
        harness._preflight_architect_mcp(
            env_overrides=env_overrides,
            bridge=bridge,
            cwd=str(client_repo_with_src),
        )


def test_preflight_raises_when_the_tool_server_cannot_start(tmp_path: Path) -> None:
    """A dead MCP server must abort BEFORE the (billable) model invocation."""
    bridge = _bridge()
    harness = OpenCodeHarness(architect_tool_registry=dict(bridge.tool_registry))
    env_overrides = {
        "OPENCODE_CONFIG_CONTENT": json.dumps(
            {
                "mcp": {
                    "ce": {
                        # Stands in for any startup failure (the real one was
                        # ModuleNotFoundError from the shadowed `src`).
                        "command": [sys.executable, "-c", "raise SystemExit(1)"],
                        "environment": {},
                    }
                }
            }
        )
    }
    with pytest.raises(RuntimeError) as excinfo:
        harness._preflight_architect_mcp(
            env_overrides=env_overrides, bridge=bridge, cwd=str(tmp_path)
        )
    message = str(excinfo.value)
    assert "close_iteration" in message and "run_eval" in message


def test_preflight_is_skipped_on_the_injected_runner_path(tmp_path: Path) -> None:
    """Tests inject a fake runner; there is no real subprocess to preflight."""
    bridge = _bridge()
    harness = OpenCodeHarness(
        architect_tool_registry=dict(bridge.tool_registry),
        runner=lambda *args, **kwargs: subprocess.CompletedProcess([], 0, "", ""),
    )
    harness._preflight_architect_mcp(
        env_overrides={}, bridge=bridge, cwd=str(tmp_path)
    )
