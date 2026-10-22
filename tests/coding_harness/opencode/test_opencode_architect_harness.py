"""Tests for OpenCode architect tool execution."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

def _seed_trace_dir(trace_dir):
    """Minimal eval trace directory: the registry checks for `index.jsonl`."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "index.jsonl").write_text("{}\n")



class RecordingRunner:
    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, "kwargs": kwargs})
        return self.result


def _evaluator_config(*, testset_path: str = "/data/test.jsonl"):
    from src.engine.evaluator.config import EvaluatorConfig

    return EvaluatorConfig(
        program_factory="pkg.program:Program",
        metric_fns=["pkg.metric:metric"],
        metric_names=["score"],
        primary_metric="score",
        testset_path=testset_path,
    )


def _architect_config(
    *,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = "openai/gpt-5.3-codex-spark",
    agent_managed_git: bool = True,
    testset_path: str = "/data/test.jsonl",
):
    from src.engine.optimizer.v3.iteration_architect import ArchitectConfig

    return ArchitectConfig(
        run_timestamp="20260604",
        additional_instructions=None,
        program_path="pkg.program:Program",
        evaluator=_evaluator_config(testset_path=testset_path),
        trainset_ids=[1],
        comparison_ids=[2],
        max_turns=17,
        coding_harness=coding_harness,  # type: ignore[arg-type]
        coding_harness_model=coding_harness_model,
        agent_managed_git=agent_managed_git,
        system_prompt_builder=lambda **kwargs: "system prompt",
    )


def test_opencode_architect_request_registers_ce_mcp_and_guard_permissions(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "session.updated",
                    "sessionID": "architect-session-1",
                }
            )
            + "\n"
            + json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}})
            + "\n",
            stderr="",
        )
    )
    testset_path = str(tmp_path / "holdout_test.jsonl")
    harness = OpenCodeHarness(
        model="openai/gpt-5.3-codex-spark",
        runner=runner,
        architect_context=IterationContext(iteration_idx=3),
        architect_config=_architect_config(testset_path=testset_path),
        architect_tool_registry={
            "run_eval": lambda eval_sample, filter=None: {"ok": True},
            "coding_agent": lambda mode, change_request: {"ok": True},
            "inspect_traces": lambda query: {"ok": True},
            "close_iteration": lambda register_branch, improvement_mode, memory_updated: {
                "ok": True
            },
        },
    )

    result = harness.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            max_turns=5,
            raw_provider_metadata={"system_prompt": "system prompt"},
        )
    )

    assert result.success is True
    assert result.session_id == "architect-session-1"
    assert len(runner.calls) == 1
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["opencode", "run"]
    assert cmd[cmd.index("--dir") + 1] == str(tmp_path)
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-5.3-codex-spark"
    assert cmd[cmd.index("--format") + 1] == "json"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[-1].startswith("/s\n\n")
    assert "architect prompt" in cmd[-1]
    assert call["kwargs"]["cwd"] == str(tmp_path)
    assert call["kwargs"]["timeout"] == 5 * 120

    opencode_config_dir = Path(call["kwargs"]["env"]["OPENCODE_CONFIG_DIR"])
    config = json.loads(call["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == "openai/gpt-4.1-mini"
    server = config["mcp"]["ce"]
    # LOCAL stdio MCP server: opencode spawns the proxy script under the current
    # interpreter; the proxy forwards tool calls over loopback HTTP to the
    # in-process control server. stdio (not remote HTTP) is what lets a long,
    # quiet run_eval block without an idle-timeout cut.
    import sys

    from src.engine.agent.harnesses.opencode.harness import (
        OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS,
        OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
        OPENCODE_ARCHITECT_MCP_TIMEOUT_MS,
        OPENCODE_PROVIDER_STREAM_TIMEOUT_MS,
    )

    assert server["type"] == "local"
    assert server["enabled"] is True
    # ABSOLUTE SCRIPT PATH, never `-m <module>`: opencode spawns this with cwd =
    # the client workspace, and `-m` would put that cwd ahead of PYTHONPATH on
    # sys.path — a client repo shipping its own `src/` then shadows CodeEvolver's
    # and the proxy dies with "No module named 'src.engine'", leaving the
    # architect with zero ce tools (opencode only WARNs and runs anyway).
    # See tests/unit/test_opencode_architect_mcp.py.
    assert server["command"] == [sys.executable, OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT]
    env_block = server["environment"]
    assert env_block["PYTHONPATH"]
    # Belt and braces with the absolute path above: keeps the script dir / cwd
    # off sys.path entirely so ONLY PYTHONPATH resolves `src`.
    assert env_block["PYTHONSAFEPATH"] == "1"
    assert env_block["CODEEVOLVER_ARCHITECT_MCP_PROXY_URL"].startswith("http://127.0.0.1:")
    assert env_block["CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN"]
    assert env_block["CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS"] == str(
        OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS
    )
    # Both MCP timeouts set high: the per-server `mcp.ce.timeout` is read first
    # in opencode's tool-exec timeout resolution, and the top-level
    # `experimental.mcp_timeout` is what it passes to callTool().
    assert server["timeout"] == OPENCODE_ARCHITECT_MCP_TIMEOUT_MS
    assert config["experimental"]["mcp_timeout"] == OPENCODE_ARCHITECT_MCP_TIMEOUT_MS
    # Provider streaming timeout — the knob that actually lets a >5min run_eval
    # survive on opencode >=1.17.11. Keyed off the model's provider prefix
    # (here "openai/gpt-5.3-codex-spark" -> "openai").
    provider_opts = config["provider"]["openai"]["options"]
    assert provider_opts["chunkTimeout"] == OPENCODE_PROVIDER_STREAM_TIMEOUT_MS
    assert provider_opts["headerTimeout"] == OPENCODE_PROVIDER_STREAM_TIMEOUT_MS
    assert provider_opts["requestTimeout"] == OPENCODE_PROVIDER_STREAM_TIMEOUT_MS
    assert config["permission"]["question"] == "deny"
    assert config["permission"]["edit"]["*"] == "allow"
    assert config["permission"]["read"][f"*{testset_path}*"] == "deny"
    assert config["permission"]["edit"][f"*{testset_path}*"] == "deny"
    assert config["permission"]["read"]["*_holdout*"] == "deny"
    # The architect's own meta-traces are fenced (only the end-of-run
    # meta-reflection agent may read them) — parity with the Claude
    # holdout-guard hook's `_architect_trace` fragment.
    assert config["permission"]["read"]["*_architect_trace*"] == "deny"
    assert config["permission"]["grep"]["*_architect_trace*"] == "deny"
    assert config["permission"]["edit"]["*_architect_trace*"] == "deny"
    # OpenCode only enforces the policy when it is also exported as
    # OPENCODE_PERMISSION; without it the holdout fence is inert (the architect
    # could read the held-out test set). Assert parity with the in-config policy.
    assert json.loads(call["kwargs"]["env"]["OPENCODE_PERMISSION"]) == (
        config["permission"]
    )
    assert opencode_config_dir
    # The proxy URL/token are also exported at the subprocess level for the JS
    # pre-edit-guard plugin (which reads process.env); they must match the values
    # embedded in the MCP `environment` block the proxy itself reads.
    assert (
        call["kwargs"]["env"]["CODEEVOLVER_ARCHITECT_MCP_PROXY_URL"]
        == env_block["CODEEVOLVER_ARCHITECT_MCP_PROXY_URL"]
    )
    assert (
        call["kwargs"]["env"]["CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN"]
        == env_block["CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN"]
    )
    plugin_path = (
        opencode_config_dir / "plugins" / "codeevolver-pre-edit-guard.js"
    )
    plugin_text = plugin_path.read_text(encoding="utf-8")
    assert "tool.execute.before" in plugin_text
    assert "/guard/pre-edit" in plugin_text

    instruction_paths = config["instructions"]
    assert len(instruction_paths) == 1
    instruction_path = opencode_config_dir / "codeevolver" / "architect_system_prompt.md"
    assert instruction_path.read_text() == "system prompt"
    assert instruction_paths[0] == str(instruction_path)
    assert not (tmp_path / ".opencode").exists()


def test_opencode_architect_surfaces_token_usage_and_model(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps({"type": "session.updated", "sessionID": "arch-1"}),
                    json.dumps(
                        {"info": {"role": "assistant", "id": "m1", "modelID": "gpt-5.4-mini"}}
                    ),
                    json.dumps(
                        {
                            "type": "step-finish",
                            "tokens": {
                                "input": 500,
                                "output": 100,
                                "reasoning": 25,
                                "cache": {"write": 10, "read": 300},
                            },
                        }
                    ),
                    json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}}),
                ]
            )
            + "\n",
            stderr="",
        )
    )
    harness = OpenCodeHarness(
        model="openai/gpt-5.3-codex-spark",
        runner=runner,
        architect_context=IterationContext(iteration_idx=3),
        architect_config=_architect_config(testset_path=str(tmp_path / "holdout_test.jsonl")),
        architect_tool_registry={
            "run_eval": lambda eval_sample, filter=None: {"ok": True},
            "coding_agent": lambda mode, change_request: {"ok": True},
            "inspect_traces": lambda query: {"ok": True},
            "close_iteration": lambda register_branch, improvement_mode, memory_updated: {
                "ok": True
            },
        },
    )

    result = harness.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            max_turns=5,
            raw_provider_metadata={"system_prompt": "system prompt"},
        )
    )

    # The architect path reads top-level model + usage off raw_provider_metadata.
    assert result.raw_provider_metadata["model"] == "gpt-5.4-mini"
    usage = result.raw_provider_metadata["usage"]
    assert usage["input_tokens"] == 500
    assert usage["output_tokens"] == 125  # output + reasoning
    assert usage["cache_creation_input_tokens"] == 10
    assert usage["cache_read_input_tokens"] == 300
    assert usage["total"] == 500 + 125 + 10 + 300


def test_opencode_architect_small_model_can_be_overridden(monkeypatch, tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    monkeypatch.setenv("CODEEVOLVER_OPENCODE_SMALL_MODEL", "openai/gpt-4.1-nano")
    runner = RecordingRunner(
        subprocess.CompletedProcess(args=["opencode"], returncode=0, stdout="done", stderr="")
    )
    harness = OpenCodeHarness(
        runner=runner,
        architect_context=IterationContext(iteration_idx=3),
        architect_config=_architect_config(),
        architect_tool_registry={
            "run_eval": lambda eval_sample, filter=None: {"ok": True},
            "coding_agent": lambda mode, change_request: {"ok": True},
            "inspect_traces": lambda query: {"ok": True},
        },
    )

    harness.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            raw_provider_metadata={"system_prompt": "system prompt"},
        )
    )

    config = json.loads(runner.calls[0]["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == "openai/gpt-4.1-nano"


def test_opencode_architect_mcp_bridge_dispatches_python_owned_tool_contracts():
    from src.engine.agent.harnesses.architect_tools import ARCHITECT_TOOL_SPECS
    from src.engine.agent.harnesses.opencode.architect_mcp import (
        OpenCodeArchitectToolBridge,
    )
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=4)
    calls: list[tuple[str, dict[str, Any]]] = []

    def run_eval(eval_sample: str, filter: dict[str, Any] | None = None) -> dict[str, Any]:
        ctx.record_eval("sha-after-eval", 0.75)
        calls.append(("run_eval", {"eval_sample": eval_sample, "filter": filter}))
        return {"ok": True, "primary_score": 0.75}

    def coding_agent(mode: str, change_request: str) -> dict[str, Any]:
        ctx.record_coding_agent(change_request)
        calls.append(
            (
                "coding_agent",
                {"mode": mode, "change_request": change_request},
            )
        )
        return {"ok": True, "output": "coded"}

    def inspect_traces(query: str) -> dict[str, Any]:
        calls.append(("inspect_traces", {"query": query}))
        return {"ok": True, "summary": "summary"}

    def close_iteration(
        register_branch: dict[str, Any],
        improvement_mode: str,
        memory_updated: bool,
    ) -> dict[str, Any]:
        ctx.close_called = True
        calls.append(
            (
                "close_iteration",
                {
                    "register_branch": register_branch,
                    "improvement_mode": improvement_mode,
                    "memory_updated": memory_updated,
                },
            )
        )
        return {"ok": True}

    bridge = OpenCodeArchitectToolBridge(
        {
            "run_eval": run_eval,
            "coding_agent": coding_agent,
            "inspect_traces": inspect_traces,
            "close_iteration": close_iteration,
        },
        agent_managed_git=True,
    )

    assert [tool["name"] for tool in bridge.list_tools()] == [
        "run_eval",
        "coding_agent",
        "inspect_traces",
        "close_iteration",
    ]
    for tool in bridge.list_tools():
        spec = ARCHITECT_TOOL_SPECS[tool["name"]]
        assert tool["description"] == spec.description
        assert tool["inputSchema"] == spec.input_schema
    assert bridge.call_tool(
        "run_eval",
        {"eval_sample": "train", "filter": {"category": "hard"}},
    )["primary_score"] == 0.75
    assert bridge.call_tool(
        "coding_agent",
        {"mode": "execute", "change_request": "make a change"},
    )["output"] == "coded"
    assert bridge.call_tool("inspect_traces", {"query": "inspect"})["summary"] == "summary"
    register_branch = {
        "branch": "codeevolver-approach-a",
        "was_primary": True,
        "is_primary": True,
    }
    assert bridge.call_tool(
        "close_iteration",
        {
            "register_branch": register_branch,
            "improvement_mode": "exploitation",
            "memory_updated": True,
        },
    )["ok"] is True

    assert ctx.eval_history == [("sha-after-eval", 0.75)]
    assert ctx.coding_agent_called is True
    assert ctx.close_called is True
    assert calls == [
        ("run_eval", {"eval_sample": "train", "filter": {"category": "hard"}}),
        (
            "coding_agent",
            {"mode": "execute", "change_request": "make a change"},
        ),
        ("inspect_traces", {"query": "inspect"}),
        (
            "close_iteration",
            {
                "register_branch": register_branch,
                "improvement_mode": "exploitation",
                "memory_updated": True,
            },
        ),
    ]


def test_opencode_architect_mcp_bridge_handles_remote_json_rpc_protocol():
    from src.engine.agent.harnesses.opencode.architect_mcp import (
        OpenCodeArchitectToolBridge,
    )

    calls: list[dict[str, Any]] = []

    def close_iteration(
        register_branch: dict[str, Any],
        improvement_mode: str,
        memory_updated: bool,
    ) -> dict[str, Any]:
        calls.append(
            {
                "register_branch": register_branch,
                "improvement_mode": improvement_mode,
                "memory_updated": memory_updated,
            }
        )
        return {"ok": True, "improvement_mode": improvement_mode}

    bridge = OpenCodeArchitectToolBridge(
        {"close_iteration": close_iteration},
        agent_managed_git=True,
    )

    initialize = bridge.handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 0,
            "method": "initialize",
            "params": {"protocolVersion": "2025-11-25"},
        }
    )
    tools = bridge.handle_mcp_message(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    )
    register_branch = {
        "branch": "codeevolver-approach-a",
        "was_primary": False,
        "is_primary": False,
    }
    call = bridge.handle_mcp_message(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "close_iteration",
                "arguments": {
                    "register_branch": register_branch,
                    "improvement_mode": "exploration",
                    "memory_updated": True,
                },
            },
        }
    )

    assert initialize["result"]["capabilities"] == {"tools": {}}
    assert tools["result"]["tools"][0]["name"] == "close_iteration"
    assert json.loads(call["result"]["content"][0]["text"]) == {
        "ok": True,
        "improvement_mode": "exploration",
    }
    assert calls == [
        {
            "register_branch": register_branch,
            "improvement_mode": "exploration",
            "memory_updated": True,
        }
    ]


def test_opencode_pre_edit_guard_passes_edits_through(tmp_path):
    """edit/write carry no control now — the guard only fences apply_patch.

    An ordering rule here (no edits before the baseline) would leak through
    Bash; the per-iteration baseline requirement is a `close_iteration` check. edit/write are still fenced against holdout paths
    by OpenCode's permission deny patterns.
    """
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=2)  # not locked, no baseline
    harness = OpenCodeHarness(architect_context=ctx)
    guard = harness._build_pre_edit_guard(str(tmp_path))

    assert guard("edit", {"filePath": "src/program.py"}) == {
        "allowed": True, "reason": None,
    }
    assert guard("write", {"filePath": "src/program.py"})["allowed"] is True
    assert guard("bash", {"command": "sed -i 's/a/b/' src/program.py"})["allowed"] is True


def test_opencode_pre_edit_guard_allows_apply_patch_with_holdout_fence(tmp_path):
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=2)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.5
    harness = OpenCodeHarness(architect_context=ctx)
    guard = harness._build_pre_edit_guard(str(tmp_path), ["holdout.jsonl"])

    good_patch = (
        "*** Begin Patch\n"
        f"*** Update File: {tmp_path}/src/program.py\n"
        "@@\n-x\n+y\n"
        "*** End Patch\n"
    )
    allowed = guard("apply_patch", {"patchText": good_patch})
    assert allowed == {"allowed": True, "reason": None}

    # apply_patch targeting the holdout (valset/test) file is denied — the fence
    # is preserved even though apply_patch is now an allowed edit tool.
    holdout_patch = (
        "*** Begin Patch\n"
        f"*** Update File: {tmp_path}/data/holdout.jsonl\n"
        "@@\n-x\n+y\n"
        "*** End Patch\n"
    )
    denied = guard("apply_patch", {"patchText": holdout_patch})
    assert denied["allowed"] is False
    assert "holdout" in denied["reason"]


def test_opencode_pre_edit_guard_denies_architect_trace_targets(tmp_path):
    """apply_patch targeting an architect meta-trace is denied — the
    `_architect_trace` fragment rides the same deny list as the holdout fence."""
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=2)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.5
    harness = OpenCodeHarness(architect_context=ctx)
    guard = harness._build_pre_edit_guard(str(tmp_path), ["_architect_trace"])

    trace_patch = (
        "*** Begin Patch\n"
        "*** Update File: /architect_traces/job1/iteration_1_architect_trace/turn_0001.json\n"
        "@@\n-x\n+y\n"
        "*** End Patch\n"
    )
    denied = guard("apply_patch", {"patchText": trace_patch})
    assert denied["allowed"] is False


def test_opencode_pre_edit_guard_apply_patch_needs_no_baseline(tmp_path):
    """apply_patch is gated on holdout targets, not on iteration state."""
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=2)  # not locked, no baseline
    harness = OpenCodeHarness(architect_context=ctx)
    guard = harness._build_pre_edit_guard(str(tmp_path), ["holdout.jsonl"])

    patch = (
        "*** Begin Patch\n"
        f"*** Update File: {tmp_path}/src/program.py\n"
        "@@\n-x\n+y\n"
        "*** End Patch\n"
    )
    assert guard("apply_patch", {"patchText": patch})["allowed"] is True


def test_opencode_pre_edit_guard_plugin_routes_apply_patch_not_blocks(tmp_path):
    """The JS plugin must route apply_patch through the guard, not hard-block it."""
    from src.engine.agent.harnesses.opencode.harness import _write_pre_edit_guard_plugin

    _write_pre_edit_guard_plugin(tmp_path)
    plugin_text = (tmp_path / "plugins" / "codeevolver-pre-edit-guard.js").read_text()
    assert "apply_patch is not available" not in plugin_text
    assert 'input.tool !== "apply_patch"' in plugin_text


def test_opencode_agent_managed_git_false_hides_close_iteration(tmp_path):
    """Runs with agent_managed_git=False expose no close_iteration."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    runner = RecordingRunner(
        subprocess.CompletedProcess(args=["opencode"], returncode=0, stdout="done", stderr="")
    )
    harness = OpenCodeHarness(
        runner=runner,
        architect_context=IterationContext(iteration_idx=5),
        architect_config=_architect_config(agent_managed_git=False),
        architect_tool_registry={
            "run_eval": lambda eval_sample, filter=None: {"ok": True},
            "coding_agent": lambda mode, change_request: {"ok": True},
            "inspect_traces": lambda query: {"ok": True},
        },
    )

    harness.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            raw_provider_metadata={"system_prompt": "system prompt"},
        )
    )

    tool_names = runner.calls[0]["kwargs"]["env"][
        "CODEEVOLVER_ARCHITECT_MCP_TOOL_NAMES"
    ]
    assert "close_iteration" not in tool_names
    assert tool_names == "run_eval,coding_agent,inspect_traces"
    config = json.loads(runner.calls[0]["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["mcp"]["ce"]["enabled"] is True


def test_opencode_architect_tools_delegate_coding_and_inspection_to_selected_harness(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import AgentRole, HarnessResult
    from src.engine.agent.harnesses.opencode.architect_mcp import (
        OpenCodeArchitectToolBridge,
    )
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        primary_branch: str = "codeevolver-main"

    class CapturingHarness:
        def __init__(self):
            self.calls = []

        def run_coding(self, request):
            self.calls.append(request)
            return HarnessResult.successful(
                output="coded",
                raw_provider_metadata={"commit_handled": True},
            )

        def run_inspection(self, request):
            self.calls.append(request)
            return HarnessResult.successful(output="summary")

    harness = CapturingHarness()
    ctx = IterationContext(iteration_idx=9)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.1
    architect = IterationArchitect(
        context=ctx,
        state=State(),
        workspace=str(tmp_path),
        config=_architect_config(),
    )
    bridge = OpenCodeArchitectToolBridge(
        architect.build_tool_registry(agent_harness=harness),
        agent_managed_git=True,
    )

    coding = bridge.call_tool(
        "coding_agent",
        {"mode": "execute", "change_request": "make a change"},
    )
    traces = tmp_path / "traces"
    traces.mkdir()
    _seed_trace_dir(traces / "iteration_9_parent")
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(traces))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    inspection = bridge.call_tool(
        "inspect_traces", {"query": "inspect traces", "trace": "parent"}
    )

    assert coding == {"ok": True, "error": None, "output": "coded"}
    assert inspection["ok"] is True, inspection
    assert [request.role for request in harness.calls] == [
        AgentRole.CODING,
        AgentRole.INSPECTION,
    ]
    assert "coding_harness" not in harness.calls[0].raw_provider_metadata
    assert "coding_harness_model" not in harness.calls[1].raw_provider_metadata


def test_opencode_mode_routes_architect_execution_and_close_missing_watchdog_resume(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import AgentRole, HarnessResult
    from src.engine.optimizer.v3 import iteration_architect as module
    from src.engine.optimizer.v3 import architect_watchdogs as wd_module
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        primary_branch: str = "codeevolver-main"

    class CapturingHarness:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.calls: list[tuple[Any, ...]] = []

        def run_architect(self, request):
            self.calls.append(("run", request))
            return HarnessResult.successful(
                output="ok",
                session_id="opencode-session-42",
            )

        def resume_architect(self, session_id, request):
            self.calls.append(("resume", session_id, request))
            return HarnessResult.successful(output="watchdog", session_id=session_id)

    memory = tmp_path / "memory"
    memory.mkdir()
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(memory))
    fake = CapturingHarness()
    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: fake,
        raising=False,
    )
    monkeypatch.setattr(wd_module, "git_is_dirty", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "git_add_and_commit", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "git_get_sha", lambda *args, **kwargs: "head-sha")
    monkeypatch.setattr(wd_module, "git_get_sha", lambda *args, **kwargs: "head-sha")
    # The session ended without a close_iteration call: the close-missing
    # watchdog resumes the captured session and demands a close.
    ctx = IterationContext(iteration_idx=6)
    ctx.record_eval("baseline", 0.1)
    ctx.record_eval("head-sha", 0.2)
    # HEAD moved off the iteration-start sha, so the no-change watchdog stays
    # quiet and the close-missing watchdog is the one that resumes.
    ctx.start_sha = "start-sha"

    architect = IterationArchitect(
        context=ctx,
        state=State(),
        workspace=str(tmp_path),
        config=_architect_config(),
    )
    architect.run()

    assert ctx.watchdog_fired is True
    assert architect._last_session_id == "opencode-session-42"
    assert [call[0] for call in fake.calls] == ["run", "resume"]
    run_request = fake.calls[0][1]
    assert run_request.role == AgentRole.ARCHITECT
    assert run_request.raw_provider_metadata["system_prompt"] == "system prompt"
    _, session_id, resume_request = fake.calls[1]
    assert session_id == "opencode-session-42"
    assert "close_iteration" in resume_request.prompt


class DispatchingRunner:
    """Routes `opencode run` and `opencode export` calls to canned results."""

    def __init__(self, run_result, export_results=None):
        self.run_result = run_result
        # session_id -> CompletedProcess (or list of them, consumed in order)
        self.export_results = dict(export_results or {})
        self.calls: list[dict[str, Any]] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, "kwargs": kwargs})
        if cmd[:2] == ["opencode", "export"]:
            result = self.export_results.get(cmd[2])
            if isinstance(result, list):
                return result.pop(0)
            if result is not None:
                return result
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="session not found"
            )
        return self.run_result

    def export_calls(self):
        return [c for c in self.calls if c["cmd"][:2] == ["opencode", "export"]]


def _architect_run_stdout(session_id="ses_arch1"):
    return (
        json.dumps({"type": "session.updated", "sessionID": session_id})
        + "\n"
        + json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}})
        + "\n"
    )


def _export_document(messages):
    return {"info": {"id": "ses_arch1"}, "messages": messages}


def _export_process(document):
    return subprocess.CompletedProcess(
        args=["opencode", "export"],
        returncode=0,
        stdout=json.dumps(document),
        stderr="Exporting session: ses_arch1",
    )


def _meta_harness(tmp_path, runner):
    from src.engine.agent.harnesses.opencode import OpenCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    config = _architect_config(testset_path=str(tmp_path / "holdout_test.jsonl"))
    config.capture_architect_traces = True
    return OpenCodeHarness(
        model="openai/gpt-5.3-codex-spark",
        runner=runner,
        architect_context=IterationContext(iteration_idx=4),
        architect_config=config,
        architect_tool_registry={
            "run_eval": lambda eval_sample, filter=None: {"ok": True},
            "coding_agent": lambda mode, change_request: {"ok": True},
            "inspect_traces": lambda query: {"ok": True},
            "close_iteration": lambda register_branch, improvement_mode, memory_updated: {
                "ok": True
            },
        },
    )


def _run_architect(harness, tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest

    return harness.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            max_turns=5,
            raw_provider_metadata={"system_prompt": "system prompt"},
        )
    )


def _read_trace_lines(iteration_idx):
    from src.engine.optimizer.v3.architect_trace import read_architect_trace

    return read_architect_trace(iteration_idx)


def test_opencode_meta_reflection_capture_exports_session_into_trace(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    document = _export_document(
        [
            {
                "info": {"role": "assistant", "modelID": "gpt-5.4-mini"},
                "parts": [
                    {"type": "reasoning", "text": "try reranking"},
                    {
                        "type": "tool",
                        "tool": "ce_run_eval",
                        "callID": "call_1",
                        "state": {
                            "status": "completed",
                            "input": {"eval_sample": "subsample"},
                            "output": '{"primary_score": 0.7}',
                        },
                    },
                ],
            }
        ]
    )
    runner = DispatchingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=_architect_run_stdout(),
            stderr="",
        ),
        export_results={"ses_arch1": _export_process(document)},
    )
    harness = _meta_harness(tmp_path, runner)

    result = _run_architect(harness, tmp_path)

    assert result.success is True
    assert [c["cmd"][2] for c in runner.export_calls()] == ["ses_arch1"]
    lines = _read_trace_lines(4)
    assert [ln["role"] for ln in lines] == ["assistant", "user", "result"]
    kinds = [b["kind"] for b in lines[0]["blocks"]]
    assert kinds == ["thinking", "tool_use"]
    assert lines[1]["blocks"][0]["tool_use_id"] == "call_1"
    assert lines[2]["meta"]["captured"] is True
    assert lines[2]["meta"]["session_id"] == "ses_arch1"


def test_opencode_meta_reflection_export_failure_falls_back_to_stub(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    runner = DispatchingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=_architect_run_stdout(),
            stderr="",
        ),
        export_results={},  # export returns rc=1
    )
    harness = _meta_harness(tmp_path, runner)

    _run_architect(harness, tmp_path)

    lines = _read_trace_lines(4)
    assert len(lines) == 1
    assert lines[0]["meta"]["captured"] is False
    assert lines[0]["meta"]["harness"] == "opencode"


def test_opencode_meta_reflection_no_session_id_writes_stub_without_export(
    monkeypatch, tmp_path
):
    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    runner = DispatchingRunner(
        subprocess.CompletedProcess(
            args=["opencode"], returncode=0, stdout="no json here", stderr=""
        ),
    )
    harness = _meta_harness(tmp_path, runner)

    _run_architect(harness, tmp_path)

    assert runner.export_calls() == []
    lines = _read_trace_lines(4)
    assert lines[0]["meta"]["captured"] is False


def test_opencode_meta_reflection_resume_rewrites_full_session_once(
    monkeypatch, tmp_path
):
    """A watchdog resume re-exports the (now longer) session; the trace file is
    rewritten as one full-session snapshot — no duplicated turns."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest

    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    first_doc = _export_document(
        [
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "text", "text": "first pass"}],
            }
        ]
    )
    full_doc = _export_document(
        [
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "text", "text": "first pass"}],
            },
            {
                "info": {"role": "user"},
                "parts": [{"type": "text", "text": "close_iteration required"}],
            },
            {
                "info": {"role": "assistant"},
                "parts": [{"type": "text", "text": "closing"}],
            },
        ]
    )
    runner = DispatchingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=_architect_run_stdout(),
            stderr="",
        ),
        export_results={
            "ses_arch1": [_export_process(first_doc), _export_process(full_doc)]
        },
    )
    harness = _meta_harness(tmp_path, runner)

    _run_architect(harness, tmp_path)
    harness.resume_architect(
        "ses_arch1",
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="watchdog re-prompt",
            max_turns=5,
            raw_provider_metadata={"system_prompt": "system prompt"},
        ),
    )

    assert [c["cmd"][2] for c in runner.export_calls()] == ["ses_arch1", "ses_arch1"]
    lines = _read_trace_lines(4)
    texts = [
        b["text"]
        for ln in lines
        for b in ln["blocks"]
        if b.get("kind") == "text"
    ]
    assert texts == ["first pass", "close_iteration required", "closing"]
    assert [ln["role"] for ln in lines] == ["assistant", "user", "assistant", "result"]


def test_provider_stream_timeout_config_keys_off_model_provider():
    from src.engine.agent.harnesses.opencode.harness import (
        OPENCODE_PROVIDER_STREAM_TIMEOUT_MS,
        _provider_stream_timeout_config,
    )

    openai = _provider_stream_timeout_config("openai/gpt-5.4-mini")
    assert openai["provider"]["openai"]["options"] == {
        "chunkTimeout": OPENCODE_PROVIDER_STREAM_TIMEOUT_MS,
        "headerTimeout": OPENCODE_PROVIDER_STREAM_TIMEOUT_MS,
        "requestTimeout": OPENCODE_PROVIDER_STREAM_TIMEOUT_MS,
    }
    # Provider id is everything before the FIRST slash (model id may contain more).
    deepinfra = _provider_stream_timeout_config("deepinfra/zai-org/GLM-5.2")
    assert list(deepinfra["provider"].keys()) == ["deepinfra"]
    # No model / no provider prefix -> no-op (don't emit a bogus provider block).
    assert _provider_stream_timeout_config(None) == {}
    assert _provider_stream_timeout_config("gpt-4") == {}


def test_opencode_version_meets_minimum():
    from src.engine.agent.harnesses.opencode.harness import (
        OPENCODE_MINIMUM_VERSION,
        _opencode_version_meets_minimum,
    )

    assert OPENCODE_MINIMUM_VERSION == (1, 17, 11)
    assert _opencode_version_meets_minimum("1.17.3") is False
    assert _opencode_version_meets_minimum("1.17.10") is False
    assert _opencode_version_meets_minimum("1.17.11") is True
    assert _opencode_version_meets_minimum("1.18.0") is True
    assert _opencode_version_meets_minimum("2.0.0") is True
    # Noisy / prefixed output: first dotted triple wins.
    assert _opencode_version_meets_minimum("opencode 1.20.2") is True
    # Unparseable -> None (caller warns and proceeds rather than blocking).
    assert _opencode_version_meets_minimum("dev") is None
    assert _opencode_version_meets_minimum("") is None


def test_verify_opencode_version_once_raises_on_old_version(monkeypatch):
    import src.engine.agent.harnesses.opencode.harness as h

    monkeypatch.setattr(h, "_OPENCODE_VERSION_CHECKED", False)
    monkeypatch.delenv("CODEEVOLVER_SKIP_OPENCODE_VERSION_CHECK", raising=False)

    class _Proc:
        stdout = "1.17.3"
        stderr = ""

    monkeypatch.setattr(h.subprocess, "run", lambda *a, **k: _Proc())
    with pytest.raises(RuntimeError, match="too old"):
        h._verify_opencode_version_once()

    # New-enough version: no raise, and the check caches (runs once).
    monkeypatch.setattr(h, "_OPENCODE_VERSION_CHECKED", False)

    class _NewProc:
        stdout = "1.17.11"
        stderr = ""

    calls = {"n": 0}

    def _run(*a, **k):
        calls["n"] += 1
        return _NewProc()

    monkeypatch.setattr(h.subprocess, "run", _run)
    h._verify_opencode_version_once()
    h._verify_opencode_version_once()
    assert calls["n"] == 1  # cached after first success

    # Skip env var bypasses entirely (no subprocess call).
    monkeypatch.setattr(h, "_OPENCODE_VERSION_CHECKED", False)
    monkeypatch.setenv("CODEEVOLVER_SKIP_OPENCODE_VERSION_CHECK", "1")

    def _boom(*a, **k):
        raise AssertionError("should not run opencode --version when skipped")

    monkeypatch.setattr(h.subprocess, "run", _boom)
    h._verify_opencode_version_once()
