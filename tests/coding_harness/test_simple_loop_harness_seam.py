"""Copyright © 2026 440 Labs LLC

Harness seam for the simple_loop arm (Baseline 0).

simple_loop runs the architect with `agent_managed_git=False` and
`stage_project_skills=False`: `inspect_traces` / `close_iteration` drop out of
the tool surface when not registered, and CodeEvolver skills plus the OpenCode
bootstrap are suppressed, while the engine 3 arms keep the full set.
"""

from __future__ import annotations

from src.engine.agent.harnesses import AgentRole
from src.engine.agent.harnesses.architect_tools import (
    mcp_architect_tool_definitions,
)
from src.engine.agent.harnesses.base import HarnessRequest
from src.engine.agent.harnesses.claude import ClaudeCodeHarness
from src.engine.agent.harnesses.opencode.architect_mcp import (
    OpenCodeArchitectToolBridge,
)
from src.engine.agent.harnesses.opencode.harness import OpenCodeHarness
from src.engine.optimizer.v3.iteration_architect import ArchitectConfig


def _cfg(**kw) -> ArchitectConfig:
    base = dict(
        run_timestamp="t",
        additional_instructions=None,
        program_path="p.m:C",
        evaluator=None,
        trainset_ids=[],
        comparison_ids=[],
        max_turns=10,
    )
    base.update(kw)
    return ArchitectConfig(**base)


def _architect_request():
    return HarnessRequest(
        role=AgentRole.ARCHITECT,
        workspace="/tmp",
        prompt="hi",
        max_turns=10,
        raw_provider_metadata={"system_prompt": "sp", "commit_changes": True},
    )


# --- mcp definitions (tool surface follows the registry) ---------------------


def test_mcp_definitions_respect_registry():
    simple = mcp_architect_tool_definitions(
        agent_managed_git=False,
        registered_names={"run_eval", "coding_agent"},
    )
    assert [t["name"] for t in simple] == ["run_eval", "coding_agent"]

    ee = mcp_architect_tool_definitions(
        agent_managed_git=True,
        registered_names={"run_eval", "coding_agent", "inspect_traces", "close_iteration"},
    )
    assert [t["name"] for t in ee] == [
        "run_eval",
        "coding_agent",
        "inspect_traces",
        "close_iteration",
    ]


# --- Claude harness seam -----------------------------------------------------


def test_claude_spec_simple_loop_drops_skill_and_extra_tools():
    harness = ClaudeCodeHarness(
        architect_config=_cfg(
            agent_managed_git=False,
            stage_project_skills=False,
        ),
        architect_tool_registry={"run_eval": lambda **k: {}, "coding_agent": lambda **k: {}},
    )
    spec = harness.build_architect_session_spec(_architect_request())
    assert spec.allowed_tools == [
        "Bash", "Read", "Write", "Edit", "Glob", "Grep",
        "mcp__ce__run_eval", "mcp__ce__coding_agent",
    ]
    assert spec.stages_project_skills is False


def test_claude_spec_ee_is_unchanged():
    harness = ClaudeCodeHarness(
        architect_config=_cfg(agent_managed_git=True),
        architect_tool_registry={
            "run_eval": 1, "coding_agent": 1, "inspect_traces": 1, "close_iteration": 1,
        },
    )
    spec = harness.build_architect_session_spec(_architect_request())
    assert spec.allowed_tools == [
        "Bash", "Read", "Write", "Edit", "Glob", "Grep", "Skill",
        "mcp__ce__run_eval", "mcp__ce__coding_agent",
        "mcp__ce__inspect_traces", "mcp__ce__close_iteration",
    ]
    assert spec.stages_project_skills is True


# --- OpenCode harness seam ---------------------------------------------------


def test_opencode_bridge_drops_unregistered_tools():
    simple = OpenCodeArchitectToolBridge(
        {"run_eval": lambda **k: {}, "coding_agent": lambda **k: {}},
        agent_managed_git=False,
    )
    assert [t["name"] for t in simple.list_tools()] == ["run_eval", "coding_agent"]

    ee = OpenCodeArchitectToolBridge(
        {"run_eval": 1, "coding_agent": 1, "inspect_traces": 1},
        agent_managed_git=False,
    )
    assert [t["name"] for t in ee.list_tools()] == [
        "run_eval", "coding_agent", "inspect_traces",
    ]


def test_opencode_session_spec_suppresses_bootstrap_for_simple_loop():
    simple = OpenCodeHarness(architect_config=_cfg(stage_project_skills=False))
    vspec = simple.build_session_spec(_architect_request())
    assert vspec.bootstrap_commands == []
    assert vspec.finalization_commands == []

    default = OpenCodeHarness()  # no architect_config -> EE behavior
    dspec = default.build_session_spec(_architect_request())
    assert dspec.bootstrap_commands == ["/s"]
    assert dspec.finalization_commands == ["/update-specs"]
