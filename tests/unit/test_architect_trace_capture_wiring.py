"""Copyright © 2026 440 Labs LLC

The Claude architect session spec must turn architect-trace capture (and the
summarized thinking display it depends on) on/off based on the
`enable_meta_reflection` config flag. Offline — no Claude Agent SDK.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.engine.agent.harnesses.base import (
    AgentRole,
    HarnessPermissionPolicy,
    HarnessRequest,
)
from src.engine.agent.harnesses.claude import ClaudeCodeHarness


def _build_spec(*, enable_meta_reflection: bool, stage_project_skills: bool = True):
    config = SimpleNamespace(
        stage_project_skills=stage_project_skills,
        capture_architect_traces=enable_meta_reflection,
        max_turns=10,
        agent_model="claude-opus-4-7",
    )
    harness = ClaudeCodeHarness(
        architect_config=config,
        architect_tool_registry={"run_eval": lambda **k: {}},
    )
    request = HarnessRequest(
        role=AgentRole.ARCHITECT,
        workspace="/workspace",
        prompt="go",
        permissions=HarnessPermissionPolicy.autonomous_policy(),
        max_turns=10,
        raw_provider_metadata={"system_prompt": "sys"},
    )
    return harness.build_architect_session_spec(request)


def test_capture_off_by_default():
    spec = _build_spec(enable_meta_reflection=False)
    assert spec.capture_architect_trace is False
    assert spec.enable_thinking_display is False


def test_capture_on_when_meta_reflection_enabled():
    spec = _build_spec(enable_meta_reflection=True)
    assert spec.capture_architect_trace is True
    assert spec.enable_thinking_display is True


def test_capture_on_for_vanilla_flag_combination():
    """A run that stages no skills must still capture traces — the two flags
    are independent.
    """
    spec = _build_spec(enable_meta_reflection=True, stage_project_skills=False)
    assert spec.capture_architect_trace is True
    assert spec.enable_thinking_display is True
