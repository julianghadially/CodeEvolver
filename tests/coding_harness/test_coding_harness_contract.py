"""Tests for the shared coding harness contract and fake implementation."""

from __future__ import annotations


def test_fake_harness_executes_each_role_with_autonomous_permissions(tmp_path):
    from src.engine.agent.harnesses import (
        AgentRole,
        FakeCodingHarness,
        HarnessPermissionPolicy,
        HarnessRequest,
        HarnessResult,
    )

    fake = FakeCodingHarness(
        results={
            AgentRole.ARCHITECT: [
                HarnessResult.successful(
                    output="architect complete",
                    session_id="architect-session",
                    raw_provider_metadata={"turns": 3},
                )
            ],
            AgentRole.CODING: [HarnessResult.successful(output="coding complete")],
            AgentRole.INSPECTION: [HarnessResult.successful(output="inspection complete")],
        }
    )
    permissions = HarnessPermissionPolicy.autonomous_policy()

    architect = fake.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="architect prompt",
            permissions=permissions,
        )
    )
    coding = fake.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="coding prompt",
            permissions=permissions,
        )
    )
    inspection = fake.run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspection prompt",
            permissions=permissions,
        )
    )

    assert architect.output == "architect complete"
    assert architect.session_id == "architect-session"
    assert coding.output == "coding complete"
    assert inspection.output == "inspection complete"
    assert [call.role for call in fake.calls] == [
        AgentRole.ARCHITECT,
        AgentRole.CODING,
        AgentRole.INSPECTION,
    ]
    assert all(call.permissions.autonomous for call in fake.calls)


def test_harness_result_normalizes_success_error_and_metadata():
    from src.engine.agent.harnesses import HarnessResult

    success = HarnessResult.successful(
        output=None,
        session_id="session-1",
        raw_provider_metadata={"provider": "fake"},
    )
    error = HarnessResult.failed(
        error=RuntimeError("agent failed"),
        output="partial output",
        raw_provider_metadata=None,
    )

    assert success.success is True
    assert success.error is None
    assert success.output == ""
    assert success.session_id == "session-1"
    assert success.raw_provider_metadata == {"provider": "fake"}

    assert error.success is False
    assert error.error == "agent failed"
    assert error.output == "partial output"
    assert error.session_id is None
    assert error.raw_provider_metadata == {}


def test_fake_harness_supports_architect_resume_semantics(tmp_path):
    from src.engine.agent.harnesses import (
        AgentRole,
        FakeCodingHarness,
        HarnessRequest,
        HarnessResult,
    )

    fake = FakeCodingHarness(
        architect_results=[
            HarnessResult.successful(output="initial", session_id="session-42"),
            HarnessResult.successful(output="watchdog", session_id="session-42"),
        ]
    )

    initial = fake.run_architect(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="initial prompt",
        )
    )
    resumed = fake.resume_architect(
        "session-42",
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="watchdog prompt",
        ),
    )

    assert initial.session_id == "session-42"
    assert resumed.output == "watchdog"
    assert fake.calls[1].resume_session_id == "session-42"
    assert fake.calls[1].prompt == "watchdog prompt"


def test_coding_and_inspection_can_run_without_persistent_sessions(tmp_path):
    from src.engine.agent.harnesses import (
        AgentRole,
        FakeCodingHarness,
        HarnessRequest,
        HarnessResult,
    )

    fake = FakeCodingHarness(
        results={
            AgentRole.CODING: [HarnessResult.successful(output="coded")],
            AgentRole.INSPECTION: [HarnessResult.successful(output="inspected")],
        }
    )

    coding = fake.run_coding(
        HarnessRequest(role=AgentRole.CODING, workspace=tmp_path, prompt="code")
    )
    inspection = fake.run_inspection(
        HarnessRequest(role=AgentRole.INSPECTION, workspace=tmp_path, prompt="inspect")
    )

    assert coding.success is True
    assert coding.session_id is None
    assert inspection.success is True
    assert inspection.session_id is None


def test_agent_tools_can_select_fake_harness_without_subprocesses(tmp_path):
    from src.engine.agent.harnesses import AgentRole, FakeCodingHarness, HarnessResult
    from src.engine.optimizer.v3.agent_tools import coding_agent_tool, inspect_traces_tool
    from src.engine.optimizer.v3.iteration_context import IterationContext

    fake = FakeCodingHarness(
        results={
            AgentRole.CODING: [HarnessResult.successful(output="fake coding output")],
            AgentRole.INSPECTION: [
                HarnessResult.successful(output="fake inspection summary")
            ],
        }
    )
    ctx = IterationContext(iteration_idx=7)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.25

    coding = coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="make a small change",
        workspace=str(tmp_path),
        agent_harness=fake,
    )
    inspection = inspect_traces_tool(
        ctx=ctx,
        query="summarize traces",
        workspace=str(tmp_path),
        agent_harness=fake,
    )

    assert coding == {"ok": True, "error": None, "output": "fake coding output"}
    assert inspection == {
        "ok": True,
        "summary": "fake inspection summary",
        "error": None,
    }
    assert ctx.coding_agent_called is True
    assert [call.role for call in fake.calls] == [AgentRole.CODING, AgentRole.INSPECTION]
