"""Regression tests for the Claude coding harness boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def _evaluator_config():
    from src.engine.evaluator.config import EvaluatorConfig

    return EvaluatorConfig(
        program_factory="pkg.program:Program",
        metric_fns=["pkg.metric:metric"],
        metric_names=["score"],
        primary_metric="score",
        testset_path="/data/test.jsonl",
    )


def _architect_config(*, coding_harness_model: str | None = None):
    from src.engine.optimizer.v3.iteration_architect import ArchitectConfig

    return ArchitectConfig(
        run_timestamp="20260604",
        additional_instructions=None,
        program_path="pkg.program:Program",
        evaluator=_evaluator_config(),
        trainset_ids=[1],
        comparison_ids=[2],
        max_turns=17,
        coding_harness="claude",
        coding_harness_model=coding_harness_model,
        agent_managed_git=True,
        system_prompt_builder=lambda **kwargs: "system prompt",
    )


def test_claude_architect_session_spec_preserves_existing_options(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.architect_tools import ARCHITECT_TOOL_SPECS
    from src.engine.agent.harnesses.claude import ClaudeCodeHarness
    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=3)
    tool_registry = {
        "run_eval": lambda **kwargs: {"ok": True},
        "coding_agent": lambda **kwargs: {"ok": True},
        "inspect_traces": lambda **kwargs: {"ok": True},
        "close_iteration": lambda **kwargs: {"ok": True},
    }
    harness = ClaudeCodeHarness(
        architect_context=ctx,
        architect_config=_architect_config(coding_harness_model="openai/gpt-5.3-codex-spark"),
        architect_tool_registry=tool_registry,
    )

    spec = harness.build_architect_session_spec(
        HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=tmp_path,
            prompt="initial prompt",
            max_turns=17,
        )
    )

    assert spec.cwd == str(tmp_path)
    assert spec.initial_prompt == "initial prompt"
    assert spec.setting_sources == ["project"]
    assert spec.permission_mode == "bypassPermissions"
    assert spec.max_turns == 17
    assert spec.allowed_tools == [
        "Bash",
        "Read",
        "Write",
        "Edit",
        "Glob",
        "Grep",
        "Skill",
        "mcp__ce__run_eval",
        "mcp__ce__coding_agent",
        "mcp__ce__inspect_traces",
        "mcp__ce__close_iteration",
    ]
    assert spec.mcp_server_name == "ce"
    assert spec.user_proxy_role == "ARCHITECT"
    assert spec.user_proxy_handles_questions is True
    # No `pre_edit_guard`: it existed only to apply the baseline-before-edit
    # ordering rule, which is gone (the requirement is a close_iteration check
    # now). Claude has no apply_patch tool, so the guard had nothing left to
    # decide on this harness.
    assert spec.hook_names == ["keep_stream_open", "holdout_guard"]
    assert spec.stages_project_skills is True
    assert spec.coding_harness_model is None
    assert ARCHITECT_TOOL_SPECS["run_eval"].description
    coding_schema = ARCHITECT_TOOL_SPECS["coding_agent"].input_schema
    assert set(coding_schema["properties"]) == {"mode", "change_request"}
    assert coding_schema["required"] == ["mode", "change_request"]
    # run_eval's filter is optional — only eval_sample is required. The Claude
    # SDK @tool decorator must receive this full schema (not a bare name->type
    # dict, which it would mark entirely required).
    assert ARCHITECT_TOOL_SPECS["run_eval"].input_schema["required"] == ["eval_sample"]


def test_claude_mode_routes_architect_execution_through_harness(monkeypatch, tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessResult
    from src.engine.optimizer.v3 import iteration_architect as module
    from src.engine.optimizer.v3 import architect_watchdogs as wd_module
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    class CapturingHarness:
        def __init__(self, **kwargs):
            self.calls = []

        def run_architect(self, request):
            self.calls.append(("run", request))
            return HarnessResult.successful(output="ok", session_id="claude-session")

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
    monkeypatch.setattr(module, "git_add_and_commit", lambda *args, **kwargs: False)
    monkeypatch.setattr(module, "git_get_sha", lambda *args, **kwargs: "start-sha")
    monkeypatch.setattr(wd_module, "git_get_sha", lambda *args, **kwargs: "start-sha")
    # A cleanly closed iteration: no watchdog resume expected.
    ctx = IterationContext(iteration_idx=5)
    ctx.close_called = True
    ctx.start_sha = "start-sha"

    architect = IterationArchitect(
        context=ctx,
        state=State(program_candidates=[]),
        workspace=str(tmp_path),
        config=_architect_config(),
    )
    architect.run()

    assert architect._last_session_id == "claude-session"
    assert len(fake.calls) == 1
    mode, request = fake.calls[0]
    assert mode == "run"
    assert request.role == AgentRole.ARCHITECT
    assert request.workspace == str(tmp_path)
    assert request.prompt == architect._build_initial_prompt()


def test_claude_architect_watchdogs_resume_captured_session(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.optimizer.v3 import iteration_architect as module
    from src.engine.optimizer.v3 import architect_watchdogs as wd_module
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    class CapturingHarness:
        def __init__(self, **kwargs):
            self.calls = []

        def run_architect(self, request):
            self.calls.append(("run", request))
            return HarnessResult.successful(output="ok", session_id="session-42")

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
    # The session ended without a close_iteration call — the close-missing
    # watchdog must resume the captured session and demand a close.
    ctx = IterationContext(iteration_idx=6)
    ctx.record_eval("baseline", 0.1)
    ctx.record_eval("head-sha", 0.2)
    # HEAD moved off the iteration-start sha, so the no-change watchdog stays
    # quiet and the close-missing watchdog is the one that resumes.
    ctx.start_sha = "start-sha"

    architect = IterationArchitect(
        context=ctx,
        state=State(program_candidates=[]),
        workspace=str(tmp_path),
        config=_architect_config(),
    )
    architect.run()

    assert ctx.watchdog_fired is True
    assert [call[0] for call in fake.calls] == ["run", "resume"]
    _, session_id, request = fake.calls[1]
    assert session_id == "session-42"
    assert "close_iteration" in request.prompt


def test_claude_mode_routes_coding_and_inspection_tools_through_harness(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import AgentRole, HarnessResult
    from src.engine.optimizer.v3.agent_tools import coding_agent_tool, inspect_traces_tool
    from src.engine.optimizer.v3.iteration_context import IterationContext

    class CapturingHarness:
        def __init__(self):
            self.calls = []

        def run_coding(self, request):
            self.calls.append(request)
            return HarnessResult.successful(output="coded")

        def run_inspection(self, request):
            self.calls.append(request)
            return HarnessResult.successful(output="inspected")

    fake = CapturingHarness()
    ctx = IterationContext(iteration_idx=7)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.25

    coding = coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="make the change",
        workspace=str(tmp_path),
        program_path="pkg.program:Program",
        agent_harness=fake,
    )
    inspection = inspect_traces_tool(
        ctx=ctx,
        query="inspect traces",
        workspace=str(tmp_path),
        agent_harness=fake,
    )

    assert coding == {"ok": True, "error": None, "output": "coded"}
    assert inspection == {"ok": True, "summary": "inspected", "error": None}
    assert [request.role for request in fake.calls] == [
        AgentRole.CODING,
        AgentRole.INSPECTION,
    ]
    assert "coding_harness_model" not in fake.calls[0].raw_provider_metadata
    assert "coding_harness_model" not in fake.calls[1].raw_provider_metadata


def test_claude_harness_preserves_coding_and_inspection_subprocess_contracts(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.claude import ClaudeCodeHarness

    calls = []

    def coding_runner(*args, **kwargs):
        calls.append(("coding", args, kwargs))
        return {"success": True, "error": None, "output": "AGENT_SUCCESS\n"}

    def inspection_runner(*args, **kwargs):
        calls.append(("inspection", args, kwargs))
        return {
            "success": True,
            "error": None,
            "proposed_change": "trace summary",
        }

    harness = ClaudeCodeHarness(
        coding_runner=coding_runner,
        inspection_runner=inspection_runner,
    )

    coding = harness.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="change request",
            max_turns=23,
            raw_provider_metadata={
                "commit_changes": True,
                "program_path": "pkg.program:Program",
                "coding_harness_model": "openai/gpt-5.3-codex-spark",
            },
        )
    )
    inspection = harness.run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect",
            max_turns=11,
            raw_provider_metadata={
                "iteration_num": 9,
                "writable": False,
                "coding_harness_model": "openai/gpt-5.3-codex-spark",
            },
        )
    )

    assert coding.success is True
    assert coding.output == "AGENT_SUCCESS\n"
    assert inspection.success is True
    assert inspection.output == "trace summary"
    assert calls == [
        (
            "coding",
            (str(tmp_path),),
            {
                "change_request": "change request",
                "change_location": None,
                "max_turns": 23,
                "commit_changes": True,
                "program_path": "pkg.program:Program",
                "coding_harness": "claude",
                "coding_harness_model": None,
            },
        ),
        (
            "inspection",
            (str(tmp_path),),
            {
                "prompt": "inspect",
                "max_turns": 11,
                "iteration_num": 9,
                "writable": False,
                "deny_paths": None,
                "coding_harness": "claude",
                "coding_harness_model": None,
                "run_as_agent_user": True,
                # Off for every caller but the clean-slate inspector.
                "allow_bash": False,
            },
        ),
    ]


def test_subprocess_service_claude_mode_delegates_to_claude_harness(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.services import subprocess_service as module

    class CapturingHarness:
        def __init__(self):
            self.calls = []

        def run_coding(self, request):
            self.calls.append(request)
            return HarnessResult(
                success=True,
                error=None,
                output="AGENT_SUCCESS\n",
                raw_provider_metadata={"commit_handled": True},
            )

        def run_inspection(self, request):
            self.calls.append(request)
            return HarnessResult(
                success=True,
                error=None,
                output="memory updated",
            )

    fake = CapturingHarness()
    created = []
    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: created.append(kwargs) or fake,
    )
    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Claude subprocess launch should be behind the harness")
        ),
    )

    coding = module.run_coding_agent(
        str(tmp_path),
        change_request="change",
        max_turns=13,
        commit_changes=True,
        program_path="pkg.program:Program",
        coding_harness="claude",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )
    inspection = module.run_inspection_agent(
        str(tmp_path),
        prompt="reflect",
        max_turns=19,
        iteration_num=4,
        writable=True,
        deny_paths=["val.jsonl"],
        coding_harness="claude",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert coding == {
        "success": True,
        "error": None,
        "output": "AGENT_SUCCESS\n",
        "session_id": None,
        "raw_provider_metadata": {"commit_handled": True},
        "usage": {},
    }
    assert inspection == {
        "success": True,
        "proposed_change": "memory updated",
        "error": None,
        "output": "memory updated",
        "session_id": None,
        "raw_provider_metadata": {},
        "usage": {},
    }
    assert [kwargs["coding_harness"] for kwargs in created] == ["claude", "claude"]
    assert [call.raw_provider_metadata for call in fake.calls] == [
        {
            "change_location": None,
            "commit_changes": True,
            "program_path": "pkg.program:Program",
        },
        {
            "iteration_num": 4,
            "writable": True,
            "deny_paths": ["val.jsonl"],
            "run_as_agent_user": True,
            "allow_bash": False,
        },
    ]
