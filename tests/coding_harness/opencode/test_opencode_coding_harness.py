"""Tests for the OpenCode coding-agent tracer bullet."""

from __future__ import annotations

import json
import os
import subprocess


class RecordingRunner:
    def __init__(self, result: subprocess.CompletedProcess[str]) -> None:
        self.result = result
        self.calls: list[dict] = []

    def __call__(self, cmd, **kwargs):
        self.calls.append({"cmd": cmd, "kwargs": kwargs})
        return self.result


def test_opencode_coding_command_uses_workspace_model_permissions_and_bootstrap(
    tmp_path,
):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {
                    "type": "session.updated",
                    "sessionID": "opencode-session-1",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "type": "message",
                    "message": {
                        "role": "assistant",
                        "parts": [{"type": "text", "text": "done"}],
                    },
                }
            )
            + "\n",
            stderr="",
        )
    )
    harness = OpenCodeHarness(model="openai/gpt-5.3-codex-spark", runner=runner)

    result = harness.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            max_turns=12,
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert result.success is True
    assert result.error is None
    assert result.output == "done"
    assert result.session_id == "opencode-session-1"
    assert result.raw_provider_metadata["provider"] == "opencode"
    assert result.raw_provider_metadata["role"] == "coding"
    assert result.raw_provider_metadata["returncode"] == 0

    assert len(runner.calls) == 1
    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["opencode", "run"]
    assert "--dir" in cmd
    assert cmd[cmd.index("--dir") + 1] == str(tmp_path)
    assert "--model" in cmd
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-5.3-codex-spark"
    assert "--format" in cmd
    assert cmd[cmd.index("--format") + 1] == "json"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[-1].startswith("/s\n\n")
    assert "make a change" in cmd[-1]
    assert call["kwargs"]["cwd"] == str(tmp_path)
    assert call["kwargs"]["text"] is True
    assert call["kwargs"]["capture_output"] is True
    assert call["kwargs"]["timeout"] == 12 * 120
    assert call["kwargs"]["env"]["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] == "true"
    assert call["kwargs"]["env"]["OPENCODE_CONFIG_DIR"]
    config = json.loads(call["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == "openai/gpt-4.1-mini"
    assert not (tmp_path / ".opencode").exists()


def test_opencode_coding_surfaces_token_usage_and_model_from_stdout(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="\n".join(
                [
                    json.dumps(
                        {"info": {"role": "assistant", "id": "m1", "modelID": "gpt-5.4-mini"}}
                    ),
                    json.dumps(
                        {
                            "type": "step-finish",
                            "tokens": {
                                "input": 200,
                                "output": 30,
                                "reasoning": 0,
                                "cache": {"write": 0, "read": 50},
                            },
                        }
                    ),
                ]
            )
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(model="openai/gpt-5.3-codex-spark", runner=runner).run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    usage = result.raw_provider_metadata["usage"]
    assert usage["input_tokens"] == 200
    assert usage["output_tokens"] == 30
    assert usage["cache_read_input_tokens"] == 50
    assert usage["total"] == 280
    # Parsed bare modelID is preferred over the requested coding_harness_model.
    assert usage["model"] == "gpt-5.4-mini"
    assert result.raw_provider_metadata["model"] == "gpt-5.4-mini"


def test_opencode_coding_falls_back_to_spec_model_when_stdout_has_no_usage(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {"message": {"parts": [{"type": "text", "text": "done"}]}}
            )
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(model="openai/gpt-5.3-codex-spark", runner=runner).run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    # No usage in stdout — but the model bucket is the requested model, never
    # "unknown". Token usage stays absent (records as zero downstream).
    assert result.raw_provider_metadata["model"] == "openai/gpt-5.3-codex-spark"
    assert "usage" not in result.raw_provider_metadata


def test_opencode_coding_fences_holdout_deny_paths(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {"message": {"parts": [{"type": "text", "text": "done"}]}}
            )
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(runner=runner).run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={
                "commit_changes": False,
                "deny_paths": ["_valset.jsonl", "holdout.jsonl"],
            },
        )
    )

    assert result.success is True
    assert result.raw_provider_metadata["deny_paths"] == [
        "_valset.jsonl",
        "holdout.jsonl",
    ]

    call = runner.calls[0]
    permission = json.loads(call["kwargs"]["env"]["OPENCODE_PERMISSION"])
    config = json.loads(call["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    # The permission config the harness records matches what OpenCode receives.
    assert config["permission"] == permission
    assert result.raw_provider_metadata["permission"] == permission
    # Coding agent keeps its edit/write/bash capabilities ...
    assert permission["read"]["*"] == "allow"
    assert permission["edit"]["*"] == "allow"
    assert permission["write"]["*"] == "allow"
    assert permission["bash"] == "allow"
    # ... but the holdout files are denied on every read/write surface.
    for pattern in ("*_valset.jsonl*", "*holdout.jsonl*"):
        assert permission["read"][pattern] == "deny"
        assert permission["grep"][pattern] == "deny"
        assert permission["glob"][pattern] == "deny"
        assert permission["edit"][pattern] == "deny"
        assert permission["write"][pattern] == "deny"
        assert permission["external_directory"][pattern] == "deny"


def test_opencode_coding_command_omits_model_when_not_configured(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="plain output",
            stderr="",
        )
    )
    harness = OpenCodeHarness(runner=runner)

    harness.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert "--model" not in runner.calls[0]["cmd"]


def test_opencode_coding_error_is_normalized(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=2,
            stdout="partial output",
            stderr="provider failed",
        )
    )
    harness = OpenCodeHarness(model="openai/gpt-5.3-codex-spark", runner=runner)

    result = harness.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert result.success is False
    assert result.error == "OpenCode exited with code 2: provider failed"
    assert result.output == "partial output"
    assert result.raw_provider_metadata["stderr"] == "provider failed"


def test_opencode_coding_verifies_binary_without_installing(monkeypatch, tmp_path):
    import src.engine.agent.harnesses.opencode.harness as opencode_module
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    monkeypatch.setattr(opencode_module.shutil, "which", lambda name: None)
    monkeypatch.setattr(
        opencode_module.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("missing opencode must fail before subprocess execution")
        ),
    )

    harness = OpenCodeHarness(model="openai/gpt-5.3-codex-spark")

    try:
        harness.run_coding(
            HarnessRequest(
                role=AgentRole.CODING,
                workspace=tmp_path,
                prompt="make a change",
                raw_provider_metadata={"commit_changes": False},
            )
        )
    except RuntimeError as exc:
        assert "opencode binary was not found" in str(exc)
    else:
        raise AssertionError("expected missing opencode binary to fail clearly")


def test_opencode_coding_does_not_import_claude_sdk(monkeypatch, tmp_path):
    import builtins
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise AssertionError("OpenCode coding mode must not import claude_agent_sdk")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="done",
            stderr="",
        )
    )

    result = OpenCodeHarness(runner=runner).run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert result.success is True
    assert result.output == "done"


def test_opencode_coding_verifies_opencode_not_claude_cli(monkeypatch, tmp_path):
    import src.engine.agent.harnesses.opencode.harness as opencode_module
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    checked_binaries = []
    monkeypatch.setattr(
        opencode_module.shutil,
        "which",
        lambda name: checked_binaries.append(name) or "/usr/bin/opencode",
    )
    monkeypatch.setattr(
        opencode_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="done",
            stderr="",
        ),
    )

    result = OpenCodeHarness().run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert result.success is True
    assert checked_binaries == ["opencode"]


def test_opencode_service_routes_coding_through_harness_without_claude(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.services import subprocess_service as module

    class CapturingOpenCodeHarness:
        def run_coding(self, request):
            calls.append(request)
            return HarnessResult.successful(
                output="opencode output",
                session_id="session-123",
                raw_provider_metadata={"provider": "opencode"},
            )

    calls = []
    created = []
    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: created.append(kwargs) or CapturingOpenCodeHarness(),
    )
    monkeypatch.setattr(
        module,
        "_run_claude_coding_agent_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCode mode must not launch Claude")
        ),
    )
    monkeypatch.setattr(module, "git_add_and_commit", lambda *args, **kwargs: True)

    result = module.run_coding_agent(
        str(tmp_path),
        change_request="make a change",
        max_turns=9,
        commit_changes=True,
        program_path="pkg.program:Program",
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result == {
        "success": True,
        "error": None,
        "output": "opencode output",
        "session_id": "session-123",
        "raw_provider_metadata": {"provider": "opencode"},
        "usage": {},
    }
    assert len(calls) == 1
    request = calls[0]
    assert created[0]["coding_harness"] == "opencode"
    assert created[0]["coding_harness_model"] == "openai/gpt-5.3-codex-spark"
    assert request.workspace == str(tmp_path)
    assert request.prompt == "make a change"
    assert request.permissions.autonomous is True
    assert request.max_turns == 9
    assert request.raw_provider_metadata == {
        "change_location": None,
        "commit_changes": True,
        "program_path": "pkg.program:Program",
    }


def test_coding_agent_tool_routes_opencode_request_to_selected_harness(tmp_path):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.optimizer.v3.agent_tools import coding_agent_tool
    from src.engine.optimizer.v3.iteration_context import IterationContext

    class CapturingHarness:
        def __init__(self):
            self.calls = []

        def run_coding(self, request):
            self.calls.append(request)
            return HarnessResult.successful(output="opencode output")

    harness = CapturingHarness()
    ctx = IterationContext(iteration_idx=11)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.5

    result = coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="make a change",
        workspace=str(tmp_path),
        program_path="pkg.program:Program",
        max_turns=7,
        agent_harness=harness,
    )

    assert result == {"ok": True, "error": None, "output": "opencode output"}
    assert len(harness.calls) == 1
    request = harness.calls[0]
    assert str(request.workspace) == str(tmp_path)
    assert request.prompt == "make a change"
    assert request.max_turns == 7
    assert request.raw_provider_metadata["commit_changes"] is True
    assert request.raw_provider_metadata["program_path"] == "pkg.program:Program"
    assert "coding_harness" not in request.raw_provider_metadata
    assert "coding_harness_model" not in request.raw_provider_metadata


def test_opencode_service_preserves_commit_result_behavior(monkeypatch, tmp_path):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.services import subprocess_service as module

    class SuccessfulOpenCodeHarness:
        def run_coding(self, request):
            return HarnessResult.successful(output="opencode output")

    commit_calls = []
    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: SuccessfulOpenCodeHarness(),
    )
    monkeypatch.setattr(
        module,
        "git_add_and_commit",
        lambda *args, **kwargs: commit_calls.append((args, kwargs)) or False,
    )

    result = module.run_coding_agent(
        str(tmp_path),
        change_request="make a change",
        commit_changes=True,
        coding_harness="opencode",
    )

    assert result == {
        "success": True,
        "error": "Agent completed but no file changes were made",
        "output": "opencode output",
        "session_id": None,
        "raw_provider_metadata": {},
        "usage": {},
    }
    assert commit_calls == [((str(tmp_path), "Code mutation"), {})]


def test_opencode_service_skips_commit_when_disabled(monkeypatch, tmp_path):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.services import subprocess_service as module

    class SuccessfulOpenCodeHarness:
        def run_coding(self, request):
            return HarnessResult.successful(output="opencode output")

    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: SuccessfulOpenCodeHarness(),
    )
    monkeypatch.setattr(
        module,
        "git_add_and_commit",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("commit_changes=False must not commit")
        ),
    )

    result = module.run_coding_agent(
        str(tmp_path),
        change_request="make a change",
        commit_changes=False,
        coding_harness="opencode",
    )

    assert result == {
        "success": True,
        "error": None,
        "output": "opencode output",
        "session_id": None,
        "raw_provider_metadata": {},
        "usage": {},
    }


def test_opencode_readonly_inspection_command_denies_write_tools_and_uses_model(
    tmp_path,
):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {
                    "sessionID": "inspect-session-1",
                    "message": {"parts": [{"type": "text", "text": "summary text"}]},
                }
            )
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(model="openai/gpt-5.3-codex-spark", runner=runner).run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect traces",
            max_turns=4,
            raw_provider_metadata={"writable": False, "deny_paths": ["_valset.jsonl"]},
        )
    )

    assert result.success is True
    assert result.output == "summary text"
    assert result.session_id == "inspect-session-1"
    assert result.raw_provider_metadata["provider"] == "opencode"
    assert result.raw_provider_metadata["role"] == "inspection"

    call = runner.calls[0]
    cmd = call["cmd"]
    assert cmd[:2] == ["opencode", "run"]
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-5.3-codex-spark"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[-1].startswith("/s\n\n")
    assert "inspect traces" in cmd[-1]

    permission = json.loads(call["kwargs"]["env"]["OPENCODE_PERMISSION"])
    config = json.loads(call["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == "openai/gpt-4.1-mini"
    assert call["kwargs"]["env"]["OPENCODE_CONFIG_DIR"]
    assert config["permission"] == permission
    assert permission["read"]["*"] == "allow"
    assert permission["grep"]["*_valset.jsonl*"] == "deny"
    assert permission["glob"]["*_valset.jsonl*"] == "deny"
    assert permission["edit"] == "deny"
    assert permission["bash"] == "deny"
    assert permission["question"] == "deny"


def test_opencode_writable_inspection_allows_edits_with_holdout_fenced(
    monkeypatch,
    tmp_path,
):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    memory_dir = tmp_path / "memory"
    traces_dir = tmp_path / "traces"
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(memory_dir))
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(traces_dir))

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps(
                {
                    "message": {
                        "parts": [
                            {
                                "type": "tool",
                                "tool": "edit",
                                "input": {"filePath": str(memory_dir / "memory.md")},
                            }
                        ]
                    }
                }
            )
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(runner=runner).run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="update memory",
            raw_provider_metadata={"writable": True, "deny_paths": ["holdout.jsonl"]},
        )
    )

    assert result.success is True
    assert result.output == "Memory updated"
    permission = json.loads(runner.calls[0]["kwargs"]["env"]["OPENCODE_PERMISSION"])
    config = json.loads(runner.calls[0]["kwargs"]["env"]["OPENCODE_CONFIG_CONTENT"])
    assert config["small_model"] == "openai/gpt-4.1-mini"
    assert config["permission"] == permission
    # Allow-by-default with holdout denies (matches architect/coding agents);
    # apply_patch is governed by `edit`, so a deny-by-default edit whitelist
    # silently blocked all writes. Holdout paths stay fenced; bash stays denied.
    assert permission["edit"]["*"] == "allow"
    assert permission["edit"]["*holdout.jsonl*"] == "deny"
    assert permission["write"]["*"] == "allow"
    assert permission["read"][str(traces_dir / "**")] == "allow"
    assert permission["read"]["*holdout.jsonl*"] == "deny"
    assert permission["bash"] == "deny"


def test_opencode_writable_inspection_fails_when_no_memory_write_is_observed(
    tmp_path,
):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout=json.dumps({"message": {"parts": [{"type": "text", "text": "done"}]}})
            + "\n",
            stderr="",
        )
    )

    result = OpenCodeHarness(runner=runner).run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="update memory",
            raw_provider_metadata={"writable": True},
        )
    )

    assert result.success is False
    assert result.error == "Writable inspection produced no writes"
    assert result.output == "No change proposed"


def test_opencode_inspection_error_is_normalized(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=3,
            stdout="partial summary",
            stderr="provider failed",
        )
    )

    result = OpenCodeHarness(runner=runner).run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect traces",
            raw_provider_metadata={"writable": False},
        )
    )

    assert result.success is False
    assert result.error == "OpenCode exited with code 3: provider failed"
    assert result.output == "partial summary"
    assert result.raw_provider_metadata["stderr"] == "provider failed"


def test_opencode_inspection_verifies_opencode_not_claude_cli(monkeypatch, tmp_path):
    import src.engine.agent.harnesses.opencode.harness as opencode_module
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    checked_binaries = []
    monkeypatch.setattr(
        opencode_module.shutil,
        "which",
        lambda name: checked_binaries.append(name) or "/usr/bin/opencode",
    )
    monkeypatch.setattr(
        opencode_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=0,
            stdout="summary text",
            stderr="",
        ),
    )

    result = OpenCodeHarness().run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect traces",
            raw_provider_metadata={"writable": False},
        )
    )

    assert result.success is True
    assert checked_binaries == ["opencode"]


def test_opencode_inspection_does_not_import_claude_sdk(monkeypatch, tmp_path):
    import builtins
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name == "claude_agent_sdk":
            raise AssertionError(
                "OpenCode inspection mode must not import claude_agent_sdk"
            )
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    runner = RecordingRunner(
        subprocess.CompletedProcess(
            args=["opencode"],
            returncode=0,
            stdout="summary text",
            stderr="",
        )
    )

    result = OpenCodeHarness(runner=runner).run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect traces",
            raw_provider_metadata={"writable": False},
        )
    )

    assert result.success is True
    assert result.output == "summary text"


def test_opencode_service_routes_inspection_through_harness_without_claude(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.services import subprocess_service as module

    class CapturingOpenCodeHarness:
        def run_inspection(self, request):
            calls.append(request)
            return HarnessResult.successful(
                output="inspection summary",
                session_id="inspect-session-123",
                raw_provider_metadata={"provider": "opencode"},
            )

    calls = []
    created = []
    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: created.append(kwargs) or CapturingOpenCodeHarness(),
    )
    monkeypatch.setattr(
        module,
        "_run_claude_inspection_agent_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCode mode must not launch Claude inspection")
        ),
    )

    result = module.run_inspection_agent(
        str(tmp_path),
        prompt="inspect traces",
        max_turns=8,
        iteration_num=3,
        writable=False,
        deny_paths=["_valset.jsonl"],
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result == {
        "success": True,
        "proposed_change": "inspection summary",
        "error": None,
        "output": "inspection summary",
        "session_id": "inspect-session-123",
        "raw_provider_metadata": {"provider": "opencode"},
        "usage": {},
    }
    assert len(calls) == 1
    request = calls[0]
    assert created[0]["coding_harness"] == "opencode"
    assert created[0]["coding_harness_model"] == "openai/gpt-5.3-codex-spark"
    assert request.workspace == str(tmp_path)
    assert request.prompt == "inspect traces"
    assert request.permissions.autonomous is True
    assert request.max_turns == 8
    assert request.raw_provider_metadata == {
        "iteration_num": 3,
        "writable": False,
        "deny_paths": ["_valset.jsonl"],
        "run_as_agent_user": True,
        # Off for every caller but the clean-slate inspector.
        "allow_bash": False,
    }


def _inspection_result():
    return subprocess.CompletedProcess(
        args=["opencode"], returncode=0, stderr="",
        stdout=json.dumps({
            "sessionID": "inspect-iso",
            "message": {"parts": [{"type": "text", "text": "ok"}]},
        }) + "\n",
    )


def test_opencode_inspection_sessions_get_isolated_data_dirs(tmp_path, monkeypatch):
    """Every inspection session points OpenCode's data/state/cache at its own
    fresh dir under ~/.codeevolver/opencode-sessions: a shared store under
    $HOME would leave the root-run integrity auditor facing ce-agent-owned
    0644 SQLite files without CAP_DAC_OVERRIDE."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    monkeypatch.setenv("HOME", str(tmp_path))
    runner = RecordingRunner(_inspection_result())
    harness = OpenCodeHarness(runner=runner)
    request = HarnessRequest(
        role=AgentRole.INSPECTION, workspace=tmp_path, prompt="inspect",
        max_turns=2, raw_provider_metadata={"writable": False},
    )
    first = harness.run_inspection(request)
    second = harness.run_inspection(request)

    envs = [c["kwargs"]["env"] for c in runner.calls]
    dirs = [e["XDG_DATA_HOME"] for e in envs]
    assert dirs[0] != dirs[1]
    for env, result in zip(envs, (first, second)):
        d = env["XDG_DATA_HOME"]
        assert env["XDG_STATE_HOME"] == d
        # The cache (models.json, ripgrep download) stays shared: a fresh one
        # per session adds a network fetch to bootstrap (flaky locally,
        # egress-fenced on the fleet).
        assert "XDG_CACHE_HOME" not in env
        assert d.startswith(str(tmp_path / ".codeevolver" / "opencode-sessions"))
        assert d != str(tmp_path / ".local" / "share")
        assert os.path.isdir(d)
        assert os.path.basename(d).startswith("inspection-")
        assert result.raw_provider_metadata["data_dir"] == d


def test_opencode_coding_sessions_keep_the_global_data_dir(tmp_path, monkeypatch):
    """Architect/coding sessions are resumed and exported from OpenCode's
    global store, so they must NOT be isolated."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    monkeypatch.setenv("HOME", str(tmp_path))
    runner = RecordingRunner(subprocess.CompletedProcess(
        args=["opencode"], returncode=0, stderr="",
        stdout=json.dumps({"sessionID": "s", "message": {"parts": []}}) + "\n",
    ))
    OpenCodeHarness(runner=runner).run_coding(
        HarnessRequest(role=AgentRole.CODING, workspace=tmp_path, prompt="edit", max_turns=2)
    )
    env = runner.calls[0]["kwargs"]["env"]
    assert "XDG_DATA_HOME" not in env and "XDG_STATE_HOME" not in env


def test_opencode_inspection_error_event_is_a_failure_not_a_report(tmp_path, monkeypatch):
    """A session that dies in bootstrap emits only `{"type":"error",…}` on
    stdout. That must come back as a FAILED result with empty output and the
    OpenCode error in `error` -- not as a "report" made of the raw event."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    monkeypatch.setenv("HOME", str(tmp_path))
    event = {"type": "error", "timestamp": 1, "sessionID": "ses_x",
             "error": {"name": "UnknownError",
                       "data": {"message": "Unexpected server error.", "ref": "err_1"}}}
    for rc in (1, 0):
        runner = RecordingRunner(subprocess.CompletedProcess(
            args=["opencode"], returncode=rc, stderr="", stdout=json.dumps(event) + "\n",
        ))
        result = OpenCodeHarness(runner=runner).run_inspection(HarnessRequest(
            role=AgentRole.INSPECTION, workspace=tmp_path, prompt="p", max_turns=2,
            raw_provider_metadata={"writable": False},
        ))
        assert result.success is False, rc
        assert result.output == ""
        assert "UnknownError: Unexpected server error. (ref err_1)" in result.error
        assert "produced no text" in result.error
        assert result.raw_provider_metadata["error_events"] == [
            "UnknownError: Unexpected server error. (ref err_1)"
        ]


def test_opencode_inspection_nonjson_stdout_is_still_output(tmp_path, monkeypatch):
    """Plain (non-event) stdout keeps flowing through as the output."""
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    monkeypatch.setenv("HOME", str(tmp_path))
    runner = RecordingRunner(subprocess.CompletedProcess(
        args=["opencode"], returncode=0, stderr="", stdout="just text\n",
    ))
    result = OpenCodeHarness(runner=runner).run_inspection(HarnessRequest(
        role=AgentRole.INSPECTION, workspace=tmp_path, prompt="p", max_turns=2,
        raw_provider_metadata={"writable": False},
    ))
    assert result.success is True and result.output.strip() == "just text"
