"""Copyright © 2026 440 Labs LLC

Close-skill watchdog (all arms): a session that called `close_iteration`
without loading the `close_iteration` skill is resumed once and told to
follow the skill. Detection keys on the harness-reported tool calls
(`raw_provider_metadata["tool_calls"]`; `skill` on OpenCode, `Skill` on
Claude) and is fail-open when the harness reported none.
"""

from __future__ import annotations

from src.engine.agent.harnesses import HarnessResult
from src.engine.optimizer.v3.iteration_architect import _close_skill_invoked
from tests.controls._arch_helpers import make_architect


def _capture_launch(monkeypatch, arch):
    calls: list[dict] = []
    monkeypatch.setattr(
        arch,
        "_launch_architect_provider",
        lambda sp, tr, *, resume_session_id=None, initial_prompt_override=None: calls.append(
            {"resume": resume_session_id, "prompt": initial_prompt_override}
        ),
    )
    return calls


def _armed(monkeypatch, tool_calls):
    arch, ctx, _ = make_architect(iteration_idx=4)
    ctx.close_called = True
    arch._last_session_id = "sess-4"
    arch._session_tool_calls = tool_calls
    return arch, ctx, _capture_launch(monkeypatch, arch)


def test_fires_once_when_close_called_without_the_skill(monkeypatch):
    arch, ctx, calls = _armed(
        monkeypatch, [{"name": "run_eval", "input": {}}, {"name": "ce_close_iteration", "input": {}}]
    )
    arch._run_close_skill_watchdog_if_needed("sp", {})
    assert len(calls) == 1 and calls[0]["resume"] == "sess-4"
    assert "`close_iteration` skill" in calls[0]["prompt"]
    assert "last call wins" in calls[0]["prompt"]
    assert ctx.watchdog_fired is True


def test_silent_when_the_skill_was_loaded_opencode_or_claude(monkeypatch):
    for call in (
        {"name": "skill", "input": {"name": "close_iteration"}},
        {"name": "Skill", "input": {"skill": "close_iteration"}},
        {"name": "skill", "input": {}},  # unreadable input: any skill load counts
    ):
        arch, ctx, calls = _armed(monkeypatch, [call, {"name": "ce_close_iteration"}])
        arch._run_close_skill_watchdog_if_needed("sp", {})
        assert calls == [] and ctx.watchdog_fired is False


def test_fires_when_only_another_skill_was_loaded(monkeypatch):
    arch, ctx, calls = _armed(
        monkeypatch, [{"name": "skill", "input": {"name": "exploration"}}]
    )
    arch._run_close_skill_watchdog_if_needed("sp", {})
    assert len(calls) == 1


def test_fail_open_without_tool_call_metadata(monkeypatch):
    arch, ctx, calls = _armed(monkeypatch, None)
    arch._run_close_skill_watchdog_if_needed("sp", {})
    assert calls == []


def test_silent_when_close_not_called_or_skills_not_staged(monkeypatch):
    arch, ctx, calls = _armed(monkeypatch, [{"name": "ce_close_iteration"}])
    ctx.close_called = False
    arch._run_close_skill_watchdog_if_needed("sp", {})
    assert calls == []

    arch, ctx, calls = _armed(monkeypatch, [{"name": "ce_close_iteration"}])
    arch.config.stage_project_skills = False
    arch._run_close_skill_watchdog_if_needed("sp", {})
    assert calls == []


def test_tool_calls_accumulate_across_initial_run_and_resumes(monkeypatch):
    arch, ctx, _ = make_architect(iteration_idx=2)

    class Harness:
        def run_architect(self, request):
            return HarnessResult.successful(
                output="", session_id="s",
                raw_provider_metadata={"tool_calls": [{"name": "run_eval"}]},
            )

        def resume_architect(self, session_id, request):
            return HarnessResult.successful(
                output="", session_id=session_id,
                raw_provider_metadata={"tool_calls": [{"name": "skill", "input": {"name": "close_iteration"}}]},
            )

    monkeypatch.setattr(arch, "_create_coding_harness", lambda **kw: Harness())
    assert arch._session_tool_calls is None
    arch._launch_architect_provider("sp", {})
    arch._launch_architect_provider("sp", {}, resume_session_id="s")
    assert [c["name"] for c in arch._session_tool_calls] == ["run_eval", "skill"]
    assert _close_skill_invoked(arch._session_tool_calls) is True


def test_close_skill_invoked_predicate():
    assert _close_skill_invoked([]) is False
    assert _close_skill_invoked([{"name": "bash", "input": {"command": "skill close_iteration"}}]) is False
    assert _close_skill_invoked([{"name": "SKILL", "input": {"name": "x/close_iteration"}}]) is True
