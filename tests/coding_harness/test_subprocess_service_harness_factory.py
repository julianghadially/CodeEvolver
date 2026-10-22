from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class RecordingHarness:
    calls: list[tuple[str, Any]]

    def run_architect(self, request):
        raise AssertionError("not used in this test")

    def resume_architect(self, session_id, request):
        raise AssertionError("not used in this test")

    def run_coding(self, request):
        from src.engine.agent.harnesses import HarnessResult

        self.calls.append(("coding", request))
        return HarnessResult.successful(
            output="coded",
            raw_provider_metadata={"provider": "fake", "commit_handled": False},
        )

    def run_inspection(self, request):
        from src.engine.agent.harnesses import HarnessResult

        self.calls.append(("inspection", request))
        return HarnessResult.successful(
            output="inspected",
            raw_provider_metadata={"provider": "fake"},
        )


def test_run_coding_agent_uses_factory_and_commits_when_harness_does_not(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import AgentRole
    from src.engine.services import subprocess_service

    harness = RecordingHarness(calls=[])
    created: list[dict[str, Any]] = []

    def fake_factory(**kwargs):
        created.append(kwargs)
        return harness

    monkeypatch.setattr(subprocess_service, "create_coding_harness", fake_factory)
    monkeypatch.setattr(
        subprocess_service, "git_add_and_commit", lambda *args, **kwargs: True
    )

    result = subprocess_service.run_coding_agent(
        str(tmp_path),
        "make a change",
        program_path="pkg.program:Program",
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result["success"] is True
    assert result["output"] == "coded"
    assert created[0]["coding_harness"] == "opencode"
    assert created[0]["coding_harness_model"] == "openai/gpt-5.3-codex-spark"
    assert harness.calls[0][0] == "coding"
    request = harness.calls[0][1]
    assert request.role == AgentRole.CODING
    assert request.raw_provider_metadata["program_path"] == "pkg.program:Program"
    assert request.raw_provider_metadata["commit_changes"] is True


def test_run_inspection_agent_uses_factory(monkeypatch, tmp_path):
    from src.engine.agent.harnesses import AgentRole
    from src.engine.services import subprocess_service

    harness = RecordingHarness(calls=[])
    created: list[dict[str, Any]] = []

    def fake_factory(**kwargs):
        created.append(kwargs)
        return harness

    monkeypatch.setattr(subprocess_service, "create_coding_harness", fake_factory)

    result = subprocess_service.run_inspection_agent(
        str(tmp_path),
        "inspect traces",
        iteration_num=7,
        writable=False,
        deny_paths=["validation.jsonl"],
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result["success"] is True
    assert result["proposed_change"] == "inspected"
    assert created[0]["coding_harness"] == "opencode"
    assert harness.calls[0][0] == "inspection"
    request = harness.calls[0][1]
    assert request.role == AgentRole.INSPECTION
    assert request.raw_provider_metadata["iteration_num"] == 7
    assert request.raw_provider_metadata["writable"] is False
    assert request.raw_provider_metadata["deny_paths"] == ["validation.jsonl"]
