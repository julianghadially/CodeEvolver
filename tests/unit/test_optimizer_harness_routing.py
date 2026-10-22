"""Issue 8 optimizer-level harness routing tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

def _seed_trace_dir(trace_dir):
    """Minimal eval trace directory: the registry checks for `index.jsonl`."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "index.jsonl").write_text("{}\n")



def _evaluator_config():
    from src.engine.evaluator.config import EvaluatorConfig

    return EvaluatorConfig(
        program_factory="pkg.program:Program",
        metric_fns=["pkg.metric:metric"],
        metric_names=["score"],
        primary_metric="score",
        testset_path="/data/test.jsonl",
    )


def _architect_config():
    from src.engine.optimizer.v3.iteration_architect import ArchitectConfig

    return ArchitectConfig(
        run_timestamp="20260604",
        additional_instructions=None,
        program_path="pkg.program:Program",
        evaluator=_evaluator_config(),
        trainset_ids=[1],
        comparison_ids=[2],
        max_turns=17,
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
        agent_managed_git=True,
        system_prompt_builder=lambda **kwargs: "system prompt",
    )


def test_architect_tool_registry_routes_nested_roles_through_selected_harness(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import AgentRole, FakeCodingHarness, HarnessResult
    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    fake = FakeCodingHarness(
        results={
            AgentRole.CODING: [HarnessResult.successful(output="coded")],
            AgentRole.INSPECTION: [HarnessResult.successful(output="inspected")],
        }
    )
    from src.engine.optimizer.v3 import agent_tools as agent_tools_module

    monkeypatch.setattr(
        agent_tools_module,
        "git_add_and_commit",
        lambda *args, **kwargs: True,
    )
    ctx = IterationContext(iteration_idx=12)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.4
    architect = IterationArchitect(
        context=ctx,
        state=State(program_candidates=[]),
        workspace=str(tmp_path),
        config=_architect_config(),
    )

    registry = architect.build_tool_registry(agent_harness=fake)
    coding = registry["coding_agent"]("execute", "make a change")
    traces = tmp_path / "traces"
    traces.mkdir()
    _seed_trace_dir(traces / "iteration_12_parent")
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(traces))
    inspection = registry["inspect_traces"]("inspect traces", "parent")

    assert coding == {"ok": True, "error": None, "output": "coded"}
    assert inspection == {"ok": True, "summary": "inspected", "error": None}
    assert [call.role for call in fake.calls] == [
        AgentRole.CODING,
        AgentRole.INSPECTION,
    ]


def test_architect_tool_registry_requires_selected_harness(tmp_path):
    import pytest

    from src.engine.optimizer.v3.iteration_architect import IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    architect = IterationArchitect(
        context=IterationContext(iteration_idx=13),
        state=State(program_candidates=[]),
        workspace=str(tmp_path),
        config=_architect_config(),
    )

    with pytest.raises(TypeError):
        architect.build_tool_registry()  # type: ignore[call-arg]


def test_main_optimizer_and_service_files_do_not_import_provider_harness_classes():
    repo_root = Path(__file__).resolve().parents[2]
    guarded_files = [
        repo_root / "src/engine/optimizer/v3/iteration_architect.py",
        repo_root / "src/engine/optimizer/v3/architect_watchdogs.py",
        repo_root / "src/engine/optimizer/v3/architect_tool_registry.py",
        repo_root / "src/engine/optimizer/v3/agent_tools.py",
        repo_root / "src/engine/optimizer/v3/optimizer_loop.py",
        repo_root / "src/engine/optimizer/v3/close_effects.py",
        repo_root / "src/engine/optimizer/v3/analysis_agent.py",
        repo_root / "src/engine/optimizer/v3/run_reporting.py",
        repo_root / "src/engine/optimizer/v3/finalization.py",
        repo_root / "src/engine/optimizer/v3/variance_estimate.py",
        repo_root / "src/engine/services/subprocess_service.py",
    ]

    forbidden = (
        "from src.engine.agent.harnesses import ClaudeCodeHarness",
        "from src.engine.agent.harnesses import OpenCodeHarness",
        "from src.engine.agent.harnesses.claude import ClaudeCodeHarness",
        "from src.engine.agent.harnesses.opencode import OpenCodeHarness",
        "ClaudeCodeHarness(",
        "OpenCodeHarness(",
    )
    for path in guarded_files:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{path.name} leaks provider detail {token}"
