"""Regression tests for workspace-specific architect prompt paths."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def test_architect_prompt_builders_use_runtime_workspace_path():
    from src.engine.optimizer.v3.prompts import (
        build_architect_system_prompt,
        build_evo_architect_system_prompt,
        build_freeform_architect_system_prompt,
    )
    from src.engine.optimizer.v3.iteration_context import IterationContext

    workspace = "/home/operator/.codeevolver/test-repos/FactChecker"
    builders = [
        build_architect_system_prompt,
        build_evo_architect_system_prompt,
        build_freeform_architect_system_prompt,
    ]

    for builder in builders:
        prompt = builder(
            additional_instructions=None,
            program_path="src.factchecker.simple.factchecker_program:FactCheckerProgram",
            workspace_path=workspace,
            context=IterationContext(iteration_idx=1),
        )

        # The runtime workspace path is rendered, never a hard-coded /workspace.
        assert workspace in prompt
        assert "/workspace/codeevolver.md" not in prompt
        assert "{workspace_path}" not in prompt


def test_iteration_architect_passes_workspace_to_system_prompt_builder(
    monkeypatch, tmp_path
):
    from src.engine.agent.harnesses import HarnessResult
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.optimizer.v3 import iteration_architect as module
    from src.engine.optimizer.v3.iteration_architect import (
        ArchitectConfig,
        IterationArchitect,
    )
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    class CapturingHarness:
        def run_architect(self, request):
            return HarnessResult.successful(output="ok", session_id="session")

        def resume_architect(self, session_id, request):
            return HarnessResult.successful(output="ok", session_id=session_id)

    captured: dict[str, Any] = {}

    def prompt_builder(**kwargs) -> str:
        captured.update(kwargs)
        return "system prompt"

    monkeypatch.setattr(
        module,
        "create_coding_harness",
        lambda **kwargs: CapturingHarness(),
        raising=False,
    )
    config = ArchitectConfig(
        run_timestamp="20260605",
        additional_instructions=None,
        program_path="pkg.program:Program",
        evaluator=EvaluatorConfig(
            program_factory="pkg.program:Program",
            metric_fns=["pkg.metric:metric"],
            metric_names=["score"],
            primary_metric="score",
        ),
        trainset_ids=[1],
        comparison_ids=[2],
        max_turns=1,
        system_prompt_builder=prompt_builder,
    )

    architect = IterationArchitect(
        context=IterationContext(iteration_idx=1),
        state=State(program_candidates=[]),
        workspace=str(tmp_path),
        config=config,
    )
    architect.run()

    assert captured["workspace_path"] == str(tmp_path)
