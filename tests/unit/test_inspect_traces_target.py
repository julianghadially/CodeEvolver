"""`inspect_traces(query, trace)` reads the file the architect names.

`trace` is required, the target is validated, and a missing file is an error
rather than a wasted sub-agent session.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.engine.agent.harnesses.architect_tools import (
    ARCHITECT_TOOL_SPECS,
    mcp_architect_tool_definitions,
)
from src.engine.agent.harnesses.base import AgentRole, HarnessRequest, HarnessResult

def _seed_trace_dir(trace_dir):
    """Minimal eval trace directory: the registry checks for `index.jsonl`."""
    trace_dir.mkdir(parents=True, exist_ok=True)
    (trace_dir / "index.jsonl").write_text("{}\n")



class _FakeHarness:
    def __init__(self) -> None:
        self.calls: list[HarnessRequest] = []

    def run_inspection(self, request: HarnessRequest) -> HarnessResult:
        self.calls.append(request)
        return HarnessResult(success=True, output="inspected", error=None)

    def run_coding(self, request):  # pragma: no cover - not exercised
        raise AssertionError


def _registry(tmp_path, monkeypatch, iteration_idx=12):
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.optimizer.v3.iteration_architect import ArchitectConfig, IterationArchitect
    from src.engine.optimizer.v3.iteration_context import IterationContext

    @dataclass
    class State:
        program_candidates: list[dict[str, Any]]

    traces = tmp_path / "traces"
    traces.mkdir()
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(traces))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    ctx = IterationContext(iteration_idx=iteration_idx)
    ctx.parent_locked = True
    ctx.initial_subsample_score = 0.4
    config = ArchitectConfig(
        run_timestamp="20260830",
        additional_instructions=None,
        program_path="pkg.program:Program",
        evaluator=EvaluatorConfig(
            program_factory="pkg.program:Program",
            metric_fns=["pkg.metric:metric"],
            metric_names=["score"],
            primary_metric="score",
            testset_path="/data/test.jsonl",
        ),
        trainset_ids=[1],
        comparison_ids=[2],
        max_turns=5,
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
        agent_managed_git=True,
        system_prompt_builder=lambda **kwargs: "system prompt",
    )
    fake = _FakeHarness()
    architect = IterationArchitect(
        context=ctx, state=State(program_candidates=[]),
        workspace=str(tmp_path), config=config,
    )
    return architect.build_tool_registry(agent_harness=fake), fake, traces


def test_parent_targets_the_baseline_file(tmp_path, monkeypatch):
    registry, fake, traces = _registry(tmp_path, monkeypatch)
    _seed_trace_dir(traces / "iteration_12_parent")
    out = registry["inspect_traces"]("why did row 3 fail?", "parent")
    assert out["ok"], out
    prompt = fake.calls[-1].prompt
    assert fake.calls[-1].role is AgentRole.INSPECTION
    assert "iteration_12_parent/" in prompt
    assert "iteration_12_child" not in prompt
    assert "BASELINE" in prompt and "why did row 3 fail?" in prompt


def test_child_before_any_post_change_eval_is_an_error_not_a_session(tmp_path, monkeypatch):
    registry, fake, traces = _registry(tmp_path, monkeypatch)
    _seed_trace_dir(traces / "iteration_12_parent")
    out = registry["inspect_traces"]("q", "child")
    assert out["ok"] is False
    assert "iteration_12_child/ does not exist yet" in out["error"]
    assert "trace='parent'" in out["error"]
    assert fake.calls == []


def test_child_reads_the_child_file_once_it_exists(tmp_path, monkeypatch):
    registry, fake, traces = _registry(tmp_path, monkeypatch)
    _seed_trace_dir(traces / "iteration_12_child")
    out = registry["inspect_traces"]("q", "child")
    assert out["ok"]
    assert "iteration_12_child/" in fake.calls[-1].prompt
    assert "AFTER the" in fake.calls[-1].prompt


def test_invalid_target_is_rejected(tmp_path, monkeypatch):
    registry, fake, _ = _registry(tmp_path, monkeypatch)
    out = registry["inspect_traces"]("q", "holdout")
    assert out["ok"] is False and "parent" in out["error"] and "child" in out["error"]
    assert fake.calls == []


def test_comparison_before_any_comparison_eval_is_an_error_not_a_session(tmp_path, monkeypatch):
    registry, fake, traces = _registry(tmp_path, monkeypatch)
    _seed_trace_dir(traces / "iteration_12_parent")
    out = registry["inspect_traces"]("q", "comparison")
    assert out["ok"] is False
    assert "iteration_12_comparison/ does not exist yet" in out["error"]
    assert "comparison" in out["error"]
    assert fake.calls == []


def test_comparison_reads_the_comparison_dir_once_it_exists(tmp_path, monkeypatch):
    """Comparison traces are inspectable: an architect may run a comparison
    eval specifically to get traces to inspect."""
    registry, fake, traces = _registry(tmp_path, monkeypatch)
    _seed_trace_dir(traces / "iteration_12_comparison")
    out = registry["inspect_traces"]("q", "comparison")
    assert out["ok"]
    assert "iteration_12_comparison/" in fake.calls[-1].prompt
    assert "comparison-set eval" in fake.calls[-1].prompt


def test_schema_requires_trace_on_both_harness_exports():
    spec = ARCHITECT_TOOL_SPECS["inspect_traces"]
    assert spec.input_schema["required"] == ["query", "trace"]
    assert spec.input_schema["properties"]["trace"]["enum"] == ["parent", "child", "comparison"]
    assert "BASELINE" in spec.description
    mcp = {d["name"]: d for d in mcp_architect_tool_definitions()}["inspect_traces"]
    assert mcp["inputSchema"]["required"] == ["query", "trace"]
