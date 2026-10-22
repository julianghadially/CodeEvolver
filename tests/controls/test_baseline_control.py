"""Copyright © 2026 440 Labs LLC

Baseline control on the delegation path.

The coding agent does not wait for an initial subsample eval to set
``ctx.initial_subsample_score``: an ordering rule there would apply to
``coding_agent`` but not to a direct Bash edit, penalising delegation for
something the shell does freely, and an agent-directed search may legitimately
need to repair a branch before a valid baseline is possible
(``specs/research_plan.md``).

The baseline requirement — every working iteration measures its program at
least once — is a ``close_iteration`` check, covered in
``test_baseline_requirement.py``. What this file still proves is that the
architect's real tool registry wires ``coding_agent`` through to the harness.
"""

from __future__ import annotations

import pytest

from src.engine.agent.harnesses import AgentRole, HarnessResult
from src.engine.optimizer.v3 import agent_tools as agent_tools_module
from src.engine.optimizer.v3.errors import ControlViolation

from tests.controls._arch_helpers import RecordingHarness, make_architect


def _registry(harness, *, locked: bool = True):
    """Build a registry over a real architect.

    ``locked=True`` mirrors the engine 3 loop, which locks the context at
    iteration construction (there is no select_candidate).
    """
    arch, ctx, _ = make_architect(locked=locked)
    registry = arch.build_tool_registry(agent_harness=harness)
    return registry, ctx


def test_coding_agent_runs_before_any_baseline_eval(monkeypatch):
    """Repairing a branch before it can be baselined is a legitimate move."""
    monkeypatch.setattr(
        agent_tools_module, "git_add_and_commit", lambda ws, msg: False
    )
    harness = RecordingHarness(
        results={AgentRole.CODING: [HarnessResult.successful(output="done")]}
    )
    registry, ctx = _registry(harness)
    assert ctx.initial_subsample_score is None

    result = registry["coding_agent"](
        mode="execute", change_request="fix the import error blocking the eval"
    )

    assert result["ok"] is True
    assert ctx.coding_agent_called is True


def test_coding_agent_allowed_after_baseline_eval(monkeypatch):
    # The registry's coding_agent commits harness changes on success; stub the
    # git commit so the control test stays offline and deterministic.
    monkeypatch.setattr(
        agent_tools_module, "git_add_and_commit", lambda ws, msg: False
    )
    harness = RecordingHarness(
        results={AgentRole.CODING: [HarnessResult.successful(output="done")]}
    )
    registry, ctx = _registry(harness)

    # Baseline eval landed — record the initial subsample score.
    ctx.initial_subsample_score = 0.40

    result = registry["coding_agent"](
        mode="execute", change_request="add re-ranking"
    )

    assert result["ok"] is True
    assert ctx.coding_agent_called is True
    assert [c.role for c in harness.calls] == [AgentRole.CODING]
