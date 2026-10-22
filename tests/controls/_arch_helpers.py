"""Copyright © 2026 440 Labs LLC

Shared fixtures for the hook/control tests.

These tests drive the **real** ``IterationArchitect`` (its
``build_tool_registry`` and watchdog methods) with a ``FakeCodingHarness`` so
they prove the controls are wired into the implemented agent — not just that
the standalone control functions raise (those are covered by
``test_agent_tools.py``). No real coding-harness session, no subprocess, no
network: everything resolves in-process.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from src.engine.agent.harnesses import FakeCodingHarness, HarnessRequest
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.optimizer.v3.iteration_architect import (
    ArchitectConfig,
    IterationArchitect,
)
from src.engine.optimizer.v3.iteration_context import IterationContext


@dataclass
class FakeState:
    """Minimal CEState stand-in: branch registry + eval counter."""

    primary_branch: str = "codeevolver-20260601120000-main"
    branches: dict[str, Any] = field(
        default_factory=lambda: {
            "codeevolver-20260601120000-main": {"last_sha": "seed", "is_primary": True}
        }
    )
    iterations: list[dict[str, Any]] = field(default_factory=list)
    total_num_evals: int = 0
    final_comparison_evals: int = 0
    comparison_credits_balance: int = 1
    comparison_evals: list[dict[str, Any]] = field(default_factory=list)

    def increment_evals(self, count: int) -> None:
        self.total_num_evals += count

    def note_final_comparison_eval(self) -> None:
        self.final_comparison_evals += 1

    def grant_comparison_credits(self, n: int) -> None:
        self.comparison_credits_balance += int(n)

    def debit_comparison_credit(self) -> None:
        self.comparison_credits_balance -= 1

    def record_comparison_eval(self, **kwargs: Any) -> None:
        self.comparison_evals.append(kwargs)

    def known_branches(self) -> set[str]:
        return set(self.branches)


class RecordingHarness(FakeCodingHarness):
    """FakeCodingHarness that also retains the full HarnessRequest objects.

    ``FakeHarnessCall`` drops ``raw_provider_metadata`` (where ``deny_paths``
    lives), so the structural fence tests need the raw requests.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.requests: list[HarnessRequest] = []

    def run_coding(self, request: HarnessRequest) -> Any:
        self.requests.append(request)
        return super().run_coding(request)

    def run_inspection(self, request: HarnessRequest) -> Any:
        self.requests.append(request)
        return super().run_inspection(request)

    def run_architect(self, request: HarnessRequest) -> Any:
        self.requests.append(request)
        return super().run_architect(request)

    def resume_architect(self, session_id: str, request: HarnessRequest) -> Any:
        self.requests.append(request)
        return super().resume_architect(session_id, request)


def make_evaluator_config(
    *,
    testset_path: str | None = None,
) -> EvaluatorConfig:
    return EvaluatorConfig(
        program_factory="my_pkg.program:make_program",
        metric_fns=["my_pkg.metrics:accuracy"],
        metric_names=["accuracy"],
        primary_metric="accuracy",
        testset_path=testset_path,
    )


def make_architect(
    *,
    iteration_idx: int = 1,
    agent_managed_git: bool = True,
    locked: bool = True,
    testset_path: str | None = None,
    state: FakeState | None = None,
    additional_deny_paths: list[str] | None = None,
) -> tuple[IterationArchitect, IterationContext, FakeState]:
    """Build a real IterationArchitect over in-memory fakes.

    ``locked=True`` mirrors the loop's iteration-start lock (engine 3: the
    loop locks the context at construction; there is no select_candidate).
    """
    ctx = IterationContext(
        iteration_idx=iteration_idx,
        parent_locked=locked,
        start_branch="codeevolver-20260601120000-main",
        start_sha="seedsha",
        primary_branch_at_start="codeevolver-20260601120000-main",
    )
    state = state or FakeState()
    cfg = ArchitectConfig(
        run_timestamp="20260601120000",
        additional_instructions=None,
        program_path="my_pkg.program:make_program",
        evaluator=make_evaluator_config(testset_path=testset_path),
        trainset_ids=list(range(50)),
        comparison_ids=list(range(10)),
        subsample_size=5,
        agent_managed_git=agent_managed_git,
        additional_deny_paths=additional_deny_paths or [],
    )
    arch = IterationArchitect(
        context=ctx,
        state=state,
        workspace="/workspace",
        config=cfg,
        rng=random.Random(0),
    )
    return arch, ctx, state
