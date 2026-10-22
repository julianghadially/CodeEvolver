"""Copyright © 2026 440 Labs LLC

BaseAgentConfig — what the coding harnesses and the shared `run_eval` tool
read about an agent session. Pure schema; no logic.

Engine 3's `ArchitectConfig` extends it with the fields its closing protocol
and prompt family need; the simple loop constructs it directly. The harness
keys its behavior off these fields only (`stage_project_skills`,
`agent_managed_git`, `capture_architect_traces`, `evaluator.testset_path`,
`additional_deny_paths`, `max_turns`); the loop calls `system_prompt_builder`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from src.engine.evaluator.config import EvaluatorConfig


@dataclass(kw_only=True)
class BaseAgentConfig:
    """Harness + eval-tool settings for one agent session."""

    additional_instructions: str | None
    program_path: str | None
    evaluator: EvaluatorConfig
    trainset_ids: list[int]
    comparison_ids: list[int] = field(default_factory=list)
    subsample_timeout: int | None = None
    comparison_timeout: int | None = None
    max_score: float = 1.0
    subsample_size: int = 20
    # Repeat count of the architect's once-per-run `run_variance_estimate`
    # (specs/archive-public/sampling_variance.md); the simple loop never exposes that tool.
    variance_estimate_samples: int = 5
    max_turns: int = 350
    coding_harness: str = "opencode"
    coding_harness_model: str | None = None
    # Bare Anthropic model id for the agent session + its sub-agents (Claude
    # harness only; None on OpenCode, whose model is `coding_harness_model`).
    agent_model: str | None = None
    # True: the agent owns branch choice and the `close_iteration` tool is
    # exposed. False: no close tool.
    agent_managed_git: bool = True
    # False suppresses CodeEvolver skill staging, the OpenCode `/s` bootstrap /
    # `/update-specs` finalization, and the Claude `Skill` tool.
    stage_project_skills: bool = True
    skills_to_stage: tuple[str, ...] | None = None
    skill_overrides: Mapping[str, Path] | None = None
    # Extra `{placeholder}` values rendered into staged skill markdown on top
    # of the directory paths (per-iteration content such as a budget note).
    skill_template_values: Mapping[str, str] | None = None
    # Renders the system prompt: (additional_instructions, program_path,
    # workspace_path, context) -> str.
    system_prompt_builder: Callable[..., str] | None = None
    # Capture the agent's own session (thinking / tool calls / results) to
    # `architect_traces_dir()` as `iteration_{N}_architect_trace/`.
    capture_architect_traces: bool = False
    # Client-supplied deny-path fragments, appended to the holdout fence.
    additional_deny_paths: list[str] = field(default_factory=list)
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md). The
    # acknowledgement note shown before a comparison run (None = the standard
    # note in `run_eval_tool.STANDARD_COMPARISON_ACK_NOTE`; arm strategies
    # override it). `comparison_closes_iteration=True` (pareto_evolution)
    # refuses every run_eval call after the iteration's comparison eval —
    # the comparison is the close step in that arm.
    comparison_ack_note: str | None = None
    comparison_closes_iteration: bool = False


__all__ = ["BaseAgentConfig"]
