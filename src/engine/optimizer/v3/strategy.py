"""Copyright © 2026 440 Labs LLC

`OptimizerStrategy` — the small seam that lets the shared optimizer loop
(`v3.optimizer_loop`) serve engine 3 methods without `if method ==` branching.

Engine 3 implements EE (agent-managed git + close_iteration),
`pareto_evolution` (the fixed-evolution research arm, `v3/pareto_evolution/`)
and `freeform` (research Baseline 1, `v3/freeform/`: EE's harness with no
explore/exploit guidance and no population management — the loop registers
HEAD at close).

Seams:
  1. **prompt_builder** — each method points at its own system-prompt builder
     in `prompts.py`. EE / evo builders are produced from one base template
     with strategy-owned sections spliced in
     (`prompts.make_architect_prompt_builder`); freeform has its own literal.
  2. **skills_to_stage** / **skill_overrides** / **skill_template_values** —
     which CodeEvolver skills the harness stages for the architect (freeform
     and pareto_evolution drop `exploration`), per-skill full `SKILL.md`
     replacements keyed by skill directory name (pareto_evolution and
     freeform each swap `close_iteration`), and a per-iteration
     callable returning extra `{placeholder}` values rendered into the staged
     skills (freeform's `{final_iteration_note}`). The staged directory keeps
     the canonical name so prompts and watchdogs that name a skill stay valid.
  3. **enable_explore_guard** — the explore-frequency guard (claimed
     exploration share < 30% after 4 iterations forces an exploration
     iteration). EE and ASA keep it; freeform and pareto_evolution (freeform
     prompt) turn it off.
  4. **Loop hooks** (`specs/research_plan.md` Variable Table):
     `on_iteration_start` runs BEFORE the architect session and before the
     iteration-start snapshot — pareto_evolution uses it to pick and check out
     the parent (`loop_owns_parent=True` then skips post-session parent
     derivation); ASA uses it to compute the eligible lineage set and the
     iteration's fired requirements; `on_seed_complete` runs after the seed step;
     `on_iteration_closed` applies admission / fitness / promotion rules to the
     close record before it is recorded; `close_iteration_mode="loop_branch"`
     registers the two-field `close_iteration` tool (no `register_branch`),
     `"head"` the one-field tool whose registration the loop derives from HEAD
     (`freeform.closing.register_head`). EE leaves all of these at their
     defaults — the agent owns them.
  5. **run_eval_gate / extra_tools / commit_before_comparison** (ASA,
     `specs/archive-public/v3_asa.md`): a per-call policy consulted by `run_eval` at the
     iteration's baseline (returns a note, or raises `RunEvalError` to refuse
     before any rows are spent); extra architect tools built per iteration
     (ASA's read-only `branch_comparison`); and whether `run_eval` commits a
     dirty tree before a comparison eval so the JSONL sha is the content.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Mapping

if TYPE_CHECKING:  # annotations only — keep strategy.py import-cycle free
    from src.engine.evaluator.config import EvaluatorConfig

    from .optimization_config import OptimizationConfig

    from .ce_state import CEState
    from .iteration_context import IterationContext
    from src.engine.schemas.iteration_history import IterationHistory
    from .records import CloseRecord

SystemPromptBuilder = Callable[..., str]
CloseIterationMode = Literal["register_branch", "loop_branch", "head"]


@dataclass(frozen=True)
class LoopServices:
    """Loop-owned handles handed to strategy hooks.

    Built once per run in `run_optimization` and threaded into the loop so a
    hook can run an eval or fork a branch with exactly the objects the loop
    itself uses (same evaluator config, same comparison ids, same rng).
    """

    workspace: str
    evaluator_config: "EvaluatorConfig"
    trainset_ids: list
    comparison_ids: list
    comparison_timeout: int | None
    run_timestamp: str
    rng: random.Random
    max_eval_rows: int | None = None
    # The run's OptimizationConfig (ASA reads `max_score`, `subsample_size`,
    # its `asa_*` knobs and the job's git creds from it). None in tests that
    # build services by hand.
    config: "OptimizationConfig | None" = None


IterationStartHook = Callable[["CEState", "IterationContext", LoopServices], None]
# (ctx, state, workspace, eval_sample) -> note appended to the run_eval
# response, or raise `RunEvalError` to refuse the call.
RunEvalGate = Callable[["IterationContext", "CEState", str, str], "str | None"]
# (ctx, state, workspace, services) -> {tool_name: callable} merged into the
# registry (the loop binds `services` before handing it to the architect).
ExtraToolsBuilder = Callable[
    ["IterationContext", "CEState", str, LoopServices], Mapping[str, Callable[..., dict]]
]
SeedHook = Callable[["CEState", "IterationHistory", LoopServices], None]
CloseHook = Callable[
    ["IterationContext", "CEState", "CloseRecord", LoopServices], None
]
SkillTemplateValues = Callable[["IterationContext"], Mapping[str, str]]


@dataclass(frozen=True)
class OptimizerStrategy:
    """Per-optimizer policy consumed by `v3.optimizer_loop.run_optimization`.

    Attributes:
        method: Stamps `result["method"]` and progress logs.
        prompt_builder: Callable returning the architect's system prompt.
        skills_to_stage: Skill directory names to stage into the harness, or
            None to stage all canonical CodeEvolver skills (test/dev only —
            every arm names its skills explicitly; spec §29).
        skill_overrides: `{skill_dir_name: path_to_replacement_SKILL.md}`;
            the replacement is staged under the canonical directory name.
        skill_template_values: Per-iteration extra `{placeholder}` values
            rendered into the staged skills; None = none.
        enable_explore_guard: Whether the explore-frequency guard runs.
        loop_owns_parent: The loop selects/checks out the parent at iteration
            start (inside `on_iteration_start`) and post-session parent
            derivation is skipped (`parent_derivation="preselected"`).
        close_iteration_mode: Which `close_iteration` tool the architect gets.
        on_iteration_start / on_seed_complete / on_iteration_closed: loop
            hooks (see module docstring); None = not used.
        run_eval_gate: Policy consulted by `run_eval` at the iteration's
            baseline (ASA); None = no gate.
        extra_tools: Builder for extra architect tools (ASA's
            `branch_comparison`); None = none.
        commit_before_comparison: `run_eval` commits a dirty tree before a
            comparison eval (ASA, pareto_evolution).
        comparison_ack_note: Arm-specific acknowledgement note shown before a
            comparison run (specs/archive-public/comparison_credits.md); None = the standard
            note in `run_eval_tool.STANDARD_COMPARISON_ACK_NOTE`.
        comparison_closes_iteration: `run_eval` refuses every call after the
            iteration's comparison eval — the comparison is the close step
            (pareto_evolution).
    """

    method: str
    prompt_builder: SystemPromptBuilder
    skills_to_stage: tuple[str, ...] | None = None
    enable_explore_guard: bool = True
    skill_overrides: Mapping[str, Path] | None = None
    skill_template_values: SkillTemplateValues | None = None
    loop_owns_parent: bool = False
    close_iteration_mode: CloseIterationMode = "register_branch"
    on_iteration_start: IterationStartHook | None = None
    on_seed_complete: SeedHook | None = None
    on_iteration_closed: CloseHook | None = None
    run_eval_gate: RunEvalGate | None = None
    extra_tools: ExtraToolsBuilder | None = None
    commit_before_comparison: bool = False
    comparison_ack_note: str | None = None
    comparison_closes_iteration: bool = False

    def __post_init__(self) -> None:
        if self.loop_owns_parent and self.on_iteration_start is None:
            raise ValueError(
                f"strategy {self.method!r}: loop_owns_parent=True requires an "
                "on_iteration_start hook that forks the child branch"
            )


__all__ = [
    "CloseHook",
    "CloseIterationMode",
    "ExtraToolsBuilder",
    "IterationStartHook",
    "LoopServices",
    "OptimizerStrategy",
    "RunEvalGate",
    "SeedHook",
    "SkillTemplateValues",
    "SystemPromptBuilder",
]
