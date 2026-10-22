"""Copyright © 2026 440 Labs LLC

EE (exploit and explore) optimizer — the production CodeEvolver optimization
method.

EE (engine 3) is the **agent-managed-git** optimizer: the
IterationArchitect owns branch choice, manages its own lineage branches, and
finalizes each iteration through the `close_iteration` tool. This module is
intentionally thin — it selects the EE `OptimizerStrategy` and delegates
the entire run (seed step, architect loop, finalization, callbacks) to the
shared `v3.optimizer_loop.run_optimization`.
"""

from __future__ import annotations

from ..optimizer_loop import run_optimization
from ..prompts import build_architect_system_prompt
from ..strategy import OptimizerStrategy

# EE's strategy: agent-managed git, only the spec'd skills staged (spec §29:
# prompt_engineering / tracing live in the engine but are never staged),
# explore-frequency guard armed.
EE_STRATEGY = OptimizerStrategy(
    method="ee",
    prompt_builder=build_architect_system_prompt,
    skills_to_stage=("close_iteration", "exploration"),
    enable_explore_guard=True,
)


def run_ee_optimization(**kwargs) -> dict:
    """Run a full EE optimization.

    Thin wrapper over the shared driver — see
    `v3.optimizer_loop.run_optimization` for the full keyword-argument
    contract. Environment-agnostic: assumes the workspace is already set up
    with the cloned repo, deps installed, and API keys in env vars. No
    database imports.
    """
    return run_optimization(strategy=EE_STRATEGY, **kwargs)


__all__ = ["EE_STRATEGY", "run_ee_optimization"]
