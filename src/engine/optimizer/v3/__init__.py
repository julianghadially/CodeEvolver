"""Copyright © 2026 440 Labs LLC

EE (exploit and explore) optimizer.

CodeEvolver v3's optimization loop: an agentic IterationArchitect picks a
parent candidate, proposes a hypothesis, runs subsample evaluations, invokes
the coding agent, and emits an outcome reflection — all inside one coding
harness session per iteration. `ce_state.CEState` is the thin state
substrate; this package owns the loop.

Sibling folders under `src/engine/optimizer/` can host other optimizers in
the future.

This package is environment-agnostic: no database or cloud SDK
imports. It runs as a single process (locally or inside the Docker runner), launched via
`optimization_entry.py`.
"""

from .iteration_context import IterationContext

__all__ = [
    "IterationContext",
]
