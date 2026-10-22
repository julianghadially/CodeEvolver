"""Copyright © 2026 440 Labs LLC

Simple loop — research Baseline 0 (`specs/v3_simple_loop.md`).

A fresh OpenCode session per iteration on one linear branch, with a single
`run_eval` tool and three controls: the eval-row budget, a post-session
constraint gate, and the post-run auditors in `src/engine/optimizer/meta/`.
"""

from .simple_loop_optimizer import run_simple_loop_optimization

__all__ = ["run_simple_loop_optimization"]
