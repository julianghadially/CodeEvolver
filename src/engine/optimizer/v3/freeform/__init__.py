"""Copyright © 2026 440 Labs LLC

Freeform optimizer — research Baseline 1 (`specs/v3_freeform.md`).

The CodeEvolver harness (memory, coding agent, inspect_traces, hooks) with
no explore/exploit guidance and no population management: the agent improves
the program as it normally would, and the loop registers whatever HEAD is at
the end of each iteration as the submitted state.

`run_freeform_optimization` is a thin wrapper that builds the freeform
`OptimizerStrategy` and calls the shared driver in `v3.optimizer_loop`.
"""

from .freeform_optimizer import run_freeform_optimization

__all__ = ["run_freeform_optimization"]
