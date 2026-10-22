"""Copyright © 2026 440 Labs LLC

EE (exploit and explore) optimizer — the canonical, production CodeEvolver
optimization method.

Engine 3: the IterationArchitect owns git — it selects/creates its own
lineage branches and finalizes each iteration through the `close_iteration`
tool (the source of truth for the primary branch).

`run_ee_optimization` is a thin wrapper that builds the EE
`OptimizerStrategy` and calls the shared driver in `v3.optimizer_loop`.
"""

from .ee_optimizer import run_ee_optimization

__all__ = ["run_ee_optimization"]
