"""Copyright © 2026 440 Labs LLC

Pareto-Evolution optimizer — the fixed-evolution research arm (Treatment B
in `specs/research_plan.md`; GEPA-inspired).

NOT the production path (that is `v3/ee/`). Runs on the same engine 3
`optimizer_loop` as EE and differs only where the research plan's Variable
Table says a fixed rule replaces the agent: parent selection (Pareto-
proportional sampling over the comparison-set instance frontier, with an
exploit-and-merge draw late in the budget), comparison-fitness timing,
candidate admission / primary promotion, and the (loop-internal) population
log. See `pareto_optimizer.py` for the hooks and `population.py` for the
state.
"""

from .pareto_optimizer import PARETO_STRATEGY, run_pareto_optimization

__all__ = ["PARETO_STRATEGY", "run_pareto_optimization"]
