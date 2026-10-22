"""Copyright © 2026 440 Labs LLC

ASA — Artificial Selection Algorithm (Treatment C in
`specs/research_plan.md`; `specs/archive-public/v3_asa.md`).

EE plus a REQUIREMENT to traverse the population tree: rule-driven prompt
injections (new lineage / eligible non-primary parent) enforced at the
run_eval baseline with a violation allowance, a read-only `branch_comparison`
tool (positive behavioral contribution + pruned Pareto score per lineage),
the seed kept as its own lineage, and loop-run comparison evals at
promotion / when the primary's record goes stale. See `asa_optimizer.py`.
"""

from .asa_optimizer import ASA_STRATEGY, run_asa_optimization

__all__ = ["ASA_STRATEGY", "run_asa_optimization"]
