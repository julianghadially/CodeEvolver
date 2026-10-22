"""Copyright © 2026 440 Labs LLC

Pareto-Evolution parent-selection policy (GEPA-inspired).

GEPA selects parents by building the per-instance Pareto frontier and
sampling a candidate with probability proportional to the number of
instances it is a frontier winner on. We reuse the same rule here, reading
the frontier from the loop-owned `Population` (per-row comparison-set
scores) — selection happens in the optimizer loop, before the architect
session, via `OptimizerStrategy.on_iteration_start`; the architect never chooses
its branch.

The candidate **merge** half of GEPA is agent-driven: the exploit-and-merge
draw below both picks the best candidate and (via `ctx.exploit_phase`)
surfaces `prompts.EVO_MERGE_NOTE`, which tells the architect to port ideas
from other branches with git. No crossover operator.
"""

from __future__ import annotations

import logging
import math
import random
from collections import Counter
from typing import TYPE_CHECKING, Any

from src.engine.paths import evo_exploit_k, evo_exploit_threshold

from ..iteration_context import IterationContext
from ..select_parent import fork_child_from
from .population import Population

if TYPE_CHECKING:
    from ..ce_state import CEState
    from ..strategy import LoopServices

logger = logging.getLogger(__name__)


def _instance_front_counts(state: Any) -> Counter[int]:
    """Count, per candidate key, how many rows it wins on the instance
    Pareto frontier.

    `get_pareto_front()` returns `{row_id: set[key]}` — the set of candidates
    tied for best on each row. A candidate's selection weight is the number
    of rows whose winner-set contains it (GEPA's Pareto-sampling rule).
    """
    counts: Counter[int] = Counter()
    getter = getattr(state, "get_pareto_front", None)
    if not callable(getter):
        return counts
    try:
        front = getter() or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("pareto: get_pareto_front() failed (%s)", exc)
        return counts
    for winners in front.values():
        for idx in winners:
            counts[int(idx)] += 1
    return counts


def _weighted_choice(weights: dict[int, int], rng: random.Random) -> int | None:
    """Pick a key with probability proportional to its positive weight."""
    items = [(k, w) for k, w in weights.items() if w > 0]
    if not items:
        return None
    total = sum(w for _, w in items)
    r = rng.random() * total
    upto = 0.0
    for k, w in items:
        upto += w
        if r <= upto:
            return k
    return items[-1][0]  # float-rounding guard


def _pareto_sample(state: Any, rng: random.Random) -> int:
    """Sample a parent key per GEPA's Pareto-proportional rule.

    Fallback chain (keeps iteration 1, where only the seed exists, working):
      1. instance-front win counts, sampled proportionally;
      2. union of all frontier members, uniform;
      3. `get_best_program()`;
      4. seed (key 0).
    """
    counts = _instance_front_counts(state)
    try:
        admitted = len(state)
    except TypeError:
        admitted = None
    logger.info(
        "pareto: pruned front holds %d of %s admitted candidates",
        len(counts), admitted if admitted is not None else "?",
    )
    chosen = _weighted_choice(dict(counts), rng)
    if chosen is not None:
        logger.info(
            "pareto: sampled candidate=%d from instance front (weights=%s)",
            chosen, dict(counts),
        )
        return chosen

    members = getattr(state, "get_frontier_members", None)
    if callable(members):
        try:
            pool = sorted(int(i) for i in (members() or set()))
        except Exception:  # noqa: BLE001
            pool = []
        if pool:
            pick = pool[rng.randrange(len(pool))]
            logger.info("pareto: empty front; uniform frontier-member candidate=%d", pick)
            return pick

    best = getattr(state, "get_best_program", None)
    if callable(best):
        try:
            return int(best())
        except Exception:  # noqa: BLE001
            pass
    return 0


def _exploit_probability(ctx: IterationContext) -> float:
    """Per-iteration probability of entering the exploit-and-merge phase.

    A smooth function of the eval-budget fraction spent this run:

        budget = ctx.evals_used / ctx.max_eval_rows
        p = min(0.5, 0.5 * (1 + tanh(k * (budget - threshold))))

    with `k = evo_exploit_k()` and `threshold = evo_exploit_threshold()`. The
    0.5 cap means even a fully-spent budget only ever gives a coin-flip chance
    of exploiting on any single iteration (at `budget == threshold` the raw
    curve already sits at exactly 0.5). Returns 0.0 when no eval-row budget is
    configured — the run never enters the exploit phase without a budget.
    """
    if not ctx.max_eval_rows:
        return 0.0
    budget = ctx.evals_used / ctx.max_eval_rows
    hyperbolic = 0.5 * (1.0 + math.tanh(evo_exploit_k() * (budget - evo_exploit_threshold())))
    return min(0.5, hyperbolic)


def choose_parent_iteration(
    pop: Population, ctx: IterationContext, rng: random.Random
) -> int:
    """Draw the exploit decision once, stamp `ctx.exploit_phase`, and return
    the parent candidate's iteration number.

    Exploit phase → the best candidate (max comparison-set score), so the
    architect — shown the merge note off the *same* `ctx.exploit_phase` flag
    (`prompts._evo_merge_note`) — can port the best ideas from other branches
    onto the best program. Otherwise → Pareto-proportional sample.
    """
    p = _exploit_probability(ctx)
    ctx.exploit_phase = rng.random() < p
    if ctx.exploit_phase:
        try:
            it = int(pop.get_best_program())
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "pareto: exploit phase get_best_program() failed (%s); "
                "falling back to pareto sample", exc,
            )
            it = _pareto_sample(pop, rng)
        logger.info(
            "iteration %d: selecting best candidate due to exploit phase "
            "trigger (p=%.3f at %.0f%% of eval budget): candidate=%d",
            ctx.iteration_idx, p,
            100.0 * ctx.evals_used / ctx.max_eval_rows if ctx.max_eval_rows else 0.0,
            it,
        )
    else:
        it = _pareto_sample(pop, rng)
        logger.info(
            "iteration %d: pareto preselect — parent candidate=%d (exploit p=%.3f)",
            ctx.iteration_idx, it, p,
        )
    return it


def evo_select_parent(
    state: "CEState",
    ctx: IterationContext,
    services: "LoopServices",
) -> None:
    """`OptimizerStrategy.on_iteration_start` for pareto_evolution.

    Loads the population, draws the parent, and forks this iteration's child
    branch from the candidate's recorded close sha (`fork_child_from`).
    """
    pop = Population.load_or_none()
    if pop is None or len(pop) == 0:
        raise RuntimeError(
            "pareto_evolution: population.json is missing or empty — the "
            "on_seed_complete hook must seed candidate 0 before iteration 1"
        )
    it = choose_parent_iteration(pop, ctx, services.rng)
    cand = pop.get(it)
    fork_child_from(
        ctx=ctx,
        parent_branch=cand["branch"],
        parent_sha=cand["sha"],
        parent_iteration=it,
        workspace=services.workspace,
        run_timestamp=services.run_timestamp,
    )


__all__ = ["choose_parent_iteration", "evo_select_parent"]
