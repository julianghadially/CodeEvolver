"""Copyright © 2026 440 Labs LLC

Loop-initiated comparison-set evaluation.

The fixed-evolution arm (`pareto_evolution`) owes every admitted candidate a
fitness measurement on the comparison set. Normally the architect runs it at
close (the evo `close_iteration` skill says so); when it did not, or when the
closing content differs from what it measured, the LOOP runs it here.

This reuses `RunEvalBinding` so the eval lands exactly where an agent call
would — counted against the budget, on `ctx.eval_records` with per-row
scores, in `comparison_set_history.jsonl` — rather than the finalization
`_eval_branch` path, which is deliberately uncounted and unrecorded.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.services.subprocess_service import clean_python_bytecode, git_checkout

from .run_eval_tool import RunEvalBinding

if TYPE_CHECKING:
    from .ce_state import CEState
    from .iteration_context import IterationContext
    from .strategy import LoopServices

logger = logging.getLogger(__name__)


def run_loop_comparison_eval(
    *,
    ctx: "IterationContext",
    state: "CEState",
    services: "LoopServices",
    ref: str,
) -> EvalResult:
    """Check out `ref` and run the comparison set on it for the loop."""
    workspace = services.workspace
    clean_python_bytecode(workspace)
    git_checkout(workspace, ref)
    binding = RunEvalBinding(
        ctx=ctx,
        workspace=workspace,
        config=services.evaluator_config,
        trainset_ids=services.trainset_ids,
        comparison_ids=services.comparison_ids,
        rng=services.rng,
        comparison_timeout=services.comparison_timeout,
        state=state,
    )
    logger.info(
        "iteration %d: loop-initiated comparison eval on %s (%d rows)",
        ctx.iteration_idx, ref, len(services.comparison_ids),
    )
    result = binding.run_comparison_for_loop()
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): loop-initiated
    # runs debit the same ledger as agent calls — a fixed rule is never
    # blocked, so the balance may go negative (overdraft).
    cfg = services.config
    if cfg is not None and getattr(cfg, "comparison_credits_per_iteration", None) is not None:
        state.debit_comparison_credit()
        logger.info(
            "iteration %d: loop comparison eval debited 1 credit (balance=%d)",
            ctx.iteration_idx, state.comparison_credits_balance,
        )
    return result


__all__ = ["run_loop_comparison_eval"]
