"""Copyright © 2026 440 Labs LLC

Eval-row budget policy for the `run_eval` tool.

Pure functions. `RunEvalBinding` reads the live counters (`ctx.max_eval_rows`,
`state.total_num_evals`, `state.final_comparison_evals`) and hands them
here; nothing in this module touches git, the evaluator or state.

The rule (specs/engine3.md):
once `rows_used >= max_rows`, `train` / `single` / `variance` evals are
refused and exactly `FINAL_COMPARISON_ALLOWANCE` more `comparison` evals are
allowed.
Under budget every sample is allowed, however large — an eval is judged by
the budget at the moment it STARTS, so the last one may overshoot.
"""

from __future__ import annotations

from typing import Any

from .errors import RunEvalError

# Comparison evals allowed AFTER the eval-row budget is reached. One —
# enough to compare the branch about to become primary against the
# incumbent during wrap-up (and to match the single post-exhaustion
# comparison the fixed-evolutionary arms get), not enough to keep
# optimizing.
FINAL_COMPARISON_ALLOWANCE = 1


def budget(ctx: Any, state: Any | None) -> tuple[int, int] | None:
    """`(rows_used, max_rows)`, or None when this run has no budget.

    `max_eval_rows` is the run-level budget stamped on the context by the
    loop; `state.total_num_evals` is the live counter every eval bumps, so
    the pair is always current — unlike `ctx.evals_used`, which is a
    start-of-iteration snapshot taken for the system prompt.
    """
    max_rows = ctx.max_eval_rows
    if not max_rows or max_rows <= 0 or state is None:
        return None
    used = int(getattr(state, "total_num_evals", 0) or 0)
    return used, int(max_rows)


def final_comparisons_left(state: Any | None, has_comparison_set: bool) -> int:
    """Post-budget comparison evals still available on this run."""
    # No comparison set on this run → no allowance to offer.
    if not has_comparison_set:
        return 0
    spent = int(getattr(state, "final_comparison_evals", 0) or 0)
    return max(0, FINAL_COMPARISON_ALLOWANCE - spent)


def enforce_eval_budget(
    eval_sample: str,
    *,
    budget: tuple[int, int] | None,
    has_comparison_set: bool,
    final_left: int,
) -> bool:
    """Refuse evals the run's eval-row budget no longer allows.

    No-op only on runs with no configured budget. Otherwise this applies
    everywhere — SaaS and research runs alike — so the wrap-up affordance
    (and the budget note describing it) means exactly one thing.

    The rule, once `rows_used >= max_rows`:
      - `train` / `single` → refused. The architect is done optimizing.
      - `comparison` → allowed `FINAL_COMPARISON_ALLOWANCE` times, then
        refused.
    Under budget every sample is allowed, however large — an eval is
    judged by the budget at the moment it STARTS, so the last one may
    legitimately overshoot.

    Returns True when this call is spending the post-budget comparison
    allowance (the caller marks it consumed once the eval has run).
    """
    if budget is None:
        return False
    used, max_rows = budget
    if used < max_rows:
        return False

    if eval_sample == "comparison":
        if not has_comparison_set:
            # This run has no comparison set; let _resolve_rows say so
            # rather than blaming the budget for an impossible call.
            return False
        if final_left > 0:
            return True
        raise RunEvalError(
            f"Eval budget spent: {used}/{max_rows} rows used, and this "
            "run's one final comparison-set eval has already been run. No "
            "further evals of any kind are available. Wrap up: make sure "
            "the branch you want as the run's primary is registered, and "
            "close the iteration."
        )

    if eval_sample == "variance":
        raise RunEvalError(
            f"Eval budget reached: {used}/{max_rows} rows used. "
            "run_variance_estimate needs eval rows for its repeated runs and "
            "none remain, so it is refused. Wrap up: make sure the branch you "
            "want as the run's primary is registered, and close the iteration."
        )

    left = final_left
    if left <= 0:
        raise RunEvalError(
            f"Eval budget spent: {used}/{max_rows} rows used. No further "
            "evals of any kind are available — make sure the branch you "
            "want as the run's primary is registered, and close the "
            "iteration."
        )
    raise RunEvalError(
        f"Eval budget reached: {used}/{max_rows} rows used. 0 rows "
        "remaining, so no further training evals can be run — "
        "eval_sample='train' and eval_sample='single' are both refused "
        f"from here on. {left} final comparison set run remaining: you "
        "may call run_eval(eval_sample='comparison') once to check the "
        "branch you intend to leave as the run's primary against the "
        "fixed comparison set. Then close the iteration."
    )


def budget_fields(
    *,
    budget: tuple[int, int] | None,
    has_comparison_set: bool,
    final_left: int,
) -> dict[str, Any]:
    """`eval_rows_remaining` + `budget_note` for the tool response.

    Emitted on every eval so the architect always learns the CURRENT
    remaining budget from the eval it just ran, rather than from the
    start-of-iteration figure in its system prompt.

    The final-comparison sentence appears only once rows are spent — while
    budget remains it would read as a spendable extra — and only on runs
    that have a comparison set to run. It is what stops "0 rows remaining"
    from reading as "stop working": the architect still has its wrap-up
    comparison.
    """
    if budget is None:
        return {}
    used, max_rows = budget
    remaining = max(0, max_rows - used)
    if remaining > 0:
        return {
            "eval_rows_remaining": remaining,
            "budget_note": f"{remaining} rows remaining.",
        }
    if not has_comparison_set:
        # No comparison set on this run — there is no wrap-up eval to
        # offer, so don't invent an allowance to report.
        return {"eval_rows_remaining": 0, "budget_note": "0 rows remaining."}
    left = final_left
    if left == 1:
        note = (
            "0 rows remaining. 1 final comparison set run remaining — no "
            "further training evals will be accepted."
        )
    elif left == 0:
        note = (
            "0 rows remaining. 0 final comparison set runs remaining — "
            "this run's eval budget is fully spent."
        )
    else:
        note = f"0 rows remaining. {left} final comparison set runs remaining."
    return {
        "eval_rows_remaining": 0,
        "final_comparison_runs_remaining": left,
        "budget_note": note,
    }


# ----------------------------------------------------------------------
# Comparison-credit ledger (specs/archive-public/comparison_credits.md)
# ----------------------------------------------------------------------
def credits_balance(state: Any | None) -> int:
    return int(getattr(state, "comparison_credits_balance", 0) or 0)


def enforce_comparison_credits(*, balance: int, budget: tuple[int, int] | None) -> None:
    """Refuse an agent-initiated comparison eval when the ledger is empty.

    Only the agent-facing path calls this; loop-initiated runs debit without
    a check (a fixed rule is never blocked — it overdraws instead), and the
    post-budget final-comparison allowance is exempt entirely.
    """
    if balance > 0:
        return
    remaining = "unknown"
    if budget is not None:
        used, max_rows = budget
        remaining = str(max(0, max_rows - used))
    raise RunEvalError(
        f"Comparison-set run refused: {balance} comparison credit(s) "
        "remaining. Credits replenish at the start of each iteration "
        f"(eval-row budget remaining: {remaining}). Run training evals, or "
        "close the iteration and run the comparison set in a later "
        "iteration."
    )


def acknowledgement_refusal(
    *,
    note: str,
    code: str,
    balance: int,
    budget: tuple[int, int] | None,
) -> dict[str, Any]:
    """The structured (non-error) response for an unacknowledged comparison
    call: the arm's acknowledgement note plus the per-iteration verification
    code. No rows are spent; the agent re-calls run_eval with
    `acknowledge_message=<code>` to proceed."""
    remaining: int | None = None
    if budget is not None:
        used, max_rows = budget
        remaining = max(0, max_rows - used)
    return {
        "ok": False,
        "refused": "acknowledgement_required",
        "note": note,
        "verification_code": code,
        "comparison_credits_remaining": balance,
        **({"eval_rows_remaining": remaining} if remaining is not None else {}),
        "next_step": (
            "To proceed, call run_eval again with the same arguments plus "
            f"acknowledge_message='{code}'. This code is valid for the rest "
            "of the iteration."
        ),
    }


def credits_fields(*, balance: int, enabled: bool) -> dict[str, Any]:
    """`comparison_credits_remaining` for the tool response (comparison evals
    only — train evals don't touch the ledger and shouldn't mention it)."""
    if not enabled:
        return {}
    return {"comparison_credits_remaining": balance}


__all__ = [
    "FINAL_COMPARISON_ALLOWANCE",
    "acknowledgement_refusal",
    "budget",
    "budget_fields",
    "credits_balance",
    "credits_fields",
    "enforce_comparison_credits",
    "enforce_eval_budget",
    "final_comparisons_left",
]
