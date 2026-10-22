# Comparison-set credit system

Implemented 2026-08-30. See `specs/engine3.md` §"Comparison-set credit system" for the cross-reference.

## Motivation
The resource use is a combination of eval runs and iterations. We measure eval rows, which roughly map to iterations. However, some methods use more budget per iteration than others. That is observable and okay.

However, budget use per iteration is an implementation detail that can bias the comparison of strategies. In particular, the comparison set is worth four subsample evaluations (200 vs 50 rows at research defaults) and can drastically change the number of total iterations for a run — the first freeform hover run spent ~67% of its 5000-row budget on comparison evals and was on track for ~9 iterations, versus 23 on engine 2 and the ~20+ the population arms (pareto_evolution, asa) need for their selection mechanisms to function.

To prevent overly budget-hungry iterations, comparison-set runs are paced by a credit ledger. It is a budget mechanic, uniform across arms — not workflow guidance — so it does not touch the freeform arm's construct (no strategy constraints) and does not cap change rounds or within-iteration ordering.

## The ledger
- **Schedule:** 1 credit granted up front at the start of the seed round; +1 at each iteration start. Config knobs `comparison_credits_initial=1` / `comparison_credits_per_iteration=1` (`OptimizationConfig`, `OptimizeRequest`, launcher kwargs, pinned in `runner/entrypoint.py::RESEARCH_DEFAULTS`). `comparison_credits_per_iteration=None` disables the whole feature (no gate, no acknowledgement). No bank cap — hoarding credits for late-run evidence is legitimate strategy.
- **Balance lives on the run state** (`CEState.comparison_credits_balance`, schema v6; `SimpleLoopState` equivalent) and is persisted, so a resume cannot re-earn credits.
- **Agent-initiated comparison evals debit 1** and are **refused at balance <= 0** (`eval_budget.enforce_comparison_credits`, before any rows are spent). Debit happens after the eval returns — a call that produced no score spends no credit.
- **Loop-initiated runs debit 1 but may overdraw** (negative balance; `loop_eval.run_loop_comparison_eval`). A fixed rule is never blocked by the ledger — evo's admission guarantee and ASA's promotion evidence stay deterministic.
- **Exempt:** the post-budget `FINAL_COMPARISON_ALLOWANCE` wrap-up run (its own persisted one-shot accounting; no ack, no debit) and the finalization evals (never counted anywhere).
- **Starting balances:** evo/ASA spend the up-front credit on their seed comparison eval and start iteration 1 at balance 1; EE/freeform/simple_loop start iteration 1 at balance 2 (their iteration 0 ran no comparison).

## Acknowledgement mechanism (in the tool, not the prompt)
The `run_eval` tool schema gains an optional `acknowledge_message` parameter. Per iteration, the loop stamps a fresh random 2-digit verification code on the context (`BaseIterationContext.comparison_ack_code`, drawn from the run rng so it is reproducible from `seed`). The first comparison call of an iteration without the correct code returns a **structured non-error refusal** (`ok: false, refused: "acknowledgement_required"`) carrying the arm's acknowledgement note (live values templated in), the code, the credit balance, and the remaining row budget — no rows spent. Re-calling with `acknowledge_message=<code>` proceeds and latches the ack for the rest of the iteration. Free-text Y/N was rejected: it needs fuzzy parsing and a destructive default.

Notes are arm-specific (`OptimizerStrategy.comparison_ack_note` → `BaseAgentConfig` → `RunEvalBinding`):
- **Standard** (`run_eval_tool.STANDARD_COMPARISON_ACK_NOTE`) — EE, freeform, simple_loop: cost framing (comparison vs subsample rows), credit schedule, balances.
- **Evo** (`pareto_optimizer.EVO_COMPARISON_ACK_NOTE`): the comparison is the close step; one run per iteration; run it at the end after committing; changes after it are reverted.
- **ASA** (`asa_optimizer.ASA_COMPARISON_ACK_NOTE`): standard framing plus "future agents pick up your work based on these results (branch_comparison tool); best practice is to run it at the end."

Every comparison-eval response carries `comparison_credits_remaining`; the `iteration_{N}_evals.jsonl` line carries `credits_balance_before`; the close record carries `comparison_credits_after` (`ITERATION_RECORD_FIELDS`, DB schema).

## Loop-driven spend
For both evo and ASA, the loop never triggers a second comparison run in an iteration where the agent already ran one.

### Evo (pareto_evolution)
The comparison run is the **close step** (`OptimizerStrategy.comparison_closes_iteration=True`):
- `commit_before_comparison=True` (ASA's seam, now shared): a dirty tree is auto-committed before the comparison eval, so the run point is always a commit.
- After the iteration's comparison eval, **every further `run_eval` call is refused** (`run_eval_tool.COMPARISON_CLOSED_NOTE`, pointing at the `close_iteration` skill). This also enforces one comparison per iteration. Session termination from inside a tool call was considered and rejected (no clean abort channel in either harness; it would strand the memory-diff watchdog's resume) — the hard refusal plus the deterministic revert below achieve the contract, and the memory-diff watchdog still runs post-session.
- At close, when the closing tree is not program-equivalent to the comparison eval's tree, the loop **reverts the branch to the comparison-run commit** (`close_effects.revert_to_comparison_point`; `comparison_state="REVERTED_TO_COMPARISON_POINT"`) instead of paying for a rerun. The candidate is admitted with the fitness vector of the content actually measured, so the population never holds a mismeasured candidate.
- **Admission is the existence of that comparison eval** (2026-08-31, `specs/engine3.md` §"Pareto admission by comparison eval"): the agent decides whether to run it; a train tie or regression is not a barrier. The backstop rerun stays only for the no-comparison-ran case **when the train subsample strictly progressed** (`RERUN_BY_LOOP`, counted, debits with overdraft). A failed revert also falls back to the rerun. An iteration with neither is `no_comparison` and spends nothing.
- The `COMPARISON_RERUN` watchdog is retired (2026-08-30) — the up-front ack note plus the deterministic revert supersede the warn-then-rerun flow.
- Credit arithmetic: at most one comparison per iteration (agent-run OR the loop fallback, never both) versus +1/iteration accrual, so evo hovers near balance 0–1. Since admission now requires the comparison eval, the agent has a reason to spend its one credit in every iteration it wants admitted; the loop's overdraft covers the improved-but-forgot case.

### ASA
Loop-driven comparison runs (all counted, all debit with overdraft, at most one per close):
1. **Seed** (`asa_on_seed_complete`): one run representing both seed branches by tree — spends the up-front credit.
2. **Promotion without tip-matching evidence**: the promoted primary lacks a comparison record matching its tip (exact or program-equivalent) → loop runs it (`reason="promotion"`).
3. **Stale primary refresh**: no promotion, and the primary's newest tip-matching record predates more than `asa_primary_stale_landed_iterations` landed iterations → loop refreshes it (`reason="stale_primary"`; skipped in the free tail).

Both (2) and (3) are **skipped when the agent ran any comparison eval this iteration** (`asa_loop_comparison_reason="skipped_agent_ran_comparison"`) — the agent's reading stands even when not tip-matching.

### Other arms
EE and freeform have no loop-driven comparison runs; all their comparison spend is agent-initiated and fully credit-gated. simple_loop's comparison evals go through the same shared `RunEvalBinding`, so the gate and ack apply there too (its uncounted finalization comparison is loop-side and exempt).

## Tests
- **Mechanism (offline):** `tests/unit/test_comparison_credits.py` — ack round-trip, gate, overdraft, final-allowance precedence over an available credit, disabled path, persistence, config knobs, `acknowledge_message` in the tool schema, `comparison_credits_after` whitelisted end to end. `tests/coding_harness/opencode/test_opencode_mcp_proxy.py` — the proxy forwards `acknowledge_message` and renders the `ok: False` refusal as a normal (non-`isError`) result.
- **Loop wiring (offline, real git):** `tests/controls/test_credits_loop_wiring.py` — drives the real `_run_optimizer_loop` with a stub architect over the production `build_tool_registry`: +N grant at iteration start, 2-digit code stamped from the run rng (reproducible from `seed`), a registry comparison call refused-then-debited against the loop's state, `comparison_credits_after` on the record, nothing when disabled.
- **Arm contracts (offline):** `tests/controls/test_evo_admission.py` (revert-to-comparison-point, failed-revert fallback, backstop), `tests/controls/test_close_side_effects.py` (real-git revert), `tests/controls/test_asa_close.py` (skip-when-agent-ran rules), `tests/unit/test_experiment_fleet.py` (research defaults pin the knobs).
- **Model behavior (live, billable):** `tests/functional/test_comparison_credits.py` via `python local_test_app.py comparison_credits [--coding-harness opencode --coding-harness-model …]`, swept in `scripts/run_tests.sh` Phase 2. Production `IterationArchitect` and production registry — nothing subclassed; only `run_eval_tool._run_eval` is patched. One credit, fixed code `73`, and the instructions never mention credits or codes: the only way the model learns the mechanism is the tool description plus the refusal. Passes when the refusal was returned at least once, the comparison eval executed exactly once, `comparison_ack_given` is set, the balance went 1 → 0, and the real close tool accepted the close. 3 attempts (model-driven). This is the one claim no offline test can make.
