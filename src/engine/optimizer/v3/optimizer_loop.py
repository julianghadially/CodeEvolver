"""Copyright © 2026 440 Labs LLC

Shared engine 3 optimizer driver.

`run_optimization` owns a full run end to end and is parameterized by an
`OptimizerStrategy` (see `v3.strategy`). Engine 3 implements EE
(agent-managed git + `close_iteration`), pareto_evolution (loop-owned
population), freeform (research Baseline 1: HEAD-registering close, no
explore guidance) and asa on this one driver. The simple_loop research
control has its own loop under `optimizer/simple_loop/`.

This module owns end to end:
  - dataset resolution (via `evaluator/dataset_config.py` — trainset, the
    fixed comparison set drawn from it with seed 42, and the optional
    held-out test set)
  - seed step: branch, codeevolver.md, memory.md, environment sanity eval,
    `CEState` construction, `IterationHistory[0]` pre-population
  - the architect loop itself (agent-owned branch choice, deterministic
    closing: parent derivation, finalize, constraint gate with
    force-revert, `CEState.record_iteration`)
  - finalization: seed + last-declared-primary scored on the held-out test
    set (uncounted against the eval budget)
  - final result dict returned to the launcher

The mechanics live in sibling modules:
  - `close_effects`   — git side-effects of closing (fallback/head close,
                        force-revert, codeevolver.md commit, anchors, …)
  - `analysis_agent`  — the per-iteration change-summary agent
  - `run_reporting`   — progress payloads + history stamping + callbacks
  - `finalization`    — post-loop scoring + post-run meta/integrity agents

Main loop (one iteration per turn):
  1. Snapshot iteration-start state (branch, sha, primary, memory.md hash)
     onto a locked `IterationContext`; arm the explore-frequency guard.
  2. Call `architect.run()` — blocks for minutes to hours. The architect
     session runs through the selected coding harness and ends with the
     `close_iteration` tool (watchdogs enforce the closing protocol).
  3. Deterministic close: fallback-close if needed, stamp `close_sha` +
     `close_tree`, derive the parent from past CLOSE shas, `context.finalize()`
     (the iteration ends on close_sha; its score is the eval whose tree is
     program-equivalent to close_tree), constraint gate (force-revert on
     violation).
  4. Run the analysis agent (per-iteration change summary — for every
     iteration that closed, landed or not) and the codeevolver.md check.
     Both can commit, so they run BEFORE the record is built and `close_sha`
     is re-resolved afterwards.
  5. Build + record the closing record on `CEState` (the close_iteration
     registration is the source of truth for the primary), anchor the end
     state under `refs/codeevolver/`, push the registered branch, report
     progress.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Callable, Any

# Soft-timeout buffer: stop the architect loop this many seconds before the
# hard job timeout so the finalization pass (held-out test evals) + status
# callbacks have time to run before the environment kills the process.
#
# Sized to the WORK finalization actually has to do, not to a round number:
# `score_seed_and_final` runs the seed on the held-out set plus a comparison
# eval, then the final primary `test_samples` times, and the post-run
# meta-reflection and run-integrity agents follow. On ARC-AGI-2 that measures
# ~8h (each 120-row held-out draw is ~1.7h against an evolved program); a
# shorter buffer leaves a deadline-triggered run unable to finish reporting
# (the callback JWT expires mid-finalization and the job never reaches a
# terminal status).
TIMEOUT_BUFFER_SECONDS = 36000  # 10 hours

# Circuit breaker for systemic architect failures (e.g. sandbox-level
# syscall unavailability, selected harness missing, auth misconfig).
# If this many consecutive architect iterations raise an exception with
# no successful iteration in between, bail out of the loop rather than
# burn the remaining eval/time budget on failures that will not
# self-heal. Reset on any iteration that exits the architect try/except
# cleanly (including skipped iterations — those are normal outcomes, not
# architect failures).
MAX_CONSECUTIVE_ARCHITECT_FAILURES = 5

# Safety iteration cap — the primary end conditions are `max_eval_rows` and
# the soft timeout. This exists only so a misconfigured run that sets neither
# still has an upper bound.
DEFAULT_MAX_ITERATIONS = 200

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DatasetConfig

from src.engine.agent.token_usage import ModelTokenLedger, TokenUsage
from src.engine.optimizer.v3.optimization_config import OptimizationConfig
from src.engine.schemas.iteration_history import IterationHistory
from src.engine.services.subprocess_service import (
    git_commit_trees,
    git_get_current_branch,
    git_get_sha,
    git_is_ancestor,
    git_list_branches,
    git_merge_base,
    git_push_authenticated,
)
from .analysis_agent import run_analysis_agent
from .ce_state import CEState
from .close_effects import (
    anchor_eval_trees,
    anchor_iteration_state,
    checkout_registered_branch,
    cleanup_dirty_workspace_if_needed,
    commit_codeevolver_md_if_modified,
    fallback_close,
    force_revert_to_parent,
    head_close,
    tree_of,
)
from .closing import build_close_record, derive_iteration_parent
from .coding_harness_config import (
    CodingHarnessConfig,
    normalize_coding_harness_config,
)
from .constraint_gate import evaluate_change_constraints
from .finalization import (
    run_post_run_integrity,
    run_post_run_meta_reflection,
    score_seed_and_final,
)
from .iteration_architect import (
    ArchitectConfig,
    IterationArchitect,
    hash_file,
    memory_md_path,
)
from .iteration_context import IterationContext
from .lm_config import build_constraint_lm_callable, resolve_agent_model_for_harness
from .program_paths import diff_is_non_program_only
from .run_metrics import derive_run_metrics, explore_guard_should_fire
from .run_reporting import (
    build_ce_state_payload,
    final_candidate,
    log_iteration_close,
    stamp_and_report_iteration,
    stamp_close_record_onto_history,
)
from .seed import run_seed_step
from .select_parent import utc_run_timestamp
from .strategy import LoopServices, OptimizerStrategy

logger = logging.getLogger(__name__)


@dataclass
class OptimizerHooks:
    """Callables the loop invokes at specific points.

    Factoring these out lets tests stand up a minimal loop without the
    full environment bring-up.
    """

    run_analysis_agent: Callable[[IterationContext], None] = lambda ctx: None
    on_iteration_recorded: Callable[[int, IterationHistory], None] = lambda i, h: None
    budget_exhausted: Callable[[int], bool] = lambda i: i >= DEFAULT_MAX_ITERATIONS



def _render_architect_markdown(iteration_idx: int, context: Any, **header: Any) -> None:
    """Best-effort `digest.md` / `transcript.md` for the iteration's architect
    trace (see `architect_trace_markdown`). Never raises."""
    try:
        from src.engine.optimizer.v3.architect_trace_markdown import (
            build_trace_header,
            render_architect_trace_markdown,
        )

        render_architect_trace_markdown(
            iteration_idx, build_trace_header(context, **header)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "iteration %d: architect trace markdown render failed (non-fatal): %s",
            iteration_idx, exc,
        )

def run_optimization(
    *,
    config: OptimizationConfig,
    strategy: OptimizerStrategy,
) -> dict:
    """Run a full optimization with the given `config` and `strategy`.

    `config` is the single `OptimizationConfig` input (the config-JSON
    contract). Its fields are unpacked to locals below so the run body reads
    naturally; `config` itself is threaded into `_run_optimizer_loop`.
    No `program_lm`: the LM that runs the client's program is configured
    inside the client repo. CodeEvolver only owns `reflection_lm` (the
    agent/summary model).

    Environment-agnostic — assumes the workspace is already set up with
    the cloned repo, deps installed, and API keys in env vars. No database
    imports.
    """
    workspace = config.workspace
    program = config.program
    metric = config.metric
    reflection_lm = config.reflection_lm
    debug_max_iterations = config.debug_max_iterations
    max_eval_rows = config.max_metric_calls
    job_timeout_seconds = config.job_timeout_seconds
    comparison_eval_timeout = config.comparison_eval_timeout
    additional_deny_paths = config.additional_deny_paths or []
    job = config.job

    coding_harness_config = normalize_coding_harness_config(
        coding_harness=config.coding_harness,
        coding_harness_model=config.coding_harness_model,
    )

    resolved_metric_names = config.metric_names or ["accuracy"]
    resolved_primary_metric = config.primary_metric or resolved_metric_names[0]

    # In Claude mode, reflection_lm also selects the Claude Agent SDK model.
    # In OpenCode mode, reflection_lm is only the seed-step LiteLLM model; the
    # harness model comes from coding_harness_model.
    agent_model = resolve_agent_model_for_harness(
        reflection_lm=reflection_lm,
        coding_harness=coding_harness_config.coding_harness,
    )

    logger.info(
        "run config: coding_harness=%s coding_harness_model=%s agent_model=%s "
        "reflection_lm=%s method=%s",
        coding_harness_config.coding_harness,
        coding_harness_config.coding_harness_model,
        agent_model,
        reflection_lm,
        strategy.method,
    )

    callbacks = job.make_callbacks()
    callbacks.set_running()

    run_start_time = time.time()
    if job_timeout_seconds is not None:
        soft_deadline_seconds = max(
            job_timeout_seconds - TIMEOUT_BUFFER_SECONDS, 0
        )
    else:
        soft_deadline_seconds = None

    # ----- Dataset resolution -----
    datasets = DatasetConfig.prepare(
        workspace,
        trainset_json=config.trainset_json,
        trainset_path=config.trainset_path,
        comparison_set_size=config.comparison_set_size,
        testset_json=config.testset_json,
        testset_path=config.testset_path,
    )

    evaluator_config = EvaluatorConfig(
        program_factory=program,
        metric_fns=[metric],
        metric_names=resolved_metric_names[:1],  # single-metric bridge
        primary_metric=resolved_primary_metric,
        num_threads=config.num_threads,
        trainset_path=datasets.trainset_rel_path,
        testset_path=datasets.testset_rel_path or None,
        input_keys=config.input_keys,
    )

    # Run-level token ledger (by model).
    run_ledger = ModelTokenLedger()

    # ----- Seed step -----
    state, history, seed_meta = run_seed_step(
        workspace=workspace,
        datasets=datasets,
        evaluator_config=evaluator_config,
        reflection_lm=reflection_lm,
        program=program,
        metric=metric,
        job=job,
        method=strategy.method,
        agent_model=agent_model,
        coding_harness=coding_harness_config.coding_harness,
        coding_harness_model=coding_harness_config.coding_harness_model,
        additional_deny_paths=additional_deny_paths,
        enable_clean_slate_inspector=config.enable_clean_slate_inspector,
    )

    run_ledger.add(
        seed_meta.get("codeevolver_md_model"),
        TokenUsage.from_dict(seed_meta.get("codeevolver_md_usage")),
    )

    # ----- Strategy services + seed hook -----
    # One rng for the whole run (seed hook + loop) so a strategy's draws are
    # reproducible from `config.seed`.
    rng = random.Random(config.seed)
    run_timestamp = utc_run_timestamp()
    services = LoopServices(
        workspace=workspace,
        evaluator_config=evaluator_config,
        trainset_ids=list(datasets.trainset_ids),
        comparison_ids=list(datasets.comparison_ids),
        comparison_timeout=comparison_eval_timeout,
        run_timestamp=run_timestamp,
        rng=rng,
        max_eval_rows=max_eval_rows,
        config=config,
    )
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): the initial
    # grant lands BEFORE the seed hook, so an arm whose seed hook runs a
    # comparison eval (pareto_evolution, asa) spends this credit on it.
    if config.comparison_credits_per_iteration is not None and datasets.comparison_ids:
        state.grant_comparison_credits(config.comparison_credits_initial)
    if strategy.on_seed_complete is not None:
        try:
            strategy.on_seed_complete(state, history, services)
        except Exception as exc:  # noqa: BLE001
            logger.exception("strategy.on_seed_complete failed")
            callbacks.set_failed(f"{type(exc).__name__}: {exc}")
            raise

    # ----- Hooks -----
    def _run_analysis_agent(ctx: IterationContext) -> None:
        run_analysis_agent(
            ctx,
            workspace=workspace,
            evaluator_config=evaluator_config,
            subsample_size=config.subsample_size,
            additional_deny_paths=additional_deny_paths,
            agent_model=agent_model,
            coding_harness_config=coding_harness_config,
        )

    def _on_iteration_recorded(iteration_idx: int, hist: IterationHistory) -> None:
        stamp_and_report_iteration(
            iteration_idx,
            hist,
            state=state,
            workspace=workspace,
            callbacks=callbacks,
            run_start_time=run_start_time,
            soft_deadline_seconds=soft_deadline_seconds,
            max_eval_rows=max_eval_rows,
        )

    def _budget_exhausted(i: int) -> bool:
        if debug_max_iterations is not None:
            if i >= debug_max_iterations + 1:  # +1 for the seed at index 0
                return True
        if callbacks.check_cancelled():
            logger.info("optimizer loop: job cancelled via /check-cancelled")
            return True
        total_evals = int(state.total_num_evals or 0)
        if max_eval_rows is not None and total_evals >= max_eval_rows:
            logger.info(
                "optimizer loop: eval-row budget exhausted (%d/%d) — ending "
                "before iteration %d",
                total_evals, max_eval_rows, i,
            )
            return True
        if soft_deadline_seconds is not None:
            elapsed = time.time() - run_start_time
            if elapsed >= soft_deadline_seconds:
                logger.info(
                    "optimizer loop: soft timeout reached (elapsed=%.0fs, "
                    "budget=%ds, buffer=%ds before hard timeout) — ending "
                    "before iteration %d",
                    elapsed, soft_deadline_seconds, TIMEOUT_BUFFER_SECONDS, i,
                )
                return True
        return i >= DEFAULT_MAX_ITERATIONS

    hooks = OptimizerHooks(
        run_analysis_agent=_run_analysis_agent,
        on_iteration_recorded=_on_iteration_recorded,
        budget_exhausted=_budget_exhausted,
    )

    # ----- Run the architect loop -----
    try:
        history = _run_optimizer_loop(
            config=config,
            strategy=strategy,
            state=state,
            workspace=workspace,
            evaluator_config=evaluator_config,
            trainset_ids=datasets.trainset_ids,
            comparison_ids=datasets.comparison_ids,
            hooks=hooks,
            history=history,
            start_iteration=1,
            agent_model=agent_model,
            coding_harness_config=coding_harness_config,
            run_ledger=run_ledger,
            services=services,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("optimizer loop raised — reporting failure to API")
        callbacks.set_failed(f"{type(exc).__name__}: {exc}")
        raise

    # ----- Finalization: seed + final primary on test + comparison sets -----
    # The final program is the LAST DECLARED PRIMARY (close_iteration is the
    # source of truth; the agent chooses). None of these evals count against
    # the eval budget (spec §24). See `finalization.score_seed_and_final`.
    scores = score_seed_and_final(
        workspace=workspace,
        datasets=datasets,
        evaluator_config=evaluator_config,
        state=state,
        history=history,
        seed_branch=seed_meta["seed_branch"],
        seed_sha=seed_meta["seed_sha"],
        comparison_eval_timeout=comparison_eval_timeout,
        test_samples=config.test_samples,
        seed_test_samples=config.seed_test_samples,
    )
    candidate = final_candidate(state)

    result = {
        "success": True,
        "method": strategy.method,
        "seed_branch": seed_meta["seed_branch"],
        "seed_sha": seed_meta["seed_sha"],
        "iterations": len(history),
        "history": dict(history),
        "primary_branch": state.primary_branch,
        "final_branch": scores.final_branch,
        "final_sha": scores.final_sha,
        "seed_train_score": scores.seed_train,
        "final_train_score": scores.final_train,
        "seed_test_score": scores.seed_test,
        "final_test_score": scores.final_test,
        "seed_test_scores": scores.seed_test_scores,
        "final_test_scores": scores.final_test_scores,
        "run_metrics": derive_run_metrics(state.iterations),
        "total_metric_calls": int(state.total_num_evals or 0),
        "finalization_error": scores.finalization_error,
    }

    callbacks.set_completed(
        final_candidate=candidate,
        seed_train_score=scores.seed_train,
        final_train_score=scores.final_train,
        seed_test_score=scores.seed_test,
        final_test_score=scores.final_test,
        seed_test_scores=scores.seed_test_scores,
        finalization_error=scores.finalization_error,
        final_test_scores=scores.final_test_scores,
        total_metric_calls=int(state.total_num_evals or 0),
        num_branches=len(state.branches),
        ce_state=build_ce_state_payload(
            state, history, num_iterations=len(history),
        ),
    )

    # Post-run agents (opt-in, advisory only). Both run AFTER the result is
    # built and MongoDB completion has been posted, so they can never affect
    # run success.
    if config.enable_meta_reflection:
        run_post_run_meta_reflection(
            workspace=workspace,
            method=strategy.method,
            history=history,
            scores=scores,
            agent_model=agent_model,
            coding_harness_config=coding_harness_config,
        )

    if config.enable_run_integrity_agent:
        stamped = run_post_run_integrity(
            workspace=workspace,
            method=strategy.method,
            history=history,
            scores=scores,
            testset_path=evaluator_config.testset_path,
            agent_model=agent_model,
            coding_harness_config=coding_harness_config,
        )
        if stamped:
            # The verdict landed on the latest record — re-flush it. Advisory
            # only: a failure here must never fail an already-completed run.
            result["history"] = dict(history)
            try:
                callbacks.update_progress(
                    current_iteration=len(history),
                    final_candidate=candidate,
                    num_branches=len(state.branches),
                    total_metric_calls=int(state.total_num_evals or 0),
                    ce_state=build_ce_state_payload(
                        state, history, num_iterations=len(history),
                    ),
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("run-integrity re-flush failed (non-fatal): %s", exc)

    # ----- Run-level token accounting (count + log) -----
    run_ledger.log_summary(logger, "run total")
    result["token_usage_by_model"] = run_ledger.to_dict()

    return result


def _run_optimizer_loop(
    *,
    config: OptimizationConfig,
    strategy: OptimizerStrategy,
    state: CEState,
    workspace: str,
    evaluator_config: EvaluatorConfig,
    trainset_ids: list[int],
    comparison_ids: list[int],
    hooks: OptimizerHooks,
    history: IterationHistory | None = None,
    start_iteration: int = 0,
    agent_model: str | None = None,
    coding_harness_config: CodingHarnessConfig | None = None,
    run_ledger: ModelTokenLedger | None = None,
    services: LoopServices | None = None,
) -> IterationHistory:
    """Run the engine 3 architect loop until the budget is exhausted.

    Each iteration: lock a fresh `IterationContext` with the iteration-start
    snapshot, run the architect (agent-owned git; ends via `close_iteration`
    + watchdogs), then perform the deterministic close: fallback-close if
    needed, derive the parent from past final shas, `finalize()`, run the
    constraint gate (force-revert on violation), and record the closing
    record on `CEState` — which applies the registered primary transition.

    The seed step is expected to have already constructed `state` and
    pre-populated `history[0]` with the seed record. Callers pass
    `start_iteration=1` so the first architect iteration lands at index 1.
    """
    additional_instructions = config.additional_instructions
    program_path = config.program
    max_iterations = config.debug_max_iterations
    seed = config.seed
    max_score = config.max_score
    subsample_size = config.subsample_size
    subsample_eval_timeout = config.subsample_eval_timeout
    comparison_eval_timeout = config.comparison_eval_timeout
    max_eval_rows = config.max_metric_calls
    # Architect trace capture serves BOTH post-run agents; keep traces on when
    # either is enabled (the ArchitectConfig field keeps its historical name).
    capture_architect_traces = (
        config.enable_meta_reflection or config.enable_run_integrity_agent
    )
    additional_deny_paths = config.additional_deny_paths or []
    reflection_lm = config.reflection_lm
    job = config.job

    if services is not None:
        # Shared with the seed hook (one rng per run, one branch timestamp).
        rng = services.rng
        run_timestamp = services.run_timestamp
    else:
        rng = random.Random(seed)
        run_timestamp = utc_run_timestamp()
    if coding_harness_config is None:
        coding_harness_config = normalize_coding_harness_config()
    if history is None:
        history = IterationHistory()
    iteration_idx = start_iteration
    consecutive_architect_failures = 0

    if run_ledger is None:
        run_ledger = ModelTokenLedger()

    def _account_iteration_tokens(ctx: IterationContext) -> None:
        by_model = ctx.token_usage_by_model
        if not by_model:
            return
        parts = ", ".join(
            f"{model}: total={usage.get('total', 0)}"
            for model, usage in sorted(by_model.items())
        )
        logger.info(
            "iteration %d token usage by model: %s", ctx.iteration_idx, parts
        )
        run_ledger.merge(by_model)

    def _program_equivalent(a: str | None, b: str | None) -> bool:
        return diff_is_non_program_only(workspace, a, b)

    prev_context: IterationContext | None = None

    # Post-change constraint gate (opt-in by `additional_instructions`).
    constraint_lm_callable = None
    if additional_instructions and additional_instructions.strip() and reflection_lm:
        constraint_lm_callable = build_constraint_lm_callable(reflection_lm)

    max_iter_idx = (
        start_iteration + max_iterations if max_iterations is not None else None
    )
    logger.info(
        "optimizer loop: starting (start_iteration=%d, max_iterations=%s, "
        "seed=%d, max_score=%s)",
        start_iteration,
        max_iterations,
        seed,
        max_score,
    )
    while True:
        if prev_context is not None:
            _account_iteration_tokens(prev_context)
            prev_context = None

        if max_iter_idx is not None and iteration_idx >= max_iter_idx:
            logger.info(
                "optimizer loop: reached max_iterations=%d, exiting", max_iterations
            )
            break
        if hooks.budget_exhausted(iteration_idx):
            logger.info(
                "optimizer loop: budget exhausted at iteration %d", iteration_idx
            )
            break

        history.create_entry(iteration_idx)
        logger.info("iteration %d: starting", iteration_idx)

        # Safety net: commit + push drift the previous iteration left behind.
        cleanup_dirty_workspace_if_needed(
            workspace=workspace,
            iteration_idx=iteration_idx,
            job=job,
        )

        # Explore-frequency guard (replaces the hereditary-distance quota).
        # Can only fire after >= 4 closed (non-skipped) iterations — early
        # iterations are never forced to explore.
        must_explore = strategy.enable_explore_guard and explore_guard_should_fire(
            state.iterations
        )
        if must_explore:
            logger.info(
                "iteration %d: explore-frequency guard fired — this iteration "
                "must explore (claimed explore fraction below quota)",
                iteration_idx,
            )

        # The context is LOCKED at construction. EE: the architect owns
        # branch choice, so there is no select_candidate step.
        # `strategy.on_iteration_start` runs BEFORE the start snapshot:
        # pareto_evolution checks out the chosen candidate and forks the child
        # branch there (so start_branch/start_sha describe the forked branch);
        # ASA computes the eligible lineage set and the fired requirements.
        # Comparison-credit ledger: +N at every iteration start, and a fresh
        # 2-digit acknowledgement code for this iteration's comparison runs
        # (specs/archive-public/comparison_credits.md). An empty code disables the gate.
        credits_enabled = (
            config.comparison_credits_per_iteration is not None and bool(comparison_ids)
        )
        if credits_enabled:
            state.grant_comparison_credits(config.comparison_credits_per_iteration)
        context = IterationContext(
            iteration_idx=iteration_idx,
            subsample_size=subsample_size,
            comparison_set_size=len(comparison_ids),
            max_eval_rows=max_eval_rows,
            evals_used=int(state.total_num_evals or 0),
            primary_branch_at_start=state.primary_branch,
            memory_md_hash_at_start=hash_file(memory_md_path()),
            must_explore=must_explore,
            parent_locked=True,
            comparison_ack_code=f"{rng.randint(10, 99)}" if credits_enabled else "",
            # Run-variance estimate (specs/archive-public/sampling_variance.md): the run's
            # one reading, rendered into every later prompt; None until taken.
            variance_estimate=getattr(state, "variance_estimate", None),
            variance_estimate_samples=config.variance_estimate_samples,
            # Kickoff note (all arms): the previous iteration was reverted by
            # the constraint gate. The fact only.
            previous_iteration_constraint_reverted=bool(
                (history.get(iteration_idx - 1) or {}).get("constraint_violation")
            ),
        )
        prev_context = context
        if strategy.on_iteration_start is not None:
            if services is None:
                raise RuntimeError(
                    f"strategy {strategy.method!r} has on_iteration_start but the "
                    "loop was started without LoopServices"
                )
            strategy.on_iteration_start(state, context, services)

        # Iteration-start snapshot.
        try:
            start_branch = git_get_current_branch(workspace)
        except Exception:  # noqa: BLE001
            start_branch = None
        try:
            start_sha = git_get_sha(workspace, "HEAD")
        except Exception:  # noqa: BLE001
            start_sha = None
        context.start_branch = start_branch
        context.start_sha = start_sha
        architect_config = ArchitectConfig(
            run_timestamp=run_timestamp,
            additional_instructions=additional_instructions,
            program_path=program_path,
            evaluator=evaluator_config,
            trainset_ids=trainset_ids,
            comparison_ids=comparison_ids,
            subsample_timeout=subsample_eval_timeout,
            comparison_timeout=comparison_eval_timeout,
            max_score=max_score,
            subsample_size=subsample_size,
            variance_estimate_samples=config.variance_estimate_samples,
            coding_harness=coding_harness_config.coding_harness,
            coding_harness_model=coding_harness_config.coding_harness_model,
            agent_managed_git=True,
            skills_to_stage=strategy.skills_to_stage,
            skill_overrides=strategy.skill_overrides,
            skill_template_values=(
                dict(strategy.skill_template_values(context))
                if strategy.skill_template_values is not None
                else None
            ),
            close_iteration_mode=strategy.close_iteration_mode,
            run_eval_gate=strategy.run_eval_gate,
            extra_tools=(
                (lambda c, s, w, _b=strategy.extra_tools, _sv=services: _b(c, s, w, _sv))
                if strategy.extra_tools is not None
                else None
            ),
            commit_before_comparison=strategy.commit_before_comparison,
            comparison_ack_note=strategy.comparison_ack_note,
            comparison_closes_iteration=strategy.comparison_closes_iteration,
            system_prompt_builder=strategy.prompt_builder,
            agent_model=agent_model,
            capture_architect_traces=capture_architect_traces,
            additional_deny_paths=additional_deny_paths,
        )

        try:
            architect = IterationArchitect(
                context=context,
                state=state,
                workspace=workspace,
                config=architect_config,
                rng=rng,
            )
            architect.run()

            # ----- Deterministic close -----
            if strategy.close_iteration_mode == "head":
                head_close(context, workspace=workspace, state=state)
            elif not context.close_called:
                fallback_close(context, workspace=workspace, state=state)

            try:
                context.close_sha = git_get_sha(workspace, context.child_branch)
            except Exception:  # noqa: BLE001
                context.close_sha = start_sha
            context.close_tree = tree_of(workspace, context.close_sha)
            # Every codeevolver-* branch that exists right now, registered or
            # not. The architect creates experiment branches mid-iteration and
            # abandons the ones that regress; those never appear in
            # `registered_branch`, so without this sweep the run record has
            # no trace that they existed.
            context.branches_seen = git_list_branches(workspace)
            # Final reconciliation pass: the fallback-close path and any
            # watchdog resume can produce commits after the close tool ran.
            # Idempotent. finalize() no longer depends on it for correctness
            # (it keys on trees), but the untested-HEAD and revert-check
            # watchdogs and the eval audit trail still read record shas.
            try:
                context.reconcile_eval_shas(
                    git_commit_trees(
                        workspace,
                        context.child_branch or "HEAD",
                        since=context.start_sha,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "iteration %d: close-time eval-sha reconciliation failed "
                    "(non-fatal): %s",
                    iteration_idx, exc,
                )

            if strategy.loop_owns_parent:
                # The loop picked the parent at iteration start
                # (`on_iteration_start` stamped parent_sha / branch / tree /
                # iteration / is_primary); deriving it from git ancestry
                # would only overwrite a known fact.
                context.parent_derivation = "preselected"
                context.new_lineage = True
            else:
                derived = derive_iteration_parent(
                    close_sha=context.close_sha or (start_sha or ""),
                    past_iterations=state.iterations,
                    seed_sha=state.seed_sha,
                    seed_branch=state.seed_branch,
                    is_ancestor=lambda a, d: git_is_ancestor(workspace, a, d),
                    merge_base=lambda a, b: git_merge_base(workspace, a, b),
                )
                context.parent_sha = derived.parent_sha
                context.parent_tree = tree_of(workspace, derived.parent_sha)
                context.parent_branch = derived.parent_branch
                context.parent_iteration = derived.parent_iteration
                context.parent_derivation = derived.via
                context.parent_is_primary = (
                    derived.parent_branch == context.primary_branch_at_start
                )
                context.new_lineage = (
                    context.child_branch is not None
                    and context.child_branch not in state.known_branches()
                )

            if context.watchdog_fired:
                history[iteration_idx]["watchdog_fired"] = True

            # Finalize: the iteration ended on close_sha; attribute a score to
            # that content and decide whether the program actually moved.
            context.finalize(_program_equivalent)
            # An iteration that ran to completion is not skipped, even if it
            # tested a change and reverted. Only the architect-exception path
            # below and a session that never registered a branch are.
            context.skipped = context.child_branch is None

        except Exception as exc:
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "iteration %d: architect failed: %s", iteration_idx, error_msg
            )
            history[iteration_idx].update(
                {"subsample_error": error_msg, "skipped": True, "changes_landed": False}
            )
            # The session died, but it may have spent budget on evals first —
            # a harness timeout after two 20-row runs is the realistic case.
            # Anchor those trees before moving on or they are referenced by
            # nothing and `git gc` prunes them, leaving no diffable trace of
            # what the iteration tried. The iteration ref is a no-op here
            # (there is no close_sha).
            anchor_iteration_state(context, workspace=workspace)
            # The only genuinely skipped shape: the architect raised, so there
            # is no branch, no close sha and nothing to attribute. Carries no
            # `close_sha`, which is exactly what excludes it from parent
            # derivation and from run_metrics.
            state.record_iteration(
                {
                    "iteration": iteration_idx,
                    "skipped": True,
                    "changes_landed": False,
                    "error": error_msg,
                }
            )
            hooks.on_iteration_recorded(iteration_idx, history)
            # The session may have run for hours before raising; its trace
            # deserves a digest with the error in the header.
            _render_architect_markdown(
                iteration_idx, context,
                harness=coding_harness_config.coding_harness,
                error=error_msg, skipped=True,
            )
            consecutive_architect_failures += 1
            if consecutive_architect_failures >= MAX_CONSECUTIVE_ARCHITECT_FAILURES:
                raise RuntimeError(
                    f"optimizer loop halting: {consecutive_architect_failures} "
                    f"consecutive architect iterations failed with no successful "
                    f"iteration in between. Last error at iteration "
                    f"{iteration_idx}: {error_msg}"
                ) from exc
            iteration_idx += 1
            continue

        # Architect ran to completion — reset the systemic-failure counter.
        consecutive_architect_failures = 0

        # ----- Constraint gate (force-revert on violation) -----
        constraint_violation_reason: str | None = None
        constraint_usage_before = (
            constraint_lm_callable.token_ledger.snapshot()
            if constraint_lm_callable is not None
            else {}
        )
        violation = evaluate_change_constraints(
            workspace=workspace,
            parent_sha=context.parent_sha,
            final_sha=context.close_sha,
            additional_instructions=additional_instructions,
            lm_callable=constraint_lm_callable,
            change_requests=context.change_requests,
            outcome_reflection=None,
        )
        if constraint_lm_callable is not None:
            for model, usage in constraint_lm_callable.token_ledger.delta_since(
                constraint_usage_before
            ).items():
                context.record_token_usage(model, usage)
        if violation is not None and violation.status == "fail":
            constraint_violation_reason = (
                violation.violated_constraint or violation.reason
            )
            logger.warning(
                "iteration %d: constraint gate fired (%s) — force-reverting "
                "%s to parent %s",
                iteration_idx,
                constraint_violation_reason,
                context.child_branch,
                (context.parent_sha or "?")[:8],
            )
            force_revert_to_parent(
                context, workspace=workspace,
                # The marker commit quotes the constraint, never the reason.
                violated_constraint=violation.violated_constraint,
            )
            history[iteration_idx]["constraint_violation"] = (
                constraint_violation_reason
            )

        # Anchor the evaluated trees now so the close record's
        # `subsample_best_sha` / `comparison_best_sha` can point at content
        # that was never committed. Idempotent; the iteration ref is written
        # at the end once close_sha is final.
        anchor_eval_trees(context, workspace=workspace)

        # ----- Close record + fixed-arm close hook, BEFORE the analysis agent -----
        #
        # pareto_evolution's hook maps the CLOSING content to a comparison
        # eval by tree (or re-runs it). It must see the tree the architect
        # left, not one the analysis agent's codeevolver.md commit has moved
        # past — even though that commit is program-equivalent by
        # construction, mapping first keeps "what the loop measured" and
        # "what the architect closed on" the same object.
        close_record = build_close_record(context, state)
        if constraint_violation_reason is not None:
            close_record["constraint_violation"] = constraint_violation_reason
            # A violating change never moves the primary.
            close_record["primary_branch_after"] = context.primary_branch_at_start
        if strategy.on_iteration_closed is not None and not context.skipped:
            # Fixed admission / fitness / promotion rules (pareto_evolution).
            # Mutates the record in place — may run a counted comparison eval
            # and always sets `primary_branch_after` to the population's best.
            # A failure here is a loop bug, not an architect failure — let it
            # propagate rather than record a half-applied transition.
            if services is None:
                raise RuntimeError(
                    f"strategy {strategy.method!r} has on_iteration_closed but "
                    "the loop was started without LoopServices"
                )
            strategy.on_iteration_closed(context, state, close_record, services)

        # ----- Change summary + codeevolver.md -----
        #
        # The analysis agent runs for every iteration that closed, including
        # ones that tested a change and reverted — a negative result is a
        # result, and skipping them would leave those iterations with no
        # written trace at all. On landing
        # iterations it also rewrites codeevolver.md in place; the loop
        # commits that on the REGISTERED branch, and that commit becomes the
        # tip the next iteration starts from — so it must land before the
        # record is written and before the push.
        if not context.skipped:
            checkout_registered_branch(context, workspace=workspace)
            hooks.run_analysis_agent(context)
            commit_codeevolver_md_if_modified(
                context, workspace=workspace, iteration_idx=iteration_idx,
            )
            # Re-resolve: the commit may have advanced the branch tip. The
            # new tip differs from the evaluated trees by codeevolver.md only
            # (non-program by definition), so nothing is re-mapped — the
            # record just has to name the tip that actually exists.
            if context.child_branch:
                try:
                    new_close = git_get_sha(workspace, context.child_branch)
                except Exception:  # noqa: BLE001
                    new_close = None
                if new_close and new_close != context.close_sha:
                    context.close_sha = new_close
                    context.close_tree = tree_of(workspace, new_close)
                    context.finalize(_program_equivalent)
                    context.branches_seen = git_list_branches(workspace)
                    close_record["close_sha"] = context.close_sha
                    close_record["close_tree"] = context.close_tree
                    close_record["changes_landed"] = context.changes_landed
                    close_record["branches_seen"] = context.branches_seen

        # ----- Record -----
        # Ledger balance AFTER the close hook, so loop-run debits are included.
        close_record["comparison_credits_after"] = (
            state.comparison_credits_balance if credits_enabled else None
        )
        stamp_close_record_onto_history(history[iteration_idx], context, close_record)
        state.record_iteration(close_record)

        # Anchor the iteration's end state so it stays reachable. The
        # architect may `git reset --hard` this branch in a later iteration,
        # which orphans the commit and makes it prunable by `git gc`. Analysis
        # needs to check these out long after the run.
        anchor_iteration_state(context, workspace=workspace)
        log_iteration_close(context, close_record)
        # Render digest.md / transcript.md for the architect session now that
        # every watchdog resume is done (OpenCode rewrites the trace directory
        # on each export) and the close facts exist for the header. Non-fatal.
        _render_architect_markdown(
            iteration_idx, context,
            harness=coding_harness_config.coding_harness,
            primary_branch_after=close_record.get("primary_branch_after"),
            constraint_violation=close_record.get("constraint_violation"),
        )

        # Push the registered branch to origin. Deterministic orchestrator
        # step — not an architect tool. Runs after the codeevolver.md commit
        # so it is included rather than stranded locally. Non-fatal.
        if job.repo_url and context.child_branch:
            push_result = git_push_authenticated(
                workspace=workspace,
                branch=context.child_branch,
                **job.git_creds(),
                set_upstream=True,
            )
            if not push_result["success"]:
                logger.warning(
                    "iteration %d: push of %s failed (continuing): %s",
                    iteration_idx, context.child_branch, push_result["stderr"],
                )

        hooks.on_iteration_recorded(iteration_idx, history)
        iteration_idx += 1

    total_iterations = iteration_idx - start_iteration
    closed_count = sum(
        1 for i in range(start_iteration, iteration_idx)
        if i in history and not history[i].get("skipped")
    )
    logger.info(
        "optimizer loop: finished (iterations=%d, closed=%d, total_evals=%d, "
        "primary=%s)",
        total_iterations,
        closed_count,
        int(state.total_num_evals or 0),
        state.primary_branch,
    )
    return history


__all__ = ["OptimizerHooks", "run_optimization"]
