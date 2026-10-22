"""Copyright © 2026 440 Labs LLC

`run_eval` tool for the IterationArchitect.

The LLM-facing tool schema is:

    run_eval(eval_sample: "single" | "train" | "comparison",
             filter: dict | None) -> dict

Subsample size is a run-level config value threaded through
`ArchitectConfig.subsample_size` into `RunEvalBinding.__init__`; the
architect does not choose it (and there is no agent-chosen sampling seed —
sampling seeds stay orchestrator-owned).

- `train` draws (and locks) the iteration's training subsample; later train
  calls reuse the locked rows so before/after deltas are fair.
- `comparison` evaluates the run's FIXED comparison set (drawn once from the
  trainset with seed 42). It exists for apples-to-apples branch comparison;
  every comparison eval is appended to `{history_dir}/comparison_set_history.jsonl`
  with the branch + sha actually evaluated. Counts against the eval budget.
- The held-out test set is never reachable from this tool (finalization-only,
  run by the loop).

Every response carries `eval_rows_remaining` + `budget_note` when the run has
an eval-row budget, so the architect learns what it has left from the tool
itself rather than from a system prompt written before the iteration started.

The budget is also ENFORCED here: once `total_num_evals >= max_eval_rows`,
train / single evals are refused and exactly one final `comparison` eval is
allowed. The loop only checks the budget BETWEEN iterations — without this gate
an architect that crossed the line mid-iteration could keep spending. An eval
that *overshoots* the budget is fine (it was started under budget); what is
refused is a training eval STARTED at or over the line. The one final
comparison is deliberate: it lets the architect check the branch it is about to
leave as primary, so wrap-up never has to ship a never-compared branch (and in
the research matrix it matches the comparison run the fixed-evolutionary arms
get after exhaustion). One rule everywhere — SaaS and research runs behave
identically, so the budget note can never mean two different things.

All orchestrator dependencies (sandbox, dataset ids, evaluator config,
rng, timeouts) are bound once per iteration on `RunEvalBinding` and closed
over. The binding's `__call__` is the callable the architect session
registers as the `run_eval` tool — the Python signature matches the LLM
schema one-to-one, so readers do not have to infer which arguments the
model supplies and which are orchestrator state.

There is deliberately no free-standing `run_tool(...)` helper. If you want
to drive a one-off eval from a test, construct a `RunEvalBinding` with your
fakes and call it.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Literal

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.sandbox_client import EvalResult, run_eval as _run_eval
from src.engine.schemas.agent_config import BaseAgentConfig
from src.engine.schemas.iteration_context import TRAIN_EVAL_SAMPLES, BaseIterationContext
from src.engine.services.subprocess_service import (
    git_add_and_commit,
    git_dirty_paths,
    git_get_current_branch,
    git_get_sha,
    git_is_dirty,
    git_tree_signature,
)
from src.engine.telemetry.trace_layout import write_provenance

from . import eval_budget, eval_filters
from .errors import RunEvalError
from src.engine.evaluator.mounted.evaluate import _is_retryable
from src.engine.telemetry.trace_layout import pretty_index_path

from .eval_budget import FINAL_COMPARISON_ALLOWANCE
from .eval_filters import normalize_filter as _normalize_filter, row_matches as _row_matches
from .eval_log import (
    VARIANCE_EVAL_SAMPLE,
    append_eval_log_line,
    append_row_scores_line,
    evals_log_path,
    failed_rows_from_trace_meta,
    new_eval_id,
    row_scores_from_trace_meta,
    sample_key_for_rows,
    sampling_note,
    split_for_suffix,
    variance_suffix,
)

logger = logging.getLogger(__name__)

EvalSample = Literal["single", "train", "comparison"]


AUTO_COMMIT_NOTE = (
    "Your uncommitted changes were auto-committed. Running comparison evals "
    "triggers a commit for tracking purposes."
)

# Acknowledgement note shown before a comparison run when no arm-specific
# note is configured (specs/archive-public/comparison_credits.md). Placeholders are filled
# at refusal time with live values.
STANDARD_COMPARISON_ACK_NOTE = (
    "The comparison set depletes budget at {comparison_set_size} rows per "
    "run, compared to {subsample_size} rows for a training subsample. "
    "Additionally, there are a fixed number of comparison set credits "
    "replenishing at the start of each iteration. Remaining credits: "
    "{credits}. Remaining eval-row budget: {rows_remaining}. Please "
    "acknowledge that you would like to continue running the comparison set."
)

# Refusal for every run_eval call after the iteration's comparison eval on
# arms where the comparison IS the close step (pareto_evolution).
COMPARISON_CLOSED_NOTE = (
    "run_eval refused: this iteration's comparison set has already been "
    "run, and on this optimizer the comparison run is the close step — no "
    "further evals are accepted this iteration. Load the `close_iteration` "
    "skill and close the iteration now. Any program changes made after the "
    "comparison run will be reverted to the comparison-run commit."
)


class RunEvalBinding:
    """LLM-facing `run_eval` tool.

    Constructed once per iteration by `IterationArchitect.build_tool_registry`
    (or `from_agent_config` at any call site that holds a `BaseAgentConfig`).
    Every iteration gets a fresh binding — there is no shared state between
    iterations because the `IterationContext` on `self.ctx` is itself
    per-iteration.

    Rules (enforced on each call; violations raise `RunEvalError`, which the
    architect's SDK proxy catches and reports back to the model):

    - The iteration context must be locked (`ctx.parent_locked`) — the loop
      locks it at iteration start (or a strategy's `on_iteration_start` hook
      does).
    - The eval-row budget is hard-enforced — see `_enforce_eval_budget`
      (policy in `eval_budget.py`).
    - On the first `train` call, `self.subsample_size` rows are drawn
      uniformly from `trainset_ids` (or from the filter-narrowed pool), stored
      on `ctx.subsample_ids`, and locked. Subsequent `train` calls in the
      same iteration reuse the locked ids. Filter rules live in
      `eval_filters.py`.
    - Every call lands in `ctx.eval_records` with the TREE of the code that
      actually ran — the identity `IterationContext.finalize()` matches the
      closing content against. Train/single calls additionally append
      `(current_head_sha, primary_score)` to `ctx.eval_history`; comparison
      calls also append to the
      comparison history JSONL.

    This module keeps every effectful step (the evaluator call, git reads,
    the ASA pre-comparison commit) so tests can patch `_run_eval` / `git_*`
    on this one namespace; the pure policy lives in `eval_budget`,
    `eval_filters` and `eval_log`.
    """

    # First-subsample resample budget: if the initial subsample scores at or
    # above `max_score` we redraw up to this many times looking for rows the
    # architect can still improve on, gated on the filtered pool being at
    # least 1.5× `subsample_size` (2× on attempt 3). Past the budget we fall
    # back to the full (filtered) trainset once.
    MAX_INITIAL_SUBSAMPLE_RESAMPLES = 3

    # Comparison evals allowed AFTER the eval-row budget is reached (see
    # `eval_budget.FINAL_COMPARISON_ALLOWANCE`).
    FINAL_COMPARISON_ALLOWANCE = FINAL_COMPARISON_ALLOWANCE

    def __init__(
        self,
        *,
        ctx: BaseIterationContext,
        workspace: str,
        config: EvaluatorConfig,
        trainset_ids: list[int],
        comparison_ids: list[int] | None = None,
        rng: random.Random | None = None,
        subsample_timeout: int | None = None,
        comparison_timeout: int | None = None,
        max_score: float = 1.0,
        subsample_size: int = 20,
        state: Any | None = None,
        run_eval_gate: Any | None = None,
        commit_before_comparison: bool = False,
        comparison_ack_note: str | None = None,
        comparison_closes_iteration: bool = False,
    ) -> None:
        self.ctx = ctx
        self.workspace = workspace
        self.config = config
        self.trainset_ids = list(trainset_ids)
        self.comparison_ids = list(comparison_ids or [])
        self.rng = rng if rng is not None else random.Random()
        self.subsample_timeout = subsample_timeout
        self.comparison_timeout = comparison_timeout
        self.max_score = max_score
        self.subsample_size = subsample_size
        # Optional — when provided, every eval calls state.increment_evals so
        # `state.total_num_evals` is the single source of truth for the
        # budget check in `ee_optimizer._budget_exhausted`.
        self.state = state
        # Strategy policy consulted once per iteration, at the BASELINE (the
        # first train/single eval): `(ctx, state, workspace, eval_sample) ->
        # note | None`, or raises RunEvalError to refuse before any rows are
        # spent (specs/archive-public/v3_asa.md §Mechanism). None = no gate.
        self.run_eval_gate = run_eval_gate
        # Commit a dirty tree before a comparison eval so the JSONL sha names
        # the evaluated content (specs/archive-public/v3_asa.md §Comparison Set Eval Integrity).
        self.commit_before_comparison = commit_before_comparison
        # Comparison-credit acknowledgement note (specs/archive-public/comparison_credits.md);
        # None = the standard note. The gate itself is keyed on
        # `ctx.comparison_ack_code` — empty means the ledger is disabled.
        self.comparison_ack_note = comparison_ack_note or STANDARD_COMPARISON_ACK_NOTE
        # pareto_evolution: the comparison run is the close step — once it has
        # run, every further run_eval call this iteration is refused.
        self.comparison_closes_iteration = comparison_closes_iteration
        # Lazy cache populated on the first filter-requiring call.
        self._trainset_rows_cache: list[dict[str, Any]] | None = None
        # Working-tree signature of the previous eval in this iteration, so each
        # `run_eval done` line can flag whether the evaluated code actually
        # changed since the last eval (catches duplicate evals on identical
        # code — e.g. the OpenCode double-fire — even with uncommitted drift).
        self._last_tree_sig: str | None = None
        # Set per `_run_once`: did this call's comparison eval auto-commit?
        self._last_auto_committed: bool = False

    @classmethod
    def from_agent_config(
        cls,
        *,
        ctx: BaseIterationContext,
        state: Any | None,
        workspace: str,
        config: BaseAgentConfig,
        rng: random.Random | None,
        run_eval_gate: Any | None = None,
        commit_before_comparison: bool = False,
    ) -> "RunEvalBinding":
        # comparison_ack_note / comparison_closes_iteration ride on the
        # BaseAgentConfig itself (arm strategies set them there).
        """Bind the eval-tool settings a `BaseAgentConfig` carries.

        The one construction path shared by the architect's tool registry,
        the simple loop's session and the loop-initiated comparison eval,
        so the evaluator config, dataset ids, timeouts, `max_score` and
        `subsample_size` are threaded identically everywhere.
        """
        return cls(
            ctx=ctx,
            workspace=workspace,
            config=config.evaluator,
            trainset_ids=config.trainset_ids,
            comparison_ids=config.comparison_ids,
            rng=rng,
            subsample_timeout=config.subsample_timeout,
            comparison_timeout=config.comparison_timeout,
            max_score=config.max_score,
            subsample_size=config.subsample_size,
            state=state,
            run_eval_gate=run_eval_gate,
            commit_before_comparison=commit_before_comparison,
            comparison_ack_note=getattr(config, "comparison_ack_note", None),
            comparison_closes_iteration=getattr(
                config, "comparison_closes_iteration", False
            ),
        )

    def __call__(
        self,
        eval_sample: EvalSample,
        filter: dict[str, Any] | None = None,
        acknowledge_message: str | None = None,
    ) -> dict[str, Any]:
        """The LLM-facing call. Matches the tool schema the model sees.

        Args:
            eval_sample: One of `"single"`, `"train"`, `"comparison"`.
                (The legacy alias `"subsample"` is normalized to `"train"`.)
            filter: Optional column-value filter applied on the *first*
                `train` call to narrow the sampling pool. Map column name
                to either a scalar (exact match) or `{op: value}` where op is
                one of `eq`, `neq`, `in`, `contains` (case-insensitive
                substring). Multiple columns AND together. Ignored for
                `single` / `comparison`. Locks for the rest of the iteration —
                passing a different filter on a later `train` call raises
                `RunEvalError`.

        The response dict includes `baseline_accepted: bool` on initial
        train calls. When `False`, every sampled row scored at ceiling
        (>= max_score), so improvement is impossible on this slice; the filter
        lock is cleared and the architect should reissue `run_eval` with a
        different filter — that call becomes the iteration's baseline.
        """
        ctx = self.ctx

        if eval_sample == "subsample":  # legacy alias
            eval_sample = "train"

        if not ctx.parent_locked:
            raise RunEvalError(
                "run_eval is not available yet — the iteration has not been "
                "initialized by the optimizer loop."
            )

        # pareto_evolution: the comparison run is the close step. Once one has
        # run this iteration, every further eval of any kind is refused —
        # the agent's next move is close_iteration (specs/archive-public/comparison_credits.md).
        if self.comparison_closes_iteration and any(
            rec.get("eval_sample") == "comparison" and not rec.get("by_loop")
            for rec in ctx.eval_records
        ):
            raise RunEvalError(COMPARISON_CLOSED_NOTE)

        # Raises when the budget forbids this call; the returned flag says
        # this comparison is spending the post-budget allowance, so it can be
        # marked consumed once it actually runs.
        is_final_comparison = self._enforce_eval_budget(eval_sample)

        # Comparison-credit ledger (specs/archive-public/comparison_credits.md). Keyed on
        # the per-iteration ack code — empty means the ledger is disabled for
        # this run. One contract, no exceptions: EVERY comparison eval —
        # the post-budget final one included — requires the acknowledgement
        # code, so the agent never hunts for a code that was not issued.
        # Only the credit balance check and the
        # debit stay exempt for the final comparison: wrap-up commonly has
        # zero credits left, and the final allowance is its own ledger.
        credits_enabled = bool(ctx.comparison_ack_code)
        if eval_sample == "comparison" and credits_enabled:
            balance = eval_budget.credits_balance(self.state)
            if not is_final_comparison:
                eval_budget.enforce_comparison_credits(
                    balance=balance, budget=self._budget()
                )
            if not ctx.comparison_ack_given:
                if (acknowledge_message or "").strip() == ctx.comparison_ack_code:
                    ctx.comparison_ack_given = True
                else:
                    budget_pair = self._budget()
                    rows_remaining = "unknown"
                    rows_left: int | None = None
                    if budget_pair is not None:
                        used, max_rows = budget_pair
                        rows_left = max(0, max_rows - used)
                        rows_remaining = str(rows_left)
                    note = self.comparison_ack_note.format(
                        comparison_set_size=len(self.comparison_ids),
                        subsample_size=self.subsample_size,
                        credits=balance,
                        rows_remaining=rows_remaining,
                    )
                    # Budget-endgame clarity: state the overshoot policy and
                    # the reserved final run so the agent does not abandon
                    # validating its best change over the ambiguity.
                    if is_final_comparison:
                        note += (
                            " This is the run's one reserved final "
                            "comparison-set eval; it will be allowed even "
                            "though the eval-row budget is spent."
                        )
                    elif rows_left is not None and 0 < rows_left < len(self.comparison_ids):
                        note += (
                            " An eval started under budget is allowed to "
                            "overshoot the remaining budget; it will run to "
                            "completion."
                        )
                    return eval_budget.acknowledgement_refusal(
                        note=note,
                        code=ctx.comparison_ack_code,
                        balance=balance,
                        budget=budget_pair,
                    )

        # Strategy gate at the iteration's baseline only (before anything is
        # stamped or spent). Later train evals are ungated; comparison evals
        # are never gated.
        gate_note: str | None = None
        # `single` smoke checks neither trigger nor consume the gate — the
        # gate belongs to the real baseline subsample.
        if (
            self.run_eval_gate is not None
            and eval_sample in TRAIN_EVAL_SAMPLES
            and eval_sample != "single"
            and not any(
                rec.get("eval_sample") in TRAIN_EVAL_SAMPLES
                and rec.get("eval_sample") != "single"
                for rec in ctx.eval_records
            )
        ):
            gate_note = self.run_eval_gate(ctx, self.state, self.workspace, eval_sample)

        normalized_filter = _normalize_filter(filter)

        if eval_sample != "train":
            if normalized_filter is not None:
                logger.info(
                    "iteration %d: run_eval ignoring filter on eval_sample=%s",
                    ctx.iteration_idx, eval_sample,
                )
            normalized_filter = None

        is_initial_subsample = (
            eval_sample == "train" and ctx.initial_subsample_score is None
        )

        # Record what the iteration's FIRST training eval actually measured.
        # Editing before the baseline is permitted — an agent-directed search
        # may need to repair the branch before a valid baseline is possible,
        # and the harness must not impose fixed rules on the arm defined by the
        # agent deciding (specs/research_plan.md). But when it happens the
        # baseline describes the parent PLUS those edits, so say so rather than
        # let a clean-looking `subsample_baseline` imply otherwise.
        if eval_sample == "train" and ctx.baseline_tree_dirty is None:
            ctx.baseline_dirty_paths = git_dirty_paths(self.workspace)
            ctx.baseline_tree_dirty = bool(ctx.baseline_dirty_paths)
            if ctx.baseline_tree_dirty:
                logger.info(
                    "iteration %d: baseline eval is running on a MODIFIED "
                    "working tree (%d path(s): %s) — the baseline measures the "
                    "parent plus these edits",
                    ctx.iteration_idx,
                    len(ctx.baseline_dirty_paths),
                    ", ".join(ctx.baseline_dirty_paths[:5]),
                )

        baseline_accepted: bool | None = None
        baseline_note: str | None = None

        if is_initial_subsample:
            result, suffix = self._run_initial_subsample(normalized_filter)
            if result.primary_score >= self.max_score:
                # Ceiling across every path we tried. Don't commit this as
                # the iteration's baseline — admission requires a strictly-
                # higher child score, which is impossible from here. Clear
                # filter lock + resample counter so the next call can
                # baseline against a different slice.
                locked_filter = ctx.subsample_filter
                attempts = ctx.subsample_resample_count
                ctx.subsample_ids = None
                ctx.subsample_filter = None
                ctx.subsample_resample_count = 0
                baseline_accepted = False
                if locked_filter:
                    baseline_note = (
                        f"All sampled rows scored >= max_score="
                        f"{self.max_score} after {attempts} resamples with "
                        f"filter {locked_filter!r}. This slice is already "
                        "fully solved and cannot serve as a baseline. Call "
                        "run_eval(eval_sample='train', ...) again with "
                        "a different filter (or drop the filter to baseline "
                        "against the full trainset); that call will become "
                        "the new baseline."
                    )
                else:
                    baseline_note = (
                        f"All sampled rows scored >= max_score="
                        f"{self.max_score} after {attempts} resamples across "
                        "the full trainset. Every row is already at ceiling, "
                        "so no change can produce a strictly-higher subsample "
                        "score. Either narrow the subsample with a filter "
                        "targeting a harder slice, or abandon this iteration."
                    )
                logger.info(
                    "iteration %d: initial subsample at ceiling (score=%s, "
                    "filter=%s); baseline_accepted=False, filter lock cleared",
                    ctx.iteration_idx, result.primary_score, locked_filter,
                )
            else:
                ctx.initial_subsample_score = result.primary_score
                ctx.baseline_sample_key = sample_key_for_rows(result.row_ids)
                baseline_accepted = True
        else:
            if eval_sample == "train":
                self._enforce_filter_lock(normalized_filter)
            row_ids, suffix, timeout = self._resolve_rows(eval_sample)
            result = self._run_once(row_ids, suffix, timeout, eval_sample)

        if is_final_comparison:
            # After the eval, not before: a call that never produced a score
            # must not burn the run's one remaining comparison.
            note = getattr(self.state, "note_final_comparison_eval", None)
            if callable(note):
                note()
            logger.info(
                "iteration %d: spent the run's final (post-budget) comparison eval",
                ctx.iteration_idx,
            )
        elif eval_sample == "comparison" and credits_enabled:
            # Same after-the-eval rule as the final allowance: a call that
            # produced no score spends no credit.
            debit = getattr(self.state, "debit_comparison_credit", None)
            if callable(debit):
                debit()

        head_sha = git_get_sha(self.workspace)
        # Content identity of what was scored. `sha` is HEAD, which is the
        # PARENT commit while the tree is dirty; `tree` survives the commit
        # the architect makes next, so the close checks can still match it.
        head_tree = git_tree_signature(self.workspace)

        # Summary-only response: the per-row feedback stays on disk (index.jsonl /
        # index_pretty_print.txt) instead of inflating the inline result past
        # the harness truncation limit and forcing score recovery via grep.
        meta_rows = [m for m in result.trace_meta if isinstance(m, dict)]
        # Rows that scored zero OR failed outright. Named for what it counts:
        # a wrong-but-completed prediction lands here too, so on a
        # low-scoring benchmark this is mostly "wrong answers", not breakage.
        # `infra_failed` below is the true failure count.
        zero_or_failed_count = sum(
            1 for m in meta_rows if m.get("failed") or not m.get("score")
        )
        # Rows whose failure matches the retry classifier's provider-fault
        # signatures: zeros in the score that are infrastructure, not the
        # program. Default-deny — an unrecognized error counts as a
        # real program failure.
        infra_failed = sum(
            1 for m in meta_rows
            if m.get("failed") and _is_retryable(m.get("error_summary"))
        )
        response: dict[str, Any] = {
            "ok": True,
            "eval_id": ctx.eval_records[-1].get("eval_id") if ctx.eval_records else None,
            "eval_sample": eval_sample,
            "suffix": suffix,
            "sha": head_sha,
            "tree": head_tree,
            "scores_by_metric": result.scores_by_metric,
            "primary_score": result.primary_score,
            "n_rows": len(result.trace_meta) or len(result.row_ids),
            "zero_or_failed_count": zero_or_failed_count,
            "infra_failed": infra_failed,
            "score_reliable": infra_failed == 0,
            "trace_dir": result.trace_dir,
            "trace_index_file": result.trace_index_file,
            "trace_pretty_file": (
                pretty_index_path(result.trace_dir) if result.trace_dir else ""
            ),
            "evals_log_file": evals_log_path(ctx.iteration_idx),
        }
        if eval_sample == "train" and ctx.subsample_filter is not None:
            # With a filter in effect the id list is the only per-call
            # evidence of WHICH rows ran, and it is small; unfiltered evals
            # can recover ids from the trace index on disk.
            response["row_ids"] = result.row_ids
        if baseline_accepted is not None:
            response["baseline_accepted"] = baseline_accepted
        if baseline_note is not None:
            response["baseline_note"] = baseline_note
        if gate_note:
            response["gate_note"] = gate_note
        if self._last_auto_committed:
            response["auto_committed"] = True
            response["auto_commit_note"] = AUTO_COMMIT_NOTE
        response["sampling_note"] = self._sampling_note(eval_sample, suffix, result)
        response.update(self._budget_fields())
        if eval_sample == "comparison":
            response.update(
                eval_budget.credits_fields(
                    balance=eval_budget.credits_balance(self.state),
                    enabled=credits_enabled,
                )
            )
            history_path = getattr(self.state, "comparison_history_path", None)
            if history_path:
                response["comparison_history_path"] = history_path
        return response

    # ----- eval-row budget (policy: eval_budget.py) -----

    def _budget(self) -> tuple[int, int] | None:
        """`(rows_used, max_rows)`, or None when this run has no budget."""
        return eval_budget.budget(self.ctx, self.state)

    def _final_comparisons_left(self) -> int:
        return eval_budget.final_comparisons_left(self.state, bool(self.comparison_ids))

    def _enforce_eval_budget(self, eval_sample: EvalSample) -> bool:
        """Refuse evals the run's eval-row budget no longer allows; True when
        this call spends the post-budget comparison allowance."""
        return eval_budget.enforce_eval_budget(
            eval_sample,
            budget=self._budget(),
            has_comparison_set=bool(self.comparison_ids),
            final_left=self._final_comparisons_left(),
        )

    def close_requires_train_eval(self) -> bool:
        """Should `close_iteration` check 5c demand a train baseline?

        5c (`agent_tools._validate_committed_close`) exists to keep every
        working iteration anchored to a train baseline. A baseline is only
        meaningful as one leg of a before/after PAIR, so demanding one leg when
        there is no room for the other is arbitrary — and it is precisely the
        state the engine's own guidance steers the architect into, producing a
        deadlock in which `close_iteration` refuses ("call run_eval(
        eval_sample='train')") while `run_eval` refuses that very call.

        Two waivers, both read off state this binding already owns so the gate
        and the refusal can never drift apart:

        1. No room for a pair — fewer than `2 * subsample_size` rows remain.
           This is the same threshold `prompts._budget_note` uses to tell the
           architect it has "no room for a before/after train pair", so the
           prompt and the gate now agree. Budget exhaustion needs no separate
           branch: 0 remaining is already below it. `_budget()` reads the LIVE
           `state.total_num_evals` while the note reads the start-of-iteration
           snapshot, and the asymmetry runs the safe way — live remaining at
           close time is <= the snapshot at iteration start, so any iteration
           where the note fired is guaranteed to get the waiver.
        2. The comparison latch is engaged — on `comparison_closes_iteration`
           arms (evo only) an agent-run comparison refuses every later eval
           (see `__call__`), so no train eval is reachable at any budget level.
           This is the `pareto_evolution-paperreview-s1` iteration 43 deadlock.

        Runs with no configured budget never waive on ground 1; there is always
        room for a pair.
        """
        if self.comparison_closes_iteration and any(
            rec.get("eval_sample") == "comparison" and not rec.get("by_loop")
            for rec in self.ctx.eval_records
        ):
            return False

        budget = self._budget()
        if budget is None:
            return True
        used, max_rows = budget
        return (max_rows - used) >= 2 * self.subsample_size

    def _budget_fields(self) -> dict[str, Any]:
        """`eval_rows_remaining` + `budget_note` for the tool response."""
        return eval_budget.budget_fields(
            budget=self._budget(),
            has_comparison_set=bool(self.comparison_ids),
            final_left=self._final_comparisons_left(),
        )

    def _sampling_note(
        self, eval_sample: str, suffix: str, result: EvalResult
    ) -> str:
        return sampling_note(eval_sample, suffix, len(result.row_ids))

    # ----- loop-initiated comparison eval (pareto_evolution) -----

    def run_comparison_for_loop(self) -> EvalResult:
        """Run the fixed comparison set on the current working tree for the
        OPTIMIZER LOOP (not the architect).

        Same recording path as an agent call — counted against the budget,
        logged on `ctx.eval_records` with per-row scores, appended to
        `comparison_set_history.jsonl` — but it bypasses the agent-facing
        budget refusal and never consumes the post-budget final-comparison
        allowance: this is the fixed-rule fitness measurement the
        pareto_evolution arm owes every admitted candidate, and refusing it
        would discard a verified improvement. Overshoot is counted honestly.
        """
        if not self.comparison_ids:
            raise RunEvalError(
                "no comparison set is configured for this run "
                "(comparison_set_size was unset or 0)."
            )
        return self._run_once(
            list(self.comparison_ids),
            "comparison",
            self.comparison_timeout,
            "comparison",
            by_loop=True,
        )

    # ----- loop-driven repeated runs (run_variance_estimate) -----

    def run_variance_samples(self, n_runs: int) -> tuple[list[int], list[EvalResult]]:
        """Evaluate the current tree `n_runs` times on ONE fresh random
        `subsample_size` sample (specs/archive-public/sampling_variance.md).

        Drawn from the full trainset — the iteration's filter lock and locked
        subsample are neither consulted nor touched, so the runs can never
        become the baseline. Each run is recorded as `eval_sample="variance"`
        through the usual `_run_once` path (counted, logged, per-row scores
        kept) under suffixes `variance`, `variance_2`, …. Every run is judged
        against the budget when it starts: a refusal (or evaluator failure)
        before the first run propagates; after that the completed runs are
        returned and the caller decides whether they are enough.
        """
        if self.subsample_size <= 0:
            raise RunEvalError(
                f"subsample_size must be positive, got {self.subsample_size}"
            )
        if not self.trainset_ids:
            raise RunEvalError("trainset is empty; cannot draw a variance sample")
        k = min(self.subsample_size, len(self.trainset_ids))
        row_ids = sorted(self.rng.sample(list(self.trainset_ids), k))
        logger.info(
            "iteration %d: run_variance_estimate drew %d ids for %d runs: %s",
            self.ctx.iteration_idx, k, n_runs, row_ids,
        )
        results: list[EvalResult] = []
        for run_idx in range(n_runs):
            try:
                self._enforce_eval_budget(VARIANCE_EVAL_SAMPLE)
                results.append(
                    self._run_once(
                        list(row_ids),
                        variance_suffix(run_idx),
                        self.subsample_timeout,
                        VARIANCE_EVAL_SAMPLE,
                    )
                )
            except Exception as exc:  # noqa: BLE001
                if run_idx == 0:
                    raise
                logger.warning(
                    "iteration %d: variance run %d/%d stopped (%s: %s); keeping "
                    "the %d completed run(s)",
                    self.ctx.iteration_idx, run_idx + 1, n_runs,
                    type(exc).__name__, exc, len(results),
                )
                break
        return row_ids, results

    # ----- internals -----

    def _commit_before_comparison_eval(self) -> bool:
        """Auto-commit before a comparison eval (see `_auto_commit_dirty_tree`)."""
        return self._auto_commit_dirty_tree("comparison")

    def _auto_commit_dirty_tree(self, reason: str) -> bool:
        """Commit a dirty tree on a named branch before a `reason` eval.

        Makes `sha`/`committed` in the JSONL line describe the evaluated
        content at write time (no reconciliation dependence). Skipped on a
        detached HEAD — the commit would be orphaned and `branch` None.
        Returns True when a commit was actually made, so the tool response
        can tell the architect (which otherwise finds a clean tree and
        "nothing to commit" with no explanation).
        """
        try:
            if not git_is_dirty(self.workspace):
                return False
            if git_get_current_branch(self.workspace) is None:
                logger.info(
                    "iteration %d: dirty tree on detached HEAD — not auto-committing "
                    "before the %s eval",
                    self.ctx.iteration_idx, reason,
                )
                return False
            committed = bool(git_add_and_commit(
                self.workspace,
                f"iter {self.ctx.iteration_idx}: auto-commit before {reason} eval",
            ))
            if committed:
                logger.info(
                    "iteration %d: auto-committed dirty tree before %s eval",
                    self.ctx.iteration_idx, reason,
                )
            return committed
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "iteration %d: auto-commit before %s eval failed: %s",
                self.ctx.iteration_idx, reason, exc,
            )
            return False

    def _run_once(
        self,
        row_ids: list[int],
        suffix: str,
        timeout: int | None,
        eval_sample: str,
        *,
        by_loop: bool = False,
    ) -> EvalResult:
        """Execute one eval pass and append to ctx.eval_history."""
        ctx = self.ctx
        eval_id = new_eval_id()

        logger.info(
            "iteration %d: run_eval start (sample=%s, suffix=%s, rows=%d, timeout=%s)",
            ctx.iteration_idx,
            eval_sample,
            suffix,
            len(row_ids),
            timeout,
        )

        self._last_auto_committed = False
        if eval_sample == "comparison" and self.commit_before_comparison:
            self._last_auto_committed = self._commit_before_comparison_eval()

        result: EvalResult = _run_eval(
            workspace=self.workspace,
            config=self.config,
            row_ids=row_ids,
            iteration_idx=ctx.iteration_idx,
            suffix=suffix,
            capture_traces=True,
            timeout=timeout,
        )

        # A provider glitch that cost retries is otherwise invisible: the row
        # recovers, the score is right, and nothing says the run fought for it.
        # Surfaced here rather than in the mounted evaluator, which runs in the
        # sandbox and has no logger.
        retry_summary = (result.raw or {}).get("retry_summary")
        if retry_summary:
            logger.warning(
                "iteration %d: %d row(s) hit a retryable provider error during "
                "%s eval (%d retry attempt(s); %d recovered, %d exhausted the "
                "allowance and scored 0.0). Retries are NOT charged to the "
                "eval-row budget.",
                ctx.iteration_idx, retry_summary.get("rows", 0), suffix,
                retry_summary.get("attempts", 0),
                retry_summary.get("recovered", 0),
                retry_summary.get("exhausted", 0),
            )

        # Increment global eval-row counter so the EE loop's budget check
        # (`ee_optimizer._budget_exhausted`) sees every row touched by the
        # architect. Scored rows only — trace_meta length is the authoritative
        # count because malformed rows may be dropped by the mounted evaluator.
        # Retried rows are counted ONCE: `_run_row` emits exactly one entry per
        # row however many attempts it took, so a provider fault never spends
        # the architect's budget.
        if self.state is not None:
            inc = len(result.trace_meta) or len(row_ids)
            try:
                self.state.increment_evals(inc)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "iteration %d: state.increment_evals(%d) failed: %s",
                    ctx.iteration_idx, inc, exc,
                )

        head_sha = git_get_sha(self.workspace)
        # Captured at the same instant as head_sha, after the eval returned,
        # so the pair describes one moment and the tree matches the commit
        # the architect makes immediately afterwards.
        head_tree = git_tree_signature(self.workspace)
        try:
            head_branch = git_get_current_branch(self.workspace)
        except Exception:  # noqa: BLE001
            head_branch = None
        n_rows = len(result.trace_meta) or len(row_ids)
        # Per-row scores are kept for the fixed comparison set — the one
        # sample whose rows are identical across branches, feeding the
        # pareto_evolution instance frontier — and for the variance runs,
        # whose per-row spread IS the measurement.
        row_scores = (
            row_scores_from_trace_meta(result.trace_meta)
            if eval_sample in ("comparison", VARIANCE_EVAL_SAMPLE)
            else None
        )
        # Does HEAD actually hold the scored content? A clean tree means the
        # commit's tree equals the working tree; a dirty one means the content
        # lives only in the tree object until the architect commits (at which
        # point `reconcile_eval_shas` flips this to True).
        committed: bool | None = None
        if head_sha and head_tree:
            try:
                committed = git_tree_signature(self.workspace, head_sha) == head_tree
            except Exception:  # noqa: BLE001
                committed = None
        sample_key = (
            sample_key_for_rows(result.row_ids)
            if eval_sample != "comparison"
            else None
        )
        ctx.record_eval(
            head_sha,
            result.primary_score,
            eval_sample=eval_sample,
            branch=head_branch,
            n_rows=n_rows,
            tree=head_tree,
            row_scores=row_scores,
            eval_id=eval_id,
            sample_key=sample_key,
            committed=committed,
            by_loop=by_loop,
            ts=time.time(),
        )
        # One line per eval, agent-readable — the per-iteration record of
        # which sha/tree each eval actually measured. The same payload is
        # stamped into the trace directory as `provenance.json`: the directory
        # is rewritten by every eval with the same suffix, so without the
        # stamp a later session cannot tell which tree its traces describe.
        eval_line = {
                "eval_id": eval_id,
                "ts": time.time(),
                "iteration": ctx.iteration_idx,
                "eval_sample": eval_sample,
                "suffix": suffix,
                "n_rows": n_rows,
                "sample_key": sample_key,
                "filter": ctx.subsample_filter if eval_sample == "train" else None,
                "score": result.primary_score,
                "scores_by_metric": result.scores_by_metric,
                "branch": head_branch,
                "sha_at_eval": head_sha,
                "tree": head_tree,
                "committed": committed,
                "auto_committed": self._last_auto_committed,
                "by_loop": by_loop,
                # Ledger balance at eval time, BEFORE this eval's debit
                # (comparison evals only; specs/archive-public/comparison_credits.md).
                **(
                    {"credits_balance_before": eval_budget.credits_balance(self.state)}
                    if eval_sample == "comparison"
                    else {}
                ),
        }
        append_eval_log_line(ctx.iteration_idx, eval_line)
        if result.trace_dir:
            write_provenance(result.trace_dir, {**eval_line, "trace_dir": result.trace_dir})
        # Durable per-row record of EVERY eval (train/comparison/single/
        # variance) — `{history_dir}/row_scores.jsonl`, collected off the
        # Docker runner unlike traces. Distinct from `row_scores` above, which is
        # kept on the EvalRecord for comparison/variance evals only.
        append_row_scores_line(
            {
                "eval_id": eval_id,
                "ts": time.time(),
                "iteration": ctx.iteration_idx,
                "eval_sample": eval_sample,
                "suffix": suffix,
                "split": split_for_suffix(suffix),
                "n_rows": n_rows,
                "score": result.primary_score,
                "branch": head_branch,
                "sha_at_eval": head_sha,
                "tree": head_tree,
                "row_scores": row_scores_from_trace_meta(result.trace_meta),
                "failed_rows": failed_rows_from_trace_meta(result.trace_meta),
            }
        )
        if eval_sample == "comparison" and self.state is not None:
            # Synchronous append so the branch/sha recorded are the ones
            # actually evaluated — immune to the agent switching branches
            # after the call.
            record = getattr(self.state, "record_comparison_eval", None)
            if callable(record):
                try:
                    record(
                        iteration=ctx.iteration_idx,
                        branch=head_branch,
                        sha=head_sha,
                        score=result.primary_score,
                        n_rows=n_rows,
                        scores_by_metric=result.scores_by_metric,
                        row_scores=row_scores,
                        eval_id=eval_id,
                        tree=head_tree,
                    )
                except Exception:  # noqa: BLE001
                    logger.warning(
                        "iteration %d: record_comparison_eval failed",
                        ctx.iteration_idx, exc_info=True,
                    )

        # Tree-based, so a commit of already-evaluated content correctly
        # reads UNCHANGED — the evaluator runs the same code either way.
        if self._last_tree_sig is None:
            change_note = "first eval"
        elif head_tree is not None and head_tree == self._last_tree_sig:
            change_note = "UNCHANGED since last eval"
        else:
            change_note = "changed since last eval"
        if head_tree is not None:
            # Never reset the baseline on a transient git failure; that would
            # report a spurious "changed" on the following call.
            self._last_tree_sig = head_tree

        logger.info(
            "iteration %d: run_eval done (id=%s, sample=%s, suffix=%s, primary_score=%s, "
            "sha=%s, tree=%s %s)",
            ctx.iteration_idx,
            eval_id,
            eval_sample,
            suffix,
            result.primary_score,
            head_sha[:8] if head_sha else None,
            (head_tree or "?")[:8],
            change_note,
        )
        return result

    def _run_initial_subsample(
        self,
        normalized_filter: dict[str, Any] | None,
    ) -> tuple[EvalResult, str]:
        """First-subsample call: resample up to MAX_INITIAL_SUBSAMPLE_RESAMPLES
        if primary_score >= max_score, then fall back to the full (filtered)
        pool.

        Each resample is additionally gated on the filtered pool being large
        enough to plausibly produce a different sample — attempts #1 and #2
        require pool >= 1.5 * self.subsample_size, attempt #3 requires >= 2x.
        If the gate fails we stop redrawing.

        Returns the final `(result, suffix)` even when `primary_score` still
        sits at ceiling. The caller (`__call__`) inspects the score and
        decides whether to commit it as the iteration's baseline or advise
        the architect to retry with a different filter.

        Subsequent subsample calls in the same iteration skip this path and
        reuse `ctx.subsample_ids` (and `ctx.subsample_filter`).
        """
        ctx = self.ctx

        # Lock filter identity for the rest of the iteration. _resolve_rows
        # reads it to narrow the candidate pool when drawing.
        ctx.subsample_filter = normalized_filter

        row_ids, suffix, timeout = self._resolve_rows("train")
        result = self._run_once(row_ids, suffix, timeout, "train")

        while result.primary_score >= self.max_score:
            if ctx.subsample_resample_count >= self.MAX_INITIAL_SUBSAMPLE_RESAMPLES:
                # Budget exhausted — swap to the full (filtered) pool and re-eval.
                full_pool = self._filtered_pool()
                logger.info(
                    "iteration %d: subsample scored %s (>= max_score=%s) "
                    "after %d resamples, falling back to full pool (size=%d, "
                    "filter=%s)",
                    ctx.iteration_idx,
                    result.primary_score,
                    self.max_score,
                    ctx.subsample_resample_count,
                    len(full_pool),
                    normalized_filter,
                )
                ctx.subsample_ids = list(full_pool)
                result = self._run_once(
                    list(full_pool),
                    "parent",
                    self.subsample_timeout,
                    "train",
                )
                break

            next_attempt = ctx.subsample_resample_count + 1
            # Attempts 1–2: pool ≥ 1.5× sample. Attempt 3: ≥ 2×. Without this
            # headroom, resampling tends to re-draw the rows we just evaluated.
            required_multiplier = 2.0 if next_attempt >= 3 else 1.5
            pool_size = len(self._filtered_pool())
            required_pool = required_multiplier * self.subsample_size
            if pool_size < required_pool:
                logger.info(
                    "iteration %d: subsample scored %s (>= max_score=%s), "
                    "skipping resample %d/%d — filtered pool size %d < required "
                    "%.1f (%.1fx subsample_size=%d, filter=%s)",
                    ctx.iteration_idx,
                    result.primary_score,
                    self.max_score,
                    next_attempt,
                    self.MAX_INITIAL_SUBSAMPLE_RESAMPLES,
                    pool_size,
                    required_pool,
                    required_multiplier,
                    self.subsample_size,
                    normalized_filter,
                )
                break

            ctx.subsample_resample_count += 1
            logger.info(
                "iteration %d: subsample scored %s (>= max_score=%s), "
                "resample %d/%d (pool=%d, required=%.1f)",
                ctx.iteration_idx,
                result.primary_score,
                self.max_score,
                ctx.subsample_resample_count,
                self.MAX_INITIAL_SUBSAMPLE_RESAMPLES,
                pool_size,
                required_pool,
            )
            ctx.subsample_ids = None  # unlock so _resolve_rows redraws
            row_ids, suffix, timeout = self._resolve_rows("train")
            result = self._run_once(row_ids, suffix, timeout, "train")

        return result, suffix

    def _resolve_rows(
        self,
        eval_sample: EvalSample,
    ) -> tuple[list[int], str, int | None]:
        """Decide which rows to evaluate and which trace suffix + timeout to use."""
        ctx = self.ctx

        if eval_sample in ("train", "subsample"):
            if ctx.subsample_ids is None:
                if self.subsample_size <= 0:
                    raise RunEvalError(
                        f"subsample_size must be positive, got {self.subsample_size}"
                    )
                if not self.trainset_ids:
                    raise RunEvalError("trainset is empty; cannot draw a subsample")
                pool = self._filtered_pool()
                if not pool:
                    raise RunEvalError(
                        f"filter {ctx.subsample_filter!r} matches 0 of "
                        f"{len(self.trainset_ids)} trainset rows; widen or "
                        "remove the filter"
                    )
                k = min(self.subsample_size, len(pool))
                ctx.subsample_ids = sorted(self.rng.sample(pool, k))
                logger.info(
                    "iteration %d: drew subsample of %d ids (pool=%d, filter=%s) ids=%s",
                    ctx.iteration_idx,
                    k,
                    len(pool),
                    ctx.subsample_filter,
                    ctx.subsample_ids,
                )
            row_ids = list(ctx.subsample_ids)
            suffix = "parent" if ctx.initial_subsample_score is None else "child"
            return row_ids, suffix, self.subsample_timeout

        if eval_sample == "comparison":
            if not self.comparison_ids:
                raise RunEvalError(
                    "no comparison set is configured for this run "
                    "(comparison_set_size was unset or 0); use "
                    "eval_sample='train' instead."
                )
            return list(self.comparison_ids), "comparison", self.comparison_timeout

        if eval_sample == "single":
            return list(self.trainset_ids[:1]), "single", self.subsample_timeout

        raise RunEvalError(f"unknown eval_sample {eval_sample!r}")

    # ----- filter pool / lock (rules: eval_filters.py) -----

    def _filtered_pool(self) -> list[int]:
        """Trainset ids matching `ctx.subsample_filter` (all ids when unset)."""
        return eval_filters.filtered_pool(
            self.trainset_ids, self.ctx.subsample_filter, self._load_trainset_rows
        )

    def _load_trainset_rows(self) -> list[dict[str, Any]]:
        """Load trainset rows from `config.trainset_path`, cached on the binding."""
        if self._trainset_rows_cache is None:
            self._trainset_rows_cache = eval_filters.load_trainset_rows(
                self.config.trainset_path, self.workspace
            )
        return self._trainset_rows_cache

    def _enforce_filter_lock(self, normalized_filter: dict[str, Any] | None) -> None:
        """Raise if the caller's filter differs from the locked filter."""
        eval_filters.enforce_filter_lock(
            self.ctx.iteration_idx, self.ctx.subsample_filter, normalized_filter
        )


__all__ = ["EvalSample", "RunEvalBinding", "RunEvalError", "row_scores_from_trace_meta"]
