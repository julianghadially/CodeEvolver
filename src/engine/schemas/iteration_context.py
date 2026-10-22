"""Copyright © 2026 440 Labs LLC

BaseIterationContext — the per-iteration state the shared instrumentation
needs, and nothing more.

`run_eval` (`optimizer/v3/run_eval_tool.py`) writes the subsample lock, the
baseline reading and the eval log here; the coding harnesses read
`iteration_idx` to name trace files. Every optimizer arm constructs one of
these (or a subclass). Engine 3's `IterationContext` extends it with closing
state and the logic that interprets it; the simple loop uses it as-is. The
two methods are append-only bookkeeping — no decisions live here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, NotRequired, TypedDict

TRAIN_EVAL_SAMPLES: tuple[str, ...] = ("train", "single", "subsample")


class EvalRecord(TypedDict):
    """One `run_eval` call, as logged on `BaseIterationContext.eval_records`.

    `sha` is HEAD at eval time (the parent commit when the tree is dirty);
    `tree` is the git tree hash of the code that actually ran and is the
    content identity engine 3 reconciles on.
    """

    sha: str | None
    tree: NotRequired[str | None]
    sha_at_eval: NotRequired[str | None]
    branch: str | None
    eval_sample: str
    score: float
    n_rows: int
    ts: float | None
    row_scores: NotRequired[dict[str, float]]
    eval_id: NotRequired[str]
    sample_key: NotRequired[str]
    committed: NotRequired[bool]
    anchor_ref: NotRequired[str]
    by_loop: NotRequired[bool]


@dataclass(kw_only=True)
class BaseIterationContext:
    """State written by `run_eval` and the harness during one agent session."""

    iteration_idx: int
    # Run-level snapshot, read by the system-prompt builders. `evals_used` is
    # the budget consumed at the START of the iteration.
    subsample_size: int = 20
    comparison_set_size: int = 0
    max_eval_rows: int | None = None
    evals_used: int = 0
    # `run_eval` refuses to run until this is True.
    parent_locked: bool = False
    # Subsample identity — locked on the first train call of the iteration;
    # later train calls reuse the same rows (and must pass the same filter).
    subsample_ids: list[Any] | None = None
    initial_subsample_score: float | None = None
    subsample_resample_count: int = 0
    subsample_filter: dict[str, Any] | None = None
    baseline_sample_key: str | None = None
    # Was the working tree already modified when the first train eval ran?
    # Observability only, recorded once.
    baseline_tree_dirty: bool | None = None
    baseline_dirty_paths: list[str] = field(default_factory=list)
    # Comparison-credit acknowledgement (specs/archive-public/comparison_credits.md). The
    # loop stamps a fresh 2-digit code at iteration start; the first
    # comparison call without it returns the acknowledgement note carrying
    # the code, and a correct code latches `comparison_ack_given` for the
    # rest of the iteration. Empty code = ledger disabled for this run.
    comparison_ack_code: str = ""
    comparison_ack_given: bool = False
    # Set by the loop from the prior iteration's history record; the kickoff
    # builders (every arm) append `constraint_gate.PREVIOUS_ITERATION_REVERTED_NOTE`
    # when True -the note says that the previous iteration was reverted due to violation of constraints in the additional instructions.
    previous_iteration_constraint_reverted: bool = False
    # `(sha, score)` per train/single eval; comparison evals excluded.
    eval_history: list[tuple[str, float]] = field(default_factory=list)
    # Full eval log — one record per `run_eval` call of any kind.
    eval_records: list[EvalRecord] = field(default_factory=list)
    # Token usage this iteration, keyed by normalized model id; a plain dict
    # so the context survives a JSON round-trip.
    token_usage_by_model: dict[str, dict[str, int]] = field(default_factory=dict)

    def record_eval(
        self,
        sha: str,
        primary_score: float,
        *,
        eval_sample: str = "train",
        branch: str | None = None,
        n_rows: int = 0,
        ts: float | None = None,
        tree: str | None = None,
        row_scores: Mapping[str, float] | None = None,
        eval_id: str | None = None,
        sample_key: str | None = None,
        committed: bool | None = None,
        by_loop: bool = False,
    ) -> None:
        """Log one run_eval call (append-only)."""
        rec = EvalRecord(
            sha=sha, tree=tree, branch=branch, eval_sample=eval_sample,
            score=primary_score, n_rows=n_rows, ts=ts,
        )
        if row_scores is not None:
            rec["row_scores"] = dict(row_scores)
        if eval_id is not None:
            rec["eval_id"] = eval_id
        if sample_key is not None:
            rec["sample_key"] = sample_key
        if committed is not None:
            rec["committed"] = committed
        if by_loop:
            rec["by_loop"] = True
        self.eval_records.append(rec)
        if eval_sample in TRAIN_EVAL_SAMPLES:
            self.eval_history.append((sha, primary_score))

    def record_token_usage(self, model: str | None, usage: Any) -> None:
        """Fold one model call's `TokenUsage` into `token_usage_by_model`."""
        from src.engine.agent.token_usage import TokenUsage, normalize_model_id

        key = normalize_model_id(model)
        prior = TokenUsage.from_dict(self.token_usage_by_model.get(key))
        self.token_usage_by_model[key] = (prior + usage).to_dict()


__all__ = ["BaseIterationContext", "EvalRecord", "TRAIN_EVAL_SAMPLES"]
