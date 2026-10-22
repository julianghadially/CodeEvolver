"""Copyright © 2026 440 Labs LLC

Per-session history record, derived from the in-process eval log.

`RunEvalBinding` appends every eval to `ctx.eval_records`, so the best train
and comparison scores of a session are a max over that list — no trace files
are parsed. `score` / `best_subsample_score` is the best training reading ON
THE BASELINE'S LOCKED ROWS (engine 3's rule, `v3/iteration_context.py::
finalize`): 1-row `single` evals and any train eval drawn on other rows
(ceiling resample) are excluded, so best-vs-baseline is always the same
sample. When the post-session constraint gate reverted the session,
`score` (the headline consumed by `candidate_scores`, the progress line and
the meta-reflection run table) is None — the reading belongs to a tree that
was discarded — while `best_subsample_score` / `comparison_best_score` keep
the measured values beside `constraint_violation` for analysis. `close_sha`
is then the revert marker commit (`constraint_check.py`), which differs from
`parent_sha` by sha but not by content, so `changes_landed` is False.
No sha is attributed to either score: the simple loop never
reconciles dirty-tree evals, so an eval's `sha` is HEAD at eval time — the
parent commit whenever the tree was dirty — and would mislead. The exact
tree of every eval is in `{traces_dir}/iteration_{N}_evals.jsonl`.
"""

from __future__ import annotations

from typing import Any

from src.engine.schemas.iteration_context import (
    TRAIN_EVAL_SAMPLES,
    BaseIterationContext,
)


def _best_train(records: list[dict], baseline_sample_key: str | None) -> float | None:
    best: float | None = None
    for rec in records:
        score = rec.get("score")
        if rec.get("eval_sample") not in TRAIN_EVAL_SAMPLES or rec.get("eval_sample") == "single":
            continue
        if not isinstance(score, (int, float)):
            continue
        key = rec.get("sample_key")
        if baseline_sample_key is not None and key is not None and key != baseline_sample_key:
            continue
        if best is None or score > best:
            best = float(score)
    return best


def _best_comparison(records: list[dict]) -> float | None:
    scores = [
        float(rec["score"])
        for rec in records
        if rec.get("eval_sample") == "comparison" and isinstance(rec.get("score"), (int, float))
    ]
    return max(scores) if scores else None


def session_record(
    ctx: BaseIterationContext,
    *,
    branch: str,
    start_sha: str,
    end_sha: str,
    rows_used: int,
    constraint_violation: str | None,
    skipped: bool,
) -> dict[str, Any]:
    """Fields for `IterationHistory[n]` after one session."""
    records = list(ctx.eval_records)
    best_train = _best_train(records, ctx.baseline_sample_key)
    return {
        "git_branch": branch,
        "parent_sha": start_sha,
        "close_sha": end_sha,
        "changes_landed": constraint_violation is None and start_sha != end_sha,
        "eval_rows_used": rows_used,
        "initial_subsample_score": ctx.initial_subsample_score,
        "best_subsample_score": best_train,
        "comparison_best_score": _best_comparison(records),
        "score": None if constraint_violation is not None else best_train,
        "constraint_violation": constraint_violation,
        "skipped": skipped,
    }


__all__ = ["session_record"]
