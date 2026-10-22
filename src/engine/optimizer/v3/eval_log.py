"""Copyright © 2026 440 Labs LLC

Eval identifiers, the per-iteration evals JSONL, per-row score extraction
and the agent-facing sampling note for the `run_eval` tool.

Pure helpers. Nothing here calls git or the evaluator.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from src.engine.evaluator.row_scores import (
    append_row_scores_line,
    failed_rows_from_trace_meta,
    row_scores_from_trace_meta,
    row_scores_log_path,
)
from src.engine.paths import history_dir, traces_dir
from src.engine.telemetry.trace_layout import is_holdout_suffix

logger = logging.getLogger(__name__)

# `run_variance_estimate` records its repeated runs under this eval_sample so
# nothing keyed on TRAIN_EVAL_SAMPLES / "comparison" ever reads them as a
# baseline, a best reading or promotion evidence (specs/archive-public/sampling_variance.md).
VARIANCE_EVAL_SAMPLE = "variance"


def variance_suffix(run_idx: int) -> str:
    """Trace suffix for the k-th variance run: `variance`, `variance_2`, ….

    Distinct per run because `trace_layout.write_eval_trace` replaces a
    same-named directory; the tool runs once per run, so the names never
    collide across iterations either. Not a holdout suffix, so the mounted
    evaluator reads the trainset and the directories stay agent-readable.
    """
    return "variance" if run_idx == 0 else f"variance_{run_idx + 1}"


def sample_key_for_rows(row_ids: Sequence[Any]) -> str:
    """Stable digest of a set of row ids — the identity of a subsample.

    Order-independent so the same locked rows always produce the same key
    regardless of evaluation order.
    """
    joined = ",".join(sorted(str(r) for r in row_ids))
    return f"{len(row_ids)}:{hashlib.sha1(joined.encode()).hexdigest()[:12]}"


def new_eval_id() -> str:
    """Short unique id for one run_eval call (12 hex chars)."""
    return uuid.uuid4().hex[:12]


def evals_log_path(iteration_idx: int) -> str:
    """`{traces_dir}/iteration_{N}_evals.jsonl` — one line per eval."""
    return os.path.join(traces_dir(), f"iteration_{iteration_idx}_evals.jsonl")


def history_evals_log_path(iteration_idx: int) -> str:
    """`{history_dir}/iteration_{N}_evals.jsonl` — the collected copy.

    `traces_dir()` is never collected off the Docker runner (`orchestrator/
    collect.py`), so the evals log is dual-written here; the
    traces copy stays because the architect and the evo rerun warning read it.
    """
    return os.path.join(history_dir(), f"iteration_{iteration_idx}_evals.jsonl")


def append_eval_log_line(iteration_idx: int, line: Mapping[str, Any]) -> None:
    """Append one eval to the per-iteration evals log (traces + history copies).
    Best-effort."""
    payload = json.dumps(dict(line)) + "\n"
    for path in (evals_log_path(iteration_idx), history_evals_log_path(iteration_idx)):
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a") as f:
                f.write(payload)
        except OSError:
            logger.warning("failed to append eval log line to %s", path, exc_info=True)


def split_for_suffix(suffix: str) -> str:
    """Map a trace suffix to the dataset split it selects.

    Mirrors the mounted evaluator's rule (`evaluator/mounted/evaluate.py`
    `_load_dataset`): `"holdout"` / `"holdout_<k>"` → test, `"comparison"` →
    comparison (a fixed trainset subsample), everything else (`"parent"`,
    `"child"`, `"single"`, …) → training.
    """
    if is_holdout_suffix(suffix):
        return "test"
    if suffix == "comparison":
        return "comparison"
    return "training"


def sampling_note(eval_sample: str, suffix: str, n_rows: int) -> str:
    """Guidance attached to every eval response, kept true to the split and
    shape actually scored.

    The split is derived from `suffix` (the same key the mounted evaluator
    uses to pick the dataset file) rather than assumed, so the note can
    never drift from the data that was run. `_resolve_rows` emits suffix
    parent/child/single (trainset) and comparison (the fixed trainset
    subsample); the test branch below is defensive — it only fires if a
    future change exposes that split to the tool.
    """
    split = split_for_suffix(suffix)
    n = n_rows

    if eval_sample == VARIANCE_EVAL_SAMPLE:
        return (
            f"This is one of the repeated variance-estimate runs on a fixed "
            f"{n}-row training sample. It is NOT a training eval: it does not "
            "set or replace this iteration's baseline and is not compared "
            "against your before/after scores."
        )

    if eval_sample == "single":
        return (
            f"This is a single {split}-set row, evaluated for quick "
            "inspection of one example's trace — it spends 1 budget row, "
            "is NOT a reliable score and should not drive a keep/revert "
            "decision. Use eval_sample='train' for a score you can act on."
        )

    if split == "comparison":
        return (
            f"This score is measured on the run's FIXED comparison set — {n} rows "
            "drawn once from the training set (seed 42) and identical for every "
            "branch and iteration. Use it for apples-to-apples comparison across "
            "branches (e.g. before promoting a new primary). It overlaps the "
            "training data, so a branch that trained on more of these rows has a "
            "slight edge. Comparison evals consume real eval budget — run them "
            "sparingly. This eval was appended to comparison_set_history.jsonl "
            "with the branch and sha that were evaluated."
        )

    # train subsample
    if split == "training":
        return (
            f"The training data subsample score is measured on {n} rows randomly drawn "
            "from the TRAINING set. A random number seed ensures that the same sample is drawn within iterations (i.e., before/after delta is fair)." 
            "However, between iterations, the seed is changed so you can learn from more examples across many iterations." 
            "WARNING: Do NOT over-rotate on specific rows. Do not over-rotate on subsample scores if they are driven by noisy sampling."
            "Subsamples when small (e.g., < 100 rows) can be quite noisy depending on the application, especially when the seed differs between iterations. "                
            "Ultimately, your work is judged by its ability to generalize beyond the training set."
            "Use the training set to test generalizable hypotheses." 
        )
    # validation / test: a fixed, held-out slice — no per-iteration redraw.
    return (
        f"This score is measured on a fixed, held-out {split}-set slice of "
        f"{n} rows. Whereas training subsamples are re-drawn between iterations, this set is fixed between iterations. "
        "Note: Subsamples when small (e.g., < 100 rows) can be quite noisy depending on the application. "
        "Factor in some expectation for variance depending on the application and sample size. "
    )


__all__ = [
    "VARIANCE_EVAL_SAMPLE",
    "append_eval_log_line",
    "append_row_scores_line",
    "evals_log_path",
    "failed_rows_from_trace_meta",
    "history_evals_log_path",
    "new_eval_id",
    "row_scores_from_trace_meta",
    "row_scores_log_path",
    "sample_key_for_rows",
    "sampling_note",
    "split_for_suffix",
    "variance_suffix",
]
