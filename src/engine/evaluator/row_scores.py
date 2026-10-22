"""Copyright © 2026 440 Labs LLC

Durable per-row scores: `{history_dir}/row_scores.jsonl`.

Every eval the engine pays for (training subsamples, comparison-set runs,
variance runs, finalization holdout draws and the final comparison) writes one
line here with the primary score of every row it scored. Otherwise per-row
scores would survive a run only for comparison evals
(`comparison_set_history.jsonl`) and inside `traces/`, which the Docker runner
never collects — so the offline analyses (`scripts/best_of_population/README.md`)
would have no per-row record of the holdout draws or training subsamples a run
already spent budget on. `history_dir()` IS collected (`orchestrator/
collect.py::JOB_TREES`), and a typical line is a few KB.

Lives in the evaluator package (not `optimizer/v3`) so `holdout.py` and
`finalization.py` can call it without the evaluator importing optimizer code.
Best-effort: a write failure is logged, never raised.
"""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from collections.abc import Mapping, Sequence
from typing import Any

from src.engine.paths import history_dir

logger = logging.getLogger(__name__)

ROW_SCORES_FILENAME = "row_scores.jsonl"


def row_scores_log_path() -> str:
    """`{history_dir}/row_scores.jsonl` — one line per eval, all splits."""
    return os.path.join(history_dir(), ROW_SCORES_FILENAME)


def row_scores_from_trace_meta(meta: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """`str(example_id) -> primary score` from the evaluator's per-row meta.

    Rows with no `example_id` or a non-numeric score are skipped. Keys are
    strings so the mapping survives a JSON round-trip unchanged.
    """
    out: dict[str, float] = {}
    for row in meta or []:
        rid = row.get("example_id")
        score = row.get("score")
        if rid is None or score is None:
            continue
        try:
            out[str(rid)] = float(score)
        except (TypeError, ValueError):
            continue
    return out


def failed_rows_from_trace_meta(meta: Sequence[Mapping[str, Any]]) -> list[str]:
    """`example_id`s whose row raised (`failed: true`), as strings."""
    return [
        str(row.get("example_id"))
        for row in meta or []
        if row.get("example_id") is not None and row.get("failed")
    ]


def append_row_scores_line(line: Mapping[str, Any]) -> None:
    """Append one eval's per-row scores. Best-effort, never raises.

    Expected keys: `eval_id, ts, iteration, eval_sample, suffix, split,
    n_rows, score, branch, sha_at_eval, tree, row_scores[, failed_rows]`.
    """
    path = row_scores_log_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(dict(line)) + "\n")
    except OSError:
        logger.warning("failed to append row scores line to %s", path, exc_info=True)


def record_eval_row_scores(
    *,
    workspace: str,
    result: Any,
    iteration: int,
    eval_sample: str,
    suffix: str,
    split: str,
    eval_id: str | None = None,
) -> None:
    """Record one `EvalResult`'s per-row scores with the git identity of the
    program that produced them (loop-side callers: finalization holdout
    draws and the final comparison). Never raises."""
    branch = sha = tree = None
    try:
        from src.engine.services.subprocess_service import (
            git_get_current_branch,
            git_get_sha,
            git_tree_signature,
        )

        sha = git_get_sha(workspace)
        tree = git_tree_signature(workspace)
        try:
            branch = git_get_current_branch(workspace)
        except Exception:  # noqa: BLE001
            branch = None
    except Exception:  # noqa: BLE001
        logger.debug("row_scores: git identity unavailable", exc_info=True)
    meta = getattr(result, "trace_meta", None) or []
    append_row_scores_line(
        {
            "eval_id": eval_id or uuid.uuid4().hex[:12],
            "ts": time.time(),
            "iteration": iteration,
            "eval_sample": eval_sample,
            "suffix": suffix,
            "split": split,
            "n_rows": len(getattr(result, "row_ids", None) or meta),
            "score": getattr(result, "primary_score", None),
            "branch": branch,
            "sha_at_eval": sha,
            "tree": tree,
            "row_scores": row_scores_from_trace_meta(meta),
            "failed_rows": failed_rows_from_trace_meta(meta),
        }
    )


__all__ = [
    "ROW_SCORES_FILENAME",
    "append_row_scores_line",
    "failed_rows_from_trace_meta",
    "record_eval_row_scores",
    "row_scores_from_trace_meta",
    "row_scores_log_path",
]
