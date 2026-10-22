"""Copyright © 2026 440 Labs LLC

Host-side entry point to the generic evaluator.

`run_eval` serializes the request, dispatches through the execution layer's
`run_evaluation()` subprocess call, and returns the parsed result plus the
trace meta path. No sandbox object dependency — takes a workspace path.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .config import EvaluatorConfig

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Return shape of `run_eval`."""

    scores_by_metric: dict[str, float]
    primary_score: float
    trace_meta: list[dict[str, Any]]
    # `{traces_dir}/iteration_{N}_{suffix}` — see telemetry/trace_layout.py.
    trace_dir: str
    trace_index_file: str
    row_ids: list[int]
    raw: dict[str, Any]


def run_eval(
    *,
    workspace: str,
    config: EvaluatorConfig,
    row_ids: list[int] | None,
    iteration_idx: int,
    suffix: str,
    capture_traces: bool = True,
    timeout: int | None = None,
) -> EvalResult:
    """Run the generic evaluator via subprocess and return structured results.

    Args:
        workspace: Path to the client repo workspace.
        config: Evaluator config describing the program and metrics.
        row_ids: Explicit dataset row ids to evaluate. When None, the mounted
            evaluator evaluates the full dataset it has loaded.
        iteration_idx: Iteration number — used to name the trace directory
            (`/traces/iteration_{N}_{suffix}/`).
        suffix: `"parent"` (pre-mutation eval), `"child"` (post-mutation eval),
            `"comparison"`, or `"holdout"` / `"holdout_<k>"` (held-out test set).
        capture_traces: When True, the mounted evaluator collects OTel spans
            into the per-row span files. Defaults to True; only disable for
            budget-sensitive single-row sanity checks.
        timeout: Optional hard timeout in seconds.

    Returns:
        EvalResult with per-metric scores and the path to the trace files.

    Raises:
        RuntimeError: If the mounted script returns an error or a malformed
            response.
    """
    from src.engine.services.subprocess_service import run_evaluation

    payload = config.to_payload(
        row_ids=row_ids,
        iteration_idx=iteration_idx,
        suffix=suffix,
        capture_traces=capture_traces,
    )
    result = run_evaluation(workspace, payload, timeout=timeout)

    if not isinstance(result, dict):
        raise RuntimeError(f"Mounted evaluator returned non-dict: {result!r}")
    if not result.get("success", False):
        err = result.get("error") or "mounted evaluator failed without an error message"
        raise RuntimeError(f"Mounted evaluator failed: {err}")

    scores = result.get("scores_by_metric")
    if not isinstance(scores, dict):
        raise RuntimeError(
            f"Mounted evaluator missing 'scores_by_metric' dict: {result!r}"
        )
    if config.primary_metric not in scores:
        raise RuntimeError(
            f"primary_metric {config.primary_metric!r} missing from evaluator scores "
            f"{list(scores.keys())!r}"
        )

    return EvalResult(
        scores_by_metric=dict(scores),
        primary_score=float(scores[config.primary_metric]),
        trace_meta=list(result.get("trace_meta") or []),
        trace_dir=str(result.get("trace_dir") or ""),
        trace_index_file=str(result.get("trace_index_file") or ""),
        row_ids=list(result.get("row_ids") or row_ids or []),
        raw=result,
    )
