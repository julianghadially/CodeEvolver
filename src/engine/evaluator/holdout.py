"""Copyright © 2026 440 Labs LLC

Held-out test-set sampling for finalization.

Every arm scores the seed and the final program on the held-out test set
at the end of a run. Benchmarks vary several points call-to-call, so one
reading of the final program is a poor estimate; `sample_test_scores` draws
the test set `num_samples` times and returns every reading so the run can
report the list AND the mean (`seed_test_scores` / `final_test_scores`
alongside the existing `seed_test_score` / `final_test_score`).

Lives in the evaluator package so both `optimizer/v3/finalization.py` and
`optimizer/simple_loop/loop.py` can import it (the simple loop's import
boundary fences v3 logic modules, not the evaluator).

Caching is NOT configured here, by decision: the client repo
owns its LM and cache settings (`dspy.LM(cache=False)`, etc.). The engine
only DETECTS a cache — when every sample's row outputs are byte-identical
it logs a warning — so a benchmark that left response caching on is
visible in the logs rather than silently reporting zero variance.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Callable

from src.engine.telemetry.trace_layout import holdout_suffix, iter_rows

from .config import EvaluatorConfig
from .dataset_config import DatasetConfig
from .row_scores import record_eval_row_scores
from .sandbox_client import EvalResult, run_eval

logger = logging.getLogger(__name__)

IDENTICAL_SAMPLES_WARNING = (
    "finalization test samples are identical across %d draws (iteration %d) "
    "— an LLM response cache may be active in the client program. Sampling "
    "adds no information until the client disables it."
)


def test_suffix(sample_idx: int) -> str:
    """Trace suffix for the k-th test-set sample: `holdout`, `holdout_2`, ….

    Every draw's trace directory therefore contains the `_holdout` fence
    fragment (`trace_layout.HOLDOUT_DENY_FRAGMENT`), which is what the agent
    deny-path hooks match; the mounted evaluator resolves these suffixes to
    the held-out file (`trace_layout.is_holdout_suffix`).
    """
    return holdout_suffix(sample_idx)


def mean_or_none(scores: list[float]) -> float | None:
    """Arithmetic mean of `scores`, or None when there are none."""
    if not scores:
        return None
    return sum(scores) / len(scores)


def sample_test_scores(
    *,
    workspace: str,
    config: EvaluatorConfig,
    datasets: DatasetConfig,
    iteration_idx: int,
    num_samples: int,
    timeout: int | None,
    run_eval_fn: Callable[..., EvalResult] = run_eval,
) -> list[float]:
    """Score the checked-out program on the held-out test set `num_samples` times.

    Returns the primary score of every sample that succeeded, in draw order.
    A failed draw is logged and skipped rather than aborting the remaining
    draws, so a transient provider error costs one reading, not all of them.
    Never touches the eval-row budget — the caller is finalization, which
    is outside it by spec (§24). `run_eval_fn` exists so callers can pass
    their own module-level `run_eval` binding (tests patch it there).
    """
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1, got {num_samples}")
    scores: list[float] = []
    results: list[EvalResult] = []
    for k in range(num_samples):
        try:
            result = run_eval_fn(
                workspace=workspace,
                config=config,
                row_ids=datasets.testset_ids,
                iteration_idx=iteration_idx,
                suffix=test_suffix(k),
                capture_traces=True,
                timeout=timeout,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "finalization: test sample %d/%d (iteration %d) failed "
                "(continuing): %s",
                k + 1, num_samples, iteration_idx, exc,
            )
            continue
        scores.append(result.primary_score)
        results.append(result)
        record_eval_row_scores(
            workspace=workspace,
            result=result,
            iteration=iteration_idx,
            eval_sample="holdout",
            suffix=test_suffix(k),
            split="test",
        )
        logger.info(
            "finalization: test sample %d/%d (iteration %d) -> %.4f",
            k + 1, num_samples, iteration_idx, result.primary_score,
        )
    if len(results) > 1 and samples_are_identical(
        [r.trace_dir for r in results]
    ):
        logger.warning(IDENTICAL_SAMPLES_WARNING, len(results), iteration_idx)
    return scores


def samples_are_identical(trace_dirs: list[str]) -> bool:
    """True when every trace directory holds the same per-row outputs.

    The cache detector. Compares the JSON-serialized `output` of each row
    across the sample trace directories; unreadable or missing ones make the
    answer False (no evidence, no warning). Failed rows are ignored — their
    `output` is null in every draw, which is not evidence of a cache — and a
    draw with no successful rows contributes no evidence either. Row order
    is by `example_id` so thread-scheduling differences do not create false
    negatives.
    """
    signatures: list[str] = []
    for path in trace_dirs:
        sig = _outputs_signature(path)
        if sig is None:
            return False
        signatures.append(sig)
    return len(signatures) > 1 and len(set(signatures)) == 1


def _outputs_signature(trace_dir: str) -> str | None:
    if not trace_dir or not os.path.isdir(trace_dir):
        return None
    rows: list[tuple[str, str]] = []
    try:
        for entry in iter_rows(trace_dir):
            if entry.get("failed"):
                continue
            rows.append((
                str(entry.get("example_id")),
                json.dumps(entry.get("output"), sort_keys=True, default=str),
            ))
    except Exception:  # noqa: BLE001
        return None
    if not rows:
        return None
    rows.sort()
    return json.dumps(rows)


__all__ = [
    "IDENTICAL_SAMPLES_WARNING",
    "mean_or_none",
    "sample_test_scores",
    "samples_are_identical",
    "test_suffix",
]
