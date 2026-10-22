"""Copyright © 2026 440 Labs LLC

Post-loop finalization for the engine 3 driver.

- `score_seed_and_final`: the seed and the LAST DECLARED PRIMARY
  (close_iteration is the source of truth; the agent chooses) scored on the
  held-out test set (its only use in the run) — drawn `test_samples` times
  each (`evaluator/holdout.py`), reported as the per-draw lists
  `seed_test_scores` / `final_test_scores` with `seed_test` / `final_test`
  their means — AND on the fixed comparison set (`seed_train_score` /
  `final_train_score` — a metrics-only train-side measurement). One checkout
  per branch runs all evals. NONE of these evals count against the eval
  budget (spec §24) and comparison runs here do not append to
  comparison_set_history.jsonl (the agent-facing log).
- `run_post_run_meta_reflection` / `run_post_run_integrity`: the opt-in,
  advisory post-run agents (`optimizer/meta/`). Both run AFTER the result is
  built and completion has been posted, so they can never affect run
  success. Their imports stay lazy — the meta package is heavy and optional.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DatasetConfig
from src.engine.evaluator.holdout import mean_or_none, sample_test_scores
from src.engine.evaluator.row_scores import record_eval_row_scores
from src.engine.evaluator.sandbox_client import run_eval
from src.engine.schemas.iteration_history import IterationHistory
from src.engine.services.subprocess_service import (
    clean_python_bytecode,
    git_checkout,
)

from .ce_state import CEState
from .coding_harness_config import CodingHarnessConfig
from .run_reporting import final_candidate

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FinalScores:
    """Held-out + comparison-set readings for the seed and the final primary.

    `seed_test` / `final_test` are the MEANS of `seed_test_scores` /
    `final_test_scores` (one entry per test-set draw). Any score is None
    (list empty) when that split is not configured or scoring failed.
    """

    final_branch: str | None
    final_sha: str | None
    seed_test: float | None = None
    final_test: float | None = None
    seed_train: float | None = None
    final_train: float | None = None
    seed_test_scores: list[float] = field(default_factory=list)
    final_test_scores: list[float] = field(default_factory=list)
    finalization_error: str | None = None


def score_seed_and_final(
    *,
    workspace: str,
    datasets: DatasetConfig,
    evaluator_config: EvaluatorConfig,
    state: CEState,
    history: IterationHistory,
    seed_branch: str,
    seed_sha: str,
    comparison_eval_timeout: int | None,
    test_samples: int = 1,
    seed_test_samples: int = 1,
) -> FinalScores:
    """Score seed + final primary on test and comparison sets (uncounted).

    The test set is drawn `seed_test_samples` times for the seed and
    `test_samples` times for the final primary. Also stamps
    `test_score` (mean) and `test_scores` (every draw) onto the seed record
    and the latest record on the final primary branch so the UI can
    star-plot the held-out datapoints. Non-fatal: a scoring failure logs
    and returns None scores.

    The seed is scored at `seed_sha` (a detached checkout), NEVER by checking
    out `seed_branch`: the run-main branch is the branch the architect works
    on (freeform always, EE/ASA whenever the agent commits on the primary),
    so its tip at finalization is the *final* program; scoring the seed by
    branch name would measure the evolved program twice. `seed_branch`
    is kept only for the returned metadata / short-circuit bookkeeping.
    """
    candidate = final_candidate(state)
    final_branch = candidate.get("git_branch")
    final_sha = candidate.get("git_sha")

    seed_test_scores: list[float] = []
    final_test_scores: list[float] = []
    seed_train_score: float | None = None
    final_train_score: float | None = None

    finalization_error: str | None = None

    if not (datasets.has_testset or datasets.comparison_ids):
        return FinalScores(final_branch=final_branch, final_sha=final_sha)

    def _eval_branch(
        ref: str | None, iteration_idx: int, num_test_samples: int
    ) -> tuple[list[float], float | None]:
        """Score git `ref` (a branch name or a commit sha) on the test set
        (`num_test_samples` draws) and the comparison set.

        Returns `(test_scores, train_score)`; the list is empty / the score
        None when that split is not configured. Deliberately does NOT call
        `state.increment_evals` — finalization evals are outside the
        eval-row budget.
        """
        if ref:
            try:
                clean_python_bytecode(workspace)
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "finalization: bytecode cleanup before checkout "
                    "failed (continuing): %s",
                    exc,
                )
            git_checkout(workspace, ref, force=True)
        test_scores: list[float] = []
        train_score: float | None = None
        if datasets.has_testset:
            test_scores = sample_test_scores(
                workspace=workspace,
                config=evaluator_config,
                datasets=datasets,
                iteration_idx=iteration_idx,
                num_samples=num_test_samples,
                timeout=comparison_eval_timeout,
                run_eval_fn=run_eval,
            )
        if datasets.comparison_ids:
            # `final_comparison` (not `comparison`) so the trace files
            # never collide with the agent's own comparison evals; the
            # mounted evaluator resolves any non-test suffix to the
            # trainset file.
            comp_eval = run_eval(
                workspace=workspace,
                config=evaluator_config,
                row_ids=datasets.comparison_ids,
                iteration_idx=iteration_idx,
                suffix="final_comparison",
                capture_traces=True,
                timeout=comparison_eval_timeout,
            )
            train_score = comp_eval.primary_score
            record_eval_row_scores(
                workspace=workspace,
                result=comp_eval,
                iteration=iteration_idx,
                eval_sample="final_comparison",
                suffix="final_comparison",
                split="comparison",
            )
        return test_scores, train_score

    try:
        logger.info(
            "finalization: scoring seed (sha=%s, test rows=%d x %d samples, "
            "comparison rows=%d)",
            (seed_sha or "")[:8],
            datasets.testset_size,
            seed_test_samples,
            len(datasets.comparison_ids),
        )
        seed_test_scores, seed_train_score = _eval_branch(
            seed_sha, iteration_idx=0, num_test_samples=seed_test_samples
        )
        if final_sha == seed_sha and test_samples <= seed_test_samples:
            # Final primary IS the seed (same commit, not merely the same
            # branch name) — one pass, identical scores. Leave the workspace
            # on the final branch (the seed pass was a detached checkout).
            final_test_scores = list(seed_test_scores)
            final_train_score = seed_train_score
            if final_branch:
                git_checkout(workspace, final_branch, force=True)
        else:
            logger.info(
                "finalization: scoring final primary (branch=%s, test x %d)",
                final_branch, test_samples,
            )
            final_test_scores, final_train_score = _eval_branch(
                final_branch, iteration_idx=1, num_test_samples=test_samples
            )
        # Stamp test scores onto the seed record and the latest record on
        # the final primary branch so the UI can star-plot the held-out
        # datapoints.
        if 0 in history:
            history[0]["test_score"] = mean_or_none(seed_test_scores)
            history[0]["test_scores"] = list(seed_test_scores)
        for iter_idx in sorted((k for k in history if k != 0), reverse=True):
            if history[iter_idx].get("git_branch") == final_branch:
                history[iter_idx]["test_score"] = mean_or_none(final_test_scores)
                history[iter_idx]["test_scores"] = list(final_test_scores)
                break
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "finalization: seed/final scoring failed (non-fatal): %s", exc,
        )
        finalization_error = f"{type(exc).__name__}: {exc}"
        # A failure mid-seed-pass leaves the workspace on a detached HEAD;
        # the meta-reflection / integrity agents that run next need the
        # final branch checked out. (The success path already ends there.)
        if final_branch:
            try:
                git_checkout(workspace, final_branch, force=True)
            except Exception as restore_exc:  # noqa: BLE001
                logger.warning(
                    "finalization: restoring branch %s failed: %s",
                    final_branch, restore_exc,
                )

    return FinalScores(
        final_branch=final_branch,
        final_sha=final_sha,
        seed_test=mean_or_none(seed_test_scores),
        final_test=mean_or_none(final_test_scores),
        seed_train=seed_train_score,
        final_train=final_train_score,
        seed_test_scores=list(seed_test_scores),
        final_test_scores=list(final_test_scores),
        finalization_error=finalization_error,
    )


def run_post_run_meta_reflection(
    *,
    workspace: str,
    method: str,
    history: IterationHistory,
    scores: FinalScores,
    agent_model: str | None,
    coding_harness_config: CodingHarnessConfig,
) -> None:
    """Post-run meta-reflection (opt-in, advisory only). Never raises."""
    try:
        from src.engine.optimizer.meta.meta_reflection import run_meta_reflection

        run_meta_reflection(
            workspace=workspace,
            method=method,
            num_iterations=len(history),
            seed_test_score=scores.seed_test,
            best_candidate_test_score=scores.final_test,
            iteration_indices=sorted(history.keys()),
            agent_model=agent_model,
            coding_harness=coding_harness_config.coding_harness,
            coding_harness_model=coding_harness_config.coding_harness_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta-reflection step failed (non-fatal): %s", exc)


def run_post_run_integrity(
    *,
    workspace: str,
    method: str,
    history: IterationHistory,
    scores: FinalScores,
    testset_path: str | None,
    agent_model: str | None,
    coding_harness_config: CodingHarnessConfig,
) -> bool:
    """Post-run run-integrity inspection (opt-in, advisory only).

    The verdict is stamped onto the LATEST iteration record; returns True
    when it was, so the caller can re-flush progress (the /progress handler
    never touches job status, and both the JSONL mirror and Mongo are
    last-write-wins per iteration, so the late flush converges). The verdict
    is NEVER used to stop or fail the run. Never raises.
    """
    try:
        from src.engine.optimizer.meta.run_integrity import (
            run_integrity_inspection,
            stamp_integrity_verdict,
        )

        verdict = run_integrity_inspection(
            workspace=workspace,
            method=method,
            num_iterations=len(history),
            best_valset_score=scores.final_train,
            seed_test_score=scores.seed_test,
            best_candidate_test_score=scores.final_test,
            iteration_indices=sorted(history.keys()),
            testset_path=testset_path,
            safe_mode=False,
            agent_model=agent_model,
            coding_harness=coding_harness_config.coding_harness,
            coding_harness_model=coding_harness_config.coding_harness_model,
        )
        if verdict is None:
            # Loud on purpose: this is the run's only cheating / functioning-
            # iteration audit, and a silent failure is easy to miss. Still
            # advisory -- the run stays completed.
            logger.error(
                "run-integrity: NO VERDICT -- the closing integrity auditor "
                "produced no output (see the preceding "
                "'meta-reflection: ... (method=run-integrity ...)' lines for "
                "the harness error); integrity_* stays None for this run"
            )
            return False
        return stamp_integrity_verdict(history, verdict) is not None
    except Exception as exc:  # noqa: BLE001
        logger.error("run-integrity step failed (non-fatal): %s", exc)
        return False


__all__ = [
    "FinalScores",
    "run_post_run_integrity",
    "run_post_run_meta_reflection",
    "score_seed_and_final",
]
