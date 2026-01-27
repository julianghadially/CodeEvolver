"""GEPA optimization orchestrator for CodeEvolver.

This module is called from the Modal function to set up and run the
GEPA optimization loop. It loads user code, creates the adapter,
and calls gepa.optimize().
"""

from pathlib import Path
from typing import Any

from gepa import optimize as gepa_optimize
from gepa.core.result import GEPAResult

from .adapter import CodeEvolverDSPyAdapter
from .callback import CallbackJobUpdater, CallbackProgressTracker
from .utils import load_import_path, resolve_dataset


def run_gepa_optimization(
    job_id: str,
    callback_url: str,
    jwt_token: str,
    workspace_path: str,
    program: str,
    metric: str,
    reflection_lm: str,
    max_metric_calls: int,
    saved_program_json_path: str | None = None,
    trainset_json: list[dict[str, Any]] | None = None,
    trainset_path: str | None = None,
    valset_json: list[dict[str, Any]] | None = None,
    valset_path: str | None = None,
    input_keys: list[str] | None = None,
    num_threads: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Run GEPA optimization. Called from the Modal function.

    This is synchronous — GEPA's optimize() is a blocking call.

    Args:
        job_id: Unique job identifier.
        workspace_path: Path to cloned user repo.
        program: Dotted import path to DSPy module class.
        metric: Dotted import path to metric function.
        reflection_lm: LM for GEPA reflection.
        max_metric_calls: Budget for optimization.
        callback_url: Base URL for HTTP callbacks to FastAPI.
        jwt_token: Job-scoped JWT for authenticating callbacks.
        saved_program_json_path: Relative path to program.json (optional).
        trainset_json: Training data as inline list of dicts.
        trainset_path: Path to training data file in repo.
        valset_json: Validation data as inline list of dicts.
        valset_path: Path to validation data file in repo.
        input_keys: Optional explicit list of input field names.
        num_threads: Parallelism for evaluation.
        seed: Random seed for reproducibility.

    Returns:
        Dict with optimization results.
    """
    ws = Path(workspace_path)
    updater = CallbackJobUpdater(callback_url, jwt_token, job_id)

    # Update job status to running
    updater.set_running()

    try:
        # Load user's metric function
        metric_fn = load_import_path(workspace_path, metric)

        # Resolve datasets
        trainset = resolve_dataset(ws, trainset_json, trainset_path, input_keys, required=True)
        valset = resolve_dataset(ws, valset_json, valset_path, input_keys, required=False)

        # Create the adapter
        adapter = CodeEvolverDSPyAdapter(
            workspace_path=workspace_path,
            program=program,
            metric_fn=metric_fn,
            saved_program_json_path=saved_program_json_path,
            num_threads=num_threads,
        )

        # Build seed candidate from program.json
        seed_candidate = adapter.build_seed_candidate()

        # Create callback progress tracker (also handles cancellation)
        tracker = CallbackProgressTracker(callback_url, jwt_token, job_id)

        # Run GEPA optimization (synchronous, blocking)
        result: GEPAResult = gepa_optimize(
            seed_candidate=seed_candidate,
            trainset=trainset,
            valset=valset,
            adapter=adapter,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            stop_callbacks=[tracker],
            seed=seed,
            raise_on_exception=False,
            display_progress_bar=True,
        )

        # Build result dict
        best_idx = result.best_idx
        result_dict = {
            "best_candidate": result.best_candidate,
            "best_score": result.val_aggregate_scores[best_idx],
            "best_idx": best_idx,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
        }

        # Persist final results via callback
        updater.set_completed(
            best_candidate=result.best_candidate,
            best_score=result.val_aggregate_scores[best_idx],
            total_metric_calls=result.total_metric_calls,
            num_candidates=result.num_candidates,
        )

        return result_dict

    except Exception as e:
        # Update job status to failed via callback
        updater.set_failed(str(e))
        raise
