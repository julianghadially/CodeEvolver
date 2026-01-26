"""GEPA optimization orchestrator for CodeEvolver.

This module is called from the Modal function to set up and run the
GEPA optimization loop. It loads user code, creates the adapter,
and calls gepa.optimize().
"""

import importlib
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import dspy
from dspy.primitives import Example

from gepa import optimize as gepa_optimize
from gepa.core.result import GEPAResult

from .adapter import CodeEvolverDSPyAdapter
from .progress import MongoDBProgressTracker


def load_metric_function(
    workspace_path: str,
    metric_path: str,
    metric_fn_name: str,
) -> Callable:
    """Dynamically load the user's metric function from their repository.

    Args:
        workspace_path: Path to the cloned repo root.
        metric_path: Relative path to metric module (e.g., "eval/metric.py").
        metric_fn_name: Function name within the module (e.g., "accuracy").

    Returns:
        The metric function callable.
    """
    ws = str(workspace_path)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    # Convert file path to Python module path
    module_path = metric_path
    if module_path.endswith(".py"):
        module_path = module_path[:-3]
    module_path = module_path.replace("/", ".").replace("\\", ".")

    mod = importlib.import_module(module_path)
    return getattr(mod, metric_fn_name)


def load_dataset(
    dataset_json: list[dict[str, Any]],
    input_keys: list[str] | None = None,
) -> list[Example]:
    """Convert JSON dataset to dspy.Example objects.

    Args:
        dataset_json: List of dicts, each representing one example.
        input_keys: Explicit input field names. If provided, these fields
            are marked as inputs via with_inputs(). If None, all keys are
            treated as both inputs and labels (DSPy default).

    Returns:
        List of dspy.Example objects.
    """
    examples = []
    for item in dataset_json:
        ex = Example(**item)
        if input_keys:
            ex = ex.with_inputs(*input_keys)
        examples.append(ex)
    return examples


def run_gepa_optimization(
    job_id: str,
    workspace_path: str,
    program_json_path: str,
    entry_point: str,
    metric_path: str,
    metric_fn_name: str,
    trainset_json: list[dict[str, Any]],
    valset_json: list[dict[str, Any]] | None,
    task_lm: str,
    reflection_lm: str,
    max_metric_calls: int,
    mongodb_url: str,
    database_name: str,
    input_keys: list[str] | None = None,
    num_threads: int = 1,
    seed: int = 0,
) -> dict[str, Any]:
    """Run GEPA optimization. Called from the Modal function.

    This is synchronous — GEPA's optimize() is a blocking call.

    Args:
        job_id: Unique job identifier.
        workspace_path: Path to cloned user repo.
        program_json_path: Relative path to program.json.
        entry_point: DSPy module class path (e.g., "fire.FIREJudge").
        metric_path: Relative path to metric module.
        metric_fn_name: Metric function name.
        trainset_json: Training data as list of dicts.
        valset_json: Validation data (optional, defaults to trainset).
        task_lm: LM for running the DSPy program.
        reflection_lm: LM for GEPA reflection.
        max_metric_calls: Budget for optimization.
        mongodb_url: MongoDB connection string.
        database_name: MongoDB database name.
        input_keys: Optional explicit list of input field names.
        num_threads: Parallelism for DSPy Evaluate.
        seed: Random seed for reproducibility.

    Returns:
        Dict with optimization results.
    """
    from pymongo import MongoClient

    # Update job status to running
    client = MongoClient(mongodb_url)
    db = client[database_name]
    db.jobs.update_one(
        {"job_id": job_id},
        {"$set": {
            "status": "running",
            "started_at": datetime.now(timezone.utc),
            "updated_at": datetime.now(timezone.utc),
        }},
    )
    client.close()

    tracker = None
    try:
        # Load user's metric function
        metric_fn = load_metric_function(workspace_path, metric_path, metric_fn_name)

        # Convert datasets to DSPy Examples
        trainset = load_dataset(trainset_json, input_keys)
        valset = load_dataset(valset_json, input_keys) if valset_json else None

        # Create the adapter
        adapter = CodeEvolverDSPyAdapter(
            workspace_path=workspace_path,
            program_json_path=program_json_path,
            entry_point=entry_point,
            metric_fn=metric_fn,
            task_lm=task_lm,
            num_threads=num_threads,
        )

        # Build seed candidate from program.json
        seed_candidate = adapter.build_seed_candidate()

        # Create MongoDB progress tracker (also handles cancellation)
        tracker = MongoDBProgressTracker(
            mongodb_url=mongodb_url,
            database_name=database_name,
            job_id=job_id,
        )

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

        # Persist final results to MongoDB
        client = MongoClient(mongodb_url)
        db = client[database_name]
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "completed",
                "best_candidate": result.best_candidate,
                "best_score": result.val_aggregate_scores[best_idx],
                "total_metric_calls": result.total_metric_calls,
                "num_candidates": result.num_candidates,
                "completed_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        client.close()

        return result_dict

    except Exception as e:
        # Update job status to failed
        client = MongoClient(mongodb_url)
        db = client[database_name]
        db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": "failed",
                "error": str(e),
                "updated_at": datetime.now(timezone.utc),
            }},
        )
        client.close()
        raise

    finally:
        if tracker:
            tracker.close()
