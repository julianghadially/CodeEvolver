"""
Copyright © 2026 440 Labs LLC

GEPA optimization orchestrator for CodeEvolver.

This module is called from the Modal function to set up and run the
GEPA optimization loop. It creates the adapter (which proxies to the
eval sandbox) and calls gepa.optimize().

No dspy is imported here — all DSPy operations happen inside the sandbox.
"""

from pathlib import Path
from typing import Any

from gepa import optimize as gepa_optimize
from gepa.core.result import GEPAResult

from .adapter import CodeEvolverDSPyAdapter
from .callback import CallbackJobUpdater, CallbackProgressTracker
from .component_selector import CodeFrequencyComponentSelector
from .utils import load_dataset_from_file


def _resolve_dataset_raw(
    workspace_path: Path,
    inline_data: list[dict[str, Any]] | None,
    file_path: str | None,
    required: bool = True,
) -> list[dict[str, Any]] | None:
    """Resolve a dataset as raw dicts (no dspy.Example conversion).

    Args:
        workspace_path: Root of the cloned repo.
        inline_data: Dataset provided inline in the API request.
        file_path: Path to a data file in the repo (relative).
        required: If True, raises when neither source is provided.

    Returns:
        List of plain dicts, or None if not required and not provided.
    """
    if inline_data is not None:
        return inline_data
    if file_path is not None:
        return load_dataset_from_file(workspace_path / file_path)
    if required:
        raise ValueError("No dataset: provide inline data or a file path")
    return None


def run_gepa_optimization(
    job_id: str,
    callback_url: str,
    jwt_token: str,
    workspace_path: str,
    program: str,
    metric: str,
    reflection_lm: str,
    max_metric_calls: int,
    sandbox_manager: Any,
    saved_program_json_path: str | None = None,
    trainset_json: list[dict[str, Any]] | None = None,
    trainset_path: str | None = None,
    valset_json: list[dict[str, Any]] | None = None,
    valset_path: str | None = None,
    input_keys: list[str] | None = None,
    num_threads: int = 1,
    seed: int = 0,
    program_lm: str = "openai/gpt-5-mini",
    additional_instructions: str | None = None,
    code_frequency: int | None = None,
    code_cutoff_step: int | None = None,
    code_lm: str = "anthropic/claude-sonnet-4-5-20250929",
    subsample_size: int = 5,
) -> dict[str, Any]:
    """Run GEPA optimization. Called from the Modal function.

    This is synchronous — GEPA's optimize() is a blocking call.

    Args:
        job_id: Unique job identifier.
        workspace_path: Path to cloned user repo (local to orchestrator).
        program: Dotted import path to DSPy module class.
        metric: Dotted import path to metric function.
        reflection_lm: LM for GEPA reflection.
        max_metric_calls: Budget for optimization.
        sandbox_manager: GEPASandbox instance (already started).
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
        additional_instructions: Client-provided guidance for GEPA optimization.
            May include constraints (off-limits changes), services (available APIs
            with keys in environment), and ideas for optimization.
        code_frequency: Number of code iterations per prompt iteration.
            If None (default), uses GEPA's default "round_robin" selector.
            If provided, uses CodeFrequencyComponentSelector:
            0=prompt only, 1=alternating (50%), 2=2 code per prompt (67%),
            3=3 code per prompt (75%).
        code_cutoff_step: Stop code mutations after this iteration. Only used
            when code_frequency is provided. Default is None (no cutoff).
        code_lm: Language model for code mutations. Default is Claude Sonnet 4.5.
        subsample_size: Number of examples per evaluation batch. Default is 5.

    Returns:
        Dict with optimization results.
    """
    ws = Path(workspace_path)
    updater = CallbackJobUpdater(callback_url, jwt_token, job_id)

    # Update job status to running
    updater.set_running()

    try:
        # Resolve datasets as raw dicts (no dspy.Example — conversion happens in sandbox)
        trainset = _resolve_dataset_raw(ws, trainset_json, trainset_path, required=True)
        valset = _resolve_dataset_raw(ws, valset_json, valset_path, required=False)

        # Create the adapter (RPC proxy to sandbox)
        adapter = CodeEvolverDSPyAdapter(
            sandbox_manager=sandbox_manager,
            program=program,
            metric=metric,
            saved_program_json_path=saved_program_json_path,
            num_threads=num_threads,
            input_keys=input_keys,
            program_lm=program_lm,
            reflection_lm=reflection_lm,
            additional_instructions=additional_instructions,
            code_lm=code_lm,
        )

        # Build seed candidate from program.json (via sandbox)
        seed_candidate = adapter.build_seed_candidate()

        # Create callback progress tracker (also handles cancellation)
        tracker = CallbackProgressTracker(callback_url, jwt_token, job_id)

        # Determine module selector based on code_frequency
        # If code_frequency is provided, use CodeFrequencyComponentSelector
        # Otherwise use GEPA's default "round_robin"
        effective_module_selector: CodeFrequencyComponentSelector | str
        if code_frequency is not None:
            effective_module_selector = CodeFrequencyComponentSelector(
                code_frequency=code_frequency,
                code_cutoff_step=code_cutoff_step,
            )
        else:
            # Use GEPA's default round-robin selection
            effective_module_selector = "round_robin"

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
            module_selector=effective_module_selector,
            subsample_size=subsample_size,
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
