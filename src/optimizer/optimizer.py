"""
Copyright © 2026 440 Labs LLC

GEPA optimization orchestrator for CodeEvolver.

This module is called from the Modal function to set up and run the
GEPA optimization loop. It creates the adapter (which proxies to the
eval sandbox) and calls gepa.optimize().

No dspy is imported here — all DSPy operations happen inside the sandbox.
"""

import json
import time
from pathlib import Path
from typing import Any

from gepa import optimize as gepa_optimize
from gepa.core.result import GEPAResult
from gepa.strategies.batch_sampler import EpochShuffledBatchSampler

from .adapter import CodeEvolverDSPyAdapter
from .callback import CallbackJobUpdater, CallbackProgressTracker
from .component_selector import CodeFrequencyComponentSelector
from .utils import (
    extract_run_timestamp_from_branch,
    get_git_branch_from_candidate,
    load_dataset_from_file,
    save_file_to_sandbox,
    subsample_validation_set,
)


def _save_best_candidate_artifact(
    sandbox_manager: Any,
    best_candidate: dict[str, str],
) -> None:
    """Save the best candidate JSON to the winning branch.

    Creates codeevolver/results/best_program_{timestamp}.json on the winning
    branch. The timestamp is extracted from the branch name for consistency
    across the run.

    Args:
        sandbox_manager: GEPASandbox instance (already started).
        best_candidate: The winning candidate dict.
    """
    try:
        import json

        # Get winning branch and run timestamp
        winning_branch = get_git_branch_from_candidate(best_candidate)
        run_timestamp = extract_run_timestamp_from_branch(winning_branch)

        if not run_timestamp:
            print(f"[OPTIMIZER] Warning: Could not extract timestamp from {winning_branch}", flush=True)
            run_timestamp = "unknown"

        # Checkout winning branch
        checkout_result = sandbox_manager.exec_bash(f"git checkout {winning_branch}")
        if checkout_result.get("returncode") != 0:
            print(
                f"[OPTIMIZER] Warning: Failed to checkout {winning_branch}: "
                f"{checkout_result.get('stderr')}",
                flush=True,
            )
            return

        # Create results directory
        sandbox_manager.exec_bash("mkdir -p codeevolver/results")

        # Format candidate as pretty JSON
        candidate_json = json.dumps(best_candidate, indent=2)

        # Save to file
        artifact_path = f"codeevolver/results/best_program_{run_timestamp}.json"
        
        save_file_to_sandbox(
            sandbox=sandbox_manager,
            content=candidate_json,
            path=artifact_path,
            push=True,
            commit_message=f"Save best candidate (score: optimized)",
            branch=winning_branch,
        )
        print(f"[OPTIMIZER] Saved best candidate to {artifact_path}", flush=True)
    except Exception as e:
        print(f"[OPTIMIZER] Warning: Failed to save artifact: {e}", flush=True)


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
    initial: int = 1,
    decay_rate: int = 25,
    decay_factor: int = 2,
    code_cutoff_step: int | None = None,
    code_lm: str = "anthropic/claude-sonnet-4-5-20250929",
    subsample_size: int = 10,
    initial_branch: str = "main",
    max_valset_size: int | None = None,
    debug: bool = False,
    debug_max_iterations: int | None = None,
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
        initial: Starting prompts per code (default: 1). The ratio represents
            prompts per code change. Uses CodeFrequencyComponentSelector by default.
        decay_rate: Iterations between each multiplier step (default: 25).
        decay_factor: Multiplier applied at each decay step (default: 2).
        code_cutoff_step: Stop code mutations after this iteration. Default is None (no cutoff).
        code_lm: Language model for code mutations. Default is Claude Sonnet 4.5.
        subsample_size: Number of examples per evaluation batch. Default is 10.
        initial_branch: Git branch to use as starting point for optimization. Default is "main".
        max_valset_size: Maximum size of validation set. If specified, randomly subsamples
            the validation set to this size using the provided seed. The same subsample
            is used throughout the optimization run. Default is None (use full validation set).

    Returns:
        Dict with optimization results.
    """
    ws = Path(workspace_path)
    updater = CallbackJobUpdater(callback_url, jwt_token, job_id)
    optimization_start = time.time()

    # Update job status to running
    updater.set_running()
    print(f"[TIMER] Starting optimization run", flush=True)

    try:
        # Resolve datasets as raw dicts (no dspy.Example — conversion happens in sandbox)
        dataset_start = time.time()
        trainset = _resolve_dataset_raw(ws, trainset_json, trainset_path, required=True)
        valset = _resolve_dataset_raw(ws, valset_json, valset_path, required=False)

        # Subsample validation set if max_valset_size is specified
        valset = subsample_validation_set(valset, max_valset_size, seed)

        print(f"[TIMER] Dataset loading took {time.time() - dataset_start:.2f}s", flush=True)

        # Create the adapter (RPC proxy to sandbox)
        adapter_start = time.time()
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
            initial_branch=initial_branch,
        )
        print(f"[TIMER] Adapter creation took {time.time() - adapter_start:.2f}s", flush=True)

        # Build seed candidate from program.json (via sandbox)
        seed_start = time.time()
        print(f"[TIMER] Starting: build_seed_candidate", flush=True)
        seed_candidate = adapter.build_seed_candidate()
        print(f"[TIMER] build_seed_candidate took {time.time() - seed_start:.2f}s", flush=True)

        # Validate sandbox environment with a small subset of data
        # This catches environment issues (missing deps, import errors) early
        # Extract prompt texts from seed_candidate (exclude _code key)
        from .utils import CODE_COMPONENT_KEY
        prompt_texts = {k: v for k, v in seed_candidate.items() if k != CODE_COMPONENT_KEY}
        git_branch = json.loads(seed_candidate.get(CODE_COMPONENT_KEY, "{}")).get("git_branch")

        validation_start = time.time()
        print(f"[TIMER] Starting: sandbox environment validation", flush=True)
        validation_result = sandbox_manager.validate_environment(
            program=program,
            metric=metric,
            batch=trainset,
            seed_candidate=prompt_texts,
            saved_program_json_path=saved_program_json_path,
            program_lm=program_lm,
            num_threads=num_threads,
            input_keys=input_keys,
            failure_score=adapter.failure_score,
            git_branch=git_branch,
            max_validation_rows=15,
            error_threshold=0.05,
            capture_traces=True,  # Enable traces for detailed error diagnostics
        )
        print(f"[TIMER] Sandbox validation took {time.time() - validation_start:.2f}s", flush=True)

        if not validation_result.get("success"):
            error_msg = validation_result.get("error", "Unknown validation error")
            print(f"[OPTIMIZER] Validation failed:\n{error_msg}", flush=True)
            updater.set_failed(error_msg)
            raise RuntimeError(f"Sandbox validation failed:\n{error_msg}")

        # Create callback progress tracker (also handles cancellation)
        tracker = CallbackProgressTracker(
            callback_url,
            jwt_token,
            job_id,
            debug_max_iterations=debug_max_iterations,
        )

        if debug_max_iterations:
            print(f"\n[DEBUG MODE] Will stop after {debug_max_iterations} iterations\n", flush=True)

        # Use CodeFrequencyComponentSelector by default
        # This controls the ratio of code mutations vs prompt mutations
        effective_module_selector = CodeFrequencyComponentSelector(
            initial=initial,
            decay_rate=decay_rate,
            decay_factor=decay_factor,
            code_cutoff_step=code_cutoff_step,
        )

        # Create batch sampler with specified minibatch size
        import random
        batch_sampler = EpochShuffledBatchSampler(
            minibatch_size=subsample_size,
            rng=random.Random(seed),
        )

        # Run GEPA optimization (synchronous, blocking)
        # Note: Per-iteration timing is inside the GEPA package.
        # We log evaluate() and propose_new_texts() timing in adapter.py
        gepa_start = time.time()
        print(f"[TIMER] Starting: gepa_optimize (main loop)", flush=True)
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
            batch_sampler=batch_sampler,
        )
        gepa_elapsed = time.time() - gepa_start
        print(f"[TIMER] gepa_optimize took {gepa_elapsed:.2f}s ({gepa_elapsed/60:.1f} minutes)", flush=True)

        # Build result dict
        best_idx = result.best_idx
        result_dict = {
            "best_candidate": result.best_candidate,
            "best_score": result.val_aggregate_scores[best_idx],
            "best_idx": best_idx,
            "num_candidates": result.num_candidates,
            "total_metric_calls": result.total_metric_calls,
            "all_candidates": result.candidates,
            "candidate_scores": result.val_aggregate_scores,
            "parent_programs": result.parents,
        }

        if debug:
            # DEBUG: Analyze candidate evolution before saving
            print("\n" + "="*80, flush=True)
            print("[DEBUG] Candidate evolution analysis:", flush=True)
            for idx in range(min(10, len(result.candidates))):
                candidate = result.candidates[idx]
                score = result.val_aggregate_scores[idx]
                is_best = (idx == best_idx)
                marker = " ← BEST" if is_best else ""

                # Count prompt length as proxy for complexity
                prompt_lengths = {k: len(v) for k, v in candidate.items() if k != "_code"}
                total_prompt_chars = sum(prompt_lengths.values())

                # Check if this is a code mutation
                has_code_change = False
                if "_code" in candidate:
                    import json as json_mod
                    code_data = json_mod.loads(candidate["_code"])
                    has_code_change = bool(code_data.get("change_request"))

                print(f"  Candidate {idx}: score={score:.4f}, prompts={total_prompt_chars}chars, "
                    f"code={'YES' if has_code_change else 'NO'}{marker}", flush=True)

                if idx < 3 or is_best:  # Show details for first 3 and best
                    for key in list(prompt_lengths.keys())[:2]:  # Show first 2 prompts
                        preview = candidate[key][:80].replace('\n', ' ')
                        print(f"      {key}: {preview}...", flush=True)
            print("="*80 + "\n", flush=True)

        # Save best candidate artifact to winning branch
        _save_best_candidate_artifact(
            sandbox_manager=sandbox_manager,
            best_candidate=result.best_candidate,
        )

        # Build a mock GEPAState for the callback (contains the essential data)
        from gepa.core.state import GEPAState
        # Create a minimal state object with the data we need
        class FinalGEPAState:
            """Minimal state container for callback serialization."""
            def __init__(self, result: GEPAResult):
                self.program_candidates = result.candidates
                self.program_full_scores_val_set = result.val_aggregate_scores
                self.parent_program_for_candidate = result.parents  # Note: singular form!
                self.i = result.num_candidates - 1  # Last iteration index
                self.total_num_evals = result.total_metric_calls or 0

        final_state = FinalGEPAState(result)
        
        if debug:
            # DEBUG: Print final state summary
            print("\n" + "="*80, flush=True)
            print("[DEBUG] Final GEPA state summary:", flush=True)
            print(f"  Total candidates: {len(result.candidates)}", flush=True)
            print(f"  Best candidate index: {best_idx}", flush=True)
            print(f"  Best score: {result.val_aggregate_scores[best_idx]:.4f}", flush=True)
            print(f"  All scores: {[f'{s:.3f}' for s in result.val_aggregate_scores[:10]]}{'...' if len(result.val_aggregate_scores) > 10 else ''}", flush=True)

            # Show best candidate prompts
            print(f"\n[DEBUG] Best candidate prompts:", flush=True)
            for key, value in result.best_candidate.items():
                if key != "_code":
                    preview = value[:150] if isinstance(value, str) else str(value)[:150]
                    print(f"  {key}: {preview}...", flush=True)

            # Show parent relationships for best candidate
            if result.parents and best_idx < len(result.parents):
                parents = result.parents[best_idx]
                print(f"\n[DEBUG] Best candidate parents: {parents}", flush=True)
                if parents:
                    for parent_idx in parents:
                        if parent_idx is not None and parent_idx < len(result.val_aggregate_scores):
                            parent_score = result.val_aggregate_scores[parent_idx]
                            print(f"    Parent {parent_idx}: score={parent_score:.4f}", flush=True)
            print("="*80 + "\n", flush=True)

        # Persist final results via callback
        updater.set_completed(
            best_candidate=result.best_candidate,
            best_score=result.val_aggregate_scores[best_idx],
            total_metric_calls=result.total_metric_calls,
            num_candidates=result.num_candidates,
            gepa_state=final_state,
        )

        total_elapsed = time.time() - optimization_start
        print(f"[TIMER] Total optimization run took {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)", flush=True)

        return result_dict

    except Exception as e:
        # Update job status to failed via callback
        total_elapsed = time.time() - optimization_start
        print(f"[TIMER] Optimization failed after {total_elapsed:.2f}s ({total_elapsed/60:.1f} minutes)", flush=True)
        updater.set_failed(str(e))
        raise
