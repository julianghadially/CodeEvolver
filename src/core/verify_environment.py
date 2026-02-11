"""
Copyright © 2026 440 Labs LLC

Sandbox environment validation for CodeEvolver.

Validates that the sandbox environment can successfully execute client code
by running evaluation on a small subset of data. This catches environment issues
(missing dependencies, import errors, configuration problems) early before
starting the full optimization.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def validate_sandbox_environment(
    sandbox: Any,
    program: str,
    metric: str,
    batch: list[dict],
    seed_candidate: dict[str, str],
    saved_program_json_path: str | None = None,
    program_lm: str = "openai/gpt-5-mini",
    num_threads: int = 1,
    input_keys: list[str] | None = None,
    failure_score: float = 0.0,
    git_branch: str | None = None,
    max_validation_rows: int = 15,
    error_threshold: float = 0.05,
    capture_traces: bool = True,
) -> dict[str, Any]:
    """Validate sandbox environment by running evaluation on a small subset.

    This early validation helps detect environment issues (missing dependencies,
    import errors, configuration problems) before starting the full optimization.
    Captures traces for detailed error diagnostics.

    Args:
        sandbox: GEPASandbox instance (must be started).
        program: Dotted import path to DSPy module class.
        metric: Dotted import path to metric function.
        batch: Full training dataset.
        seed_candidate: Seed candidate dict with prompt texts (excludes _code).
        saved_program_json_path: Optional path to program.json.
        program_lm: LM model string for DSPy.
        num_threads: Parallelism for evaluation.
        input_keys: Fields to mark as inputs on examples.
        failure_score: Score to assign on evaluation failure.
        git_branch: Optional git branch to checkout.
        max_validation_rows: Maximum number of rows to validate (default 15).
        error_threshold: Maximum allowed error rate (default 0.05 = 5%).
        capture_traces: Whether to capture execution traces (default True).

    Returns:
        Dict with 'success' (bool), 'error' (str) if failed, 'error_details' (list),
        and 'error_rate' (float).
    """
    logger.info(f"Starting sandbox validation on {min(len(batch), max_validation_rows)} rows...")
    print(
        f"[VALIDATION] Testing {min(len(batch), max_validation_rows)} rows, "
        f"threshold: {error_threshold:.1%}, capture_traces: {capture_traces}",
        flush=True
    )

    # Take first N rows for validation
    validation_batch = batch[:max_validation_rows]

    try:
        # Run evaluation via exec_prebuilt
        result = sandbox.exec_prebuilt({
            "command": "evaluate",
            "program": program,
            "metric": metric,
            "saved_program_json_path": saved_program_json_path,
            "candidate": seed_candidate,
            "batch": validation_batch,
            "capture_traces": capture_traces,
            "num_threads": num_threads,
            "input_keys": input_keys or [],
            "failure_score": failure_score,
            "program_lm": program_lm,
            "git_branch": git_branch,
        })

        if not result.get("success", False):
            error_msg = (
                f"Sandbox validation failed during evaluation: {result.get('error', 'unknown')}\n\n"
                f"This indicates a critical environment issue that prevents evaluation."
            )
            if result.get("traceback"):
                error_msg += f"\n\nTraceback:\n{result.get('traceback')}"
            return {"success": False, "error": error_msg, "error_rate": 1.0}

        # Count errors (scores at failure_score indicate errors)
        scores = result.get("scores", [])
        outputs = result.get("outputs", [])
        trajectories = result.get("trajectories", [])

        error_count = sum(1 for score in scores if score == failure_score)
        error_rate = error_count / len(validation_batch) if validation_batch else 0.0

        logger.info(
            f"Validation results: {error_count}/{len(validation_batch)} errors "
            f"({error_rate:.1%}), threshold: {error_threshold:.1%}"
        )
        print(
            f"[VALIDATION] Results: {error_count}/{len(validation_batch)} errors "
            f"({error_rate:.1%})",
            flush=True
        )

        if error_rate > error_threshold:
            # Collect detailed error information including traces
            error_details = _collect_error_details(
                validation_batch, outputs, scores, trajectories, failure_score
            )

            # Build detailed error message
            error_msg = _build_error_message(
                error_count, len(validation_batch), error_rate, error_threshold, error_details
            )

            return {
                "success": False,
                "error": error_msg,
                "error_details": error_details,
                "error_rate": error_rate,
            }

        logger.info("Sandbox validation passed!")
        print("[VALIDATION] Passed!", flush=True)
        return {"success": True, "error_rate": error_rate}

    except Exception as e:
        error_msg = (
            f"Sandbox validation failed with exception: {str(e)}\n\n"
            f"This indicates a critical environment issue that prevents evaluation."
        )
        logger.error(f"Validation exception: {e}", exc_info=True)
        return {"success": False, "error": error_msg, "error_rate": 1.0}


def _collect_error_details(
    validation_batch: list[dict],
    outputs: list,
    scores: list[float],
    trajectories: list | None,
    failure_score: float,
) -> list[dict]:
    """Collect detailed error information for failed rows.

    Args:
        validation_batch: Input data batch.
        outputs: Evaluation outputs.
        scores: Evaluation scores.
        trajectories: Optional execution traces.
        failure_score: Score indicating failure.

    Returns:
        List of error detail dicts.
    """
    error_details = []

    for i, (output, score) in enumerate(zip(outputs, scores)):
        if score == failure_score:
            error_detail = {
                "row_index": i,
                "input": validation_batch[i],
                "output": output,
                "score": score,
            }

            # Add trajectory/trace information if available
            if trajectories and i < len(trajectories) and trajectories[i]:
                traj = trajectories[i]
                if isinstance(traj, dict):
                    # Extract trace information for debugging
                    trace = traj.get("trace", [])
                    error_detail["trace_length"] = len(trace)

                    # Look for failed predictions or exceptions in trace
                    failed_steps = []
                    for step_idx, step in enumerate(trace):
                        if isinstance(step, dict):
                            step_output = step.get("output", {})
                            if isinstance(step_output, dict) and step_output.get("__failed__"):
                                failed_steps.append({
                                    "step": step_idx,
                                    "signature": step.get("signature_key", "unknown"),
                                    "error": step_output.get("completion_text", ""),
                                })

                    if failed_steps:
                        error_detail["failed_steps"] = failed_steps

            error_details.append(error_detail)

    return error_details


def _build_error_message(
    error_count: int,
    total_count: int,
    error_rate: float,
    error_threshold: float,
    error_details: list[dict],
) -> str:
    """Build a detailed error message for validation failures.

    Args:
        error_count: Number of failed rows.
        total_count: Total number of validation rows.
        error_rate: Calculated error rate.
        error_threshold: Configured error threshold.
        error_details: List of error detail dicts.

    Returns:
        Formatted error message string.
    """
    error_msg = (
        f"Sandbox validation failed: {error_count}/{total_count} rows "
        f"({error_rate:.1%}) had errors (threshold: {error_threshold:.1%}).\n\n"
        f"This indicates the sandbox environment may be missing dependencies, "
        f"have import errors, or configuration issues.\n\n"
        f"Error details:\n"
    )

    # Show first 5 errors with trace information
    for detail in error_details[:5]:
        error_msg += f"\n--- Row {detail['row_index']} ---\n"
        error_msg += f"Input: {str(detail['input'])[:200]}\n"
        error_msg += f"Output: {str(detail['output'])[:500]}\n"
        error_msg += f"Score: {detail['score']}\n"

        if detail.get("trace_length"):
            error_msg += f"Trace length: {detail['trace_length']} steps\n"

        if detail.get("failed_steps"):
            error_msg += "Failed steps:\n"
            for step in detail["failed_steps"]:
                error_msg += f"  - Step {step['step']} ({step['signature']}): {step['error'][:200]}\n"

    if len(error_details) > 5:
        error_msg += f"\n... and {len(error_details) - 5} more errors"

    return error_msg
