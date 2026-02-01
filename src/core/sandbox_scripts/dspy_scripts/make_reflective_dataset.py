"""Make reflective dataset command for DSPy programs.

Builds a reflective dataset from serialized execution traces,
matching trace entries to predictors using signature fingerprinting.
"""

import random
from typing import Any

from . import build_program, signature_key
from ..utils import get_logger, make_success_result

log = get_logger("reflective")

# Reserved key for code component (must match adapter.py)
CODE_COMPONENT_KEY = "_code"


def build_code_reflective_dataset(
    trajectories: list[dict],
    scores: list[float],
    failure_score: float,
) -> list[dict[str, Any]]:
    """Build feedback for code reflection.

    Includes both:
    - Low-scoring examples (accuracy issues)
    - Failed examples (code exceptions)

    The reflective LM has agency to decide what to prioritize.

    Args:
        trajectories: List of trajectory dicts with example, prediction, trace.
        scores: List of scores for each trajectory.
        failure_score: Score threshold for failures.

    Returns:
        List of feedback items with input, output, score, and optionally exception.
    """
    items = []

    for i, data in enumerate(trajectories):
        if data is None:
            continue

        score = scores[i] if i < len(scores) else failure_score
        example = data.get("example", {})
        prediction = data.get("prediction", {})

        # Include examples that are: low-scoring OR had code failures
        is_low_score = score < 1.0  # Include any imperfect score
        has_exception = prediction and isinstance(prediction, dict) and prediction.get("__failed__")

        if is_low_score or has_exception:
            item = {
                "input": example,
                "output": prediction,
                "score": score,
            }
            # Only include exception field if there was an actual code failure
            if has_exception:
                item["exception"] = prediction.get("__error__", "Unknown error")

            items.append(item)

    return items


def handle(cmd: dict, workspace: str) -> dict:
    """Build reflective dataset from serialized traces.

    Matches trace entries to predictors using signature_key fingerprinting
    (sorted input+output field names) instead of live DSPy object comparison.
    Special handling for _code component which gets overall system feedback.

    Args:
        cmd: Command dict with keys:
            - program: Dotted import path to DSPy module class
            - saved_program_json_path: Optional path to program.json
            - candidate: Dict of predictor name -> instruction text
            - trajectories: List of serialized trace data
            - scores: List of scores for each trajectory
            - components_to_update: List of predictor names to include (may include "_code")
            - failure_score: Score threshold for failures
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success' and 'reflective_dataset' (component name -> examples)
    """
    program_path = cmd["program"]
    saved_json = cmd.get("saved_program_json_path")
    candidate = cmd.get("candidate", {})
    trajectories = cmd.get("trajectories", [])
    scores = cmd.get("scores", [])
    components_to_update = cmd.get("components_to_update", [])
    failure_score = cmd.get("failure_score", 0.0)

    log.info(f"Building reflective dataset: {len(trajectories)} trajectories, components={components_to_update}")

    ret_d: dict[str, list[dict[str, Any]]] = {}

    # Handle _code component specially - it gets overall system feedback
    if CODE_COMPONENT_KEY in components_to_update:
        code_feedback = build_code_reflective_dataset(trajectories, scores, failure_score)
        if code_feedback:
            ret_d[CODE_COMPONENT_KEY] = code_feedback
        log.info(f"Built code reflective dataset with {len(code_feedback)} items")

    # Filter out _code from components that need DSPy predictor handling
    prompt_components = [c for c in components_to_update if c != CODE_COMPONENT_KEY]

    if not prompt_components:
        # Only _code was requested
        if ret_d:
            log.info(f"Built reflective dataset with {len(ret_d)} components")
            return make_success_result({"reflective_dataset": ret_d}, logs=log.get_logs())
        else:
            log.warning("No valid feedback found for code component")
            return {"success": False, "error": "No valid feedback found.", "logs": log.get_logs()}

    # Build program with candidate instructions to get predictor signatures
    program = build_program(workspace, program_path, saved_json, candidate)
    log.info(f"Built program: {type(program).__name__}")

    # Build signature_key -> predictor_name mapping
    pred_sig_keys: dict[str, list[str]] = {}
    for name, pred in program.named_predictors():
        key = signature_key(pred.signature)
        pred_sig_keys.setdefault(key, []).append(name)

    rng = random.Random(0)

    for pred_name in prompt_components:
        # Find the predictor module for this component
        module = None
        for name, m in program.named_predictors():
            if name == pred_name:
                module = m
                break
        if module is None:
            continue

        target_key = signature_key(module.signature)

        items: list[dict[str, Any]] = []
        for data in trajectories:
            if data is None:
                continue
            trace = data.get("trace", [])
            example = data.get("example")
            score = data.get("score", failure_score)

            # Find trace entries matching this predictor's signature
            matching = [t for t in trace
                        if t["signature_key"] == target_key
                        and not t["output"].get("__failed__", False)]
            failed = [t for t in trace
                      if t["signature_key"] == target_key
                      and t["output"].get("__failed__", False)]

            if not matching:
                if failed:
                    selected = failed[0]
                else:
                    continue
            else:
                # Check if the overall prediction was a failure
                prediction = data.get("prediction")
                if prediction and isinstance(prediction, dict) and prediction.get("__failed__", False):
                    continue
                selected = rng.choice(matching)

            new_inputs = selected["inputs"]
            pred_output = selected["output"]

            # Build output representation
            if pred_output.get("__failed__", False):
                new_outputs = (
                    "Couldn't parse the output as per the expected format. "
                    f"Raw response:\n```\n{pred_output.get('completion_text', '')}\n```"
                )
            else:
                new_outputs = pred_output

            # Build feedback from score and ground truth
            feedback_parts = [f"Score: {score}"]
            if example is not None:
                input_key_set = set(new_inputs.keys())
                for k, v in example.items():
                    if k not in input_key_set:
                        feedback_parts.append(f"Expected {k}: {v}")

            items.append({
                "Inputs": new_inputs,
                "Generated Outputs": new_outputs,
                "Feedback": "\n".join(feedback_parts),
            })

        if items:
            ret_d[pred_name] = items

    if not ret_d:
        log.warning("No valid predictions found for any module")
        return {"success": False, "error": "No valid predictions found for any module.", "logs": log.get_logs()}

    log.info(f"Built reflective dataset with {len(ret_d)} components")
    return make_success_result({"reflective_dataset": ret_d}, logs=log.get_logs())
