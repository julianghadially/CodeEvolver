"""Make reflective dataset command for DSPy programs.

Builds a reflective dataset from serialized execution traces,
matching trace entries to predictors using signature fingerprinting.
"""

import random
from typing import Any

from . import build_program, signature_key


def handle(cmd: dict, workspace: str) -> dict:
    """Build reflective dataset from serialized traces.

    Matches trace entries to predictors using signature_key fingerprinting
    (sorted input+output field names) instead of live DSPy object comparison.

    Args:
        cmd: Command dict with keys:
            - program: Dotted import path to DSPy module class
            - saved_program_json_path: Optional path to program.json
            - candidate: Dict of predictor name -> instruction text
            - trajectories: List of serialized trace data
            - scores: List of scores for each trajectory
            - components_to_update: List of predictor names to include
            - failure_score: Score threshold for failures
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success' and 'reflective_dataset' (predictor name -> examples)
    """
    program_path = cmd["program"]
    saved_json = cmd.get("saved_program_json_path")
    candidate = cmd.get("candidate", {})
    trajectories = cmd.get("trajectories", [])
    scores = cmd.get("scores", [])
    components_to_update = cmd.get("components_to_update", [])
    failure_score = cmd.get("failure_score", 0.0)

    # Build program with candidate instructions to get predictor signatures
    program = build_program(workspace, program_path, saved_json, candidate)

    # Build signature_key -> predictor_name mapping
    pred_sig_keys: dict[str, list[str]] = {}
    for name, pred in program.named_predictors():
        key = signature_key(pred.signature)
        pred_sig_keys.setdefault(key, []).append(name)

    rng = random.Random(0)

    ret_d: dict[str, list[dict[str, Any]]] = {}
    for pred_name in components_to_update:
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
        return {"success": False, "error": "No valid predictions found for any module."}

    return {"success": True, "reflective_dataset": ret_d}
