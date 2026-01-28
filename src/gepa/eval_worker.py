"""Eval worker that runs inside the Modal sandbox.

Standalone script. Reads a JSON command file, dispatches to the appropriate
handler, and prints EVAL_RESULT:{json} to stdout.

Commands:
  - build_seed_candidate: Extract initial instructions from DSPy program
  - evaluate: Run DSPy program on batch, return scores/outputs/traces
  - make_reflective_dataset: Build reflective dataset from serialized traces

This file is copied into the sandbox image and executed via sandbox.exec().
It has access to the client's full Python environment (dspy, their code, etc.).
"""

import argparse
import importlib
import json
import os
import random
import sys
import traceback
from pathlib import Path
from typing import Any


def load_import_path(workspace_path: str, dotted_path: str) -> Any:
    """Import an attribute from a dotted path relative to the workspace."""
    ws = str(workspace_path)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    module_path, attr_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def _build_program(workspace_path: str, program_path: str,
                   saved_program_json_path: str | None,
                   candidate: dict[str, str] | None = None):
    """Instantiate DSPy module, load program.json, optionally apply candidate instructions."""
    import dspy

    cls = load_import_path(workspace_path, program_path)
    program = cls()

    if saved_program_json_path:
        full_path = Path(workspace_path) / saved_program_json_path
        if full_path.exists():
            program.load(str(full_path))

    if candidate:
        for name, pred in program.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

    return program


def _signature_key(sig) -> str:
    """Build a fingerprint key from a DSPy signature's field names.

    Used to match serialized trace entries to predictors across the
    process boundary.
    """
    input_fields = sorted(sig.input_fields.keys())
    output_fields = sorted(sig.output_fields.keys())
    return f"{','.join(input_fields)}->{','.join(output_fields)}"


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def handle_build_seed_candidate(cmd: dict, workspace: str) -> dict:
    """Extract initial instructions from the DSPy program."""
    program_path = cmd["program"]
    saved_json = cmd.get("saved_program_json_path")

    program = _build_program(workspace, program_path, saved_json)

    candidate = {}
    for name, pred in program.named_predictors():
        candidate[name] = pred.signature.instructions

    return {"success": True, "candidate": candidate}


def handle_evaluate(cmd: dict, workspace: str) -> dict:
    """Run evaluation: load program, apply candidate, score batch."""
    import dspy
    from dspy.primitives import Example

    program_path = cmd["program"]
    metric_path = cmd["metric"]
    saved_json = cmd.get("saved_program_json_path")
    candidate = cmd.get("candidate", {})
    batch = cmd.get("batch", [])
    capture_traces = cmd.get("capture_traces", False)
    num_threads = cmd.get("num_threads", 1)
    input_keys = cmd.get("input_keys", [])
    failure_score = cmd.get("failure_score", 0.0)

    # Load metric function
    metric_fn = load_import_path(workspace, metric_path)

    # Build program with candidate instructions
    program = _build_program(workspace, program_path, saved_json, candidate)

    # Convert batch dicts to dspy.Example
    examples = []
    for item in batch:
        ex = Example(**item)
        if input_keys:
            ex = ex.with_inputs(*input_keys)
        examples.append(ex)

    if capture_traces:
        return _evaluate_with_traces(
            program, metric_fn, examples, failure_score, num_threads
        )
    else:
        return _evaluate_simple(
            program, metric_fn, examples, failure_score, num_threads
        )


def _evaluate_simple(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate using dspy.Evaluate — returns outputs + scores, no traces."""
    import dspy

    evaluator = dspy.Evaluate(
        devset=examples,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=False,
        return_all_scores=True,
        return_outputs=True,
    )

    result = evaluator(program)

    # dspy.Evaluate returns (aggregate_score, results_list)
    # results_list is [(example, prediction, score), ...]
    aggregate_score, results_list = result

    outputs = []
    scores = []
    for ex, pred, score in results_list:
        if pred is not None:
            outputs.append(dict(pred))
        else:
            outputs.append(None)
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(failure_score)

    return {"success": True, "outputs": outputs, "scores": scores}


def _evaluate_with_traces(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate with trace capture for reflective dataset building."""
    try:
        from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data
    except ImportError:
        # Fallback if bootstrap_trace_data not available in this dspy version
        result = _evaluate_simple(program, metric_fn, examples, failure_score, num_threads)
        result["trajectories"] = []
        return result

    trace_data = bootstrap_trace_data(
        program=program,
        dataset=examples,
        metric=metric_fn,
        num_threads=num_threads,
    )

    # Serialize trace data for cross-process transfer
    trajectories = []
    scores = []
    outputs = []

    for data in trace_data:
        if data is None:
            trajectories.append(None)
            scores.append(failure_score)
            outputs.append(None)
            continue

        trace = data.get("trace", [])
        example = data.get("example")
        prediction = data.get("prediction")
        score = data.get("score")

        # Serialize score
        try:
            score_val = float(score) if score is not None else failure_score
        except (TypeError, ValueError):
            if isinstance(score, dict):
                score_val = float(score.get("score", failure_score))
            elif hasattr(score, "score"):
                score_val = float(getattr(score, "score", failure_score))
            else:
                score_val = failure_score
        scores.append(score_val)

        # Serialize prediction
        if prediction is not None:
            try:
                # Check for FailedPrediction
                from dspy.teleprompt.bootstrap_trace import FailedPrediction
                if isinstance(prediction, FailedPrediction):
                    outputs.append({"__failed__": True, "completion_text": getattr(prediction, "completion_text", "")})
                else:
                    outputs.append(dict(prediction))
            except ImportError:
                outputs.append(dict(prediction))
        else:
            outputs.append(None)

        # Serialize trace entries
        serialized_trace = []
        for entry in trace:
            # entry is (predictor, inputs, output) tuple
            pred_module, inputs, output = entry

            sig_key = _signature_key(pred_module.signature)

            # Serialize inputs
            ser_inputs = {k: str(v) for k, v in inputs.items()}

            # Serialize output
            try:
                from dspy.teleprompt.bootstrap_trace import FailedPrediction
                if isinstance(output, FailedPrediction):
                    ser_output = {"__failed__": True, "completion_text": getattr(output, "completion_text", "")}
                else:
                    ser_output = {k: str(v) for k, v in output.items()}
            except ImportError:
                ser_output = {k: str(v) for k, v in output.items()}

            serialized_trace.append({
                "signature_key": sig_key,
                "inputs": ser_inputs,
                "output": ser_output,
            })

        # Serialize example
        ser_example = dict(example) if example is not None else None

        trajectories.append({
            "trace": serialized_trace,
            "example": ser_example,
            "prediction": outputs[-1],
            "score": score_val,
        })

    return {
        "success": True,
        "outputs": outputs,
        "scores": scores,
        "trajectories": trajectories,
    }


def handle_make_reflective_dataset(cmd: dict, workspace: str) -> dict:
    """Build reflective dataset from serialized traces.

    Matches trace entries to predictors using signature_key fingerprinting
    (sorted input+output field names) instead of live DSPy object comparison.
    """
    program_path = cmd["program"]
    saved_json = cmd.get("saved_program_json_path")
    candidate = cmd.get("candidate", {})
    trajectories = cmd.get("trajectories", [])
    scores = cmd.get("scores", [])
    components_to_update = cmd.get("components_to_update", [])
    failure_score = cmd.get("failure_score", 0.0)

    # Build program with candidate instructions to get predictor signatures
    program = _build_program(workspace, program_path, saved_json, candidate)

    # Build signature_key -> predictor_name mapping
    pred_sig_keys: dict[str, list[str]] = {}
    for name, pred in program.named_predictors():
        key = _signature_key(pred.signature)
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

        target_key = _signature_key(module.signature)

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


# ---------------------------------------------------------------------------
# Main dispatcher
# ---------------------------------------------------------------------------

HANDLERS = {
    "build_seed_candidate": handle_build_seed_candidate,
    "evaluate": handle_evaluate,
    "make_reflective_dataset": handle_make_reflective_dataset,
}


def main():
    parser = argparse.ArgumentParser(description="GEPA eval worker")
    parser.add_argument("--workspace", required=True, help="Path to cloned repo")
    parser.add_argument("--command-file", required=True, help="Path to JSON command file")
    args = parser.parse_args()

    # Read command file
    with open(args.command_file) as f:
        cmd = json.load(f)

    command_name = cmd.get("command")
    handler = HANDLERS.get(command_name)
    if handler is None:
        result = {"success": False, "error": f"Unknown command: {command_name}"}
    else:
        try:
            result = handler(cmd, args.workspace)
        except Exception as e:
            result = {
                "success": False,
                "error": f"{type(e).__name__}: {e}",
                "traceback": traceback.format_exc(),
            }

    # Output result on a single line with known prefix for parsing
    print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
    main()
