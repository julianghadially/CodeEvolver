"""Legacy trace capture for DSPy versions < 3.0.0 (e.g., 2.6.13).

This module provides trace capture compatibility for older DSPy versions
that don't have bootstrap_trace_data but support dspy.context(trace=[]).

bootstrap_trace.py was added to DSPy in version 3.0.0 (August 2025).
For DSPy >= 3.0.0, use the modern bootstrap_trace_data implementation.
"""

import concurrent.futures
import sys

# Import dspy at module level (must be available in sandbox venv)
try:
    import dspy
    from dspy.primitives import Example as DspyExample
except ImportError as _import_err:
    print(f"[evaluate_legacy.py IMPORT ERROR] Cannot import dspy: {_import_err}", file=sys.stderr)
    raise

# Import from parent package
from ai_frameworks.mounted.dspy import signature_key
from sandbox.mounted.utils import get_logger, make_success_result

log = get_logger("evaluate_legacy")


def evaluate_with_traces_legacy(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate with trace capture for DSPy 2.6.x using dspy.context().

    This implementation is compatible with DSPy 2.6.x which doesn't have
    bootstrap_trace_data but supports native trace capture via dspy.context.

    Args:
        program: DSPy module to evaluate
        metric_fn: Metric function
        examples: List of dspy.Example objects
        failure_score: Score to use on failures
        num_threads: Number of parallel threads

    Returns:
        Dict with outputs, scores, and trajectories
    """
    log.info(f"[LEGACY] Running DSPy {dspy.__version__} trace capture on {len(examples)} examples...")

    def evaluate_single_example(example):
        """Evaluate one example and capture its trace."""
        try:
            # Capture trace using dspy.context
            with dspy.context(trace=[]):
                # Run program
                prediction = program(**example.inputs())

                # Get captured trace from dspy.settings
                trace = dspy.settings.trace if hasattr(dspy.settings, 'trace') else []

                # Score the prediction
                try:
                    score = metric_fn(example, prediction)
                except Exception as e:
                    log.warning(f"Metric failed: {e}")
                    score = failure_score
                    prediction = {"__failed__": True, "__error__": str(e)}

                return {
                    "example": example,
                    "prediction": prediction,
                    "trace": trace,
                    "score": score,
                }
        except Exception as e:
            log.warning(f"Example evaluation failed: {e}")
            import traceback
            tb = traceback.format_exc()
            return {
                "example": example,
                "prediction": {"__failed__": True, "__error__": str(e), "traceback": tb},
                "trace": [],
                "score": failure_score,
            }

    # Execute evaluations (parallel if num_threads > 1)
    if num_threads > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            trace_data = list(executor.map(evaluate_single_example, examples))
    else:
        trace_data = [evaluate_single_example(ex) for ex in examples]

    log.info(f"[LEGACY] Captured {len(trace_data)} trace results")

    # Serialize trace data for cross-process transfer
    trajectories = []
    scores = []
    outputs = []

    for idx, data in enumerate(trace_data):
        if data is None:
            trajectories.append(None)
            scores.append(failure_score)
            outputs.append(None)
            continue

        trace = data.get("trace", [])
        example = data.get("example")
        prediction = data.get("prediction")
        score = data.get("score", failure_score)

        if idx == 0:
            log.info(f"[LEGACY] First item: trace_len={len(trace)}, has_example={example is not None}, has_pred={prediction is not None}, score={score}")

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
            if isinstance(prediction, dict) and prediction.get("__failed__"):
                outputs.append(prediction)
            else:
                try:
                    outputs.append(dict(prediction))
                except (TypeError, ValueError):
                    outputs.append({"value": str(prediction)})
        else:
            outputs.append(None)

        # Serialize trace entries
        serialized_trace = []
        for entry_idx, entry in enumerate(trace):
            if idx == 0 and entry_idx == 0:
                log.info(f"[LEGACY] First trace entry: type={type(entry)}, len={len(entry) if hasattr(entry, '__len__') else 'N/A'}")

            # entry is (predictor, inputs, output) tuple
            try:
                pred_module, inputs, output = entry
            except (ValueError, TypeError):
                log.warning(f"[LEGACY] Malformed trace entry at idx={idx}, entry_idx={entry_idx}")
                continue

            sig_key = signature_key(pred_module.signature)

            # Serialize inputs
            ser_inputs = {k: str(v) for k, v in inputs.items()}

            # Serialize output
            if isinstance(output, dict) and output.get("__failed__"):
                ser_output = output
            else:
                try:
                    ser_output = {k: str(v) for k, v in output.items()} if hasattr(output, 'items') else {"value": str(output)}
                except (TypeError, ValueError):
                    ser_output = {"value": str(output)}

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

    log.info(f"[LEGACY] Evaluation complete: {len(outputs)} outputs, {len(trajectories)} trajectories")
    return make_success_result(
        {"outputs": outputs, "scores": scores, "trajectories": trajectories},
        logs=log.get_logs()
    )
