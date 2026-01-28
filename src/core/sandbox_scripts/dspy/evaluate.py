"""Evaluate command for DSPy programs.

Runs a DSPy program on a batch of examples, applying candidate instructions
and scoring with the provided metric. Optionally captures execution traces
for reflective dataset building.
"""

from . import build_program, load_import_path, signature_key


def handle(cmd: dict, workspace: str) -> dict:
    """Run evaluation: load program, apply candidate, score batch.

    Args:
        cmd: Command dict with keys:
            - program: Dotted import path to DSPy module class
            - metric: Dotted import path to metric function
            - saved_program_json_path: Optional path to program.json
            - candidate: Dict of predictor name -> instruction text
            - batch: List of example dicts
            - capture_traces: Whether to capture DSPy traces
            - num_threads: Parallelism for evaluation
            - input_keys: Fields to mark as inputs on examples
            - failure_score: Score to use on failures
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success', 'outputs', 'scores', and optionally 'trajectories'
    """
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
    program = build_program(workspace, program_path, saved_json, candidate)

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

            sig_key = signature_key(pred_module.signature)

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
