"""Evaluate command for DSPy programs.

Runs a DSPy program on a batch of examples, applying candidate instructions
and scoring with the provided metric. Optionally captures execution traces
for reflective dataset building.

Requires DSPy >= 3.0.0.
"""

from . import build_program, load_import_path, signature_key
from ..utils import get_logger, make_error_result, make_success_result

log = get_logger("evaluate")


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
            - program_lm: LM model string for DSPy (e.g., 'openai/gpt-4o-mini')
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
    capture_traces = cmd.get("capture_traces", True)
    num_threads = cmd.get("num_threads", 1)
    input_keys = cmd.get("input_keys", [])
    failure_score = cmd.get("failure_score", 0.0)
    program_lm = cmd.get("program_lm", "openai/gpt-5-mini")

    log.info(f"DSPy version: {dspy.__version__}")

    # Configure DSPy with the specified LM
    log.info(f"Configuring DSPy with LM: {program_lm}")
    lm = dspy.LM(program_lm)
    dspy.configure(lm=lm)
    log.info(f"Program: {program_path}, Metric: {metric_path}")
    log.info(f"Batch size: {len(batch)}, capture_traces: {capture_traces}, num_threads: {num_threads}")

    # Load metric function
    metric_fn = load_import_path(workspace, metric_path)
    log.info(f"Loaded metric: {metric_fn}")

    # Build program with candidate instructions
    program = build_program(workspace, program_path, saved_json, candidate)
    log.info(f"Built program: {type(program).__name__}")

    # Convert batch dicts to dspy.Example
    examples = []
    for item in batch:
        ex = Example(**item)
        if input_keys:
            ex = ex.with_inputs(*input_keys)
        examples.append(ex)
    log.info(f"Converted {len(examples)} examples")

    if capture_traces:
        log.info("Running evaluation with traces...")
        return _evaluate_with_traces(
            program, metric_fn, examples, failure_score, num_threads
        )
    else:
        log.info("Running simple evaluation...")
        return _evaluate_simple(
            program, metric_fn, examples, failure_score, num_threads
        )


def _evaluate_simple(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate using dspy.Evaluate — returns outputs + scores, no traces.

    Requires DSPy >= 3.0.0.
    """
    import dspy

    log.info("Creating dspy.Evaluate...")
    evaluator = dspy.Evaluate(
        devset=examples,
        metric=metric_fn,
        num_threads=num_threads,
        display_progress=False,
        return_all_scores=True,
    )

    log.info("Running evaluator...")
    result = evaluator(program)
    log.info(f"Evaluator returned: type={type(result)}, has_results={hasattr(result, 'results')}")

    # DSPy 3.0+: result.results is a list of (example, prediction, score) tuples
    results_list = result.results
    log.info(f"results_list: len={len(results_list)}")

    outputs = []
    scores = []
    for _example, pred, score in results_list:
        if pred is not None:
            outputs.append(dict(pred))
        else:
            outputs.append(None)
        try:
            scores.append(float(score))
        except (TypeError, ValueError):
            scores.append(failure_score)

    log.info(f"Simple evaluation complete: {len(outputs)} outputs")
    return make_success_result(
        {"outputs": outputs, "scores": scores},
        logs=log.get_logs()
    )


def _evaluate_with_traces(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate with trace capture for reflective dataset building.

    Note: Requires DSPy >= 2.6.0 with bootstrap_trace support.
    Traces are essential for GEPA's reflective mutation - failures will propagate.
    """
    from dspy.teleprompt.bootstrap_trace import bootstrap_trace_data

    log.info(f"Calling bootstrap_trace_data with {len(examples)} examples...")
    trace_data = bootstrap_trace_data(
        program=program,
        dataset=examples,
        metric=metric_fn,
        num_threads=num_threads,
    )
    log.info(f"bootstrap_trace_data returned: type={type(trace_data)}, len={len(trace_data) if hasattr(trace_data, '__len__') else 'N/A'}")

    # Serialize trace data for cross-process transfer
    trajectories = []
    scores = []
    outputs = []

    for idx, data in enumerate(trace_data):
        if idx == 0:
            log.info(f"First trace_data item: type={type(data)}, keys={data.keys() if isinstance(data, dict) else 'N/A'}")

        if data is None:
            trajectories.append(None)
            scores.append(failure_score)
            outputs.append(None)
            continue

        trace = data.get("trace", [])
        example = data.get("example")
        prediction = data.get("prediction")
        score = data.get("score")

        if idx == 0:
            log.info(f"First item: trace_len={len(trace)}, has_example={example is not None}, has_pred={prediction is not None}, score={score}")

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
            from dspy.teleprompt.bootstrap_trace import FailedPrediction
            if isinstance(prediction, FailedPrediction):
                outputs.append({"__failed__": True, "completion_text": getattr(prediction, "completion_text", "")})
            else:
                outputs.append(dict(prediction))
        else:
            outputs.append(None)

        # Serialize trace entries
        serialized_trace = []
        for entry_idx, entry in enumerate(trace):
            if idx == 0 and entry_idx == 0:
                log.info(f"First trace entry: type={type(entry)}, len={len(entry) if hasattr(entry, '__len__') else 'N/A'}")
            # entry is (predictor, inputs, output) tuple
            pred_module, inputs, output = entry

            sig_key = signature_key(pred_module.signature)

            # Serialize inputs
            ser_inputs = {k: str(v) for k, v in inputs.items()}

            # Serialize output
            from dspy.teleprompt.bootstrap_trace import FailedPrediction
            if isinstance(output, FailedPrediction):
                ser_output = {"__failed__": True, "completion_text": getattr(output, "completion_text", "")}
            else:
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

    log.info(f"Evaluation complete: {len(outputs)} outputs, {len(trajectories)} trajectories")
    return make_success_result(
        {"outputs": outputs, "scores": scores, "trajectories": trajectories},
        logs=log.get_logs()
    )
