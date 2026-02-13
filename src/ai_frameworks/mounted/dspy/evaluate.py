"""Evaluate command for DSPy programs.

Runs a DSPy program on a batch of examples, applying candidate instructions
and scoring with the provided metric. Optionally captures execution traces
for reflective dataset building.

Requires DSPy >= 3.0.0.
"""

# CRITICAL: Import dspy at module level BEFORE any git checkout or module clearing.
#
# Architecture:
# - This script runs under VENV Python (/workspace/.venv/bin/python via PATH)
# - DSPy is installed in the venv's site-packages (from client's requirements.txt)
# - Importing here ensures dspy module references are stable before any
#   workspace manipulation (git checkout, sys.modules clearing)
#
# Previous bug: Importing dspy AFTER importlib.invalidate_caches() caused
# progressive module corruption ("cannot import name 'Example' from 'dspy.primitives'
# (unknown location)") after 5-6 optimizer iterations.

import sys

from sandbox.mounted.debug_env import get_dspy_import_diagnostic

try:
    import dspy
    from dspy.primitives import Example as DspyExample
except ImportError as _import_err:
    _diag = get_dspy_import_diagnostic(_import_err)
    print(f"[evaluate.py IMPORT ERROR]\n{_diag}", file=sys.stderr)
    raise ImportError(_diag) from _import_err

from . import build_program, load_import_path, signature_key
from sandbox.mounted.git_commands import checkout_branch_if_needed
from sandbox.mounted.utils import get_logger, make_error_result, make_success_result

log = get_logger("evaluate")

# Version detection for legacy compatibility
def _get_dspy_version_tuple():
    """Parse DSPy version into tuple for comparison."""
    try:
        version_str = dspy.__version__
        parts = version_str.split('.')
        return tuple(int(p) for p in parts[:3])  # (major, minor, patch)
    except Exception:
        # If we can't parse version, assume it's recent
        return (999, 0, 0)

_DSPY_VERSION = _get_dspy_version_tuple()
_USE_LEGACY_TRACE_CAPTURE = _DSPY_VERSION < (3, 0, 0)

if _USE_LEGACY_TRACE_CAPTURE:
    log.info(f"[VERSION] DSPy {dspy.__version__} (< 3.0.0) detected - using legacy trace capture")
else:
    log.info(f"[VERSION] DSPy {dspy.__version__} (>= 3.0.0) detected - using modern trace capture")


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
            - git_branch: Optional branch to checkout before evaluation
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success', 'outputs', 'scores', and optionally 'trajectories'
    """
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
    git_branch = cmd.get("git_branch")

    # Checkout the specified branch if provided (needed for code-mutated candidates)
    if git_branch:
        checkout_branch_if_needed(workspace, git_branch, log)

    # Use module-level dspy imports (imported at top before any workspace manipulation)
    Example = DspyExample

    lm = dspy.LM(program_lm)
    dspy.configure(lm=lm)
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
        log.info("Running evaluation with traces...")
        # Route to legacy implementation for older DSPy versions
        if _USE_LEGACY_TRACE_CAPTURE:
            return _evaluate_with_traces_legacy_wrapper(
                program, metric_fn, examples, failure_score, num_threads
            )
        else:
            return _evaluate_with_traces(
                program, metric_fn, examples, failure_score, num_threads
            )
    else:
        log.info("Running simple evaluation without traces...")
        return _evaluate_simple(
            program, metric_fn, examples, failure_score, num_threads
        )


def _evaluate_with_traces_legacy_wrapper(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Wrapper to call legacy trace capture for DSPy < 2.7.0."""
    try:
        from .dspy_legacy.evaluate import evaluate_with_traces_legacy
        return evaluate_with_traces_legacy(program, metric_fn, examples, failure_score, num_threads)
    except ImportError as e:
        log.error(f"Failed to import legacy trace capture: {e}")
        # Fallback to simple evaluation without traces
        log.warning("Falling back to simple evaluation without traces")
        return _evaluate_simple(program, metric_fn, examples, failure_score, num_threads)


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

    result = evaluator(program)
    
    # DSPy 3.0+: result.results is a list of (example, prediction, score) tuples
    results_list = result.results
    
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

    return make_success_result(
        {"outputs": outputs, "scores": scores},
        logs=log.get_logs()
    )


def _evaluate_with_traces(program, metric_fn, examples, failure_score, num_threads) -> dict:
    """Evaluate with trace capture for reflective dataset building.

    Modern implementation for DSPy >= 3.0.0 using bootstrap_trace_data.
    For DSPy < 3.0.0, see dspy_legacy/evaluate.py which uses dspy.context().

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
