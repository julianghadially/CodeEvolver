"""Runtime tests for agent behavior during CodeEvolver optimization.

Results are persisted to the directory returned by
``src.engine.paths.runtime_tests_dir()`` (non-local default
``/app/runtime_tests``; local deploys ``~/.codeevolver/runtime_tests``)
as JSON so we can track metrics across iterations and warn on degraded
behavior.

Usage from inspection_agent.py:
    from src.engine.agent.mounted.runtime_tests import (
        print_tool_call_diagnostics, check_trace_inspection, check_scores_contain_feedback,
    )

    print_tool_call_diagnostics(tool_calls)
    check_trace_inspection(tool_calls, iteration_num)
    check_scores_contain_feedback(iteration_num)
"""

import json
import os
import re

from src.engine.paths import runtime_tests_dir, traces_dir
from src.engine.telemetry.trace_layout import eval_trace_dir, index_path, read_index
LOW_INSPECTION_THRESHOLD = 4  # Warn after this many consecutive low-inspection iterations
NO_FEEDBACK_THRESHOLD = 4  # Warn after this many consecutive iterations with no feedback
MIN_FEEDBACK_NONNUMERIC_CHARS = 10  # Minimum non-numeric chars to count as real feedback


def print_tool_call_diagnostics(tool_calls: list[dict]):
    """Print compact summary of all tool calls made by the reflection agent."""
    print(f"[REFLECT:summary] Total tool calls: {len(tool_calls)}", flush=True)
    for i, tc in enumerate(tool_calls):
        tool = tc["tool"]
        inp = tc["input"]
        if tool == "Read":
            print(f"[REFLECT:summary]   [{i}] Read({inp.get('file_path', '?')})", flush=True)
        elif tool == "Grep":
            print(f"[REFLECT:summary]   [{i}] Grep('{inp.get('pattern', '?')}', {inp.get('path', '?')})", flush=True)
        elif tool == "Glob":
            print(f"[REFLECT:summary]   [{i}] Glob('{inp.get('pattern', '?')}', {inp.get('path', '?')})", flush=True)
        else:
            print(f"[REFLECT:summary]   [{i}] {tool}({json.dumps(inp)[:200]})", flush=True)


def _load_test_results(test_name: str) -> dict:
    """Load existing test results JSON for a given test name."""
    filepath = os.path.join(runtime_tests_dir(), f"{test_name}.json")
    if os.path.exists(filepath):
        try:
            with open(filepath) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_test_result(test_name: str, iteration_num: int, result: dict):
    """Save a result dict keyed by iteration number."""
    base = runtime_tests_dir()
    os.makedirs(base, exist_ok=True)
    results = _load_test_results(test_name)
    results[str(iteration_num)] = result

    filepath = os.path.join(base, f"{test_name}.json")
    with open(filepath, "w") as f:
        json.dump(results, f, indent=2)


def _count_consecutive_trailing_low(results: dict, min_calls: int = 2) -> int:
    """Count consecutive iterations at the tail with trace_call_count < min_calls."""
    if not results:
        return 0
    sorted_keys = sorted(results.keys(), key=lambda k: int(k))
    count = 0
    for key in reversed(sorted_keys):
        entry = results[key]
        call_count = entry.get("trace_call_count", 0) if isinstance(entry, dict) else 0
        if call_count < min_calls:
            count += 1
        else:
            break
    return count


def check_trace_inspection(tool_calls: list[dict], iteration_num: int):
    """Check if the reflection agent inspected trajectory files this iteration.

    Measures:
    - file_exists: whether the expected JSONL file is on disk
    - trace_call_count: how many tool calls targeted trajectory files

    Saves per-iteration results to /app/runtime_tests/trace_inspection.json.
    Warns if the trajectory file doesn't exist (hard fail), or if trace_call_count
    has been low for LOW_INSPECTION_THRESHOLD consecutive iterations.
    """
    if iteration_num == 0:
        return

    test_name = "trace_inspection"

    # --- Check trace directory existence ---
    expected_dir = eval_trace_dir(traces_dir(), iteration_num, "parent")
    file_exists = os.path.exists(index_path(expected_dir))

    if not file_exists:
        print(
            f"[REFLECT:check] iteration {iteration_num}: FAIL — "
            f"trajectory directory missing: {expected_dir}",
            flush=True,
        )
        _save_test_result(test_name, iteration_num, {
            "file_exists": False,
            "trace_call_count": 0,
        })
        return

    # --- Count tool calls that target trajectory files ---
    # A trace read names the traces dir, or an `iteration_<N>_<suffix>`
    # directory's index / row / span file.
    traces_root = traces_dir().lower()
    trace_calls = []
    for tc in tool_calls:
        inp_str = json.dumps(tc["input"]).lower()
        if (
            "trajector" in inp_str
            or traces_root in inp_str
            or ("iteration_" in inp_str and (
                "index.jsonl" in inp_str or "row.json" in inp_str or "span_" in inp_str
            ))
        ):
            trace_calls.append(tc)

    trace_call_count = len(trace_calls)

    # Save result
    _save_test_result(test_name, iteration_num, {
        "file_exists": True,
        "trace_call_count": trace_call_count,
    })

    # Print current status
    print(
        f"[REFLECT:check] iteration {iteration_num}: "
        f"trace_call_count={trace_call_count}, file_exists={file_exists}",
        flush=True,
    )

    # Check for consecutive low-inspection iterations
    results = _load_test_results(test_name)
    consecutive_low = _count_consecutive_trailing_low(results)
    if consecutive_low >= LOW_INSPECTION_THRESHOLD:
        print(
            f"[REFLECT:check] WARNING — trace inspection has been low "
            f"({consecutive_low} consecutive iterations with <2 calls). "
            f"The reflection agent may not be using trajectory data effectively.",
            flush=True,
        )


def _count_nonnumeric_chars(text: str) -> int:
    """Count characters that are not digits, dots, or whitespace."""
    return len(re.sub(r"[\d.\s]", "", text))


def _count_consecutive_trailing_no_feedback(results: dict) -> int:
    """Count consecutive iterations at the tail with examples_with_feedback == 0."""
    if not results:
        return 0
    sorted_keys = sorted(results.keys(), key=lambda k: int(k))
    count = 0
    for key in reversed(sorted_keys):
        entry = results[key]
        if isinstance(entry, dict) and entry.get("examples_with_feedback", 0) == 0:
            count += 1
        else:
            break
    return count


def check_scores_contain_feedback(iteration_num: int):
    """Check if the trajectory JSONL contains meaningful feedback text from the metric.

    Reads the JSONL file and counts how many examples have a "feedback" field
    with at least MIN_FEEDBACK_NONNUMERIC_CHARS non-numeric characters.

    Saves per-iteration results to /app/runtime_tests/scores_contain_feedback.json.
    Warns after NO_FEEDBACK_THRESHOLD consecutive iterations with zero feedback.
    """
    if iteration_num == 0:
        return

    test_name = "scores_contain_feedback"

    expected_dir = eval_trace_dir(traces_dir(), iteration_num, "parent")
    if not os.path.exists(index_path(expected_dir)):
        print(
            f"[REFLECT:check] iteration {iteration_num}: scores_contain_feedback — "
            f"skipped (trajectory directory missing)",
            flush=True,
        )
        return

    # The index carries each row's feedback; no need to open the row files.
    total_examples = 0
    examples_with_feedback = 0
    try:
        for obj in read_index(expected_dir):
            total_examples += 1
            feedback = obj.get("feedback")
            if feedback and isinstance(feedback, str):
                if _count_nonnumeric_chars(feedback) >= MIN_FEEDBACK_NONNUMERIC_CHARS:
                    examples_with_feedback += 1
    except Exception as e:
        print(
            f"[REFLECT:check] iteration {iteration_num}: scores_contain_feedback — "
            f"error reading file: {e}",
            flush=True,
        )
        return

    # Save result
    _save_test_result(test_name, iteration_num, {
        "total_examples": total_examples,
        "examples_with_feedback": examples_with_feedback,
    })

    # Print current status
    print(
        f"[REFLECT:check] iteration {iteration_num}: scores_contain_feedback — "
        f"{examples_with_feedback}/{total_examples} examples have feedback text",
        flush=True,
    )

    # Check for consecutive iterations with no feedback
    results = _load_test_results(test_name)
    consecutive_no_feedback = _count_consecutive_trailing_no_feedback(results)
    if consecutive_no_feedback >= NO_FEEDBACK_THRESHOLD:
        print(
            f"[REFLECT:check] WARNING — Scores are missing feedback for "
            f"{consecutive_no_feedback} iterations. Please provide score with "
            f"feedback (ScoreWithFeedback if using DSPy).",
            flush=True,
        )
