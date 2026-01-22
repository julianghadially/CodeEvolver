"""DSPy program execution.

Handles loading and running DSPy programs inside the sandbox.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ProgramOutput:
    """Output from running a program on a single example."""

    example_id: int
    output: Any
    error: str | None = None


@dataclass
class ProgramRunResult:
    """Result from running a program on all examples."""

    success: bool
    outputs: list[ProgramOutput]
    traces: list[Any] | None = None
    error: str | None = None


def load_program_json(workspace_path: Path, program_json_path: str) -> dict[str, Any]:
    """
    Load program.json from the workspace.

    Args:
        workspace_path: Path to the git workspace
        program_json_path: Relative path to program.json

    Returns:
        Parsed program JSON

    Raises:
        FileNotFoundError: If program.json doesn't exist
        json.JSONDecodeError: If JSON is invalid
    """
    full_path = workspace_path / program_json_path
    if not full_path.exists():
        raise FileNotFoundError(f"Program JSON not found: {full_path}")

    with open(full_path) as f:
        return json.load(f)


def save_program_json(
    workspace_path: Path,
    program_json_path: str,
    program_json: dict[str, Any],
) -> None:
    """
    Save program.json to the workspace.

    Args:
        workspace_path: Path to the git workspace
        program_json_path: Relative path to program.json
        program_json: Program data to save
    """
    full_path = workspace_path / program_json_path
    full_path.parent.mkdir(parents=True, exist_ok=True)

    with open(full_path, "w") as f:
        json.dump(program_json, f, indent=2)


def apply_prompt_mutation(
    program_json: dict[str, Any],
    candidate: dict[str, str],
) -> dict[str, Any]:
    """
    Apply prompt mutation to program.json.

    Updates signature.instructions for the specified components.

    Args:
        program_json: Original program JSON
        candidate: Dict mapping component_name -> new instruction text

    Returns:
        Modified program JSON

    Raises:
        KeyError: If component not found in program
    """
    modified = program_json.copy()

    for component_name, new_instruction in candidate.items():
        if component_name not in modified:
            raise KeyError(f"Component not found in program: {component_name}")

        component = modified[component_name]
        if "signature" not in component:
            raise KeyError(f"Component {component_name} has no signature")

        component["signature"]["instructions"] = new_instruction

    return modified


async def run_program(
    workspace_path: Path,
    program_json_path: str,
    entry_point: str,
    test_examples: list[dict[str, Any]],
    capture_traces: bool = False,
) -> ProgramRunResult:
    """
    Run a DSPy program on test examples.

    Args:
        workspace_path: Path to the git workspace
        program_json_path: Relative path to program.json
        entry_point: DSPy module class (e.g., 'fire.FIREJudge')
        test_examples: List of example dicts to run
        capture_traces: Whether to capture execution traces

    Returns:
        ProgramRunResult with outputs and optional traces
    """
    # For now, return placeholder outputs
    # Full implementation requires DSPy runtime integration
    outputs = []
    for i, example in enumerate(test_examples):
        outputs.append(
            ProgramOutput(
                example_id=i,
                output={
                    "_placeholder": True,
                    "_note": "DSPy runtime integration pending",
                    "input": example,
                },
            )
        )

    traces = [] if capture_traces else None

    return ProgramRunResult(
        success=True,
        outputs=outputs,
        traces=traces,
    )


def generate_runner_script(
    workspace_path: str,
    program_json_path: str,
    entry_point: str,
    test_examples: list[dict[str, Any]],
    capture_traces: bool = False,
) -> str:
    """
    Generate a Python script that runs the DSPy program.

    This script is written to the sandbox and executed there.

    Args:
        workspace_path: Path to workspace in the sandbox
        program_json_path: Path to program.json
        entry_point: DSPy module class to run
        test_examples: Examples to run
        capture_traces: Whether to capture traces

    Returns:
        Python script as a string
    """
    examples_json = json.dumps(test_examples)

    return f'''#!/usr/bin/env python3
"""Auto-generated script for running DSPy program."""

import json
import os
import sys

# Add workspace to path for imports
workspace = "{workspace_path}"
sys.path.insert(0, workspace)

# Load environment variables
env_file = f"{{workspace}}/.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.replace("export ", "").strip()
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value

try:
    import dspy

    # Load program.json
    program_json_path = f"{{workspace}}/{program_json_path}"
    with open(program_json_path) as f:
        program_json = json.load(f)

    # Parse entry_point to get module and class
    entry_point = "{entry_point}"
    module_path, class_name = entry_point.rsplit(".", 1)

    # Import the module dynamically
    import importlib
    module = importlib.import_module(module_path)
    program_class = getattr(module, class_name)

    # Instantiate and load state
    program = program_class()
    program.load(program_json_path)

    # Run on test examples
    test_examples = {examples_json}
    outputs = []
    traces = [] if {capture_traces} else None

    for i, example in enumerate(test_examples):
        try:
            # Convert dict to DSPy Example
            dspy_example = dspy.Example(**example)
            result = program(dspy_example)
            outputs.append({{
                "example_id": i,
                "output": result.toDict() if hasattr(result, "toDict") else str(result),
            }})
        except Exception as e:
            outputs.append({{
                "example_id": i,
                "output": None,
                "error": str(e),
            }})

    # Output results as JSON
    print("RUNNER_RESULT:" + json.dumps({{"outputs": outputs, "traces": traces}}))

except ImportError as e:
    print(f"RUNNER_ERROR: Import failed: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"RUNNER_ERROR: {{e}}")
    sys.exit(1)
'''


def parse_runner_output(stdout: str, stderr: str, returncode: int) -> ProgramRunResult:
    """
    Parse the output from running the program runner script.

    Args:
        stdout: Standard output from the script
        stderr: Standard error from the script
        returncode: Exit code from the script

    Returns:
        ProgramRunResult with parsed outputs
    """
    if returncode != 0:
        error_lines = [
            line
            for line in (stdout + "\n" + stderr).split("\n")
            if "RUNNER_ERROR" in line
        ]
        error_msg = error_lines[0].replace("RUNNER_ERROR: ", "") if error_lines else stderr
        return ProgramRunResult(success=False, outputs=[], error=error_msg)

    # Find the result line
    for line in stdout.split("\n"):
        if line.startswith("RUNNER_RESULT:"):
            result_json = line[len("RUNNER_RESULT:") :]
            try:
                data = json.loads(result_json)
                outputs = [
                    ProgramOutput(
                        example_id=o["example_id"],
                        output=o.get("output"),
                        error=o.get("error"),
                    )
                    for o in data.get("outputs", [])
                ]
                return ProgramRunResult(
                    success=True,
                    outputs=outputs,
                    traces=data.get("traces"),
                )
            except json.JSONDecodeError as e:
                return ProgramRunResult(
                    success=False,
                    outputs=[],
                    error=f"Failed to parse runner output: {e}",
                )

    return ProgramRunResult(
        success=False,
        outputs=[],
        error="No result found in runner output",
    )
