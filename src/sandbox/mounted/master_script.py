"""Copyright © 2026 440 Labs LLC
Master script dispatcher for sandbox prebuilt commands.

Standalone script that runs inside the Modal sandbox. Reads a JSON command
file, dispatches to the appropriate handler module, and prints
EVAL_RESULT:{json} to stdout.

This file is the entry point for all prebuilt scripts in the sandbox.
"""

import argparse
import json
import sys

from sandbox.mounted.debug_env import _log_environment_debug
from sandbox.mounted.utils import get_logger, make_error_result

# Command handlers are imported dynamically based on command type
# Currently supported: dspy.build_seed_candidate, dspy.evaluate, dspy.make_reflective_dataset

HANDLERS = {}
log = get_logger("master")



def _register_dspy_handlers():
    """Register DSPy command handlers."""
    # Log environment before attempting imports (helps debug iteration 7-8 failures)
    _log_environment_debug(log)

    try:
        from ai_frameworks.mounted.dspy import build_seed_candidate
        from ai_frameworks.mounted.dspy import evaluate
        from ai_frameworks.mounted.dspy import make_reflective_dataset

        HANDLERS["build_seed_candidate"] = build_seed_candidate.handle
        HANDLERS["evaluate"] = evaluate.handle
        HANDLERS["make_reflective_dataset"] = make_reflective_dataset.handle
    except ImportError as e:
        log.exception(f"Failed to import DSPy handlers: {e}")
        # Re-raise with more context
        raise ImportError(
            f"DSPy handler import failed: {e}\n"
            f"Python: {sys.executable}\n"
            f"sys.path[0:3]: {sys.path[:3]}\n"
            f"Check if venv is active and dspy is installed."
        ) from e


def main():
    parser = argparse.ArgumentParser(description="Sandbox prebuilt script dispatcher")
    parser.add_argument("--workspace", required=True, help="Path to cloned repo")
    parser.add_argument("--command-file", required=True, help="Path to JSON command file")
    args = parser.parse_args()

    # Read command file
    with open(args.command_file) as f:
        cmd = json.load(f)

    command_name = cmd.get("command")
    log.info(f"Dispatching command: {command_name}")

    # Lazy-load handlers based on command prefix
    # For now, all commands are DSPy commands
    if not HANDLERS:
        _register_dspy_handlers()

    handler = HANDLERS.get(command_name)
    if handler is None:
        result = {"success": False, "error": f"Unknown command: {command_name}", "logs": log.get_logs()}
    else:
        try:
            result = handler(cmd, args.workspace)
        except Exception as e:
            log.exception(f"Handler {command_name} raised exception")
            result = make_error_result(e, logs=log.get_logs())

    # Output result on a single line with known prefix for parsing
    print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
    main()
