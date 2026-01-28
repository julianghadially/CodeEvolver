"""Master script dispatcher for sandbox prebuilt commands.

Standalone script that runs inside the Modal sandbox. Reads a JSON command
file, dispatches to the appropriate handler module, and prints
EVAL_RESULT:{json} to stdout.

This file is the entry point for all prebuilt scripts in the sandbox.
"""

import argparse
import json
import traceback

# Command handlers are imported dynamically based on command type
# Currently supported: dspy.build_seed_candidate, dspy.evaluate, dspy.make_reflective_dataset

HANDLERS = {}


def _register_dspy_handlers():
    """Register DSPy command handlers."""
    from sandbox_scripts.dspy import build_seed_candidate
    from sandbox_scripts.dspy import evaluate
    from sandbox_scripts.dspy import make_reflective_dataset

    HANDLERS["build_seed_candidate"] = build_seed_candidate.handle
    HANDLERS["evaluate"] = evaluate.handle
    HANDLERS["make_reflective_dataset"] = make_reflective_dataset.handle


def main():
    parser = argparse.ArgumentParser(description="Sandbox prebuilt script dispatcher")
    parser.add_argument("--workspace", required=True, help="Path to cloned repo")
    parser.add_argument("--command-file", required=True, help="Path to JSON command file")
    args = parser.parse_args()

    # Read command file
    with open(args.command_file) as f:
        cmd = json.load(f)

    command_name = cmd.get("command")

    # Lazy-load handlers based on command prefix
    # For now, all commands are DSPy commands
    if not HANDLERS:
        _register_dspy_handlers()

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
