"""Copyright © 2026 440 Labs LLC

Master script dispatcher for evaluation commands.

Standalone script that `subprocess_service.run_evaluation` runs in a
subprocess. Reads a JSON command file, dispatches to the registered handler
(the generic evaluator, `src/engine/evaluator/mounted/evaluate.py`), and
prints EVAL_RESULT:{json} to stdout.
"""

import argparse
import json
import sys
from pathlib import Path

# Add the engine root to sys.path so `from src.engine.X import Y` resolves.
# This script lives at `<root>/src/engine/sandbox/mounted/master_script.py`,
# so `parents[4]` is `<root>`. Mirrors `src.engine.paths.engine_root()` but
# inline because we can't import that module before sys.path is set up.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[4])
if _ENGINE_ROOT not in sys.path:
    sys.path.insert(0, _ENGINE_ROOT)

from src.engine.sandbox.mounted.utils import get_logger, make_error_result

HANDLERS = {}
log = get_logger("master")


def _register_handlers():
    """Register sandbox command handlers.

    Currently only the generic evaluator is registered. Future handlers (e.g.
    a tracing-agent command, a rerun-single-row helper) can be added here.
    """
    try:
        from src.engine.evaluator.mounted import evaluate

        HANDLERS["evaluate"] = evaluate.handle
    except ImportError as e:
        log.exception(f"Failed to import evaluator handler: {e}")
        raise ImportError(
            f"Evaluator handler import failed: {e}\n"
            f"Python: {sys.executable}\n"
            f"sys.path[0:3]: {sys.path[:3]}\n"
            f"Check that src/engine/evaluator/mounted/ is importable."
        ) from e


def main():
    parser = argparse.ArgumentParser(description="Sandbox prebuilt script dispatcher")
    parser.add_argument("--workspace", required=True, help="Path to cloned repo")
    parser.add_argument("--command-file", required=True, help="Path to JSON command file")
    args = parser.parse_args()

    with open(args.command_file) as f:
        cmd = json.load(f)

    command_name = cmd.get("command")
    log.info(f"Dispatching command: {command_name}")

    if not HANDLERS:
        _register_handlers()

    handler = HANDLERS.get(command_name)
    if handler is None:
        result = {
            "success": False,
            "error": f"Unknown command: {command_name}",
            "logs": log.get_logs(),
        }
    else:
        try:
            result = handler(cmd, args.workspace)
        except Exception as e:
            log.exception(f"Handler {command_name} raised exception")
            result = make_error_result(e, logs=log.get_logs())

    print(f"EVAL_RESULT:{json.dumps(result)}")


if __name__ == "__main__":
    main()
