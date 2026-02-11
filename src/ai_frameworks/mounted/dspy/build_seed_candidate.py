"""Build seed candidate command for DSPy programs.

Extracts initial instructions from all predictors in a DSPy module.
"""

from . import build_program
from sandbox.mounted.utils import get_logger, make_success_result

log = get_logger("build_seed")


def handle(cmd: dict, workspace: str) -> dict:
    """Extract initial instructions from the DSPy program.

    Args:
        cmd: Command dict with keys:
            - program: Dotted import path to DSPy module class
            - saved_program_json_path: Optional path to program.json
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success' and 'candidate' (predictor name -> instruction text)
    """
    program_path = cmd["program"]
    saved_json = cmd.get("saved_program_json_path")

    log.info(f"Building program: {program_path}")
    program = build_program(workspace, program_path, saved_json)
    log.info(f"Program type: {type(program).__name__}")

    candidate = {}
    for name, pred in program.named_predictors():
        candidate[name] = pred.signature.instructions
        log.info(f"Predictor '{name}': {len(pred.signature.instructions)} chars")

    log.info(f"Extracted {len(candidate)} predictors")
    return make_success_result({"candidate": candidate}, logs=log.get_logs())
