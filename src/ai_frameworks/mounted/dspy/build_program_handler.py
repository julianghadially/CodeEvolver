"""Build DSPy program from a GEPA candidate and save in DSPy-native format.

Takes a candidate dict (predictor_name -> instruction text) and reconstructs
a full DSPy Module with those instructions applied, then saves via program.save().
"""

from . import build_program
from sandbox.mounted.git_commands import checkout_branch_if_needed
from sandbox.mounted.utils import get_logger, make_error_result, make_success_result

log = get_logger("build_program")


def handle(cmd: dict, workspace: str) -> dict:
    """Build a DSPy program from a candidate and save to disk.

    Args:
        cmd: Command dict with keys:
            - program: Dotted import path to DSPy module class
            - saved_program_json_path: Optional path to program.json
            - candidate: Dict of predictor_name -> instruction text
            - output_path: Relative path to save the DSPy program (e.g. "codeevolver/results/optimized_program_20260224.json")
            - git_branch: Optional branch to checkout before building
        workspace: Path to cloned client repository

    Returns:
        Dict with 'success', 'output_path', 'predictor_count'
    """
    try:
        program_path = cmd["program"]
        saved_json = cmd.get("saved_program_json_path")
        candidate = cmd.get("candidate", {})
        output_path = cmd["output_path"]
        git_branch = cmd.get("git_branch")

        # Checkout the specified branch if provided
        if git_branch:
            checkout_branch_if_needed(workspace, git_branch, log)

        log.info(f"Building program: {program_path} with {len(candidate)} predictor instructions")

        # build_program instantiates the DSPy module and applies candidate instructions
        program = build_program(workspace, program_path, saved_json, candidate=candidate)
        log.info(f"Program type: {type(program).__name__}")

        # Ensure output directory exists
        import os
        from pathlib import Path

        full_output_path = Path(workspace) / output_path
        full_output_path.parent.mkdir(parents=True, exist_ok=True)

        # Patch Retrieve.dump_state to accept json_mode kwarg.
        # DSPy 3.x BaseModule.save() passes json_mode= to every parameter's
        # dump_state(), but Retrieve.dump_state() doesn't accept it.
        import dspy
        _orig_retrieve_dump = dspy.Retrieve.dump_state

        def _patched_retrieve_dump(self, **kwargs):
            kwargs.pop("json_mode", None)
            return _orig_retrieve_dump(self)

        dspy.Retrieve.dump_state = _patched_retrieve_dump

        # Save in DSPy-native format (state JSON, loadable via program.load())
        try:
            program.save(str(full_output_path))
        finally:
            dspy.Retrieve.dump_state = _orig_retrieve_dump
        log.info(f"Saved DSPy program to {output_path}")

        predictor_count = len(list(program.named_predictors()))

        return make_success_result(
            {
                "output_path": output_path,
                "predictor_count": predictor_count,
            },
            logs=log.get_logs(),
        )

    except Exception as e:
        log.exception(f"Failed to build/save program")
        return make_error_result(e, logs=log.get_logs())
