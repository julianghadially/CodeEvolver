"""DSPy-specific sandbox scripts for GEPA optimization.

Shared utilities used across DSPy command handlers.
"""

import importlib
import sys
from pathlib import Path
from typing import Any


def load_import_path(workspace_path: str, dotted_path: str) -> Any:
    """Import an attribute from a dotted path relative to the workspace."""
    ws = str(workspace_path)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    module_path, attr_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def build_program(workspace_path: str, program_path: str,
                  saved_program_json_path: str | None,
                  candidate: dict[str, str] | None = None):
    """Instantiate DSPy module, load program.json, optionally apply candidate instructions."""
    import dspy

    cls = load_import_path(workspace_path, program_path)
    program = cls()

    if saved_program_json_path:
        full_path = Path(workspace_path) / saved_program_json_path
        if full_path.exists():
            program.load(str(full_path))

    if candidate:
        for name, pred in program.named_predictors():
            if name in candidate:
                pred.signature = pred.signature.with_instructions(candidate[name])

    return program


def signature_key(sig) -> str:
    """Build a fingerprint key from a DSPy signature's field names.

    Used to match serialized trace entries to predictors across the
    process boundary.
    """
    input_fields = sorted(sig.input_fields.keys())
    output_fields = sorted(sig.output_fields.keys())
    return f"{','.join(input_fields)}->{','.join(output_fields)}"
