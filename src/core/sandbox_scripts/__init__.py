"""Sandbox scripts that run inside client Modal sandboxes.

These scripts are copied into the sandbox image and executed
via sandbox.exec(). They have access to the client's full
Python environment (dspy, their code, etc.).
"""
from .debug_env import (
    get_debug_python_command,
    get_debug_env_info,
    get_dspy_import_diagnostic,
    _log_environment_debug,
)

__all__ = [
    "get_debug_python_command",
    "get_debug_env_info",
    "get_dspy_import_diagnostic",
    "_log_environment_debug",
]
