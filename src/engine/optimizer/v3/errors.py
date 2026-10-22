"""Copyright © 2026 440 Labs LLC

Shared exception types for EE architect tools.

A tool raises `ControlViolation` (or a subclass) when the architect calls it
in an invalid order or with invalid arguments. The iteration architect's
Claude Agent SDK proxy catches these and delivers the error message back to
the model as a structured tool-error response so the model can re-order
its calls.

Keeping these in a standalone module avoids circular imports between
`agent_tools.py` and the individual tool modules.
"""

from __future__ import annotations


class ControlViolation(ValueError):
    """Raised when a tool is called in an invalid order for the iteration."""


class RunEvalError(ControlViolation):
    """Specific control violation for the `run_eval` tool."""


class SelectCandidateError(ControlViolation):
    """Specific control violation for `select_parent` (preselect path)."""


class CloseIterationError(ControlViolation):
    """Specific control violation for the `close_iteration` tool."""


__all__ = [
    "ControlViolation",
    "RunEvalError",
    "SelectCandidateError",
    "CloseIterationError",
]
