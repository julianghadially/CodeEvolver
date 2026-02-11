"""Sandbox infrastructure for CodeEvolver Agents.

Contains sandbox lifecycle management and Modal deployment utilities.
"""

from .client_sandbox import ClientSandbox
from .verify_environment import validate_sandbox_environment

__all__ = [
    "ClientSandbox",
    "validate_sandbox_environment",
]
