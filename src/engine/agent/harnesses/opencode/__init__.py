"""OpenCode provider adapter for CodeEvolver coding harnesses."""

from .architect_mcp import (
    OpenCodeArchitectControlServer,
    OpenCodeArchitectToolBridge,
)
from .commands import OpenCodeSessionSpec
from .harness import (
    OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
    OPENCODE_ARCHITECT_TIMEOUT_SECONDS_PER_TURN,
    OPENCODE_BOOTSTRAP_COMMANDS,
    OPENCODE_CODING_TIMEOUT_SECONDS_PER_TURN,
    OPENCODE_CODE_CHANGE_FINALIZATION_COMMANDS,
    OPENCODE_INSPECTION_TIMEOUT_SECONDS_PER_TURN,
    OpenCodeHarness,
    OpenCodeHarnessNotImplementedError,
)

__all__ = [
    "OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT",
    "OPENCODE_ARCHITECT_TIMEOUT_SECONDS_PER_TURN",
    "OPENCODE_BOOTSTRAP_COMMANDS",
    "OPENCODE_CODING_TIMEOUT_SECONDS_PER_TURN",
    "OPENCODE_CODE_CHANGE_FINALIZATION_COMMANDS",
    "OPENCODE_INSPECTION_TIMEOUT_SECONDS_PER_TURN",
    "OpenCodeArchitectControlServer",
    "OpenCodeArchitectToolBridge",
    "OpenCodeHarness",
    "OpenCodeHarnessNotImplementedError",
    "OpenCodeSessionSpec",
]
