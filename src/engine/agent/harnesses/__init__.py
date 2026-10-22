"""Shared coding harness contract and test fakes."""

from .base import (
    AgentRole,
    CodingHarness,
    HarnessPermissionPolicy,
    HarnessRequest,
    HarnessResult,
)
from .claude import ClaudeCodeHarness, ClaudeHarnessNotWiredError
from .factory import create_coding_harness
from .fake import FakeCodingHarness, FakeHarnessCall
from .opencode import OpenCodeHarness, OpenCodeHarnessNotImplementedError


__all__ = [
    "AgentRole",
    "ClaudeCodeHarness",
    "ClaudeHarnessNotWiredError",
    "CodingHarness",
    "create_coding_harness",
    "FakeCodingHarness",
    "FakeHarnessCall",
    "HarnessPermissionPolicy",
    "HarnessRequest",
    "HarnessResult",
    "OpenCodeHarness",
    "OpenCodeHarnessNotImplementedError",
]
