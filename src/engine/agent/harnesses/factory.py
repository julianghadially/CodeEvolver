"""Provider selection for CodeEvolver coding harnesses."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from .base import CodingHarness
from .claude import ClaudeCodeHarness
from .opencode import OpenCodeHarness


def create_coding_harness(
    *,
    coding_harness: str,
    coding_harness_model: str | None = None,
    architect_context: Any | None = None,
    architect_config: Any | None = None,
    architect_tool_registry: Mapping[str, Callable[..., dict]] | None = None,
    coding_runner: Callable[..., dict] | None = None,
    inspection_runner: Callable[..., dict] | None = None,
) -> CodingHarness:
    """Create the selected provider adapter from run-level config."""
    if coding_harness == "claude":
        return ClaudeCodeHarness(
            architect_context=architect_context,
            architect_config=architect_config,
            architect_tool_registry=architect_tool_registry,
            coding_runner=coding_runner,
            inspection_runner=inspection_runner,
        )
    if coding_harness == "opencode":
        return OpenCodeHarness(
            model=coding_harness_model,
            architect_context=architect_context,
            architect_config=architect_config,
            architect_tool_registry=dict(architect_tool_registry or {}),
        )
    raise NotImplementedError(
        f"{coding_harness!r} coding harness execution is not implemented"
    )


__all__ = ["create_coding_harness"]
