"""Run-level coding harness config normalization."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from typing import Literal


CodingHarness = Literal["opencode", "claude"]

DEFAULT_CODING_HARNESS: CodingHarness = "opencode"
SUPPORTED_CODING_HARNESSES: tuple[CodingHarness, ...] = ("opencode", "claude")


@dataclass(frozen=True)
class CodingHarnessConfig:
    coding_harness: CodingHarness
    coding_harness_model: str | None = None


def normalize_coding_harness_config(
    *,
    coding_harness: str | None = None,
    coding_harness_model: str | None = None,
) -> CodingHarnessConfig:
    """Validate and normalize run-level coding harness config."""
    selected = DEFAULT_CODING_HARNESS if coding_harness is None else coding_harness
    if selected not in SUPPORTED_CODING_HARNESSES:
        supported = ", ".join(SUPPORTED_CODING_HARNESSES)
        raise ValueError(
            f"Unknown coding_harness {selected!r}. Expected one of: {supported}."
        )

    if selected == "claude":
        return CodingHarnessConfig(coding_harness="claude", coding_harness_model=None)

    _verify_opencode_binary()
    return CodingHarnessConfig(
        coding_harness="opencode",
        coding_harness_model=coding_harness_model,
    )


def _verify_opencode_binary() -> None:
    if shutil.which("opencode"):
        return
    raise RuntimeError(
        "OpenCode harness selected but the opencode binary was not found on PATH. "
        "Install OpenCode before running with coding_harness='opencode'."
    )


__all__ = [
    "CodingHarness",
    "CodingHarnessConfig",
    "DEFAULT_CODING_HARNESS",
    "SUPPORTED_CODING_HARNESSES",
    "normalize_coding_harness_config",
]
