"""Provider-neutral contract for CodeEvolver coding harnesses."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol


class AgentRole(str, Enum):
    """CodeEvolver agent roles handled by a coding harness."""

    ARCHITECT = "architect"
    CODING = "coding"
    INSPECTION = "inspection"


@dataclass(frozen=True)
class HarnessPermissionPolicy:
    """Permission behavior requested from the provider harness."""

    autonomous: bool = True

    @classmethod
    def autonomous_policy(cls) -> "HarnessPermissionPolicy":
        return cls(autonomous=True)


@dataclass(frozen=True)
class HarnessRequest:
    """Role-aware request passed to a coding harness."""

    role: AgentRole
    workspace: str | Path
    prompt: str
    permissions: HarnessPermissionPolicy = field(
        default_factory=HarnessPermissionPolicy.autonomous_policy
    )
    max_turns: int | None = None
    raw_provider_metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HarnessResult:
    """Normalized result returned by any provider harness."""

    success: bool
    error: str | None
    output: str
    session_id: str | None = None
    raw_provider_metadata: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def successful(
        cls,
        *,
        output: str | None = "",
        session_id: str | None = None,
        raw_provider_metadata: Mapping[str, Any] | None = None,
    ) -> "HarnessResult":
        return cls(
            success=True,
            error=None,
            output=output or "",
            session_id=session_id,
            raw_provider_metadata=dict(raw_provider_metadata or {}),
        )

    @classmethod
    def failed(
        cls,
        *,
        error: str | BaseException,
        output: str | None = "",
        session_id: str | None = None,
        raw_provider_metadata: Mapping[str, Any] | None = None,
    ) -> "HarnessResult":
        return cls(
            success=False,
            error=str(error),
            output=output or "",
            session_id=session_id,
            raw_provider_metadata=dict(raw_provider_metadata or {}),
        )


class CodingHarness(Protocol):
    """Small CodeEvolver-specific harness contract."""

    def run_architect(self, request: HarnessRequest) -> HarnessResult:
        """Start an architect session and return normalized session metadata."""
        ...

    def resume_architect(
        self, session_id: str, request: HarnessRequest
    ) -> HarnessResult:
        """Resume an architect session, used by watchdog follow-ups."""
        ...

    def run_coding(self, request: HarnessRequest) -> HarnessResult:
        """Run a coding-agent request to completion."""
        ...

    def run_inspection(self, request: HarnessRequest) -> HarnessResult:
        """Run an inspection-agent request to completion."""
        ...
