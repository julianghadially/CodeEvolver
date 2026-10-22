"""Fake coding harness for focused optimizer wiring tests."""

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .base import AgentRole, HarnessPermissionPolicy, HarnessRequest, HarnessResult


@dataclass(frozen=True)
class FakeHarnessCall:
    role: AgentRole
    workspace: str
    prompt: str
    permissions: HarnessPermissionPolicy
    max_turns: int | None
    resume_session_id: str | None = None


class FakeCodingHarness:
    """Deterministic in-memory harness used by unit tests."""

    def __init__(
        self,
        *,
        results: Mapping[AgentRole, Iterable[HarnessResult]] | None = None,
        architect_results: Iterable[HarnessResult] | None = None,
    ) -> None:
        self._results: dict[AgentRole, deque[HarnessResult]] = defaultdict(deque)
        if results:
            for role, role_results in results.items():
                self._results[role].extend(role_results)
        if architect_results:
            self._results[AgentRole.ARCHITECT].extend(architect_results)
        self.calls: list[FakeHarnessCall] = []

    def run_architect(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.ARCHITECT)
        return self._record_and_pop(request)

    def resume_architect(
        self, session_id: str, request: HarnessRequest
    ) -> HarnessResult:
        self._require_role(request, AgentRole.ARCHITECT)
        return self._record_and_pop(request, resume_session_id=session_id)

    def run_coding(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.CODING)
        return self._record_and_pop(request)

    def run_inspection(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.INSPECTION)
        return self._record_and_pop(request)

    def _record_and_pop(
        self,
        request: HarnessRequest,
        *,
        resume_session_id: str | None = None,
    ) -> HarnessResult:
        self.calls.append(
            FakeHarnessCall(
                role=request.role,
                workspace=str(request.workspace),
                prompt=request.prompt,
                permissions=request.permissions,
                max_turns=request.max_turns,
                resume_session_id=resume_session_id,
            )
        )
        if self._results[request.role]:
            return self._results[request.role].popleft()
        return HarnessResult.successful(
            output="",
            raw_provider_metadata={"provider": "fake"},
        )

    @staticmethod
    def _require_role(request: HarnessRequest, expected: AgentRole) -> None:
        if request.role != expected:
            raise ValueError(
                f"{expected.value} harness call received {request.role.value!r} request"
            )
