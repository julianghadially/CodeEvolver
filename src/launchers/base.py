"""OptimizationLauncher protocol.

Implemented by `LocalLauncher` (subprocess on the host). The call shape is
fire-and-forget — the optimizer reports back through the API's
internal callback endpoints, not through this return value.
"""
from __future__ import annotations

from typing import Protocol


class OptimizationLauncher(Protocol):
    async def spawn(self, **kwargs) -> str:
        """Launch an optimization run. Returns an implementation-specific id."""
        ...

    async def cancel(self, job_id: str) -> bool:
        """Best-effort cancel; True if a live process was signaled."""
        ...
