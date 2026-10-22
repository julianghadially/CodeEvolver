"""Optimization launcher.

The /optimize API endpoint calls `spawn_optimization(**kwargs)`; this module
routes that call to the local subprocess launcher. Optimizations run only in
local mode (directly on a host, or inside the Docker runner); the hosted
portal API rejects /optimize before reaching here.

Cancellation flows through the same dispatch via `cancel_optimization(job_id)`,
which SIGTERMs the tracked subprocess.
"""
from __future__ import annotations

from src.config import settings
from src.launchers.base import OptimizationLauncher


def _get_launcher() -> OptimizationLauncher:
    from src.launchers.local_launcher import LocalLauncher
    return LocalLauncher()


async def spawn_optimization(**kwargs) -> str:
    """Fire-and-forget optimization launch. Returns an implementation-specific call id."""
    # Resolve the optimizer strategy from the STRATEGY env var (default 'freeform')
    # unless the caller passed an explicit `method`, so the launcher honors it
    # without a /optimize request field.
    kwargs.setdefault("method", settings.strategy)
    return await _get_launcher().spawn(**kwargs)


async def cancel_optimization(job_id: str) -> bool:
    """Best-effort cancel. Returns True if a live process was signaled."""
    return await _get_launcher().cancel(job_id)


__all__ = ["spawn_optimization", "cancel_optimization", "OptimizationLauncher"]
