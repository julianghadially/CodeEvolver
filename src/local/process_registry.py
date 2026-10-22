"""Live-process registry for local optimization runs.

`LocalLauncher.spawn()` registers each asyncio subprocess by job_id.
`LocalLauncher.cancel()` and the FastAPI shutdown hook iterate this
registry to terminate orphans. Module-level state is fine — the API
process is the only writer.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable

logger = logging.getLogger(__name__)

_registry: dict[str, asyncio.subprocess.Process] = {}


def register(job_id: str, proc: asyncio.subprocess.Process) -> None:
    _registry[job_id] = proc


def unregister(job_id: str) -> None:
    _registry.pop(job_id, None)


def get(job_id: str) -> asyncio.subprocess.Process | None:
    return _registry.get(job_id)


def all_job_ids() -> Iterable[str]:
    return list(_registry.keys())


async def terminate(job_id: str, kill_after: float = 10.0) -> bool:
    """SIGTERM the tracked process; SIGKILL after `kill_after` seconds.

    Returns True if a live process was signaled.
    """
    proc = _registry.get(job_id)
    if proc is None or proc.returncode is not None:
        return False
    try:
        proc.terminate()
    except ProcessLookupError:
        unregister(job_id)
        return False
    try:
        await asyncio.wait_for(proc.wait(), timeout=kill_after)
    except asyncio.TimeoutError:
        logger.warning("Job %s did not exit on SIGTERM; sending SIGKILL.", job_id)
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    finally:
        unregister(job_id)
    return True
