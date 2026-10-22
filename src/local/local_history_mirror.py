"""Local-mode JSONL history mirror.

Cloud deploys persist iteration history in `JobRecord.ce_state_history`
on MongoDB Atlas. Local deploys point Mongo at a single-node Community
instance — no replica set, no oplog, easy to corrupt or stop. To make
sure the iteration record a researcher needs to report back to a client
survives a Mongo outage, we write each iteration as a JSONL line to
`<local_history_root>/{job_id}.jsonl` **before** updating Mongo. A final
snapshot lands at `<local_history_root>/{job_id}.final.json` on terminal
status.

A small CLI at `scripts/export_history.py` reconstructs the cloud-shaped
iteration history from these files when Mongo is unavailable.

All functions are no-ops when `settings.sandbox_mode != "local"` so the
cloud product never pays the extra disk I/O. The pairing of "Mongo write
+ local mirror" is owned by `src/db/job_writes.py`; callers should go
through that module rather than invoking these functions directly.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import settings

logger = logging.getLogger(__name__)


def _enabled() -> bool:
    return settings.sandbox_mode == "local"


def _job_history_path(job_id: str) -> Path:
    root = settings.local_history_root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{job_id}.jsonl"


def _job_final_path(job_id: str) -> Path:
    root = settings.local_history_root
    root.mkdir(parents=True, exist_ok=True)
    return root / f"{job_id}.final.json"


def append_iteration(
    job_id: str,
    iteration_history: dict[str, Any] | None,
    progress: dict[str, Any] | None = None,
) -> None:
    """Append one or more iteration records as JSONL lines.

    `iteration_history` is the `ce_state["iteration_history"]` dict the
    optimizer sends on progress callbacks (iteration index → record).
    Each entry produces one JSONL line:

        {"job_id": "...", "iteration": "3", "record": {...},
         "progress": {...}, "ts": "2026-..."}

    `progress` is the surrounding progress envelope (current_iteration,
    best_score, etc.) — copied onto every iteration line in this batch so
    the operator can correlate the per-iteration record with the run
    state at the time it landed.
    """
    if not _enabled() or not iteration_history:
        return
    path = _job_history_path(job_id)
    ts = datetime.now(timezone.utc).isoformat()
    try:
        with path.open("a") as f:
            for idx_key, record in iteration_history.items():
                line = {
                    "job_id": job_id,
                    "iteration": str(idx_key),
                    "record": record,
                    "progress": progress or {},
                    "ts": ts,
                }
                f.write(json.dumps(line, default=str) + "\n")
    except OSError as exc:
        # Never let mirror failure break the API handler. Log and move on —
        # Mongo remains the primary store; the mirror is best-effort backup.
        logger.warning(
            "local_history_mirror.append_iteration failed for %s: %s", job_id, exc
        )


def finalize(
    job_id: str,
    status: str,
    summary: dict[str, Any],
) -> None:
    """Write the run's terminal snapshot to `{job_id}.final.json`.

    `summary` is whatever the API's status handler has assembled —
    best_candidate, best_score, num_candidates, error, etc. Anything
    JSON-serializable is fine.
    """
    if not _enabled():
        return
    path = _job_final_path(job_id)
    try:
        with path.open("w") as f:
            json.dump(
                {
                    "job_id": job_id,
                    "status": status,
                    "summary": summary,
                    "ts": datetime.now(timezone.utc).isoformat(),
                },
                f,
                default=str,
                indent=2,
            )
    except OSError as exc:
        logger.warning(
            "local_history_mirror.finalize failed for %s: %s", job_id, exc
        )
