"""Write helpers for the `jobs` collection.

The API's internal callback handlers (`/internal/job/{id}/status`,
`/internal/job/{id}/progress`) need to update Mongo and — in local mode —
mirror the same record to a JSONL file before the Mongo write so a Mongo
crash never strands the operator's iteration history. Doing that pairing
inline in `src/api/main.py` makes the handlers carry dotted-key upsert
details and the cloud/local distinction; centralizing it here lets the
handlers stay declarative.

In cloud mode `local_history_mirror.*` are no-ops, so the only side
effect is the Mongo `update_one`.
"""
from __future__ import annotations

import logging
from typing import Any

from src.db.mongo import get_database
from src.local import local_history_mirror

logger = logging.getLogger(__name__)


async def record_iteration_progress(
    job_id: str,
    base_update: dict[str, Any],
    raw_history: dict[str, Any] | None,
) -> None:
    """Persist one progress callback worth of state.

    `base_update` carries the flat top-level fields (`current_iteration`,
    `final_train_score`, `memory_md`, …). `raw_history` is the optimizer's
    `ce_state["iteration_history"]` (iteration index → record); it may
    be `None` when no iteration completed this tick.

    When `raw_history` is non-empty: we write the JSONL mirror first
    (write-ahead), then upsert each iteration under a dotted key so
    out-of-order callbacks never clobber earlier entries, then re-merge
    the full history into `ce_state.iteration_history` via an
    aggregation pipeline (so legacy jobs with `ce_state: null` don't
    500 the handler).
    """
    db = get_database()

    if raw_history:
        # Local-mode JSONL mirror — write-ahead before Mongo so iteration
        # history survives a Mongo crash. No-op on cloud.
        local_history_mirror.append_iteration(job_id, raw_history, base_update)

        job = await db.jobs.find_one(
            {"job_id": job_id},
            {"ce_state_history": 1},
        )
        if job and job.get("ce_state_history") is None:
            await db.jobs.update_one(
                {"job_id": job_id}, {"$set": {"ce_state_history": {}}}
            )

        history_update: dict[str, object] = {
            f"ce_state_history.{str(idx_key)}": record
            for idx_key, record in raw_history.items()
        }
        if history_update:
            await db.jobs.update_one({"job_id": job_id}, {"$set": history_update})
            logger.info(
                "[PROGRESS] Upserted %d iteration(s) for %s",
                len(history_update),
                job_id,
            )

        # Aggregation-pipeline update so `ce_state` is coalesced to {}
        # when null (legacy jobs inserted before the default became a
        # dict). Without this, Mongo rejects the dotted $set with
        # "cannot create field ... in element {ce_state: null}", which
        # 500s the handler and leaves `current_iteration` stale.
        await db.jobs.update_one(
            {"job_id": job_id},
            [
                {
                    "$set": {
                        "ce_state": {
                            "$mergeObjects": [
                                {"$ifNull": ["$ce_state", {}]},
                                {"iteration_history": raw_history},
                            ]
                        }
                    }
                }
            ],
        )

    result = await db.jobs.update_one({"job_id": job_id}, {"$set": base_update})
    logger.info(
        "[PROGRESS] MongoDB update for %s: matched=%d, modified=%d",
        job_id,
        result.matched_count,
        result.modified_count,
    )


async def record_job_status_update(
    job_id: str,
    status: str,
    update: dict[str, Any],
) -> None:
    """Persist a status-callback update.

    For terminal statuses (`completed` / `failed`) the assembled `update`
    is also snapshot-written to the local-mode JSONL mirror before the
    Mongo write, so a Mongo failure still leaves the operator with the
    final summary on disk.
    """
    if status in ("completed", "failed"):
        local_history_mirror.finalize(job_id, status, update)

    db = get_database()
    await db.jobs.update_one({"job_id": job_id}, {"$set": update})
