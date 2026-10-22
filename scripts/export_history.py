#!/usr/bin/env python3
"""Reconstruct a job's iteration history from the local JSONL mirror.

Local-mode runs write each iteration to `<local_history_root>/{job_id}.jsonl`
before Mongo. If Mongo is unavailable (or you just want to inspect a run
without booting the API), this CLI emits the same shape that
`GET /job/{job_id}/iterations` would return on the cloud.

Usage:
    python scripts/export_history.py <job_id>
    python scripts/export_history.py <job_id> --history-root ~/.codeevolver/history
    python scripts/export_history.py <job_id> --pretty
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    entries: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(
                    f"[WARN] {path}:{line_no} skipped (invalid JSON): {exc}",
                    file=sys.stderr,
                )
    return entries


def reconstruct(job_id: str, history_root: Path) -> dict:
    """Replay the JSONL mirror into the `GET /job/{job_id}/iterations` shape.

    Later writes for the same iteration index win — matches the dotted-key
    upsert behavior the API uses against Mongo. The final snapshot, if
    present, contributes job-level summary fields.
    """
    jsonl_path = history_root / f"{job_id}.jsonl"
    final_path = history_root / f"{job_id}.final.json"

    iteration_history: dict[str, dict] = {}
    last_progress: dict = {}
    for entry in _load_jsonl(jsonl_path):
        idx = entry.get("iteration")
        if idx is None:
            continue
        iteration_history[str(idx)] = entry.get("record", {})
        progress = entry.get("progress")
        if progress:
            last_progress = progress

    summary: dict = {}
    status = "unknown"
    if final_path.exists():
        with final_path.open() as f:
            final_blob = json.load(f)
        summary = final_blob.get("summary", {}) or {}
        status = final_blob.get("status", "unknown")

    return {
        "job_id": job_id,
        "status": status,
        "iteration_history": iteration_history,
        "summary": summary,
        "last_progress": last_progress,
        "source": {
            "jsonl": str(jsonl_path),
            "final_json": str(final_path) if final_path.exists() else None,
        },
    }


def _default_history_root() -> Path:
    return Path.home() / ".codeevolver" / "history"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job_id", help="Job id to reconstruct (e.g., job_abc123).")
    parser.add_argument(
        "--history-root",
        type=Path,
        default=_default_history_root(),
        help="Directory holding {job_id}.jsonl + {job_id}.final.json "
             "(default: ~/.codeevolver/history).",
    )
    parser.add_argument(
        "--pretty", action="store_true",
        help="Pretty-print the JSON output.",
    )
    args = parser.parse_args()

    result = reconstruct(args.job_id, args.history_root)
    indent = 2 if args.pretty else None
    print(json.dumps(result, default=str, indent=indent))
    return 0


if __name__ == "__main__":
    sys.exit(main())
