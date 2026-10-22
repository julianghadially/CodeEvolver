"""Copyright © 2026 440 Labs LLC

Readers over one collected run directory's per-row records.

`<run>/.codeevolver/history/<job_id>/row_scores.jsonl` holds one line per
eval with `row_scores: {str(row_idx): score}` and `failed_rows`. The seed's
held-out eval is `iteration == 0, split == "test"`; the final program's
held-out draws are `iteration == 1` with suffixes `holdout`, `holdout_2`, ...
(`finalization.py` hard-codes those iteration indices).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from scripts.results_tables import RUN_ID_RE, required_label

from .catalog import RunRecord

ROW_SCORES = "row_scores.jsonl"


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def parse_run_id(run_id: str) -> dict[str, Any] | None:
    m = RUN_ID_RE.match(run_id)
    if not m:
        return None
    return {"arm": m.group("arm"), "bench": m.group("bench"), "seed": int(m.group("seed")),
            "label": m.group("label")}


def void_reason(run_id: str) -> str | None:
    """Why a run directory must not be read (results_tables' replacement
    table), or None when it is the directory of record for its cell."""
    parts = parse_run_id(run_id)
    if parts is None:
        return f"{run_id}: not a <arm>-<bench>-s<seed>[-<label>] run id"
    need = required_label(parts["bench"], parts["arm"], parts["seed"])
    if need != parts["label"]:
        if need:
            return f"void: cell reads only the -{need} directory"
        return f"ignored: unexpected label -{parts['label']}"
    return None


def row_score_lines(run: RunRecord) -> list[dict]:
    if not run.history_dir:
        return []
    path = Path(run.history_dir) / ROW_SCORES
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def seed_holdout_line(run: RunRecord) -> dict | None:
    lines = [l for l in row_score_lines(run) if l.get("split") == "test" and l.get("iteration") == 0]
    return lines[0] if lines else None


def final_holdout_lines(run: RunRecord) -> list[dict]:
    return [l for l in row_score_lines(run) if l.get("split") == "test" and l.get("iteration") == 1]


def settings_models(run: RunRecord) -> dict:
    s = read_json(Path(run.artifacts_dir) / "settings.json") or {}
    return s.get("models") or {}


def posthoc_rescore(run: RunRecord) -> dict | None:
    return read_json(Path(run.artifacts_dir) / "posthoc_holdout_rescore.json")


_SUFFIX_RE = re.compile(r"^holdout(?:_(\d+))?$")


def holdout_draw_index(suffix: str) -> int | None:
    m = _SUFFIX_RE.match(suffix or "")
    if not m:
        return None
    return int(m.group(1) or 1)


__all__ = [
    "final_holdout_lines", "holdout_draw_index", "parse_run_id", "posthoc_rescore", "read_json",
    "row_score_lines", "seed_holdout_line", "settings_models", "void_reason",
]
