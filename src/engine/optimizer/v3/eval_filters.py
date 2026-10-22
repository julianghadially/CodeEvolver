"""Copyright © 2026 440 Labs LLC

Row-filter parsing, matching and trainset loading for the `run_eval` tool.

Pure helpers: no git, no evaluator, no state. `RunEvalBinding` owns the per-iteration
filter LOCK and the loaded-rows cache; this module owns the rules.

A filter maps column name -> scalar (exact match) or `{op: value}` where op
is one of `eq`, `neq`, `in`, `contains` (case-insensitive substring).
Multiple columns AND together.
"""

from __future__ import annotations

import csv
import io
import json
import os
from collections.abc import Callable
from typing import Any

from .errors import RunEvalError

ALLOWED_OPS = frozenset({"eq", "neq", "in", "contains"})


def normalize_filter(
    filter: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Validate and canonicalize a filter dict.

    Returns None if filter is None or an empty dict. Scalar column values are
    expanded to `{"eq": value}` so downstream matching is uniform. Raises
    `RunEvalError` on malformed input.
    """
    if filter is None:
        return None
    if not isinstance(filter, dict):
        raise RunEvalError(
            f"filter must be a dict mapping column -> value (or op map), "
            f"got {type(filter).__name__}"
        )
    if not filter:
        return None
    normalized: dict[str, dict[str, Any]] = {}
    for col, spec in filter.items():
        if not isinstance(col, str):
            raise RunEvalError(
                f"filter column names must be strings, got {type(col).__name__}"
            )
        if isinstance(spec, dict):
            if not spec:
                raise RunEvalError(
                    f"filter[{col!r}] is an empty op map; use a scalar for "
                    "equality or specify one of eq/neq/in/contains"
                )
            bad = [op for op in spec if op not in ALLOWED_OPS]
            if bad:
                raise RunEvalError(
                    f"filter[{col!r}] has unknown operators {bad}; allowed: "
                    f"{sorted(ALLOWED_OPS)}"
                )
            for op, val in spec.items():
                if op == "in" and not isinstance(val, (list, tuple, set)):
                    raise RunEvalError(
                        f"filter[{col!r}]['in'] must be a list/tuple/set, "
                        f"got {type(val).__name__}"
                    )
                if op == "contains" and not isinstance(val, str):
                    raise RunEvalError(
                        f"filter[{col!r}]['contains'] must be a string, "
                        f"got {type(val).__name__}"
                    )
            # Canonicalize `in` to a sorted list so dict-equality lock checks
            # are order-insensitive.
            if "in" in spec:
                spec = dict(spec)
                spec["in"] = sorted(list(spec["in"]), key=lambda v: (type(v).__name__, v))
            normalized[col] = dict(spec)
        else:
            normalized[col] = {"eq": spec}
    return normalized


def row_matches(row: dict[str, Any], normalized_filter: dict[str, Any]) -> bool:
    """Evaluate a normalized filter against a single row (AND of all columns)."""
    for col, ops in normalized_filter.items():
        value = row.get(col)
        for op, target in ops.items():
            if op == "eq":
                if value != target:
                    return False
            elif op == "neq":
                if value == target:
                    return False
            elif op == "in":
                if value not in target:
                    return False
            elif op == "contains":
                if not isinstance(value, str):
                    return False
                if target.lower() not in value.lower():
                    return False
    return True


def load_trainset_rows(path: str | None, workspace: str) -> list[dict[str, Any]]:
    """Load trainset rows from `path` (resolved against `workspace` when relative).

    Supports `.json` (a list of dicts), `.jsonl` (one dict per line), and
    `.csv` (one dict per row, keyed by header), matching the mounted
    evaluator's conventions (`evaluate.py::_load_dataset`). Non-dict rows
    become empty dicts so filter lookups don't explode on malformed input.
    """
    if not path:
        raise RunEvalError(
            "filter requires EvaluatorConfig.trainset_path to be set"
        )
    full_path = path if os.path.isabs(path) else os.path.join(workspace, path)
    try:
        with open(full_path) as f:
            content = f.read()
    except FileNotFoundError as exc:
        raise RunEvalError(
            f"trainset not found at {full_path!r}; cannot apply filter"
        ) from exc

    if path.endswith(".jsonl"):
        loaded: Any = [
            json.loads(line) for line in content.splitlines() if line.strip()
        ]
    elif path.endswith(".csv"):
        # One dict per row keyed by the header. Values stay strings — the
        # mounted evaluator parses the same file the same way, so the row
        # schema (and thus filter columns) match.
        loaded = list(csv.DictReader(io.StringIO(content)))
    else:
        loaded = json.loads(content)
        if not isinstance(loaded, list):
            raise RunEvalError(
                f"trainset at {path!r} must be a list of dicts"
            )

    return [r if isinstance(r, dict) else {} for r in loaded]


def filtered_pool(
    trainset_ids: list[int],
    normalized_filter: dict[str, Any] | None,
    load_rows: Callable[[], list[dict[str, Any]]],
) -> list[int]:
    """Return the trainset ids that match `normalized_filter`.

    When no filter is set, returns the full trainset_ids without calling
    `load_rows`. When a filter is set, `load_rows()` supplies the trainset
    rows (the binding caches them) and must line up 1:1 with `trainset_ids`.
    """
    if not normalized_filter:
        return list(trainset_ids)
    rows = load_rows()
    if len(rows) != len(trainset_ids):
        raise RunEvalError(
            f"trainset file has {len(rows)} rows but binding was "
            f"configured with {len(trainset_ids)} ids; filter cannot "
            "be applied"
        )
    matched: list[int] = []
    for row_id, row in zip(trainset_ids, rows):
        if row_matches(row, normalized_filter):
            matched.append(row_id)
    return matched


def enforce_filter_lock(
    iteration_idx: int,
    locked_filter: dict[str, Any] | None,
    normalized_filter: dict[str, Any] | None,
) -> None:
    """Raise if the caller's filter differs from the locked filter."""
    if normalized_filter is None:
        return
    if normalized_filter == locked_filter:
        return
    raise RunEvalError(
        f"subsample filter is locked for iteration {iteration_idx} "
        f"(locked={locked_filter!r}, attempted={normalized_filter!r}); "
        "finish this iteration and change the filter on the next one"
    )


__all__ = [
    "ALLOWED_OPS",
    "enforce_filter_lock",
    "filtered_pool",
    "load_trainset_rows",
    "normalize_filter",
    "row_matches",
]
