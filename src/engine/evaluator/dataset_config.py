"""Copyright © 2026 440 Labs LLC

Dataset configuration for the CodeEvolver optimizers (engine 3).

Encapsulates train/test resolution and the fixed comparison-set draw so the
optimizer and entry points never have to touch JSON files or sampling logic
directly. Keeping this out of `EvaluatorConfig` lets the mounted evaluator
stay a thin row-runner (it only needs a dataset path and row ids) while the
orchestrator owns the sizing + split decisions.

Engine 3 has no validation set. The comparison set is a fixed subsample of
the TRAINING set (seed 42) used by the architect for apples-to-apples branch
comparison; it overlaps the training data by design and counts against the
eval-row budget.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Canonical relative paths used when the caller supplies datasets inline as
# JSON. Paths provided by the client are respected as-is — these defaults
# only come into play for the inline-JSON path.
_INLINE_TRAINSET_REL = ".codeevolver/trainset.json"
_INLINE_TESTSET_REL = ".codeevolver/testset.json"

# Fixed seed for the comparison-set draw. Deliberately a constant (not the run
# seed): the comparison set must be identical across runs of the same dataset
# so branch scores stay comparable within and across runs.
COMPARISON_SET_SEED = 42

# `{history_dir}/datasets.json` — written once per run by the seed step.
DATASETS_MANIFEST_FILENAME = "datasets.json"


@dataclass
class DatasetConfig:
    """Orchestrator-side dataset resolution result.

    After `DatasetConfig.prepare()` runs, the trainset (and the optional test
    set) is guaranteed to live at its relative path inside the sandbox
    workspace, row id ranges are resolved, and the fixed comparison set (if
    any) is drawn once for the whole run from the trainset with seed
    `COMPARISON_SET_SEED` (42).
    """

    trainset_rel_path: str
    trainset_ids: list[int]
    # Fixed comparison set drawn from `trainset_ids`. Empty when
    # `comparison_set_size` was None/0 (comparison evals then unavailable).
    comparison_ids: list[int] = field(default_factory=list)
    # Held-out test set. Optional — `testset_rel_path` is "" and `testset_ids`
    # is empty when no test set was provided. Scored only at finalization
    # (seed + best candidate); never subsampled, never seen by agents.
    testset_rel_path: str = ""
    testset_ids: list[int] = field(default_factory=list)

    @property
    def trainset_size(self) -> int:
        return len(self.trainset_ids)

    @property
    def has_testset(self) -> bool:
        return bool(self.testset_ids)

    @property
    def testset_size(self) -> int:
        return len(self.testset_ids)

    @property
    def comparison_set_size(self) -> int:
        return len(self.comparison_ids)

    def manifest(self, workspace: str) -> dict[str, Any]:
        """Everything an offline analysis (`scripts/best_of_population/`) needs to re-address rows.

        Row ids are positional indices into the dataset files, so the files'
        sha256 is what makes an id comparable across runs and candidates; the
        comparison ids are persisted because nothing else records them (the
        draw is deterministic, but the analysis should not have to know the
        rule). Written once per run to `{history_dir}/datasets.json` by the
        seed step.
        """
        return {
            "trainset_rel_path": self.trainset_rel_path,
            "trainset_size": self.trainset_size,
            "trainset_sha256": file_sha256(workspace, self.trainset_rel_path),
            "testset_rel_path": self.testset_rel_path,
            "testset_size": self.testset_size,
            "testset_sha256": (
                file_sha256(workspace, self.testset_rel_path)
                if self.testset_rel_path
                else None
            ),
            "comparison_set_seed": COMPARISON_SET_SEED,
            "comparison_set_size": self.comparison_set_size,
            "comparison_ids": list(self.comparison_ids),
        }

    def write_manifest(self, workspace: str, path: str) -> None:
        """Best-effort write of `manifest()` to `path`."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f:
                json.dump(self.manifest(workspace), f, indent=2)
        except OSError:
            logger.warning("failed to write dataset manifest to %s", path, exc_info=True)

    @classmethod
    def prepare(
        cls,
        workspace: str,
        *,
        trainset_json: list | None,
        trainset_path: str | None,
        comparison_set_size: int | None = None,
        testset_json: list | None = None,
        testset_path: str | None = None,
    ) -> "DatasetConfig":
        """Resolve all datasets and derive the fixed comparison set.

        Either the inline JSON or a relative path must be provided for the
        train split; the test split is always optional. When inline JSON is
        provided, the data is written to a canonical location inside the
        workspace. When a path is provided, the file is read once to learn
        its length. The comparison set is drawn from the trainset with the
        fixed `COMPARISON_SET_SEED` and clamped to the trainset size;
        `comparison_set_size=None`/0 disables it (empty `comparison_ids`).
        """
        trainset_rel, trainset_len = _resolve_split(
            workspace,
            label="trainset",
            rows=trainset_json,
            path=trainset_path,
            inline_rel=_INLINE_TRAINSET_REL,
        )
        # Optional held-out test set. `_resolve_split` returns ("", 0) when
        # neither inline rows nor a path are supplied.
        testset_rel, testset_len = _resolve_split(
            workspace,
            label="testset",
            rows=testset_json,
            path=testset_path,
            inline_rel=_INLINE_TESTSET_REL,
            optional=True,
        )

        if trainset_len == 0:
            raise ValueError("trainset is empty — nothing to optimize over")

        trainset_ids = list(range(trainset_len))
        testset_ids = list(range(testset_len))

        if comparison_set_size:
            k = min(comparison_set_size, trainset_len)
            rng = random.Random(COMPARISON_SET_SEED)
            comparison_ids = sorted(rng.sample(trainset_ids, k))
            # Diagnostic: the draw is deterministic given trainset_len, so these
            # ids must be identical run-to-run for a fixed dataset. Logged so a
            # score swing can be attributed to program/metric nondeterminism
            # rather than a shifting comparison slice.
            logger.info(
                "comparison set: drew %d of %d trainset ids with seed=%d -> %s",
                k, trainset_len, COMPARISON_SET_SEED, comparison_ids,
            )
        else:
            comparison_ids = []
            logger.info(
                "comparison set: disabled (comparison_set_size=%s)", comparison_set_size,
            )

        return cls(
            trainset_rel_path=trainset_rel,
            trainset_ids=trainset_ids,
            comparison_ids=comparison_ids,
            testset_rel_path=testset_rel,
            testset_ids=testset_ids,
        )


def file_sha256(workspace: str, path: str) -> str | None:
    """sha256 hex of a dataset file (absolute or workspace-relative); None if unreadable."""
    if not path:
        return None
    full = path if os.path.isabs(path) else os.path.join(workspace, path)
    try:
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _resolve_split(
    workspace: str,
    *,
    label: str,
    rows: list | None,
    path: str | None,
    inline_rel: str,
    optional: bool = False,
) -> tuple[str, int]:
    """Return `(relative_path, row_count)` for one split.

    When `optional` is True and neither inline rows nor a path are supplied,
    returns `("", 0)` instead of raising — used for the held-out test set.
    """
    if rows is None and path is None and optional:
        return "", 0
    if rows is not None:
        if not isinstance(rows, list):
            raise ValueError(f"{label}_json must be a list of dicts")
        full_path = os.path.join(workspace, inline_rel)
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        with open(full_path, "w") as f:
            json.dump(rows, f)
        _log_schema(label, inline_rel, rows)
        return inline_rel, len(rows)
    if path:
        # Path may be absolute or relative to workspace
        full_path = path if os.path.isabs(path) else os.path.join(workspace, path)
        try:
            with open(full_path) as f:
                content = f.read()
        except FileNotFoundError:
            raise RuntimeError(f"failed to read {label} at {path}: file not found")
        if path.endswith(".jsonl"):
            loaded: Any = [
                json.loads(line) for line in content.splitlines() if line.strip()
            ]
        elif path.endswith(".csv"):
            # Each CSV row becomes a dict keyed by the header. Values stay
            # strings — the mounted evaluator (`evaluate.py::_load_dataset`)
            # parses the same file the same way, so the row schema matches.
            loaded = list(csv.DictReader(io.StringIO(content)))
        else:
            loaded = json.loads(content)
            if not isinstance(loaded, list):
                raise ValueError(f"{label} at {path} must be a list of dicts")
        _log_schema(label, path, loaded)
        return path, len(loaded)
    raise ValueError(f"{label}: provide either {label}_json or {label}_path")


def _log_schema(label: str, path: str, rows: list) -> None:
    """Log first-row keys + a truncated preview so schema mismatches are visible.

    The VALUE preview is suppressed for the testset. The holdout file itself is
    sequestered under /opt/holdout (root-owned, mode 700), but this log is
    written to the driver's HOME as mode 0644 and `.codeevolver/logs` is not on
    any deny path -- so previewing a test row would put a held-out row and its
    gold labels where the agent user can read them. Keys
    and row count are what make a schema mismatch visible; the values were
    never needed for it.
    """
    if not rows:
        logger.info("dataset %s (%s): empty", label, path)
        return
    first = rows[0]
    if isinstance(first, dict):
        keys = sorted(first.keys())
        if label == "testset":
            logger.info(
                "dataset %s (%s): n=%d keys=%s first_row=<withheld: holdout>",
                label, path, len(rows), keys,
            )
            return
        preview = {k: _truncate(first[k]) for k in list(first.keys())[:6]}
        logger.info(
            "dataset %s (%s): n=%d keys=%s first_row=%s",
            label, path, len(rows), keys, preview,
        )
    else:
        logger.info(
            "dataset %s (%s): n=%d non-dict row type=%s",
            label, path, len(rows), type(first).__name__,
        )


def _truncate(value: Any, limit: int = 80) -> Any:
    s = repr(value)
    return s if len(s) <= limit else s[:limit] + "…"


__all__ = ["DatasetConfig", "COMPARISON_SET_SEED", "DATASETS_MANIFEST_FILENAME", "file_sha256"]
