"""Copyright © 2026 440 Labs LLC

Resumable per-row score store: `scores/<bench>/rows.jsonl` + `evals.jsonl`.

`ScoreStore` keys a row line by `(job_id, tree, split, row_id, draw)`;
`AuditStore` keys by `(job_id, tree, purpose, row_id, draw)` so the audit's
phases (screen / select / confirm / final) never alias: a resumed stage only
fills keys that are missing, and a cached observation can never be
re-labelled as a new repetition. A stage consults `missing_rows` before
spending and appends after every `run_eval` call, so an interrupted stage
resumes where it stopped.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

ROW_KEY = ("job_id", "tree", "split", "row_id", "draw")
AUDIT_ROW_KEY = ("job_id", "tree", "purpose", "row_id", "draw")


class ScoreStore:
    def __init__(self, rows_path: Path | str, evals_path: Path | str) -> None:
        self.rows_path = Path(rows_path)
        self.evals_path = Path(evals_path)
        self._rows: dict[tuple, dict] = {}
        self._evals: list[dict] = []
        self._load()

    # -- persistence -------------------------------------------------------
    def _load(self) -> None:
        if self.rows_path.exists():
            for line in self.rows_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._rows[self._key(rec)] = rec  # later lines win (fill overrides)
        if self.evals_path.exists():
            for line in self.evals_path.read_text().splitlines():
                if line.strip():
                    try:
                        self._evals.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    @staticmethod
    def _key(rec: dict) -> tuple:
        return (rec["job_id"], rec["tree"], rec["split"], int(rec["row_id"]), int(rec["draw"]))

    def _write(self, eval_line: dict, row_lines: Iterable[dict], *, override: bool) -> int:
        self.rows_path.parent.mkdir(parents=True, exist_ok=True)
        added = 0
        with open(self.rows_path, "a") as f:
            for rec in row_lines:
                k = self._key(rec)
                if k in self._rows and not override:
                    continue
                self._rows[k] = rec
                f.write(json.dumps(rec) + "\n")
                added += 1
            f.flush()
            os.fsync(f.fileno())
        with open(self.evals_path, "a") as f:
            self._evals.append(eval_line)
            f.write(json.dumps(eval_line) + "\n")
        return added

    def append_eval(self, eval_line: dict, row_lines: Iterable[dict]) -> int:
        """Append one eval and its rows (idempotent by key). Returns rows added."""
        return self._write(eval_line, row_lines, override=False)

    # -- queries -----------------------------------------------------------
    def have(self, job_id: str, tree: str, split: str, row_id: int, draw: int) -> bool:
        return (job_id, tree, split, int(row_id), int(draw)) in self._rows

    def missing_rows(
        self, job_id: str, tree: str, split: str, row_ids: Iterable[int], draw: int
    ) -> list[int]:
        return [r for r in row_ids if not self.have(job_id, tree, split, r, draw)]

    def draws_for(self, job_id: str, tree: str, split: str, row_id: int) -> list[int]:
        return sorted(
            k[4] for k in self._rows
            if k[0] == job_id and k[1] == tree and k[2] == split and k[3] == int(row_id)
        )

    def matrix(
        self, job_id: str, tree: str, split: str, *, failed_as_zero: bool = False
    ) -> dict[int, list[float]]:
        """`row_id -> scores by draw` (draw order). Failed rows are dropped
        unless `failed_as_zero`."""
        out: dict[int, list[tuple[int, float]]] = defaultdict(list)
        for k, rec in self._rows.items():
            if k[0] != job_id or k[1] != tree or k[2] != split:
                continue
            if rec.get("failed") and not failed_as_zero:
                continue
            score = rec.get("score")
            if score is None:
                if not failed_as_zero:
                    continue
                score = 0.0
            out[k[3]].append((k[4], float(score)))
        return {rid: [s for _, s in sorted(v)] for rid, v in out.items()}

    def rows(self) -> list[dict]:
        return list(self._rows.values())

    def evals(self) -> list[dict]:
        return list(self._evals)

    def testset_hashes(self) -> set[str]:
        return {r["testset_sha256"] for r in self._rows.values()
                if r.get("split") == "test" and r.get("testset_sha256")}

    def per_row_latency(self) -> float | None:
        """Median seconds per row over recorded evals (for cost estimates)."""
        lat = [e["elapsed_s"] / e["n_rows"] for e in self._evals
               if e.get("elapsed_s") and e.get("n_rows")]
        if not lat:
            return None
        lat.sort()
        return lat[len(lat) // 2]


class AuditStore(ScoreStore):
    """Phase-keyed store: `(job_id, tree, purpose, row_id, draw)`.

    `purpose` is the audit phase (`screen`, `select`, `confirm`, `final`).
    The `split` positional of the base-class queries is reinterpreted as the
    phase so `evaluate.execute_tasks` needs no changes.
    """

    def __init__(self, rows_path: Path | str, evals_path: Path | str,
                 attempts_path: Path | str | None = None) -> None:
        self.attempts_path = Path(attempts_path) if attempts_path else Path(evals_path).with_name("attempts.jsonl")
        super().__init__(rows_path, evals_path)

    @staticmethod
    def _key(rec: dict) -> tuple:
        return (rec["job_id"], rec["tree"], rec["purpose"], int(rec["row_id"]), int(rec["draw"]))

    def append_fill(self, eval_line: dict, row_lines: Iterable[dict]) -> int:
        """Apply a fill pass: only rows whose stored record is `failed` are
        replaced (the new line carries `fill=True`). Returns rows replaced."""
        replace = []
        for rec in row_lines:
            k = self._key(rec)
            prior = self._rows.get(k)
            if prior is None or prior.get("failed"):
                rec = dict(rec, fill=True)
                replace.append(rec)
        return self._write(eval_line, replace, override=True)

    def record(self, job_id: str, tree: str, phase: str, row_id: int, draw: int) -> dict | None:
        return self._rows.get((job_id, tree, phase, int(row_id), int(draw)))

    def failed_rows(self, job_id: str, tree: str, phase: str, draw: int) -> list[int]:
        return sorted(k[3] for k, rec in self._rows.items()
                      if k[0] == job_id and k[1] == tree and k[2] == phase and k[4] == int(draw)
                      and rec.get("failed"))

    def scores(self, job_id: str, tree: str, phase: str, draw: int) -> dict[int, float]:
        """`row_id -> score` for one (tree, phase, draw); failed rows keep the
        evaluator's score (0.0 when it recorded none)."""
        out: dict[int, float] = {}
        for k, rec in self._rows.items():
            if k[0] == job_id and k[1] == tree and k[2] == phase and k[4] == int(draw):
                s = rec.get("score")
                out[k[3]] = float(s) if s is not None else 0.0
        return out

    def log_attempt(self, **fields: Any) -> None:
        self.attempts_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.attempts_path, "a") as f:
            f.write(json.dumps({"ts": time.time(), **fields}) + "\n")


def row_line(
    *, bench: str, job_id: str, arm: str, tree: str, sha: str, split: str,
    row_id: int, draw: int, score: float | None, failed: bool, purpose: str,
    eval_id: str, testset_sha256: str | None, trainset_sha256: str | None,
) -> dict[str, Any]:
    return {
        "bench": bench, "job_id": job_id, "arm": arm, "tree": tree, "sha": sha,
        "split": split, "row_id": int(row_id), "draw": int(draw), "score": score,
        "failed": bool(failed), "purpose": purpose, "eval_id": eval_id,
        "ts": time.time(), "testset_sha256": testset_sha256,
        "trainset_sha256": trainset_sha256,
    }


__all__ = ["AUDIT_ROW_KEY", "ROW_KEY", "AuditStore", "ScoreStore", "row_line"]
