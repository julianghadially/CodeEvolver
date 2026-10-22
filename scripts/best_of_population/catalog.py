"""Copyright © 2026 440 Labs LLC

Run discovery and candidate enumeration over collected run directories.

Reads only collected artifacts: `<run_dir>/status.json`, `settings.json`,
`candidates.jsonl` and the job trees under `<run_dir>/.codeevolver/history/
<job_id>/` (or `~/.codeevolver/history/<job_id>/`). Everything is keyed by
`job_id` — `run_id` (`<arm>-<bench>-s<seed>[-<label>]`) is reused across
re-dispatched attempts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from .config import AnalysisConfig, AuditConfig

RUN_DIR_SKIP = {"repos"}


@dataclass
class RunRecord:
    run_id: str
    bench: str
    arm: str
    seed: str
    job_id: str
    status: str
    artifacts_dir: str
    final_branch: str | None = None
    final_sha: str | None = None
    final_test_score: float | None = None
    final_test_scores: list[float] = field(default_factory=list)
    seed_test_score: float | None = None
    seed_test_scores: list[float] = field(default_factory=list)
    bundle_path: str | None = None
    history_dir: str | None = None
    traces_dir: str | None = None
    testset_sha256: str | None = None
    trainset_sha256: str | None = None
    note: str | None = None

    @property
    def usable(self) -> bool:
        return self.status == "completed" and bool(self.final_sha) and bool(self.bundle_path)


@dataclass
class CandidateRecord:
    job_id: str
    bench: str
    arm: str
    iteration: int
    branch: str | None
    sha: str
    source: str  # "close" | "seed" | "final"
    is_final: bool = False
    is_seed: bool = False
    tree: str | None = None
    reqhash: str | None = None
    dataset_drift: bool = False
    best_subsample_score: float | None = None
    comparison_best_score: float | None = None
    is_primary: bool | None = None


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _first_existing(*paths: Path) -> str | None:
    for p in paths:
        if p.exists():
            return str(p)
    return None


def _run_record_from_dir(cfg: AnalysisConfig | AuditConfig, run_dir: Path) -> RunRecord | None:
    status = _read_json(run_dir / "status.json")
    if not status:
        return None
    job_id = status.get("job_id")
    if not job_id:
        return None
    settings = _read_json(run_dir / "settings.json") or {}
    hashes = settings.get("dataset_hashes") or {}
    final = status.get("final_candidate") or {}
    ce_home = Path(cfg.ce_home).expanduser()
    history_dir = _first_existing(
        run_dir / ".codeevolver" / "history" / job_id, ce_home / "history" / job_id
    )
    traces_dir = _first_existing(
        run_dir / ".codeevolver" / "traces" / job_id, ce_home / "traces" / job_id
    )
    manifest = _read_json(Path(history_dir) / "datasets.json") if history_dir else None
    bundle = run_dir / "repo.bundle"
    return RunRecord(
        run_id=status.get("run_id") or run_dir.name,
        bench=status.get("bench") or "",
        arm=status.get("arm") or "",
        seed=str(status.get("seed") or ""),
        job_id=job_id,
        status=status.get("status") or "unknown",
        artifacts_dir=str(run_dir),
        final_branch=final.get("git_branch"),
        final_sha=final.get("git_sha"),
        final_test_score=status.get("final_test_score"),
        final_test_scores=list(status.get("final_test_scores") or []),
        seed_test_score=status.get("seed_test_score"),
        seed_test_scores=list(status.get("seed_test_scores") or []),
        bundle_path=str(bundle) if bundle.exists() else None,
        history_dir=history_dir,
        traces_dir=traces_dir,
        testset_sha256=(manifest or {}).get("testset_sha256") or hashes.get("testset"),
        trainset_sha256=(manifest or {}).get("trainset_sha256") or hashes.get("trainset"),
    )


def discover_runs_in(cfg: AnalysisConfig | AuditConfig, roots: Iterable[Path | str]) -> list[RunRecord]:
    """Every run with a `status.json` under `roots`, one record per job_id
    (first root wins; a later copy only lends a missing bundle/traces dir)."""
    seen: dict[str, RunRecord] = {}
    for root in roots:
        root = Path(root).expanduser()
        if not root.is_dir():
            continue
        for run_dir in sorted(root.iterdir()):
            if not run_dir.is_dir() or run_dir.name in RUN_DIR_SKIP:
                continue
            rec = _run_record_from_dir(cfg, run_dir)
            if rec is None:
                continue
            prior = seen.get(rec.job_id)
            if prior is None:
                seen[rec.job_id] = rec
            else:
                if not prior.traces_dir and rec.traces_dir:
                    prior.traces_dir = rec.traces_dir
                if not prior.bundle_path and rec.bundle_path:
                    prior.bundle_path = rec.bundle_path
    runs = list(seen.values())
    if cfg.benches:
        runs = [r for r in runs if r.bench in cfg.benches]
    return runs


def discover_runs(cfg: AnalysisConfig | AuditConfig) -> list[RunRecord]:
    if isinstance(cfg, AuditConfig):
        return discover_runs_in(cfg, cfg.runs_roots)
    return discover_runs_in(cfg, (cfg.runs_home, cfg.data_runs))


def candidates_from_run(cfg: AnalysisConfig | AuditConfig, run: RunRecord) -> list[CandidateRecord]:
    """Closed-state candidates from `candidates.jsonl` (iteration >= 1 unless
    `include_seed`, non-skipped, with a close sha) plus the final program when
    its sha is not among them. Trees are resolved later by `candidates.
    materialize_arm`, which also dedupes by tree."""
    include_seed = bool(getattr(cfg, "include_seed", False))
    path = Path(run.artifacts_dir) / "candidates.jsonl"
    out: list[CandidateRecord] = []
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        it = int(rec.get("iteration", -1))
        sha = rec.get("close_sha") or rec.get("sha")
        if not sha or rec.get("skipped"):
            continue
        if it == 0 and not include_seed:
            continue
        out.append(
            CandidateRecord(
                job_id=run.job_id,
                bench=run.bench,
                arm=run.arm,
                iteration=it,
                branch=rec.get("branch"),
                sha=sha,
                source="seed" if it == 0 else "close",
                is_final=(sha == run.final_sha),
                is_seed=(it == 0),
                best_subsample_score=rec.get("best_subsample_score"),
                comparison_best_score=rec.get("comparison_best_score"),
                is_primary=rec.get("is_primary"),
            )
        )
    if run.final_sha and not any(c.sha == run.final_sha for c in out):
        out.append(
            CandidateRecord(
                job_id=run.job_id, bench=run.bench, arm=run.arm, iteration=-1,
                branch=run.final_branch, sha=run.final_sha, source="final",
                is_final=True,
            )
        )
    return out


__all__ = [
    "CandidateRecord",
    "RunRecord",
    "candidates_from_run",
    "discover_runs",
    "discover_runs_in",
]
