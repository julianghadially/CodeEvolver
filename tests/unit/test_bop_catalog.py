"""Unit tests for scripts/best_of_population/catalog.py."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.best_of_population.catalog import (
    candidates_from_run,
    discover_runs,
)
from scripts.best_of_population.config import AnalysisConfig

pytestmark = pytest.mark.unit


def _run(root: Path, run_id: str, *, job_id: str, arm: str, score: float | None, status: str = "completed",
         final_sha: str = "f" * 40, with_bundle: bool = True, candidates: list[dict] | None = None) -> Path:
    d = root / run_id
    d.mkdir(parents=True)
    st = {"run_id": run_id, "bench": "hover", "arm": arm, "seed": "1", "job_id": job_id, "status": status,
          "final_candidate": {"git_branch": "b", "git_sha": final_sha}, "final_test_score": score,
          "final_test_scores": [score] if score is not None else [], "seed_test_scores": [0.5]}
    (d / "status.json").write_text(json.dumps(st))
    if with_bundle:
        (d / "repo.bundle").write_bytes(b"x")
    if candidates is not None:
        (d / "candidates.jsonl").write_text("\n".join(json.dumps(c) for c in candidates) + "\n")
    return d


def test_discover_keys_by_job_id_and_prefers_collected_copy(tmp_path):
    runs_home, data_runs = tmp_path / "ce-runs", tmp_path / "ce-data"
    _run(runs_home, "ee-hover-s1", job_id="job_a", arm="ee", score=0.8)
    live = _run(data_runs, "ee-hover-s1", job_id="job_a", arm="ee", score=0.8, with_bundle=False)
    (live / ".codeevolver" / "traces" / "job_a").mkdir(parents=True)
    _run(data_runs, "ee-hover-s1-old", job_id="job_b", arm="ee", score=0.7)
    (runs_home / "repos").mkdir()
    (runs_home / "no-status").mkdir()
    cfg = AnalysisConfig(out=str(tmp_path / "out"), runs_home=str(runs_home), data_runs=str(data_runs),
                         ce_home=str(tmp_path / "ce-home"))
    runs = discover_runs(cfg)
    assert sorted(r.job_id for r in runs) == ["job_a", "job_b"]
    a = next(r for r in runs if r.job_id == "job_a")
    assert a.artifacts_dir == str(runs_home / "ee-hover-s1")
    assert a.traces_dir == str(live / ".codeevolver" / "traces" / "job_a")  # borrowed from the live tree


def test_candidates_filter_skipped_seed_and_always_include_final(tmp_path):
    root = tmp_path / "ce-runs"
    cands = [
        {"iteration": 0, "close_sha": "s" * 40, "skipped": False, "branch": "main"},
        {"iteration": 1, "close_sha": "a" * 40, "skipped": False, "branch": "main", "is_primary": True},
        {"iteration": 2, "close_sha": None, "skipped": True, "branch": "main"},
        {"iteration": 3, "close_sha": "b" * 40, "skipped": False, "branch": "main"},
    ]
    _run(root, "ee-hover-s1", job_id="j1", arm="ee", score=0.8, final_sha="c" * 40, candidates=cands)
    cfg = AnalysisConfig(out=str(tmp_path / "out"), runs_home=str(root), data_runs=str(tmp_path / "none"))
    run = discover_runs(cfg)[0]
    out = candidates_from_run(cfg, run)
    assert [c.iteration for c in out] == [1, 3, -1]
    assert out[-1].is_final and out[-1].source == "final"
    cfg_seed = cfg.with_overrides(include_seed=True)
    assert [c.iteration for c in candidates_from_run(cfg_seed, run)][0] == 0
