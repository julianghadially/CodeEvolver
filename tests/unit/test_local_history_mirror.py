"""Tests for the local JSONL history mirror.

The mirror must:
  - no-op when sandbox_mode != "local"
  - append one JSONL line per iteration record
  - write a final.json snapshot on terminal status
  - survive caller failures without raising (Mongo is primary; mirror is backup)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.local import local_history_mirror


@pytest.fixture
def history_root(tmp_path, monkeypatch):
    from src.config import settings
    root = tmp_path / "history"
    monkeypatch.setattr(settings, "local_history_root", root)
    monkeypatch.setattr(settings, "sandbox_mode", "local")
    return root


def test_append_iteration_writes_jsonl_lines(history_root):
    iteration_history = {
        "1": {"subsample_score": 0.6, "branch": "codeevolver-1"},
        "2": {"subsample_score": 0.7, "branch": "codeevolver-2"},
    }
    local_history_mirror.append_iteration(
        "job_abc", iteration_history, progress={"current_iteration": 2}
    )

    path = history_root / "job_abc.jsonl"
    assert path.exists()
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert len(lines) == 2
    assert {l["iteration"] for l in lines} == {"1", "2"}
    assert lines[0]["record"]["branch"] == "codeevolver-1"
    assert lines[0]["progress"]["current_iteration"] == 2
    assert "ts" in lines[0]


def test_append_iteration_is_appendable(history_root):
    local_history_mirror.append_iteration("job_xyz", {"1": {"v": 0.5}})
    local_history_mirror.append_iteration("job_xyz", {"2": {"v": 0.6}})
    path = history_root / "job_xyz.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines() if line]
    assert [l["iteration"] for l in lines] == ["1", "2"]


def test_finalize_writes_terminal_snapshot(history_root):
    summary = {"status": "completed", "best_score": 0.88}
    local_history_mirror.finalize("job_q", "completed", summary)
    path = history_root / "job_q.final.json"
    assert path.exists()
    blob = json.loads(path.read_text())
    assert blob["status"] == "completed"
    assert blob["summary"]["best_score"] == 0.88
    assert blob["job_id"] == "job_q"
    assert "ts" in blob


def test_mirror_noops_when_not_local(tmp_path, monkeypatch):
    from src.config import settings
    monkeypatch.setattr(settings, "local_history_root", tmp_path / "history")
    monkeypatch.setattr(settings, "sandbox_mode", "cloud")
    local_history_mirror.append_iteration("job_cloud", {"1": {"v": 0.5}})
    local_history_mirror.finalize("job_cloud", "completed", {})
    assert not (tmp_path / "history").exists()


def test_append_handles_empty_history(history_root):
    """A None or empty iteration_history should be a no-op (not crash)."""
    local_history_mirror.append_iteration("job_z", None)
    local_history_mirror.append_iteration("job_z", {})
    assert not (history_root / "job_z.jsonl").exists()
