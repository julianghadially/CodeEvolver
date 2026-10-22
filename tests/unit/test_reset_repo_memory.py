"""Unit tests for reset_repo's memory-root cleanup.

A full `reset_repo.py -D` reset must clear not only per-job artifact subdirs
(`memory/{job_id}`) but also any loose files written directly to the *root* of
the memory folder. Those appear when the architect writes memory without a
`CODEEVOLVER_JOB_ID` set (e.g. `memory.md`, `iteration_change_summaries.md`); they
are not job-scoped, so the per-job sweep never catches them and they
contaminate the next run.
"""
from __future__ import annotations

import os

import pytest

from scripts.reset_repo import clear_memory_root_files, delete_job_artifacts


@pytest.fixture
def memory_root(tmp_path, monkeypatch):
    """Point memory_dir() at a tmp folder with loose files + a job subdir."""
    mem = tmp_path / "memory"
    mem.mkdir()
    (mem / "memory.md").write_text("notes\n")
    (mem / "iteration_3_change_summary.md").write_text("hist\n")
    (mem / ".DS_Store").write_text("junk\n")
    job_dir = mem / "job_abc123"
    job_dir.mkdir()
    (job_dir / "memory.md").write_text("job notes\n")

    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(mem))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    return mem


def test_clears_loose_root_files(memory_root):
    clear_memory_root_files(dry_run=False)

    assert not (memory_root / "memory.md").exists()
    assert not (memory_root / "iteration_3_change_summary.md").exists()


def test_preserves_job_subdir_and_hidden_files(memory_root):
    clear_memory_root_files(dry_run=False)

    # Per-job subdir is left to delete_job_artifacts; dotfiles are skipped.
    assert (memory_root / "job_abc123" / "memory.md").exists()
    assert (memory_root / ".DS_Store").exists()


def test_dry_run_removes_nothing(memory_root):
    clear_memory_root_files(dry_run=True)

    assert (memory_root / "memory.md").exists()
    assert (memory_root / "iteration_3_change_summary.md").exists()


def test_missing_root_is_noop(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "absent"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    # Must not raise when the memory folder does not exist.
    clear_memory_root_files(dry_run=False)


def test_delete_job_artifacts_clears_architect_traces(tmp_path, monkeypatch):
    """`-D` must remove the per-job architect_traces subdir, not just
    traces/memory/runtime_tests — it's the sibling dir holding the
    IterationArchitect session traces (`iteration_{N}_architect_trace/`).
    """
    job_id = "job_abc123"
    arch = tmp_path / "architect_traces"
    job_dir = arch / job_id
    job_dir.mkdir(parents=True)
    (job_dir / "iteration_1_architect_trace").mkdir()
    (job_dir / "iteration_1_architect_trace" / "index.jsonl").write_text("turn\n")

    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(arch))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)

    delete_job_artifacts(job_id, dry_run=False)
    assert not job_dir.exists()
