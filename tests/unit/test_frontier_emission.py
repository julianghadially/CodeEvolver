"""Engine-side emission for the offline analyses
(read by scripts/best_of_population/): per-row score sidecar,
evals-log dual write, dataset manifest, collect.py refspecs, and the
per-call evaluator command file."""

from __future__ import annotations

import json
import os
import subprocess
import threading
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def job_dirs(tmp_path, monkeypatch):
    monkeypatch.setenv("SANDBOX_MODE", "local")
    monkeypatch.setenv("CODEEVOLVER_HISTORY_DIR", str(tmp_path / "history"))
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("CODEEVOLVER_JOB_ID", "job_t")
    return tmp_path


def test_row_scores_sidecar_and_evals_dual_write(job_dirs):
    from src.engine.evaluator.row_scores import append_row_scores_line, row_scores_log_path
    from src.engine.optimizer.v3.eval_log import append_eval_log_line, evals_log_path, history_evals_log_path

    append_eval_log_line(3, {"eval_id": "e1", "score": 0.5})
    assert Path(evals_log_path(3)).exists() and Path(history_evals_log_path(3)).exists()
    assert json.loads(Path(history_evals_log_path(3)).read_text())["eval_id"] == "e1"
    assert str(job_dirs / "history" / "job_t") in history_evals_log_path(3)

    append_row_scores_line({"eval_id": "e1", "split": "test", "row_scores": {"0": 1.0}})
    lines = Path(row_scores_log_path()).read_text().splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["row_scores"] == {"0": 1.0}


def test_record_eval_row_scores_reads_git_identity(job_dirs):
    from src.engine.evaluator.row_scores import record_eval_row_scores, row_scores_log_path

    ws = job_dirs / "ws"
    ws.mkdir()
    subprocess.run(["git", "-C", str(ws), "init", "-q"], check=True)
    subprocess.run(["git", "-C", str(ws), "-c", "user.email=t@t", "-c", "user.name=t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)

    class R:
        primary_score = 0.5
        row_ids = [0, 1]
        trace_meta = [{"example_id": 0, "score": 1.0, "failed": False},
                      {"example_id": 1, "score": 0.0, "failed": True}]

    record_eval_row_scores(workspace=str(ws), result=R(), iteration=1, eval_sample="holdout",
                           suffix="holdout", split="test")
    rec = json.loads(Path(row_scores_log_path()).read_text().splitlines()[0])
    assert rec["row_scores"] == {"0": 1.0, "1": 0.0} and rec["failed_rows"] == ["1"]
    assert rec["sha_at_eval"] and rec["tree"] and rec["split"] == "test" and rec["n_rows"] == 2


def test_dataset_manifest_hashes_and_comparison_ids(tmp_path):
    from src.engine.evaluator.dataset_config import DatasetConfig

    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "train.json").write_text(json.dumps([{"q": i} for i in range(10)]))
    (tmp_path / "data" / "test.json").write_text(json.dumps([{"q": i} for i in range(3)]))
    ds = DatasetConfig.prepare(str(tmp_path), trainset_json=None, trainset_path="data/train.json",
                               comparison_set_size=4, testset_path="data/test.json")
    m = ds.manifest(str(tmp_path))
    assert m["comparison_ids"] == ds.comparison_ids and len(m["comparison_ids"]) == 4
    assert m["trainset_size"] == 10 and m["testset_size"] == 3
    assert len(m["trainset_sha256"]) == 64 and len(m["testset_sha256"]) == 64
    ds.write_manifest(str(tmp_path), str(tmp_path / "h" / "datasets.json"))
    assert json.loads((tmp_path / "h" / "datasets.json").read_text()) == m


def test_collect_refspecs_include_job_anchors():
    from orchestrator.collect import bundle_refspecs

    assert bundle_refspecs("ee-hover-s1", None) == ["refs/heads/*:refs/runs/ee-hover-s1/*"]
    specs = bundle_refspecs("ee-hover-s1", "job_x")
    assert "refs/codeevolver/*:refs/jobs/job_x/codeevolver/*" in specs
    assert "refs/heads/*:refs/jobs/job_x/heads/*" in specs


def test_run_evaluation_uses_a_private_command_file(tmp_path, monkeypatch):
    """Two concurrent calls must not share a command file."""
    from src.engine.services import subprocess_service as ss

    seen: list[str] = []
    lock = threading.Lock()

    def fake_run(cmd, **kw):
        path = cmd[cmd.index("--command-file") + 1]
        with lock:
            seen.append(path)
        payload = json.loads(Path(path).read_text())
        # simulate a slow evaluator so the calls overlap
        threading.Event().wait(0.05)

        class P:
            returncode = 0
            stdout = "EVAL_RESULT:" + json.dumps({"success": True, "echo": payload["tag"]})
            stderr = ""
        return P()

    monkeypatch.setattr(ss.subprocess, "run", fake_run)
    results: dict[str, dict] = {}

    def call(tag):
        results[tag] = ss.run_evaluation(str(tmp_path), {"tag": tag})

    ts = [threading.Thread(target=call, args=(t,)) for t in ("a", "b")]
    [t.start() for t in ts]; [t.join() for t in ts]
    assert results["a"]["echo"] == "a" and results["b"]["echo"] == "b"
    assert len(set(seen)) == 2 and all(not os.path.exists(p) for p in seen)
    assert "/tmp/prebuilt_eval_command.json" not in seen
