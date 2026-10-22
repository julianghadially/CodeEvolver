"""`run_evaluation` persists the eval subprocess's stderr next to the traces.

Stderr is kept for SUCCESSFUL evals too, so lines such as the seed programs'
`[WARNING] GMI error ... retrying this call on DeepInfra` (printed by the
client's LM wrapper) are not discarded.
"""

import json
import os
import subprocess
from types import SimpleNamespace

import pytest

from src.engine.services import subprocess_service as svc

FALLBACK_LINE = (
    "[WARNING] GMI error on openai/deepseek-ai/DeepSeek-V4-Flash (status=402, "
    "APIError: litellm.APIError: OpenAIException - Error code: 402 - "
    "{'error': 'Insufficient balance', 'reason': 'model_access_denied'}) "
    "-- retrying this call on DeepInfra (deepinfra/deepseek-ai/DeepSeek-V4-Flash)\n"
    "[WARNING] DeepInfra fallback succeeded in 61.2s (gmi_status=402)\n"
)


def _fake_run(payload: dict, stderr: str, returncode: int = 0):
    def run(cmd, **kwargs):
        stdout = "noise\nEVAL_RESULT:" + json.dumps(payload) + "\n"
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)

    return run


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    ws = tmp_path / "ws"
    ws.mkdir()
    monkeypatch.setattr(svc, "_build_env", lambda workspace: dict(os.environ))
    return ws


def test_successful_eval_stderr_lands_in_the_trace_dir(workspace, tmp_path, monkeypatch):
    trace_dir = tmp_path / "traces" / "iteration_3_child"
    trace_dir.mkdir(parents=True)
    payload = {"success": True, "trace_dir": str(trace_dir), "scores_by_metric": {"m": 0.5}}
    monkeypatch.setattr(subprocess, "run", _fake_run(payload, FALLBACK_LINE))

    result = svc.run_evaluation(str(workspace), {"kind": "eval"})

    assert result == payload
    log = trace_dir / svc.EVAL_STDERR_FILENAME
    assert log.read_text() == FALLBACK_LINE


def test_quiet_eval_writes_no_file(workspace, tmp_path, monkeypatch):
    trace_dir = tmp_path / "iteration_3_child"
    trace_dir.mkdir()
    payload = {"success": True, "trace_dir": str(trace_dir)}
    monkeypatch.setattr(subprocess, "run", _fake_run(payload, ""))

    assert svc.run_evaluation(str(workspace), {}) == payload
    assert not (trace_dir / svc.EVAL_STDERR_FILENAME).exists()


def test_missing_trace_dir_is_not_fatal(workspace, tmp_path, monkeypatch):
    payload = {"success": True, "trace_dir": str(tmp_path / "does-not-exist")}
    monkeypatch.setattr(subprocess, "run", _fake_run(payload, FALLBACK_LINE))
    assert svc.run_evaluation(str(workspace), {}) == payload

    payload = {"success": False, "error": "boom"}  # no trace_dir key at all
    monkeypatch.setattr(subprocess, "run", _fake_run(payload, FALLBACK_LINE))
    assert svc.run_evaluation(str(workspace), {}) == payload


def test_oversized_stderr_keeps_the_tail(workspace, tmp_path, monkeypatch):
    trace_dir = tmp_path / "iteration_9_parent"
    trace_dir.mkdir()
    monkeypatch.setattr(svc, "EVAL_STDERR_MAX_BYTES", 100)
    stderr = ("x" * 500) + "\n" + FALLBACK_LINE  # the interesting part is at the END
    payload = {"success": True, "trace_dir": str(trace_dir)}
    monkeypatch.setattr(subprocess, "run", _fake_run(payload, stderr))

    svc.run_evaluation(str(workspace), {})

    text = (trace_dir / svc.EVAL_STDERR_FILENAME).read_text()
    assert text.startswith("[truncated: kept the last 100 of")
    assert text.endswith(stderr[-100:])


def test_failed_script_still_reports_stderr_in_the_error(workspace, monkeypatch):
    monkeypatch.setattr(subprocess, "run", _fake_run({}, "Traceback ... KeyError: 'GMI_API_KEY'", returncode=1))
    result = svc.run_evaluation(str(workspace), {})
    assert result["success"] is False
    assert "KeyError: 'GMI_API_KEY'" in result["error"]
