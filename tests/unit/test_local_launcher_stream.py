"""Regression test for LocalLauncher._stream_output oversized-line handling.

The final `OPTIMIZATION_RESULT:{json}` marker serializes the full iteration
history onto a single line and routinely exceeds asyncio's 64 KiB
StreamReader separator limit. The old `async for raw_line in proc.stdout`
loop raised `LimitOverrunError` ("Separator is not found, and chunk exceed
the limit") on such a line, crashing the stream task and dropping the
result. The chunked reader must parse the marker regardless of line length.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from src.launchers.local_launcher import LocalLauncher


class _FakeProc:
    """Minimal stand-in for asyncio.subprocess.Process feeding a StreamReader."""

    def __init__(self, data: bytes):
        # Default limit (64 KiB) mirrors create_subprocess_exec — the whole
        # point is that chunked .read() ignores it. Built inside the running
        # loop so the StreamReader binds to it.
        self.stdout = asyncio.StreamReader()
        self.stdout.feed_data(data)
        self.stdout.feed_eof()
        self.returncode = 0

    async def wait(self):
        return self.returncode


class _FakeEnv:
    def stop(self):
        pass


def _run_stream(tmp_path, monkeypatch, data: bytes) -> str:
    # Neutralize the process-registry side effects.
    from src.local import process_registry
    monkeypatch.setattr(process_registry, "unregister", lambda job_id: None)

    launcher = LocalLauncher()
    log_path = tmp_path / "job.log"

    async def _go():
        proc = _FakeProc(data)  # StreamReader must bind to the running loop
        await launcher._stream_output("job_test", proc, log_path, _FakeEnv())

    asyncio.run(_go())
    return log_path.read_text()


def test_stream_parses_oversized_optimization_result(tmp_path, monkeypatch, caplog):
    """A >64 KiB marker line is parsed without raising LimitOverrunError."""
    big_history = {str(i): {"change_request": "x" * 2000} for i in range(60)}
    payload = {"success": True, "history": big_history}
    marker = "OPTIMIZATION_RESULT:" + json.dumps(payload, separators=(",", ":"))
    assert len(marker.encode()) > 65536, "test line must exceed the 64 KiB limit"

    data = (b"some normal log line\n" + marker.encode() + b"\n")
    log_text = _run_stream(tmp_path, monkeypatch, data)

    # The oversized marker must have been written to the log (proving the
    # stream loop did not crash on it) and no stream error logged.
    assert "OPTIMIZATION_RESULT:" in log_text
    assert "some normal log line" in log_text
    assert "Error streaming optimizer output" not in caplog.text


def test_stream_handles_unterminated_final_line(tmp_path, monkeypatch, caplog):
    """A marker with no trailing newline (EOF mid-line) is still flushed."""
    marker = 'OPTIMIZATION_RESULT:{"success":true}'
    data = b"line one\n" + marker.encode()  # no trailing \n
    log_text = _run_stream(tmp_path, monkeypatch, data)

    assert "OPTIMIZATION_RESULT:" in log_text
    assert "Error streaming optimizer output" not in caplog.text
