"""Copyright © 2026 440 Labs LLC

Unit tests for `src.engine.telemetry.trace_json` (the chunked pretty-JSON
writer) and the harness gate that turns chunking on.
"""

from __future__ import annotations

import json
import random

import pytest

from src.engine.paths import OPENCODE_TRACE_MAX_LINE_CHARS, trace_max_line_chars
from src.engine.telemetry.trace_json import (
    CHUNKED_KEY,
    chunk_long_strings,
    dump_trace_json,
    escaped_len,
    join_chunked_strings,
    split_string,
)


def test_escaped_width_is_what_lands_on_disk():
    text = 'a"b\\c\nd\x01é東'
    assert escaped_len(text) == len(json.dumps(text, ensure_ascii=False)) - 2


def _random_text(rng: random.Random, n: int) -> str:
    alphabet = "abc def\nghi \t 東京 🙂 "
    return "".join(rng.choice(alphabet) for _ in range(n))


@pytest.mark.parametrize("n", [0, 1, 1899, 1900, 1901, 5000, 300_000])
def test_split_string_roundtrips_and_respects_limit(n):
    rng = random.Random(n)
    text = _random_text(rng, n)
    pieces = split_string(text, 1900)
    assert "".join(pieces) == text
    assert all(escaped_len(p) <= 1900 for p in pieces)
    assert text == "" or all(pieces)  # no empty chunks


def test_split_string_prefers_whitespace_cut():
    text = "word " * 500  # 2500 chars
    pieces = split_string(text, 1900)
    assert pieces[0].endswith(" ")
    assert len(pieces[0]) <= 1900


def test_split_string_hard_cuts_without_whitespace():
    text = "x" * 4000
    assert split_string(text, 1900) == ["x" * 1900, "x" * 1900, "x" * 200]


def test_dump_trace_json_no_line_exceeds_limit():
    rng = random.Random(7)
    doc = {
        "prompt": _random_text(rng, 40_000),
        "nested": {"list": [_random_text(rng, 3000), 5, None, {"deep": "y" * 2500}]},
        "short": "ok",
    }
    text = dump_trace_json(doc, max_line_chars=1900)
    assert max(len(line) for line in text.splitlines()) < 2000
    assert join_chunked_strings(json.loads(text)) == doc


def test_dump_trace_json_unset_limit_is_plain_json():
    doc = {"prompt": "x" * 5000, "n": 1}
    assert dump_trace_json(doc, max_line_chars=None) == json.dumps(
        doc, indent=2, ensure_ascii=False, default=str
    )
    assert dump_trace_json(doc, max_line_chars=0) == dump_trace_json(
        doc, max_line_chars=None
    )


def test_chunk_marker_shape():
    out = chunk_long_strings({"a": "z" * 2000, "b": ["short"]}, 1900)
    assert set(out["a"]) == {CHUNKED_KEY}
    assert out["b"] == ["short"]
    assert join_chunked_strings(out) == {"a": "z" * 2000, "b": ["short"]}


def test_trace_max_line_chars_env(monkeypatch):
    monkeypatch.delenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", raising=False)
    assert trace_max_line_chars() is None
    monkeypatch.setenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", "0")
    assert trace_max_line_chars() is None
    monkeypatch.setenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", "junk")
    assert trace_max_line_chars() is None
    monkeypatch.setenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", "1900")
    assert trace_max_line_chars() == 1900
    assert OPENCODE_TRACE_MAX_LINE_CHARS < 2000


def test_run_entry_sets_gate_only_for_opencode(tmp_path, monkeypatch):
    from src.engine.optimizer.v3 import _entry
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    # Only the harness gate is under test; skip the full config schema.
    monkeypatch.setattr(OptimizationConfig, "from_dict", staticmethod(lambda d: d))
    captured: dict = {}

    def runner(*, config):
        captured["gate"] = trace_max_line_chars()
        return {"ok": True}

    def _run(coding_harness):
        monkeypatch.delenv("CODEEVOLVER_TRACE_MAX_LINE_CHARS", raising=False)
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps({
            "workspace": str(tmp_path),
            "coding_harness": coding_harness,
        }))
        monkeypatch.setattr("sys.argv", ["entry", "--config", str(cfg)])
        _entry.run_entry(runner, method="test")
        return captured["gate"]

    assert _run("opencode") == OPENCODE_TRACE_MAX_LINE_CHARS
    assert _run("claude") is None
    # The default harness is opencode, so an unset field gates too.
    assert _run(None) == OPENCODE_TRACE_MAX_LINE_CHARS
