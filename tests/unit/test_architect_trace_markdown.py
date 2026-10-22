"""Copyright © 2026 440 Labs LLC

Unit tests for the rendered architect-trace surfaces (`digest.md`,
`transcript.md`, `header.json`, run-level `README.md`). Fully offline: turns
are seeded through `architect_trace.append_architect_turn` / the OpenCode
stub writer, then rendered by `architect_trace_markdown`.
"""

from __future__ import annotations

import json
import os
import re

import pytest

from src.engine.optimizer.v3 import architect_trace as at
from src.engine.optimizer.v3 import architect_trace_markdown as md


@pytest.fixture
def trace_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_ARCHITECT_TRACES_DIR", str(tmp_path))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    return tmp_path


def _read(d, name):
    with open(os.path.join(d, name), encoding="utf-8") as f:
        return f.read()


def _run_eval_json(**over):
    base = {
        "ok": True, "eval_id": "9e9cb0eb53f1", "eval_sample": "train",
        "suffix": "parent", "sha": "36b3216fdaeeb975", "tree": "729fae923d5a4fd1",
        "row_ids": list(range(50)), "primary_score": 0.87032,
        "trace_dir": "/artifacts/traces/iteration_3_parent",
        "trace_meta": [
            {"example_id": i, "score": 0.0 if i < 2 else 0.99, "failed": i == 1}
            for i in range(50)
        ],
    }
    base.update(over)
    return json.dumps(base)


def _seed_basic(iteration=31):
    """Prompt, assistant (thinking+text+tool_use), run_eval result, error,
    inspect_traces, close, result."""
    at.append_architect_turn(
        iteration, role="user", blocks=[{"kind": "text", "text": "Improve the program."}],
    )
    at.append_architect_turn(
        iteration, role="assistant",
        blocks=[
            {"kind": "thinking", "text": "let me think " * 20},
            {"kind": "text", "text": "Now let me run the   baseline evaluation."},
            {"kind": "tool_use", "name": "ce_run_eval", "id": "c1", "input": {"eval_sample": "train"}},
            {"kind": "tool_use", "name": "bash", "id": "c2",
             "input": {"command": "git status", "workdir": "/w"}},
        ],
    )
    at.append_architect_turn(
        iteration, role="user",
        blocks=[
            {"kind": "tool_result", "tool_use_id": "c1", "is_error": False,
             "content": _run_eval_json()},
            {"kind": "tool_result", "tool_use_id": "c2", "is_error": True,
             "content": "fatal: not a git repository " * 3},
        ],
    )
    at.append_architect_turn(
        iteration, role="assistant",
        blocks=[{"kind": "tool_use", "name": "ce_inspect_traces", "id": "c3",
                 "input": {"query": "why did row 1 fail?", "trace": "parent"}}],
    )
    at.append_architect_turn(
        iteration, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "c3", "is_error": False,
                 "content": json.dumps({"ok": True, "summary": "Row 1 failed because " + "x" * 3000})}],
    )
    at.append_architect_turn(
        iteration, role="assistant",
        blocks=[{"kind": "tool_use", "name": "ce_close_iteration", "id": "c4",
                 "input": {"memory_updated": True}}],
    )
    at.append_architect_turn(
        iteration, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "c4", "is_error": False,
                 "content": json.dumps({"ok": True, "branch": "codeevolver-x", "note": "Iteration closed."})}],
    )
    at.append_architect_turn(
        iteration, role="result", blocks=[],
        usage={"input_tokens": 1234, "output_tokens": 56},
        meta={"harness": "opencode", "captured": True, "model": "m1", "num_messages": 7},
    )
    return at.architect_trace_path(iteration)


def test_render_writes_digest_transcript_and_header(trace_dir):
    d = _seed_basic()
    out = md.render_trace_dir(d, header={"iteration": 31, "branch": "codeevolver-x"})
    assert out is not None
    digest_path, transcript_path = out
    assert os.path.isfile(digest_path) and os.path.isfile(transcript_path)
    names = sorted(os.listdir(d))
    assert "digest.md" in names and "transcript.md" in names and "header.json" in names
    assert not [n for n in names if n.endswith(".tmp")]
    hdr = json.loads(_read(d, "header.json"))
    assert hdr["iteration"] == 31 and hdr["branch"] == "codeevolver-x"
    assert hdr["trace_stats"]["turns"] == 8
    assert hdr["trace_stats"]["tool_calls"] == {
        "ce_run_eval": 1, "bash": 1, "ce_inspect_traces": 1, "ce_close_iteration": 1,
    }


def test_digest_omits_thinking_transcript_includes_it(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d)
    digest, transcript = _read(d, "digest.md"), _read(d, "transcript.md")
    assert "let me think" not in digest
    assert re.search(r"_\(thinking omitted: \d[\d,]* chars — see transcript\.md\)_", digest)
    assert "- thinking omitted here:" in digest
    assert "**thinking**" in transcript and "let me think let me think" in transcript
    assert "thinking omitted" not in transcript.split("### turn 0000")[1]


def test_tool_result_caps_differ_and_point_to_turn_file(trace_dir):
    at.append_architect_turn(
        40, role="assistant",
        blocks=[{"kind": "tool_use", "name": "read", "id": "r1", "input": {"filePath": "/w/a.py"}},
                {"kind": "tool_use", "name": "read", "id": "r2", "input": {"filePath": "/w/b.py"}}],
    )
    at.append_architect_turn(
        40, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "r1", "is_error": False, "content": "A" * 3000},
                {"kind": "tool_result", "tool_use_id": "r2", "is_error": False, "content": "B" * 500}],
    )
    d = at.architect_trace_path(40)
    md.render_trace_dir(d)
    digest, transcript = _read(d, "digest.md"), _read(d, "transcript.md")
    # digest: 1,000 of the 3,000 A's + pointer; the 500 B's untouched, no pointer
    assert "A" * 1000 + "…" in digest and "A" * 1001 not in digest
    assert digest.count("→ full: turn_0001.json") == 1
    assert "B" * 500 in digest
    # transcript: 2,000 of the A's (line-wrapped at 1,800 inside the fence,
    # so compare with newlines removed)
    flat = transcript.replace("\n", "")
    assert "A" * 2000 + "…" in flat and "A" * 2001 not in flat
    assert transcript.count("→ full: turn_0001.json") == 1


def test_run_eval_result_renders_structured_line_shared_with_index(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    line = (
        "run_eval train/parent score=0.8703 rows=50 zero/failed=2 "
        "eval_id=9e9cb0eb53f1 sha=36b3216f tree=729fae92"
    )
    assert f"- **result** {line}" in digest
    # same formatter behind the index summary
    with open(os.path.join(d, "index.jsonl")) as f:
        index = [json.loads(ln) for ln in f if ln.strip()]
    assert index[2]["summary"].startswith(line)
    # auto_committed + ok:false variants
    at.append_architect_turn(
        41, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "x", "is_error": False,
                 "content": _run_eval_json(auto_committed=True)},
                {"kind": "tool_result", "tool_use_id": "y", "is_error": False,
                 "content": json.dumps({"ok": False, "error": "budget exhausted", "eval_sample": "train"})}],
    )
    d41 = at.architect_trace_path(41)
    md.render_trace_dir(d41)
    digest41 = _read(d41, "digest.md")
    assert line + " auto_committed" in digest41
    assert "- **ERROR** result: budget exhausted" in digest41


def test_inspect_traces_result_first_800_chars(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d)
    digest, transcript = _read(d, "digest.md"), _read(d, "transcript.md")
    body = "Row 1 failed because " + "x" * 3000
    assert "- **result** inspect_traces" in digest
    assert body[:800] + "…" in digest and body[:801] not in digest
    assert body[:2000] + "…" in transcript.replace("\n", "")


def test_user_text_turns_marked_prompt_then_watchdog_resume(trace_dir):
    at.append_architect_turn(50, role="user", blocks=[{"kind": "text", "text": "\"/s\n\nImprove the program.\""}])
    at.append_architect_turn(50, role="assistant", blocks=[{"kind": "text", "text": "ok"}])
    at.append_architect_turn(
        50, role="user",
        blocks=[{"kind": "text", "text": "\"/s\n\nYou are about to end iteration 50 WITHOUT having made any code change\""}],
    )
    d = at.architect_trace_path(50)
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    assert "### turn 0000 · Prompt" in digest
    assert "### turn 0002 · Watchdog resume — no-change watchdog" in digest
    assert "Improve the program." in digest
    assert "WITHOUT having made any code change" in digest


def test_prompt_kind_meta_overrides_ordinal_heuristic(trace_dir):
    # Claude path: the first user text turn is explicitly a resume (session
    # capture started mid-way) — meta wins over "first == Prompt".
    at.append_session_prompt_turn(51, "The session is about to exit but you have not called the `close_iteration` tool", resume=True)
    at.append_session_prompt_turn(51, "Improve.", resume=False)
    d = at.architect_trace_path(51)
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    assert "### turn 0000 · Watchdog resume — close-missing watchdog" in digest
    assert "### turn 0001 · Prompt" in digest


def test_error_results_verbatim_and_flagged(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    assert "- **ERROR** result (bash)" in digest
    assert "fatal: not a git repository fatal: not a git repository" in digest
    assert "tool errors: 1" in digest.splitlines()[0:12].__str__()


def test_every_line_under_cap_with_and_without_whitespace(trace_dir):
    spaced = ("word " * 1500).strip()      # 7,499 chars, whitespace everywhere
    solid = "z" * 6000                     # no whitespace at all
    at.append_architect_turn(
        60, role="assistant",
        blocks=[{"kind": "thinking", "text": spaced}, {"kind": "text", "text": solid}],
    )
    at.append_architect_turn(
        60, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "q", "is_error": False, "content": solid}],
    )
    d = at.architect_trace_path(60)
    md.render_trace_dir(d)
    for name in ("digest.md", "transcript.md"):
        lines = _read(d, name).splitlines()
        assert max(len(ln) for ln in lines) < 1900, name
    transcript_lines = _read(d, "transcript.md").splitlines()
    spaced_lines = [ln for ln in transcript_lines if ln.startswith("word ")]
    assert spaced_lines and all(ln.endswith("word") or ln.endswith("word ") for ln in spaced_lines)


def test_backtick_payloads_do_not_break_fences(trace_dir):
    payload = "here is code:\n```python\nprint(1)\n```\nand a longer run ````x````"
    at.append_architect_turn(
        61, role="user",
        blocks=[{"kind": "tool_result", "tool_use_id": "q", "is_error": False, "content": payload}],
    )
    d = at.architect_trace_path(61)
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    fence_lines = [ln for ln in digest.splitlines() if re.fullmatch(r"`{5,}", ln)]
    assert len(fence_lines) == 2  # opener + closer, longer than any run inside
    assert "```python" in digest and "````x````" in digest


def test_elapsed_only_when_ts_differ(trace_dir, monkeypatch):
    # identical stamps → no elapsed anywhere
    stamps = iter(["2026-08-30T16:00:00+00:00"] * 3)
    monkeypatch.setattr(at, "_now_iso", lambda: next(stamps))
    at.append_architect_turn(70, role="assistant", blocks=[{"kind": "text", "text": "a"}])
    at.append_architect_turn(70, role="assistant", blocks=[{"kind": "text", "text": "b"}])
    d = at.architect_trace_path(70)
    md.render_trace_dir(d)
    digest = _read(d, "digest.md")
    assert "(+" not in digest and "- elapsed:" not in digest
    # distinct stamps → per-turn delta and header elapsed
    stamps2 = iter(["2026-08-30T16:00:00+00:00", "2026-08-30T16:00:12+00:00", "2026-08-30T17:42:10+00:00"])
    monkeypatch.setattr(at, "_now_iso", lambda: next(stamps2))
    for t in ("a", "b", "c"):
        at.append_architect_turn(71, role="assistant", blocks=[{"kind": "text", "text": t}])
    d71 = at.architect_trace_path(71)
    md.render_trace_dir(d71)
    digest71 = _read(d71, "digest.md")
    assert "### turn 0001 · assistant (+12.0s)" in digest71
    assert "### turn 0002 · assistant (+1h 41m 58s)" in digest71
    assert "- elapsed: 1h 42m 10s" in digest71


def test_header_fields_and_trace_stats(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d, header={
        "iteration": 31, "branch": "codeevolver-x", "close_sha": "36b3216fdaee",
        "changes_landed": True, "primary_branch_after": "codeevolver-x",
        "baseline_score": 0.419, "best_train_score": 0.8703, "comparison_best_score": 0.869,
        "improvement_mode": "exploit", "eval_rows_spent": 350,
        "evals_by_sample": {"train": 5, "comparison": 2}, "watchdog_fired": True,
    })
    head = "\n".join(_read(d, "digest.md").splitlines()[:12])
    assert "# Iteration 31 architect session — digest" in head
    assert "branch: codeevolver-x · close_sha: 36b3216f · changes_landed: yes · primary after: codeevolver-x" in head
    assert "train subsample: baseline 0.4190 → best 0.8703 · comparison best: 0.8690 · mode: exploit" in head
    assert "eval rows spent: 350 (comparison×2, train×5) · watchdog resumes: yes" in head
    assert "turns: 8 (prompt 1, assistant 3, tool-result 3, result 1)" in head
    assert "tool calls: ce_run_eval 1, bash 1, ce_inspect_traces 1, ce_close_iteration 1 · tool errors: 1" in head
    assert "harness: opencode · model: m1 · tokens: input_tokens=1,234, output_tokens=56" in head
    # header=None → n/a everywhere it is loop-sourced, stats still computed
    md.render_trace_dir(d, header=None)
    head2 = "\n".join(_read(d, "transcript.md").splitlines()[:12])
    assert "# Iteration 31 architect session — transcript" in head2
    assert "branch: n/a · close_sha: n/a" in head2 and "turns: 8" in head2


def test_stub_trace_renders_not_captured(trace_dir):
    at.append_opencode_stub(8, stdout_tail="last lines of stdout")
    d = at.architect_trace_path(8)
    md.render_trace_dir(d, header={"iteration": 8})
    digest, transcript = _read(d, "digest.md"), _read(d, "transcript.md")
    assert "**NOT CAPTURED**: OpenCode session export unavailable" in digest
    assert "last lines of stdout" in digest
    assert len(digest) < 3000 and len(transcript) < 3000
    assert json.loads(_read(d, "header.json"))["trace_stats"]["captured"] is False


def test_missing_dir_is_noop(trace_dir):
    d = at.architect_trace_path(99)
    assert md.render_trace_dir(d) is None
    assert not os.path.exists(d)
    assert md.render_architect_trace_markdown(99) is None
    assert not os.path.exists(os.path.join(str(trace_dir), "README.md"))


def test_render_is_non_fatal(trace_dir, monkeypatch, caplog):
    d = _seed_basic()

    def boom(*_a, **_k):
        raise OSError("disk full")

    monkeypatch.setattr(md.os, "replace", boom)
    with caplog.at_level("WARNING"):
        assert md.render_trace_dir(d) is None
    assert "non-fatal" in caplog.text


def test_build_trace_header_base_and_v3_contexts():
    from src.engine.schemas.iteration_context import BaseIterationContext

    base = BaseIterationContext(iteration_idx=3)
    base.record_eval("s1", 0.4, eval_sample="train", n_rows=20)
    base.record_eval("s2", 0.5, eval_sample="train", n_rows=20)
    base.record_eval("s2", 0.45, eval_sample="comparison", n_rows=30)
    base.initial_subsample_score = 0.4
    h = md.build_trace_header(base)
    assert h["iteration"] == 3 and h["eval_rows_spent"] == 70
    assert h["evals_by_sample"] == {"train": 2, "comparison": 1}
    assert h["baseline_score"] == 0.4
    assert h["best_train_score"] is None and h["close_sha"] is None and h["watchdog_fired"] is None
    # overrides win
    assert md.build_trace_header(base, best_train_score=0.9, harness="claude")["best_train_score"] == 0.9

    from src.engine.optimizer.v3.iteration_context import IterationContext

    ctx = IterationContext(iteration_idx=4)
    ctx.child_branch = "codeevolver-y"
    ctx.close_sha = "abc123"
    ctx.best_subsample_score = 0.7
    ctx.comparison_best_score = 0.65
    ctx.watchdog_fired = True
    ctx.changes_landed = True
    h2 = md.build_trace_header(ctx)
    assert h2["branch"] == "codeevolver-y" and h2["close_sha"] == "abc123"
    assert h2["best_train_score"] == 0.7 and h2["comparison_best_score"] == 0.65
    assert h2["watchdog_fired"] is True and h2["changes_landed"] is True


def test_run_readme_one_row_per_iteration_sorted(trace_dir):
    _seed_basic(iteration=10)
    _seed_basic(iteration=2)
    at.append_opencode_stub(7)
    md.render_architect_trace_markdown(10, {"iteration": 10, "branch": "b10", "baseline_score": 0.4,
                                            "best_train_score": 0.5, "changes_landed": True,
                                            "eval_rows_spent": 100}, update_readme=False)
    md.render_architect_trace_markdown(2, {"iteration": 2, "branch": "b2"}, update_readme=False)
    md.render_architect_trace_markdown(7, {"iteration": 7})
    readme = _read(str(trace_dir), "README.md")
    rows = [ln for ln in readme.splitlines() if ln.startswith("| ") and not ln.startswith("| iter")]
    assert [r.split("|")[1].strip() for r in rows] == ["2", "7", "10"]
    assert "[digest](iteration_10_architect_trace/digest.md)" in rows[2]
    assert "| b10 | 0.4000 → 0.5000 | n/a | yes |" in rows[2] and "| 100 |" in rows[2]
    assert "not captured" in rows[1]
    # every link resolves
    for m in re.finditer(r"\]\(([^)]+)\)", readme):
        assert os.path.isfile(os.path.join(str(trace_dir), m.group(1))), m.group(1)


def test_render_is_idempotent_and_overwrites(trace_dir):
    d = _seed_basic()
    md.render_trace_dir(d, header={"iteration": 31, "branch": "first"})
    md.render_trace_dir(d, header={"iteration": 31, "branch": "second"})
    assert "branch: second" in _read(d, "digest.md")
    assert "branch: first" not in _read(d, "digest.md")
    assert json.loads(_read(d, "header.json"))["branch"] == "second"
    assert not [n for n in os.listdir(d) if n.endswith(".tmp")]


def test_tool_use_rendering_edit_bash_and_transcript_input(trace_dir):
    at.append_architect_turn(
        80, role="assistant",
        blocks=[
            {"kind": "tool_use", "name": "edit", "id": "e1",
             "input": {"filePath": "/w/prog.py", "oldString": "a = 1", "newString": "a = 2  # " + "y" * 400}},
            {"kind": "tool_use", "name": "bash", "id": "b1",
             "input": {"command": "python3 - <<'EOF'\nprint('multi')\nEOF", "workdir": "/w"}},
            {"kind": "tool_use", "name": "ce_coding_agent", "id": "a1",
             "input": {"change_request": "Do X.\nThen Y.", "mode": "fast"}},
        ],
    )
    d = at.architect_trace_path(80)
    md.render_trace_dir(d)
    digest, transcript = _read(d, "digest.md"), _read(d, "transcript.md")
    assert "- **call** `edit(filePath=/w/prog.py, new: a = 2 # " in digest
    assert "y" * 200 not in digest  # head of new content only
    assert "- **call** `bash(command=⤵, workdir=/w)`" in digest
    assert "print('multi')" in digest
    assert "- **call** `ce_coding_agent(change_request=⤵, mode=fast)`" in digest
    assert "Then Y." in digest
    assert "  input:" not in digest
    assert "  input:" in transcript and '"oldString": "a = 1"' in transcript
