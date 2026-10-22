"""Copyright © 2026 440 Labs LLC

Unit tests for the start-of-run clean-slate inspector (offline).

Covers the prompt builder (the three inspection tasks, deny-path listing,
no-contents guardrail), verdict parsing, and the driver's abort semantics:
an explicit `contaminated` verdict raises `CleanSlateContaminationError`;
everything else (clean verdict, unparseable output, agent error) fails OPEN.
"""

from __future__ import annotations

import os
import subprocess

import pytest

from src.engine.optimizer.v3 import clean_slate as cs


def _prompt(**overrides):
    kwargs = dict(
        workspace="/workspace/repo",
        initial_branch="main",
        deny_paths=["data/bench_test.json", "_holdout"],
        testset_rel_path="data/bench_test.json",
    )
    kwargs.update(overrides)
    return cs.build_clean_slate_prompt(**kwargs)


# ---------------------------------------------------------------- prompt


def test_prompt_frames_scientist_and_three_tasks():
    prompt = _prompt()
    assert "scientist" in prompt.lower()
    assert "research project" in prompt
    assert "self-improving AI optimization" in prompt
    # The three inspection tasks.
    assert "Git history" in prompt
    assert "deny paths" in prompt.lower()
    assert "Container workspace" in prompt
    assert "Docker container" in prompt
    # Prior-agent leftovers language from the requirement.
    assert "memory from old runs" in prompt or "memory from old" in prompt


def test_prompt_lists_deny_paths_and_forbids_reading_contents():
    prompt = _prompt()
    assert "data/bench_test.json" in prompt
    assert "_holdout" in prompt
    # The contents-read guardrail (belt; the armed deny paths are suspenders).
    # Load-bearing now that the agent has a shell: its report lands beside
    # memory.md, so a quoted holdout row would be architect-readable.
    assert "never the contents of a data file" in prompt
    assert "NAMES, PATHS and ROW/BYTE COUNTS ONLY" in prompt


def test_prompt_states_the_shell_is_read_only_and_enforced():
    prompt = _prompt()
    # The agent must know the guard exists, or it will "helpfully" check out a
    # branch to look at it and trip CleanSlateMutationError.
    assert "READ-ONLY" in prompt
    assert "fingerprinted before and after" in prompt
    for forbidden in ("git checkout", "git reset", "git stash", "git gc"):
        assert forbidden in prompt
    # ...and what the shell is FOR: the channels that survive branch deletion.
    for allowed in ("git reflog", "git fsck", "git branch -a"):
        assert allowed in prompt


def test_prompt_gives_every_check_a_body():
    """Check 2 shipped as a bare heading with no instructions under it.

    The original assertions only grepped for substrings, so an empty task
    body passed. Anchor on the content each check must actually carry.
    """
    prompt = _prompt()
    body = prompt.split("## Inspect three things", 1)[1]
    one, rest = body.split("2. **Repo data folders.**", 1)
    two, three = rest.split("3. **Container workspace.**", 1)
    assert "git log" in one and "reflog" in one
    assert "deny" in two.lower() and len(two.split()) > 40
    assert "Docker container" in three


def test_prompt_accepts_sibling_benchmarks_and_this_runs_own_job_dir():
    """Two false-positive sources that would abort a healthy run.

    The merged LangProBe repo hosts hover AND ragqa, so each run's workspace
    legitimately carries the other's test set; and this run's own job
    directory exists by the time the agent looks.
    """
    prompt = _prompt()
    assert "another benchmark's train/test data is normal" in prompt.lower()
    assert "only OTHER jobs' directories" in prompt


def test_prompt_handles_empty_deny_paths_and_missing_testset():
    prompt = _prompt(deny_paths=[], testset_rel_path=None)
    assert "(none configured)" in prompt
    assert "No test-set path was configured" in prompt


# ---------------------------------------------------------------- parsing


def test_parse_verdict_clean_and_contaminated():
    clean = '```json\n{"status": "clean", "findings": []}\n```'
    assert cs.parse_clean_slate_verdict(clean) == {"status": "clean", "findings": []}

    dirty = (
        '```json\n{"status": "contaminated", "findings": '
        '[{"path": "/old/memory.md", "reason": "prior-run memory"}]}\n```'
    )
    v = cs.parse_clean_slate_verdict(dirty)
    assert v["status"] == "contaminated"
    assert v["findings"][0]["path"] == "/old/memory.md"


def test_prompt_explains_uncertain_status():
    prompt = _prompt()
    assert '"uncertain"' in prompt
    assert "cannot confidently classify" in prompt


def test_parse_verdict_keeps_uncertain():
    v = cs.parse_clean_slate_verdict(
        '```json\n{"status": "uncertain", "findings": '
        '[{"path": "notes.md", "reason": "cannot tell if prior-run notes"}]}\n```'
    )
    assert v["status"] == "uncertain"
    assert v["findings"][0]["path"] == "notes.md"


def test_parse_verdict_tolerates_garbage():
    assert cs.parse_clean_slate_verdict(None) is None
    assert cs.parse_clean_slate_verdict("prose") is None
    v = cs.parse_clean_slate_verdict('```json\n{"status": "sparkling"}\n```')
    assert v["status"] == "unparsed"
    # Non-list findings are dropped, non-dict entries filtered.
    v = cs.parse_clean_slate_verdict(
        '```json\n{"status": "clean", "findings": "nope"}\n```'
    )
    assert v["findings"] == []


# ---------------------------------------------------------------- driver


def _run(monkeypatch, tmp_path, agent_result):
    # The report lives in logs_dir(), NOT memory_dir() — memory_dir is the
    # directory the architect reads.
    monkeypatch.setenv("CODEEVOLVER_LOGS_DIR", str(tmp_path))
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    captured = {}

    def fake_run_inspection_agent(**kwargs):
        captured.update(kwargs)
        return agent_result

    import src.engine.services.subprocess_service as svc

    monkeypatch.setattr(svc, "run_inspection_agent", fake_run_inspection_agent)

    status = cs.run_clean_slate_inspection(
        workspace="/workspace",
        initial_branch="main",
        deny_paths=["data/bench_test.json"],
        testset_rel_path="data/bench_test.json",
        coding_harness="claude",
    )
    return status, captured


def test_clean_verdict_writes_report_and_returns_clean(tmp_path, monkeypatch):
    report = 'Inspected.\n```json\n{"status": "clean", "findings": []}\n```'
    status, captured = _run(
        monkeypatch, tmp_path, {"success": True, "proposed_change": report}
    )
    assert status == "clean"
    assert os.path.exists(os.path.join(str(tmp_path), "clean_slate_report.md"))
    # Unlike the integrity agent, the holdout fences ARE armed here.
    assert captured["deny_paths"] == ["data/bench_test.json"]
    assert captured["writable"] is False
    # The one inspection caller that gets a shell: `git log` / `branch -a` /
    # `reflog` / `fsck` have no Read/Grep/Glob equivalent, and checks 1 and 3
    # are unanswerable without them.
    assert captured["allow_bash"] is True


def test_contaminated_verdict_raises_with_findings(tmp_path, monkeypatch):
    report = (
        'Found it.\n```json\n{"status": "contaminated", "findings": '
        '[{"path": "~/.codeevolver/memory/memory.md", '
        '"reason": "memory from a prior run"}]}\n```'
    )
    with pytest.raises(cs.CleanSlateContaminationError) as exc_info:
        _run(monkeypatch, tmp_path, {"success": True, "proposed_change": report})
    err = exc_info.value
    assert "memory from a prior run" in str(err)
    assert err.findings[0]["path"] == "~/.codeevolver/memory/memory.md"
    assert err.report_path and err.report_path.endswith("clean_slate_report.md")
    # The report was still written before the abort.
    assert os.path.exists(os.path.join(str(tmp_path), "clean_slate_report.md"))
    assert err.to_dict()["findings"] == err.findings


def test_uncertain_verdict_fails_open_with_status(tmp_path, monkeypatch):
    report = (
        'Unsure about one file.\n```json\n{"status": "uncertain", "findings": '
        '[{"path": "scratch/ideas.md", "reason": "could be prior-run notes"}]}\n```'
    )
    status, _ = _run(
        monkeypatch, tmp_path, {"success": True, "proposed_change": report}
    )
    assert status == "uncertain"
    # The report is still written for the operator to review.
    assert os.path.exists(os.path.join(str(tmp_path), "clean_slate_report.md"))


def test_unparseable_output_fails_open(tmp_path, monkeypatch):
    status, _ = _run(
        monkeypatch, tmp_path,
        {"success": True, "proposed_change": "prose without a verdict"},
    )
    assert status == "unparsed"


def test_empty_output_fails_open(tmp_path, monkeypatch):
    status, _ = _run(
        monkeypatch, tmp_path, {"success": False, "proposed_change": ""}
    )
    assert status == "error"


def test_agent_raise_fails_open(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path))
    import src.engine.services.subprocess_service as svc

    def boom(**kwargs):
        raise RuntimeError("harness exploded")

    monkeypatch.setattr(svc, "run_inspection_agent", boom)
    status = cs.run_clean_slate_inspection(
        workspace="/workspace",
        initial_branch="main",
        deny_paths=[],
        testset_rel_path=None,
        coding_harness="claude",
    )
    assert status == "error"


# ------------------------------------------------- mutation guard (A1b)


def _repo(tmp_path):
    """A real one-commit git repo, mirroring the squashed seed workspace."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "program.py").write_text("x = 1\n")
    (ws / ".gitignore").write_text("ignored/\n")
    (ws / "ignored").mkdir()
    (ws / "ignored" / "cache.bin").write_text("junk\n")
    for cmd in (
        ["init", "-q", "-b", "main"],
        ["add", "-A"],
        ["-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "seed"],
    ):
        subprocess.run(["git", *cmd], cwd=ws, check=True, capture_output=True)
    return ws


def _run_against(monkeypatch, tmp_path, workspace, side_effect):
    """Drive the inspector with an agent that does `side_effect(workspace)`."""
    monkeypatch.setenv("CODEEVOLVER_LOGS_DIR", str(tmp_path / "mem"))
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "mem"))
    monkeypatch.delenv("CODEEVOLVER_JOB_ID", raising=False)
    (tmp_path / "mem").mkdir(exist_ok=True)

    import src.engine.services.subprocess_service as svc

    def fake_agent(**kwargs):
        side_effect(workspace)
        return {
            "success": True,
            "proposed_change": 'ok\n```json\n{"status": "clean", "findings": []}\n```',
        }

    monkeypatch.setattr(svc, "run_inspection_agent", fake_agent)
    return cs.run_clean_slate_inspection(
        workspace=str(workspace),
        initial_branch="main",
        deny_paths=[],
        testset_rel_path=None,
        coding_harness="claude",
    )


def test_read_only_session_passes_the_guard(tmp_path, monkeypatch):
    """The happy path must not be noisy — including git's own read commands.

    `git status`/`log` touch .git (index stat cache, gc --auto bookkeeping),
    and `_snapshot_repo` itself calls `git_tree_signature`, which writes loose
    objects through a throwaway index by design. None of that is a mutation of
    what the guard measures, and a guard that fired on it would be useless.
    """
    ws = _repo(tmp_path)

    def read_only(workspace):
        for args in (["status"], ["log", "--stat"], ["branch", "-a"], ["reflog"]):
            subprocess.run(["git", *args], cwd=workspace, capture_output=True)

    assert _run_against(monkeypatch, tmp_path, ws, read_only) == "clean"


@pytest.mark.parametrize(
    "name,mutate",
    [
        # A commit moves HEAD but can leave the tree hash equal to nothing else.
        ("commit", lambda ws: (
            (ws / "program.py").write_text("x = 2\n"),
            subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True),
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t",
                 "commit", "-q", "-m", "nope"],
                cwd=ws, capture_output=True,
            ),
        )),
        # A new branch touches neither HEAD content nor the working tree.
        ("branch", lambda ws: subprocess.run(
            ["git", "checkout", "-q", "-b", "scratch"], cwd=ws, capture_output=True,
        )),
        # An uncommitted edit moves the tree hash and nothing else.
        ("dirty_edit", lambda ws: (ws / "program.py").write_text("x = 99\n")),
        # A scratch file moves neither HEAD, refs, nor the committed tree.
        ("untracked_file", lambda ws: (ws / "notes.md").write_text("hi\n")),
        # A newly-ignored top-level path: `status --ignored` collapses ignored
        # DIRECTORIES to one entry, so a brand-new one still shows up.
        ("ignored_dir", lambda ws: (
            (ws / "mlruns").mkdir(),
            (ws / ".gitignore").write_text("ignored/\nmlruns/\n"),
        )),
    ],
)
def test_any_mutation_hard_fails(tmp_path, monkeypatch, name, mutate):
    ws = _repo(tmp_path)
    with pytest.raises(cs.CleanSlateMutationError) as exc_info:
        _run_against(monkeypatch, tmp_path, ws, mutate)
    err = exc_info.value
    assert err.changed, f"{name}: guard fired but named no changed key"
    assert "read-only" in str(err)
    assert err.to_dict()["changed"] == err.changed


def test_mutation_beats_fail_open(tmp_path, monkeypatch):
    """A crashed session that also mutated must raise, not return "error".

    Fail-open is right for "the agent could not tell us anything". It is wrong
    for "the agent broke the workspace on its way out" — that verdict is not
    unknown, it is bad.
    """
    ws = _repo(tmp_path)
    monkeypatch.setenv("CODEEVOLVER_LOGS_DIR", str(tmp_path / "mem"))
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "mem"))
    (tmp_path / "mem").mkdir(exist_ok=True)
    import src.engine.services.subprocess_service as svc

    def boom(**kwargs):
        subprocess.run(["git", "checkout", "-q", "-b", "scratch"],
                       cwd=ws, capture_output=True)
        raise RuntimeError("harness exploded")

    monkeypatch.setattr(svc, "run_inspection_agent", boom)
    with pytest.raises(cs.CleanSlateMutationError):
        cs.run_clean_slate_inspection(
            workspace=str(ws), initial_branch="main",
            deny_paths=[], testset_rel_path=None, coding_harness="claude",
        )


def test_report_is_written_to_logs_not_memory(tmp_path, monkeypatch):
    """The architect reads memory_dir. The inspector must not write into it.

    The inspector walks the repo with a shell, and unlike the post-loop
    run-integrity report this one exists before the architect's first turn.
    logs/ is collected by orchestrator/collect.py exactly like memory, so
    moving it costs no durability.
    """
    from src.engine import paths

    monkeypatch.setenv("SANDBOX_MODE", "local")
    monkeypatch.setenv("CODEEVOLVER_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("CODEEVOLVER_MEMORY_DIR", str(tmp_path / "memory"))
    monkeypatch.setenv("CODEEVOLVER_JOB_ID", "job_abc123")

    report = paths.clean_slate_report_path()
    assert report.endswith("job_abc123_clean_slate_report.md")
    assert str(tmp_path / "logs") in report
    assert str(tmp_path / "memory") not in report
    # Job id in the FILENAME, mirroring detailed_agent_log_path(): logs_dir()
    # is deliberately not job-suffixed.
    assert paths.logs_dir() == str(tmp_path / "logs")

    monkeypatch.delenv("CODEEVOLVER_JOB_ID")
    assert paths.clean_slate_report_path().endswith("clean_slate_report.md")


def test_report_is_fenced_from_every_agent():
    """The report sits beside memory.md, which the architect reads freely."""
    from src.engine.optimizer.v3.iteration_architect import (
        _build_holdout_deny_paths,
    )

    deny = _build_holdout_deny_paths("data/bench_test.json", None)
    assert cs.CLEAN_SLATE_REPORT_DENY_FRAGMENT in deny
    # Substring-matched, so the fragment must cover the real filename.
    assert cs.CLEAN_SLATE_REPORT_DENY_FRAGMENT in "clean_slate_report.md"
