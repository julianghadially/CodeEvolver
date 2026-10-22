"""Unit tests for the post-change constraint gate.

`evaluate_change_constraints` computes the real `git diff parent..final` inside
a workspace and judges it with a (here mocked) LM callable. These tests use a
throwaway git repo so the diff path is exercised end-to-end without any LLM call.
"""

import json
import subprocess

import pytest

from src.engine.optimizer.v3.constraint_gate import evaluate_change_constraints


def _run(cwd, *args):
    subprocess.run(args, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def git_repo(tmp_path):
    """A repo with two commits; yields (workspace, parent_sha, final_sha)."""
    ws = tmp_path
    _run(ws, "git", "init", "-q")
    _run(ws, "git", "config", "user.email", "t@example.com")
    _run(ws, "git", "config", "user.name", "Test")
    (ws / "program.py").write_text("def forward(q):\n    return predict(q)\n")
    _run(ws, "git", "add", "-A")
    _run(ws, "git", "commit", "-q", "-m", "parent")
    parent_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()
    (ws / "program.py").write_text(
        "def forward(q):\n    docs = web_search(q)\n    return predict(q, docs)\n"
    )
    _run(ws, "git", "add", "-A")
    _run(ws, "git", "commit", "-q", "-m", "final")
    final_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()
    return str(ws), parent_sha, final_sha


def _fail_lm(prompt):
    # Stash the prompt so the test can assert the diff was included.
    _fail_lm.last_prompt = prompt
    return json.dumps({
        "status": "fail",
        "reason": "Adds a web_search call.",
        "violated_constraint": "Do not add network calls.",
    })


def test_gate_disabled_without_instructions(git_repo):
    ws, parent, final = git_repo
    assert evaluate_change_constraints(
        workspace=ws, parent_sha=parent, final_sha=final,
        additional_instructions=None, lm_callable=_fail_lm,
    ) is None
    assert evaluate_change_constraints(
        workspace=ws, parent_sha=parent, final_sha=final,
        additional_instructions="   ", lm_callable=_fail_lm,
    ) is None


def test_gate_disabled_without_lm_callable(git_repo):
    ws, parent, final = git_repo
    assert evaluate_change_constraints(
        workspace=ws, parent_sha=parent, final_sha=final,
        additional_instructions="Do not add network calls.", lm_callable=None,
    ) is None


def test_gate_noop_when_parent_equals_final(git_repo):
    ws, parent, _ = git_repo
    assert evaluate_change_constraints(
        workspace=ws, parent_sha=parent, final_sha=parent,
        additional_instructions="Do not add network calls.", lm_callable=_fail_lm,
    ) is None


def _pass_lm(prompt):
    return json.dumps({
        "status": "pass",
        "reason": "Only adds retrieval context.",
        "violated_constraint": None,
    })


def test_gate_returns_fail_on_violating_diff(git_repo, caplog):
    ws, parent, final = git_repo
    with caplog.at_level("INFO"):
        result = evaluate_change_constraints(
            workspace=ws, parent_sha=parent, final_sha=final,
            additional_instructions="Do not add network calls.",
            lm_callable=_fail_lm,
            change_requests=["Add web search"],
        )
    assert result is not None
    assert result.status == "fail"
    assert result.violated_constraint == "Do not add network calls."
    # The real diff body was handed to the checker.
    assert "web_search" in _fail_lm.last_prompt
    # The verdict is logged (WARNING on fail) with the judged sha range.
    fail_lines = [r for r in caplog.records if "constraint gate: FAIL" in r.getMessage()]
    assert len(fail_lines) == 1 and fail_lines[0].levelname == "WARNING"
    msg = fail_lines[0].getMessage()
    assert f"{parent[:8]}..{final[:8]}" in msg and "Do not add network calls." in msg


def test_gate_logs_a_pass_verdict(git_repo, caplog):
    """A passing gate leaves a verdict line in the logs."""
    ws, parent, final = git_repo
    with caplog.at_level("INFO"):
        result = evaluate_change_constraints(
            workspace=ws, parent_sha=parent, final_sha=final,
            additional_instructions="Do not add network calls.",
            lm_callable=_pass_lm,
        )
    assert result is not None and result.status == "pass"
    pass_lines = [r for r in caplog.records if "constraint gate: pass" in r.getMessage()]
    assert len(pass_lines) == 1 and pass_lines[0].levelname == "INFO"
    assert f"{parent[:8]}..{final[:8]}" in pass_lines[0].getMessage()


# --- Revert visibility: message + marker commit (all arms) -------------------


def test_revert_message_quotes_constraint_but_never_the_reason():
    from src.engine.optimizer.v3.constraint_gate import build_constraint_revert_message

    msg = build_constraint_revert_message(3, "Do not add network calls.")
    assert msg.startswith("Revert iteration 3:")
    assert "constraint checker" in msg
    assert "Violated constraint: Do not add network calls." in msg
    # The judge's rationale is never part of the message.
    assert "web_search" not in msg and "reason" not in msg.lower()
    assert "(not quoted by the checker)" in build_constraint_revert_message(4, None)


def test_revert_marker_is_an_empty_commit_carrying_the_message(git_repo):
    from src.engine.optimizer.v3.constraint_gate import (
        build_constraint_revert_message,
        commit_constraint_revert_marker,
    )

    ws, parent, final = git_repo
    # The loop's reset happens first; the marker sits on top of the reset point.
    _run(ws, "git", "reset", "-q", "--hard", parent)
    marker = commit_constraint_revert_marker(
        ws, build_constraint_revert_message(2, "Do not add network calls.")
    )
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ws, capture_output=True, text=True).stdout.strip()
    assert marker == head and head not in (parent, final)
    tree = lambda ref: subprocess.run(
        ["git", "rev-parse", f"{ref}^{{tree}}"], cwd=ws, capture_output=True, text=True
    ).stdout.strip()
    assert tree("HEAD") == tree(parent), "the marker changes no content"
    assert subprocess.run(["git", "rev-parse", "HEAD~1"], cwd=ws, capture_output=True, text=True).stdout.strip() == parent
    body = subprocess.run(["git", "log", "-1", "--format=%B"], cwd=ws, capture_output=True, text=True).stdout
    assert "Revert iteration 2" in body and "Violated constraint: Do not add network calls." in body


def test_revert_marker_failure_is_non_fatal(tmp_path):
    from src.engine.optimizer.v3.constraint_gate import commit_constraint_revert_marker

    # Not a git repo: the commit fails, the helper returns None, nothing raises.
    assert commit_constraint_revert_marker(str(tmp_path), "Revert iteration 1") is None
