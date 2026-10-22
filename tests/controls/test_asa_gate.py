"""ASA run_eval gate + commit-before-comparison against a real git repo
(specs/archive-public/v3_asa.md §Mechanism, §Rule 1, §Rule 2, §Comparison Set Eval Integrity)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from src.engine.evaluator.sandbox_client import EvalResult
from src.engine.optimizer.v3.asa.asa_optimizer import asa_run_eval_gate
from src.engine.optimizer.v3.asa.prompt_sections import (
    REQ_ELIGIBLE_NONPRIMARY,
    REQ_NEW_LINEAGE,
)
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.errors import RunEvalError
from src.engine.optimizer.v3.iteration_context import IterationContext
from src.engine.optimizer.v3.run_eval_tool import AUTO_COMMIT_NOTE, RunEvalBinding
from src.engine.services.subprocess_service import git_get_sha, git_is_dirty

from tests.controls._arch_helpers import make_evaluator_config

MAIN = "codeevolver-t-main"
SEED_LINEAGE = "codeevolver-t-seed"
OTHER = "codeevolver-t-react"


def _git(ws: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *args], check=True, capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _git(ws, "init", "-q", "-b", MAIN)
    _git(ws, "config", "user.email", "t@t")
    _git(ws, "config", "user.name", "t")
    (ws / "prog.py").write_text("x = 1\n")
    _git(ws, "add", "-A")
    _git(ws, "commit", "-qm", "seed")
    seed_sha = git_get_sha(str(ws))
    state = CEState(seed_branch=MAIN, seed_sha=seed_sha,
                    comparison_history_path=str(tmp_path / "cmp.jsonl"))
    # a registered non-primary lineage with one commit on it
    _git(ws, "checkout", "-qb", OTHER)
    (ws / "prog.py").write_text("x = 2\n")
    _git(ws, "commit", "-qam", "react")
    other_sha = git_get_sha(str(ws))
    state.register_lineage(OTHER, sha=other_sha, iteration=1)
    state.iterations.append({"iteration": 1, "branch": OTHER, "close_sha": other_sha, "skipped": False})
    _git(ws, "checkout", "-q", MAIN)
    return ws, state


def _ctx(**kw) -> IterationContext:
    base = dict(iteration_idx=4, subsample_size=5, parent_locked=True, primary_branch_at_start=MAIN,
                start_branch=MAIN, asa_allowance_left=0)
    base.update(kw)
    return IterationContext(**base)


# --- gate compliance ----------------------------------------------------------

def test_no_requirement_means_no_gate(repo):
    ws, state = repo
    assert asa_run_eval_gate(_ctx(), state, str(ws), "train") is None


def test_rule1_refuses_known_branch_and_accepts_fresh_fork(repo):
    ws, state = repo
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE])
    with pytest.raises(RunEvalError, match="create a new branch"):
        asa_run_eval_gate(ctx, state, str(ws), "train")
    _git(ws, "checkout", "-qb", "codeevolver-t-fresh")          # fork first (same content)
    assert asa_run_eval_gate(ctx, state, str(ws), "train") is None
    _git(ws, "checkout", "-qb", "not-prefixed")                  # wrong prefix
    with pytest.raises(RunEvalError):
        asa_run_eval_gate(ctx, state, str(ws), "train")


def test_rule1_refuses_detached_head(repo):
    ws, state = repo
    _git(ws, "checkout", "-q", git_get_sha(str(ws)))
    with pytest.raises(RunEvalError):
        asa_run_eval_gate(_ctx(asa_requirements=[REQ_NEW_LINEAGE]), state, str(ws), "train")


def test_rule2_accepts_eligible_branch_or_fork_from_it(repo):
    ws, state = repo
    ctx = _ctx(asa_requirements=[REQ_ELIGIBLE_NONPRIMARY], asa_eligible_at_start=[OTHER])
    with pytest.raises(RunEvalError, match="eligible non-primary"):   # on primary
        asa_run_eval_gate(ctx, state, str(ws), "train")
    _git(ws, "checkout", "-q", OTHER)
    assert asa_run_eval_gate(ctx, state, str(ws), "train") is None
    _git(ws, "checkout", "-qb", "codeevolver-t-react-v2")            # fork of eligible
    (ws / "prog.py").write_text("x = 3\n")                           # dirty tree is fine
    assert asa_run_eval_gate(ctx, state, str(ws), "train") is None


def test_both_rules_satisfied_by_fork_from_eligible(repo):
    ws, state = repo
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE, REQ_ELIGIBLE_NONPRIMARY], asa_eligible_at_start=[OTHER])
    _git(ws, "checkout", "-q", OTHER)
    with pytest.raises(RunEvalError):                                  # eligible but not new
        asa_run_eval_gate(ctx, state, str(ws), "train")
    _git(ws, "checkout", "-qb", "codeevolver-t-react-v2")
    assert asa_run_eval_gate(ctx, state, str(ws), "train") is None


def test_allowance_warns_once_and_holds_for_the_iteration(repo):
    ws, state = repo
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE], asa_allowance_left=1)
    note = asa_run_eval_gate(ctx, state, str(ws), "train")
    assert note and "WARNING" in note and ctx.asa_violation_granted
    # a later gated call in the same iteration still proceeds, even with the
    # allowance notionally spent
    ctx.asa_allowance_left = 0
    assert "WARNING" in asa_run_eval_gate(ctx, state, str(ws), "train")


# --- binding integration ------------------------------------------------------

def _eval_result(row_ids):
    return EvalResult(scores_by_metric={"accuracy": 0.5}, primary_score=0.5,
                      trace_meta=[{"row_id": r, "example_id": r, "score": 0.5} for r in row_ids],
                      trace_dir="/t.jsonl", trace_index_file="/ti.jsonl", row_ids=list(row_ids), raw={})


class _Fake:
    def __init__(self):
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kw):
        self.calls.append(kw)
        return _eval_result(kw.get("row_ids") or [])


def _binding(ws, state, ctx, **kw):
    return RunEvalBinding(
        ctx=ctx, workspace=str(ws), config=make_evaluator_config(), trainset_ids=list(range(20)),
        comparison_ids=[1, 2, 3], subsample_size=5, state=state, **kw,
    )


def test_binding_refuses_before_spending_and_only_gates_the_baseline(repo, monkeypatch):
    ws, state = repo
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(ws.parent / "traces"))
    fake = _Fake()
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE])
    b = _binding(ws, state, ctx, run_eval_gate=asa_run_eval_gate)
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        with pytest.raises(RunEvalError):
            b("train")
        assert fake.calls == [] and state.total_num_evals == 0
        _git(ws, "checkout", "-qb", "codeevolver-t-fresh")
        assert b("train")["ok"]
        assert len(fake.calls) == 1
        # second train eval: back on the primary is NOT gated (baseline only)
        _git(ws, "checkout", "-q", MAIN)
        assert b("train")["ok"] and len(fake.calls) == 2


def test_single_smoke_eval_neither_triggers_nor_consumes_the_gate(repo, monkeypatch):
    """A 1-row `single` on the primary must not be refused by rule 1 and must
    not count as "the baseline ran" — the real train baseline is still gated."""
    ws, state = repo
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(ws.parent / "traces"))
    fake = _Fake()
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE])
    b = _binding(ws, state, ctx, run_eval_gate=asa_run_eval_gate)
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        assert b("single")["ok"]            # on primary: smoke check is ungated
        assert len(fake.calls) == 1
        with pytest.raises(RunEvalError):    # the baseline itself is still gated
            b("train")
        assert len(fake.calls) == 1
        _git(ws, "checkout", "-qb", "codeevolver-t-fresh2")
        assert b("train")["ok"] and len(fake.calls) == 2


def test_binding_comparison_evals_are_never_gated_and_warning_note_surfaces(repo, monkeypatch):
    ws, state = repo
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(ws.parent / "traces"))
    fake = _Fake()
    ctx = _ctx(asa_requirements=[REQ_NEW_LINEAGE], asa_allowance_left=1)
    b = _binding(ws, state, ctx, run_eval_gate=asa_run_eval_gate)
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        assert b("comparison")["ok"]           # on primary, no refusal
        resp = b("train")
        assert resp["ok"] and "WARNING" in resp["gate_note"]


def test_commit_before_comparison_commits_named_branch_only(repo, monkeypatch):
    ws, state = repo
    monkeypatch.setenv("CODEEVOLVER_TRACES_DIR", str(ws.parent / "traces"))
    fake = _Fake()
    (ws / "prog.py").write_text("x = 9\n")
    assert git_is_dirty(str(ws))
    b = _binding(ws, state, _ctx(), commit_before_comparison=True)
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        resp = b("comparison")
    assert not git_is_dirty(str(ws))
    assert resp["sha"] == git_get_sha(str(ws))
    # The architect is told — otherwise it finds a clean tree and
    # "nothing to commit" with no explanation.
    assert resp["auto_committed"] is True
    assert resp["auto_commit_note"] == AUTO_COMMIT_NOTE
    line = Path(state.comparison_history_path).read_text().strip().splitlines()[-1]
    assert '"tree"' in line and resp["sha"] in line
    # clean tree on the same binding: nothing to commit, no note
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        resp2 = b("comparison")
    assert "auto_committed" not in resp2 and "auto_commit_note" not in resp2
    # detached HEAD: left dirty
    _git(ws, "checkout", "-q", git_get_sha(str(ws)))
    (ws / "prog.py").write_text("x = 10\n")
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        resp3 = _binding(ws, state, _ctx(), commit_before_comparison=True)("comparison")
    assert git_is_dirty(str(ws))
    assert "auto_committed" not in resp3
    # EE default: never commits
    _git(ws, "checkout", "-q", "-f", MAIN)
    (ws / "prog.py").write_text("x = 11\n")
    with patch("src.engine.optimizer.v3.run_eval_tool._run_eval", fake):
        resp4 = _binding(ws, state, _ctx())("comparison")
    assert git_is_dirty(str(ws))
    assert "auto_committed" not in resp4
