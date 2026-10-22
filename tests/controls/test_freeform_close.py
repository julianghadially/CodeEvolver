"""Copyright © 2026 440 Labs LLC

Controls for the freeform (Baseline 1) HEAD-registering close protocol,
against a real git repo:

- the one-field `close_iteration` tool auto-commits a dirty tree, keeps a
  `codeevolver-` branch as-is, renames an unprefixed branch (suffixing on
  collision), names a detached HEAD, and still refuses an unmeasured change;
- the loop's post-session `close_effects.head_close` re-registers wherever HEAD ended up
  and doubles as the fallback close when the tool was never called;
- the close record / strategy hook make the primary follow HEAD, so the
  run's final candidate is the last close's branch;
- the real `IterationArchitect` binds the one-field tool for
  `close_iteration_mode="head"`.
"""

from __future__ import annotations

import inspect
import subprocess

import pytest

from src.engine.optimizer.v3 import close_effects as loop_module
from src.engine.optimizer.v3 import run_reporting
from src.engine.optimizer.v3.agent_tools import close_iteration_head_tool
from src.engine.optimizer.v3.ce_state import CEState
from src.engine.optimizer.v3.closing import build_close_record
from src.engine.optimizer.v3.errors import CloseIterationError
from src.engine.optimizer.v3.freeform.freeform_optimizer import (
    freeform_on_iteration_closed,
)
from src.engine.optimizer.v3.head_close import register_head
from src.engine.optimizer.v3.iteration_architect import ArchitectConfig
from src.engine.optimizer.v3.iteration_context import IterationContext
from tests.controls._arch_helpers import (
    RecordingHarness,
    make_architect,
    make_evaluator_config,
)

MAIN = "codeevolver-20260601120000-main"


def _repo(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args], cwd=ws, check=True, capture_output=True, text=True
        ).stdout.strip()

    git("init", "-q", "-b", "main")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (ws / "prog.py").write_text("x = 1\n")
    git("add", "-A")
    git("commit", "-qm", "seed")
    git("checkout", "-qb", MAIN)
    return ws, git, git("rev-parse", "HEAD")


def _ctx(start_sha: str, *, evaluated: bool = True) -> IterationContext:
    ctx = IterationContext(
        iteration_idx=2,
        parent_locked=True,
        start_branch=MAIN,
        start_sha=start_sha,
        primary_branch_at_start=MAIN,
    )
    if evaluated:
        ctx.record_eval(start_sha, 0.5, eval_sample="train", branch=MAIN, n_rows=5)
    return ctx


def _edit_and_commit(ws, git, text="x = 2\n"):
    (ws / "prog.py").write_text(text)
    git("commit", "-qam", "change")
    return git("rev-parse", "HEAD")


# --- the tool ----------------------------------------------------------------


def test_dirty_tree_is_auto_committed_and_branch_registered(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    (ws / "prog.py").write_text("x = 2\n")

    out = close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=True)

    assert out["ok"] is True and out["branch"] == MAIN
    assert git("status", "--porcelain") == ""
    assert git("log", "-1", "--format=%s") == "iter 2: uncommitted work at close (auto-committed)"
    assert ctx.child_branch == MAIN and ctx.close_called is True
    assert ctx.registered_branch == {
        "branch": MAIN, "was_primary": True, "is_primary": True,
        "promotion_evidence": None,
    }
    assert ctx.claimed_improvement_mode is None
    assert ctx.memory_updated_claimed is True and ctx.closed_by_fallback is False


def test_unprefixed_branch_is_renamed(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    git("checkout", "-qb", "feature/rerank")
    _edit_and_commit(ws, git)

    out = close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=False)

    assert out["branch"] == "codeevolver-feature/rerank"
    assert git("rev-parse", "--abbrev-ref", "HEAD") == "codeevolver-feature/rerank"
    branches = git("branch", "--list", "--format=%(refname:short)").splitlines()
    assert "feature/rerank" not in branches
    assert ctx.registered_branch["was_primary"] is False
    assert ctx.registered_branch["is_primary"] is True


def test_rename_collision_appends_sha(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    git("branch", "codeevolver-feat")  # already taken
    git("checkout", "-qb", "feat")
    sha = _edit_and_commit(ws, git)

    entry = register_head(ctx, str(ws))

    assert entry["branch"] == f"codeevolver-feat-{sha[:8]}"
    assert git("rev-parse", "--abbrev-ref", "HEAD") == entry["branch"]


def test_detached_head_gets_a_named_branch(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    git("checkout", "-q", "--detach")
    sha = _edit_and_commit(ws, git)

    out = close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=True)

    expected = f"codeevolver-20260601120000-main-detached-{sha[:8]}"
    assert out["branch"] == expected
    assert git("rev-parse", "--abbrev-ref", "HEAD") == expected
    assert git("rev-parse", "HEAD") == sha
    # Re-registering the same state is idempotent.
    assert register_head(ctx, str(ws))["branch"] == expected


def test_prefixed_branch_is_left_alone(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    git("checkout", "-qb", "codeevolver-lm-rerank")
    _edit_and_commit(ws, git)
    assert register_head(ctx, str(ws))["branch"] == "codeevolver-lm-rerank"
    assert git("rev-parse", "--abbrev-ref", "HEAD") == "codeevolver-lm-rerank"


def test_unmeasured_change_is_still_rejected(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed, evaluated=False)
    _edit_and_commit(ws, git)
    with pytest.raises(CloseIterationError, match="never measured"):
        close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=True)
    assert ctx.close_called is False


def test_no_op_iteration_closes_without_an_eval(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed, evaluated=False)
    out = close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=True)
    assert out["ok"] is True and ctx.close_called is True


# --- post-session re-registration / fallback ---------------------------------


def test_head_close_reregisters_where_head_ended_up(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    close_iteration_head_tool(ctx=ctx, workspace=str(ws), memory_updated=True)
    assert ctx.child_branch == MAIN
    # The agent kept going after calling the tool.
    git("checkout", "-qb", "codeevolver-after-close")
    _edit_and_commit(ws, git, "x = 3\n")

    loop_module.head_close(ctx, workspace=str(ws), state=None)

    assert ctx.child_branch == "codeevolver-after-close"
    assert ctx.registered_branch["branch"] == "codeevolver-after-close"
    assert ctx.closed_by_fallback is False and ctx.close_called is True


def test_head_close_is_the_fallback_when_tool_never_called(tmp_path, monkeypatch):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    ctx.memory_md_hash_at_start = "h1"
    monkeypatch.setattr(loop_module, "hash_file", lambda p: "h2")
    (ws / "prog.py").write_text("x = 9\n")  # dirty at session end

    loop_module.head_close(ctx, workspace=str(ws), state=None)

    assert ctx.close_called is True and ctx.closed_by_fallback is True
    assert ctx.memory_updated_claimed is True
    assert ctx.child_branch == MAIN and git("status", "--porcelain") == ""


def test_head_close_degrades_to_generic_fallback_on_git_failure(monkeypatch):
    ctx = _ctx("seed")
    calls = []
    monkeypatch.setattr(loop_module, "register_head", lambda c, w: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(loop_module, "fallback_close", lambda c, **kw: calls.append(kw))
    loop_module.head_close(ctx, workspace="/nowhere", state="S")
    assert calls == [{"workspace": "/nowhere", "state": "S"}]


# --- primary follows HEAD ----------------------------------------------------


def test_close_record_and_hook_make_primary_follow_head(tmp_path):
    ws, git, seed = _repo(tmp_path)
    ctx = _ctx(seed)
    git("checkout", "-qb", "codeevolver-new-lineage")
    sha = _edit_and_commit(ws, git)
    register_head(ctx, str(ws))
    ctx.close_sha = sha

    state = CEState(seed_branch=MAIN, seed_sha=seed)
    rec = build_close_record(ctx, state)
    assert rec["branch"] == "codeevolver-new-lineage"
    assert rec["primary_branch_after"] == "codeevolver-new-lineage"
    assert rec["is_primary"] is True and rec["was_primary"] is False
    assert rec["claimed_improvement_mode"] is None
    assert rec["promotion_evidence"] is None

    # The constraint-violation override pins the primary back to the start;
    # the freeform hook re-applies "primary follows HEAD".
    rec["primary_branch_after"] = MAIN
    freeform_on_iteration_closed(ctx, state, rec, services=None)
    assert rec["primary_branch_after"] == "codeevolver-new-lineage"

    state.record_iteration(rec)
    assert state.primary_branch == "codeevolver-new-lineage"
    assert run_reporting.final_candidate(state) == {
        "git_branch": "codeevolver-new-lineage", "git_sha": sha, "is_primary": True,
    }


# --- architect binding -------------------------------------------------------


def test_architect_binds_one_field_close_tool_for_head_mode():
    arch, ctx, _ = make_architect()
    arch.config = ArchitectConfig(
        run_timestamp="20260601120000",
        additional_instructions=None,
        program_path="my_pkg.program:make_program",
        evaluator=make_evaluator_config(),
        trainset_ids=list(range(50)),
        comparison_ids=list(range(10)),
        subsample_size=5,
        close_iteration_mode="head",
    )
    registry = arch.build_tool_registry(agent_harness=RecordingHarness())
    params = list(inspect.signature(registry["close_iteration"]).parameters)
    assert params == ["memory_updated"]
