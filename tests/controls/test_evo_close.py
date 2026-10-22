"""Copyright © 2026 440 Labs LLC

Controls for the pareto_evolution close protocol:

- the loop-branch `close_iteration` tool refuses a session that left the
  loop-assigned branch, accepts one that stayed on it, and never touches the
  primary (the loop's close hook decides that);
- the real `IterationArchitect` registers the one-field variant
  (`memory_updated` only — no improvement_mode, as the arm uses the
  freeform prompt) when `close_iteration_mode="loop_branch"`
  and refuses to bind it without a forked child branch;
- the loop's fallback close prefers the loop-owned child branch over
  wherever HEAD wandered.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.engine.optimizer.v3 import agent_tools as agent_tools_module
from src.engine.optimizer.v3 import close_effects as loop_module
from src.engine.optimizer.v3.agent_tools import close_iteration_loop_branch_tool
from src.engine.optimizer.v3.errors import CloseIterationError
from src.engine.optimizer.v3.iteration_context import IterationContext
from tests.controls._arch_helpers import RecordingHarness, make_architect


def _registry(arch):
    return arch.build_tool_registry(agent_harness=RecordingHarness())

_CHILD = "codeevolver-20260601120000-iter2"


def _ctx(child: str | None = _CHILD) -> IterationContext:
    ctx = IterationContext(
        iteration_idx=2,
        parent_locked=True,
        start_sha="parentsha",
        start_branch=child,
        child_branch=child,
        parent_branch="codeevolver-20260601120000-main",
        parent_sha="parentsha",
        parent_is_primary=True,
        primary_branch_at_start="codeevolver-20260601120000-main",
    )
    ctx.record_eval("childsha", 0.6, eval_sample="train", branch=child, n_rows=5)
    return ctx


def _patch_git(monkeypatch, *, current: str, dirty: bool = False) -> None:
    monkeypatch.setattr(agent_tools_module, "git_get_current_branch", lambda ws: current)
    monkeypatch.setattr(agent_tools_module, "git_branch_exists", lambda ws, b: True)
    monkeypatch.setattr(agent_tools_module, "git_is_dirty", lambda ws: dirty)
    monkeypatch.setattr(agent_tools_module, "git_get_sha", lambda ws, ref="HEAD": "childsha")
    monkeypatch.setattr(agent_tools_module, "git_commit_trees", lambda *a, **k: [])


def _call(ctx: IterationContext, **overrides: Any) -> dict:
    kwargs = dict(ctx=ctx, workspace="/workspace", memory_updated=True)
    kwargs.update(overrides)
    return close_iteration_loop_branch_tool(**kwargs)


def test_loop_branch_close_rejects_wrong_head(monkeypatch):
    ctx = _ctx()
    _patch_git(monkeypatch, current="codeevolver-20260601120000-main")
    with pytest.raises(CloseIterationError, match="Do not switch branches"):
        _call(ctx)
    assert ctx.close_called is False


def test_loop_branch_close_accepts_child_and_never_sets_primary(monkeypatch):
    ctx = _ctx()
    _patch_git(monkeypatch, current=_CHILD)
    out = _call(ctx)
    assert out["ok"] is True and out["branch"] == _CHILD
    assert ctx.close_called is True and ctx.closed_by_fallback is False
    assert ctx.child_branch == _CHILD
    assert ctx.claimed_improvement_mode is None  # never asked on this arm
    assert ctx.memory_updated_claimed is True
    entry = ctx.registered_branch
    assert entry["branch"] == _CHILD
    assert entry["is_primary"] is False
    assert entry["was_primary"] is True  # parent was the primary
    assert entry["promotion_evidence"] is None


def test_loop_branch_close_keeps_shared_checks(monkeypatch):
    ctx = _ctx()
    _patch_git(monkeypatch, current=_CHILD, dirty=True)
    with pytest.raises(CloseIterationError, match="uncommitted"):
        _call(ctx)
    _patch_git(monkeypatch, current=_CHILD)
    # The tool has no improvement_mode parameter at all.
    with pytest.raises(TypeError, match="improvement_mode"):
        _call(ctx, improvement_mode="exploitation")
    assert _call(ctx, memory_updated="false")["ok"] is True
    assert ctx.memory_updated_claimed is False


def test_loop_branch_close_requires_child_branch(monkeypatch):
    ctx = _ctx(child=None)
    _patch_git(monkeypatch, current=_CHILD)
    with pytest.raises(CloseIterationError, match="no loop-assigned branch"):
        _call(ctx)


def test_architect_registers_one_field_tool_in_loop_branch_mode():
    arch, ctx, _ = make_architect(iteration_idx=2)
    arch.config.close_iteration_mode = "loop_branch"
    ctx.child_branch = _CHILD
    registry = _registry(arch)
    import inspect

    params = list(inspect.signature(registry["close_iteration"]).parameters)
    assert params == ["memory_updated"]


@pytest.mark.parametrize("mode", ["register_branch", "loop_branch", "head"])
def test_close_tool_registry_signature_matches_spec_required_fields(mode):
    """Both harnesses build the LLM-facing schema from the spec and dispatch
    `**args` into the registry callable, so the two must agree or every close
    call raises TypeError. Nothing else checks this."""
    import inspect

    from src.engine.agent.harnesses.architect_tools import architect_tool_spec

    arch, ctx, _ = make_architect(iteration_idx=2)
    arch.config.close_iteration_mode = mode
    ctx.child_branch = _CHILD
    params = list(inspect.signature(_registry(arch)["close_iteration"]).parameters)
    spec = architect_tool_spec("close_iteration", close_iteration_mode=mode)
    assert params == spec.input_schema["required"]
    assert set(params) == set(spec.input_schema["properties"])


def test_architect_refuses_loop_branch_mode_without_forked_child():
    arch, ctx, _ = make_architect(iteration_idx=2)
    arch.config.close_iteration_mode = "loop_branch"
    ctx.child_branch = None
    with pytest.raises(RuntimeError, match="fork ctx.child_branch"):
        _registry(arch)


def test_architect_default_mode_keeps_three_field_tool():
    arch, ctx, _ = make_architect(iteration_idx=2)
    import inspect

    params = list(inspect.signature(_registry(arch)["close_iteration"]).parameters)
    assert params == ["register_branch", "improvement_mode", "memory_updated"]


def test_fallback_close_prefers_loop_owned_child_branch(monkeypatch):
    from src.engine.optimizer.v3.ce_state import CEState

    state = CEState(seed_branch="codeevolver-20260601120000-main", seed_sha="seed")
    ctx = _ctx()
    ctx.close_called = False
    # The architect wandered off to another codeevolver- branch and died.
    monkeypatch.setattr(
        loop_module, "git_get_current_branch", lambda ws: "codeevolver-20260601120000-stray"
    )
    monkeypatch.setattr(loop_module, "hash_file", lambda p: None)
    loop_module.fallback_close(ctx, workspace="/ws", state=state)
    assert ctx.child_branch == _CHILD
    assert ctx.registered_branch["branch"] == _CHILD
    assert ctx.closed_by_fallback is True


def test_fallback_close_without_child_uses_head_branch(monkeypatch):
    from src.engine.optimizer.v3.ce_state import CEState

    state = CEState(seed_branch="codeevolver-20260601120000-main", seed_sha="seed")
    ctx = _ctx(child=None)
    ctx.close_called = False
    monkeypatch.setattr(
        loop_module, "git_get_current_branch", lambda ws: "codeevolver-20260601120000-x"
    )
    monkeypatch.setattr(loop_module, "hash_file", lambda p: None)
    loop_module.fallback_close(ctx, workspace="/ws", state=state)
    assert ctx.child_branch == "codeevolver-20260601120000-x"


# --- fork_child_from against a real git repo ---------------------------------


def _init_repo(tmp_path):
    import subprocess

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
    seed_sha = git("rev-parse", "HEAD")
    git("checkout", "-qb", "codeevolver-20260601120000-main")
    (ws / "prog.py").write_text("x = 2\n")
    git("commit", "-qam", "moved tip")
    return ws, git, seed_sha


def test_fork_child_from_uses_candidate_sha_not_branch_tip(tmp_path):
    from src.engine.optimizer.v3.select_parent import fork_child_from

    ws, git, seed_sha = _init_repo(tmp_path)
    ctx = IterationContext(
        iteration_idx=4, primary_branch_at_start="codeevolver-20260601120000-main"
    )
    child = fork_child_from(
        ctx=ctx,
        parent_branch="codeevolver-20260601120000-main",
        parent_sha=seed_sha,
        parent_iteration=0,
        workspace=str(ws),
        run_timestamp="20260601120000",
    )
    assert child == "codeevolver-20260601120000-iter4"
    assert git("rev-parse", "--abbrev-ref", "HEAD") == child
    # Forked from the recorded candidate sha, not the (moved) branch tip.
    assert git("rev-parse", "HEAD") == seed_sha
    assert ctx.child_branch == child and ctx.parent_sha == seed_sha
    assert ctx.parent_iteration == 0 and ctx.parent_locked is True
    assert ctx.parent_is_primary is True
    assert ctx.parent_tree == git("rev-parse", f"{seed_sha}^{{tree}}")

    # A crashed attempt left the branch behind (with a commit): re-forking
    # resets it to the parent instead of failing.
    (ws / "prog.py").write_text("x = 3\n")
    git("commit", "-qam", "half-finished")
    ctx2 = IterationContext(iteration_idx=4)
    fork_child_from(
        ctx=ctx2,
        parent_branch="codeevolver-20260601120000-main",
        parent_sha=seed_sha,
        parent_iteration=0,
        workspace=str(ws),
        run_timestamp="20260601120000",
    )
    assert git("rev-parse", "HEAD") == seed_sha
    assert ctx2.parent_is_primary is False  # no primary_branch_at_start given
