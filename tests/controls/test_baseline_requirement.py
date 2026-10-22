"""Copyright © 2026 440 Labs LLC

Where the per-iteration baseline requirement lives, and what it records.

It does not live at the tool layer as "no Edit/Write before the baseline
eval", for two reasons:

- **Wrong layer.** Applying an ordering rule to tool calls means enumerating
  the ways a file can change. A hook covering Edit/Write/apply_patch leaks
  through Bash — a primary edit path for current models — so a model that
  prefers Edit is constrained and one that prefers `sed -i` is not. In a
  cross-model, cross-arm experiment that is friction the harness introduces,
  not a control it applies.
- **Wrong rule.** `specs/research_plan.md` defines the agent-directed arm by
  the agent making these calls; it holds the *existence* of a per-iteration
  baseline constant across arms, not its ordering. An agent may legitimately
  need to repair a branch before a valid baseline is possible.

Where it lives instead:

- `close_iteration` check 5c — a training eval ran this iteration.
  Deterministic and unbypassable; nothing to enumerate.
- `run_eval` records `baseline_tree_dirty` — did that baseline measure the
  parent, or the parent plus edits already made? Observability, not a gate.

The only thing the pre-edit hook decides is the `apply_patch` holdout
fence, covered at the bottom.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.engine.optimizer.v3.architect_hooks import check_pre_edit_guard_decision
from src.engine.optimizer.v3.errors import CloseIterationError
from src.engine.optimizer.v3.iteration_context import IterationContext


# --- close gate: a training eval ran this iteration --------------------------


def _close_kwargs(ctx, workspace, **overrides):
    kwargs = {
        "ctx": ctx,
        "workspace": workspace,
        "register_branch": {
            "branch": "codeevolver-x",
            "was_primary": True,
            "is_primary": True,
        },
        "improvement_mode": "exploitation",
        "memory_updated": True,
    }
    kwargs.update(overrides)
    return kwargs


def _repo(tmp_path):
    import subprocess

    def git(*args):
        subprocess.run(["git", *args], cwd=tmp_path, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "t@t")
    git("config", "user.name", "t")
    (tmp_path / "program.py").write_text("K = 7\n")
    git("add", "-A")
    git("commit", "-q", "-m", "seed")
    git("checkout", "-q", "-b", "codeevolver-x")
    return tmp_path


def _ctx_at(repo) -> IterationContext:
    from src.engine.services.subprocess_service import git_get_sha

    return IterationContext(
        iteration_idx=1,
        parent_locked=True,
        start_branch="codeevolver-x",
        start_sha=git_get_sha(str(repo)),
        primary_branch_at_start="codeevolver-x",
    )


def _commit_a_change(repo):
    """Move HEAD, so the iteration counts as having done work."""
    import subprocess

    (repo / "program.py").write_text("K = 20\n")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "change"], cwd=repo, check=True,
        capture_output=True,
    )


def test_close_is_rejected_when_work_landed_but_nothing_was_measured(tmp_path):
    """Check 5c: a changed program with no training eval at all cannot close.

    5c is the baseline requirement; the closing tree itself need not have
    been evaluated.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_tool(**_close_kwargs(ctx, str(repo)))


def test_a_comparison_eval_alone_does_not_satisfy_the_requirement(tmp_path):
    """Comparison evals compare branches; they are not the iteration's baseline.

    Check #5 would accept one (it asks only whether HEAD's program was
    evaluated), so this is the check that keeps every working iteration
    anchored to a training measurement.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool
    from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)
    ctx.record_eval(
        git_get_sha(str(repo)), 0.9, eval_sample="comparison",
        tree=git_tree_signature(str(repo)),
    )

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_tool(**_close_kwargs(ctx, str(repo)))


def test_a_single_row_smoke_eval_alone_does_not_satisfy_the_requirement(tmp_path):
    """`single` is a 1-row smoke check ("does it still run"), not a measurement
    of the program. Check 5c wants a training subsample."""
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool
    from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)
    ctx.record_eval(
        git_get_sha(str(repo)), 1.0, eval_sample="single", n_rows=1,
        tree=git_tree_signature(str(repo)),
    )

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_tool(**_close_kwargs(ctx, str(repo)))


def test_close_is_waived_when_no_train_pair_is_affordable(tmp_path):
    """5c is enforced only while a before/after train pair is affordable.

    Otherwise the engine would demand a train baseline the architect cannot
    run (`run_eval` refuses it because the budget is spent), and the only
    escape would be reverting validated work.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)

    result = close_iteration_tool(
        **_close_kwargs(ctx, str(repo), require_train_eval=False)
    )
    assert result["ok"] is True


def test_close_still_demands_a_train_eval_while_the_pair_is_affordable(tmp_path):
    """The waiver must not become a general escape hatch.

    With rows to spare, a changed program with only a comparison eval is still
    refused — this is the research condition held constant across arms.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool
    from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)
    ctx.record_eval(
        git_get_sha(str(repo)), 0.9, eval_sample="comparison",
        tree=git_tree_signature(str(repo)),
    )

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_tool(
            **_close_kwargs(ctx, str(repo), require_train_eval=True)
        )


def test_the_waiver_reaches_the_loop_branch_and_head_close_variants(tmp_path):
    """All three close tools share `_validate_committed_close`, so all three
    must honour the waiver, including the evo (loop_branch) variant."""
    from src.engine.optimizer.v3.agent_tools import (
        close_iteration_head_tool,
        close_iteration_loop_branch_tool,
    )

    repo = _repo(tmp_path)
    # Both contexts must be pinned to the pre-change sha, or HEAD matches
    # `start_sha` and the no-op exemption fires instead of 5c.
    ctx = _ctx_at(repo)
    ctx.child_branch = "codeevolver-x"
    ctx2 = _ctx_at(repo)
    _commit_a_change(repo)

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_loop_branch_tool(
            ctx=ctx, workspace=str(repo),
            memory_updated=True, require_train_eval=True,
        )
    assert close_iteration_loop_branch_tool(
        ctx=ctx, workspace=str(repo),
        memory_updated=True, require_train_eval=False,
    )["ok"] is True

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_head_tool(
            ctx=ctx2, workspace=str(repo), memory_updated=True,
            require_train_eval=True,
        )
    assert close_iteration_head_tool(
        ctx=ctx2, workspace=str(repo), memory_updated=True,
        require_train_eval=False,
    )["ok"] is True


def test_a_deliberate_no_op_iteration_may_close_without_an_eval(tmp_path):
    """Declining an iteration must not cost a 50-row eval.

    HEAD unmoved plus a clean tree (check #4) is a genuine abort — the
    no-change watchdog explicitly offers it as an option.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)

    assert close_iteration_tool(**_close_kwargs(ctx, str(repo)))["ok"] is True


def test_close_is_accepted_once_a_training_eval_has_run(tmp_path):
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool
    from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)
    ctx.record_eval(
        git_get_sha(str(repo)), 0.5, eval_sample="train",
        tree=git_tree_signature(str(repo)),
    )
    ctx.initial_subsample_score = 0.5

    assert close_iteration_tool(**_close_kwargs(ctx, str(repo)))["ok"] is True


def test_a_ceiling_baseline_still_lets_the_iteration_close(tmp_path):
    """`initial_subsample_score` is deliberately NOT set at ceiling.

    Keying the check on "a train eval ran" rather than on a committed baseline
    score is what keeps an all-rows-already-solved iteration closable.
    """
    from src.engine.optimizer.v3.agent_tools import close_iteration_tool
    from src.engine.services.subprocess_service import git_get_sha, git_tree_signature

    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    _commit_a_change(repo)
    ctx.record_eval(
        git_get_sha(str(repo)), 1.0, eval_sample="train",
        tree=git_tree_signature(str(repo)),
    )
    assert ctx.initial_subsample_score is None  # ceiling: not committed

    assert close_iteration_tool(**_close_kwargs(ctx, str(repo)))["ok"] is True


# --- run_eval records what the baseline measured -----------------------------


def _binding(ctx, repo, monkeypatch, score=0.5):
    """A RunEvalBinding whose evaluator is faked but whose git reads are real."""
    from src.engine.evaluator.config import EvaluatorConfig
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3 import run_eval_tool as module
    from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding

    def fake_run_eval(**kwargs):
        return EvalResult(
            scores_by_metric={"accuracy": score},
            primary_score=score,
            trace_meta=[{"example_id": i, "score": score} for i in range(3)],
            trace_dir="/traces/t.jsonl",
            trace_index_file="/traces/t_index.jsonl",
            row_ids=[1, 2, 3],
            raw={},
        )

    monkeypatch.setattr(module, "_run_eval", fake_run_eval)
    return RunEvalBinding(
        ctx=ctx,
        workspace=str(repo),
        config=EvaluatorConfig(
            program_factory="p:make",
            metric_fns=["m:acc"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
        ),
        trainset_ids=list(range(50)),
        rng=random.Random(0),
        subsample_size=3,
    )


def test_baseline_on_a_clean_tree_is_recorded_as_clean(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)

    _binding(ctx, repo, monkeypatch)(eval_sample="train")

    assert ctx.baseline_tree_dirty is False
    assert ctx.baseline_dirty_paths == []


def test_baseline_on_a_modified_tree_is_recorded_with_the_paths(
    tmp_path, monkeypatch
):
    """Editing before the baseline is allowed — and it is written down.

    `subsample_baseline` then describes the parent PLUS these edits, so the
    iteration's baseline→final delta understates the change. That is a reading
    instruction for the analysis, not a violation.
    """
    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    (repo / "program.py").write_text("K = 20\n")  # edited before any eval

    _binding(ctx, repo, monkeypatch)(eval_sample="train")

    assert ctx.baseline_tree_dirty is True
    assert ctx.baseline_dirty_paths == ["program.py"]
    assert ctx.history_fields()["baseline_ran_on_modified_tree"] is True


def test_an_untracked_new_file_counts_as_a_modified_tree(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    (repo / "reranker.py").write_text("def rerank(): ...\n")

    _binding(ctx, repo, monkeypatch)(eval_sample="train")

    assert ctx.baseline_tree_dirty is True
    assert "reranker.py" in ctx.baseline_dirty_paths


def test_purity_is_recorded_once_from_the_first_training_eval(
    tmp_path, monkeypatch
):
    """A later dirty eval is the NORMAL flow (edit -> eval) and must not
    retroactively mark the baseline dirty."""
    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)
    binding = _binding(ctx, repo, monkeypatch)

    binding(eval_sample="train")           # clean baseline
    (repo / "program.py").write_text("K = 20\n")
    binding(eval_sample="train")           # the change being measured

    assert ctx.baseline_tree_dirty is False
    assert ctx.baseline_dirty_paths == []


def test_no_training_eval_leaves_purity_unknown(tmp_path):
    repo = _repo(tmp_path)
    ctx = _ctx_at(repo)

    assert ctx.baseline_tree_dirty is None
    assert ctx.history_fields()["baseline_ran_on_modified_tree"] is None


# --- the one thing the pre-edit hook still decides ---------------------------


_HOLDOUT = ["data/hoverBench_test.json", "_holdout"]


def _decide(tool_name: str, tool_input: dict[str, Any], deny=None):
    return check_pre_edit_guard_decision(
        IterationContext(iteration_idx=1),
        "/workspace",
        tool_name=tool_name,
        tool_input=tool_input,
        deny_paths=deny if deny is not None else _HOLDOUT,
    )


def test_apply_patch_targeting_the_holdout_is_denied():
    patch = (
        "*** Begin Patch\n"
        "*** Update File: data/hoverBench_test.json\n"
        "+leak\n"
        "*** End Patch\n"
    )
    decision = _decide("apply_patch", {"patchText": patch})

    assert decision.allowed is False
    assert "holdout" in (decision.reason or "")


def test_apply_patch_targeting_the_program_is_allowed():
    """No ordering control remains — only the holdout fence."""
    patch = (
        "*** Begin Patch\n"
        "*** Update File: langProBe/hover/hover_program.py\n"
        "+K = 20\n"
        "*** End Patch\n"
    )
    assert _decide("apply_patch", {"patchText": patch}).allowed is True


def test_an_unparseable_patch_is_allowed_now_that_there_is_no_ordering_rule():
    """The old fail-closed branch existed to catch an unreadable patch landing
    before the iteration was initialized. With that rule gone, an unparseable
    patch has no holdout target to deny."""
    assert _decide("apply_patch", {"patchText": "garbage"}).allowed is True


@pytest.mark.parametrize(
    "tool, tool_input",
    [
        ("edit", {"filePath": "/workspace/program.py"}),
        ("write", {"filePath": "/workspace/program.py"}),
        ("bash", {"command": "sed -i 's/k=7/k=20/' program.py"}),
        ("bash", {"command": "git checkout other-branch"}),
        ("read", {"filePath": "/workspace/program.py"}),
    ],
)
def test_every_other_tool_passes_through(tool, tool_input):
    """edit/write/bash are fenced by the permission config and the holdout
    hook; there is no ordering control left for this guard to apply."""
    assert _decide(tool, tool_input).allowed is True


# --- the waiver is wired into the REAL architect registry -------------------
#
# The tests above call the close tools directly. These drive
# `IterationArchitect.build_tool_registry`, so they prove the registry actually
# consults `RunEvalBinding.close_requires_train_eval()` at close time against a
# live budget — the wiring an end-to-end run would exercise, without an LLM.


def _wired_close(monkeypatch, *, used: int, max_eval_rows: int | None):
    """Real registry + real budget; HEAD moved, zero train evals."""
    from src.engine.optimizer.v3 import agent_tools as agent_tools_module
    from tests.controls._arch_helpers import RecordingHarness, make_architect

    arch, ctx, state = make_architect()          # subsample_size=5 → pair = 10
    ctx.max_eval_rows = max_eval_rows
    state.total_num_evals = used
    monkeypatch.setattr(agent_tools_module, "git_branch_exists", lambda ws, b: True)
    monkeypatch.setattr(agent_tools_module, "git_is_dirty", lambda ws: False)
    monkeypatch.setattr(
        agent_tools_module, "git_get_sha", lambda ws, ref="HEAD": "movedsha"
    )
    monkeypatch.setattr(agent_tools_module, "git_commit_trees", lambda *a, **k: [])
    registry = arch.build_tool_registry(agent_harness=RecordingHarness())
    return registry, ctx


def test_registry_close_is_refused_while_a_pair_is_affordable(monkeypatch):
    registry, _ = _wired_close(monkeypatch, used=0, max_eval_rows=100)
    with pytest.raises(CloseIterationError, match="never measured it"):
        registry["close_iteration"](
            register_branch={
                "branch": "codeevolver-20260601120000-main",
                "was_primary": True, "is_primary": True,
            },
            improvement_mode="exploitation",
            memory_updated=True,
        )


def test_registry_close_succeeds_once_the_pair_is_unaffordable(monkeypatch):
    # 96/100 used → 4 rows left, below the 10-row pair. The deadlock state.
    registry, _ = _wired_close(monkeypatch, used=96, max_eval_rows=100)
    result = registry["close_iteration"](
        register_branch={
            "branch": "codeevolver-20260601120000-main",
            "was_primary": True, "is_primary": True,
        },
        improvement_mode="exploitation",
        memory_updated=True,
    )
    assert result["ok"] is True
