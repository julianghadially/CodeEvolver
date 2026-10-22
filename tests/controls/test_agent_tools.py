"""Copyright © 2026 440 Labs LLC

Unit tests for the engine 3 architect agent tools: control violations (wrong
call order), run_eval subsample identity, the coding-agent round cap, and the
`close_iteration` tool's synchronous validations.
"""

from __future__ import annotations

import random
from typing import Any

import pytest

from src.engine.agent.harnesses import AgentRole, FakeCodingHarness, HarnessResult
from src.engine.evaluator.config import EvaluatorConfig
from src.engine.optimizer.v3 import agent_tools as agent_tools_module
from src.engine.optimizer.v3.agent_tools import (
    ControlViolation,
    RunEvalBinding,
    close_iteration_tool,
    coding_agent_tool,
)
from src.engine.optimizer.v3.errors import CloseIterationError
from src.engine.optimizer.v3.iteration_context import IterationContext


# --- Fakes -------------------------------------------------------------------


class FakeSandbox:
    def __init__(self, eval_results: list[dict]):
        self.eval_results = list(eval_results)
        self.calls: list[dict] = []
        self.head_sha = "head_sha_initial"


def _make_evaluator_config() -> EvaluatorConfig:
    return EvaluatorConfig(
        program_factory="my_pkg.program:make_program",
        metric_fns=["my_pkg.metrics:accuracy"],
        metric_names=["accuracy"],
        primary_metric="accuracy",
    )


def _successful_eval_result(primary: float = 0.5, trace_meta_rows: int = 3) -> dict:
    return {
        "success": True,
        "scores_by_metric": {"accuracy": primary},
        "row_ids": [1, 2, 3],
        "trace_dir": "/traces/iteration_0_parent",
        "trace_index_file": "/traces/iteration_0_parent/index.jsonl",
        "trace_meta": [
            {"example_id": i, "score": primary, "failed": False} for i in range(trace_meta_rows)
        ],
    }


def _locked_context(iteration_idx: int = 0) -> IterationContext:
    """Engine 3: the loop locks the context at construction."""
    return IterationContext(
        iteration_idx=iteration_idx,
        parent_locked=True,
        start_branch="codeevolver-20260601120000-main",
        start_sha="seedsha",
        primary_branch_at_start="codeevolver-20260601120000-main",
    )


# --- run_eval ---------------------------------------------------------------


def _make_run_eval_binding(
    ctx: IterationContext,
    sandbox: "FakeSandbox",
    *,
    monkeypatch: pytest.MonkeyPatch,
    trainset_size: int = 100,
    comparison_size: int = 20,
    seed: int = 0,
    subsample_size: int = 5,
    state: Any | None = None,
) -> RunEvalBinding:
    """Build a RunEvalBinding with test fakes.

    Keeping this in a helper highlights what the binding owns
    (orchestrator state: ctx, sandbox, config, dataset ids, rng, timeouts,
    subsample_size) versus what the LLM supplies on each call (only
    eval_sample + filter).
    """
    from src.engine.evaluator.sandbox_client import EvalResult
    from src.engine.optimizer.v3 import run_eval_tool as module

    def fake_run_eval(
        *,
        workspace: str,
        config: EvaluatorConfig,
        row_ids: list[int] | None,
        iteration_idx: int,
        suffix: str,
        capture_traces: bool = True,
        timeout: int | None = None,
    ) -> EvalResult:
        sandbox.calls.append(
            {
                "workspace": workspace,
                "row_ids": row_ids,
                "iteration_idx": iteration_idx,
                "suffix": suffix,
                "capture_traces": capture_traces,
                "timeout": timeout,
            }
        )
        result = sandbox.eval_results.pop(0)
        return EvalResult(
            scores_by_metric=result["scores_by_metric"],
            primary_score=result["scores_by_metric"]["accuracy"],
            trace_meta=result["trace_meta"],
            trace_dir=result["trace_dir"],
            trace_index_file=result["trace_index_file"],
            row_ids=result["row_ids"],
            raw=result,
        )

    monkeypatch.setattr(module, "_run_eval", fake_run_eval)
    monkeypatch.setattr(module, "git_get_sha", lambda workspace, ref="HEAD": sandbox.head_sha)
    monkeypatch.setattr(module, "git_get_current_branch", lambda workspace: "codeevolver-branch")
    # None (not a string): exercises the sha-only backward-compat path, so
    # every existing eval_history/eval_records assertion below stays valid.
    monkeypatch.setattr(
        module, "git_tree_signature", lambda workspace, ref=None: None
    )

    return RunEvalBinding(
        ctx=ctx,
        workspace="/workspace",
        config=_make_evaluator_config(),
        trainset_ids=list(range(trainset_size)),
        comparison_ids=list(range(comparison_size)),
        rng=random.Random(seed),
        subsample_size=subsample_size,
        state=state,
    )


def test_run_eval_requires_locked_context(monkeypatch):
    sandbox = FakeSandbox([_successful_eval_result()])
    ctx = IterationContext(iteration_idx=0)  # the loop has not locked it yet
    binding = _make_run_eval_binding(ctx, sandbox, monkeypatch=monkeypatch)

    # LLM-facing call — only two kwargs.
    with pytest.raises(ControlViolation, match="initialized by the optimizer loop"):
        binding(eval_sample="train")


def test_run_eval_subsample_identity_preserved_across_calls(monkeypatch):
    sandbox = FakeSandbox(
        [
            _successful_eval_result(primary=0.40),
            _successful_eval_result(primary=0.55),
        ]
    )
    ctx = _locked_context()

    binding = _make_run_eval_binding(ctx, sandbox, seed=42, monkeypatch=monkeypatch)

    first = binding(eval_sample="train")
    locked_ids = ctx.subsample_ids

    second = binding(eval_sample="train")

    assert ctx.subsample_ids == locked_ids, "subsample ids must be reused"
    assert ctx.initial_subsample_score == 0.40
    assert ctx.eval_history == [("head_sha_initial", 0.40), ("head_sha_initial", 0.55)]
    # Every call also lands in the full eval log with its sample kind.
    assert [rec["eval_sample"] for rec in ctx.eval_records] == ["train", "train"]
    # First call has suffix=parent, second has suffix=child.
    assert first["suffix"] == "parent"
    assert second["suffix"] == "child"


def test_run_eval_accepts_legacy_subsample_alias(monkeypatch):
    sandbox = FakeSandbox([_successful_eval_result(primary=0.40)])
    ctx = _locked_context()
    binding = _make_run_eval_binding(ctx, sandbox, monkeypatch=monkeypatch)

    result = binding(eval_sample="subsample")

    assert result["eval_sample"] == "train"
    assert ctx.initial_subsample_score == 0.40


def test_run_eval_subsample_rejects_nonpositive_subsample_size(monkeypatch):
    sandbox = FakeSandbox([])
    ctx = _locked_context()

    # Configured size of 0 is a construction-time mistake; the binding surfaces
    # it when it tries to draw the first subsample.
    binding = _make_run_eval_binding(
        ctx,
        sandbox,
        subsample_size=0,
        monkeypatch=monkeypatch,
    )

    with pytest.raises(ControlViolation, match="subsample_size must be positive"):
        binding(eval_sample="train")


def test_run_eval_rejects_unknown_eval_sample(monkeypatch):
    sandbox = FakeSandbox([])
    ctx = _locked_context()

    binding = _make_run_eval_binding(ctx, sandbox, monkeypatch=monkeypatch)

    with pytest.raises(ControlViolation, match="unknown eval_sample"):
        binding(eval_sample="bogus")  # type: ignore[arg-type]


def test_run_eval_comparison_requires_configured_comparison_set(monkeypatch):
    sandbox = FakeSandbox([])
    ctx = _locked_context()

    binding = _make_run_eval_binding(
        ctx, sandbox, comparison_size=0, monkeypatch=monkeypatch
    )

    with pytest.raises(ControlViolation, match="no comparison set"):
        binding(eval_sample="comparison")


def test_run_eval_comparison_records_history_but_not_eval_history(monkeypatch):
    class _State:
        def __init__(self):
            self.comparison_evals: list[dict] = []
            self.total_num_evals = 0

        def increment_evals(self, count):
            self.total_num_evals += count

        def record_comparison_eval(self, **kwargs):
            self.comparison_evals.append(kwargs)

    state = _State()
    sandbox = FakeSandbox([_successful_eval_result(primary=0.62)])
    ctx = _locked_context()
    binding = _make_run_eval_binding(
        ctx, sandbox, monkeypatch=monkeypatch, state=state
    )

    result = binding(eval_sample="comparison")

    assert result["suffix"] == "comparison"
    # The fixed comparison ids were evaluated verbatim.
    assert sandbox.calls[0]["row_ids"] == list(range(20))
    # Comparison evals never enter eval_history (finalize() must not see
    # them) — only the full eval_records log + the comparison history.
    assert ctx.eval_history == []
    assert [rec["eval_sample"] for rec in ctx.eval_records] == ["comparison"]
    assert len(state.comparison_evals) == 1
    recorded = state.comparison_evals[0]
    assert recorded["sha"] == "head_sha_initial"
    assert recorded["score"] == 0.62


# --- coding_agent -----------------------------------------------------------


def test_coding_agent_has_no_ordering_control():
    """No baseline-first rule on the delegation path.

    A tool-layer ordering rule would apply to `coding_agent` but not to a
    direct Bash edit, penalising delegation for something the shell does
    freely. The per-iteration baseline requirement is a `close_iteration`
    check (5c), which no tool call can route around.
    """
    ctx = _locked_context()
    assert ctx.initial_subsample_score is None
    harness = FakeCodingHarness(
        results={AgentRole.CODING: [HarnessResult.successful(output="ok")]}
    )

    result = coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="repair the branch before baselining",
        workspace="/workspace",
        agent_harness=harness,
    )

    assert result["ok"] is True
    assert ctx.coding_agent_called is True


def test_coding_agent_tracing_mode_is_not_implemented():
    ctx = IterationContext(iteration_idx=0)
    harness = FakeCodingHarness()

    with pytest.raises(NotImplementedError):
        coding_agent_tool(
            ctx=ctx,
            mode="tracing",
            change_request="instrument pipeline",
            workspace="/workspace",
            agent_harness=harness,
        )


def test_coding_agent_success_updates_context():
    ctx = _locked_context()
    ctx.initial_subsample_score = 0.40
    harness = FakeCodingHarness(
        results={AgentRole.CODING: [HarnessResult.successful(output="ok")]}
    )

    result = coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="add re-ranking",
        workspace="/workspace",
        agent_harness=harness,
    )

    assert result["ok"] is True
    assert ctx.coding_agent_called
    assert ctx.change_requests == ["add re-ranking"]
    assert [call.role for call in harness.calls] == [AgentRole.CODING]


def test_coding_agent_has_no_round_cap():
    """Change rounds are bounded by the eval budget, not a counter.

    A round cap would only apply to the `coding_agent` path; an architect
    that edits directly is never subject to it, so it would control nothing
    while penalising the delegating path.
    """
    ctx = _locked_context()
    ctx.initial_subsample_score = 0.40
    harness = FakeCodingHarness(
        results={
            AgentRole.CODING: [
                HarnessResult.successful(output=f"ok {i}") for i in range(6)
            ]
        }
    )

    for i in range(6):
        result = coding_agent_tool(
            ctx=ctx,
            mode="execute",
            change_request=f"round {i}",
            workspace="/workspace",
            agent_harness=harness,
        )
        assert result["ok"] is True, f"round {i} was rejected"

    assert len(harness.calls) == 6
    assert len(ctx.change_requests) == 6


def test_coding_agent_records_token_usage_onto_context():
    ctx = _locked_context()
    ctx.initial_subsample_score = 0.40
    # The harness surfaces the subprocess agent's token usage (and model) in
    # raw_provider_metadata["usage"], exactly as ClaudeCodeHarness.run_coding does.
    harness = FakeCodingHarness(
        results={
            AgentRole.CODING: [
                HarnessResult.successful(
                    output="ok",
                    raw_provider_metadata={
                        "usage": {
                            "input_tokens": 120,
                            "output_tokens": 30,
                            "model": "anthropic/claude-opus-4-8",
                        }
                    },
                )
            ]
        }
    )

    coding_agent_tool(
        ctx=ctx,
        mode="execute",
        change_request="add re-ranking",
        workspace="/workspace",
        agent_harness=harness,
    )

    # Model id is normalized (anthropic/ stripped) and counts land in the bucket.
    assert ctx.token_usage_by_model["claude-opus-4-8"]["input_tokens"] == 120
    assert ctx.token_usage_by_model["claude-opus-4-8"]["output_tokens"] == 30
    assert ctx.token_usage_by_model["claude-opus-4-8"]["total"] == 150


# --- close_iteration --------------------------------------------------------

_PRIMARY = "codeevolver-20260601120000-main"
_CHILD = "codeevolver-20260601130000-rerank"


def _close_kwargs(ctx: IterationContext, **overrides: Any) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "ctx": ctx,
        "workspace": "/workspace",
        "register_branch": {
            "branch": _CHILD,
            "was_primary": False,
            "is_primary": False,
        },
        "improvement_mode": "exploration",
        "memory_updated": True,
    }
    kwargs.update(overrides)
    return kwargs


def _patch_close_git(
    monkeypatch: pytest.MonkeyPatch,
    *,
    branch_exists: bool = True,
    dirty: bool = False,
    head_sha: str = "childsha",
    commit_trees: list[tuple[str, str]] | None = None,
) -> None:
    monkeypatch.setattr(
        agent_tools_module, "git_branch_exists", lambda ws, branch: branch_exists
    )
    monkeypatch.setattr(agent_tools_module, "git_is_dirty", lambda ws: dirty)
    monkeypatch.setattr(
        agent_tools_module, "git_get_sha", lambda ws, ref="HEAD": head_sha
    )
    # Empty by default, so close-time reconciliation is a no-op and every
    # existing case exercises the same sha comparison it always did.
    monkeypatch.setattr(
        agent_tools_module,
        "git_commit_trees",
        lambda *a, **k: list(commit_trees or []),
    )


def _evaluated_context() -> IterationContext:
    """Locked context whose registered-branch HEAD has been evaluated."""
    ctx = _locked_context(iteration_idx=2)
    ctx.record_eval("childsha", 0.55, eval_sample="train", branch=_CHILD, n_rows=5)
    return ctx


def test_close_iteration_rejects_list_register_branch(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="never a list"):
        close_iteration_tool(
            **_close_kwargs(ctx, register_branch=[{"branch": _CHILD}])
        )
    assert ctx.close_called is False


def test_close_iteration_rejects_missing_branch(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="non-empty string"):
        close_iteration_tool(
            **_close_kwargs(ctx, register_branch={"is_primary": False})
        )


def test_close_iteration_rejects_prefixless_branch(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="codeevolver-"):
        close_iteration_tool(
            **_close_kwargs(
                ctx,
                register_branch={"branch": "feature/rerank", "is_primary": False},
            )
        )


def test_close_iteration_rejects_nonexistent_branch(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch, branch_exists=False)

    with pytest.raises(CloseIterationError, match="does not exist"):
        close_iteration_tool(**_close_kwargs(ctx))


def test_close_iteration_rejects_dirty_tree(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch, dirty=True)

    with pytest.raises(CloseIterationError, match="uncommitted changes"):
        close_iteration_tool(**_close_kwargs(ctx))


def test_close_iteration_rejects_a_changed_program_that_was_never_measured(monkeypatch):
    """Check 5c: work landed but no training eval ran at all this iteration."""
    ctx = _locked_context(iteration_idx=2)
    assert ctx.eval_records == []
    # HEAD moved past the iteration-start sha without any eval.
    _patch_close_git(monkeypatch, head_sha="unevaluated_sha")

    with pytest.raises(CloseIterationError, match="never measured it"):
        close_iteration_tool(**_close_kwargs(ctx))
    assert ctx.close_called is False


def test_close_iteration_allows_unmoved_head_without_evals(monkeypatch):
    # A deliberate no-op: HEAD is still the iteration-start sha and the tree is
    # clean, so neither check #5 nor the baseline requirement (5c) applies —
    # declining an iteration must not cost a 50-row eval.
    ctx = _locked_context(iteration_idx=2)
    _patch_close_git(monkeypatch, head_sha=ctx.start_sha)

    result = close_iteration_tool(**_close_kwargs(ctx))

    assert result["ok"] is True
    assert ctx.close_called is True


def test_close_iteration_reconciles_a_dirty_tree_eval_before_checking(monkeypatch):
    """edit -> run_eval -> commit -> close must pass without a re-eval.

    `run_eval` stamps HEAD, which is the PARENT commit while the tree is
    dirty, so the eval lands under a sha that is not the branch HEAD at close.
    Reconciliation (between the clean-tree check and this one) re-points it
    onto the commit whose tree matches, and the plain sha check then passes.
    """
    ctx = _locked_context(iteration_idx=2)
    ctx.record_eval(
        ctx.start_sha, 0.925, eval_sample="train", branch=_CHILD, tree="tree_abc"
    )
    _patch_close_git(
        monkeypatch,
        head_sha="committed_sha",
        commit_trees=[("committed_sha", "tree_abc")],
    )

    result = close_iteration_tool(**_close_kwargs(ctx))

    assert result["ok"] is True
    assert ctx.eval_records[0]["sha"] == "committed_sha"
    assert ctx.eval_records[0]["sha_at_eval"] == ctx.start_sha
    # No re-eval was needed to satisfy the check.
    assert len(ctx.eval_records) == 1


def test_close_iteration_accepts_a_head_whose_tree_was_never_evaluated(monkeypatch):
    """The closing tree need not have been evaluated.

    A post-eval edit — a formatter run, a small follow-up — does not force a
    redundant subsample eval. The close record never claims the closing
    commit was measured (it carries `subsample_best`, not a head score), so
    there is nothing for the gate to protect; the agent-directed arm trusts
    the agent's judgement that the edit is immaterial.
    """
    ctx = _locked_context(iteration_idx=2)
    ctx.record_eval(
        ctx.start_sha, 0.5, eval_sample="train", branch=_CHILD, tree="tree_abc"
    )
    _patch_close_git(
        monkeypatch,
        head_sha="committed_sha",
        commit_trees=[("committed_sha", "tree_never_evaluated")],
    )

    result = close_iteration_tool(**_close_kwargs(ctx))

    assert result["ok"] is True
    assert ctx.close_called is True
    assert len(ctx.eval_records) == 1, "no re-eval was demanded"


def test_close_iteration_unwraps_a_legacy_nested_entry(monkeypatch):
    """A `register_branch` nested under a codebase folder name (an older
    prompt shape) still closes: a single nested entry is unwrapped rather
    than rejected."""
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    result = close_iteration_tool(
        **_close_kwargs(
            ctx,
            register_branch={
                "my-improvised-name": {
                    "branch": _CHILD, "was_primary": False, "is_primary": False,
                }
            },
        )
    )

    assert result["ok"] is True
    assert ctx.registered_branch["branch"] == _CHILD


def test_close_iteration_rejects_a_multi_entry_nested_payload(monkeypatch):
    """Multi-repo was never implemented; more than one entry has no flat
    reading, so it must not be silently resolved to one of them."""
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="non-empty string"):
        close_iteration_tool(
            **_close_kwargs(
                ctx,
                register_branch={
                    "a": {"branch": _CHILD, "was_primary": False, "is_primary": False},
                    "b": {"branch": _CHILD, "was_primary": False, "is_primary": False},
                },
            )
        )


def test_close_iteration_survives_a_git_failure_during_reconciliation(monkeypatch):
    """Degrades to sha-only behaviour instead of crashing the close path."""
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    def _boom(*a, **k):
        raise RuntimeError("git exploded")

    monkeypatch.setattr(agent_tools_module, "git_commit_trees", _boom)

    result = close_iteration_tool(**_close_kwargs(ctx))
    assert result["ok"] is True


def test_close_iteration_rejects_bad_improvement_mode(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="improvement_mode"):
        close_iteration_tool(**_close_kwargs(ctx, improvement_mode="vibes"))


def test_close_iteration_rejects_promotion_without_evidence(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    with pytest.raises(CloseIterationError, match="promotion_evidence"):
        close_iteration_tool(
            **_close_kwargs(
                ctx,
                register_branch={
                    "branch": _CHILD,
                    "was_primary": False,
                    "is_primary": True,
                },
            )
        )
    assert ctx.close_called is False


def test_close_iteration_promotion_with_evidence_succeeds(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    result = close_iteration_tool(
        **_close_kwargs(
            ctx,
            register_branch={
                "branch": _CHILD,
                "was_primary": False,
                "is_primary": True,
                "promotion_evidence": "comparison 0.62 vs primary 0.55",
            },
        )
    )

    assert result["ok"] is True
    assert result["effective_primary"] == _CHILD
    assert result["primary_fallback"] is False


def test_close_iteration_happy_path_sets_context_and_recall_amends(monkeypatch):
    ctx = _evaluated_context()
    _patch_close_git(monkeypatch)

    result = close_iteration_tool(**_close_kwargs(ctx))

    assert result["ok"] is True
    assert result["registered_branch"] == _CHILD
    assert ctx.close_called is True
    assert ctx.closed_by_fallback is False
    assert ctx.child_branch == _CHILD
    assert ctx.claimed_improvement_mode == "exploration"
    assert ctx.memory_updated_claimed is True
    assert ctx.registered_branch == {
        "branch": _CHILD,
        "was_primary": False,
        "is_primary": False,
        "promotion_evidence": None,
    }

    # Re-callable: the last call wins (e.g. after a watchdog resume).
    result2 = close_iteration_tool(
        **_close_kwargs(ctx, improvement_mode="exploitation", memory_updated=False)
    )

    assert result2["ok"] is True
    assert ctx.claimed_improvement_mode == "exploitation"
    assert ctx.memory_updated_claimed is False
    assert ctx.close_called is True
