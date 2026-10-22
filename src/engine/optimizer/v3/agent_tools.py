"""Copyright © 2026 440 Labs LLC

Agent-facing tool surface for the IterationArchitect.

Architectural notes for readers:

- **LLM-facing vs Python-facing signatures.** Every tool in this module has
  a public signature that matches exactly what the language model sees.
  Orchestrator dependencies (sandbox, state, dataset ids, rng, timeouts) are
  bound elsewhere — either on a binding class (`RunEvalBinding`) or passed
  as kwargs by `iteration_architect.build_tool_registry` to a free function
  (`coding_agent_tool`, `inspect_traces_tool`, `close_iteration_tool`). The
  tool registry is the single place where orchestrator state is closed over;
  the architect's SDK proxy calls the closure with *only* the arguments
  the model supplied.

- **Error handling.** Control-order violations raise `ControlViolation` (or
  a subclass in `errors.py`). The selected architect harness catches any
  exception from a tool and converts it into a structured tool-error response
  for the model — per-tool try/except blocks here would be redundant.

- **Re-exports.** This module re-exports the individual tool entry points
  so `iteration_architect.py` has a single import site.
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from src.engine.agent.harnesses import (
    AgentRole,
    CodingHarness as AgentCodingHarness,
    HarnessPermissionPolicy,
    HarnessRequest,
    HarnessResult,
)
from src.engine.agent.token_usage import TokenUsage
from .closing import resolve_primary, _coerce_bool
from .head_close import register_head
from .records import RegisteredBranch
from .errors import (
    CloseIterationError,
    ControlViolation,
    RunEvalError,
    SelectCandidateError,
)
from src.engine.services.subprocess_service import (
    git_add_and_commit,
    git_branch_exists,
    git_commit_trees,
    git_get_current_branch,
    git_get_sha,
    git_is_dirty,
)
from .iteration_context import TRAIN_EVAL_SAMPLES, IterationContext
from .run_eval_tool import RunEvalBinding

logger = logging.getLogger(__name__)


CodingAgentMode = Literal["execute", "tracing"]


def _record_harness_usage(
    ctx: IterationContext,
    harness_result: HarnessResult,
    *,
    fallback_model: str | None = None,
) -> None:
    """Fold a subprocess tool's token usage onto the iteration context.

    The harness surfaces the subprocess agent's token usage (and the model it
    ran on) in `raw_provider_metadata["usage"]`. Best-effort: an OpenCode
    harness (or any harness that doesn't report usage) yields an empty dict,
    which records as zero under the resolved model.
    """
    usage_dict = (harness_result.raw_provider_metadata or {}).get("usage") or {}
    model = usage_dict.get("model") or fallback_model
    ctx.record_token_usage(model, TokenUsage.from_dict(usage_dict))


def coding_agent_tool(
    *,
    ctx: IterationContext,
    mode: CodingAgentMode,
    change_request: str,
    workspace: str,
    agent_harness: AgentCodingHarness,
    program_path: str | None = None,
    max_turns: int = 50,
    agent_model: str | None = None,
    deny_paths: list[str] | None = None,
    commit_harness_changes: bool = False,
) -> dict[str, Any]:
    """Wrapper over execution.run_coding_agent.

    LLM-facing signature (after `iteration_architect.build_tool_registry`
    binds `ctx`, `workspace`, `program_path`, `max_turns`):

        coding_agent(mode: "execute" | "tracing", change_request: str) -> dict

    `mode="tracing"` is deferred — raises `NotImplementedError` so the
    architect sees a clear error and proceeds with `mode="execute"`.
    """
    if mode == "tracing":
        raise NotImplementedError(
            "coding_agent(mode='tracing') is deferred. Use mode='execute' "
            "and assume the client program is already OTel-instrumented."
        )
    if mode != "execute":
        raise ControlViolation(
            f"coding_agent mode must be 'execute' or 'tracing', got {mode!r}"
        )

    logger.info(
        "iteration %d: coding_agent called (mode=%s, change_request=%r)",
        ctx.iteration_idx,
        mode,
        change_request[:200],
    )

    harness_result = agent_harness.run_coding(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=workspace,
            prompt=change_request,
            permissions=HarnessPermissionPolicy.autonomous_policy(),
            max_turns=max_turns,
            raw_provider_metadata={
                "mode": mode,
                "program_path": program_path,
                "commit_changes": True,
                "agent_model": agent_model,
                "deny_paths": deny_paths,
            },
        )
    )
    if (
        commit_harness_changes
        and harness_result.success
        and not harness_result.raw_provider_metadata.get("commit_handled")
    ):
        try:
            committed = git_add_and_commit(workspace, "Code mutation")
        except RuntimeError as exc:
            harness_result = HarnessResult.failed(
                error=f"Git commit failed: {exc}",
                output=harness_result.output,
                session_id=harness_result.session_id,
                raw_provider_metadata=harness_result.raw_provider_metadata,
            )
        else:
            raw_metadata = dict(harness_result.raw_provider_metadata)
            raw_metadata["commit_handled"] = True
            raw_metadata["committed"] = committed
            harness_result = HarnessResult(
                success=harness_result.success,
                error=harness_result.error,
                output=harness_result.output,
                session_id=harness_result.session_id,
                raw_provider_metadata=raw_metadata,
            )
    result = {
        "success": harness_result.success,
        "error": harness_result.error,
        "output": harness_result.output,
    }
    ctx.record_coding_agent(change_request)
    _record_harness_usage(ctx, harness_result, fallback_model=agent_model)

    logger.info(
        "iteration %d: coding_agent returned (success=%s)",
        ctx.iteration_idx,
        bool(result.get("success")),
    )
    return {
        "ok": bool(result.get("success")),
        "error": result.get("error"),
        "output": result.get("output"),
    }


def inspect_traces_tool(
    *,
    ctx: IterationContext,
    query: str,
    workspace: str,
    agent_harness: AgentCodingHarness,
    iteration_num: int | None = None,
    agent_model: str | None = None,
    deny_paths: list[str] | None = None,
) -> dict[str, Any]:
    """Wrapper over subprocess_service.run_inspection_agent in read-only mode.

    LLM-facing signature (after binding): `inspect_traces(query: str) -> dict`.
    The full trace-agent prompt (see `prompts.build_trace_inspection_prompt`)
    is built by `iteration_architect.build_tool_registry` from the LLM's
    `query` so the file paths are consistent with the iteration's suffix.

    `deny_paths` fences the trace-inspection agent off the valset and the
    held-out test set (and their trace files) — it should only ever read the
    iteration's own trace files.
    """
    harness_result = agent_harness.run_inspection(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=workspace,
            prompt=query,
            permissions=HarnessPermissionPolicy.autonomous_policy(),
            max_turns=15,
            raw_provider_metadata={
                "iteration_num": (
                    iteration_num if iteration_num is not None else ctx.iteration_idx
                ),
                "writable": False,
                "agent_model": agent_model,
                "deny_paths": deny_paths,
            },
        )
    )
    result = {
        "success": harness_result.success,
        "proposed_change": harness_result.output,
        "error": harness_result.error,
    }
    _record_harness_usage(ctx, harness_result, fallback_model=agent_model)
    return {
        "ok": bool(result.get("success")),
        "summary": result.get("proposed_change"),
        "error": result.get("error"),
    }


def close_iteration_tool(
    *,
    ctx: IterationContext,
    workspace: str,
    register_branch: Any,
    improvement_mode: str,
    memory_updated: Any,
    require_train_eval: bool = True,
) -> dict[str, Any]:
    """The engine 3 `close_iteration` tool — finalize an iteration.

    LLM-facing signature (after binding):

        close_iteration(register_branch: dict, improvement_mode: str,
                        memory_updated: bool) -> dict

    Synchronous validation only; the deterministic close (parent derivation,
    finalize, constraint gate, state write) runs post-session in the loop.
    Violations raise `CloseIterationError` — a structured tool error the
    model can fix and retry. The tool is re-callable: the last successful
    call wins, so a watchdog resume can amend an earlier registration.
    """
    # 1) Shape: a FLAT OBJECT describing the one branch being submitted —
    # never a list. A legacy single-entry payload nested under a codebase
    # folder name is unwrapped rather than rejected so a model working
    # from an older prompt still closes.
    if isinstance(register_branch, list):
        raise CloseIterationError(
            "register_branch must be a JSON object describing the one branch "
            "you are submitting, never a list. Use "
            '{"branch": ..., "was_primary": ..., "is_primary": ..., '
            '"promotion_evidence": ...}.'
        )
    if not isinstance(register_branch, dict) or not register_branch:
        raise CloseIterationError(
            "register_branch must be a non-empty JSON object: "
            '{"branch": ..., "was_primary": ..., "is_primary": ..., '
            '"promotion_evidence": ...}.'
        )
    if "branch" not in register_branch and len(register_branch) == 1:
        only_key, only_value = next(iter(register_branch.items()))
        if isinstance(only_value, dict) and "branch" in only_value:
            logger.info(
                "iteration %d: register_branch arrived in the legacy form "
                "nested under %r; unwrapping to the flat entry",
                ctx.iteration_idx, only_key,
            )
            register_branch = only_value

    branch = register_branch.get("branch")
    if not isinstance(branch, str) or not branch.strip():
        raise CloseIterationError(
            "register_branch.branch must be a non-empty string."
        )
    branch = branch.strip()
    if not branch.startswith("codeevolver-"):
        raise CloseIterationError(
            f"register_branch.branch={branch!r} must start with "
            "'codeevolver-'. Work must live on codeevolver- branches."
        )
    ws_entry = RegisteredBranch(
        branch=branch,
        was_primary=_coerce_bool(register_branch.get("was_primary")),
        is_primary=_coerce_bool(register_branch.get("is_primary")),
        promotion_evidence=(
            register_branch.get("promotion_evidence") or ""
        ).strip() or None,
    )
    ws_branch = ws_entry["branch"]

    _validate_committed_close(
        ctx, workspace, ws_branch, require_train_eval=require_train_eval
    )

    # 6) improvement_mode / memory_updated.
    memory_updated_bool = _validate_claims(improvement_mode, memory_updated)

    # 7) New-primary promotion requires evidence.
    prior_primary = ctx.primary_branch_at_start
    if (
        ws_entry["is_primary"]
        and prior_primary
        and ws_branch != prior_primary
        and not ws_entry["promotion_evidence"]
    ):
        raise CloseIterationError(
            f"Promoting {ws_branch!r} to primary (replacing {prior_primary!r}) "
            "requires promotion_evidence — e.g. comparison-set scores for both "
            "branches. Provide evidence or set is_primary=false."
        )

    # 8) Primary fallback (warn, never reject).
    effective_primary, primary_fallback = resolve_primary(
        ws_entry, prior_primary or ws_branch,
    )

    ctx.registered_branch = ws_entry
    ctx.claimed_improvement_mode = improvement_mode
    ctx.memory_updated_claimed = memory_updated_bool
    ctx.child_branch = ws_branch
    ctx.close_called = True
    ctx.closed_by_fallback = False

    logger.info(
        "iteration %d: close_iteration registered branch=%s primary=%s "
        "(fallback=%s) mode=%s memory_updated=%s",
        ctx.iteration_idx, ws_branch, effective_primary, primary_fallback,
        improvement_mode, memory_updated_bool,
    )
    return {
        "ok": True,
        "registered_branch": ws_branch,
        "effective_primary": effective_primary,
        "primary_fallback": primary_fallback,
        "note": (
            "Iteration registered. You may exit now. Re-calling "
            "close_iteration amends this registration (last call wins)."
        ),
    }


def close_iteration_loop_branch_tool(
    *,
    ctx: IterationContext,
    workspace: str,
    memory_updated: Any,
    require_train_eval: bool = True,
) -> dict[str, Any]:
    """The loop-owned-branch `close_iteration` tool (pareto_evolution).

    LLM-facing signature (after binding):

        close_iteration(memory_updated: bool) -> dict

    The optimizer forked this iteration's branch (`ctx.child_branch`) and
    owns candidate retention and the primary, so there is nothing to
    register: the tool only checks the architect stayed on its branch and
    left it committed + evaluated, and records the memory claim. No
    improvement_mode — the arm never hears of exploration. A
    synthesized single-entry registration keeps `closing.build_close_record`
    unchanged; the strategy's close hook decides the primary afterwards.
    """
    branch = ctx.child_branch
    if not branch:
        raise CloseIterationError(
            "This iteration has no loop-assigned branch; cannot close."
        )
    try:
        current = git_get_current_branch(workspace)
    except Exception:  # noqa: BLE001
        current = None
    if current != branch:
        raise CloseIterationError(
            f"You are on branch {current!r} but this iteration's work must be "
            f"on {branch!r}. Do not switch branches: `git checkout {branch}` "
            "(bring your commits with you if needed) and call close_iteration "
            "again."
        )

    _validate_committed_close(
        ctx, workspace, branch, require_train_eval=require_train_eval
    )
    memory_updated_bool = _coerce_bool(memory_updated)

    ctx.registered_branch = RegisteredBranch(
        branch=branch,
        was_primary=bool(ctx.parent_is_primary),
        is_primary=False,
        promotion_evidence=None,
    )
    ctx.memory_updated_claimed = memory_updated_bool
    ctx.close_called = True
    ctx.closed_by_fallback = False

    logger.info(
        "iteration %d: close_iteration (loop branch) branch=%s memory_updated=%s",
        ctx.iteration_idx, branch, memory_updated_bool,
    )
    return {
        "ok": True,
        "branch": branch,
        "note": (
            "Iteration closed. You may exit now. Re-calling close_iteration "
            "amends the claim (last call wins)."
        ),
    }


def close_iteration_head_tool(
    *,
    ctx: IterationContext,
    workspace: str,
    memory_updated: Any,
    require_train_eval: bool = True,
) -> dict[str, Any]:
    """The HEAD-registering `close_iteration` tool (freeform).

    LLM-facing signature (after binding):

        close_iteration(memory_updated: bool) -> dict

    The agent registers nothing: `register_head` derives the submitted
    branch from HEAD (auto-committing a dirty tree, prefixing / naming the
    branch as needed) and the shared committed-close checks then run on
    that branch. No improvement_mode — the arm never hears of exploration.
    """
    entry = register_head(ctx, workspace)
    branch = entry["branch"]
    _validate_committed_close(
        ctx, workspace, branch, require_train_eval=require_train_eval
    )
    memory_updated_bool = _coerce_bool(memory_updated)

    ctx.memory_updated_claimed = memory_updated_bool
    ctx.close_called = True
    ctx.closed_by_fallback = False

    logger.info(
        "iteration %d: close_iteration (head) branch=%s memory_updated=%s",
        ctx.iteration_idx, branch, memory_updated_bool,
    )
    return {
        "ok": True,
        "branch": branch,
        "note": (
            "Iteration closed. You may exit now. Whatever HEAD is when the "
            "session ends is your submitted state. Re-calling close_iteration "
            "amends the claim (last call wins)."
        ),
    }


def _validate_claims(improvement_mode: str, memory_updated: Any) -> bool:
    """Check 6 (EE `register_branch` tool only): improvement_mode /
    memory_updated. Returns the coerced bool."""
    if improvement_mode not in ("exploration", "exploitation"):
        raise CloseIterationError(
            "improvement_mode must be 'exploration' or 'exploitation', got "
            f"{improvement_mode!r}."
        )
    return _coerce_bool(memory_updated)


def _validate_committed_close(
    ctx: IterationContext,
    workspace: str,
    ws_branch: str,
    *,
    require_train_eval: bool = True,
) -> None:
    """Checks 3 / 4 / 4b / 5c — shared by both close_iteration variants.

    The branch exists, the tree is clean, eval shas are reconciled onto their
    commits, and — when `require_train_eval` — a train baseline ran.

    `require_train_eval` comes from `RunEvalBinding.close_requires_train_eval`
    (see its docstring): False means no train eval is reachable this iteration,
    so demanding one would deadlock the close rather than enforce anything. It
    defaults True so every caller that does not thread it keeps the strict
    behaviour.

    There is deliberately no "the program at HEAD was evaluated" check. It
    would force a redundant subsample eval whenever the architect touched a
    program file after its last eval (even a formatter run) for no upside:
    the close record never claims the closing commit was measured (see
    `records.CloseRecord`), the constraint gate diffs commits not evals, and
    the held-out finalization measures the final primary regardless. The
    agent-directed arm trusts the agent's judgement that a post-eval edit is
    immaterial; the evolution arm re-measures the closing content on the
    comparison set by tree when it differs (`comparison_record_for_tree`).
    """
    # 3) The registered workspace branch must actually exist in git.
    if not git_branch_exists(workspace, ws_branch):
        raise CloseIterationError(
            f"Branch {ws_branch!r} does not exist in the workspace repo. "
            "Register the branch your committed work is on."
        )

    # 4) Committed-work nudge. The post-session workstation-committed check
    # is the safety net; here we push the agent to commit deliberately.
    if git_is_dirty(workspace):
        raise CloseIterationError(
            "The working tree has uncommitted changes. Commit your work "
            "(git add -A && git commit) — or git reset --hard to discard it — "
            "then call close_iteration again."
        )

    # 4b) Reconcile eval shas onto their commits.
    #
    # `run_eval` stamps HEAD's sha, but the evaluator imports client code from
    # the WORKING TREE — so an eval of uncommitted edits is recorded under the
    # PARENT commit's sha. Check #4 above has just confirmed a clean tree, so
    # every such eval's content either now exists as a commit (and gets
    # re-pointed here) or genuinely does not (and stays anchored as a bare
    # tree by the loop).
    #
    # Reconciling HERE, at the dirty->committed boundary, is what lets the
    # revert check and every other consumer stay plain sha comparisons, and
    # what marks each record `committed` once its content has a commit.
    try:
        reconciled = ctx.reconcile_eval_shas(
            git_commit_trees(workspace, ws_branch, since=ctx.start_sha)
        )
    except Exception as exc:  # noqa: BLE001 - degrade to sha-only behaviour
        logger.warning(
            "iteration %d: close-time eval-sha reconciliation failed "
            "(non-fatal): %s",
            ctx.iteration_idx, exc,
        )
    else:
        if reconciled:
            logger.info(
                "iteration %d: re-pointed %d eval record(s) onto their "
                "committed sha via tree match",
                ctx.iteration_idx, reconciled,
            )

    head_sha = git_get_sha(workspace, ws_branch)

    # 5c) A training baseline ran this iteration.
    #
    # This is where the per-iteration baseline requirement lives. It is not
    # enforced at the tool layer as "no edits before the baseline": such a rule
    # could only be applied by pattern-matching tool calls, leaks through
    # Bash (a primary edit path for current models), and penalises whichever
    # change path happens to be matched. Editing before the baseline is
    # permitted; an agent-directed search may need to repair a branch before a
    # valid baseline is possible, and `research_plan.md` holds the *existence*
    # of a per-iteration baseline constant across arms, not its ordering.
    #
    # As a close check it is deterministic and unbypassable: no tool call can
    # route around it, and there is nothing to enumerate.
    #
    # Keyed on "a train eval ran", not on `initial_subsample_score` being set —
    # a ceiling baseline is deliberately not committed as the score, and an
    # iteration whose every sampled row is already solved must still close.
    #
    # Exempt: a genuine no-op. HEAD unmoved from the iteration's start and (per
    # check #4) a clean tree means the architect deliberately aborted without
    # touching the program — the no-change watchdog explicitly offers that. It
    # should not cost a 50-row eval to decline an iteration.
    # A 1-row `single` smoke check does not count as measuring the program.
    #
    # Also exempt: no train eval is REACHABLE this iteration (`require_train_eval`
    # False — fewer than 2x subsample rows left, or the evo comparison latch is
    # engaged). Enforcing a baseline the architect cannot run does not anchor
    # anything; it deadlocks the close and forces a validated change to be
    # reverted just to escape.
    if not require_train_eval:
        logger.info(
            "close check 5c waived: no train eval is available this iteration "
            "(budget below one before/after pair, or the comparison run has "
            "already closed evals on this arm)."
        )
    elif head_sha != ctx.start_sha and not any(
        rec.get("eval_sample") in TRAIN_EVAL_SAMPLES
        and rec.get("eval_sample") != "single"
        for rec in ctx.eval_records
    ):
        logger.info("CloseIterationError surfaced to agent for iteration with no eval.")
        raise CloseIterationError(
            "This iteration changed the program but never measured it. Every "
            "iteration that does work must run at least one training eval — "
            "call run_eval(eval_sample='train'), then close_iteration again. "
            "(An iteration that deliberately changes nothing may close without "
            "one.)"
        )


__all__ = [
    "CodingAgentMode",
    "ControlViolation",
    "RunEvalBinding",
    "RunEvalError",
    "SelectCandidateError",
    "close_iteration_head_tool",
    "close_iteration_loop_branch_tool",
    "close_iteration_tool",
    "coding_agent_tool",
    "inspect_traces_tool",
]
