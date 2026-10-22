"""Copyright © 2026 440 Labs LLC

The architect's tool dispatch table and the holdout deny-path fence.

`IterationArchitect.build_tool_registry` delegates here;
the functional tests that subclass the architect keep overriding that method.
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping

from src.engine.agent.harnesses import CodingHarness as AgentCodingHarness

from .agent_tools import (
    RunEvalBinding,
    close_iteration_head_tool,
    close_iteration_loop_branch_tool,
    close_iteration_tool,
    coding_agent_tool,
    inspect_traces_tool,
)
from .clean_slate import CLEAN_SLATE_REPORT_DENY_FRAGMENT
from .prompts import build_trace_inspection_prompt
from .variance_estimate import run_variance_estimate_tool
from src.engine.paths import traces_dir
from src.engine.telemetry.trace_layout import (
    HOLDOUT_DENY_FRAGMENT,
    INDEX_FILE,
    eval_trace_dir,
)

# The end-of-run meta-reflection agent is the ONLY consumer of architect traces
# we also want to add architect traces to the deny list for the per-iteration agents (nice to have). This variable is for appending outside of the main function.
# Substring fragment: every `iteration_{N}_architect_trace/` directory.
ARCHITECT_TRACE_DENY_FRAGMENT = "_architect_trace"
# `inspect_traces(trace=...)` accepted values: the iteration's train-eval
# files plus its latest comparison-set eval (without "comparison" the agent
# has to hand-roll trace reading for comparison evals).
INSPECT_TRACE_TARGETS = frozenset({"parent", "child", "comparison"})

# CodeEvolver's own constraint-checker test fixtures (`tests/functional/
# test_data/constraint_diffs/`) hold real optimization diffs. Fence them from
# every agent so a running architect can never Read/Grep/Bash these files and
# lift program ideas from them. Substring-matched (any path containing the
# directory name); `constraint_diffs` is specific enough not to collide with a
# client repo's own files. The architect's own session is fenced with the same
# fragment in `architect_hooks._make_holdout_guard_hook`.
CONSTRAINT_FIXTURE_DENY_FRAGMENT = "constraint_diffs"


def _build_holdout_deny_paths(
    testset_path: str | None,
    additional_deny_paths: list[str] | None = None,
) -> list[str]:
    """Path fragments the coding / inspection sub-agents must never touch.

    Covers the held-out test set (absolute + basename) and its eval trace
    directories (`_holdout` — every `iteration_{N}_holdout[_k]/`), plus
    CodeEvolver's constraint fixtures. Matches the fence the architect's own
    session gets via the holdout guard hook. Any client-supplied
    `additional_deny_paths` are appended (substring fragments), never
    replacing the built-in fences.
    """
    deny: list[str] = []
    if testset_path:
        deny.append(testset_path)
        deny.append(os.path.basename(testset_path))
    deny.append(HOLDOUT_DENY_FRAGMENT)
    deny.append(CONSTRAINT_FIXTURE_DENY_FRAGMENT)
    deny.append(CLEAN_SLATE_REPORT_DENY_FRAGMENT)
    if additional_deny_paths:
        deny.extend(p for p in additional_deny_paths if p)
    return deny


def build_tool_registry(
    *,
    context: Any,
    state: Any,
    workspace: str,
    cfg: Any,
    rng: random.Random,
    agent_harness: AgentCodingHarness,
) -> dict[str, Callable[..., dict]]:
    """Build the tool dispatch table the SDK will register.

    Each entry is a callable whose signature matches the tool schema the
    language model sees one-to-one. `cfg` is the `ArchitectConfig`.
    """
    # Holdout fence shared by the coding agent and the trace-inspection
    # agent: neither may read, open, or edit the held-out test set (or
    # its trace files). The architect's own SDK session is fenced
    # separately via the holdout guard hook.
    holdout_deny_paths = _build_holdout_deny_paths(
        cfg.evaluator.testset_path,
        cfg.additional_deny_paths,
    )
    # Also fence the architect's own meta-traces from these sub-agents.
    holdout_deny_paths.append(ARCHITECT_TRACE_DENY_FRAGMENT)

    run_eval = RunEvalBinding.from_agent_config(
        ctx=context,
        state=state,
        workspace=workspace,
        config=cfg,
        rng=rng,
        run_eval_gate=cfg.run_eval_gate,
        commit_before_comparison=cfg.commit_before_comparison,
    )

    def coding_agent(mode: str, change_request: str) -> dict:
        return coding_agent_tool(
            ctx=context,
            mode=mode,  # type: ignore[arg-type]
            change_request=change_request,
            workspace=workspace,
            program_path=cfg.program_path,
            agent_model=cfg.agent_model,
            deny_paths=holdout_deny_paths,
            agent_harness=agent_harness,
            commit_harness_changes=True,
        )

    def inspect_traces(query: str, trace: str) -> dict:
        # `trace` names which of the iteration's eval trace dirs to read:
        # `parent` = the baseline (first run_eval(train), before changes),
        # `child` = the latest post-change train eval, `comparison` = the
        # latest comparison-set eval. Spelled out because the file the
        # architect *thinks* it is reading and the one the sub-agent
        # actually reads must never diverge silently.
        if trace not in INSPECT_TRACE_TARGETS:
            return {
                "ok": False,
                "error": (
                    f"trace must be one of {sorted(INSPECT_TRACE_TARGETS)}, "
                    f"got {trace!r}"
                ),
            }
        trace_dir = Path(eval_trace_dir(traces_dir(), context.iteration_idx, trace))
        if not (trace_dir / INDEX_FILE).exists():
            if trace == "child":
                hint = (
                    "no train eval has run since your last change; use "
                    "trace='parent' for the baseline"
                )
            elif trace == "comparison":
                hint = (
                    "no comparison eval has run yet this iteration — run "
                    "run_eval(eval_sample='comparison') first"
                )
            else:
                hint = "run the baseline run_eval(train) first"
            return {
                "ok": False,
                "error": f"{trace_dir.name}/ does not exist yet — {hint}",
            }
        full_prompt = build_trace_inspection_prompt(
            query=query,
            iteration_num=context.iteration_idx,
            suffix=trace,
            subsample_size=cfg.subsample_size,
        )
        return inspect_traces_tool(
            ctx=context,
            query=full_prompt,
            workspace=workspace,
            agent_model=cfg.agent_model,
            deny_paths=holdout_deny_paths,
            agent_harness=agent_harness,
        )

    def close_iteration(
        register_branch: dict,
        improvement_mode: str,
        memory_updated: bool,
    ) -> dict:
        return close_iteration_tool(
            ctx=context,
            workspace=workspace,
            register_branch=register_branch,
            improvement_mode=improvement_mode,
            memory_updated=memory_updated,
            require_train_eval=run_eval.close_requires_train_eval(),
        )

    def close_iteration_loop_branch(memory_updated: bool) -> dict:
        return close_iteration_loop_branch_tool(
            ctx=context,
            workspace=workspace,
            memory_updated=memory_updated,
            require_train_eval=run_eval.close_requires_train_eval(),
        )

    def close_iteration_head(memory_updated: bool) -> dict:
        return close_iteration_head_tool(
            ctx=context,
            workspace=workspace,
            memory_updated=memory_updated,
            require_train_eval=run_eval.close_requires_train_eval(),
        )

    registry: dict[str, Callable[..., dict]] = {
        "run_eval": run_eval,
        "coding_agent": coding_agent,
        "inspect_traces": inspect_traces,
    }
    if cfg.agent_managed_git:
        if cfg.close_iteration_mode == "loop_branch":
            if not context.child_branch:
                raise RuntimeError(
                    "close_iteration_mode='loop_branch' requires the loop "
                    "to fork ctx.child_branch before the architect runs"
                )
            registry["close_iteration"] = close_iteration_loop_branch
        elif cfg.close_iteration_mode == "head":
            registry["close_iteration"] = close_iteration_head
        else:
            registry["close_iteration"] = close_iteration
    # Once-per-run noise estimate (specs/archive-public/sampling_variance.md): exposed on
    # every arm until the run has one. The registry is built once per
    # iteration, so within the iteration that takes it the tool stays
    # registered (a second call is a structured refusal) and it is absent
    # from every later iteration. `getattr`: test doubles stand in for
    # CEState here.
    if getattr(state, "variance_estimate", None) is None:
        def run_variance_estimate() -> dict:
            return run_variance_estimate_tool(
                ctx=context,
                state=state,
                binding=run_eval,
                workspace=workspace,
                n_runs=cfg.variance_estimate_samples,
            )

        registry["run_variance_estimate"] = run_variance_estimate
    if cfg.extra_tools is not None:
        # Strategy-owned tools (ASA's `branch_comparison`). Names must
        # exist in `architect_tools.ARCHITECT_TOOL_SPECS`; both harnesses
        # expose exactly the registry's members.
        extra: Mapping[str, Callable[..., dict]] = dict(
            cfg.extra_tools(context, state, workspace)
        )
        for name, fn in extra.items():
            if name in registry:
                raise RuntimeError(f"extra_tools may not override built-in tool {name!r}")
            registry[name] = fn
    return registry


__all__ = [
    "ARCHITECT_TRACE_DENY_FRAGMENT",
    "INSPECT_TRACE_TARGETS",
    "CONSTRAINT_FIXTURE_DENY_FRAGMENT",
    "_build_holdout_deny_paths",
    "build_tool_registry",
]
