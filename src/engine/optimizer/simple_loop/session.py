"""Copyright © 2026 440 Labs LLC

One agent session: a fresh OpenCode process with a single `run_eval` tool.

The harness receives the `ArchitectConfig` built by the loop
(`agent_managed_git=False`, `stage_project_skills=False`), so the MCP bridge
exposes exactly `["run_eval"]`, no skills or bootstrap commands are staged,
and the process is spawned as the unprivileged agent user when one is
configured. The loop never spawns agent processes itself.
"""

from __future__ import annotations

import logging
import random

from src.engine.agent.harnesses import (
    AgentRole,
    HarnessPermissionPolicy,
    HarnessRequest,
    HarnessResult,
    create_coding_harness,
)
from src.engine.agent.token_usage import TokenUsage
from src.engine.schemas.agent_config import BaseAgentConfig
from src.engine.schemas.iteration_context import BaseIterationContext
from src.engine.optimizer.v3.run_eval_tool import RunEvalBinding

from .prompt import build_kickoff_message

logger = logging.getLogger(__name__)


def build_run_eval(
    *,
    ctx: BaseIterationContext,
    state: object,
    workspace: str,
    config: BaseAgentConfig,
    rng: random.Random,
) -> RunEvalBinding:
    """The session's only tool — the shared CodeEvolver eval mechanic."""
    return RunEvalBinding.from_agent_config(
        ctx=ctx, state=state, workspace=workspace, config=config, rng=rng
    )


def run_session(
    *,
    ctx: BaseIterationContext,
    state: object,
    workspace: str,
    config: BaseAgentConfig,
    rng: random.Random,
) -> HarnessResult:
    """Run one fresh agent session to completion and record its token usage."""
    system_prompt = config.system_prompt_builder(
        additional_instructions=config.additional_instructions,
        program_path=config.program_path,
        workspace_path=workspace,
        context=ctx,
    )
    harness = create_coding_harness(
        coding_harness=config.coding_harness,
        coding_harness_model=config.coding_harness_model,
        architect_context=ctx,
        architect_config=config,
        architect_tool_registry={
            "run_eval": build_run_eval(
                ctx=ctx, state=state, workspace=workspace, config=config, rng=rng
            )
        },
    )
    request = HarnessRequest(
        role=AgentRole.ARCHITECT,
        workspace=workspace,
        prompt=build_kickoff_message(ctx),
        permissions=HarnessPermissionPolicy.autonomous_policy(),
        max_turns=config.max_turns,
        raw_provider_metadata={"system_prompt": system_prompt},
    )
    logger.info("simple loop session %d: spawning agent", ctx.iteration_idx)
    result = harness.run_architect(request)
    meta = result.raw_provider_metadata or {}
    ctx.record_token_usage(meta.get("model"), TokenUsage.from_sdk_usage(meta.get("usage")))
    logger.info("simple loop session %d: agent exited", ctx.iteration_idx)
    return result


__all__ = ["build_run_eval", "run_session"]
