"""Copyright © 2026 440 Labs LLC

IterationArchitect — one selected coding harness session per iteration.

The architect runs in-process with cwd set to the configured client workspace.
Harness-provided tools operate directly on the client's filesystem.

Engine 3: the architect owns git — it checks out / creates its own
`codeevolver-` branches and finalizes each iteration through the
`close_iteration` tool. The loop locks the `IterationContext` at iteration
start (there is no `select_candidate`); post-session watchdogs verify the
trust-but-verify closing protocol (a close happened, work is committed, HEAD
was evaluated, regressions were deliberate, memory.md was updated).

The simple_loop baseline does NOT use this class.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Protocol

from src.engine.agent.harnesses import (
    AgentRole,
    ClaudeHarnessNotWiredError,
    CodingHarness as AgentCodingHarness,
    HarnessPermissionPolicy,
    HarnessRequest,
    create_coding_harness,
)
from src.engine.agent.harnesses.architect_tools import CloseIterationMode
from src.engine.agent.token_usage import TokenUsage
from src.engine.schemas.agent_config import BaseAgentConfig

from src.engine.services.logger import log_session_debug_context
from src.engine.services.subprocess_service import (
    git_add_and_commit,
    git_commit_trees,
    git_get_sha,
)

from .architect_helpers import SKILLS_DIR
from .architect_tool_registry import (
    ARCHITECT_TRACE_DENY_FRAGMENT,
    CONSTRAINT_FIXTURE_DENY_FRAGMENT,
    _build_holdout_deny_paths,
    build_tool_registry,
)
from .architect_watchdogs import (
    CLOSE_MISSING,
    CLOSE_SKILL,
    MEMORY_DIFF,
    NO_CHANGE,
    POST_RECONCILE_WATCHDOGS,
    PRE_COMMIT_WATCHDOGS,
    REVERT_CHECK,
    Watchdog,
    _close_skill_invoked,
    hash_file,
    memory_md_path,
    registered_head_sha,
    run_watchdog,
)
from .constraint_gate import PREVIOUS_ITERATION_REVERTED_NOTE
from .iteration_context import IterationContext

logger = logging.getLogger(__name__)


class _StateLike(Protocol):
    primary_branch: str


@dataclass(kw_only=True)
class ArchitectConfig(BaseAgentConfig):
    """Engine-3 architect session config.

    Harness- and eval-tool-facing fields live on `BaseAgentConfig`
    (`src/engine/schemas/agent_config.py`); the fields below are engine-3
    additions (branch naming, the close-iteration variant, the derived agent
    model). `system_prompt_builder` defaults to the EE builder.
    """

    run_timestamp: str
    # Which `close_iteration` tool the architect gets: "register_branch" (EE),
    # "loop_branch" (pareto_evolution — the loop forked the child branch) or
    # "head" (freeform — the loop registers whatever HEAD is at close).
    close_iteration_mode: CloseIterationMode = "register_branch"
    # Strategy seams threaded from `OptimizerStrategy` (ASA uses all three):
    # a run_eval policy consulted at the baseline, extra architect tools built
    # per iteration, and commit-before-comparison-eval.
    run_eval_gate: Callable[..., Any] | None = None
    extra_tools: Callable[..., Mapping[str, Callable[..., dict]]] | None = None
    commit_before_comparison: bool = False

    def __post_init__(self) -> None:
        if self.system_prompt_builder is None:
            from .prompts import build_architect_system_prompt
            self.system_prompt_builder = build_architect_system_prompt


class IterationArchitect:
    """Runs a single iteration end-to-end via the selected coding harness.

    Instantiated directly by the optimizer loop with a workspace path.
    The harness session runs with cwd pointing to that workspace.
    """

    def __init__(
        self,
        *,
        context: IterationContext,
        state: _StateLike,
        workspace: str,
        config: ArchitectConfig,
        rng: random.Random | None = None,
    ) -> None:
        self.context = context
        self.state = state
        self.workspace = workspace
        self.config = config
        self.rng = rng or random.Random()
        self._last_session_id: str | None = None
        # Tool calls observed across the initial session and every watchdog
        # resume (harness `raw_provider_metadata["tool_calls"]`). None until a
        # harness reports the key — harnesses that don't are fail-open for the
        # close-skill watchdog.
        self._session_tool_calls: list[dict[str, Any]] | None = None

    # ----- Public entry point -----

    def run(self) -> None:
        """Run one iteration. Blocks until the architect session exits.

        Launches a single architect harness session with:
        - The architect system prompt (built from `prompts.py`).
        - Custom tools (`run_eval`, `coding_agent`, `inspect_traces`,
          `close_iteration`) registered via the harness tool-dispatch
          surface.
        - Project skills loaded from SKILLS_DIR — `close_iteration` (invoked
          to finalize the iteration) and, on the EE / ASA arms only,
          `exploration` (macro-approach experimentation guidance).
        - Read / Grep / Glob / Bash available for ad-hoc file and git ops.

        The session is expected to block for minutes to hours. Afterwards a
        series of one-shot watchdogs enforces the closing protocol (see the
        `_run_*_watchdog_if_needed` methods).
        """
        logger.info(
            "iteration %d: IterationArchitect spawning (workspace=%s, skills_dir=%s)",
            self.context.iteration_idx,
            self.workspace,
            SKILLS_DIR,
        )
        log_session_debug_context(
            logger=logger,
            iteration_idx=self.context.iteration_idx,
            workspace=self.workspace,
        )
        system_prompt = self.config.system_prompt_builder(
            additional_instructions=self.config.additional_instructions,
            program_path=self.config.program_path,
            workspace_path=self.workspace,
            context=self.context,
        )
        role_harness = self._create_coding_harness()
        tool_registry = self.build_tool_registry(agent_harness=role_harness)
        self._last_session_id = None

        try:
            self._launch_architect_provider(system_prompt, tool_registry)
        except ClaudeHarnessNotWiredError as exc:
            logger.warning(
                "iteration %d: architect session not yet wired (%s). "
                "Recording iteration as skipped so the loop can progress.",
                self.context.iteration_idx,
                exc,
            )
            return
        except Exception:
            logger.exception("iteration %d: architect session crashed", self.context.iteration_idx)
            raise

        # Closing-protocol watchdogs, in order. Each is one-shot; each resume
        # may change git/close state, so later checks re-read live state.
        # Decisions + prompts live in `architect_watchdogs.py`.
        for watchdog in PRE_COMMIT_WATCHDOGS:
            self._run_watchdog(watchdog, system_prompt, tool_registry)
        self._commit_workstation_drift()
        # Must run after the drift commit (the last step that can turn dirty
        # content into a commit) and before the sha-keyed watchdogs below.
        self._reconcile_eval_shas()
        # No untested-HEAD watchdog: the close record
        # never claims the closing commit was measured, so re-prompting for a
        # re-eval only cost budget. The evolution arm gets a targeted warning
        # instead, because there the loop WILL re-measure by rule.
        for watchdog in POST_RECONCILE_WATCHDOGS:
            self._run_watchdog(watchdog, system_prompt, tool_registry)
        logger.info("iteration %d: IterationArchitect completed", self.context.iteration_idx)

    # ----- Watchdogs (see architect_watchdogs.py) -----

    def _run_watchdog(
        self,
        watchdog: Watchdog,
        system_prompt: str,
        tool_registry: dict[str, Callable[..., dict]],
    ) -> None:
        run_watchdog(
            self, watchdog, system_prompt, tool_registry, self._launch_architect_provider
        )

    def _run_no_change_watchdog_if_needed(self, system_prompt, tool_registry) -> None:
        self._run_watchdog(NO_CHANGE, system_prompt, tool_registry)

    def _run_close_missing_watchdog_if_needed(self, system_prompt, tool_registry) -> None:
        self._run_watchdog(CLOSE_MISSING, system_prompt, tool_registry)

    def _run_close_skill_watchdog_if_needed(self, system_prompt, tool_registry) -> None:
        self._run_watchdog(CLOSE_SKILL, system_prompt, tool_registry)

    def _run_revert_check_if_needed(self, system_prompt, tool_registry) -> None:
        self._run_watchdog(REVERT_CHECK, system_prompt, tool_registry)

    def _run_memory_diff_watchdog_if_needed(self, system_prompt, tool_registry) -> None:
        self._run_watchdog(MEMORY_DIFF, system_prompt, tool_registry)

    # ----- Deterministic close-time git steps -----

    def _commit_workstation_drift(self) -> None:
        """Workstation-committed check: commit any uncommitted work at close.

        No resume — this is the deterministic safety net behind the
        close_iteration tool's dirty-tree rejection. Multi-repo note: only
        the workspace repo is committed today; other repos registered in
        register_branch are the agent's responsibility.
        """
        iteration_idx = self.context.iteration_idx
        try:
            committed = git_add_and_commit(
                self.workspace,
                f"iter {iteration_idx}: uncommitted work at close (auto-committed)",
            )
            if committed:
                drift_sha = git_get_sha(self.workspace, "HEAD")
                logger.info(
                    "iteration %d: committed close-time drift as %s",
                    iteration_idx, drift_sha[:8],
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "iteration %d: failed to commit close-time drift "
                "(non-fatal): %s",
                iteration_idx, exc,
            )

    def _reconcile_eval_shas(self) -> None:
        """Promote dirty-tree eval records onto the commits holding their code.

        `run_eval` stamps HEAD's sha, but the evaluator runs the working tree,
        so an eval of uncommitted edits carries the PARENT's sha. Left alone,
        `finalize()` would attribute the score to the parent's content,
        and record the iteration as skipped despite real work landing.
        """
        ref = self.context.child_branch or "HEAD"
        try:
            n = self.context.reconcile_eval_shas(
                git_commit_trees(
                    self.workspace, ref, since=self.context.start_sha
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "iteration %d: eval-sha reconciliation failed (non-fatal): %s",
                self.context.iteration_idx, exc,
            )
            return
        if n:
            logger.info(
                "iteration %d: re-pointed %d eval record(s) onto their "
                "committed sha via tree match",
                self.context.iteration_idx, n,
            )

    def _registered_head_sha(self) -> str | None:
        """HEAD of the branch the architect registered (falls back to HEAD)."""
        return registered_head_sha(self)

    # ----- Session launching -----

    def _launch_architect_provider(
        self,
        system_prompt: str,
        tool_registry: dict[str, Callable[..., dict]],
        *,
        resume_session_id: str | None = None,
        initial_prompt_override: str | None = None,
    ) -> None:
        initial_prompt = self._build_initial_prompt(initial_prompt_override)
        harness = self._create_coding_harness(architect_tool_registry=tool_registry)
        request = HarnessRequest(
            role=AgentRole.ARCHITECT,
            workspace=self.workspace,
            prompt=initial_prompt,
            permissions=HarnessPermissionPolicy.autonomous_policy(),
            max_turns=self.config.max_turns,
            raw_provider_metadata={"system_prompt": system_prompt},
        )
        if resume_session_id:
            result = harness.resume_architect(resume_session_id, request)
        else:
            result = harness.run_architect(request)
        if result.session_id:
            self._last_session_id = result.session_id
        calls = (result.raw_provider_metadata or {}).get("tool_calls")
        if isinstance(calls, list):
            if self._session_tool_calls is None:
                self._session_tool_calls = []
            self._session_tool_calls.extend(c for c in calls if isinstance(c, dict))

        # Record usage
        # When resuming after watchdog, sessions accumulate into the same model bucket.
        meta = result.raw_provider_metadata or {}
        self.context.record_token_usage(
            meta.get("model"),
            TokenUsage.from_sdk_usage(meta.get("usage")),
        )

    def _create_coding_harness(
        self,
        *,
        architect_tool_registry: dict[str, Callable[..., dict]] | None = None,
    ) -> AgentCodingHarness:
        return create_coding_harness(
            coding_harness=self.config.coding_harness,
            coding_harness_model=self.config.coding_harness_model,
            architect_context=self.context if architect_tool_registry is not None else None,
            architect_config=self.config if architect_tool_registry is not None else None,
            architect_tool_registry=architect_tool_registry,
        )

    def _build_initial_prompt(
        self,
        initial_prompt_override: str | None = None,
    ) -> str:
        if initial_prompt_override:
            return initial_prompt_override

        # `must_explore` is set by the loop's explore-frequency guard, which
        # can only fire after >= 4 CLOSED (non-skipped) iterations with a
        # claimed exploration share below quota — early iterations (incl.
        # iteration 1) never carry this note; the agent is free to explore
        # or exploit as it chooses.
        must_explore_note = (
            "This iteration MUST be an exploration iteration (the run's "
            "exploration share has fallen below quota) — see the "
            "`exploration` skill. "
            if self.context.must_explore
            else ""
        )
        # ASA rule notes (specs/archive-public/v3_asa.md §Rule 1 / §Rule 2). Empty for every
        # other arm — only ASA's iteration-start hook sets `asa_requirements`.
        asa_notes = ""
        if getattr(self.context, "asa_requirements", None):
            from .asa.prompt_sections import user_message_notes

            asa_notes = user_message_notes(self.context)
        # Constraint-gate revert note (every arm, same wording): the previous
        # iteration was reverted for violating the constraints — not which
        # constraint, and never the judge's reasoning.
        reverted_note = (
            PREVIOUS_ITERATION_REVERTED_NOTE + " "
            if self.context.previous_iteration_constraint_reverted
            else ""
        )
        return (
            f"Improve the program, iteratively. Use the tools at your disposal, including evaluations (via `run_eval`), traces, skills, etc. "
            + must_explore_note
            + asa_notes
            + reverted_note
        )

    def build_tool_registry(
        self,
        *,
        agent_harness: AgentCodingHarness,
    ) -> dict[str, Callable[..., dict]]:
        """Build the tool dispatch table the SDK will register.

        Each entry is a callable whose signature matches the tool schema the
        language model sees one-to-one (see `architect_tool_registry`).
        """
        return build_tool_registry(
            context=self.context,
            state=self.state,
            workspace=self.workspace,
            cfg=self.config,
            rng=self.rng,
            agent_harness=agent_harness,
        )


__all__ = [
    "ARCHITECT_TRACE_DENY_FRAGMENT",
    "CONSTRAINT_FIXTURE_DENY_FRAGMENT",
    "ArchitectConfig",
    "IterationArchitect",
    "SKILLS_DIR",
    "_build_holdout_deny_paths",
    "_close_skill_invoked",
    "hash_file",
    "memory_md_path",
]
