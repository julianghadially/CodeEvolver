"""Claude Code implementation of the CodeEvolver coding harness."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from src.engine import paths

from .architect_tools import (
    ARCHITECT_TOOL_SPECS,
    architect_tool_spec,
    ordered_architect_tool_names,
)
from .base import AgentRole, HarnessRequest, HarnessResult

logger = logging.getLogger(__name__)


class ClaudeHarnessNotWiredError(RuntimeError):
    """Raised when Claude Code / Claude Agent SDK dependencies are unavailable."""


@dataclass(frozen=True)
class ClaudeArchitectSessionSpec:
    cwd: str
    initial_prompt: str
    system_prompt: str
    setting_sources: list[str]
    allowed_tools: list[str]
    permission_mode: str
    max_turns: int
    resume: str | None
    mcp_server_name: str
    user_proxy_role: str
    user_proxy_handles_questions: bool
    hook_names: list[str]
    stages_project_skills: bool
    coding_harness_model: str | None
    # Meta-reflection capture: when on, serialize the architect's own session
    # (thinking/text/tool_use/tool_result) to architect meta-traces, and enable
    # summarized thinking display so the thinking text is actually streamed.
    capture_architect_trace: bool = False
    enable_thinking_display: bool = False


class ClaudeCodeHarness:
    """Claude Code / Claude Agent SDK backed harness.

    The architect role runs in-process through ``claude_agent_sdk``. Coding and
    inspection roles preserve the existing run-to-completion subprocess scripts.
    """

    def __init__(
        self,
        *,
        architect_context: Any | None = None,
        architect_config: Any | None = None,
        architect_tool_registry: Mapping[str, Callable[..., dict]] | None = None,
        coding_runner: Callable[..., dict] | None = None,
        inspection_runner: Callable[..., dict] | None = None,
    ) -> None:
        self.architect_context = architect_context
        self.architect_config = architect_config
        self.architect_tool_registry = dict(architect_tool_registry or {})
        self._coding_runner = coding_runner
        self._inspection_runner = inspection_runner
        self._last_session_id: str | None = None

    @property
    def coding_harness_model(self) -> None:
        """Claude ignores the OpenCode-specific harness model selector."""
        return None

    def build_architect_session_spec(
        self,
        request: HarnessRequest,
        *,
        resume_session_id: str | None = None,
    ) -> ClaudeArchitectSessionSpec:
        self._require_role(request, AgentRole.ARCHITECT)
        config = self._require_architect_config()
        system_prompt = str(request.raw_provider_metadata.get("system_prompt") or "")
        stages_project_skills = bool(getattr(config, "stage_project_skills", True))
        # The meta-reflection flag does double duty: capture the architect's own
        # session AND enable summarized thinking so the thinking text streams.
        capture_architect_trace = bool(getattr(config, "capture_architect_traces", False))
        allowed_tools = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
        if stages_project_skills:
            allowed_tools.append("Skill")
        # MCP tools are derived from the registered tool set (in the canonical
        # `ordered_architect_tool_names()` order) so an optimizer that
        # registers a smaller surface automatically drops the rest.
        allowed_tools.extend(
            f"mcp__ce__{name}"
            for name in ordered_architect_tool_names()
            if name in self.architect_tool_registry
        )
        return ClaudeArchitectSessionSpec(
            cwd=str(request.workspace),
            initial_prompt=request.prompt,
            system_prompt=system_prompt,
            setting_sources=["project"],
            allowed_tools=allowed_tools,
            permission_mode="bypassPermissions",
            max_turns=request.max_turns or config.max_turns,
            resume=resume_session_id,
            mcp_server_name="ce",
            user_proxy_role="ARCHITECT",
            user_proxy_handles_questions=True,
            hook_names=["keep_stream_open", "holdout_guard"],
            stages_project_skills=stages_project_skills,
            coding_harness_model=None,
            capture_architect_trace=capture_architect_trace,
            enable_thinking_display=capture_architect_trace,
        )

    def run_architect(self, request: HarnessRequest) -> HarnessResult:
        return self._run_architect_session(request)

    def resume_architect(
        self, session_id: str, request: HarnessRequest
    ) -> HarnessResult:
        return self._run_architect_session(request, resume_session_id=session_id)

    def run_coding(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.CODING)
        metadata = dict(request.raw_provider_metadata or {})
        runner = self._coding_runner
        if runner is None:
            from src.engine.services.subprocess_service import (
                _run_claude_coding_agent_subprocess,
            )

            runner = _run_claude_coding_agent_subprocess

        runner_kwargs = {
            "change_request": request.prompt,
            "change_location": metadata.get("change_location"),
            "max_turns": request.max_turns or 50,
            "commit_changes": metadata.get("commit_changes", True),
            "program_path": metadata.get("program_path"),
            "coding_harness": "claude",
            "coding_harness_model": None,
        }
        if metadata.get("agent_model"):
            runner_kwargs["model"] = metadata["agent_model"]
        if metadata.get("deny_paths"):
            runner_kwargs["deny_paths"] = metadata["deny_paths"]
        result = runner(str(request.workspace), **runner_kwargs)
        return HarnessResult(
            success=bool(result.get("success")),
            error=result.get("error"),
            output=result.get("output") or "",
            session_id=result.get("session_id"),
            raw_provider_metadata={
                "provider": "claude",
                "role": "coding",
                "commit_handled": bool(metadata.get("commit_changes", True)),
                "usage": result.get("usage") or {},
            },
        )

    def run_inspection(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.INSPECTION)
        metadata = dict(request.raw_provider_metadata or {})
        runner = self._inspection_runner
        if runner is None:
            from src.engine.services.subprocess_service import (
                _run_claude_inspection_agent_subprocess,
            )

            runner = _run_claude_inspection_agent_subprocess

        runner_kwargs = {
            "prompt": request.prompt,
            "max_turns": request.max_turns or 20,
            "iteration_num": metadata.get("iteration_num"),
            "writable": metadata.get("writable", False),
            "deny_paths": metadata.get("deny_paths"),
            "coding_harness": "claude",
            "coding_harness_model": None,
            "run_as_agent_user": metadata.get("run_as_agent_user", True),
            "allow_bash": metadata.get("allow_bash", False),
        }
        if metadata.get("agent_model"):
            runner_kwargs["model"] = metadata["agent_model"]
        result = runner(str(request.workspace), **runner_kwargs)
        return HarnessResult(
            success=bool(result.get("success")),
            error=result.get("error"),
            output=result.get("proposed_change") or "",
            session_id=result.get("session_id"),
            raw_provider_metadata={
                "provider": "claude",
                "role": "inspection",
                "usage": result.get("usage") or {},
            },
        )

    def _run_architect_session(
        self,
        request: HarnessRequest,
        *,
        resume_session_id: str | None = None,
    ) -> HarnessResult:
        try:
            import anyio
            from claude_agent_sdk import (
                AssistantMessage,
                ClaudeAgentOptions,
                HookMatcher,
                ResultMessage,
                TextBlock,
                ThinkingBlock,
                ToolResultBlock,
                ToolUseBlock,
                UserMessage,
                query,
            )
        except ImportError as exc:
            raise ClaudeHarnessNotWiredError(
                f"claude-agent-sdk not importable: {exc}"
            ) from exc

        try:
            from src.engine.agent.mounted.environment_setup import (
                create_keep_stream_hook,
                create_prompt_stream,
            )
            from src.engine.agent.mounted.user_proxy import create_user_proxy
            from src.engine.optimizer.v3.architect_helpers import (
                _log_assistant_message,
                _log_result_message,
                _stage_skills_into_workspace,
            )
            from src.engine.optimizer.v3.architect_trace import (
                append_architect_turn,
                append_session_prompt_turn,
                serialize_assistant_blocks,
                serialize_user_blocks,
            )
            from src.engine.optimizer.v3.architect_hooks import (
                _make_holdout_guard_hook,
            )
        except ImportError as exc:
            raise ClaudeHarnessNotWiredError(
                f"architect helpers not importable: {exc}"
            ) from exc

        config = self._require_architect_config()
        context = self._require_architect_context()
        iteration_idx = context.iteration_idx
        spec = self.build_architect_session_spec(
            request,
            resume_session_id=resume_session_id,
        )

        ce_mcp_server = self._build_mcp_server(workspace=str(request.workspace))
        user_proxy_base = create_user_proxy(
            spec.user_proxy_role,
            handle_questions=spec.user_proxy_handles_questions,
        )
        keep_stream_open = create_keep_stream_hook()
        holdout_guard = _make_holdout_guard_hook(
            iteration_idx,
            testset_path=config.evaluator.testset_path,
            extra_deny_paths=getattr(config, "additional_deny_paths", None),
        )
        prompt_stream, done_event = create_prompt_stream(
            spec.initial_prompt,
            timeout=spec.max_turns * 120,
        )
        if spec.stages_project_skills:
            _stage_skills_into_workspace(
                spec.cwd,
                skills=getattr(config, "skills_to_stage", None),
                overrides=getattr(config, "skill_overrides", None),
                extra_values=getattr(config, "skill_template_values", None),
            )

        def _forward_cli_stderr(line: str) -> None:
            logger.warning("iteration %d: claude CLI stderr | %s", iteration_idx, line)

        options_kwargs: dict[str, Any] = {
            "cwd": spec.cwd,
            "system_prompt": spec.system_prompt,
            "setting_sources": spec.setting_sources,
            "mcp_servers": {spec.mcp_server_name: ce_mcp_server},
            "allowed_tools": spec.allowed_tools,
            "permission_mode": spec.permission_mode,
            "max_turns": spec.max_turns,
            "can_use_tool": user_proxy_base,
            "stderr": _forward_cli_stderr,
            "hooks": {
                "PreToolUse": [
                    HookMatcher(
                        matcher=None,
                        hooks=[keep_stream_open, holdout_guard],
                    ),
                ],
            },
        }
        if spec.resume:
            options_kwargs["resume"] = spec.resume
        if config.agent_model:
            options_kwargs["model"] = config.agent_model
        if paths.agent_user():
            # Docker runner only (env-gated): the SDK passes this straight
            # through to anyio.open_process(user=...), so the claude CLI child
            # runs as the unprivileged agent user and the kernel enforces the
            # holdout fence. Tool bodies of the in-process ce MCP server still
            # run in this (driver) process.
            options_kwargs["user"] = paths.agent_user()
        if spec.enable_thinking_display:
            # Capture the architect's reasoning for end-of-run meta-reflection.
            # `display: "summarized"` is required for thinking TEXT to stream on
            # Opus 4.7+/4.8 (otherwise blocks arrive signature-only/empty). This
            # is a SESSION option, not a model change — `config.agent_model` is
            # untouched, so CLAUDE.md's "never change the language model" rule is
            # not affected.
            options_kwargs["thinking"] = {"type": "adaptive", "display": "summarized"}

        try:
            from src.engine.services.asyncio_utils import install_pidfd_child_watcher

            active_watcher = install_pidfd_child_watcher()
            logger.info(
                "iteration %d: active child watcher before anyio.run: %s",
                iteration_idx,
                active_watcher,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "iteration %d: install_pidfd_child_watcher raised (non-fatal): %s",
                iteration_idx,
                exc,
            )

        result_payload: dict[str, Any] = {
            "output": "",
            "session_id": None,
            "raw_provider_metadata": {},
        }
        # Names-only record of every tool_use block, so the optimizer can see
        # post-session which tools (incl. the `Skill` tool) the session used.
        # OpenCode's harness emits the same key from its stdout parser.
        tool_calls: list[dict[str, Any]] = []

        async def run_session():
            # Own the SDK query generator's lifecycle explicitly. Breaking out
            # of a bare `async for` does NOT close the iterator (PEP 533 was
            # deferred — see claude_agent_sdk/_internal/client.py), so the
            # public query() generator and its inner _process_query_inner would
            # be left dangling until anyio's loop-shutdown shutdown_asyncgens()
            # finalizes them. That finalization races with the outer
            # generator's own `finally: await inner.aclose()`, surfacing as
            # "RuntimeError: aclose(): asynchronous generator is already
            # running" on stderr. Closing the generator ourselves, on this
            # task, before anyio.run() returns makes teardown deterministic.
            session_query = query(
                prompt=prompt_stream(),
                options=ClaudeAgentOptions(**options_kwargs),
            )
            try:
                async for message in session_query:
                    if isinstance(message, AssistantMessage):
                        _log_assistant_message(
                            iteration_idx,
                            message,
                            TextBlock,
                            ToolUseBlock,
                        )
                        for block in getattr(message, "content", None) or []:
                            if isinstance(block, ToolUseBlock):
                                tool_calls.append({
                                    "name": getattr(block, "name", "?"),
                                    "input": getattr(block, "input", None) or {},
                                })
                        if spec.capture_architect_trace:
                            append_architect_turn(
                                iteration_idx,
                                role="assistant",
                                blocks=serialize_assistant_blocks(
                                    message, TextBlock, ThinkingBlock, ToolUseBlock
                                ),
                            )
                        continue
                    if isinstance(message, UserMessage):
                        # Carries ToolResultBlocks (the outputs/files the
                        # architect's tools returned). Only relevant for capture.
                        if spec.capture_architect_trace:
                            append_architect_turn(
                                iteration_idx,
                                role="user",
                                blocks=serialize_user_blocks(message, ToolResultBlock),
                            )
                        continue
                    if isinstance(message, ResultMessage):
                        done_event.set()
                        sid = getattr(message, "session_id", None)
                        if sid:
                            self._last_session_id = sid
                            result_payload["session_id"] = sid
                        result_payload["output"] = getattr(message, "result", "") or ""
                        result_payload["raw_provider_metadata"] = {
                            "subtype": getattr(message, "subtype", None),
                            "stop_reason": getattr(message, "stop_reason", None),
                            "num_turns": getattr(message, "num_turns", None),
                            "duration_ms": getattr(message, "duration_ms", None),
                            # Token accounting: this session's cumulative usage and
                            # the model it ran on. The optimizer folds these into
                            # the per-iteration / run-level token ledgers.
                            "usage": dict(getattr(message, "usage", None) or {}),
                            "model": getattr(message, "model", None)
                            or options_kwargs.get("model"),
                            "total_cost_usd": getattr(message, "total_cost_usd", None),
                            "tool_calls": list(tool_calls),
                        }
                        _log_result_message(iteration_idx, message)
                        if spec.capture_architect_trace:
                            append_architect_turn(
                                iteration_idx,
                                role="result",
                                blocks=[],
                                usage=getattr(message, "usage", None),
                                meta={
                                    "subtype": getattr(message, "subtype", None),
                                    "stop_reason": getattr(message, "stop_reason", None),
                                    "num_turns": getattr(message, "num_turns", None),
                                    "duration_ms": getattr(message, "duration_ms", None),
                                    "total_cost_usd": getattr(
                                        message, "total_cost_usd", None
                                    ),
                                    "session_id": sid,
                                },
                            )
                        if message.is_error:
                            raise RuntimeError(
                                f"Architect session error (subtype={message.subtype!r} "
                                f"stop_reason={message.stop_reason!r} "
                                f"num_turns={message.num_turns} "
                                f"duration_ms={message.duration_ms}): {message.result}"
                            )
                        logger.info(
                            "iteration %d: architect session exited cleanly (%s turns)",
                            iteration_idx,
                            getattr(message, "num_turns", "?"),
                        )
                        break
            finally:
                # Deterministic, same-task close. The generator is suspended at
                # `yield msg` here (never concurrently driven by another task),
                # so this aclose() runs the SDK's ordered teardown without the
                # finalizer race. Guard it anyway: the result is already
                # captured in result_payload, so a teardown hiccup must not sink
                # an otherwise-successful iteration.
                try:
                    await session_query.aclose()
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "iteration %d: architect session generator aclose "
                        "raised during teardown (non-fatal): %s",
                        iteration_idx,
                        exc,
                    )

        if spec.capture_architect_trace:
            # The prompt reaches the SDK as a stream the capture loop never
            # sees; record it so the trace shows where each session (initial
            # + every watchdog resume) began and what it was told.
            append_session_prompt_turn(
                iteration_idx, spec.initial_prompt, resume=bool(spec.resume),
            )
        anyio.run(run_session)
        return HarnessResult.successful(
            output=result_payload["output"],
            session_id=result_payload["session_id"],
            raw_provider_metadata=result_payload["raw_provider_metadata"],
        )

    def _build_mcp_server(self, *, workspace: str | None = None) -> Any:
        from claude_agent_sdk import create_sdk_mcp_server, tool

        config = self._require_architect_config()
        context = self._require_architect_context()
        iteration_idx = context.iteration_idx
        tool_registry = self.architect_tool_registry

        def _serialize(result: Any) -> dict[str, Any]:
            try:
                text = json.dumps(result, default=str)
            except Exception:
                text = repr(result)
            return {"content": [{"type": "text", "text": text}]}

        def _error(exc: BaseException, tool_name: str) -> dict[str, Any]:
            logger.exception(
                "iteration %d: tool %s raised: %s",
                iteration_idx,
                tool_name,
                exc,
            )
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
            return {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "is_error": True,
            }

        close_iteration_mode = getattr(
            config, "close_iteration_mode", "register_branch"
        )
        def _make_tool(name: str):
            spec = architect_tool_spec(
                name,
                close_iteration_mode=close_iteration_mode,
            )

            @tool(spec.name, spec.description, spec.input_schema)
            async def _impl(args, _name=name):
                try:
                    return _serialize(tool_registry[_name](**args))
                except Exception as exc:  # noqa: BLE001
                    return _error(exc, _name)

            return _impl

        # Register only the tools the optimizer actually put in the registry,
        # in the canonical order. Engine 3 EE registers {run_eval,
        # coding_agent, inspect_traces, close_iteration}; a smaller registry
        # yields a smaller tool list.
        tools = [
            _make_tool(name)
            for name in ordered_architect_tool_names()
            if name in tool_registry
        ]

        return create_sdk_mcp_server(name="ce", tools=tools)

    def _require_architect_config(self) -> Any:
        if self.architect_config is None:
            raise ValueError("Claude architect execution requires architect_config")
        return self.architect_config

    def _require_architect_context(self) -> Any:
        if self.architect_context is None:
            raise ValueError("Claude architect execution requires architect_context")
        return self.architect_context

    @staticmethod
    def _require_role(request: HarnessRequest, expected: AgentRole) -> None:
        if request.role != expected:
            raise ValueError(
                f"{expected.value} Claude harness call received "
                f"{request.role.value!r} request"
            )


__all__ = [
    "ClaudeArchitectSessionSpec",
    "ClaudeCodeHarness",
    "ClaudeHarnessNotWiredError",
]
