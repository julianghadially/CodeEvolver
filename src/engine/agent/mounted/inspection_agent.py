#!/usr/bin/env python3
"""Inspection sub-agent script, executed as a subprocess.

A restricted Claude Agent SDK session used by the EE optimizer for:
- Architect-facing trace inspection (read-only)
- Valset reflection / memory curation (writable with deny-path fence)

Tool surface: Read/Grep/Glob, plus Write/Edit when ``writable=True``. No Bash
and no coding tools — that's `coding_agent.py`. Not to be confused with the
v1 "reflection agent" that proposed change requests; that role is gone, the
IterationArchitect owns change requests now.

Accepts parameters via a JSON config file:
    python /app/src/engine/agent/mounted/inspection_agent.py --config /tmp/nested_inspection_config.json

Config JSON structure:
    {
        "workspace_path": "/workspace",
        "prompt": "...",
        "max_turns": 20,
        "iteration_num": 3,
        "writable": false,
        "deny_paths": ["..."],
        "model": "claude-opus-4-7"
    }

Outputs markers:
    - INSPECT_ANALYSIS:           single-line JSON with the sub-agent's final
                                  `message.result` text (used by read-only
                                  `inspect_traces`)
    - INSPECT_WRITE_SUCCESS:      writable sub-agent wrote at least one file
    - INSPECT_NO_OUTPUT:          session ended without producing a recognized
                                  output for the current mode
    - INSPECT_ERROR:              session ended in an error state
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Add the engine root to sys.path. This script lives at
# `<root>/src/engine/agent/mounted/inspection_agent.py`, so `parents[4]` is
# `<root>`. Mirrors `src.engine.paths.engine_root()` but inline because we
# can't import that module before sys.path is set up.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.engine.sandbox.mounted.utils import timer_printer
from src.engine.agent.mounted.environment_setup import (
    setup_sandbox_env,
    verify_claude_cli,
    create_keep_stream_hook,
    create_prompt_stream,
)
from src.engine.agent.mounted.user_proxy import create_user_proxy
from src.engine.agent.mounted.runtime_tests import (
    print_tool_call_diagnostics, check_trace_inspection, check_scores_contain_feedback,
)
from src.engine.agent.token_usage import TokenUsage


def parse_config() -> dict:
    """Parse config from JSON file specified via --config argument."""
    parser = argparse.ArgumentParser(description="Inspection agent")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        return json.load(f)


def _make_deny_paths_hook(deny_paths: list[str]):
    """PreToolUse hook that denies tool calls referencing any fragment in ``deny_paths``.

    Substring match on both absolute and relative path fragments, and on
    bash commands. Returns a no-op hook when ``deny_paths`` is empty so the
    caller can wire it unconditionally.
    """
    fragments = [p for p in deny_paths if p]

    if not fragments:
        async def _noop(_input, _tool_use_id, _context):
            return {}
        return _noop

    def _mentions_any(*candidates):
        for candidate in candidates:
            if not candidate:
                continue
            text = str(candidate)
            for fragment in fragments:
                if fragment in text:
                    return fragment
        return None

    async def deny_paths_hook(input_data, _tool_use_id, _context):
        tool_name = (
            input_data.get("tool_name", "")
            if isinstance(input_data, dict)
            else getattr(input_data, "tool_name", "")
        )
        tool_input = (
            input_data.get("tool_input", {})
            if isinstance(input_data, dict)
            else getattr(input_data, "tool_input", {})
        )
        tool_input = tool_input or {}

        matched = None
        if tool_name in ("Read", "Edit", "Write"):
            matched = _mentions_any(tool_input.get("file_path", ""))
        elif tool_name == "Grep":
            matched = _mentions_any(
                tool_input.get("path", ""), tool_input.get("pattern", "")
            )
        elif tool_name == "Glob":
            matched = _mentions_any(
                tool_input.get("path", ""), tool_input.get("pattern", "")
            )
        elif tool_name == "Bash":
            matched = _mentions_any(tool_input.get("command", ""))

        if matched is None:
            return {}

        reason = (
            f"Access to {matched!r} is prohibited for this inspection agent. "
            "This path contains validation or holdout data."
        )
        print(
            f"[INSPECT] Denied {tool_name} call referencing deny_paths fragment "
            f"{matched!r}. tool_input={tool_input!r}",
            flush=True,
        )
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": reason,
            }
        }

    return deny_paths_hook


def main_sync():
    """Entry point: load config, setup environment, and run the inspection agent."""
    config = parse_config()

    workspace = config["workspace_path"]
    prompt_text = config["prompt"]
    max_turns = config.get("max_turns", 20)
    iteration_num = config.get("iteration_num")
    writable = config.get("writable", False)
    allow_bash = config.get("allow_bash", False)
    deny_paths = config.get("deny_paths") or []
    model = config.get("model") or None

    # Setup environment (no venv needed for inspection - read-only or memory-only)
    setup_sandbox_env(workspace, prefix="INSPECT", add_venv_to_path=False)
    verify_claude_cli("INSPECT")

    try:
        import anyio
        from claude_agent_sdk import (
            ClaudeAgentOptions, query, ResultMessage, AssistantMessage,
            ToolUseBlock, TextBlock, HookMatcher,
        )

        print(
            f"[INSPECT] Starting inspection... "
            f"(writable={writable}, allow_bash={allow_bash})"
        )
        sys.stdout.flush()

        # Create helpers from utils
        user_proxy = create_user_proxy("INSPECT", handle_questions=False)
        keep_stream_open = create_keep_stream_hook()
        deny_paths_hook = _make_deny_paths_hook(deny_paths)
        prompt_stream, done_event = create_prompt_stream(prompt_text, timeout=900)

        # Mutable holder so the nested run_inspection() can surface the session's
        # token usage back to main_sync for the parent process to parse.
        usage_holder: dict = {}

        # Configure tools based on writable mode
        if writable:
            allowed_tools = ["Read", "Grep", "Glob", "Write", "Edit"]
        else:
            allowed_tools = ["Read", "Grep", "Glob"]
        # Bash is the clean-slate inspector's exception: `git log` / `branch -a`
        # / `reflog` / `fsck` have no Read/Grep/Glob equivalent, and a verdict
        # from an agent that cannot read git history is not worth acting on.
        # `deny_paths_hook` already matches Bash command text, and the caller
        # proves the session was read-only (clean_slate.py::_snapshot_repo).
        if allow_bash:
            allowed_tools = [*allowed_tools, "Bash"]

        async def run_inspection():
            result_text = None  # Final synthesis from message.result
            tool_calls = []  # Track all tool calls for diagnostics
            session_query = None  # SDK generator; aclosed deterministically below

            try:
                options_kwargs = {
                    "cwd": workspace,
                    "allowed_tools": allowed_tools,
                    "permission_mode": "bypassPermissions",
                    "max_turns": max_turns,
                    "can_use_tool": user_proxy,
                    "hooks": {
                        "PreToolUse": [
                            HookMatcher(
                                matcher=None,
                                hooks=[keep_stream_open, deny_paths_hook],
                            )
                        ]
                    },
                }
                # Model selected via reflection_lm (None → SDK default).
                if model:
                    options_kwargs["model"] = model

                # Bind the generator so we can aclose() it deterministically in
                # `finally`. The `is_error` path below `return`s out of this
                # loop, which (PEP 533 deferral) would otherwise leave the SDK
                # query() generator dangling for the loop-shutdown finalizer to
                # race against its own inner aclose() — the same
                # "aclose(): asynchronous generator is already running" the
                # architect session hit. See claude.py run_session().
                session_query = query(
                    prompt=prompt_stream(),
                    options=ClaudeAgentOptions(**options_kwargs),
                )
                async for message in session_query:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                print(f"[INSPECT:reasoning] {block.text[:500]}", flush=True)
                            elif isinstance(block, ToolUseBlock):
                                tool_input_str = json.dumps(block.input)[:300]
                                print(f"[INSPECT:tool] {block.name}({tool_input_str})", flush=True)
                                tool_calls.append({
                                    "tool": block.name,
                                    "input": block.input,
                                })

                    if isinstance(message, ResultMessage):
                        # Signal done immediately so prompt_stream can exit
                        done_event.set()

                        # Capture cumulative session token usage for accounting.
                        usage_holder.update(dict(getattr(message, "usage", None) or {}))

                        if message.is_error:
                            print(
                                f"INSPECT_ERROR: is_error=True "
                                f"result={message.result!r} "
                                f"stop_reason={getattr(message, 'stop_reason', None)!r} "
                                f"num_turns={getattr(message, 'num_turns', None)} "
                                f"duration_ms={getattr(message, 'duration_ms', None)} "
                                f"tool_calls={len(tool_calls)}"
                            )
                            print_tool_call_diagnostics(tool_calls)
                            return None

                        # Capture final synthesis. message.result only exists
                        # on successful terminations; tolerate absence for
                        # forward-compat.
                        candidate = getattr(message, "result", None)
                        if isinstance(candidate, str) and candidate.strip():
                            result_text = candidate
            except Exception as e:
                print(f"[INSPECT] Exception during query: {type(e).__name__}: {e}")
                # Don't exit - try to return any partial output
                pass
            finally:
                done_event.set()
                if session_query is not None:
                    try:
                        await session_query.aclose()
                    except Exception as exc:  # noqa: BLE001
                        print(
                            f"[INSPECT] session generator aclose raised during "
                            f"teardown (non-fatal): {type(exc).__name__}: {exc}"
                        )

            # --- Diagnostics: tool call summary and trajectory file usage ---
            print_tool_call_diagnostics(tool_calls)
            if iteration_num is not None:
                check_trace_inspection(tool_calls, iteration_num)
                check_scores_contain_feedback(iteration_num)

            if writable:
                # In writable mode, the agent writes directly to files.
                # Check if any Write/Edit tools were used as a success indicator.
                write_tools_used = [t for t in tool_calls if t["tool"] in ("Write", "Edit")]
                if write_tools_used:
                    print("INSPECT_WRITE_SUCCESS")
                else:
                    print("INSPECT_NO_OUTPUT")
                return None
            elif result_text:
                # Read-only `inspect_traces` path: return the sub-agent's
                # final synthesis as a single-line JSON payload. Using
                # separators=(",", ":") and the single-line form keeps the
                # marker safely scannable by `run_inspection_agent` even when
                # `result_text` contains embedded newlines (they get escaped
                # as `\n` inside the JSON string).
                print(
                    "INSPECT_ANALYSIS:"
                    + json.dumps({"text": result_text}, separators=(",", ":")),
                    flush=True,
                )
                return None
            else:
                print("INSPECT_NO_OUTPUT")
                return None

        anyio.run(run_inspection)

        # Emit token usage (single tiny line) for the parent process to parse.
        usage_payload = TokenUsage.from_sdk_usage(usage_holder).to_dict()
        usage_payload["model"] = model
        print("INSPECT_USAGE:" + json.dumps(usage_payload, separators=(",", ":")), flush=True)

    except ImportError as e:
        print(f"INSPECT_ERROR: claude-agent-sdk not installed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"INSPECT_ERROR: {e}")
        print(f"[INSPECT] Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
