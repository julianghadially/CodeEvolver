#!/usr/bin/env python3
"""Agent script for code mutation, executed as a subprocess.

Accepts parameters via a JSON config file:
    python /app/agent/mounted/coding_agent.py --config /tmp/agent_config.json

Config JSON structure:
    {
        "workspace_path": "/workspace",
        "change_request": "...",
        "change_location": "src/foo.py",
        "max_turns": 50
    }

Outputs markers: AGENT_SUCCESS, AGENT_NO_CHANGES, AGENT_ERROR
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Add the engine root to sys.path. This script lives at
# `<root>/src/engine/agent/mounted/coding_agent.py`, so `parents[4]` is
# `<root>`. Mirrors `src.engine.paths.engine_root()` but inline because we
# can't import that module before sys.path is set up.
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from src.engine.sandbox.mounted.utils import timer_printer, verify_changes_with_git
from src.engine.agent.mounted.environment_setup import (
    setup_sandbox_env,
    verify_claude_cli,
    create_keep_stream_hook,
    create_deny_paths_hook,
    create_prompt_stream,
)
from src.engine.agent.mounted.user_proxy import create_user_proxy
from src.engine.agent.token_usage import TokenUsage


def parse_config() -> dict:
    """Parse config from JSON file specified via --config argument."""
    parser = argparse.ArgumentParser(description="Code mutation agent")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        return json.load(f)


def main_sync():
    """Entry point: load config, setup environment, and run the agent."""
    config = parse_config()

    workspace = config["workspace_path"]
    change_request = config["change_request"]
    change_location = config.get("change_location") or None
    max_turns = config.get("max_turns", 50)
    model = config.get("model") or None
    deny_paths = config.get("deny_paths") or []

    # Setup environment
    setup_sandbox_env(workspace, prefix="AGENT", add_venv_to_path=True)
    verify_claude_cli("AGENT")

    try:
        import anyio
        from claude_agent_sdk import (
            ClaudeAgentOptions,
            query,
            AssistantMessage,
            ResultMessage,
            ToolUseBlock,
            TextBlock,
            HookMatcher,
        )

        prompt_text = change_request
        if change_location:
            prompt_text = f"Focus on {change_location}. " + prompt_text

        timer_printer("SDK imports complete")
        print(f"[AGENT] Starting code mutation...")
        print(f"[AGENT] Change request: {change_request[:500]}...")
        sys.stdout.flush()

        # Create helpers from utils
        user_proxy = create_user_proxy("AGENT", handle_questions=True)
        keep_stream_open = create_keep_stream_hook()
        deny_paths_hook = create_deny_paths_hook(
            deny_paths,
            log_prefix="AGENT",
            reason=(
                "Access to this path is prohibited: it is the validation set or "
                "the held-out test set, which are locked for the entire run. No "
                "agent may read, open, or edit them."
            ),
        )
        prompt_stream, done_event = create_prompt_stream(prompt_text, timeout=900)

        # Mutable holder so the nested run_agent() can surface the session's
        # token usage back to main_sync (anyio.run only returns the bool).
        usage_holder: dict = {}

        async def run_agent():
            tool_uses = []
            error_occurred = False
            error_message = None
            timer_printer("Starting agent query loop")

            options_kwargs = dict(
                cwd=workspace,
                # Full set of code editing tools
                allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                # bypassPermissions: Auto-approve tool execution permissions
                permission_mode="bypassPermissions",
                max_turns=max_turns,
                # User proxy callback: handles plan mode approval and questions
                can_use_tool=user_proxy,
                # keep_stream_open keeps can_use_tool alive; deny_paths_hook
                # fences the valset + held-out test set (no-op when empty).
                hooks={
                    "PreToolUse": [
                        HookMatcher(
                            matcher=None,
                            hooks=[keep_stream_open, deny_paths_hook],
                        )
                    ]
                },
            )
            # Model selected via reflection_lm (None → SDK default).
            if model:
                options_kwargs["model"] = model

            try:
                async for message in query(
                    prompt=prompt_stream(),
                    options=ClaudeAgentOptions(**options_kwargs),
                ):
                    
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                tool_uses.append(block.name)

                    if isinstance(message, ResultMessage):
                        # Signal done immediately so prompt_stream can exit
                        done_event.set()

                        # Capture cumulative session token usage for accounting.
                        usage_holder.update(dict(getattr(message, "usage", None) or {}))

                        if message.is_error:
                            error_occurred = True
                            error_message = message.result
                            print(f"[AGENT] ERROR: {message.result}")
                        else:
                            print(f"[AGENT] Completed in {message.num_turns} turns")
                            if message.total_cost_usd:
                                print(f"[AGENT] Cost: ${message.total_cost_usd:.4f}")
            except Exception as e:
                print(f"[AGENT] Exception during query: {type(e).__name__}: {e}")
                error_occurred = True
                error_message = str(e)
            finally:
                done_event.set()

            # Summary
            edit_tools = [t for t in tool_uses if t in ["Write", "Edit"]]
            
            if error_occurred:
                raise Exception(f"Agent error: {error_message}")

            if not edit_tools:
                print("[AGENT] WARNING: No file modifications were made!")

            return len(edit_tools) > 0

        changes_made = anyio.run(run_agent)

        # Emit token usage (single tiny line) for the parent process to parse.
        usage_payload = TokenUsage.from_sdk_usage(usage_holder).to_dict()
        usage_payload["model"] = model
        print("AGENT_USAGE:" + json.dumps(usage_payload, separators=(",", ":")), flush=True)

        # Verify changes with git
        has_changes, changed_files = verify_changes_with_git(workspace)

        if has_changes:
            print("AGENT_SUCCESS")
        elif changes_made:
            print("[AGENT] Tools reported changes but git shows none - files may have been reverted")
            print("AGENT_SUCCESS")
        else:
            print("[AGENT] WARNING: No changes detected in git status")
            print("AGENT_NO_CHANGES")

    except ImportError as e:
        print(f"AGENT_ERROR: claude-agent-sdk not installed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"AGENT_ERROR: {e}")
        print(f"[AGENT] Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
