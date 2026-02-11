#!/usr/bin/env python3
"""Agent script for code mutation, executed inside Modal sandbox.

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

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Import utilities from sandbox (mounted at /app/sandbox/, /app/agent/)
sys.path.insert(0, "/app")
from sandbox.mounted.utils import timer_printer, verify_changes_with_git
from agent.mounted.environment_setup import (
    setup_sandbox_env,
    verify_claude_cli,
    create_keep_stream_hook,
    create_prompt_stream,
)
from agent.mounted.user_proxy import create_user_proxy


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
        print(f"[AGENT] Workspace: {workspace}")
        print(f"[AGENT] Change request: {change_request[:500]}...")
        sys.stdout.flush()

        # Create helpers from utils
        user_proxy = create_user_proxy("AGENT", handle_questions=True)
        keep_stream_open = create_keep_stream_hook()
        prompt_stream, done_event = create_prompt_stream(prompt_text, timeout=900)

        async def run_agent():
            tool_uses = []
            error_occurred = False
            error_message = None
            timer_printer("Starting agent query loop")

            try:
                async for message in query(
                    prompt=prompt_stream(),
                    options=ClaudeAgentOptions(
                        cwd=workspace,
                        # Full set of code editing tools
                        allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                        # bypassPermissions: Auto-approve tool execution permissions
                        permission_mode="bypassPermissions",
                        max_turns=max_turns,
                        # User proxy callback: handles plan mode approval and questions
                        can_use_tool=user_proxy,
                        # Required dummy hook to keep stream open for can_use_tool
                        hooks={
                            "PreToolUse": [HookMatcher(matcher=None, hooks=[keep_stream_open])]
                        },
                    ),
                ):
                    # Log what Claude is doing for observability
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, ToolUseBlock):
                                tool_uses.append(block.name)
                                timer_printer(f"Tool: {block.name}")
                                print(f"[AGENT] Tool: {block.name}")
                                if block.name in ["Write", "Edit"]:
                                    file_path = block.input.get("file_path", "unknown")
                                    print(f"[AGENT]   -> {file_path}")
                            elif isinstance(block, TextBlock):
                                # Log first 200 chars of Claude's thinking
                                text_preview = block.text[:200].replace("\n", " ")
                                print(f"[AGENT] Claude: {text_preview}...")
                        sys.stdout.flush()

                    elif isinstance(message, ResultMessage):
                        timer_printer("Agent result received")
                        # Signal done immediately so prompt_stream can exit
                        done_event.set()

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
            print(f"[AGENT] Tools used: {tool_uses}")
            edit_tools = [t for t in tool_uses if t in ["Write", "Edit"]]
            print(f"[AGENT] File modifications: {len(edit_tools)}")

            if error_occurred:
                raise Exception(f"Agent error: {error_message}")

            if not edit_tools:
                print("[AGENT] WARNING: No file modifications were made!")

            return len(edit_tools) > 0

        changes_made = anyio.run(run_agent)

        # Verify changes with git
        timer_printer("Verifying git changes")
        has_changes, changed_files = verify_changes_with_git(workspace)

        if has_changes:
            print(f"[AGENT] Git shows {len(changed_files)} changed files:")
            for f in changed_files[:10]:  # Show first 10
                print(f"[AGENT]   {f}")
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
