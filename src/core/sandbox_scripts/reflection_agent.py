#!/usr/bin/env python3
"""Reflection agent script with structured output, executed inside Modal sandbox.

Accepts parameters via a JSON config file:
    python /app/sandbox_scripts/reflection_agent.py --config /tmp/reflection_config.json

Config JSON structure:
    {
        "workspace_path": "/workspace",
        "prompt": "...",
        "output_schema": {...},
        "max_turns": 20
    }

Outputs markers: REFLECT_STRUCTURED_OUTPUT:, REFLECT_NO_OUTPUT, REFLECT_ERROR
"""

import argparse
import json
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Import utilities from sandbox (mounted at /app/sandbox_scripts/)
sys.path.insert(0, "/app")
from sandbox_scripts.utils import timer_printer
from sandbox_scripts.environment_setup import (
    setup_sandbox_env,
    verify_claude_cli,
    create_keep_stream_hook,
    create_prompt_stream,
)
from sandbox_scripts.user_proxy import create_user_proxy


def parse_config() -> dict:
    """Parse config from JSON file specified via --config argument."""
    parser = argparse.ArgumentParser(description="Reflection agent")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        return json.load(f)


def main_sync():
    """Entry point: load config, setup environment, and run the reflection agent."""
    config = parse_config()

    workspace = config["workspace_path"]
    prompt_text = config["prompt"]
    output_schema = config["output_schema"]
    max_turns = config.get("max_turns", 20)

    # Setup environment (no venv needed for reflection - read-only)
    setup_sandbox_env(workspace, prefix="REFLECT", add_venv_to_path=False)
    verify_claude_cli("REFLECT")

    try:
        import anyio
        from claude_agent_sdk import ClaudeAgentOptions, query, ResultMessage, HookMatcher

        timer_printer("SDK imports complete")
        print(f"[REFLECT] Starting reflection...")
        print(f"[REFLECT] Workspace: {workspace}")
        print(f"[REFLECT] Prompt (first 300 chars): {prompt_text[:300]}...")
        sys.stdout.flush()

        # Create helpers from utils
        user_proxy = create_user_proxy("REFLECT", handle_questions=False)
        keep_stream_open = create_keep_stream_hook()
        prompt_stream, done_event = create_prompt_stream(prompt_text, timeout=900)

        async def run_reflection():
            structured_output = None
            timer_printer("Starting reflection query loop")

            try:
                async for message in query(
                    prompt=prompt_stream(),
                    options=ClaudeAgentOptions(
                        cwd=workspace,
                        # Read-only tools only - no edits
                        allowed_tools=["Read", "Grep", "Glob"],
                        # bypassPermissions: Auto-approve tool execution permissions
                        permission_mode="bypassPermissions",
                        max_turns=max_turns,
                        output_format={
                            "type": "json_schema",
                            "schema": output_schema,
                        },
                        # User proxy callback
                        can_use_tool=user_proxy,
                        # Required dummy hook to keep stream open for can_use_tool
                        hooks={
                            "PreToolUse": [HookMatcher(matcher=None, hooks=[keep_stream_open])]
                        },
                    ),
                ):
                    if isinstance(message, ResultMessage):
                        timer_printer("Reflection result received")
                        # Signal done immediately so prompt_stream can exit
                        done_event.set()

                        if message.is_error:
                            print(f"REFLECT_ERROR: {message.result}")
                            return None

                        # Get structured output from ResultMessage
                        if hasattr(message, "structured_output") and message.structured_output:
                            structured_output = message.structured_output
                            print("[REFLECT] Got structured output")
            except Exception as e:
                print(f"[REFLECT] Exception during query: {type(e).__name__}: {e}")
                # Don't exit - try to return any partial output
                pass
            finally:
                done_event.set()

            if structured_output:
                # Output as JSON for reliable parsing
                print("REFLECT_STRUCTURED_OUTPUT:")
                print(json.dumps(structured_output))
            else:
                print("REFLECT_NO_OUTPUT")

            return structured_output

        anyio.run(run_reflection)

    except ImportError as e:
        print(f"REFLECT_ERROR: claude-agent-sdk not installed: {e}")
        sys.exit(1)
    except Exception as e:
        import traceback
        print(f"REFLECT_ERROR: {e}")
        print(f"[REFLECT] Traceback: {traceback.format_exc()}")
        sys.exit(1)


if __name__ == "__main__":
    main_sync()
