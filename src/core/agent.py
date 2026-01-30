"""Claude agent execution for code mutations.

The agent runs inside the Modal sandbox where Claude's native tools
(Bash, Read, Edit, Glob, Grep) work via subprocess.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

from .system_prompt import get_code_mutation_prompt


@dataclass
class AgentResult:
    """Result from running the Claude agent."""

    success: bool
    error: str | None = None
    output: str | None = None


async def run_code_mutation_agent(
    workspace_path: str,
    change_request: str,
    change_location: str | None = None,
) -> AgentResult:
    """
    Run the Claude agent to apply a code mutation.

    This function should be called from within a Modal sandbox where
    the Claude Agent SDK is installed and has access to the filesystem.

    Args:
        workspace_path: Path to the git workspace
        change_request: Natural language description of code change
        change_location: Optional module path hint

    Returns:
        AgentResult with success status and any error/output
    """
    try:
        from claude_agent_sdk import ClaudeAgentOptions, query
    except ImportError:
        return AgentResult(
            success=False,
            error="claude-agent-sdk not installed. Install with: pip install claude-agent-sdk",
        )

    prompt = get_code_mutation_prompt(
        change_request=change_request,
        change_location=change_location,
        workspace_path=workspace_path,
    )

    try:
        output_messages = []

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace_path,
                # Full set of code editing tools (Write for new files, Edit for existing)
                allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
            ),
        ):
            if hasattr(message, "content"):
                output_messages.append(str(message.content))

        return AgentResult(
            success=True,
            output="\n".join(output_messages) if output_messages else None,
        )

    except Exception as e:
        return AgentResult(success=False, error=str(e))


def generate_agent_script(
    workspace_path: str,
    change_request: str,
    change_location: str | None = None,
    max_turns: int = 50,
) -> str:
    """
    Generate a Python script that runs the Claude agent.

    This script is written to the sandbox and executed there,
    allowing the agent to use native tools via subprocess.

    IMPORTANT: Requires Claude Code CLI to be installed in the sandbox.
    The Python SDK is just a wrapper that spawns the CLI subprocess.

    Args:
        workspace_path: Path to the workspace in the sandbox
        change_request: Natural language change description
        change_location: Optional module path hint
        max_turns: Maximum conversation turns (prevents runaway agents)

    Returns:
        Python script as a string
    """
    # Escape any quotes in the change request
    escaped_request = change_request.replace('"""', '\\"\\"\\"').replace("\\", "\\\\")
    escaped_location = (change_location or "").replace('"""', '\\"\\"\\"')

    return f'''#!/usr/bin/env python3
"""Auto-generated agent script for code mutation.

This script runs the Claude Agent SDK to apply code changes.
The SDK spawns Claude Code CLI as a subprocess.
"""

import os
import sys
import subprocess

# Load environment variables from .env if present
env_file = "{workspace_path}/.env"
if os.path.exists(env_file):
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.replace("export ", "").strip()
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value

# Add venv to PATH if it exists (for client code execution via agent's Bash tool)
venv_bin = "{workspace_path}/.venv/bin"
if os.path.exists(venv_bin):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")
    print(f"[AGENT] Added venv to PATH: {{venv_bin}}")

# Verify Claude Code CLI is installed
try:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    print(f"[AGENT] Claude Code CLI version: {{result.stdout.strip()}}")
except FileNotFoundError:
    print("AGENT_ERROR: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
    sys.exit(1)

try:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, ResultMessage, ToolUseBlock, TextBlock

    change_request = """{escaped_request}"""
    change_location = """{escaped_location}""" or None
    workspace = "{workspace_path}"

    prompt = change_request
    if change_location:
        prompt = f"Focus on {{change_location}}. " + prompt

    print(f"[AGENT] Starting code mutation...")
    print(f"[AGENT] Workspace: {{workspace}}")
    print(f"[AGENT] Change request: {{change_request[:200]}}...")

    async def main():
        tool_uses = []
        error_occurred = False
        error_message = None

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace,
                # Full set of code editing tools
                allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
                max_turns={max_turns},
            ),
        ):
            # Log what Claude is doing for observability
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, ToolUseBlock):
                        tool_uses.append(block.name)
                        print(f"[AGENT] Tool: {{block.name}}")
                        if block.name in ["Write", "Edit"]:
                            file_path = block.input.get("file_path", "unknown")
                            print(f"[AGENT]   -> {{file_path}}")
                    elif isinstance(block, TextBlock):
                        # Log first 200 chars of Claude's thinking
                        text_preview = block.text[:200].replace("\\n", " ")
                        print(f"[AGENT] Claude: {{text_preview}}...")

            elif isinstance(message, ResultMessage):
                if message.is_error:
                    error_occurred = True
                    error_message = message.result
                    print(f"[AGENT] ERROR: {{message.result}}")
                else:
                    print(f"[AGENT] Completed in {{message.num_turns}} turns")
                    if message.total_cost_usd:
                        print(f"[AGENT] Cost: ${{message.total_cost_usd:.4f}}")

        # Summary
        print(f"[AGENT] Tools used: {{tool_uses}}")
        edit_tools = [t for t in tool_uses if t in ["Write", "Edit"]]
        print(f"[AGENT] File modifications: {{len(edit_tools)}}")

        if error_occurred:
            raise Exception(f"Agent error: {{error_message}}")

        if not edit_tools:
            print("[AGENT] WARNING: No file modifications were made!")

        return len(edit_tools) > 0

    changes_made = anyio.run(main)

    # Verify changes with git
    os.chdir(workspace)
    git_status = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    changed_files = [line for line in git_status.stdout.strip().split("\\n") if line]

    if changed_files:
        print(f"[AGENT] Git shows {{len(changed_files)}} changed files:")
        for f in changed_files[:10]:  # Show first 10
            print(f"[AGENT]   {{f}}")
        print("AGENT_SUCCESS")
    elif changes_made:
        print("[AGENT] Tools reported changes but git shows none - files may have been reverted")
        print("AGENT_SUCCESS")
    else:
        print("[AGENT] WARNING: No changes detected in git status")
        print("AGENT_NO_CHANGES")

except ImportError as e:
    print(f"AGENT_ERROR: claude-agent-sdk not installed: {{e}}")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"AGENT_ERROR: {{e}}")
    print(f"[AGENT] Traceback: {{traceback.format_exc()}}")
    sys.exit(1)
'''


def parse_agent_output(stdout: str, stderr: str, returncode: int) -> AgentResult:
    """
    Parse the output from running the agent script.

    Args:
        stdout: Standard output from the script
        stderr: Standard error from the script
        returncode: Exit code from the script

    Returns:
        AgentResult with parsed status
    """
    if returncode != 0:
        error_lines = [
            line
            for line in (stdout + "\n" + stderr).split("\n")
            if "AGENT_ERROR" in line
        ]
        error_msg = error_lines[0].replace("AGENT_ERROR: ", "") if error_lines else stderr
        return AgentResult(success=False, error=error_msg, output=stdout)

    if "AGENT_SUCCESS" in stdout:
        return AgentResult(success=True, output=stdout)

    if "AGENT_NO_CHANGES" in stdout:
        # Agent ran successfully but made no changes
        # This is still technically a "success" but the caller should check output
        return AgentResult(
            success=True,
            output=stdout,
            error="Agent completed but no file changes were made",
        )

    return AgentResult(
        success=False,
        error="Agent did not complete successfully",
        output=stdout,
    )
