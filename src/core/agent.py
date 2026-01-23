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
                allowed_tools=["Bash", "Read", "Edit", "Glob", "Grep"],
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
) -> str:
    """
    Generate a Python script that runs the Claude agent.

    This script is written to the sandbox and executed there,
    allowing the agent to use native tools via subprocess.

    Args:
        workspace_path: Path to the workspace in the sandbox
        change_request: Natural language change description
        change_location: Optional module path hint

    Returns:
        Python script as a string
    """
    # Escape any quotes in the change request
    escaped_request = change_request.replace('"""', '\\"\\"\\"').replace("\\", "\\\\")
    escaped_location = (change_location or "").replace('"""', '\\"\\"\\"')

    return f'''#!/usr/bin/env python3
"""Auto-generated agent script for code mutation."""

import os
import sys

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

try:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query

    change_request = """{escaped_request}"""
    change_location = """{escaped_location}""" or None
    workspace = "{workspace_path}"

    prompt = change_request
    if change_location:
        prompt = f"Focus on {{change_location}}. " + prompt

    async def main():
        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace,
                allowed_tools=["Bash", "Read", "Edit", "Glob", "Grep"],
                permission_mode="acceptEdits",
            ),
        ):
            pass

    anyio.run(main)
    print("AGENT_SUCCESS")

except ImportError as e:
    print(f"AGENT_ERROR: claude-agent-sdk not installed: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"AGENT_ERROR: {{e}}")
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

    return AgentResult(
        success=False,
        error="Agent did not complete successfully",
        output=stdout,
    )
