"""Claude agent deployment and output parsing for code mutations and reflection.

Contains config builders and output parsers for the agent scripts that run
inside the Modal sandbox (agent/mounted/coding_agent.py, agent/mounted/reflection_agent.py).
"""

import json
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


def build_agent_config(
    workspace_path: str,
    change_request: str,
    change_location: str | None = None,
    max_turns: int = 50,
    program_path: str | None = None,
) -> dict[str, Any]:
    """Build a config dict for the sandbox agent script.

    The config is written as JSON to the sandbox and passed to
    agent/mounted/coding_agent.py via --config.

    When program_path is provided, the change_request is wrapped with the
    full system prompt from get_code_mutation_prompt() (including constraints,
    instructions about codeevolver.md, etc.).

    Args:
        workspace_path: Path to the workspace in the sandbox.
        change_request: Natural language change description.
        change_location: Optional module path hint.
        max_turns: Maximum conversation turns (prevents runaway agents).
        program_path: Dotted import path to parent module class.

    Returns:
        Config dict for JSON serialization.
    """
    # When program_path is provided, wrap with full system prompt
    if program_path:
        full_prompt = get_code_mutation_prompt(
            change_request=change_request,
            change_location=change_location,
            workspace_path=workspace_path,
            program_path=program_path,
        )
        config: dict[str, Any] = {
            "workspace_path": workspace_path,
            "change_request": full_prompt,
            "max_turns": max_turns,
        }
    else:
        config = {
            "workspace_path": workspace_path,
            "change_request": change_request,
            "max_turns": max_turns,
        }
        if change_location:
            config["change_location"] = change_location
    return config


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


@dataclass
class ReflectionResult:
    """Result from running the reflection agent."""

    success: bool
    output: str | None = None
    error: str | None = None


def build_reflection_config(
    workspace_path: str,
    prompt: str,
    output_schema: dict[str, Any],
    max_turns: int = 20,
) -> dict[str, Any]:
    """Build a config dict for the sandbox reflection agent script.

    The config is written as JSON to the sandbox and passed to
    agent/mounted/reflection_agent.py via --config.

    Args:
        workspace_path: Path to the workspace in the sandbox.
        prompt: Reflection prompt.
        output_schema: JSON schema for structured output (from Pydantic model_json_schema()).
        max_turns: Maximum conversation turns.

    Returns:
        Config dict for JSON serialization.
    """
    return {
        "workspace_path": workspace_path,
        "prompt": prompt,
        "output_schema": output_schema,
        "max_turns": max_turns,
    }


def parse_reflection_output(
    stdout: str,
    stderr: str,
    returncode: int,
    output_key: str,
) -> ReflectionResult:
    """Parse reflection agent output with structured JSON.

    Args:
        stdout: Standard output from the script.
        stderr: Standard error from the script.
        returncode: Exit code from the script.
        output_key: Key to extract from structured output (e.g., "architecture", "change_request").

    Returns:
        ReflectionResult with parsed output.
    """
    if returncode != 0:
        error_lines = [
            line for line in stdout.split("\n")
            if "REFLECT_ERROR" in line
        ]
        error_msg = (
            error_lines[0].replace("REFLECT_ERROR: ", "")
            if error_lines else stderr[:500]
        )
        return ReflectionResult(success=False, error=error_msg)

    # Parse structured output
    for line in stdout.split("\n"):
        if line.startswith("REFLECT_STRUCTURED_OUTPUT:"):
            # Next line contains the JSON
            continue
        if line.strip().startswith("{"):
            try:
                data = json.loads(line.strip())
                output_value = data.get(output_key, "")
                if output_value:
                    return ReflectionResult(success=True, output=output_value)
            except json.JSONDecodeError:
                pass

    if "REFLECT_NO_OUTPUT" in stdout:
        return ReflectionResult(success=True, output=None, error="No output produced")

    return ReflectionResult(success=False, error="Failed to parse reflection output")
