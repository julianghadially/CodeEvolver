"""Claude agent execution for code mutations and reflection.

The agent runs inside the Modal sandbox where Claude's native tools
(Bash, Read, Edit, Glob, Grep) work via subprocess.
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
    # Escape for embedding in triple-quoted string
    # Order matters: escape backslashes first, then triple quotes
    escaped_request = change_request.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    escaped_location = (change_location or "").replace("\\", "\\\\").replace('"""', '\\"\\"\\"')

    return f'''#!/usr/bin/env python3
"""Auto-generated agent script for code mutation.

This script runs the Claude Agent SDK to apply code changes.
The SDK spawns Claude Code CLI as a subprocess.
"""

import os
import sys
import subprocess
import warnings

# Suppress asyncio warnings about unawaited tasks
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Load environment variables from .env if present
env_file = "{workspace_path}/.env"
print(f"[AGENT] Looking for .env at: {{env_file}}")
if os.path.exists(env_file):
    print("[AGENT] Found .env file, loading...")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.replace("export ", "").strip()
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value
                if key == "ANTHROPIC_API_KEY":
                    print(f"[AGENT] Loaded ANTHROPIC_API_KEY ({{len(value)}} chars)")
else:
    print("[AGENT] WARNING: No .env file found!")

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("AGENT_ERROR: ANTHROPIC_API_KEY not set in environment")
    sys.exit(1)
else:
    print(f"[AGENT] ANTHROPIC_API_KEY is set ({{len(os.environ['ANTHROPIC_API_KEY'])}} chars)")

# Add venv to PATH if it exists (for client code execution via agent's Bash tool)
venv_bin = "{workspace_path}/.venv/bin"
if os.path.exists(venv_bin):
    os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")
    print(f"[AGENT] Added venv to PATH: {{venv_bin}}")

# Verify Claude Code CLI is installed
try:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    print(f"[AGENT] Claude Code CLI version: {{result.stdout.strip()}}")
    if result.returncode != 0:
        print(f"[AGENT] CLI stderr: {{result.stderr}}")
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
    print(f"[AGENT] Change request: {{change_request[:500]}}...")
    sys.stdout.flush()

    async def main():
        tool_uses = []
        error_occurred = False
        error_message = None

        try:
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=workspace,
                    # Full set of code editing tools
                    allowed_tools=["Bash", "Read", "Write", "Edit", "Glob", "Grep"],
                    # bypassPermissions: Runs without ANY prompts - for autonomous execution
                    # This prevents the agent from getting stuck on plan mode approval
                    permission_mode="bypassPermissions",
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
                    sys.stdout.flush()

                elif isinstance(message, ResultMessage):
                    if message.is_error:
                        error_occurred = True
                        error_message = message.result
                        print(f"[AGENT] ERROR: {{message.result}}")
                    else:
                        print(f"[AGENT] Completed in {{message.num_turns}} turns")
                        if message.total_cost_usd:
                            print(f"[AGENT] Cost: ${{message.total_cost_usd:.4f}}")
        except Exception as e:
            print(f"[AGENT] Exception during query: {{type(e).__name__}}: {{e}}")
            error_occurred = True
            error_message = str(e)

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


@dataclass
class ReflectionResult:
    """Result from running the reflection agent."""

    success: bool
    output: str | None = None
    error: str | None = None


def generate_reflection_agent_script(
    workspace_path: str,
    prompt: str,
    output_schema: dict[str, Any],
    max_turns: int = 20,
) -> str:
    """Generate a Python script that runs the reflection agent with structured output.

    The reflection agent uses read-only tools (Read, Grep, Glob) to analyze
    the codebase and returns structured JSON output.

    Args:
        workspace_path: Path to the workspace in the sandbox.
        prompt: Reflection prompt.
        output_schema: JSON schema for structured output (from Pydantic model_json_schema()).
        max_turns: Maximum conversation turns.

    Returns:
        Python script as a string.
    """
    # Escape for embedding in triple-quoted string
    # Order matters: escape backslashes first, then triple quotes
    escaped_prompt = prompt.replace("\\", "\\\\").replace('"""', '\\"\\"\\"')
    schema_json = json.dumps(output_schema)

    return f'''#!/usr/bin/env python3
"""Auto-generated reflection agent script with structured output.

Analyzes codebase with read-only tools and returns validated JSON.
"""

import os
import sys
import subprocess
import json
import warnings

# Suppress asyncio warnings about unawaited tasks
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*was never awaited.*")

# Load environment variables
env_file = "{workspace_path}/.env"
print(f"[REFLECT] Looking for .env at: {{env_file}}")
if os.path.exists(env_file):
    print("[REFLECT] Found .env file, loading...")
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.replace("export ", "").strip()
                value = value.strip().strip("'").strip('"')
                os.environ[key] = value
                if key == "ANTHROPIC_API_KEY":
                    print(f"[REFLECT] Loaded ANTHROPIC_API_KEY ({{len(value)}} chars)")
else:
    print("[REFLECT] WARNING: No .env file found!")

# Check for API key
if not os.environ.get("ANTHROPIC_API_KEY"):
    print("REFLECT_ERROR: ANTHROPIC_API_KEY not set in environment")
    sys.exit(1)
else:
    print(f"[REFLECT] ANTHROPIC_API_KEY is set ({{len(os.environ['ANTHROPIC_API_KEY'])}} chars)")

# Verify Claude Code CLI is installed
try:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
    print(f"[REFLECT] Claude Code CLI version: {{result.stdout.strip()}}")
except FileNotFoundError:
    print("REFLECT_ERROR: Claude Code CLI not found")
    sys.exit(1)

try:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query, ResultMessage

    prompt = """{escaped_prompt}"""
    workspace = "{workspace_path}"
    output_schema = {schema_json}

    print(f"[REFLECT] Starting reflection...")
    print(f"[REFLECT] Workspace: {{workspace}}")
    print(f"[REFLECT] Prompt (first 300 chars): {{prompt[:300]}}...")
    sys.stdout.flush()

    async def main():
        structured_output = None

        try:
            async for message in query(
                prompt=prompt,
                options=ClaudeAgentOptions(
                    cwd=workspace,
                    # Read-only tools only - no edits
                    allowed_tools=["Read", "Grep", "Glob"],
                    # bypassPermissions: Runs without ANY prompts - for autonomous execution
                    permission_mode="bypassPermissions",
                    max_turns={max_turns},
                    output_format={{
                        "type": "json_schema",
                        "schema": output_schema,
                    }},
                ),
            ):
                if isinstance(message, ResultMessage):
                    if message.is_error:
                        print(f"REFLECT_ERROR: {{message.result}}")
                        return None

                    # Get structured output from ResultMessage
                    if hasattr(message, "structured_output") and message.structured_output:
                        structured_output = message.structured_output
                        print(f"[REFLECT] Got structured output")
        except Exception as e:
            print(f"[REFLECT] Exception during query: {{type(e).__name__}}: {{e}}")
            # Don't exit - try to return any partial output
            pass

        if structured_output:
            # Output as JSON for reliable parsing
            print("REFLECT_STRUCTURED_OUTPUT:")
            print(json.dumps(structured_output))
        else:
            print("REFLECT_NO_OUTPUT")

        return structured_output

    anyio.run(main)

except ImportError as e:
    print(f"REFLECT_ERROR: claude-agent-sdk not installed: {{e}}")
    sys.exit(1)
except Exception as e:
    import traceback
    print(f"REFLECT_ERROR: {{e}}")
    print(f"[REFLECT] Traceback: {{traceback.format_exc()}}")
    sys.exit(1)
'''


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
