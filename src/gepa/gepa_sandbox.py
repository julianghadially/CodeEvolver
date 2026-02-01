"""GEPA-specific client sandbox implementation.

Extends ClientSandbox with exec_prebuilt() for DSPy command execution
and exec_agent() for coding agent code mutations.
"""

import base64
import json
import logging

from ..core.agent import generate_agent_script, parse_agent_output
from ..core.client_sandbox import ClientSandbox, _get_agent_capable_sandbox_image
from ..services.git_sandbox import SandboxGitService

logger = logging.getLogger(__name__)


class GEPASandbox(ClientSandbox):
    """Modal Sandbox for GEPA DSPy program evaluation and code mutations.

    Inherits generic sandbox lifecycle from ClientSandbox and implements:
    - exec_prebuilt(): Execute DSPy-specific commands via master.py
    - exec_agent(): Execute coding agent for code mutations

    When started with use_venv=True:
    - System Python has claude-agent-sdk, anyio (for agent execution)
    - Venv at /workspace/.venv has client deps (DSPy, etc.)
    - Agent's Bash tool uses venv PATH for client code execution

    Usage:
        sandbox = GEPASandbox(app, repo_url, ...)
        sandbox.start(use_venv=True)
        try:
            result = sandbox.exec_agent("Add docstring to main function")
            result = sandbox.exec_prebuilt({"command": "evaluate", ...})
        finally:
            sandbox.stop()
    """

    # Use agent-capable image with Node.js, npm, Claude Code CLI, and claude-agent-sdk
    _image_builder = staticmethod(_get_agent_capable_sandbox_image)

    def exec_agent(
        self,
        change_request: str,
        change_location: str | None = None,
        max_turns: int = 50,
        commit_changes: bool = True,
    ) -> dict:
        """Execute coding agent for code mutations.

        Uses system Python to run Claude Agent SDK.
        Agent's Bash tool uses venv PATH for client code execution.

        Args:
            change_request: Natural language description of code change.
            change_location: Optional module path hint (e.g., "src/core/agent.py").
            max_turns: Maximum conversation turns (prevents runaway agents).
            commit_changes: If True, commit changes after successful mutation.

        Returns:
            Dict with 'success', 'error', 'output' keys.

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        logger.info(f"Executing agent for: {change_request[:100]}...")

        # Generate agent script
        script = generate_agent_script(
            workspace_path=self._workspace,
            change_request=change_request,
            change_location=change_location,
            max_turns=max_turns,
        )

        # Write script to sandbox
        self._sandbox.exec(
            "bash", "-c",
            f"cat > /tmp/agent_script.py << 'EOFAGENT'\n{script}\nEOFAGENT",
        ).wait()

        # Execute with system Python (has claude-agent-sdk)
        # Source .env first for API keys
        p = self._sandbox.exec(
            "bash", "-c",
            f"set -a && source {self._workspace}/.env 2>/dev/null; set +a; "
            f"python /tmp/agent_script.py",
        )
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        # Log agent output for debugging
        if stdout:
            logger.info(f"Agent stdout:\n{stdout[:2000]}")
        if stderr:
            logger.warning(f"Agent stderr:\n{stderr[:1000]}")

        result = parse_agent_output(stdout, stderr, p.returncode)

        # Commit changes if requested and successful
        if result.success and commit_changes:
            git = SandboxGitService(self._sandbox, self._workspace)
            git.configure_user()
            commit_result = git.stage_and_commit(
                f"Code mutation: {change_request[:50]}..."
            )
            if not commit_result.success and "no changes" not in commit_result.operation:
                return {
                    "success": False,
                    "error": f"Git commit failed: {commit_result.stderr}",
                    "output": result.output,
                }

        return {
            "success": result.success,
            "error": result.error,
            "output": result.output,
        }

    def exec_prebuilt(self, command: dict) -> dict:
        """Execute a prebuilt DSPy command via master.py.

        Writes the command to a temp file, executes master.py,
        and parses the EVAL_RESULT: prefix from stdout.

        Args:
            command: Dict with at minimum a "command" key.
                Supported commands: build_seed_candidate, evaluate, make_reflective_dataset

        Returns:
            Parsed JSON result dict from the handler.

        Raises:
            RuntimeError: If the sandbox is not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Base64-encode the command JSON for safe file transfer
        cmd_json = json.dumps(command)
        cmd_b64 = base64.b64encode(cmd_json.encode()).decode()

        # Write command file via bash (base64 decode for safety)
        write_p = self._sandbox.exec(
            "bash", "-c",
            f"echo '{cmd_b64}' | base64 -d > /tmp/prebuilt_command.json",
        )
        write_p.wait()

        # Execute master.py dispatcher (source .env first to load environment variables)
        p = self._sandbox.exec(
            "bash", "-c",
            f"set -a && source {self._workspace}/.env 2>/dev/null; set +a; "
            f"PYTHONPATH=/app:$PYTHONPATH python /app/sandbox_scripts/master.py "
            f"--workspace {self._workspace} "
            f"--command-file /tmp/prebuilt_command.json",
        )
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        # Print stderr which contains our debug output (print shows in Modal logs)
        if stderr:
            print(f"[SANDBOX STDERR]\n{stderr}", flush=True)

        if p.returncode != 0:
            print(f"[SANDBOX ERROR] master.py failed (rc={p.returncode}): {stderr}", flush=True)
            return {
                "success": False,
                "error": f"Prebuilt script exit code {p.returncode}: {stderr[:2000]}",
            }

        # Parse EVAL_RESULT: from stdout
        for line in stdout.split("\n"):
            if line.startswith("EVAL_RESULT:"):
                payload = line[len("EVAL_RESULT:"):]
                try:
                    return json.loads(payload)
                except json.JSONDecodeError as e:
                    return {
                        "success": False,
                        "error": f"Failed to parse script output: {e}",
                    }

        # No EVAL_RESULT found
        logger.error(f"No EVAL_RESULT in stdout. stdout={stdout[:2000]}, stderr={stderr[:2000]}")
        return {
            "success": False,
            "error": f"Prebuilt script produced no EVAL_RESULT. stderr: {stderr[:2000]}",
        }

    def exec_bash(self, command: str) -> dict:
        """Execute a simple bash command in the sandbox.

        Args:
            command: Bash command to execute.

        Returns:
            Dict with 'stdout', 'stderr', 'returncode' keys.

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        p = self._sandbox.exec(
            "bash", "-c",
            f"cd {self._workspace} && {command}",
        )
        p.wait()

        return {
            "stdout": p.stdout.read(),
            "stderr": p.stderr.read(),
            "returncode": p.returncode,
        }

    def exec_reflection_agent(
        self,
        prompt: str,
        max_turns: int = 20,
    ) -> dict:
        """Execute reflection agent with read-only tools.

        Uses Claude Agent SDK with only Read, Grep, Glob tools (no edits).
        The agent analyzes the codebase and feedback to propose a change.

        Args:
            prompt: Reflection prompt asking for a proposed change.
            max_turns: Maximum conversation turns.

        Returns:
            Dict with 'success', 'proposed_change', 'error' keys.

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        logger.info(f"Executing reflection agent: {prompt[:100]}...")

        # Generate reflection agent script
        script = self._generate_reflection_agent_script(
            workspace_path=self._workspace,
            prompt=prompt,
            max_turns=max_turns,
        )

        # Write script to sandbox
        self._sandbox.exec(
            "bash", "-c",
            f"cat > /tmp/reflection_script.py << 'EOFREFLECT'\n{script}\nEOFREFLECT",
        ).wait()

        # Execute with system Python (has claude-agent-sdk)
        p = self._sandbox.exec(
            "bash", "-c",
            f"set -a && source {self._workspace}/.env 2>/dev/null; set +a; "
            f"python /tmp/reflection_script.py",
        )
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        # Log for debugging
        if stderr:
            logger.warning(f"Reflection agent stderr:\n{stderr[:1000]}")

        # Parse the output
        return self._parse_reflection_output(stdout, stderr, p.returncode)

    def _generate_reflection_agent_script(
        self,
        workspace_path: str,
        prompt: str,
        max_turns: int = 20,
    ) -> str:
        """Generate Python script for reflection agent.

        Similar to coding agent but with read-only tools (Read, Grep, Glob).

        Args:
            workspace_path: Path to the workspace in the sandbox.
            prompt: Reflection prompt.
            max_turns: Maximum conversation turns.

        Returns:
            Python script as a string.
        """
        escaped_prompt = prompt.replace('"""', '\\"\\"\\"').replace("\\", "\\\\")

        return f'''#!/usr/bin/env python3
"""Auto-generated reflection agent script.

Analyzes codebase with read-only tools and proposes a change.
"""

import os
import sys
import subprocess

# Load environment variables
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

# Verify Claude Code CLI is installed
try:
    result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
except FileNotFoundError:
    print("REFLECT_ERROR: Claude Code CLI not found")
    sys.exit(1)

try:
    import anyio
    from claude_agent_sdk import ClaudeAgentOptions, query, AssistantMessage, ResultMessage, TextBlock

    prompt = """{escaped_prompt}"""
    workspace = "{workspace_path}"

    async def main():
        proposed_change = None

        async for message in query(
            prompt=prompt,
            options=ClaudeAgentOptions(
                cwd=workspace,
                # Read-only tools only - no edits
                allowed_tools=["Read", "Grep", "Glob"],
                permission_mode="acceptEdits",
                max_turns={max_turns},
            ),
        ):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        # Capture the last text block as the proposed change
                        proposed_change = block.text

            elif isinstance(message, ResultMessage):
                if message.is_error:
                    print(f"REFLECT_ERROR: {{message.result}}")
                    sys.exit(1)

        if proposed_change:
            print(f"REFLECT_PROPOSED_CHANGE: {{proposed_change}}")
        else:
            print("REFLECT_NO_CHANGE")

    anyio.run(main)

except ImportError as e:
    print(f"REFLECT_ERROR: claude-agent-sdk not installed: {{e}}")
    sys.exit(1)
except Exception as e:
    print(f"REFLECT_ERROR: {{e}}")
    sys.exit(1)
'''

    def _parse_reflection_output(
        self,
        stdout: str,
        stderr: str,
        returncode: int,
    ) -> dict:
        """Parse reflection agent output.

        Args:
            stdout: Standard output from the script.
            stderr: Standard error from the script.
            returncode: Exit code from the script.

        Returns:
            Dict with 'success', 'proposed_change', 'error' keys.
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
            return {
                "success": False,
                "proposed_change": "No change proposed",
                "error": error_msg,
            }

        # Parse proposed change
        for line in stdout.split("\n"):
            if line.startswith("REFLECT_PROPOSED_CHANGE: "):
                proposed = line[len("REFLECT_PROPOSED_CHANGE: "):]
                return {
                    "success": True,
                    "proposed_change": proposed,
                    "error": None,
                }

        if "REFLECT_NO_CHANGE" in stdout:
            return {
                "success": True,
                "proposed_change": "No change proposed",
                "error": None,
            }

        return {
            "success": False,
            "proposed_change": "No change proposed",
            "error": "Failed to parse reflection output",
        }
