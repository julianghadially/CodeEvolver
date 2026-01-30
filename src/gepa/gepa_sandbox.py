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
