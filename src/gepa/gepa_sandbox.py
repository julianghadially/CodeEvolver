"""GEPA-specific client sandbox implementation.

Extends ClientSandbox with exec_prebuilt() for DSPy command execution
and exec_agent() for coding agent code mutations.
"""

import base64
import json
import logging

from ..core.agent import (
    generate_agent_script,
    generate_reflection_agent_script,
    parse_agent_output,
    parse_reflection_output,
)
from ..core.client_sandbox import ClientSandbox, _get_agent_capable_sandbox_image
from ..schemas.lm_output_schemas import ArchitectureOutput, ChangeRequestOutput
from ..services.git_sandbox import SandboxGitService
from ..services.github_app import GitHubAppService

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

    # Commit message for code mutations (set by adapter before mutations begin)
    # Format: "codeevolver mutation. Date: YYYYMMDDHHmmss"
    commit_message: str | None = None

    def exec_agent(
        self,
        change_request: str,
        change_location: str | None = None,
        max_turns: int = 50,
        commit_changes: bool = True,
        push_branch: str | None = None,
    ) -> dict:
        """Execute coding agent for code mutations.

        Uses system Python to run Claude Agent SDK.
        Agent's Bash tool uses venv PATH for client code execution.

        Args:
            change_request: Natural language description of code change.
            change_location: Optional module path hint (e.g., "src/core/agent.py").
            max_turns: Maximum conversation turns (prevents runaway agents).
            commit_changes: If True, commit changes after successful mutation.
            push_branch: If provided, push to this branch after committing.

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

        # Commit and optionally push changes if requested and successful
        if result.success and commit_changes:
            git = SandboxGitService(self._sandbox, self._workspace)
            git.configure_user()

            # Use custom commit message or default
            msg = self.commit_message or f"Code mutation: {change_request[:50]}..."
            commit_result = git.stage_and_commit(msg)

            if not commit_result.success and "no changes" not in commit_result.operation:
                return {
                    "success": False,
                    "error": f"Git commit failed: {commit_result.stderr}",
                    "output": result.output,
                }

            # Push if branch is specified (use authenticated push for fresh token)
            if push_branch and commit_result.success:
                push_result = self.push_authenticated(push_branch)
                if not push_result.get("success"):
                    return {
                        "success": False,
                        "error": f"Git push failed: {push_result.get('stderr')}",
                        "output": result.output,
                    }
                logger.info(f"Pushed to origin/{push_branch}")

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
        # Use venv Python (where client deps like dspy are installed)
        venv_path_export = f'export PATH="{self._workspace}/.venv/bin:$PATH" && ' if self._use_venv else ""
        p = self._sandbox.exec(
            "bash", "-c",
            f"set -a && source {self._workspace}/.env 2>/dev/null; set +a; "
            f"{venv_path_export}"
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

    def push_authenticated(self, branch: str, set_upstream: bool = True) -> dict:
        """Push to remote with GitHub token, refreshing if needed.

        GitHub tokens expire after 1 hour. This method refreshes the token
        via callback to FastAPI before pushing to ensure authentication works.

        Args:
            branch: Branch name to push.
            set_upstream: If True, set upstream tracking (-u).

        Returns:
            Dict with 'success', 'stdout', 'stderr', 'returncode' keys.

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Refresh token before push (tokens expire after 1 hour)
        # This ensures we have a valid token for long-running optimizations
        if self._callback_url and self._jwt_token and self._job_id:
            refreshed = self.refresh_github_token()
            if refreshed:
                logger.info("Using refreshed GitHub token for push")

        # If no github_token, fall back to unauthenticated push
        if not self.github_token:
            logger.warning("No github_token - attempting unauthenticated push")
            return self._exec_push(branch, set_upstream)

        try:
            # Get authenticated URL using token (refreshed or original)
            auth_url = GitHubAppService.get_authenticated_repo_url(self.repo_url, self.github_token)

            # Update remote URL with token
            update_result = self._sandbox.exec(
                "bash", "-c",
                f"cd {self._workspace} && git remote set-url origin '{auth_url}'",
            )
            update_result.wait()
            if update_result.returncode != 0:
                logger.warning(f"Failed to update remote URL: {update_result.stderr.read()}")
                return self._exec_push(branch, set_upstream)

            # Push with authenticated remote
            result = self._exec_push(branch, set_upstream)

            # Reset remote URL to non-authenticated version (security best practice)
            # This prevents the token from being visible in git remote -v output
            reset_result = self._sandbox.exec(
                "bash", "-c",
                f"cd {self._workspace} && git remote set-url origin '{self.repo_url}'",
            )
            reset_result.wait()

            return result

        except Exception as e:
            logger.error(f"Error during authenticated push: {e}")
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def _exec_push(self, branch: str, set_upstream: bool = True) -> dict:
        """Execute git push command.

        Args:
            branch: Branch name to push.
            set_upstream: If True, set upstream tracking (-u).

        Returns:
            Dict with 'success', 'stdout', 'stderr', 'returncode' keys.
        """
        if set_upstream:
            cmd = f"cd {self._workspace} && git push -u origin {branch}"
        else:
            cmd = f"cd {self._workspace} && git push origin {branch}"

        p = self._sandbox.exec("bash", "-c", cmd)
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        return {
            "success": p.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": p.returncode,
        }

    def exec_reflection_agent(
        self,
        prompt: str,
        output_type: str = "change_request",
        max_turns: int = 20,
    ) -> dict:
        """Execute reflection agent with read-only tools and structured output.

        Uses Claude Agent SDK with only Read, Grep, Glob tools (no edits).
        Returns validated JSON via structured output.

        Args:
            prompt: Reflection prompt asking for a proposed change.
            output_type: Type of output schema to use ("architecture" or "change_request").
            max_turns: Maximum conversation turns.

        Returns:
            Dict with 'success', 'proposed_change', 'error' keys.

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        logger.info(f"Executing reflection agent ({output_type}): {prompt[:100]}...")

        # Select schema based on output type
        if output_type == "architecture":
            schema = ArchitectureOutput.model_json_schema()
            output_key = "architecture"
        else:
            schema = ChangeRequestOutput.model_json_schema()
            output_key = "change_request"

        # Generate reflection agent script with structured output
        script = generate_reflection_agent_script(
            workspace_path=self._workspace,
            prompt=prompt,
            output_schema=schema,
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

        # Parse the structured output
        result = parse_reflection_output(stdout, stderr, p.returncode, output_key)

        return {
            "success": result.success,
            "proposed_change": result.output or "No change proposed",
            "error": result.error,
        }
