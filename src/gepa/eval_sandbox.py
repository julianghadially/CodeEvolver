"""Modal Sandbox lifecycle manager for GEPA evaluation.

Manages a long-lived Modal Sandbox that runs client code (DSPy programs,
metrics, etc.) in an isolated environment. The orchestrator communicates
with the sandbox via JSON IPC through eval_worker.py.

This provides process-level isolation: the GEPA orchestrator (Modal Function)
has no dspy installed, while the sandbox has the client's full requirements.txt.
"""

import base64
import json
import logging
import os
from typing import Any

import modal

from ..services.github_app import GitHubAppService

logger = logging.getLogger(__name__)


def _get_eval_image() -> modal.Image:
    """Build the minimal sandbox image for evaluation.

    Only includes git and Python. The client's requirements.txt
    (which includes dspy) is installed at sandbox start time.
    This avoids any version conflict with the orchestrator.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git")
        .add_local_file(
            "src/gepa/eval_worker.py",
            remote_path="/app/eval_worker.py",
        )
    )


class GEPAEvalSandbox:
    """Manages a Modal Sandbox for GEPA evaluation.

    Created once per optimization job. Clones the client repo, installs
    client deps, and runs eval_worker.py commands on demand via
    sandbox.exec().

    Usage:
        sandbox = GEPAEvalSandbox(app, repo_url, ...)
        sandbox.start()
        try:
            result = sandbox.exec_command({"command": "evaluate", ...})
        finally:
            sandbox.stop()
    """

    def __init__(
        self,
        app: modal.App,
        repo_url: str,
        installation_id: int | None = None,
        timeout: int = 3600,
        cpu: int = 2,
        memory: int = 4096,
    ):
        self.app = app
        self.repo_url = repo_url
        self.installation_id = installation_id
        self.timeout = timeout
        self.cpu = cpu
        self.memory = memory
        self._sandbox: modal.Sandbox | None = None
        self._workspace = "/workspace"

    def start(self) -> None:
        """Create sandbox, clone repo, pip install client requirements."""
        logger.info("Starting GEPA eval sandbox...")

        # Handle private repo authentication
        authenticated_url = self.repo_url
        if self.installation_id:
            token = GitHubAppService.get_installation_token(self.installation_id)
            if token:
                authenticated_url = GitHubAppService.get_authenticated_repo_url(
                    self.repo_url, token
                )

        # Collect API keys from environment to pass into sandbox
        env_vars = {}
        for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_KEY"):
            val = os.environ.get(key)
            if val:
                env_vars[key] = val

        # Create the Modal sandbox
        self._sandbox = modal.Sandbox.create(
            app=self.app,
            image=_get_eval_image(),
            timeout=self.timeout,
            cpu=self.cpu,
            memory=self.memory,
            encrypted_env_vars=env_vars,
        )
        logger.info(f"Sandbox created: {self._sandbox.object_id}")

        # Clone repository
        logger.info("Cloning repository into sandbox...")
        p = self._sandbox.exec(
            "git", "clone", authenticated_url, self._workspace,
        )
        p.wait()
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"git clone failed: {stderr}")

        # Install client's requirements.txt
        logger.info("Installing client dependencies...")
        p = self._sandbox.exec(
            "bash", "-c",
            f"if [ -f {self._workspace}/requirements.txt ]; then "
            f"pip install -r {self._workspace}/requirements.txt; fi",
        )
        p.wait()
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"pip install failed: {stderr}")

        logger.info("GEPA eval sandbox ready.")

    def exec_command(self, command: dict) -> dict:
        """Send a JSON command to sandbox / eval_worker and return the JSON result.

        Writes the command to a temp file, executes eval_worker.py,
        and parses the EVAL_RESULT: prefix from stdout.

        Args:
            command: Dict with at minimum a "command" key.

        Returns:
            Parsed JSON result dict from eval_worker.

        Raises:
            RuntimeError: If the sandbox is not started or the worker fails.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Base64-encode the command JSON for safe file transfer
        cmd_json = json.dumps(command)
        cmd_b64 = base64.b64encode(cmd_json.encode()).decode()

        # Write command file via bash (base64 decode for safety)
        write_p = self._sandbox.exec(
            "bash", "-c",
            f"echo '{cmd_b64}' | base64 -d > /tmp/eval_command.json",
        )
        write_p.wait()

        # Execute eval_worker
        p = self._sandbox.exec(
            "python", "/app/eval_worker.py",
            "--workspace", self._workspace,
            "--command-file", "/tmp/eval_command.json",
        )
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()

        if p.returncode != 0:
            logger.error(f"eval_worker failed (rc={p.returncode}): {stderr}")
            return {
                "success": False,
                "error": f"eval_worker exit code {p.returncode}: {stderr[:2000]}",
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
                        "error": f"Failed to parse eval_worker output: {e}",
                    }

        # No EVAL_RESULT found
        logger.error(f"No EVAL_RESULT in stdout. stdout={stdout[:2000]}, stderr={stderr[:2000]}")
        return {
            "success": False,
            "error": f"eval_worker produced no EVAL_RESULT. stderr: {stderr[:2000]}",
        }

    def reinstall_deps(self) -> None:
        """Re-run pip install (for use if requirements.txt changes)."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started.")
        p = self._sandbox.exec(
            "pip", "install", "-r", f"{self._workspace}/requirements.txt",
        )
        p.wait()

    def stop(self) -> None:
        """Terminate the sandbox."""
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
                logger.info("GEPA eval sandbox terminated.")
            except Exception as e:
                logger.warning(f"Failed to terminate sandbox: {e}")
            self._sandbox = None
