"""Base class for client code sandboxes.

Provides generic sandbox lifecycle management for running client code
in an isolated Modal Sandbox environment. Subclasses implement specific
prebuilt script execution logic.
"""

from abc import ABC, abstractmethod
import base64
import json
import logging
import os
from typing import Any

import httpx
import modal

from ..services.github_app import GitHubAppService

logger = logging.getLogger(__name__)


def _get_base_sandbox_image(python_version: str = "3.11") -> modal.Image:
    """Build the minimal sandbox image.

    Only includes git and Python. Client dependencies are installed
    at sandbox start time from their requirements.txt.
    """
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git")
        .add_local_dir(
            "src/core/sandbox_scripts",
            remote_path="/app/sandbox_scripts",
        )
    )


def _get_agent_capable_sandbox_image(python_version: str = "3.11") -> modal.Image:
    """Build sandbox image with coding agent capabilities.

    System Python: claude-agent-sdk, anyio (for agent execution)
    System: Node.js, npm, Claude Code CLI
    Client deps: installed at runtime into /workspace/.venv
    """
    return (
        modal.Image.debian_slim(python_version=python_version)
        .apt_install("git", "curl", "nodejs", "npm")
        .run_commands("npm install -g @anthropic-ai/claude-code")
        .pip_install("claude-agent-sdk>=0.1.21", "anyio>=4.0.0")
        .add_local_dir(
            "src/core/sandbox_scripts",
            remote_path="/app/sandbox_scripts",
        )
    )


class ClientSandbox(ABC):
    """Base class for client code sandboxes.

    Manages a Modal Sandbox lifecycle: creation, repo cloning,
    dependency installation, command execution, and termination.

    Subclasses implement exec_prebuilt() for specific script execution.

    Usage:
        sandbox = ConcreteClientSandbox(app, repo_url, ...)
        sandbox.start()
        try:
            result = sandbox.exec_prebuilt({"command": "some_command", ...})
        finally:
            sandbox.stop()
    """

    # Subclasses can override this to use a different image
    _image_builder = _get_base_sandbox_image

    def __init__(
        self,
        app: modal.App,
        repo_url: str,
        github_token: str | None = None,
        timeout: int = 3600,
        cpu: int = 2,
        memory: int = 4096,
        # Token refresh callback (for long-running jobs)
        callback_url: str | None = None,
        jwt_token: str | None = None,
        job_id: str | None = None,
    ):
        self.app = app
        self.repo_url = repo_url
        self.github_token = github_token  # Pre-generated token (no private key needed)
        self.timeout = timeout
        self.cpu = cpu
        self.memory = memory
        self._sandbox: modal.Sandbox | None = None
        self._workspace = "/workspace"
        self._use_venv = True
        # For token refresh (GitHub tokens expire after 1 hour)
        self._callback_url = callback_url
        self._jwt_token = jwt_token
        self._job_id = job_id

    def start(
        self,
        python_version: str = "3.11",
        use_venv: bool = True,
        branch: str | None = None,
    ) -> None:
        """Create sandbox, clone repo, pip install client requirements.

        Args:
            python_version: Python version for the sandbox image.
            use_venv: If True, create a venv at /workspace/.venv for client deps.
                      This isolates client deps from system Python (where agent SDK lives).
            branch: Git branch to clone. If None, clones the repository's default branch.
        """
        logger.info(f"Starting {self.__class__.__name__}...")
        self._use_venv = use_venv

        # Handle private repo authentication using pre-generated token
        authenticated_url = self.repo_url
        if self.github_token:
            authenticated_url = GitHubAppService.get_authenticated_repo_url(
                self.repo_url, self.github_token
            )

        # Create the Modal sandbox
        self._sandbox = modal.Sandbox.create(
            app=self.app,
            image=self.__class__._image_builder(python_version=python_version),
            timeout=self.timeout,
            cpu=self.cpu,
            memory=self.memory,
        )
        logger.info(f"Sandbox created: {self._sandbox.object_id}")

        # Clone repository (optionally with specific branch)
        logger.info(f"Cloning repository into sandbox...{f' (branch: {branch})' if branch else ''}")
        clone_args = ["git", "clone"]
        if branch:
            clone_args.extend(["--branch", branch])
        clone_args.extend([authenticated_url, self._workspace])
        p = self._sandbox.exec(*clone_args)
        p.wait()
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"git clone failed: {stderr}")

        # Ensure .venv and .env are in .gitignore (prevents pushing sandbox artifacts)
        self._ensure_gitignore_entries([".venv", ".env"])

        # Inject environment variables after clone
        # (temporary - will be replaced by secrets manager)
        env_vars = dict(os.environ)
        if env_vars:
            self._inject_env_vars(env_vars)

        # Install client's requirements.txt
        logger.info("Installing client dependencies...")

        # First check if requirements.txt exists
        check_p = self._sandbox.exec(
            "bash", "-c",
            f"if [ -f {self._workspace}/requirements.txt ]; then "
            f"echo 'REQUIREMENTS_FOUND'; cat {self._workspace}/requirements.txt; "
            f"else echo 'REQUIREMENTS_NOT_FOUND'; fi",
        )
        check_p.wait()
        check_output = check_p.stdout.read()
        if "REQUIREMENTS_FOUND" in check_output:
            logger.info(f"Found requirements.txt:\n{check_output}")
        else:
            logger.warning("No requirements.txt found in client repo")

        if use_venv:
            # Create venv for client deps (isolates from system Python)
            logger.info("Creating venv at /workspace/.venv...")
            p = self._sandbox.exec(
                "python", "-m", "venv", f"{self._workspace}/.venv",
            )
            p.wait()
            if p.returncode != 0:
                stderr = p.stderr.read()
                raise RuntimeError(f"venv creation failed: {stderr}")

            # Install deps into venv
            p = self._sandbox.exec(
                "bash", "-c",
                f"if [ -f {self._workspace}/requirements.txt ]; then "
                f"{self._workspace}/.venv/bin/pip install -r {self._workspace}/requirements.txt; fi",
            )
        else:
            # Install into system Python (original behavior)
            p = self._sandbox.exec(
                "bash", "-c",
                f"if [ -f {self._workspace}/requirements.txt ]; then "
                f"pip install -r {self._workspace}/requirements.txt; fi",
            )
        p.wait()
        stdout = p.stdout.read()
        stderr = p.stderr.read()
        if p.returncode != 0:
            raise RuntimeError(f"pip install failed: {stderr}")
        logger.info(f"pip install completed. stdout: {stdout[:500]}")

        logger.info(f"{self.__class__.__name__} ready.")

    def _inject_env_vars(self, env_vars: dict[str, str]) -> None:
        """Inject environment variables into the sandbox.
        
        Writes environment variables to a file that can be sourced.
        Note: This is temporary and will be replaced by secrets manager.
        """
        if not env_vars:
            return
        
        # Filter out sensitive or unnecessary env vars
        filtered_vars = {
            k: v for k, v in env_vars.items()
            if not k.startswith("_") and k not in ["PATH", "HOME", "USER"]
        }
        
        if not filtered_vars:
            return
            
        env_content = "\n".join(f"export {k}='{v}'" for k, v in filtered_vars.items())
        self._sandbox.exec(
            "bash",
            "-c",
            f"cat > {self._workspace}/.env << 'EOFENV'\n{env_content}\nEOFENV",
        ).wait()

    def _ensure_gitignore_entries(self, entries: list[str]) -> None:
        """Ensure specified entries are in .gitignore.

        Creates .gitignore if it doesn't exist, or appends missing entries.
        This prevents sandbox artifacts (.venv, .env) from being committed.

        Args:
            entries: List of patterns to add to .gitignore (e.g., [".venv", ".env"])
        """
        if self._sandbox is None:
            return

        gitignore_path = f"{self._workspace}/.gitignore"

        # Build a bash script that:
        # 1. Creates .gitignore if it doesn't exist
        # 2. Ensures file ends with newline before appending
        # 3. Appends each entry if not already present
        script_lines = [
            f"touch {gitignore_path}",
            # Add newline at end of file if not present (prevents concatenation with last line)
            f"[ -s {gitignore_path} ] && [ -n \"$(tail -c1 {gitignore_path})\" ] && echo '' >> {gitignore_path}",
        ]
        for entry in entries:
            # Use grep to check if entry exists, append if not
            script_lines.append(
                f"grep -qxF '{entry}' {gitignore_path} || echo '{entry}' >> {gitignore_path}"
            )

        script = " && ".join(script_lines)
        p = self._sandbox.exec("bash", "-c", script)
        p.wait()

        if p.returncode == 0:
            logger.info(f"Ensured .gitignore contains: {entries}")
        else:
            logger.warning(f"Failed to update .gitignore: {p.stderr.read()}")

    def exec_bash(self, command: str) -> tuple[int, str, str]:
        """Execute a bash command in the sandbox.

        Args:
            command: Bash command string to execute.

        Returns:
            Tuple of (return_code, stdout, stderr).

        Raises:
            RuntimeError: If sandbox not started.
        """
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started. Call start() first.")

        # Prepend venv to PATH when using venv isolation
        if self._use_venv:
            command = f'export PATH="{self._workspace}/.venv/bin:$PATH" && {command}'

        p = self._sandbox.exec("bash", "-c", command)
        p.wait()

        return p.returncode, p.stdout.read(), p.stderr.read()

    @abstractmethod
    def exec_prebuilt(self, command: dict) -> dict:
        """Execute a prebuilt script command in the sandbox.

        Subclasses implement this to run their specific scripts.

        Args:
            command: Dict with command parameters.

        Returns:
            Dict with execution results.
        """
        pass

    def reinstall_deps(self) -> None:
        """Re-run pip install (for use if requirements.txt changes)."""
        if self._sandbox is None:
            raise RuntimeError("Sandbox not started.")

        if self._use_venv:
            p = self._sandbox.exec(
                f"{self._workspace}/.venv/bin/pip",
                "install", "-r", f"{self._workspace}/requirements.txt",
            )
        else:
            p = self._sandbox.exec(
                "pip", "install", "-r", f"{self._workspace}/requirements.txt",
            )
        p.wait()

    def stop(self) -> None:
        """Terminate the sandbox."""
        if self._sandbox is not None:
            try:
                self._sandbox.terminate()
                logger.info(f"{self.__class__.__name__} terminated.")
            except Exception as e:
                logger.warning(f"Failed to terminate sandbox: {e}")
            self._sandbox = None

    def refresh_github_token(self) -> str | None:
        """Refresh the GitHub token by calling the FastAPI callback endpoint.

        GitHub installation tokens expire after 1 hour. For long-running
        optimizations, call this to get a fresh token before git operations.

        Returns:
            Fresh token string, or None if refresh failed.
        """
        if not all([self._callback_url, self._jwt_token, self._job_id]):
            logger.warning("Cannot refresh token: missing callback_url, jwt_token, or job_id")
            return None

        url = f"{self._callback_url}/internal/job/{self._job_id}/github-token"

        try:
            with httpx.Client(timeout=30.0) as client:
                response = client.get(
                    url,
                    headers={"Authorization": f"Bearer {self._jwt_token}"},
                )
                response.raise_for_status()
                data = response.json()

                if data.get("error"):
                    logger.warning(f"Token refresh failed: {data['error']}")
                    return None

                new_token = data.get("token")
                if new_token:
                    self.github_token = new_token
                    logger.info("GitHub token refreshed successfully")
                    return new_token

                return None

        except httpx.HTTPStatusError as e:
            logger.warning(f"Token refresh HTTP error: {e.response.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Token refresh failed: {e}")
            return None
