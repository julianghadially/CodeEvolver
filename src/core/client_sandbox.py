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
        self._use_venv = True

    def start(self, python_version: str = "3.11", use_venv: bool = True) -> None:
        """Create sandbox, clone repo, pip install client requirements.

        Args:
            python_version: Python version for the sandbox image.
            use_venv: If True, create a venv at /workspace/.venv for client deps.
                      This isolates client deps from system Python (where agent SDK lives).
        """
        logger.info(f"Starting {self.__class__.__name__}...")
        self._use_venv = use_venv

        # Handle private repo authentication
        authenticated_url = self.repo_url
        if self.installation_id:
            token = GitHubAppService.get_installation_token(self.installation_id)
            if token:
                authenticated_url = GitHubAppService.get_authenticated_repo_url(
                    self.repo_url, token
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

        # Clone repository
        logger.info("Cloning repository into sandbox...")
        p = self._sandbox.exec(
            "git", "clone", authenticated_url, self._workspace,
        )
        p.wait()
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"git clone failed: {stderr}")

        # Inject environment variables after clone
        # (temporary - will be replaced by secrets manager)
        env_vars = dict(os.environ)
        if env_vars:
            self._inject_env_vars(env_vars)

        # Install client's requirements.txt
        logger.info("Installing client dependencies...")
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
        if p.returncode != 0:
            stderr = p.stderr.read()
            raise RuntimeError(f"pip install failed: {stderr}")

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
