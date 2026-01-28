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

    def start(self, python_version: str = "3.11") -> None:
        """Create sandbox, clone repo, pip install client requirements."""
        logger.info(f"Starting {self.__class__.__name__}...")

        # Handle private repo authentication
        authenticated_url = self.repo_url
        if self.installation_id:
            token = GitHubAppService.get_installation_token(self.installation_id)
            if token:
                authenticated_url = GitHubAppService.get_authenticated_repo_url(
                    self.repo_url, token
                )

        # Pass all environment variables to sandbox (temporary - will be replaced by secrets manager)
        env_vars = dict(os.environ)

        # Create the Modal sandbox
        self._sandbox = modal.Sandbox.create(
            app=self.app,
            image=self._image_builder(python_version = python_version),
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

        logger.info(f"{self.__class__.__name__} ready.")

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
