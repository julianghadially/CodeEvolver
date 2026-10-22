"""Local execution environment for the local deployment.

Exposes a sandbox-style public surface (`start`, `write_file`, `stop`,
`refresh_github_token`) but operates **in-place** on the repo path the
user registered via `POST /local/connect-repo`. No clone, no managed
workspace dir; each job runs against the existing checkout. A repo-path
lock (`src/local/repo_lock.py`) prevents two concurrent runs on the same
tree.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

from src.local.repo_lock import RepoLock

logger = logging.getLogger(__name__)

PRE_PUSH_HOOK_SCRIPT = r"""#!/bin/sh
# CodeEvolver branch guard — blocks pushes to non-codeevolver branches.
# When no origin remote is configured, git never invokes this hook.
while read local_ref local_sha remote_ref remote_sha; do
    branch_name="${remote_ref#refs/heads/}"
    case "$branch_name" in
        codeevolver-*)
            ;;
        *)
            echo "BRANCH GUARD: Push to '$branch_name' blocked. Only codeevolver-* branches are allowed." >&2
            exit 1
            ;;
    esac
done
exit 0
"""


class LocalEnvironment:
    """In-place local workspace lifecycle.

    Usage:
        env = LocalEnvironment(repo_path=..., callback_url=..., jwt_token=..., job_id=...)
        env.start(python_version="3.11")
        try:
            # optimizer subprocess does its work against env.workspace_path
            ...
        finally:
            env.stop()
    """

    DEFAULT_VENV_DIRNAME = "env"

    def __init__(
        self,
        repo_path: str,
        github_token: str | None = None,
        callback_url: str | None = None,
        jwt_token: str | None = None,
        job_id: str | None = None,
        venv_dirname: str = DEFAULT_VENV_DIRNAME,
    ) -> None:
        self.repo_path = Path(repo_path).expanduser().resolve()
        self.github_token = github_token
        self._callback_url = callback_url
        self._jwt_token = jwt_token
        self._job_id = job_id
        self._venv_dirname = venv_dirname
        self._lock = RepoLock(self.repo_path)
        self._started = False

    @property
    def workspace_path(self) -> str:
        return str(self.repo_path)

    @property
    def venv_path(self) -> Path:
        return self.repo_path / self._venv_dirname

    def start(
        self,
        python_version: str = "3.11",
        use_venv: bool = True,
        branch: str | None = None,
        additional_pip_list: list[str] | None = None,
    ) -> None:
        """Acquire the repo lock, validate state, prepare venv + deps + git guard."""
        logger.info("Starting LocalEnvironment on %s", self.repo_path)
        if not self.repo_path.exists():
            raise FileNotFoundError(
                f"Registered repo path does not exist: {self.repo_path}"
            )
        if not (self.repo_path / ".git").exists():
            raise RuntimeError(
                f"{self.repo_path} is not a git repository (no .git/)."
            )

        self._lock.acquire()
        self._started = True

        try:
            self._install_pre_push_guard()
            self._ensure_gitignore_entries(
                [".venv", "env", ".env", ".claude/", ".opencode/",  ".codeevolver/"]
            )
            self._install_runtime_deps()

            if use_venv:
                self._ensure_venv(python_version)
                self._install_client_requirements()
                if additional_pip_list:
                    self._pip_install_into_venv(additional_pip_list)
                self._install_telemetry_runtime()
            else:
                logger.info("use_venv=False — skipping venv setup.")

            logger.info("LocalEnvironment ready at %s", self.repo_path)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        """Release the repo lock. Leaves the workspace and codeevolver-* branches in place."""
        if self._started:
            self._lock.release()
            self._started = False

    def write_file(self, path: str, content: str) -> None:
        """Write content to a file inside the workspace (mkdir -p the parent)."""
        target = self._resolve_workspace_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def exec_bash(self, command: str) -> tuple[int, str, str]:
        """Run a bash command inside the workspace with the venv on PATH."""
        env_cmd = command
        if self.venv_path.exists():
            env_cmd = f'export PATH="{self.venv_path / "bin"}:$PATH" && {command}'
        result = subprocess.run(
            ["bash", "-c", env_cmd],
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        return result.returncode, result.stdout, result.stderr

    def reinstall_deps(self) -> None:
        """Re-run uv pip install into the venv (use when requirements.txt changes)."""
        if not self.venv_path.exists():
            logger.warning("reinstall_deps called but venv is missing; skipping.")
            return
        self._install_client_requirements()

    def refresh_github_token(self) -> str | None:
        """Call the API's GitHub-token refresh endpoint.

        Works because the optimizer's HTTP callback path is
        environment-agnostic.
        """
        if not all([self._callback_url, self._jwt_token, self._job_id]):
            logger.warning(
                "Cannot refresh token: missing callback_url, jwt_token, or job_id"
            )
            return None
        url = f"{self._callback_url}/internal/job/{self._job_id}/github-token"
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with httpx.Client(timeout=60.0) as client:
                    response = client.get(
                        url,
                        headers={"Authorization": f"Bearer {self._jwt_token}"},
                    )
                    response.raise_for_status()
                    data = response.json()
                    if data.get("error"):
                        logger.warning("Token refresh failed: %s", data["error"])
                        return None
                    new_token = data.get("token")
                    if new_token:
                        self.github_token = new_token
                        logger.info("GitHub token refreshed successfully")
                        return new_token
                    return None
            except httpx.HTTPStatusError as e:
                logger.warning("Token refresh HTTP error: %s", e.response.status_code)
                return None
            except Exception as e:  # pragma: no cover — transient
                if attempt < max_retries - 1:
                    import time
                    wait = 5 * (attempt + 1)
                    logger.warning(
                        "Token refresh attempt %d failed: %s. Retrying in %ds...",
                        attempt + 1, e, wait,
                    )
                    time.sleep(wait)
                else:
                    logger.warning(
                        "Token refresh failed after %d attempts: %s",
                        max_retries, e,
                    )
                    return None
        return None

    # ---------------- internal helpers ----------------

    def _resolve_workspace_path(self, path: str) -> Path:
        """Resolve `path` relative to the workspace (absolute paths pass through)."""
        p = Path(path)
        return p if p.is_absolute() else self.repo_path / p

    def _resolve_uv_binary(self) -> str:
        """Locate the `uv` binary on the host PATH.

        uv is the installer for the client venv (see `_install_client_requirements`).
        It is a host-level binary, not a venv package — it installs into any venv
        via `uv pip install --python <venv>/bin/python`, so the venv itself needs
        neither pip nor uv inside it. We standardize on uv (no pip fallback) so
        installs are reproducible and immune to pip-less venvs.
        """
        uv = shutil.which("uv")
        if not uv:
            raise RuntimeError(
                "uv is required but was not found on PATH. Install it with "
                "`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`."
            )
        return uv

    def _ensure_venv(self, python_version: str) -> None:
        if self.venv_path.exists():
            # Reuse the existing venv as-is. uv installs into it regardless of
            # how it was created (pip- or uv-made) and regardless of whether it
            # contains pip, so we never clobber a user's venv.
            logger.info("Venv already present at %s; skipping create.", self.venv_path)
            return
        uv = self._resolve_uv_binary()
        python_bin = self._resolve_python_binary(python_version)
        logger.info("Creating venv at %s using %s (via uv)", self.venv_path, python_bin)
        result = subprocess.run(
            [uv, "venv", "--python", python_bin, str(self.venv_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"venv creation failed: {result.stderr}")

    def _resolve_python_binary(self, python_version: str) -> str:
        """Find a python<version> binary on PATH; fall back to sys.executable."""
        candidate = f"python{python_version}"
        path = shutil.which(candidate)
        if path:
            return path
        major = python_version.split(".", 1)[0]
        path = shutil.which(f"python{major}")
        if path:
            logger.warning(
                "python%s not found; falling back to %s — version may not match.",
                python_version, path,
            )
            return path
        logger.warning(
            "No python%s on PATH; falling back to %s.",
            python_version, sys.executable,
        )
        return sys.executable

    def _install_client_requirements(self) -> None:
        requirements = self.repo_path / "requirements.txt"
        if not requirements.exists():
            logger.warning("No requirements.txt at %s", requirements)
            return
        logger.info("Installing client requirements into %s", self.venv_path)
        result = subprocess.run(
            self._uv_install_cmd(["-r", str(requirements)]),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"uv pip install failed: {result.stderr}")
        logger.info("Client requirements installed.")

    def _pip_install_into_venv(self, packages: list[str]) -> None:
        result = subprocess.run(
            self._uv_install_cmd(packages),
            cwd=self.repo_path,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"uv pip install {packages} failed: {result.stderr}")

    def _uv_install_cmd(self, args: list[str]) -> list[str]:
        """Build a `uv pip install` command targeting the client venv's Python.

        uv installs into the venv via its interpreter path — no pip needed
        inside the venv (see `_resolve_uv_binary`).
        """
        venv_python = self.venv_path / "bin" / "python"
        return [
            self._resolve_uv_binary(), "pip", "install",
            "--python", str(venv_python), *args,
        ]

    def _install_telemetry_runtime(self) -> None:
        """OpenTelemetry must live in the client venv (evaluator subprocess uses it)."""
        try:
            self._pip_install_into_venv(
                ["opentelemetry-api>=1.27.0", "opentelemetry-sdk>=1.27.0"]
            )
            logger.info("OpenTelemetry installed into client venv.")
        except RuntimeError as e:
            logger.warning(
                "OTel venv install failed: %s — traces will be empty for this run.",
                e,
            )

    def _install_runtime_deps(self) -> None:
        """Run each non-comment line in `runtime_requirements.txt` as a bash command."""
        runtime_file = self.repo_path / "runtime_requirements.txt"
        if not runtime_file.exists():
            logger.info("No runtime_requirements.txt found (optional).")
            return
        commands = []
        for raw in runtime_file.read_text().splitlines():
            line = raw.strip()
            if line and not line.startswith("#"):
                commands.append(line)
        if not commands:
            return
        logger.info("Running %d runtime install commands...", len(commands))
        for cmd in commands:
            logger.info("Running: %s", cmd)
            result = subprocess.run(
                ["bash", "-c", cmd],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                logger.warning(
                    "Runtime install command failed: %s\n%s",
                    cmd, result.stderr,
                )

    def _install_pre_push_guard(self) -> None:
        """Install the git pre-push hook that blocks non-codeevolver branches.

        Short-circuits when no `origin` remote is configured — git only
        invokes pre-push when there is a push target, so an absent remote
        means the hook would never fire anyway; we still install it so
        that adding a remote later picks up the guard automatically.
        """
        hooks_dir = self.repo_path / ".git" / "hooks"
        if not hooks_dir.parent.exists():
            logger.warning("No .git directory; skipping pre-push guard install.")
            return
        hooks_dir.mkdir(parents=True, exist_ok=True)
        hook_path = hooks_dir / "pre-push"
        hook_path.write_text(PRE_PUSH_HOOK_SCRIPT)
        hook_path.chmod(0o755)
        logger.info("Installed pre-push branch guard at %s", hook_path)

    def _ensure_gitignore_entries(self, entries: list[str]) -> None:
        gitignore = self.repo_path / ".gitignore"
        existing = (
            gitignore.read_text().splitlines() if gitignore.exists() else []
        )
        existing_set = {line.strip() for line in existing}
        appended = [e for e in entries if e not in existing_set]
        if not appended:
            return
        with gitignore.open("a") as f:
            if existing and existing[-1].strip():
                f.write("\n")
            for entry in appended:
                f.write(f"{entry}\n")
        logger.info("Appended to .gitignore: %s", appended)
