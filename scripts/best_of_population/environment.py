"""Copyright © 2026 440 Labs LLC

Client venvs and secrets for offline evaluation.

One venv per (bench, sha256 of `requirements.txt`) under `<out>/venvs/`,
built the way `src/local/local_environment.py` builds a run's venv (uv, no
pip fallback, OpenTelemetry installed so traces carry per-row scores) but
NOT bound to a repo path: it is symlinked as `<worktree>/env`, which is the
directory `subprocess_service._build_env` prepends when `SANDBOX_MODE=local`.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path

TELEMETRY_PACKAGES = ("opentelemetry-api", "opentelemetry-sdk")
VENV_LINK_NAME = "env"


def requirements_hash(text: str | None) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()[:12]


def _uv() -> str:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required on PATH (the engine installs client deps with uv only)")
    return uv


def python_for(version: str | None) -> str:
    if version:
        cand = shutil.which(f"python{version}")
        if cand:
            return cand
    return shutil.which("python3.11") or shutil.which("python3") or sys.executable


def ensure_venv(
    venvs_root: Path, bench: str, reqhash: str, requirements_text: str | None,
    *, python_version: str | None = None,
) -> Path:
    """Create (once) and return `<venvs_root>/<bench>-<reqhash>`."""
    venv = venvs_root / f"{bench}-{reqhash}"
    marker = venv / ".frontier_ready"
    if marker.exists():
        return venv
    venvs_root.mkdir(parents=True, exist_ok=True)
    uv = _uv()
    if not (venv / "bin" / "python").exists():
        subprocess.run([uv, "venv", "--python", python_for(python_version), str(venv)], check=True)
    py = str(venv / "bin" / "python")
    if requirements_text:
        req = venv / "requirements.txt"
        req.write_text(requirements_text)
        subprocess.run([uv, "pip", "install", "--python", py, "-r", str(req)], check=True)
    subprocess.run([uv, "pip", "install", "--python", py, *TELEMETRY_PACKAGES], check=True)
    marker.write_text("ok\n")
    return venv


def link_venv(worktree: Path, venv: Path) -> None:
    link = worktree / VENV_LINK_NAME
    if link.is_symlink() or link.exists():
        if link.is_symlink() and os.readlink(link) == str(venv):
            return
        if link.is_symlink():
            link.unlink()
        else:
            raise RuntimeError(f"{link} exists and is not a symlink; refusing to replace")
    link.symlink_to(venv, target_is_directory=True)


def load_dotenv(path: str | None) -> None:
    if not path:
        return
    from src.engine.optimizer.v3._entry import _load_dotenv

    _load_dotenv(os.path.expanduser(path))


def missing_provider_keys(settings: dict | None) -> list[str]:
    """Keys the run had (`settings.json["provider_keys"]` == "present …")
    that are absent from this process's environment."""
    if not settings:
        return []
    out = []
    for name, value in (settings.get("provider_keys") or {}).items():
        if isinstance(value, str) and value.startswith("present") and not os.environ.get(name):
            out.append(name)
    return out


__all__ = [
    "VENV_LINK_NAME", "ensure_venv", "link_venv", "load_dotenv",
    "missing_provider_keys", "python_for", "requirements_hash",
]
