"""Repo-path lock for in-place local optimizations.

Local-mode optimizations run directly on the operator's repo checkout
(no clone). Two concurrent runs on the same repo would race on the git
tree, so `LocalEnvironment.start()` takes an exclusive `fcntl.flock` on
`<repo_path>/.codeevolver.lock` and holds it for the lifetime of the run.

If the lock is already held, `acquire()` raises `RepoLockedError` — the
API translates that into a 409 Conflict for the second `/optimize`.
"""
from __future__ import annotations

import fcntl
import os
from pathlib import Path


class RepoLockedError(RuntimeError):
    """Raised when the repo lockfile is already held by another process."""


class RepoLock:
    """Non-blocking exclusive flock on `<repo_path>/.codeevolver.lock`."""

    LOCK_FILENAME = ".codeevolver.lock"

    def __init__(self, repo_path: str | os.PathLike[str]) -> None:
        self.repo_path = Path(repo_path)
        self.lock_path = self.repo_path / self.LOCK_FILENAME
        self._fd: int | None = None

    def acquire(self) -> None:
        if self._fd is not None:
            return  # already held by this instance
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(fd)
            raise RepoLockedError(
                f"Another optimization is already running on {self.repo_path}"
            ) from exc
        # Stamp the lockfile with our PID for diagnostics.
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except OSError:
            pass
        self._fd = fd

    def release(self) -> None:
        if self._fd is None:
            return
        try:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
        finally:
            try:
                os.close(self._fd)
            finally:
                self._fd = None

    def __enter__(self) -> "RepoLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
