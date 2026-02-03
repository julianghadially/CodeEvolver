"""Git operations for Modal sandbox execution.

This module provides git operations that execute inside a Modal sandbox.
Unlike git_service.py (which uses GitPython locally), these functions
execute bash commands within the sandbox's isolated filesystem.

Pattern follows agent.py - functions that execute commands and return
structured results with full logging.
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import modal


@dataclass
class GitResult:
    """Result from a git operation in the sandbox."""

    success: bool
    stdout: str
    stderr: str
    returncode: int
    operation: str

    def log(self, prefix: str = "[git]") -> None:
        """Log the result of this git operation."""
        status = "OK" if self.success else "FAILED"
        print(f"{prefix} {self.operation}: {status} (code={self.returncode})")
        if self.stdout.strip():
            for line in self.stdout.strip().split("\n")[:10]:  # First 10 lines
                print(f"{prefix}   {line}")
        if self.stderr.strip() and not self.success:
            print(f"{prefix}   stderr: {self.stderr[:500]}")


class SandboxGitService:
    """Git operations for Modal sandbox execution.

    This class wraps git bash commands for execution inside a Modal sandbox.
    All operations are logged for debugging and return structured results.

    Usage:
        git_service = SandboxGitService(sandbox, "/workspace")
        result = git_service.configure_user("bot@example.com", "Bot")
        if not result.success:
            raise RuntimeError(result.stderr)
    """

    def __init__(self, sandbox: "modal.Sandbox", workspace: str):
        """
        Initialize the git service for a sandbox.

        Args:
            sandbox: Modal sandbox instance
            workspace: Path to the git repository in the sandbox
        """
        self.sandbox = sandbox
        self.workspace = workspace

    def _exec(self, *args: str, operation: str) -> GitResult:
        """
        Execute a git command in the sandbox.

        Args:
            *args: Command arguments (e.g., "status", "--porcelain")
            operation: Human-readable name for logging

        Returns:
            GitResult with command output
        """
        cmd = ["git", "-C", self.workspace, *args]
        p = self.sandbox.exec(*cmd)
        p.wait()

        return GitResult(
            success=p.returncode == 0,
            stdout=p.stdout.read(),
            stderr=p.stderr.read(),
            returncode=p.returncode,
            operation=operation,
        )

    def configure_user(
        self,
        email: str = "codeevolver@anthropic.com",
        name: str = "CodeEvolver Agent",
    ) -> GitResult:
        """
        Configure git user.name and user.email.

        Required before commits will work in a fresh sandbox.

        Args:
            email: Git user email
            name: Git user name

        Returns:
            GitResult (returns result of user.name config)
        """
        # Set email
        result_email = self._exec(
            "config", "user.email", email,
            operation="config user.email",
        )
        result_email.log()

        if not result_email.success:
            return result_email

        # Set name
        result_name = self._exec(
            "config", "user.name", name,
            operation="config user.name",
        )
        result_name.log()

        return result_name

    def status(self, porcelain: bool = True) -> GitResult:
        """
        Get git status.

        Args:
            porcelain: If True, use --porcelain for machine-readable output

        Returns:
            GitResult with status output
        """
        args = ["status"]
        if porcelain:
            args.append("--porcelain")

        result = self._exec(*args, operation="status")
        result.log()
        return result

    def status_short(self) -> GitResult:
        """Get short branch status (git status -sb)."""
        result = self._exec("status", "-sb", operation="status -sb")
        result.log()
        return result

    def add_all(self) -> GitResult:
        """
        Stage all changes (git add -A).

        Relies on .gitignore to exclude sandbox artifacts (.venv, .env).
        The ClientSandbox.start() method ensures these entries exist in .gitignore.

        Returns:
            GitResult
        """
        result = self._exec("add", "-A", operation="add -A")
        result.log()
        return result

    def commit(self, message: str) -> GitResult:
        """
        Commit staged changes.

        Args:
            message: Commit message

        Returns:
            GitResult
        """
        result = self._exec("commit", "-m", message, operation="commit")
        result.log()
        return result

    def log_recent(self, count: int = 3) -> GitResult:
        """
        Get recent commit history.

        Args:
            count: Number of commits to show

        Returns:
            GitResult with commit log
        """
        result = self._exec(
            "log", f"-{count}", "--oneline",
            operation=f"log -{count}",
        )
        result.log()
        return result

    def push(self, branch: str, set_upstream: bool = True) -> GitResult:
        """
        Push to remote.

        Args:
            branch: Branch name to push
            set_upstream: If True, set upstream tracking (-u)

        Returns:
            GitResult
        """
        args = ["push"]
        if set_upstream:
            args.extend(["-u", "origin", branch])
        else:
            args.extend(["origin", branch])

        result = self._exec(*args, operation=f"push origin/{branch}")
        result.log()
        return result

    def checkout(self, branch: str, create: bool = False) -> GitResult:
        """
        Checkout a branch.

        Args:
            branch: Branch name
            create: If True, create the branch (-b)

        Returns:
            GitResult
        """
        args = ["checkout"]
        if create:
            args.append("-b")
        args.append(branch)

        result = self._exec(*args, operation=f"checkout {'-b ' if create else ''}{branch}")
        result.log()
        return result

    def diff_stat(self) -> GitResult:
        """Get diff statistics for staged and unstaged changes."""
        result = self._exec("diff", "--stat", operation="diff --stat")
        result.log()
        return result

    def has_changes(self) -> bool:
        """
        Check if there are any uncommitted changes.

        Returns:
            True if there are staged or unstaged changes
        """
        result = self.status(porcelain=True)
        return bool(result.stdout.strip())

    def stage_and_commit(self, message: str) -> GitResult:
        """
        Stage all changes and commit.

        This is a convenience method that combines add_all() and commit().
        It checks for changes before committing to avoid empty commits.

        Args:
            message: Commit message

        Returns:
            GitResult from commit (or status if no changes)
        """
        # Stage all changes
        add_result = self.add_all()
        if not add_result.success:
            return add_result

        # Check if there are changes to commit
        status_result = self.status(porcelain=True)
        if not status_result.stdout.strip():
            print("[git] No changes to commit")
            return GitResult(
                success=True,
                stdout="",
                stderr="",
                returncode=0,
                operation="stage_and_commit (no changes)",
            )

        # Commit
        return self.commit(message)

    def stage_commit_push(self, message: str, branch: str) -> GitResult:
        """
        Stage, commit, and push changes.

        This is the full workflow for pushing code changes to remote.

        Args:
            message: Commit message
            branch: Branch name to push

        Returns:
            GitResult from push (or earlier step if failed)
        """
        # Stage and commit
        commit_result = self.stage_and_commit(message)
        if not commit_result.success:
            return commit_result

        # Show what we're about to push
        self.log_recent(3)
        self.status_short()

        # Push
        return self.push(branch)


def clone_repository(
    sandbox: "modal.Sandbox",
    repo_url: str,
    workspace: str,
) -> GitResult:
    """
    Clone a repository in the sandbox.

    Args:
        sandbox: Modal sandbox instance
        repo_url: Git repository URL (may include auth token)
        workspace: Destination path

    Returns:
        GitResult
    """
    p = sandbox.exec("git", "clone", repo_url, workspace)
    p.wait()

    result = GitResult(
        success=p.returncode == 0,
        stdout=p.stdout.read(),
        stderr=p.stderr.read(),
        returncode=p.returncode,
        operation="clone",
    )
    result.log()
    return result
