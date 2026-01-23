"""Git operations service for cloning and managing repositories."""

import shutil
from pathlib import Path
from uuid import uuid4

import httpx
from git import Repo
from git.exc import GitCommandError

from ..config import settings
from .github_app import GitHubAppService


class GitService:
    """Service for git repository operations."""

    @staticmethod
    def generate_client_id() -> str:
        """Placeholder function for generating a unique client identifier."""
        return f"client_{uuid4().hex[:12]}"

    @staticmethod
    def get_workspace_path(client_id: str) -> Path:
        """Get the workspace path for a client."""
        return Path(settings.workspace_root) / client_id

    @staticmethod
    def get_main_repo_path(client_id: str) -> Path:
        """Get the path to the main repository clone."""
        return GitService.get_workspace_path(client_id) / "main"

    @staticmethod
    def clone_repository(
        repo_url: str,
        client_id: str,
        installation_id: int | None = None,
    ) -> Path:
        """
        Clone a repository to the client's workspace.

        Supports both public and private repositories. For private repos,
        provide a GitHub App installation_id to authenticate.

        Args:
            repo_url: URL of the git repository to clone
            client_id: Unique client identifier
            installation_id: Optional GitHub App installation ID for private repos

        Returns:
            Path to the cloned repository (main worktree)

        Raises:
            GitCommandError: If cloning fails
            ValueError: If repo_url is invalid or authentication fails
        """
        workspace_path = GitService.get_workspace_path(client_id)
        main_path = workspace_path / "main"

        workspace_path.mkdir(parents=True, exist_ok=True)

        # Handle authentication for private repositories
        authenticated_url = repo_url
        if installation_id:
            try:
                token = GitHubAppService.get_installation_token(installation_id)
                authenticated_url = GitHubAppService.get_authenticated_repo_url(
                    repo_url, token
                )
            except ValueError as e:
                shutil.rmtree(workspace_path, ignore_errors=True)
                raise ValueError(f"GitHub App authentication failed: {e}") from e

        try:
            Repo.clone_from(authenticated_url, main_path)
        except GitCommandError as e:
            shutil.rmtree(workspace_path, ignore_errors=True)
            raise ValueError(f"Failed to clone repository: {e}") from e

        return main_path

    @staticmethod
    def create_worktree(
        client_id: str,
        program_id: str,
        parent_branch: str = "main",
    ) -> tuple[Path, str]:
        """
        Create a git worktree for a new program mutation.

        Uses git worktree for efficient parallel branch access.
        Creates a new branch from the parent and checks it out in a separate directory.

        Args:
            client_id: Client identifier
            program_id: Program identifier for this mutation
            parent_branch: Branch to create from (default: main)

        Returns:
            Tuple of (worktree_path, branch_name)

        Raises:
            ValueError: If main repo doesn't exist or worktree creation fails
        """
        main_path = GitService.get_main_repo_path(client_id)
        if not main_path.exists():
            raise ValueError(f"Main repository not found for client {client_id}")

        branch_name = f"program_{program_id}"
        worktree_path = GitService.get_workspace_path(client_id) / program_id

        repo = Repo(main_path)

        try:
            # Fetch latest from remote if available
            if repo.remotes:
                repo.remotes.origin.fetch()

            # Create new branch from parent
            if parent_branch in repo.heads:
                base_commit = repo.heads[parent_branch].commit
            elif parent_branch.startswith("program_"):
                # Parent is another program branch - check if it exists
                if parent_branch in repo.heads:
                    base_commit = repo.heads[parent_branch].commit
                else:
                    # Fall back to main/master
                    base_commit = repo.heads[repo.active_branch.name].commit
            else:
                base_commit = repo.heads[repo.active_branch.name].commit

            # Create and add worktree with new branch
            repo.git.worktree("add", "-b", branch_name, str(worktree_path), base_commit.hexsha)

        except GitCommandError as e:
            shutil.rmtree(worktree_path, ignore_errors=True)
            raise ValueError(f"Failed to create worktree: {e}") from e

        return worktree_path, branch_name

    @staticmethod
    def get_worktree_path(client_id: str, program_id: str) -> Path:
        """Get the worktree path for a specific program."""
        return GitService.get_workspace_path(client_id) / program_id

    @staticmethod
    def remove_worktree(client_id: str, program_id: str) -> None:
        """
        Remove a worktree and its associated branch.

        Args:
            client_id: Client identifier
            program_id: Program identifier
        """
        main_path = GitService.get_main_repo_path(client_id)
        worktree_path = GitService.get_worktree_path(client_id, program_id)

        if main_path.exists():
            repo = Repo(main_path)
            try:
                repo.git.worktree("remove", str(worktree_path), "--force")
            except GitCommandError:
                # Worktree may already be removed, try manual cleanup
                shutil.rmtree(worktree_path, ignore_errors=True)

    @staticmethod
    def commit_changes(
        worktree_path: Path,
        message: str,
    ) -> str:
        """
        Commit all changes in a worktree.

        Args:
            worktree_path: Path to the worktree
            message: Commit message

        Returns:
            Commit SHA
        """
        repo = Repo(worktree_path)
        repo.git.add("-A")

        if repo.is_dirty() or repo.untracked_files:
            commit = repo.index.commit(message)
            return commit.hexsha

        return repo.head.commit.hexsha

    @staticmethod
    def cleanup_workspace(client_id: str) -> None:
        """Remove a client's workspace directory."""
        workspace_path = GitService.get_workspace_path(client_id)
        shutil.rmtree(workspace_path, ignore_errors=True)

    @staticmethod
    def fetch_github_file(
        repo_url: str,
        file_path: str,
        branch: str = "main",
        installation_id: int | None = None,
    ) -> str:
        """
        Fetch a file from a GitHub repository via the raw content API.

        Args:
            repo_url: Repository URL (https://github.com/owner/repo)
            file_path: Path to file within the repository
            branch: Branch name to fetch from
            installation_id: Optional GitHub App installation ID for private repos

        Returns:
            File contents as a string

        Raises:
            ValueError: If the file cannot be fetched
        """
        # Parse owner/repo from URL
        if repo_url.startswith("https://github.com/"):
            parts = repo_url.replace("https://github.com/", "").rstrip("/").split("/")
        elif repo_url.startswith("git@github.com:"):
            parts = repo_url.replace("git@github.com:", "").replace(".git", "").split("/")
        else:
            raise ValueError(f"Unsupported repository URL format: {repo_url}")

        if len(parts) < 2:
            raise ValueError(f"Could not parse owner/repo from URL: {repo_url}")

        owner = parts[0]
        repo = parts[1].replace(".git", "")

        # Build headers
        headers = {
            "Accept": "application/vnd.github.raw+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        # Add auth token if we have installation ID
        if installation_id:
            token = GitHubAppService.get_installation_token(installation_id)
            if token:
                headers["Authorization"] = f"Bearer {token}"

        # Use GitHub API to fetch file contents
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{file_path}?ref={branch}"

        with httpx.Client() as client:
            response = client.get(api_url, headers=headers, timeout=30.0)

            if response.status_code == 404:
                raise ValueError(
                    f"File not found: {file_path} on branch {branch} in {owner}/{repo}"
                )
            elif response.status_code != 200:
                raise ValueError(
                    f"Failed to fetch file: HTTP {response.status_code} - {response.text}"
                )

            return response.text
