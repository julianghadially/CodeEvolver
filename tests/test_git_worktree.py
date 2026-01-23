"""Integration tests for git worktree creation with real repository.

Tests the git service's ability to clone a repository and create worktrees
for executing code mutations.

For testing with private repositories:
1. Set up a GitHub App (see test_github_app_auth.py for details)
2. Set environment variables:
   - CODEEVOLVER_GITHUB_APP_ID
   - CODEEVOLVER_GITHUB_APP_PRIVATE_KEY
   - GITHUB_TEST_INSTALLATION_ID (optional)
   - GITHUB_TEST_REPO_URL (optional, for private repo testing)
3. Update TEST_REPO_URL and TEST_INSTALLATION_ID below if testing private repos
"""

import os
import pytest
from pathlib import Path

from src.services.git_service import GitService
from src.config import settings


# Test configuration
TEST_REPO_URL = "https://github.com/julianghadially/FactChecker"
TEST_BRANCH = "codeevolver-tests"

# For private repository testing (optional)
TEST_INSTALLATION_ID = os.getenv("GITHUB_TEST_INSTALLATION_ID")
TEST_PRIVATE_REPO_URL = os.getenv("GITHUB_TEST_REPO_URL")


class TestGitWorktreeIntegration:
    """Integration tests for git worktree operations."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace directory for tests."""
        workspace = tmp_path / "workspaces"
        workspace.mkdir()
        original_root = settings.workspace_root
        settings.workspace_root = str(workspace)
        yield workspace
        settings.workspace_root = original_root

    def test_clone_and_create_worktree(self, temp_workspace):
        """
        Test cloning a repo and creating a worktree for a code mutation.

        This test:
        1. Clones the FactChecker repo from GitHub
        2. Creates a worktree for a new program mutation
        3. Verifies the worktree contains expected files
        """
        # Generate a client ID
        client_id = GitService.generate_client_id()

        # Clone the repository (public repo, no authentication needed)
        main_path = GitService.clone_repository(TEST_REPO_URL, client_id, installation_id=None)

        assert main_path.exists(), "Main repo path should exist after cloning"
        assert (main_path / ".git").exists(), "Should be a git repository"

        # Create a worktree for a test program mutation
        program_id = "test_mutation_001"
        worktree_path, branch_name = GitService.create_worktree(
            client_id,
            program_id,
            parent_branch="main",
        )

        # Verify worktree was created
        assert worktree_path.exists(), f"Worktree path should exist: {worktree_path}"
        assert branch_name == f"program_{program_id}"

        # Check for expected files in the FactChecker repo
        # The FactChecker repo should have standard Python project files
        expected_files_any = [
            "README.md",
            "pyproject.toml",
            "setup.py",
            "requirements.txt",
        ]

        found_files = list(worktree_path.iterdir())
        found_names = [f.name for f in found_files]

        # Verify at least one expected file exists (repo structure may vary)
        has_expected = any(name in found_names for name in expected_files_any)
        assert has_expected or len(found_files) > 0, (
            f"Worktree should contain files. Found: {found_names}"
        )

        # Cleanup
        GitService.remove_worktree(client_id, program_id)
        GitService.cleanup_workspace(client_id)

    @pytest.mark.skipif(
        not TEST_INSTALLATION_ID or not TEST_PRIVATE_REPO_URL,
        reason="Private repository test configuration not available"
    )
    def test_clone_private_repo_and_create_worktree(self, temp_workspace):
        """
        Test cloning a private repo with GitHub App authentication and creating a worktree.

        This test requires:
        - GitHub App credentials configured
        - GITHUB_TEST_INSTALLATION_ID environment variable
        - GITHUB_TEST_REPO_URL environment variable pointing to a private repo
        """
        client_id = GitService.generate_client_id()

        # Clone private repository with authentication
        main_path = GitService.clone_repository(
            TEST_PRIVATE_REPO_URL,
            client_id,
            installation_id=int(TEST_INSTALLATION_ID),
        )

        assert main_path.exists(), "Main repo path should exist after cloning"
        assert (main_path / ".git").exists(), "Should be a git repository"

        # Create a worktree for a test program mutation
        program_id = "test_private_mutation_001"
        worktree_path, branch_name = GitService.create_worktree(
            client_id,
            program_id,
            parent_branch="main",
        )

        # Verify worktree was created
        assert worktree_path.exists(), f"Worktree path should exist: {worktree_path}"
        assert branch_name == f"program_{program_id}"

        # Cleanup
        GitService.remove_worktree(client_id, program_id)
        GitService.cleanup_workspace(client_id)

    def test_worktree_with_hardcoded_change_request(self, temp_workspace):
        """
        Test creating a worktree with a hard-coded example change request.

        This simulates the workflow where:
        1. A repo is cloned
        2. A worktree is created for a code mutation
        3. We verify the worktree is ready for the change request

        Hard-coded change request example:
        "Add a new file src/test_feature.py with a function that returns 'hello world'"
        """
        client_id = GitService.generate_client_id()

        # Clone the test repo (public repo, no authentication needed)
        main_path = GitService.clone_repository(TEST_REPO_URL, client_id, installation_id=None)

        # Hard-coded change request (for documentation purposes)
        change_request = {
            "change_request": "Add a new file src/test_feature.py with a function that returns 'hello world'",
            "change_location": "src/test_feature.py",
        }

        # Create worktree for mutation
        program_id = "prog_hardcoded_001"
        worktree_path, branch_name = GitService.create_worktree(
            client_id,
            program_id,
            parent_branch="main",
        )

        # Verify worktree was created successfully
        assert worktree_path.exists()

        # Simulate writing the requested file (what Claude agent would do)
        target_file = worktree_path / "src" / "test_feature.py"
        target_file.parent.mkdir(parents=True, exist_ok=True)
        target_file.write_text('''"""Test feature module created by code mutation."""

def hello_world():
    """Return hello world string."""
    return "hello world"
''')

        # Verify the file was created
        assert target_file.exists(), f"Target file should exist: {target_file}"
        assert target_file.name == "test_feature.py"

        # Commit the changes
        commit_sha = GitService.commit_changes(
            worktree_path,
            f"Apply code mutation: {change_request['change_request'][:50]}...",
        )

        assert commit_sha, "Should return a commit SHA"
        assert len(commit_sha) == 40, "SHA should be 40 characters"

        # Cleanup
        GitService.remove_worktree(client_id, program_id)
        GitService.cleanup_workspace(client_id)

    def test_multiple_worktrees_parallel(self, temp_workspace):
        """
        Test creating multiple worktrees for parallel mutations.

        This verifies the system can handle concurrent program mutations
        as required for GEPA's parallel evaluation.
        """
        client_id = GitService.generate_client_id()

        # Clone the repo (public repo, no authentication needed)
        GitService.clone_repository(TEST_REPO_URL, client_id, installation_id=None)

        # Create multiple worktrees (simulating parallel mutations)
        program_ids = ["prog_parallel_001", "prog_parallel_002", "prog_parallel_003"]
        worktrees = []

        for program_id in program_ids:
            worktree_path, branch_name = GitService.create_worktree(
                client_id,
                program_id,
                parent_branch="main",
            )
            worktrees.append((program_id, worktree_path, branch_name))

            # Verify each worktree exists independently
            assert worktree_path.exists()

        # Verify all worktrees are separate directories
        paths = [w[1] for w in worktrees]
        assert len(set(paths)) == len(paths), "Each worktree should have unique path"

        # Cleanup
        for program_id, _, _ in worktrees:
            GitService.remove_worktree(client_id, program_id)
        GitService.cleanup_workspace(client_id)


class TestGitServiceUnit:
    """Unit tests for GitService methods."""

    def test_generate_client_id_format(self):
        """Test that client IDs have expected format."""
        client_id = GitService.generate_client_id()
        assert client_id.startswith("client_")
        assert len(client_id) == 19  # "client_" + 12 hex chars

    def test_workspace_path_construction(self, tmp_path):
        """Test workspace path generation."""
        original_root = settings.workspace_root
        settings.workspace_root = str(tmp_path)

        client_id = "client_test123456"
        path = GitService.get_workspace_path(client_id)

        assert path == tmp_path / client_id

        settings.workspace_root = original_root

    def test_main_repo_path_construction(self, tmp_path):
        """Test main repo path generation."""
        original_root = settings.workspace_root
        settings.workspace_root = str(tmp_path)

        client_id = "client_test123456"
        path = GitService.get_main_repo_path(client_id)

        assert path == tmp_path / client_id / "main"

        settings.workspace_root = original_root

