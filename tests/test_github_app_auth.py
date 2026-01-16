"""Tests for GitHub App authentication and private repository access.

These tests verify that the codebase can connect to private GitHub repositories
using GitHub App authentication via direct REST API calls.

To run these tests:
1. Set up a GitHub App (see instructions in GITHUB_APP_SETUP.md)
2. Set environment variables:
   - CODEEVOLVER_GITHUB_APP_ID
   - CODEEVOLVER_GITHUB_APP_PRIVATE_KEY (PEM format or base64 encoded)
3. Create a test private repository
4. Install the GitHub App on that repository
5. Get the installation ID
6. Run: pytest tests/test_github_app_auth.py -v

For integration tests with real GitHub:
- Use a test private repository
- Set GITHUB_TEST_INSTALLATION_ID environment variable
- Set GITHUB_TEST_REPO_URL environment variable
"""

import os
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import httpx

from src.services.git_service import GitService
from src.services.github_app import GitHubAppService
from src.config import settings


class TestGitHubAppService:
    """Unit tests for GitHub App authentication service."""

    def test_get_authenticated_repo_url_https(self):
        """Test converting HTTPS repo URL to authenticated format."""
        token = "test_token_12345"
        repo_url = "https://github.com/user/repo"
        
        authenticated = GitHubAppService.get_authenticated_repo_url(repo_url, token)
        
        assert token in authenticated
        assert "github.com" in authenticated
        assert authenticated.startswith("https://")
        assert "x-access-token" in authenticated
        assert "@github.com" in authenticated

    def test_get_authenticated_repo_url_ssh(self):
        """Test converting SSH repo URL to authenticated HTTPS format."""
        token = "test_token_12345"
        repo_url = "git@github.com:user/repo.git"
        
        authenticated = GitHubAppService.get_authenticated_repo_url(repo_url, token)
        
        assert token in authenticated
        assert "github.com" in authenticated
        assert authenticated.startswith("https://")
        assert "x-access-token" in authenticated

    def test_get_authenticated_repo_url_with_git_suffix(self):
        """Test handling repo URLs with .git suffix."""
        token = "test_token_12345"
        repo_url = "https://github.com/user/repo.git"
        
        authenticated = GitHubAppService.get_authenticated_repo_url(repo_url, token)
        
        assert token in authenticated
        assert ".git" in authenticated or authenticated.endswith(".git")
        assert "x-access-token" in authenticated

    @patch('src.services.github_app.httpx.Client')
    @patch('src.services.github_app.jwt.encode')
    def test_get_installation_token_success(self, mock_jwt_encode, mock_client_class):
        """Test successful token generation via REST API."""
        # Mock JWT generation
        mock_jwt_encode.return_value = "mock_jwt_token"
        
        # Mock HTTP client and response
        mock_response = Mock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "token": "test_installation_token",
            "expires_at": "2026-01-16T12:34:56Z"
        }
        mock_response.raise_for_status = Mock()
        
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        # Set up config
        with patch.object(settings, 'github_app_id', '12345'):
            with patch.object(settings, 'github_app_private_key', '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----'):
                token = GitHubAppService.get_installation_token(123456)
                
                assert token == "test_installation_token"
                mock_client.post.assert_called_once()
                call_args = mock_client.post.call_args
                assert "app/installations/123456/access_tokens" in call_args[0][0]
                assert "Bearer mock_jwt_token" in call_args[1]["headers"]["Authorization"]

    def test_get_installation_token_missing_credentials(self):
        """Test error when GitHub App credentials are missing."""
        with patch.object(settings, 'github_app_id', None):
            with patch.object(settings, 'github_app_private_key', None):
                with pytest.raises(ValueError, match="GitHub App credentials not configured"):
                    GitHubAppService.get_installation_token(123456)

    @patch('src.services.github_app.httpx.Client')
    @patch('src.services.github_app.jwt.encode')
    def test_get_installation_token_404_error(self, mock_jwt_encode, mock_client_class):
        """Test handling 404 (installation not found) error."""
        mock_jwt_encode.return_value = "mock_jwt_token"
        
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = "Not Found"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Not Found", request=Mock(), response=mock_response
        )
        
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        with patch.object(settings, 'github_app_id', '12345'):
            with patch.object(settings, 'github_app_private_key', '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----'):
                with pytest.raises(ValueError, match="GitHub App installation not found"):
                    GitHubAppService.get_installation_token(123456)

    @patch('src.services.github_app.httpx.Client')
    @patch('src.services.github_app.jwt.encode')
    def test_get_installation_token_401_error(self, mock_jwt_encode, mock_client_class):
        """Test handling 401 (authentication failed) error."""
        mock_jwt_encode.return_value = "mock_jwt_token"
        
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.text = "Unauthorized"
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Unauthorized", request=Mock(), response=mock_response
        )
        
        mock_client = Mock()
        mock_client.__enter__ = Mock(return_value=mock_client)
        mock_client.__exit__ = Mock(return_value=None)
        mock_client.post.return_value = mock_response
        mock_client_class.return_value = mock_client

        with patch.object(settings, 'github_app_id', '12345'):
            with patch.object(settings, 'github_app_private_key', '-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----'):
                with pytest.raises(ValueError, match="GitHub App authentication failed"):
                    GitHubAppService.get_installation_token(123456)


class TestGitServiceWithAuth:
    """Tests for GitService with GitHub App authentication."""

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace directory for tests."""
        workspace = tmp_path / "workspaces"
        workspace.mkdir()
        original_root = settings.workspace_root
        settings.workspace_root = str(workspace)
        yield workspace
        settings.workspace_root = original_root

    @patch('src.services.git_service.GitHubAppService.get_installation_token')
    @patch('src.services.git_service.GitHubAppService.get_authenticated_repo_url')
    @patch('src.services.git_service.Repo.clone_from')
    def test_clone_repository_with_auth(
        self,
        mock_clone,
        mock_get_auth_url,
        mock_get_token,
        temp_workspace,
    ):
        """Test cloning a repository with GitHub App authentication."""
        mock_get_token.return_value = "test_token_12345"
        mock_get_auth_url.return_value = "https://x-access-token:test_token_12345@github.com/user/repo.git"
        
        client_id = GitService.generate_client_id()
        repo_url = "https://github.com/user/repo"
        installation_id = 123456

        result = GitService.clone_repository(repo_url, client_id, installation_id)

        mock_get_token.assert_called_once_with(installation_id)
        mock_get_auth_url.assert_called_once_with(repo_url, "test_token_12345")
        mock_clone.assert_called_once()
        
        # Verify the authenticated URL was used
        call_args = mock_clone.call_args[0]
        assert "x-access-token" in call_args[0]

    @patch('src.services.git_service.Repo.clone_from')
    def test_clone_repository_public(self, mock_clone, temp_workspace):
        """Test cloning a public repository without authentication."""
        client_id = GitService.generate_client_id()
        repo_url = "https://github.com/user/repo"

        GitService.clone_repository(repo_url, client_id, installation_id=None)

        # Should clone directly without authentication
        mock_clone.assert_called_once()
        call_args = mock_clone.call_args[0]
        assert call_args[0] == repo_url

    @patch('src.services.git_service.GitHubAppService.get_installation_token')
    def test_clone_repository_auth_failure(self, mock_get_token, temp_workspace):
        """Test handling authentication failures."""
        mock_get_token.side_effect = ValueError("Authentication failed")
        
        client_id = GitService.generate_client_id()
        repo_url = "https://github.com/user/repo"
        installation_id = 123456

        with pytest.raises(ValueError, match="GitHub App authentication failed"):
            GitService.clone_repository(repo_url, client_id, installation_id)


@pytest.mark.integration
class TestGitHubAppIntegration:
    """Integration tests for GitHub App authentication.
    
    These tests require:
    - A GitHub App with ID and private key
    - A test private repository
    - The GitHub App installed on that repository
    - Environment variables set:
      - CODEEVOLVER_GITHUB_APP_ID
      - CODEEVOLVER_GITHUB_APP_PRIVATE_KEY
      - GITHUB_TEST_INSTALLATION_ID (optional, for full integration test)
      - GITHUB_TEST_REPO_URL (optional, for full integration test)
    """

    @pytest.fixture
    def temp_workspace(self, tmp_path):
        """Create a temporary workspace directory for tests."""
        workspace = tmp_path / "workspaces"
        workspace.mkdir()
        original_root = settings.workspace_root
        settings.workspace_root = str(workspace)
        yield workspace
        settings.workspace_root = original_root

    @pytest.mark.skipif(
        not os.getenv("CODEEVOLVER_GITHUB_APP_ID") or not os.getenv("CODEEVOLVER_GITHUB_APP_PRIVATE_KEY"),
        reason="GitHub App credentials not configured"
    )
    def test_get_installation_token_real(self):
        """Test getting a real installation token from GitHub via REST API."""
        installation_id = int(os.getenv("GITHUB_TEST_INSTALLATION_ID", "0"))
        
        if installation_id == 0:
            pytest.skip("GITHUB_TEST_INSTALLATION_ID not set")

        token = GitHubAppService.get_installation_token(installation_id)
        
        assert token is not None
        assert len(token) > 0
        assert isinstance(token, str)

    @pytest.mark.skipif(
        not os.getenv("CODEEVOLVER_GITHUB_APP_ID") or not os.getenv("CODEEVOLVER_GITHUB_APP_PRIVATE_KEY"),
        reason="GitHub App credentials not configured"
    )
    @pytest.mark.skipif(
        not os.getenv("GITHUB_TEST_INSTALLATION_ID") or not os.getenv("GITHUB_TEST_REPO_URL"),
        reason="Test repository not configured"
    )
    def test_clone_private_repo_real(self, temp_workspace):
        """Test cloning a real private repository with GitHub App authentication.
        
        This is a full integration test that:
        1. Authenticates with GitHub App via REST API
        2. Clones a private repository
        3. Verifies the repository was cloned successfully
        """
        installation_id = int(os.getenv("GITHUB_TEST_INSTALLATION_ID"))
        repo_url = os.getenv("GITHUB_TEST_REPO_URL")
        
        client_id = GitService.generate_client_id()

        try:
            main_path = GitService.clone_repository(
                repo_url,
                client_id,
                installation_id=installation_id,
            )

            assert main_path.exists(), "Repository should be cloned"
            assert (main_path / ".git").exists(), "Should be a git repository"
            
            # Verify we can access the repository
            from git import Repo
            repo = Repo(main_path)
            assert repo.remotes.origin.url is not None

        finally:
            # Cleanup
            GitService.cleanup_workspace(client_id)

    @pytest.mark.skipif(
        not os.getenv("CODEEVOLVER_GITHUB_APP_ID") or not os.getenv("CODEEVOLVER_GITHUB_APP_PRIVATE_KEY"),
        reason="GitHub App credentials not configured"
    )
    @pytest.mark.skipif(
        not os.getenv("GITHUB_TEST_INSTALLATION_ID") or not os.getenv("GITHUB_TEST_REPO_URL"),
        reason="Test repository not configured"
    )
    def test_verify_installation_access(self):
        """Test verifying installation access to a repository via REST API."""
        installation_id = int(os.getenv("GITHUB_TEST_INSTALLATION_ID"))
        repo_url = os.getenv("GITHUB_TEST_REPO_URL")
        
        has_access = GitHubAppService.verify_installation_access(installation_id, repo_url)
        
        assert has_access is True, "Installation should have access to the test repository"


class TestConfigWithGitHubApp:
    """Tests for configuration with GitHub App settings."""

    def test_base64_private_key_decoding(self):
        """Test that base64 encoded private keys are decoded correctly."""
        import base64
        
        # Create a fake PEM key
        fake_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----"
        encoded_key = base64.b64encode(fake_key.encode()).decode()
        
        # Test that Settings decodes it
        with patch.dict(os.environ, {
            'CODEEVOLVER_GITHUB_APP_ID': '12345',
            'CODEEVOLVER_GITHUB_APP_PRIVATE_KEY': encoded_key,
        }, clear=False):
            # Reload settings - need to clear module cache to get new instance
            import importlib
            import src.config
            importlib.reload(src.config)
            test_settings = src.config.Settings()
            
            # Should be decoded
            assert test_settings.github_app_private_key == fake_key

    def test_raw_private_key_passthrough(self):
        """Test that raw PEM keys are passed through without decoding."""
        fake_key = "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC\n-----END PRIVATE KEY-----"
        
        with patch.dict(os.environ, {
            'CODEEVOLVER_GITHUB_APP_ID': '12345',
            'CODEEVOLVER_GITHUB_APP_PRIVATE_KEY': fake_key,
        }, clear=False):
            import importlib
            import src.config
            importlib.reload(src.config)
            test_settings = src.config.Settings()
            
            # Should remain as-is
            assert test_settings.github_app_private_key == fake_key
