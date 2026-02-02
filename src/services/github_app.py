"""GitHub App authentication service for private repository access.

Uses direct GitHub REST API calls instead of PyGithub to avoid LGPL licensing issues.
All dependencies are MIT/Apache-2.0 licensed.
"""

import time
from typing import Optional

import httpx
import jwt

from ..config import settings


class GitHubAppService:
    """Service for authenticating with GitHub App and generating installation tokens.

    Uses direct REST API calls to GitHub's API, avoiding PyGithub's LGPL license.
    """

    GITHUB_API_BASE = "https://api.github.com"

    @staticmethod
    def _generate_jwt() -> str:
        """
        Generate a JWT token for GitHub App authentication.

        Returns:
            JWT token string

        Raises:
            ValueError: If GitHub App credentials are not configured
        """
        app_id = settings.github_app_id
        private_key = settings.github_app_private_key

        if not app_id or not private_key:
            raise ValueError(
                "GitHub App credentials not configured. "
                "Set CODEEVOLVER_GITHUB_APP_ID and CODEEVOLVER_GITHUB_APP_PRIVATE_KEY"
            )

        # Validate private key format before attempting JWT generation
        if not private_key.strip().startswith("-----BEGIN"):
            raise ValueError(
                "Invalid private key format. Expected PEM format starting with "
                "'-----BEGIN PRIVATE KEY-----' or '-----BEGIN RSA PRIVATE KEY-----'. "
                "If your key is base64 encoded, ensure CODEEVOLVER_GITHUB_APP_PRIVATE_KEY "
                "contains the base64-encoded PEM key."
            )

        try:
            # JWT payload for GitHub App
            now = int(time.time())
            payload = {
                "iat": now - 60,  # Issued at time (60 seconds in the past to allow for clock skew)
                "exp": now + (10 * 60),  # JWT expiration time (10 minutes maximum)
                "iss": app_id,  # GitHub App's identifier
            }

            # Generate JWT using RS256 algorithm
            # Wrap in try/except to prevent private key from appearing in traceback
            # Note: The key may still appear in jwt library's internal traceback,
            # but we prevent it from appearing in our error messages
            try:
                token = jwt.encode(
                    payload,
                    private_key,
                    algorithm="RS256",
                )
            except jwt.exceptions.InvalidKeyError:
                # Raise a new exception without chaining to prevent key exposure
                raise ValueError(
                    "Invalid private key format. The key could not be parsed. "
                    "Ensure it's a valid RSA private key in PEM format."
                ) from None
            except Exception as e:
                # Catch any other JWT-related exceptions
                raise ValueError(
                    f"Failed to generate JWT token: {type(e).__name__}: {e}"
                ) from None

            return token

        except ValueError:
            # Re-raise ValueError as-is (these are our sanitized errors)
            raise
        except Exception:
            # Catch any unexpected exceptions and sanitize
            raise ValueError(
                "Failed to generate JWT token. Check that your GitHub App private key "
                "is correctly formatted and base64 decoded if needed."
            ) from None

    @staticmethod
    def get_installation_token(installation_id: int) -> Optional[str]:
        """
        Generate an installation access token for a GitHub App installation.

        Uses direct REST API call to GitHub:
        POST /app/installations/{installation_id}/access_tokens

        Args:
            installation_id: The GitHub App installation ID

        Returns:
            Installation access token, or None if authentication fails

        Raises:
            ValueError: If GitHub App credentials are not configured or API call fails
        """
        try:
            # Generate JWT for app authentication
            jwt_token = GitHubAppService._generate_jwt()

            # Request installation token from GitHub API
            url = f"{GitHubAppService.GITHUB_API_BASE}/app/installations/{installation_id}/access_tokens"

            with httpx.Client() as client:
                response = client.post(
                    url,
                    headers={
                        "Authorization": f"Bearer {jwt_token}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    timeout=30.0,
                )

                response.raise_for_status()
                data = response.json()
                return data.get("token")

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(
                    f"GitHub App installation not found: {installation_id}. "
                    "Verify the installation ID and that the app is installed on the repository."
                ) from e
            elif e.response.status_code == 401:
                raise ValueError(
                    "GitHub App authentication failed. Check your App ID and private key."
                ) from e
            else:
                raise ValueError(
                    f"Failed to generate installation token: HTTP {e.response.status_code} - {e.response.text}"
                ) from e
        except httpx.RequestError as e:
            raise ValueError(f"Failed to connect to GitHub API: {e}") from e
        except Exception as e:
            raise ValueError(f"Failed to generate installation token: {e}") from e

    @staticmethod
    def get_authenticated_repo_url(repo_url: str, token: str) -> str:
        """
        Convert a repository URL to use token authentication.

        Args:
            repo_url: Original repository URL (https://github.com/user/repo)
            token: GitHub installation token

        Returns:
            Authenticated repository URL
        """
        # Handle different URL formats
        if repo_url.startswith("https://github.com/"):
            # Extract owner/repo from URL
            parts = repo_url.replace("https://github.com/", "").rstrip("/").split("/")
            if len(parts) >= 2:
                owner = parts[0]
                repo = parts[1].replace(".git", "")
                # Use x-access-token format for GitHub App tokens
                return f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"

        elif repo_url.startswith("git@github.com:"):
            # SSH URL - convert to HTTPS with token
            parts = repo_url.replace("git@github.com:", "").rstrip("/").replace(".git", "")
            if "/" in parts:
                return f"https://x-access-token:{token}@github.com/{parts}.git"

        # If we can't parse it, try appending token to existing URL
        if "://" in repo_url and "@" not in repo_url:
            # Insert token before domain
            if repo_url.startswith("https://"):
                return repo_url.replace("https://", f"https://x-access-token:{token}@")
            elif repo_url.startswith("http://"):
                return repo_url.replace("http://", f"http://x-access-token:{token}@")

        raise ValueError(f"Unable to parse repository URL: {repo_url}")

    @staticmethod
    def verify_installation_access(installation_id: int, repo_url: str) -> bool:
        """
        Verify that the GitHub App installation has access to the repository.

        Uses direct REST API call to check repository access.

        Args:
            installation_id: The GitHub App installation ID
            repo_url: Repository URL to verify access for

        Returns:
            True if installation has access, False otherwise
        """
        try:
            token = GitHubAppService.get_installation_token(installation_id)
            if not token:
                return False

            # Extract owner/repo from URL
            parts = repo_url.replace("https://github.com/", "").replace("git@github.com:", "").rstrip("/").replace(".git", "").split("/")
            if len(parts) < 2:
                return False

            owner = parts[0]
            repo_name = parts[1]

            # Try to access the repository via REST API
            url = f"{GitHubAppService.GITHUB_API_BASE}/repos/{owner}/{repo_name}"

            with httpx.Client() as client:
                response = client.get(
                    url,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Accept": "application/vnd.github+json",
                        "X-GitHub-Api-Version": "2022-11-28",
                    },
                    timeout=30.0,
                )

                # 200 means we have access, 404 means no access, other errors are ambiguous
                return response.status_code == 200

        except Exception:
            return False
