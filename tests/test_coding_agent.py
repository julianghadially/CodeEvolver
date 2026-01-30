"""Integration test for the coding agent executing code mutations.

This test verifies that the coding agent can:
1. Clone a repository
2. Create a branch
3. Apply a code mutation (remove evidence_stance field)
4. Push to remote
5. Verify the change was made correctly

To run this test:
1. In one terminal, run: modal serve modal_app.py
2. Set environment variable: GITHUB_TEST_INSTALLATION_ID=104592180
3. In another terminal, run: pytest tests/test_coding_agent.py -v -s -m integration

The test calls the Modal app's /change_request HTTP endpoint.
Watch the 'modal serve' terminal to see live logs.

Modal note:
modal.Function.from_name() only works for deployed apps (via modal deploy), not for modal serve
Direct import gets the function object, but .remote() tries to hydrate against Modal's API which fails for non-deployed apps
Since your FastAPI app is running at https://julianghadially--codeevolver-fastapi-app-dev.modal.run, the correct approach is to call the sandbox function via an HTTP endpoint
"""

import os
from datetime import datetime

import httpx
import pytest

from src.services.git_service import GitService


# Test configuration
TEST_REPO_URL = "https://github.com/julianghadially/FactChecker"
TEST_FILE_PATH = "src/factchecker/signatures/evidence_summarizer.py"
CHANGE_REQUEST = """Remove the evidence_stance field from the evidence summarizer module and signature. Ensure the full pipeline runs.

Specifically:
1. Remove the `evidence_stance: str` field from the EvidenceSummarizerSignature class
2. Remove any references to evidence_stance in the EvidenceSummarizer module
3. Update any code that depends on this field"""

# Modal app URL - set via environment variable or use default for modal serve
# Format: https://<username>--<app-name>-<function-name>-dev.modal.run
DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"


def get_modal_app_url() -> str:
    """Get the Modal app URL from environment or use default."""
    return os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL)


def generate_test_branch_name() -> str:
    """Generate a test branch name with current timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    return f"codeevolver-test-{timestamp}"


@pytest.mark.integration
class TestCodingAgent:
    """Integration tests for the coding agent with real GitHub operations.

    Prerequisites:
    - Run 'modal serve modal_app.py' in another terminal
    - Set GITHUB_TEST_INSTALLATION_ID environment variable
    - Optionally set MODAL_APP_URL if using a deployed app
    """

    @pytest.fixture
    def installation_id(self) -> int:
        """Get the GitHub App installation ID from environment."""
        installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
        if not installation_id:
            pytest.skip("GITHUB_TEST_INSTALLATION_ID not set")
        return int(installation_id)

    @pytest.fixture
    def branch_name(self) -> str:
        """Generate a unique branch name for this test run."""
        return generate_test_branch_name()

    @pytest.fixture
    def modal_url(self) -> str:
        """Get the Modal app URL."""
        return get_modal_app_url()

    @pytest.mark.asyncio
    async def test_coding_agent_removes_evidence_stance(
        self,
        installation_id: int,
        branch_name: str,
        modal_url: str,
    ):
        """
        Test that the coding agent can remove the evidence_stance field.

        This test:
        1. Calls the Modal app's /execute_sandbox endpoint via HTTP
        2. The app creates a sandbox, clones repo, creates branch
        3. Applies the code mutation to remove evidence_stance
        4. Pushes the changes
        5. Fetches the file and verifies evidence_stance is removed
        """
        print(f"\n{'='*60}")
        print(f"Test branch: {branch_name}")
        print(f"Repository: {TEST_REPO_URL}")
        print(f"Installation ID: {installation_id}")
        print(f"Modal App URL: {modal_url}")
        print(f"{'='*60}")
        print(f"\nWATCH THE 'modal serve' TERMINAL FOR LIVE LOGS\n")
        print(f"Change request: {CHANGE_REQUEST[:100]}...")
        print(f"{'='*60}\n")

        # First, check that the Modal app is reachable
        # Note: follow_redirects=True is needed because Modal may return 303 redirects
        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            # Health check
            try:
                health_response = await client.get(f"{modal_url}/health")
                if health_response.status_code != 200:
                    pytest.fail(
                        f"Modal app health check failed!\n"
                        f"URL: {modal_url}/health\n"
                        f"Status: {health_response.status_code}\n"
                        f"Make sure 'modal serve modal_app.py' is running."
                    )
                print(f"✓ Modal app is healthy: {health_response.json()}")
            except httpx.ConnectError as e:
                pytest.fail(
                    f"Cannot connect to Modal app at {modal_url}\n"
                    f"Error: {e}\n\n"
                    f"Make sure 'modal serve modal_app.py' is running in another terminal.\n"
                    f"The URL should match the one shown in the modal serve output."
                )

            # Debug: Check secrets configuration
            debug_response = await client.get(f"{modal_url}/debug_secrets")
            if debug_response.status_code == 200:
                import json
                debug_info = debug_response.json()
                print(f"\n{'='*60}")
                print("DEBUG: Secrets Configuration")
                print(json.dumps(debug_info, indent=2))
                print(f"{'='*60}\n")
            else:
                print(f"Warning: Could not fetch debug_secrets: {debug_response.status_code}")

            # Call the /change_request endpoint
            print("\nCalling /change_request via HTTP...")
            print("(This may take several minutes - watch the modal serve terminal)\n")

            request_payload = {
                "repo_url": TEST_REPO_URL,
                "change_request": CHANGE_REQUEST,
                "installation_id": installation_id,
                "branch_name": branch_name,
                "push_to_remote": True,
            }

            response = await client.post(
                f"{modal_url}/change_request",
                json=request_payload,
            )

            if response.status_code != 200:
                pytest.fail(
                    f"execute_sandbox request failed!\n"
                    f"Status: {response.status_code}\n"
                    f"Response: {response.text}"
                )

            result = response.json()

        print(f"\n{'='*60}")
        print(f"Result success: {result['success']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("branch_name"):
            print(f"Branch: {result['branch_name']}")
        if result.get("output"):
            print(f"Output (truncated): {result['output'][:500]}...")
        print(f"{'='*60}\n")

        # Check that the mutation succeeded
        assert result["success"], f"Mutation failed: {result.get('error')}"
        assert result["branch_name"] == branch_name

        # Fetch the file from the new branch and verify evidence_stance is removed
        print(f"Fetching file from branch {branch_name} to verify changes...")
        file_content = GitService.fetch_github_file(
            repo_url=TEST_REPO_URL,
            file_path=TEST_FILE_PATH,
            branch=branch_name,
            installation_id=installation_id,
        )

        print(f"File fetched ({len(file_content)} characters)")

        # Verify evidence_stance is not present
        assert "evidence_stance: str" not in file_content, (
            f"evidence_stance: str should have been removed from the file. "
            f"File still contains this field."
        )

        print(f"\n{'='*60}")
        print("TEST PASSED: evidence_stance field was successfully removed!")
        print(f"Check the branch at: {TEST_REPO_URL}/tree/{branch_name}")
        print(f"{'='*60}\n")


@pytest.mark.integration
class TestFetchGitHubFile:
    """Tests for the fetch_github_file utility function."""

    @pytest.fixture
    def installation_id(self) -> int:
        """Get the GitHub App installation ID from environment."""
        installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
        if not installation_id:
            pytest.skip("GITHUB_TEST_INSTALLATION_ID not set")
        return int(installation_id)

    def test_fetch_github_file_from_main(self, installation_id: int):
        """Test fetching a file from the main branch."""
        content = GitService.fetch_github_file(
            repo_url=TEST_REPO_URL,
            file_path=TEST_FILE_PATH,
            branch="main",
            installation_id=installation_id,
        )

        # The main branch should still have evidence_stance
        assert len(content) > 0
        assert "evidence_stance: str" in content, (
            "Main branch should contain evidence_stance: str"
        )
        print(f"\nFetched file from main branch ({len(content)} chars)")

    def test_fetch_github_file_from_test_branch(self, installation_id: int):
        """
        Test fetching a file from a codeevolver-tests branch.

        This verifies the file on a branch where the change has already been made.
        Use this test to verify previous test runs worked correctly.
        """
        try:
            content = GitService.fetch_github_file(
                repo_url=TEST_REPO_URL,
                file_path=TEST_FILE_PATH,
                branch="codeevolver-tests",
                installation_id=installation_id,
            )

            # The test branch should NOT have evidence_stance
            assert "evidence_stance: str" not in content, (
                "codeevolver-tests branch should NOT contain evidence_stance: str"
            )
            print(f"\nFetched file from codeevolver-tests branch ({len(content)} chars)")
            print("Verified: evidence_stance has been removed")
        except ValueError as e:
            if "not found" in str(e).lower():
                pytest.skip("codeevolver-tests branch does not exist yet")
            raise
