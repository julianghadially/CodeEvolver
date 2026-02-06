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
import time

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
        start_time = time.time()

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

        print(f"Test duration: {(time.time() - start_time)/60:.2f} minutes")
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
        Test fetching a file from the most recent codeevolver-test-* branch.

        This verifies the file on a branch where the change has already been made.
        Use this test to verify previous test runs worked correctly.
        Finds the most recent branch by sorting alphanumerically (format: YYYYMMDDHHMM).
        """
        # Find all codeevolver-test-* branches and pick the most recent
        branches = GitService.list_branches(
            repo_url=TEST_REPO_URL,
            prefix="codeevolver-test-",
            installation_id=installation_id,
        )

        if not branches:
            pytest.skip("No codeevolver-test-* branches exist yet")

        # Pick the most recent (last alphanumerically due to YYYYMMDDHHMM format)
        latest_branch = branches[-1]
        print(f"\nFound {len(branches)} test branches, using most recent: {latest_branch}")

        try:
            content = GitService.fetch_github_file(
                repo_url=TEST_REPO_URL,
                file_path=TEST_FILE_PATH,
                branch=latest_branch,
                installation_id=installation_id,
            )

            # The test branch should NOT have evidence_stance
            assert "evidence_stance: str" not in content, (
                f"{latest_branch} branch should NOT contain evidence_stance: str"
            )
            print(f"Fetched file from {latest_branch} ({len(content)} chars)")
            print("Verified: evidence_stance has been removed")
        except ValueError as e:
            if "not found" in str(e).lower():
                pytest.skip(f"File not found on branch {latest_branch}")
            raise


# Change request for the plan mode test - a large multi-file change
PLAN_MODE_CHANGE_REQUEST = """Please plan your changes before making them. This is a complex multi-file change.

Add a new ResearchModule before the JudgeModule and wrap it in a pipeline module called FactCheckerPipeline. The pipeline should perform web research to gather evidence
before judging statements. The module should:

1. Create a new files src/factchecker/modules/research_module.py and src/factchecker/signatures/research_signature.py containing:
   - A ResearchSignature (dspy.Signature) with inputs: statement (str), topic (str)
     and outputs: search_queries (list[str]) - generates 2-3 targeted search queries
   - A ResearchModule (dspy.Module) that:
     a. Uses dspy.ChainOfThought(ResearchSignature) to generate search queries
     b. Has a forward() method that takes statement and topic, generates queries,
        and returns a dspy.Prediction with fields: queries (list), evidence_summary (str)
     c. For now, the evidence_summary can be a placeholder that concatenates the queries
        (actual SERPER/Firecrawl integration will be added later)

2. Create a new file src/factchecker/signatures/research_strategy.py containing:
   - A ResearchStrategy (dspy.Signature) that defines the input/output contract:
     inputs: statement, topic
     outputs: search_queries, reasoning

3. Wrap the judge, module and research modules into a pipeline, called FactCheckerPipeline, as your best seat fit?
4. Update codeevolver.md if it exists to reflect the new module in the architecture

Keep the implementation clean and consistent with existing module patterns in the codebase.

Again, Please plan your changes before making them, using plan mode."""

# Directory to check for new files after the plan mode test
MODULES_DIR = "src/factchecker/modules"
SIGNATURES_DIR = "src/factchecker/signatures"


@pytest.mark.integration
class TestPlanModeExecution:
    """Integration test for autonomous plan mode execution.

    Tests that the user proxy correctly auto-approves plan mode,
    allowing the agent to plan and execute a complex multi-file change.

    Uses the 'simple' branch as the starting point and creates a new
    test branch for changes (does NOT modify the 'simple' branch).

    Prerequisites:
    - Run 'modal serve modal_app.py' in another terminal
    - Set GITHUB_TEST_INSTALLATION_ID environment variable
    """

    @pytest.fixture
    def installation_id(self) -> int:
        installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
        if not installation_id:
            pytest.skip("GITHUB_TEST_INSTALLATION_ID not set")
        return int(installation_id)

    @pytest.fixture
    def branch_name(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        return f"codeevolver-test-plan-{timestamp}"

    @pytest.fixture
    def modal_url(self) -> str:
        return get_modal_app_url()

    @pytest.mark.asyncio
    async def test_plan_mode_adds_research_module(
        self,
        installation_id: int,
        branch_name: str,
        modal_url: str,
    ):
        """
        Test that the agent can use plan mode to add a research module.

        This test:
        1. Clones from the 'simple' branch (initial_branch)
        2. Creates a new test branch (does NOT modify 'simple')
        3. Sends a complex change request that encourages plan mode
        4. Verifies new files were created in modules/ and/or signatures/
        """
        print(f"\n{'='*60}")
        print(f"PLAN MODE TEST")
        print(f"Initial branch: simple")
        print(f"Test branch: {branch_name}")
        print(f"Repository: {TEST_REPO_URL}")
        print(f"{'='*60}\n")
        start_time = time.time()

        # Baseline: count files on the 'simple' branch before changes
        try:
            modules_before = GitService.list_directory_files(
                repo_url=TEST_REPO_URL,
                directory_path=MODULES_DIR,
                branch="simple",
                installation_id=installation_id,
            )
        except ValueError:
            modules_before = []

        try:
            signatures_before = GitService.list_directory_files(
                repo_url=TEST_REPO_URL,
                directory_path=SIGNATURES_DIR,
                branch="simple",
                installation_id=installation_id,
            )
        except ValueError:
            signatures_before = []

        total_before = len(modules_before) + len(signatures_before)
        print(f"Baseline file count on 'simple' branch:")
        print(f"  {MODULES_DIR}/: {len(modules_before)} files - {modules_before}")
        print(f"  {SIGNATURES_DIR}/: {len(signatures_before)} files - {signatures_before}")
        print(f"  Total: {total_before}")

        async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
            # Health check
            try:
                health_response = await client.get(f"{modal_url}/health")
                assert health_response.status_code == 200, (
                    f"Modal app health check failed (status {health_response.status_code}). "
                    f"Make sure 'modal serve modal_app.py' is running."
                )
            except httpx.ConnectError as e:
                pytest.fail(
                    f"Cannot connect to Modal app at {modal_url}. "
                    f"Make sure 'modal serve modal_app.py' is running.\nError: {e}"
                )

            print(f"\nSending change request (plan mode encouraged)...")
            print(f"Change request: {PLAN_MODE_CHANGE_REQUEST[:150]}...")

            request_payload = {
                "repo_url": TEST_REPO_URL,
                "change_request": PLAN_MODE_CHANGE_REQUEST,
                "installation_id": installation_id,
                "branch_name": branch_name,
                "push_to_remote": True,
                "initial_branch": "simple",
            }

            response = await client.post(
                f"{modal_url}/change_request",
                json=request_payload,
            )

            if response.status_code != 200:
                pytest.fail(
                    f"Change request failed!\n"
                    f"Status: {response.status_code}\n"
                    f"Response: {response.text}"
                )

            result = response.json()

        elapsed = (time.time() - start_time) / 60
        print(f"\n{'='*60}")
        print(f"Completed in {elapsed:.2f} minutes")
        print(f"Result success: {result['success']}")
        if result.get("error"):
            print(f"Error: {result['error']}")
        if result.get("output"):
            # Check if plan mode was used
            output = result["output"]
            if "EnterPlanMode" in output:
                print("Plan mode: USED (EnterPlanMode detected in output)")
            if "User proxy: Auto-approving ExitPlanMode" in output:
                print("User proxy: WORKING (auto-approved ExitPlanMode)")
            print(f"Output (truncated): {output[:800]}...")
        print(f"{'='*60}\n")

        assert result["success"], f"Agent failed: {result.get('error')}"
        assert result["branch_name"] == branch_name

        # Verify new files were created by comparing file counts
        # The test branch should have MORE files than the simple branch
        try:
            modules_after = GitService.list_directory_files(
                repo_url=TEST_REPO_URL,
                directory_path=MODULES_DIR,
                branch=branch_name,
                installation_id=installation_id,
            )
        except ValueError:
            modules_after = []

        try:
            signatures_after = GitService.list_directory_files(
                repo_url=TEST_REPO_URL,
                directory_path=SIGNATURES_DIR,
                branch=branch_name,
                installation_id=installation_id,
            )
        except ValueError:
            signatures_after = []

        total_after = len(modules_after) + len(signatures_after)

        print(f"File count on test branch '{branch_name}':")
        print(f"  {MODULES_DIR}/: {len(modules_after)} files - {modules_after}")
        print(f"  {SIGNATURES_DIR}/: {len(signatures_after)} files - {signatures_after}")
        print(f"  Total: {total_after}")

        new_module_files = set(modules_after) - set(modules_before)
        new_signature_files = set(signatures_after) - set(signatures_before)
        print(f"\nNew files created:")
        print(f"  Modules: {new_module_files or 'none'}")
        print(f"  Signatures: {new_signature_files or 'none'}")

        assert total_after > total_before, (
            f"Expected new files to be created. "
            f"Before: {total_before} files ({modules_before} + {signatures_before}), "
            f"After: {total_after} files ({modules_after} + {signatures_after}). "
            f"The agent did not create any new files in {MODULES_DIR}/ or {SIGNATURES_DIR}/."
        )

        print(f"\n{'='*60}")
        print(f"TEST PASSED: Agent created {total_after - total_before} new file(s)!")
        print(f"Check the branch at: {TEST_REPO_URL}/tree/{branch_name}")
        print(f"{'='*60}\n")
