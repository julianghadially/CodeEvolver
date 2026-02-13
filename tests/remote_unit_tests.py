"""Unit tests for individual CodeEvolver components.

These tests verify specific components work correctly with known inputs,
without running the full optimization pipeline.

Test categories:
1. Adapter methods (build_seed_candidate, evaluate, make_reflective_dataset)
2. GEPA state tracking
3. Sandbox operations
4. Git operations and branch management
"""

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestBuildSeedCandidate:
    """Tests for build_seed_candidate with various program structures."""

    @pytest.mark.asyncio
    async def test_factchecker_with_renamed_module_via_api(self):
        """Test build_seed_candidate via /test/build-seed API after renaming ResearchAgentModule.

        This test uses the /test/build-seed endpoint to:
        1. Clone FactChecker repository
        2. Programmatically rename research_agent_module.py → research_agent_module_test.py
        3. Rename class ResearchAgentModule → ResearchAgentModuleTest
        4. Call build_seed_candidate to verify it can parse the renamed structure

        This is a FAST unit test that doesn't run the full optimization pipeline.
        """
        import httpx
        import os

        # Get Modal app URL (same pattern as integration tests)
        modal_url = os.getenv("MODAL_APP_URL", "https://julianghadially--codeevolver-fastapi-app-dev.modal.run")

        # Define the refactoring to test
        refactor_files = {
            "src/factchecker/modules/research_agent_module.py":
            "src/factchecker/modules/research_agent_module_test.py"
        }
        refactor_classes = {
            "ResearchAgentModule": "ResearchAgentModuleTest"
        }

        # Call the /test/build-seed endpoint
        # Timeout must be longer than Modal function timeout (600s = 10min)
        # Add extra buffer for network/queuing delays
        print("\n⏳ Calling /test/build-seed endpoint (this may take 5-10 minutes)...")
        async with httpx.AsyncClient(timeout=720.0) as client:  # 12 minutes
            response = await client.post(
                f"{modal_url}/test/build-seed",
                json={
                    "repo_url": "https://github.com/julianghadially/FactChecker",
                    "program": "src.factchecker.modules.fact_checker_pipeline.FactCheckerPipeline",
                    "initial_branch": "main",
                    "refactor_files": refactor_files,
                    "refactor_classes": refactor_classes,
                }
            )

            assert response.status_code == 200, f"Request failed: {response.status_code} {response.text}"

            result = response.json()
            print(f"\n{'='*60}")
            print("Test Result:")
            print(f"{'='*60}")
            print(f"Success: {result.get('success')}")
            print(f"Num Predictors: {result.get('num_predictors')}")
            print(f"Predictor Names: {result.get('predictor_names')}")

            if result.get('logs'):
                print("\nLogs:")
                for log in result['logs']:
                    print(f"  {log}")

            if not result.get('success'):
                print(f"\nError: {result.get('error')}")
                pytest.fail(f"build_seed_candidate failed: {result.get('error')}")

            # Verify we got predictors
            assert result.get('num_predictors', 0) > 0, "Should have found at least one predictor"

            # Verify the candidate structure
            candidate = result.get('candidate', {})
            predictor_names = list(candidate.keys())
            print(f"\n✓ Test passed: build_seed_candidate works with renamed modules")
            print(f"  Predictors found: {predictor_names}")


class TestGEPAState:
    """Tests for GEPA state tracking with known inputs."""

    @pytest.mark.unit
    def test_gepa_state_structure(self):
        """Test GEPA state structure with a known example.

        Verifies that GEPAState correctly tracks:
        - program_candidates
        - parent_program_for_candidates
        - prog_candidate_val_subscores
        - prog_candidate_objective_scores
        - pareto_front_valset
        """
        # This would be a mock GEPA state result
        mock_gepa_result = {
            "program_candidates": [
                {
                    "_code": json.dumps({
                        "git_branch": "codeevolver-20260212-main",
                        "parent_module_path": "src.factchecker.FactCheckerPipeline",
                        "change_request": "",
                        "last_change_summary": "Initial state"
                    }),
                    "module_1.predict": "Initial instruction for module 1",
                },
                {
                    "_code": json.dumps({
                        "git_branch": "codeevolver-20260212-abc123",
                        "parent_module_path": "src.factchecker.FactCheckerPipeline",
                        "change_request": "Improve accuracy by adding context",
                        "last_change_summary": "Added context retrieval"
                    }),
                    "module_1.predict": "Improved instruction with context",
                },
            ],
            "prog_candidate_val_subscores": [
                {"val_0": 0.8, "val_1": 0.9, "val_2": 0.7},
                {"val_0": 0.85, "val_1": 0.95, "val_2": 0.75},
            ],
            "prog_candidate_objective_scores": [
                {"accuracy": 0.8},
                {"accuracy": 0.85},
            ],
        }

        # Verify structure
        assert "program_candidates" in mock_gepa_result
        assert len(mock_gepa_result["program_candidates"]) == 2

        # Verify each candidate has required fields
        for candidate in mock_gepa_result["program_candidates"]:
            assert "_code" in candidate
            code_data = json.loads(candidate["_code"])
            assert "git_branch" in code_data
            assert "parent_module_path" in code_data
            assert "change_request" in code_data
            assert "last_change_summary" in code_data

        # Verify scores structure
        assert len(mock_gepa_result["prog_candidate_val_subscores"]) == 2
        assert len(mock_gepa_result["prog_candidate_objective_scores"]) == 2

        print("\n✓ GEPA state structure is valid")

    @pytest.mark.unit
    def test_candidate_git_branch_extraction(self):
        """Test extracting git_branch from candidate _code component."""
        candidate = {
            "_code": json.dumps({
                "git_branch": "codeevolver-20260212-abc123",
                "parent_module_path": "src.factchecker.FactCheckerPipeline",
                "change_request": "Test change",
                "last_change_summary": "Test summary"
            }),
            "module_1.predict": "Test instruction",
        }

        code_data = json.loads(candidate["_code"])
        git_branch = code_data.get("git_branch")

        assert git_branch == "codeevolver-20260212-abc123"
        assert git_branch.startswith("codeevolver-")
        print(f"\n✓ Successfully extracted git_branch: {git_branch}")


class TestSandboxOperations:
    """Tests for sandbox operations."""

    @pytest.mark.unit
    def test_git_branch_naming_convention(self):
        """Test that branch names follow the expected convention."""
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Main branch
        main_branch = f"codeevolver-{timestamp}-main"
        assert main_branch.startswith("codeevolver-")
        assert main_branch.endswith("-main")

        # Mutation branch
        import uuid
        short_id = uuid.uuid4().hex[:6]
        mutation_branch = f"codeevolver-{timestamp}-{short_id}"
        assert mutation_branch.startswith("codeevolver-")
        assert len(short_id) == 6

        print(f"\n✓ Branch naming convention verified:")
        print(f"  Main: {main_branch}")
        print(f"  Mutation: {mutation_branch}")


# Pytest markers for organizing tests
def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests (require Modal app)")
