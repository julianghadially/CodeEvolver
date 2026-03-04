"""Integration test for /test/dspy-save-program endpoint.

Tests that dspy_save_program can:
1. Accept a GEPA candidate dict (predictor_name -> instruction text)
2. Reconstruct a DSPy program inside a GEPASandbox
3. Save it in DSPy-native format (loadable via program.load())
4. Commit and push the result to the winning branch

To run:
1. Deploy or serve the Modal app:  modal serve modal_app.py
2. Set env:  export GITHUB_TEST_INSTALLATION_ID=104592180
3. Run:  python -m pytest tests/test_dspy_save_program.py -v -s -m integration

Target repository: https://github.com/julianghadially/LangProBe-CodeEvolver
"""

import json
import os
from pathlib import Path

import httpx
import pytest

# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------
REPO_URL = "https://github.com/julianghadially/LangProBe-CodeEvolver"

#standard test
GIT_BRANCH = "codeevolver-20260220211311-435004"
PROGRAM = "langProBe.hover.hover_pipeline.HoverMultiHopPipeline"
CANDIDATE_PATH = Path(__file__).parent.parent / "experiments" / "hover" / "candidate_shortlived.json"
#temp test
GIT_BRANCH = "codeevolver-20260303031536-d01394"
PROGRAM = "langProPlus.hotpotGEPA.hotpot_pipeline.HotpotMultiHopPipeline"
CANDIDATE_PATH = Path(__file__).parent.parent / "experiments" / "hotpotGEPA" / "candidate.json"


def load_prompt_candidate() -> dict[str, str]:
    """Load candidate JSON and strip the _code key (prompt-only)."""
    with open(CANDIDATE_PATH) as f:
        raw = json.load(f)
    return {k: v for k, v in raw.items() if k != "_code"}


def get_installation_id() -> int | None:
    """Get GitHub App installation ID from environment."""
    val = os.getenv("GITHUB_TEST_INSTALLATION_ID")
    return int(val) if val else None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
def test_dspy_save_program(modal_url):
    """Build a DSPy program from a GEPA candidate and push to the winning branch."""
    installation_id = get_installation_id()
    if installation_id is None:
        pytest.skip("GITHUB_TEST_INSTALLATION_ID not set")

    config = {
        "repo_url": REPO_URL,
        "program": PROGRAM,
        "candidate": load_prompt_candidate(),
        "git_branch": GIT_BRANCH,
        "push_to_remote": True,
        "installation_id": installation_id,
    }

    # Sandbox startup + repo clone + venv install takes ~4 minutes.
    # Modal returns 303 redirects every 150s for long-running requests;
    # follow_redirects=True polls until the response is ready.
    with httpx.Client(timeout=900, follow_redirects=True) as client:
        print(f"\nPOST {modal_url}/test/dspy-save-program")
        print(f"Repo: {config['repo_url']}")
        print(f"Branch: {config['git_branch']}")
        print(f"Program: {config['program']}")
        print(f"Installation ID: {installation_id}")
        print(f"Candidate predictors: {list(config['candidate'].keys())}")

        response = client.post(
            f"{modal_url}/test/dspy-save-program",
            json=config,
        )

        assert response.status_code == 200, (
            f"POST /test/dspy-save-program failed: {response.status_code} {response.text}"
        )

        data = response.json()
        print(f"\nResponse: {data}")

        # Verify success
        assert data["success"] is True, f"dspy_save_program failed: {data.get('error')}"

        # Verify output path
        assert data.get("output_path"), "output_path should be set"
        assert data["output_path"].startswith("codeevolver/results/optimized_program_")

        # Verify predictor count
        assert data.get("predictor_count", 0) > 0, "Should have at least one predictor"

        # Print summary
        print(f"\nDSPy Save Program Summary:")
        print(f"  Output path: {data['output_path']}")
        print(f"  Predictor count: {data['predictor_count']}")
        if data.get("logs"):
            print(f"  Logs:")
            for log in data["logs"]:
                print(f"    {log}")
