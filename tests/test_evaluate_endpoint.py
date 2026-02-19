"""Integration test for /test/evaluate endpoint.

This test verifies that the evaluate endpoint can:
1. Accept a POST request with a DSPy program, metric, and inline batch
2. Run evaluation in a GEPASandbox (production-like environment)
3. Return scores, outputs, and summary stats

To run this test:
1. In one terminal, run: modal serve modal_app.py
2. In another terminal, run: pytest tests/test_evaluate_endpoint.py -v -s -m integration

Target repository: https://github.com/julianghadially/FactChecker
"""

import os

import httpx
import pytest

# Modal app URL
DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"

# Small inline batch for quick evaluation (~3-5 examples)
FACTCHECKER_BATCH = [
    {"statement": "The Eiffel Tower is located in Paris, France."},
    {"statement": "The Great Wall of China is visible from space with the naked eye."},
    {"statement": "Water boils at 100 degrees Celsius at sea level."},
]

EVALUATE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/FactChecker",
    "program": "src.factchecker.modules.judge_module.JudgeModule",
    "metric": "src.codeevolver.metric.metric",
    "batch": FACTCHECKER_BATCH,
    "input_keys": ["statement"],
    "git_branch": "simple",
    "max_rows": 5,
    "program_lm": "openai/gpt-5-mini",
    "num_threads": 1,
    "capture_traces": False,
}


def get_modal_app_url() -> str:
    return os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL)


@pytest.mark.integration
def test_evaluate_endpoint():
    """Test /test/evaluate with FactChecker on a small inline batch."""
    modal_url = get_modal_app_url()

    with httpx.Client(timeout=600) as client:
        print(f"\nPOST {modal_url}/test/evaluate")
        print(f"Config: {EVALUATE_CONFIG}")

        response = client.post(f"{modal_url}/test/evaluate", json=EVALUATE_CONFIG)

        assert response.status_code == 200, (
            f"POST /test/evaluate failed: {response.status_code} {response.text}"
        )

        data = response.json()
        print(f"\nResponse: {data}")

        # Verify success
        assert data["success"] is True, f"Evaluation failed: {data.get('error')}"

        # Verify scores
        assert data["mean_score"] is not None, "mean_score should not be None"
        assert data["mean_score"] >= 0, "mean_score should be non-negative"
        assert data["scores"] is not None, "scores should not be None"
        assert len(data["scores"]) > 0, "scores should have at least one entry"
        assert len(data["scores"]) == len(FACTCHECKER_BATCH), (
            f"Expected {len(FACTCHECKER_BATCH)} scores, got {len(data['scores'])}"
        )

        # Verify num_examples
        assert data["num_examples"] == len(FACTCHECKER_BATCH)

        # Print summary
        print(f"\nEvaluation Summary:")
        print(f"  Mean score: {data['mean_score']:.4f}")
        print(f"  Scores: {data['scores']}")
        print(f"  Num examples: {data['num_examples']}")
        if data.get("logs"):
            print(f"  Logs:")
            for log in data["logs"]:
                print(f"    {log}")
