"""Integration test for GEPA prompt optimization via the /optimize API.

This test verifies that the GEPA optimization loop can:
1. Accept an optimization job via POST /optimize
2. Run at least 2 optimization iterations
3. Produce non-zero evaluation scores

To run this test:
1. In one terminal, run: modal serve modal_app.py
2. In another terminal, run: pytest tests/test_gepa.py -v -s -m integration

The test calls the Modal app's /optimize endpoint and polls /job/{job_id}
for progress. Watch the 'modal serve' terminal to see live logs.

Target repository: https://github.com/julianghadially/FactChecker
"""

import os
import time

import httpx
import pytest


# Modal app URL
DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"

# FactChecker optimization configuration
OPTIMIZE_CONFIG = {
    "repo_url": "https://github.com/julianghadially/FactChecker",
    "program": "src.factchecker.simple.modules.judge_module.JudgeModule", #src.factchecker.modules.fact_checker_pipeline.FactCheckerPipeline
    "metric": "src.codeevolver.metric.metric",
    "trainset_path": "data/FacTool_QA_train_normalized.jsonl",
    "input_keys": ["statement"],
    "reflection_lm": "openai/gpt-5-mini",
    "max_metric_calls": 1000,
    "num_threads": 5,
    "seed": 42,
}

# Polling configuration
POLL_INTERVAL_SECONDS = 30
MAX_POLL_DURATION_SECONDS = 1800  # 30 minutes
REQUIRED_ITERATIONS = 2


def get_modal_app_url() -> str:
    return os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL)


@pytest.mark.integration
class TestGEPAOptimization:
    """Integration tests for GEPA prompt optimization.

    Prerequisites:
    - Run 'modal serve modal_app.py' in another terminal
    - Optionally set MODAL_APP_URL if using a deployed app
    """

    @pytest.fixture
    def modal_url(self) -> str:
        return get_modal_app_url()

    @pytest.mark.asyncio
    async def test_prompt_optimizer(self, modal_url: str):
        """Test that GEPA prompt optimization runs and produces non-zero scores.

        Steps:
        1. Health-check the Modal app
        2. POST /optimize with FactChecker config
        3. Poll GET /job/{job_id} until current_iteration >= 2
        4. Assert best_score is present and > 0
        """
        async with httpx.AsyncClient(
            timeout=120.0, follow_redirects=True
        ) as client:
            # --- Health check ---
            try:
                health = await client.get(f"{modal_url}/health")
                assert health.status_code == 200, (
                    f"Health check failed: {health.status_code}. "
                    "Is 'modal serve modal_app.py' running?"
                )
                print(f"Health check passed: {health.json()}")
            except httpx.ConnectError as exc:
                pytest.fail(
                    f"Cannot connect to Modal app at {modal_url}: {exc}\n"
                    "Make sure 'modal serve modal_app.py' is running."
                )

            # --- Submit optimization job ---
            print(f"\nSubmitting optimization job...")
            print(f"  Repository: {OPTIMIZE_CONFIG['repo_url']}")
            print(f"  Program:    {OPTIMIZE_CONFIG['program']}")
            print(f"  Metric:     {OPTIMIZE_CONFIG['metric']}")
            print(f"  Trainset:   {OPTIMIZE_CONFIG['trainset_path']}")

            response = await client.post(
                f"{modal_url}/optimize",
                json=OPTIMIZE_CONFIG,
            )
            assert response.status_code == 200, (
                f"POST /optimize failed: {response.status_code} {response.text}"
            )

            result = response.json()
            job_id = result["job_id"]
            assert result["status"] != "failed", (
                f"Job creation failed immediately: {result}"
            )
            print(f"  Job created: {job_id} (status: {result['status']})")

            # --- Poll for progress ---
            print(f"\nPolling job {job_id} every {POLL_INTERVAL_SECONDS}s "
                  f"(max {MAX_POLL_DURATION_SECONDS}s)...")
            print(f"Waiting for {REQUIRED_ITERATIONS} completed iterations...\n")

            start_time = time.time()
            final_status = None

            while time.time() - start_time < MAX_POLL_DURATION_SECONDS:
                job_response = await client.get(f"{modal_url}/job/{job_id}")
                assert job_response.status_code == 200, (
                    f"GET /job/{job_id} failed: {job_response.status_code} "
                    f"{job_response.text}"
                )

                status = job_response.json()
                state = status["status"]
                iteration = status.get("current_iteration") or 0
                best_score = status.get("best_score")
                num_candidates = status.get("num_candidates")
                metric_calls = status.get("total_metric_calls")
                error = status.get("error")

                elapsed = int(time.time() - start_time)
                score_str = f"  score={best_score:.4f}" if best_score is not None else ""
                cand_str = f"  candidates={num_candidates}" if num_candidates else ""
                calls_str = f"  metric_calls={metric_calls}" if metric_calls else ""
                print(
                    f"  [{elapsed:>4d}s] status={state}  "
                    f"iteration={iteration}{score_str}{cand_str}{calls_str}"
                )

                if state == "failed":
                    pytest.fail(
                        f"Optimization job failed after {elapsed}s.\n"
                        f"  Error: {error}\n"
                        f"  Iteration: {iteration}\n"
                        f"  Full response: {status}"
                    )

                if state == "cancelled":
                    pytest.fail(f"Optimization job was cancelled at iteration {iteration}.")

                if state == "completed":
                    final_status = status
                    break

                if iteration >= REQUIRED_ITERATIONS:
                    final_status = status
                    break

                await _async_sleep(POLL_INTERVAL_SECONDS)

            # --- Assertions ---
            assert final_status is not None, (
                f"Timed out after {MAX_POLL_DURATION_SECONDS}s waiting for "
                f"{REQUIRED_ITERATIONS} iterations. Last iteration: {iteration}"
            )

            actual_iteration = final_status.get("current_iteration") or 0
            assert actual_iteration >= REQUIRED_ITERATIONS, (
                f"Expected >= {REQUIRED_ITERATIONS} iterations, got {actual_iteration}"
            )

            best_score = final_status.get("best_score")
            assert best_score is not None, (
                f"best_score is None after {actual_iteration} iterations. "
                f"Full response: {final_status}"
            )
            assert best_score > 0, (
                f"Expected non-zero best_score, got {best_score}. "
                f"Full response: {final_status}"
            )

            print(f"\n{'='*60}")
            print(f"PASSED: {actual_iteration} iterations completed")
            print(f"  Best score:    {best_score}")
            print(f"  Candidates:    {final_status.get('num_candidates')}")
            print(f"  Metric calls:  {final_status.get('total_metric_calls')}")
            if final_status.get("best_candidate"):
                candidate_keys = list(final_status["best_candidate"].keys())
                print(f"  Candidate keys: {candidate_keys}")
            print(f"{'='*60}")


async def _async_sleep(seconds: float):
    """Async-compatible sleep using asyncio."""
    import asyncio
    await asyncio.sleep(seconds)
