"""Integration test for GEPA optimization via the /optimize API.

This test verifies that the GEPA optimization loop can:
1. Accept an optimization job via POST /optimize
2. Run at least 5 optimization iterations
3. Produce improving evaluation scores
4. Track code mutations in git branches with meaningful diffs

To run this test:
1. In one terminal, run: modal serve modal_app.py
2. In another terminal, run: pytest tests/test_gepa.py -v -s -m integration

The test calls the Modal app's /optimize endpoint and polls /job/{job_id}
for progress. Watch the 'modal serve' terminal to see live logs.

Target repository: https://github.com/julianghadially/FactChecker
"""

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from src.schemas.job_schemas import OptimizationResult
from experiments.FactChecker.input import OPTIMIZE_CONFIG
from experiments.run import run_optimization

# Modal app URL
DEFAULT_MODAL_URL = "https://julianghadially--codeevolver-fastapi-app-dev.modal.run"

POLL_INTERVAL_SECONDS = 30
MAX_POLL_DURATION_SECONDS = 3600  # 60 minutes (increased for 5 iterations)
REQUIRED_ITERATIONS = 5


def get_modal_app_url() -> str:
    return os.getenv("MODAL_APP_URL", DEFAULT_MODAL_URL)

async def run_optimization_test(modal_url: str) -> OptimizationResult:
    import asyncio
    result = await run_optimization(modal_url, OPTIMIZE_CONFIG)
    job_id = result["final_status"]["job_id"]
    if result.final_status["status"] in ["failed", "cancelled"]:
        pytest.fail(f"Optimization job failed: result = {result}")

    with httpx.AsyncClient() as client:
        # --- Poll for progress ---
        print(f"\nPolling job {job_id} every {POLL_INTERVAL_SECONDS}s "
                f"(max {MAX_POLL_DURATION_SECONDS}s)...")
        print(f"Waiting for {REQUIRED_ITERATIONS} completed iterations...\n")

        start_time = time.time()
        final_status = None
        score_history = []
        last_score = None

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

            # Track score history (only when score changes)
            if best_score is not None and best_score != last_score:
                score_history.append(best_score)
                last_score = best_score

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

            await asyncio.sleep(POLL_INTERVAL_SECONDS)

        elapsed = int(time.time() - start_time)

        assert final_status is not None, (
            f"Timed out after {MAX_POLL_DURATION_SECONDS}s waiting for "
            f"{REQUIRED_ITERATIONS} iterations."
        )

    return OptimizationResult(
        final_status=final_status,
        score_history=score_history,
        job_id=job_id,
        elapsed_seconds=elapsed,
    )




@pytest.mark.integration
class TestGEPAOptimization:
    """Integration tests for GEPA optimization.

    The optimizer runs ONCE via a class-scoped fixture, then multiple tests
    verify different aspects of the results. This avoids running the expensive
    optimization multiple times.

    Prerequisites:
    - Run 'modal serve modal_app.py' in another terminal
    - Optionally set MODAL_APP_URL if using a deployed app
    """

    @pytest.fixture(scope="class")
    def modal_url(self) -> str:
        return get_modal_app_url()

    @pytest.fixture(scope="class")
    async def optimization_result(self, modal_url: str) -> OptimizationResult:
        """Run optimization once for all tests in this class."""
        return await run_optimization(modal_url)

    @pytest.mark.asyncio
    async def test_optimization_completes(self, optimization_result: OptimizationResult):
        """Test that optimization completes with required iterations."""
        status = optimization_result.final_status
        iteration = status.get("current_iteration") or 0

        assert iteration >= REQUIRED_ITERATIONS, (
            f"Expected >= {REQUIRED_ITERATIONS} iterations, got {iteration}"
        )

        print(f"\nPASSED: Optimization completed {iteration} iterations "
              f"in {optimization_result.elapsed_seconds}s")

    @pytest.mark.asyncio
    async def test_tracking_structure(self, optimization_result: OptimizationResult):
        """Test that optimization tracks expected fields in candidates."""
        status = optimization_result.final_status
        best_candidate = status.get("best_candidate")

        assert best_candidate is not None, "best_candidate is None"

        candidate_keys = list(best_candidate.keys())
        print(f"\nCandidate keys: {candidate_keys}")

        # Check for required tracking fields
        assert "_code" in best_candidate, (
            f"_code not in candidate. Keys: {candidate_keys}"
        )

        # Validate _code structure (git_branch is stored inside _code)
        code_data = json.loads(best_candidate["_code"])
        assert "git_branch" in code_data, (
            f"git_branch not in _code. Keys: {list(code_data.keys())}"
        )
        assert "architecture" in code_data, (
            f"architecture not in _code. Keys: {list(code_data.keys())}"
        )
        assert "change_request" in code_data, (
            f"change_request not in _code. Keys: {list(code_data.keys())}"
        )

        # Check other tracked fields
        assert status.get("best_score") is not None, "best_score not tracked"
        assert status.get("num_candidates") is not None, "num_candidates not tracked"
        assert status.get("total_metric_calls") is not None, "total_metric_calls not tracked"

        print(f"\nPASSED: All expected fields tracked")
        print(f"  git_branch: {code_data['git_branch']}")
        print(f"  _code keys: {list(code_data.keys())}")
        print(f"  best_score: {status.get('best_score')}")
        print(f"  num_candidates: {status.get('num_candidates')}")
        print(f"  total_metric_calls: {status.get('total_metric_calls')}")

    @pytest.mark.asyncio
    async def test_score_improves(self, optimization_result: OptimizationResult):
        """Test that the optimization score improves over iterations."""
        score_history = optimization_result.score_history

        assert len(score_history) >= 2, (
            f"Not enough score history to verify improvement. "
            f"History: {score_history}"
        )

        first_score = score_history[0]
        final_score = score_history[-1]

        # Score should improve (or at least not get worse)
        assert final_score >= first_score, (
            f"Score did not improve: {first_score} -> {final_score}. "
            f"Full history: {score_history}"
        )

        improvement = final_score - first_score
        print(f"\nPASSED: Score improved")
        print(f"  First score: {first_score:.4f}")
        print(f"  Final score: {final_score:.4f}")
        print(f"  Improvement: {improvement:.4f}")
        print(f"  Full history: {score_history}")

    @pytest.mark.asyncio
    async def test_mutated_branch_has_meaningful_diff(self, optimization_result: OptimizationResult):
        """Test that a mutated branch contains meaningful code changes.

        Loads a non-parent branch from the candidates and verifies the git diff
        contains more than one line of changes.
        """
        status = optimization_result.final_status

        def get_branch_from_candidate(candidate: dict) -> str | None:
            """Extract git_branch from _code component."""
            if not candidate:
                return None
            code_str = candidate.get("_code")
            if not code_str:
                return None
            try:
                code_data = json.loads(code_str)
                return code_data.get("git_branch")
            except (json.JSONDecodeError, TypeError):
                return None

        # Get all candidates to find a mutated branch
        all_candidates = status.get("all_candidates", [])
        best_candidate = status.get("best_candidate")

        # Try to find a mutated branch (not the parent/initial branch)
        # The first candidate is typically the parent, so we want any other
        mutated_branch = None
        parent_branch = None

        if all_candidates and len(all_candidates) > 1:
            # First is typically parent
            parent_branch = get_branch_from_candidate(all_candidates[0])
            # Pick a later candidate
            for candidate in all_candidates[1:]:
                branch = get_branch_from_candidate(candidate)
                if branch and branch != parent_branch:
                    mutated_branch = branch
                    break

        # Fall back to best_candidate if it's different from parent
        if not mutated_branch and best_candidate:
            mutated_branch = get_branch_from_candidate(best_candidate)
            if mutated_branch == parent_branch:
                mutated_branch = None

        assert mutated_branch is not None, (
            f"Could not find a mutated branch. "
            f"all_candidates has {len(all_candidates)} entries. "
            f"best_candidate branch: {get_branch_from_candidate(best_candidate)}"
        )

        print(f"\nChecking diff for mutated branch: {mutated_branch}")
        if parent_branch:
            print(f"  Parent branch: {parent_branch}")

        # Clone the repo and check the diff
        repo_url = OPTIMIZE_CONFIG["repo_url"]

        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone the repo
            initial_branch = OPTIMIZE_CONFIG.get("initial_branch", "main")
            clone_result = subprocess.run(
                ["git", "clone", "--depth", "100", "--branch", initial_branch, repo_url, tmpdir],
                capture_output=True,
                text=True,
            )
            assert clone_result.returncode == 0, (
                f"Failed to clone repo: {clone_result.stderr}"
            )

            # Fetch the mutated branch
            fetch_result = subprocess.run(
                ["git", "fetch", "origin", mutated_branch],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            assert fetch_result.returncode == 0, (
                f"Failed to fetch branch {mutated_branch}: {fetch_result.stderr}"
            )

            # Get the diff between initial branch and the mutated branch
            diff_result = subprocess.run(
                ["git", "diff", f"origin/{initial_branch}", f"origin/{mutated_branch}", "--stat"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            assert diff_result.returncode == 0, (
                f"Failed to get diff: {diff_result.stderr}"
            )

            diff_stat = diff_result.stdout.strip()
            print(f"\nDiff stat:\n{diff_stat}")

            # Get the actual diff for line count
            full_diff_result = subprocess.run(
                ["git", "diff", f"origin/{initial_branch}", f"origin/{mutated_branch}"],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            full_diff = full_diff_result.stdout

            # Count changed lines (lines starting with + or - but not ++ or --)
            changed_lines = [
                line for line in full_diff.split('\n')
                if (line.startswith('+') or line.startswith('-'))
                and not line.startswith('+++')
                and not line.startswith('---')
            ]

            num_changed_lines = len(changed_lines)
            print(f"\nNumber of changed lines: {num_changed_lines}")

            assert num_changed_lines > 1, (
                f"Expected more than 1 line of changes in mutated branch, "
                f"got {num_changed_lines}.\n"
                f"Diff stat:\n{diff_stat}"
            )

            print(f"\nPASSED: Mutated branch has {num_changed_lines} changed lines")
            print(f"  Branch: {mutated_branch}")
            print(f"  Diff preview (first 500 chars):\n{full_diff[:500]}")
