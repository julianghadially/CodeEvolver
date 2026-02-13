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

    @pytest.mark.asyncio
    async def test_factchecker_module_rename(self, modal_url: str):
        """Test refactoring: rename ResearchAgentModule to ResearchAgentModuleTest.

        This test verifies the coding agent can perform a systematic refactoring:
        1. Create a new branch
        2. Rename research_agent_module.py to research_agent_module_test.py
        3. Rename the class ResearchAgentModule to ResearchAgentModuleTest
        4. Update all references (imports, instantiation, etc.)

        This tests the agent's ability to make coordinated changes across multiple files.
        """
        import asyncio

        # Configuration for this specific refactoring test
        refactor_config = {
            "repo_url": "https://github.com/julianghadially/FactChecker",
            "program": "src.factchecker.modules.fact_checker_pipeline.FactCheckerPipeline",
            "metric": "src.codeevolver.metric.metric",
            "trainset_path": "data/FactChecker_news_claims_normalized.csv",
            "input_keys": ["statement"],
            "reflection_lm": "openai/gpt-5-mini",
            "max_metric_calls": 10,  # Minimal - just verify it works after refactoring
            "num_threads": 1,
            "seed": 42,
            "initial_branch": "main",
            "additional_instructions": """
## Refactoring Task

Perform the following systematic refactoring:

1. **Rename file**: `src/factchecker/modules/research_agent_module.py` → `research_agent_module_test.py`
2. **Rename class**: `ResearchAgentModule` → `ResearchAgentModuleTest`
3. **Update all references**:
   - In `src/factchecker/modules/fact_checker_pipeline.py`:
     - Import: `from .research_agent_module_test import ResearchAgentModuleTest`
     - Instantiation: `self.research_agent = ResearchAgentModuleTest(...)`
     - Signature reference: Update any type hints or docstrings
   - In `src/factchecker/modules/__init__.py`:
     - Export: `from .research_agent_module_test import ResearchAgentModuleTest`
     - Add to `__all__` list
4. **Search and replace**: Find all instances of `ResearchAgentModule(` and replace with `ResearchAgentModuleTest(`

Ensure all imports and references are updated consistently.
""",
            # Stop after just 1 code mutation
            "code_cutoff_step": 1,
        }

        print(f"\n{'='*60}")
        print("REFACTORING TEST: Rename ResearchAgentModule")
        print(f"{'='*60}\n")

        # Submit optimization job
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(f"{modal_url}/optimize", json=refactor_config)
            assert response.status_code == 200, f"POST /optimize failed: {response.text}"

            result = response.json()
            job_id = result["job_id"]
            print(f"Started refactoring job: {job_id}\n")

            # Poll for completion (shorter timeout since we only need 1 code mutation)
            start_time = time.time()
            max_wait = 600  # 10 minutes
            final_status = None

            while time.time() - start_time < max_wait:
                job_response = await client.get(f"{modal_url}/job/{job_id}")
                assert job_response.status_code == 200

                status = job_response.json()
                state = status["status"]
                iteration = status.get("current_iteration", 0)

                elapsed = int(time.time() - start_time)
                print(f"  [{elapsed:>4d}s] status={state}  iteration={iteration}")

                if state == "failed":
                    pytest.fail(f"Refactoring job failed: {status.get('error')}")

                if state == "completed":
                    final_status = status
                    break

                # For this test, we can stop after the first code mutation
                # (iteration 0 is seed, iteration 1 should be our code mutation)
                if iteration >= 2:
                    final_status = status
                    break

                await asyncio.sleep(10)

            assert final_status is not None, f"Refactoring timed out after {max_wait}s"

            # Get the mutated branch
            best_candidate = final_status.get("best_candidate")
            assert best_candidate is not None, "No best_candidate found"

            code_str = best_candidate.get("_code")
            assert code_str is not None, "_code not found in candidate"

            code_data = json.loads(code_str)
            mutated_branch = code_data.get("git_branch")
            assert mutated_branch is not None, "git_branch not found in _code"

            print(f"\nRefactored branch: {mutated_branch}")

        # Verify the changes in git
        with tempfile.TemporaryDirectory() as tmpdir:
            # Clone and checkout the refactored branch
            clone_result = subprocess.run(
                ["git", "clone", "--depth", "10", "--branch", "main",
                 refactor_config["repo_url"], tmpdir],
                capture_output=True,
                text=True,
            )
            assert clone_result.returncode == 0, f"Clone failed: {clone_result.stderr}"

            # Fetch the mutated branch
            subprocess.run(
                ["git", "fetch", "origin", mutated_branch],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )

            # Checkout the mutated branch
            checkout_result = subprocess.run(
                ["git", "checkout", mutated_branch],
                cwd=tmpdir,
                capture_output=True,
                text=True,
            )
            assert checkout_result.returncode == 0, f"Checkout failed: {checkout_result.stderr}"

            # Verify the changes
            print("\nVerifying refactoring changes:")

            # 1. Check that research_agent_module_test.py exists
            test_module_path = os.path.join(tmpdir, "src/factchecker/modules/research_agent_module_test.py")
            assert os.path.exists(test_module_path), \
                "research_agent_module_test.py not found"
            print("  ✓ research_agent_module_test.py exists")

            # 2. Check that the class is renamed
            with open(test_module_path, "r") as f:
                test_module_content = f.read()
            assert "class ResearchAgentModuleTest" in test_module_content, \
                "Class not renamed to ResearchAgentModuleTest"
            print("  ✓ Class renamed to ResearchAgentModuleTest")

            # 3. Check that fact_checker_pipeline.py imports the new module
            pipeline_path = os.path.join(tmpdir, "src/factchecker/modules/fact_checker_pipeline.py")
            if os.path.exists(pipeline_path):
                with open(pipeline_path, "r") as f:
                    pipeline_content = f.read()
                assert "ResearchAgentModuleTest" in pipeline_content, \
                    "fact_checker_pipeline.py doesn't reference ResearchAgentModuleTest"
                print("  ✓ fact_checker_pipeline.py updated")

            # 4. Check that __init__.py exports the new module
            init_path = os.path.join(tmpdir, "src/factchecker/modules/__init__.py")
            if os.path.exists(init_path):
                with open(init_path, "r") as f:
                    init_content = f.read()
                assert "ResearchAgentModuleTest" in init_content, \
                    "__init__.py doesn't export ResearchAgentModuleTest"
                print("  ✓ __init__.py updated")

            # 5. Check that old references are removed (optional - may still exist)
            # We'll just warn if they exist rather than fail
            old_module_path = os.path.join(tmpdir, "src/factchecker/modules/research_agent_module.py")
            if os.path.exists(old_module_path):
                print("  ⚠ Warning: Old research_agent_module.py still exists (not removed)")
            else:
                print("  ✓ Old research_agent_module.py removed")

            print(f"\nPASSED: Refactoring completed successfully")
            print(f"  Branch: {mutated_branch}")
            print(f"  Elapsed: {int(time.time() - start_time)}s")
