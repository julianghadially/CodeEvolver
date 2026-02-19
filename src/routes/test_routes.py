"""Test endpoints for component testing without running full optimization.

These endpoints allow testing individual components like:
- build_seed_candidate with various program structures
- GEPA state tracking with known inputs
- Adapter methods in isolation
- Git operations and branch management

Usage:
    POST /test/build-seed - Test build_seed_candidate with a repo
    POST /test/evaluate - Run evaluation on a dataset batch
    POST /test/gepa-state - Test GEPA state structure validation
    POST /test/git-ops - Test git branch operations
    GET /test/list - List available test operations
"""

import json
import sys
import tempfile
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# Import Modal test functions from modal_app
# These are defined in modal_app.py but delegate to src/test_functions.py for implementation
try:
    from modal_app import test_build_seed_candidate, test_evaluate
except ImportError:
    # If running locally without Modal, functions won't be available
    test_build_seed_candidate = None
    test_evaluate = None

router = APIRouter(prefix="/test", tags=["testing"])


class BuildSeedTestRequest(BaseModel):
    """Request to test build_seed_candidate."""
    repo_url: str = Field(..., description="Git repository URL")
    program: str = Field(..., description="Dotted import path to DSPy module")
    initial_branch: str = Field(default="main", description="Initial branch to start from")
    saved_program_json_path: str | None = Field(default=None, description="Optional path to program.json")
    installation_id: int | None = Field(default=None, description="GitHub App installation ID")

    # Optional: Perform refactoring before testing
    refactor_files: dict[str, str] | None = Field(
        default=None,
        description="Optional file renames to apply before testing. Format: {'old_path': 'new_path'}"
    )
    refactor_classes: dict[str, str] | None = Field(
        default=None,
        description="Optional class renames. Format: {'OldClass': 'NewClass'}"
    )


class BuildSeedTestResponse(BaseModel):
    """Response from build_seed_candidate test."""
    success: bool
    candidate: dict[str, str] | None = None
    num_predictors: int | None = None
    predictor_names: list[str] | None = None
    error: str | None = None
    logs: list[str] | None = None


class GEPAStateTestRequest(BaseModel):
    """Request to test GEPA state structure."""
    mock_state: dict = Field(..., description="Mock GEPA state to validate")


class GEPAStateTestResponse(BaseModel):
    """Response from GEPA state validation."""
    valid: bool
    errors: list[str] | None = None
    warnings: list[str] | None = None


class GitOpsTestRequest(BaseModel):
    """Request to test git operations."""
    operation: Literal["create_branch", "checkout", "diff", "parse_branch"] = Field(
        ..., description="Git operation to test"
    )
    repo_url: str | None = Field(default=None, description="Repository URL (for clone operations)")
    branch_name: str | None = Field(default=None, description="Branch name")
    parent_branch: str | None = Field(default=None, description="Parent branch name")


class GitOpsTestResponse(BaseModel):
    """Response from git operations test."""
    success: bool
    result: dict | None = None
    error: str | None = None


class EvaluateTestRequest(BaseModel):
    """Request to test evaluation on a dataset batch."""
    repo_url: str = Field(..., description="Git repository URL")
    program: str = Field(..., description="Dotted import path to DSPy module")
    metric: str = Field(..., description="Dotted import path to metric function")
    batch: list[dict] | None = Field(default=None, description="Inline dataset (list of dicts)")
    batch_path: str | None = Field(default=None, description="Path to data file in repo (JSON/JSONL/CSV)")
    candidate: dict[str, str] | None = Field(default=None, description="Explicit prompt texts (if None, uses program defaults)")
    git_branch: str = Field(default="main", description="Branch to evaluate")
    saved_program_json_path: str | None = Field(default=None, description="Optional path to program.json")
    program_lm: str = Field(default="openai/gpt-5-mini", description="LM for evaluation")
    num_threads: int = Field(default=1, description="Parallelism for evaluation")
    input_keys: list[str] | None = Field(default=None, description="Explicit input field names")
    max_rows: int = Field(default=20, description="Cap on dataset size")
    installation_id: int | None = Field(default=None, description="GitHub App installation ID for private repos")
    capture_traces: bool = Field(default=False, description="Whether to capture DSPy traces")


class EvaluateTestResponse(BaseModel):
    """Response from evaluation test."""
    success: bool
    mean_score: float | None = None
    scores: list[float] | None = None
    num_examples: int | None = None
    outputs: list | None = None
    error: str | None = None
    logs: list[str] | None = None


@router.get("/list")
async def list_test_operations():
    """List available test operations."""
    return {
        "operations": [
            {
                "endpoint": "/test/build-seed",
                "method": "POST",
                "description": "Test build_seed_candidate with a repository",
                "use_cases": [
                    "Verify a DSPy program can be parsed",
                    "Test with refactored code structures",
                    "Validate predictor extraction"
                ]
            },
            {
                "endpoint": "/test/gepa-state",
                "method": "POST",
                "description": "Validate GEPA state structure",
                "use_cases": [
                    "Test state tracking with known examples",
                    "Verify candidate structure",
                    "Validate score tracking"
                ]
            },
            {
                "endpoint": "/test/git-ops",
                "method": "POST",
                "description": "Test git operations",
                "use_cases": [
                    "Test branch naming conventions",
                    "Verify git diff parsing",
                    "Test branch creation"
                ]
            },
            {
                "endpoint": "/test/evaluate",
                "method": "POST",
                "description": "Run evaluation on a dataset batch without full optimization",
                "use_cases": [
                    "Quick evaluation of a DSPy program on a small batch",
                    "Test that metric + program work together",
                    "Get baseline scores before running optimization"
                ]
            }
        ]
    }


@router.post("/build-seed", response_model=BuildSeedTestResponse)
async def test_build_seed_candidate_endpoint(request: BuildSeedTestRequest) -> BuildSeedTestResponse:
    """Test build_seed_candidate with a repository.

    This endpoint:
    1. Clones the repository
    2. Optionally applies refactoring (file/class renames)
    3. Calls build_seed_candidate to extract predictors
    4. Returns the candidate structure

    Example use case: Test that build_seed_candidate works after
    renaming ResearchAgentModule → ResearchAgentModuleTest in FactChecker.
    """
    if test_build_seed_candidate is None:
        raise HTTPException(
            status_code=503,
            detail="Modal not available. Run with 'modal serve modal_app.py'."
        )

    try:
        # Call Modal function to execute test in sandbox
        result = await test_build_seed_candidate.remote.aio(
            repo_url=request.repo_url,
            program=request.program,
            initial_branch=request.initial_branch,
            saved_program_json_path=request.saved_program_json_path,
            installation_id=request.installation_id,
            refactor_files=request.refactor_files,
            refactor_classes=request.refactor_classes,
        )

        return BuildSeedTestResponse(
            success=result.get("success", False),
            candidate=result.get("candidate"),
            num_predictors=result.get("num_predictors"),
            predictor_names=result.get("predictor_names"),
            error=result.get("error"),
            logs=result.get("logs"),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Modal not available. Run with 'modal serve modal_app.py'."
        )
    except Exception as e:
        return BuildSeedTestResponse(
            success=False,
            error=f"Test failed: {e}"
        )


@router.post("/gepa-state", response_model=GEPAStateTestResponse)
async def test_gepa_state(request: GEPAStateTestRequest) -> GEPAStateTestResponse:
    """Validate GEPA state structure with a known example.

    This tests that the state tracking correctly handles:
    - program_candidates with _code components
    - prog_candidate_val_subscores
    - prog_candidate_objective_scores
    - parent_program_for_candidates
    """
    errors = []
    warnings = []
    state = request.mock_state

    # Validate required fields
    required_fields = [
        "program_candidates",
        "prog_candidate_val_subscores",
        "prog_candidate_objective_scores",
    ]
    for field in required_fields:
        if field not in state:
            errors.append(f"Missing required field: {field}")

    if errors:
        return GEPAStateTestResponse(valid=False, errors=errors)

    # Validate candidates structure
    candidates = state.get("program_candidates", [])
    for i, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            errors.append(f"Candidate {i} is not a dict")
            continue

        # Check for _code component
        if "_code" not in candidate:
            errors.append(f"Candidate {i} missing _code component")
            continue

        # Validate _code structure
        try:
            code_data = json.loads(candidate["_code"])
            required_code_fields = ["git_branch", "parent_module_path", "change_request", "last_change_summary"]
            for field in required_code_fields:
                if field not in code_data:
                    errors.append(f"Candidate {i} _code missing field: {field}")

            # Validate git_branch format
            git_branch = code_data.get("git_branch", "")
            if not git_branch.startswith("codeevolver-"):
                warnings.append(f"Candidate {i} git_branch doesn't follow naming convention: {git_branch}")

        except json.JSONDecodeError:
            errors.append(f"Candidate {i} _code is not valid JSON")

    # Validate scores match candidates
    num_candidates = len(candidates)
    num_subscores = len(state.get("prog_candidate_val_subscores", []))
    num_obj_scores = len(state.get("prog_candidate_objective_scores", []))

    if num_subscores != num_candidates:
        errors.append(f"Subscores count ({num_subscores}) != candidates count ({num_candidates})")
    if num_obj_scores != num_candidates:
        errors.append(f"Objective scores count ({num_obj_scores}) != candidates count ({num_candidates})")

    return GEPAStateTestResponse(
        valid=len(errors) == 0,
        errors=errors if errors else None,
        warnings=warnings if warnings else None,
    )


@router.post("/git-ops", response_model=GitOpsTestResponse)
async def test_git_operations(request: GitOpsTestRequest) -> GitOpsTestResponse:
    """Test git operations in isolation."""
    try:
        if request.operation == "parse_branch":
            # Test branch name parsing
            if not request.branch_name:
                raise ValueError("branch_name required for parse_branch operation")

            # Extract timestamp and id from branch name
            # Format: codeevolver-{YYYYMMDDHHmmss}-{uuid} or codeevolver-{YYYYMMDDHHmmss}-main
            parts = request.branch_name.split("-")
            if len(parts) < 3 or parts[0] != "codeevolver":
                return GitOpsTestResponse(
                    success=False,
                    error=f"Branch name doesn't match expected format: {request.branch_name}"
                )

            timestamp = parts[1]
            suffix = "-".join(parts[2:])
            is_main = suffix == "main"

            return GitOpsTestResponse(
                success=True,
                result={
                    "timestamp": timestamp,
                    "suffix": suffix,
                    "is_main_branch": is_main,
                    "is_mutation_branch": not is_main,
                }
            )

        elif request.operation == "create_branch":
            # Test branch name generation
            from datetime import datetime
            import uuid

            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            short_id = uuid.uuid4().hex[:6]
            main_branch = f"codeevolver-{timestamp}-main"
            mutation_branch = f"codeevolver-{timestamp}-{short_id}"

            return GitOpsTestResponse(
                success=True,
                result={
                    "main_branch": main_branch,
                    "mutation_branch": mutation_branch,
                    "timestamp": timestamp,
                }
            )

        else:
            raise ValueError(f"Unsupported operation: {request.operation}")

    except Exception as e:
        return GitOpsTestResponse(
            success=False,
            error=str(e)
        )


@router.post("/evaluate", response_model=EvaluateTestResponse)
async def test_evaluate_endpoint(request: EvaluateTestRequest) -> EvaluateTestResponse:
    """Run evaluation on a dataset batch without full optimization.

    This endpoint:
    1. Clones the repository and sets up the sandbox
    2. Optionally builds the seed candidate (if no explicit candidate provided)
    3. Runs exec_prebuilt(evaluate) on the batch
    4. Returns scores, outputs, and summary stats

    Use this to quickly test that a program + metric work together,
    or to get baseline scores before running a full optimization.
    """
    if test_evaluate is None:
        raise HTTPException(
            status_code=503,
            detail="Modal not available. Run with 'modal serve modal_app.py'."
        )

    try:
        result = await test_evaluate.remote.aio(
            repo_url=request.repo_url,
            program=request.program,
            metric=request.metric,
            batch=request.batch,
            batch_path=request.batch_path,
            candidate=request.candidate,
            git_branch=request.git_branch,
            saved_program_json_path=request.saved_program_json_path,
            program_lm=request.program_lm,
            num_threads=request.num_threads,
            input_keys=request.input_keys,
            max_rows=request.max_rows,
            installation_id=request.installation_id,
            capture_traces=request.capture_traces,
        )

        return EvaluateTestResponse(
            success=result.get("success", False),
            mean_score=result.get("mean_score"),
            scores=result.get("scores"),
            num_examples=result.get("num_examples"),
            outputs=result.get("outputs"),
            error=result.get("error"),
            logs=result.get("logs"),
        )

    except ImportError:
        raise HTTPException(
            status_code=503,
            detail="Modal not available. Run with 'modal serve modal_app.py'."
        )
    except Exception as e:
        return EvaluateTestResponse(
            success=False,
            error=f"Test failed: {e}"
        )
