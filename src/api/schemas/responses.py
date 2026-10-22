from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
from src.shared.enums import JobStatus
from src.db.db_schemas.db_schemas import IterationHistoryRecord


class GetProgramResponse(BaseModel):
    """Response payload for GET /program/{program_id}."""

    program_id: str
    repo_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    status: str
    created_at: datetime


class OptimizeResponse(BaseModel):
    """Response payload for POST /optimize."""

    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    """Response payload for GET /job/{job_id}."""

    job_id: str
    status: JobStatus
    repo_url: str | None = None
    config: dict[str, Any] | None = None
    current_iteration: int | None = None
    total_metric_calls: int | None = None
    num_branches: int | None = None
    # dict[str, Any], not dict[str, str]: `final_candidate` (v3/run_reporting.py)
    # carries `is_primary: True`, a bool. Typing the values as str made
    # GET /job/{id} raise a 500 the moment a candidate was registered --
    # which is the endpoint runner/entrypoint.py polls for terminal status,
    # so a run could never be seen to finish and never exported artifacts.
    final_candidate: dict[str, Any] | None = None
    seed_train_score: float | None = None
    final_train_score: float | None = None
    seed_test_score: float | None = None
    final_test_score: float | None = None
    seed_test_scores: list[float] | None = None
    final_test_scores: list[float] | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None
    name: str | None = None  # Display name; portal shows it instead of job_id
    # Long markdown reports: returned by GET /job/{id} only; GET /jobs leaves
    # them None to keep the list response small.
    memory_md: str | None = None
    meta_reflection_md: str | None = None
    integrity_audit_md: str | None = None


class CancelCheckResponse(BaseModel):
    """GET /internal/job/{job_id}/check-cancelled."""

    cancelled: bool


class CancelJobResponse(BaseModel):
    """POST /job/{job_id}/cancel."""

    job_id: str
    status: str
    already_terminal: bool = False


class CleanupBranchesResponse(BaseModel):
    """Response payload for POST /cleanup-branches."""

    deleted: list[str] = Field(default_factory=list, description="Branch names that were deleted")
    skipped: list[str] = Field(default_factory=list, description="Branch names that were skipped (in except_branches)")
    errors: list[str] = Field(default_factory=list, description="Branches that failed to delete, with error details")


class JobDetailedStateResponse(BaseModel):
    """Response payload for GET /job/{job_id}/state with full CE state."""

    job_id: str
    status: JobStatus
    # Engine 3 branch registry: branch -> {last_sha, is_primary, ...}.
    branches: dict[str, Any] | None = None
    primary_branch: str | None = None
    # dict[str, Any], not dict[str, str]: `final_candidate` (v3/run_reporting.py)
    # carries `is_primary: True`, a bool. Typing the values as str made
    # GET /job/{id} raise a 500 the moment a candidate was registered --
    # which is the endpoint runner/entrypoint.py polls for terminal status,
    # so a run could never be seen to finish and never exported artifacts.
    final_candidate: dict[str, Any] | None = None
    seed_train_score: float | None = None
    final_train_score: float | None = None
    seed_test_score: float | None = None
    final_test_score: float | None = None
    seed_test_scores: list[float] | None = None
    final_test_scores: list[float] | None = None
    num_branches: int | None = None
    total_metric_calls: int | None = None
    num_iterations: int | None = None


class JobListResponse(BaseModel):
    """Response payload for GET /jobs."""

    jobs: list[JobStatusResponse]
    total: int
    skip: int
    limit: int


class IterationHistoryResponse(BaseModel):
    """Response payload for GET /job/{job_id}/iterations.

    iterations is a dict keyed by iteration number (as string for MongoDB compat),
    each value is an IterationHistoryRecord.
    """

    job_id: str
    status: JobStatus
    iterations: dict[str, IterationHistoryRecord]
    total_iterations: int


class ImportJobHistoryResponse(BaseModel):
    """Response payload for POST /import-job-history."""

    job_id: str
    created: bool  # True if a new job record was inserted, False if updated
    total_iterations: int


class RepoResponse(BaseModel):
    """Single repo in GET /repos response."""

    repo_id: str
    repo_url: str
    github_owner: str | None = None
    github_repo_name: str | None = None
    installation_id: int | None = None
    status: str
    created_at: datetime | None = None


class RepoListResponse(BaseModel):
    """Response payload for GET /repos."""

    repos: list[RepoResponse]
    total: int
    skip: int
    limit: int
