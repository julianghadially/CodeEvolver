from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
from .program_schemas import ProgramStatus
from .job_schemas import JobStatus

class ProgramRecord(BaseModel):
    """Database record for a program version."""

    client_id: str
    program_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProgramStatus = ProgramStatus.PENDING


class ClientRecord(BaseModel):
    """Database record for a connected client."""

    client_id: str
    repo_url: str
    workspace_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "connected"

class JobRecord(BaseModel):
    """MongoDB document for an optimization job."""

    job_id: str
    client_id: str
    repo_url: str
    config: dict[str, Any]
    status: JobStatus = JobStatus.PENDING
    current_iteration: int | None = None
    total_metric_calls: int | None = None
    num_candidates: int | None = None
    best_candidate: dict[str, str] | None = None
    best_score: float | None = None
    error: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # GEPA State (stored on completion)
    gepa_state: dict[str, Any] | None = None  # Full GEPAState serialized
    program_candidates: list[dict[str, str]] | None = None  # All candidates
    candidate_scores: list[float] | None = None  # Validation scores per candidate
    parent_programs: list[list[int] | None] | None = None  # Parent indices for each candidate

    # Per-iteration history: dict keyed by candidate index (string for MongoDB)
    gepa_state_history: dict[str, Any] | None = None
