from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from .program_schemas import PipelineOutput
from .job_schemas import JobStatus

class GetProgramResponse(BaseModel):
    """Response payload for GET /program/{program_id}."""

    program_id: str
    client_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    status: str
    created_at: datetime

class ConnectGitResponse(BaseModel):
    """Response payload for /connect-git endpoint."""

    client_id: str = Field(..., description="Unique identifier for this client")
    status: str = Field(..., description="Connection status")


class ChangeResponse(BaseModel):
    """Response payload for POST /change_request endpoint."""

    success: bool
    branch_name: str | None = Field(default=None)
    error: str | None = Field(default=None)
    output: str | None = Field(default=None, description="Agent execution output/logs")


class OptimizeResponse(BaseModel):
    """Response payload for POST /optimize."""

    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    """Response payload for GET /job/{job_id}."""

    job_id: str
    status: JobStatus
    current_iteration: int | None = None
    total_metric_calls: int | None = None
    num_candidates: int | None = None
    best_candidate: dict[str, str] | None = None
    best_score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None

class OptimizeResponse(BaseModel):
    """Response payload for POST /optimize."""

    job_id: str
    status: str = "pending"


class JobStatusResponse(BaseModel):
    """Response payload for GET /job/{job_id}."""

    job_id: str
    status: JobStatus
    current_iteration: int | None = None
    total_metric_calls: int | None = None
    num_candidates: int | None = None
    best_candidate: dict[str, str] | None = None
    best_score: float | None = None
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    updated_at: datetime | None = None




class CancelCheckResponse(BaseModel):
    """GET /internal/job/{job_id}/check-cancelled."""

    cancelled: bool

