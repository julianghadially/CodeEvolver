"""Optimization job schemas for GEPA integration."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizeRequest(BaseModel):
    """Request payload for POST /optimize."""

    repo_url: str = Field(..., description="Git repository URL")
    program_json_path: str = Field(
        ...,
        description="Path to program.json from project root",
        examples=["src/fire/program.json"],
    )
    entry_point: str = Field(
        ...,
        description="DSPy module class (e.g., 'fire.FIREJudge')",
    )

    # Metric
    metric_path: str = Field(
        ...,
        description="Path to metric script in repo (e.g., 'eval/metric.py')",
    )
    metric_fn_name: str = Field(
        default="metric",
        description="Metric function name within the script",
    )

    # Dataset
    trainset: list[dict[str, Any]] = Field(
        ...,
        description="Training dataset as list of dicts",
    )
    valset: list[dict[str, Any]] | None = Field(
        default=None,
        description="Validation dataset (defaults to trainset if not provided)",
    )
    input_keys: list[str] | None = Field(
        default=None,
        description="Explicit input field names for dspy.Example.with_inputs()",
    )

    # LM configuration
    task_lm: str = Field(
        default="openai/gpt-5-mini",
        description="LM for running the DSPy program",
    )
    reflection_lm: str = Field(
        default="openai/gpt-5-mini",
        description="LM for GEPA reflection (instruction proposal)",
    )

    # Optimization config
    max_metric_calls: int = Field(
        default=1000,
        description="Max metric evaluations (optimization budget)",
    )
    num_threads: int = Field(
        default=1,
        description="Threads for parallel DSPy evaluation",
    )
    seed: int = Field(
        default=0,
        description="Random seed for reproducibility",
    )

    # Authentication
    installation_id: int | None = Field(
        default=None,
        description="GitHub App installation ID for private repos",
    )


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
