"""Optimization job schemas for GEPA integration."""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class OptimizeRequest(BaseModel):
    """Request payload for POST /optimize.

    Users must provide either inline `trainset` data or a `trainset_path`
    pointing to a data file in their repository.
    """

    repo_url: str = Field(..., description="Git repository URL")
    program: str = Field(
        ...,
        description="Dotted import path of dspy.module program that will undergo optimization (e.g., 'src.fire.AgentPipeline'). Specify from project root and include the class itself (not just the file)",
    )
    metric: str = Field(
        ...,
        description=(
            "Dotted import path of dspy metric function (e.g., 'eval.metric.accuracy). Specify from project root and include the class itself (not just the file)"
        ),
    )
    saved_program_json_path: str | None = Field(
        default=None,
        description="Path to a previously DSPY saved program.json (e.g., from a prior optimization). JSON includes module structures, instructions, etc. Specify from the project root (optional)",
        examples=["program.json"],
    )
    # Dataset — provide inline data OR a path to a file in the repo
    trainset: list[dict[str, Any]] | None = Field(
        default=None,
        description="Training dataset as inline list of dicts",
    )
    trainset_path: str | None = Field(
        default=None,
        description="Path to training data file in repo (json/jsonl/csv)",
    )
    valset: list[dict[str, Any]] | None = Field(
        default=None,
        description="Validation dataset (defaults to trainset if not provided)",
    )
    valset_path: str | None = Field(
        default=None,
        description="Path to validation data file in repo (json/jsonl/csv)",
    )
    input_keys: list[str] | None = Field(
        default=None,
        description="Explicit input field names for dspy.Example.with_inputs()",
    )

    # LM configuration
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

    @model_validator(mode="after")
    def check_trainset_provided(self) -> "OptimizeRequest":
        if self.trainset is None and self.trainset_path is None:
            raise ValueError("Provide either 'trainset' (inline data) or 'trainset_path' (file in repo)")
        return self


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
