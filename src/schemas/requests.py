"""Program execution and mutation related schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field, model_validator

from .program_schemas import MutationType
from .job_schemas import JobStatus

class ConnectGitRequest(BaseModel):
    """Request payload for /connect-git endpoint."""

    repo_url: str = Field(
        ...,
        description="URL of the git repository to connect",
        examples=["https://github.com/user/project"],
    )
    installation_id: int | None = Field(
        default=None,
        description="GitHub App installation ID for private repository access",
    )


class ChangeRequest(BaseModel):
    """Request payload for POST /change_request endpoint.

    Execute a code change via the Claude coding agent.
    """

    repo_url: str = Field(..., description="Git repository URL to clone")
    change_request: str = Field(..., description="Natural language description of code change")
    change_location: str | None = Field(
        default=None,
        description="Module path hint for code changes (e.g., 'src/core/agent.py')",
    )
    branch_name: str | None = Field(
        default=None,
        description="Branch name to create (auto-generated if not provided)",
    )
    push_to_remote: bool = Field(
        default=False,
        description="If True, push changes to remote after mutation",
    )
    installation_id: int | None = Field(
        default=None,
        description="GitHub App installation ID for private repository access",
    )
    initial_branch: str = Field(
        default="main",
        description="Git branch to clone from before creating the new branch",
    )


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
    program_lm: str = Field(
        default="openai/gpt-5-mini",
        description="LM for running the DSPy program during evaluation",
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

    # Additional guidance for GEPA optimization
    additional_instructions: str | None = Field(
        default=None,
        description=(
            "Client-provided guidance for GEPA optimization. "
            "May include: constraints (changes that are off-limits), "
            "services (external APIs available with keys in environment), "
            "and ideas for optimization approaches."
        ),
    )

    # Code mutation frequency control with exponential decay
    initial: int | None = Field(
        default=None,
        description=(
            "Starting prompts per code change. "
            "0=code only, 1=alternating (1:1), 2=2 prompts per code, etc. "
            "If None, uses GEPA's default round-robin selector."
        ),
    )
    decay_rate: int = Field(
        default=25,
        description="Iterations between each multiplier step for prompts-per-code ratio",
    )
    decay_factor: int = Field(
        default=2,
        description="Multiplier applied at each decay step (e.g., 2 means ratio doubles)",
    )
    code_cutoff_step: int | None = Field(
        default=None,
        description="Stop code mutations after this iteration (None=no cutoff)",
    )

    # Git branch configuration
    initial_branch: str = Field(
        default="main",
        description="Git branch to use as the starting point for optimization",
    )

    @model_validator(mode="after")
    def check_trainset_provided(self) -> "OptimizeRequest":
        if self.trainset is None and self.trainset_path is None:
            raise ValueError("Provide either 'trainset' (inline data) or 'trainset_path' (file in repo)")
        return self

class JobStatusUpdateRequest(BaseModel):
    """PUT /internal/job/{job_id}/status — update job status from GEPA sandbox."""

    status: str
    best_candidate: dict[str, str] | None = None
    best_score: float | None = None
    total_metric_calls: int | None = None
    num_candidates: int | None = None
    error: str | None = None


class JobProgressUpdateRequest(BaseModel):
    """PUT /internal/job/{job_id}/progress — iteration progress from GEPA."""

    current_iteration: int
    best_score: float
    best_candidate: dict[str, str]
    total_metric_calls: int
    num_candidates: int


class GitHubTokenResponse(BaseModel):
    """GET /internal/job/{job_id}/github-token — refresh GitHub token."""

    token: str | None = None
    error: str | None = None
