"""Program execution and mutation related models."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MutationType(str, Enum):
    """Type of mutation to apply."""

    PROMPT = "prompt"
    CODE = "code"


class ProgramStatus(str, Enum):
    """Status of a program execution."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


class ExecuteStepRequest(BaseModel):
    """Request payload for /execute_step endpoint."""

    client_id: str = Field(..., description="Client identifier from /connect-git")
    program_id: str = Field(..., description="New program id for this mutation")
    parent_program_id: str = Field(..., description="Parent program id")
    mutation_type: MutationType = Field(..., description="Type of mutation: prompt or code")

    # Program location
    program_json_path: str = Field(
        ...,
        description="Path to program.json from project root",
        examples=["src/fire/program.json"],
    )
    entry_point: str = Field(
        ...,
        description="DSPy module to instantiate (e.g., 'fire.FIREJudge')",
    )

    # For prompt mutations
    candidate: dict[str, str] | None = Field(
        default=None,
        description="For prompt mutations: component_name -> new instruction text",
    )

    # For code mutations
    change_request: str | None = Field(
        default=None,
        description="Natural language description of code change",
    )
    change_location: str | None = Field(
        default=None,
        description="Module path hint for code changes (optional)",
    )

    # Test data
    test_examples: list[dict[str, Any]] = Field(
        default_factory=list,
        description="DSPy Examples to run after mutation",
    )
    capture_traces: bool = Field(
        default=False,
        description="Whether to return execution traces",
    )


class PipelineOutput(BaseModel):
    """Output from running program on a single example."""

    example_id: int
    output: Any = Field(..., description="DSPy forward() return value")


class ExecuteStepResponse(BaseModel):
    """Response payload for /execute_step endpoint."""

    program_id: str
    status: Literal["success", "failed"]
    pipeline_outputs: list[PipelineOutput] = Field(default_factory=list)
    traces: list[Any] | None = Field(default=None, description="Execution traces if captured")
    branch_name: str | None = Field(default=None, description="Branch name for code mutations")
    program_json: dict[str, Any] | None = Field(
        default=None,
        description="Updated program state after mutation",
    )
    error: str | None = Field(default=None, description="Error message if failed")


class ProgramRecord(BaseModel):
    """Database record for a program version."""

    client_id: str
    program_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProgramStatus = ProgramStatus.PENDING


class GetProgramResponse(BaseModel):
    """Response payload for GET /program/{program_id}."""

    program_id: str
    client_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    status: str
    created_at: datetime
