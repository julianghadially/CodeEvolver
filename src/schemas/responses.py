from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
from typing import Any, Literal
from .program_schemas import PipelineOutput


class GetProgramResponse(BaseModel):
    """Response payload for GET /program/{program_id}."""

    program_id: str
    client_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    status: str
    created_at: datetime

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


class ConnectGitResponse(BaseModel):
    """Response payload for /connect-git endpoint."""

    client_id: str = Field(..., description="Unique identifier for this client")
    status: str = Field(..., description="Connection status")


class ExecuteSandboxResponse(BaseModel):
    """Response payload for /execute_sandbox endpoint."""

    status: Literal["success", "failed"]
    program_id: str | None = Field(default=None)
    program_json: dict[str, Any] | None = Field(default=None)
    pipeline_outputs: list[Any] | None = Field(default=None)
    traces: list[Any] | None = Field(default=None)
    branch_name: str | None = Field(default=None)
    error: str | None = Field(default=None)
