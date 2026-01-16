"""Program execution and mutation related schemas."""

from datetime import datetime
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field

from .program_schemas import MutationType


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



