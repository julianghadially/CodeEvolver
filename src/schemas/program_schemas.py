"""AI workflow program schemas."""
from enum import Enum
from pydantic import BaseModel, Field
from typing import Any

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

class PipelineOutput(BaseModel):
    """Output from running program on a single example."""

    example_id: int
    output: Any = Field(..., description="DSPy forward() return value")
