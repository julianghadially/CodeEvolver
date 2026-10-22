"""Shared enums and types used across API, engine, and DB layers."""

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


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class PipelineOutput(BaseModel):
    """Output from running program on a single example."""

    example_id: int
    output: Any = Field(..., description="DSPy forward() return value")
