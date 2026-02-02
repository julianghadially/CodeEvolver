"""Pydantic schemas for structured LM outputs.

Used with Claude Agent SDK's output_format to get validated JSON responses.
"""

from pydantic import BaseModel, Field


class ArchitectureOutput(BaseModel):
    """Schema for architecture summary output from reflection agent."""

    architecture: str = Field(
        description="Architecture summary as a single string. Use newlines for formatting."
    )


class ChangeRequestOutput(BaseModel):
    """Schema for code change request output from reflection agent."""

    change_request: str = Field(
        description="Specific, actionable change request in 1-5 sentences that a coding agent can execute immediately."
    )
