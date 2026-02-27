"""Pydantic schemas for structured LM outputs.

Used with Claude Agent SDK's output_format to get validated JSON responses.
"""

from typing import Literal

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


class ConstraintCheckResult(BaseModel):
    """Schema for constraint checker output."""

    status: Literal["pass", "fail"] = Field(
        description="Whether the change request passes or fails the constraint check."
    )
    reason: str = Field(
        description="Brief explanation of why the check passed or failed."
    )
    violated_constraint: str | None = Field(
        default=None,
        description="The specific constraint quoted from additional_instructions that was violated, if any."
    )
