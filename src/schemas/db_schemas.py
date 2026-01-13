from pydantic import BaseModel, Field
from datetime import datetime
from typing import Any
from .program_schemas import ProgramStatus

class ProgramRecord(BaseModel):
    """Database record for a program version."""

    client_id: str
    program_id: str
    parent_program_id: str
    program_json: dict[str, Any]
    branch_name: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: ProgramStatus = ProgramStatus.PENDING


class ClientRecord(BaseModel):
    """Database record for a connected client."""

    client_id: str
    repo_url: str
    workspace_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "connected"
