"""Client and repository related models."""

from datetime import datetime

from pydantic import BaseModel, Field


class ConnectGitRequest(BaseModel):
    """Request payload for /connect-git endpoint."""

    repo_url: str = Field(
        ...,
        description="URL of the git repository to connect",
        examples=["https://github.com/user/project"],
    )


class ConnectGitResponse(BaseModel):
    """Response payload for /connect-git endpoint."""

    client_id: str = Field(..., description="Unique identifier for this client")
    status: str = Field(..., description="Connection status")


class ClientRecord(BaseModel):
    """Database record for a connected client."""

    client_id: str
    repo_url: str
    workspace_path: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "connected"
