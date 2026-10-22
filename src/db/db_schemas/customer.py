"""Customer and authentication database schemas."""

from datetime import datetime
from pydantic import BaseModel, Field


class CustomerRecord(BaseModel):
    """MongoDB `customers` collection document."""

    clerk_id: str
    clerk_email: str
    first_name: str | None = None
    last_name: str | None = None
    allowed: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)


class AuthRecord(BaseModel):
    """MongoDB `auth` collection document.

    Stores hashed + encrypted API keys linked to a Clerk user.
    - api_hash: SHA-256 hex digest of the raw API key (used for lookup)
    - api_key_rev: Fernet-encrypted raw API key (used for key retrieval/recovery)
    """

    clerk_id: str
    api_hash: str
    api_key_rev: str
    revoked: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
