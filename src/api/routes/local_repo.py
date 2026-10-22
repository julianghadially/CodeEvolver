"""Local-mode repo registration.

Replaces the GitHub-App webhook flow for local deploys. The operator
points the API at a repo checkout on disk; we write a `RepoRecord` with
`installation_id=None` (the optimizer's GitHub-token codepaths already
gracefully no-op when this is missing) and a `workspace_path` that the
`LocalLauncher` later uses as the in-place workspace.

Mounted in `src/api/main.py` only when `settings.sandbox_mode == "local"`.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from src.api.auth.dependencies import get_current_user
from src.db import get_database
from src.db.db_schemas.customer import CustomerRecord
from src.db.db_schemas.db_schemas import RepoRecord

router = APIRouter(prefix="/local", tags=["local"])


class ConnectLocalRepoRequest(BaseModel):
    local_path: str = Field(..., description="Absolute path to the repo checkout on disk")
    repo_url: str | None = Field(
        None,
        description=(
            "Optional remote URL (e.g., https://github.com/foo/bar). "
            "When omitted, defaults to file://<local_path>."
        ),
    )


class ConnectLocalRepoResponse(BaseModel):
    repo_id: str
    repo_url: str
    workspace_path: str


@router.post("/connect-repo", response_model=ConnectLocalRepoResponse)
async def connect_local_repo(
    payload: ConnectLocalRepoRequest,
    user: CustomerRecord = Depends(get_current_user),
) -> ConnectLocalRepoResponse:
    """Register a local repo checkout for optimization.

    Idempotent: re-registering the same `local_path` updates the existing
    record (matched on `(repo_url, clerk_id)`) so the operator does not
    accumulate duplicates after re-runs.
    """
    local_path = Path(payload.local_path).expanduser().resolve()
    if not local_path.exists():
        raise HTTPException(
            status_code=400,
            detail=f"local_path does not exist: {local_path}",
        )
    if not (local_path / ".git").exists():
        raise HTTPException(
            status_code=400,
            detail=f"local_path is not a git repository (no .git/): {local_path}",
        )

    repo_url = payload.repo_url or f"file://{local_path}"
    db = get_database()

    existing = await db["repos"].find_one(
        {"repo_url": repo_url, "clerk_id": user.clerk_id}
    )
    if existing:
        await db["repos"].update_one(
            {"_id": existing["_id"]},
            {"$set": {"workspace_path": str(local_path), "status": "connected"}},
        )
        repo_id = existing["repo_id"]
    else:
        repo_id = uuid.uuid4().hex
        record = RepoRecord(
            repo_id=repo_id,
            clerk_id=user.clerk_id,
            repo_url=repo_url,
            workspace_path=str(local_path),
            installation_id=None,
        )
        await db["repos"].insert_one(record.model_dump())

    return ConnectLocalRepoResponse(
        repo_id=repo_id,
        repo_url=repo_url,
        workspace_path=str(local_path),
    )
