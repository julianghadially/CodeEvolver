"""Auth-related API routes (key retrieval)."""

from fastapi import APIRouter, Depends, HTTPException

from src.config import settings
from src.db import get_database
from src.db.db_schemas.customer import CustomerRecord
from .dependencies import get_current_user
from .utils import decrypt_api_key
from .middleware import resolve_clerk_id

router = APIRouter(tags=["auth"])


@router.get("/get-api-key")
async def get_api_key(user: CustomerRecord = Depends(get_current_user)):
    """Retrieve the user's API key (requires Clerk JWT auth).

    Looks up the non-revoked auth record for the user and decrypts
    the Fernet-encrypted API key.
    """
    if not settings.snoopy_key:
        raise HTTPException(status_code=500, detail="Encryption key not configured")

    db = get_database()
    auth_doc = await db.auth.find_one({"clerk_id": user.clerk_id, "revoked": False})
    if not auth_doc:
        raise HTTPException(status_code=404, detail="No active API key found")

    raw_key = decrypt_api_key(auth_doc["api_key_rev"], settings.snoopy_key)
    return {"api_key": raw_key}
