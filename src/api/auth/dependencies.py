"""FastAPI dependencies for authentication."""

from fastapi import Depends, Request, HTTPException

from src.config import settings
from src.db import get_database
from src.db.db_schemas.customer import CustomerRecord
from .local_user import LOCAL_STUB_USER
from .middleware import resolve_clerk_id


async def get_current_user(request: Request) -> CustomerRecord:
    """FastAPI dependency: resolves auth header to a CustomerRecord.

    Usage:
        @app.post("/endpoint")
        async def endpoint(user: CustomerRecord = Depends(get_current_user)):
            ...

    Local-mode short-circuit: when `settings.sandbox_mode == "local"` the API
    binds to 127.0.0.1 and serves a single operator. Skip Clerk + DB lookup
    entirely and return the stub user defined in `local_user.py`.
    """
    if settings.sandbox_mode == "local":
        return LOCAL_STUB_USER

    clerk_id = await resolve_clerk_id(request)

    db = get_database()
    customer = await db.customers.find_one({"clerk_id": clerk_id})
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
    if not customer.get("allowed", True):
        raise HTTPException(status_code=403, detail="Account disabled")

    return CustomerRecord(**customer)
