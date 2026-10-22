"""Clerk webhook handler for user.created events.

When a user signs up via Clerk, this endpoint:
1. Verifies the Svix webhook signature
2. Extracts user data from the Clerk payload
3. Generates an API key pair (raw + SHA-256 hash + Fernet encrypted)
4. Creates a CustomerRecord and AuthRecord in MongoDB
"""

from fastapi import APIRouter, Request, HTTPException

from svix.webhooks import Webhook, WebhookVerificationError

from src.config import settings
from src.db import get_database
from src.db.db_schemas.customer import CustomerRecord, AuthRecord
from .utils import generate_api_key, hash_api_key, encrypt_api_key

router = APIRouter(tags=["clerk"])


@router.post("/clerk/new-user")
async def clerk_new_user(request: Request):
    """Handle Clerk user.created webhook event."""
    if not settings.clerk_wh_signature:
        raise HTTPException(status_code=500, detail="Clerk webhook signature not configured")

    headers = {
        "svix-id": request.headers.get("svix-id", ""),
        "svix-timestamp": request.headers.get("svix-timestamp", ""),
        "svix-signature": request.headers.get("svix-signature", ""),
    }
    body = await request.body()

    wh = Webhook(settings.clerk_wh_signature)
    try:
        payload = wh.verify(body, headers)
    except WebhookVerificationError:
        raise HTTPException(status_code=400, detail="Invalid Svix signature")

    if payload.get("type") != "user.created":
        return {"ok": True, "skipped": True}

    data = payload["data"]
    clerk_id = data["id"]

    # Extract primary email
    emails = data.get("email_addresses", [])
    primary_email_id = data.get("primary_email_address_id")
    if primary_email_id and emails:
        clerk_email = next(
            (e["email_address"] for e in emails if e["id"] == primary_email_id),
            emails[0]["email_address"] if emails else "",
        )
    elif emails:
        clerk_email = emails[0]["email_address"]
    else:
        clerk_email = ""

    first_name = data.get("first_name")
    last_name = data.get("last_name")

    # Generate API key pair
    raw_key = generate_api_key()
    api_hash = hash_api_key(raw_key)
    api_key_rev = encrypt_api_key(raw_key, settings.snoopy_key)

    db = get_database()

    # Check for duplicate clerk_id
    existing = await db.customers.find_one({"clerk_id": clerk_id})
    if existing:
        return {"ok": True, "skipped": True, "reason": "customer already exists"}

    await db.customers.insert_one(
        CustomerRecord(
            clerk_id=clerk_id,
            clerk_email=clerk_email,
            first_name=first_name,
            last_name=last_name,
        ).model_dump()
    )

    await db.auth.insert_one(
        AuthRecord(
            clerk_id=clerk_id,
            api_hash=api_hash,
            api_key_rev=api_key_rev,
        ).model_dump()
    )

    return {"ok": True}
