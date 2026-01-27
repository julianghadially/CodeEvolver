"""JWT service for job-scoped authentication.

Mints and validates short-lived HS256 tokens scoped to a single job_id.
Used by the GEPA sandbox to authenticate callback requests to the FastAPI
internal endpoints.
"""

import time

import jwt

from src.config import settings


def mint_job_token(job_id: str, ttl_seconds: int = 3600) -> str:
    """Create a JWT scoped to a single job.

    Args:
        job_id: The job this token authorises updates for.
        ttl_seconds: Token lifetime in seconds (default 1 hour).

    Returns:
        Encoded JWT string.

    Raises:
        ValueError: If jwt_secret is not configured.
    """
    if not settings.jwt_secret:
        raise ValueError("CODEEVOLVER_JWT_SECRET must be set to mint JWTs")

    now = int(time.time())
    payload = {
        "job_id": job_id,
        "scope": "job:update",
        "iat": now,
        "exp": now + ttl_seconds,
    }
    return jwt.encode(payload, settings.jwt_secret, algorithm="HS256")


def validate_job_token(token: str, expected_job_id: str) -> dict:
    """Decode and validate a job-scoped JWT.

    Args:
        token: Encoded JWT string.
        expected_job_id: The job_id the token must be scoped to.

    Returns:
        Decoded payload dict.

    Raises:
        jwt.ExpiredSignatureError: Token has expired.
        jwt.InvalidTokenError: Token is invalid.
        ValueError: Token job_id does not match expected_job_id, or secret not set.
    """
    if not settings.jwt_secret:
        raise ValueError("CODEEVOLVER_JWT_SECRET must be set to validate JWTs")

    payload = jwt.decode(token, settings.jwt_secret, algorithms=["HS256"])

    if payload.get("job_id") != expected_job_id:
        raise ValueError(
            f"Token job_id '{payload.get('job_id')}' does not match "
            f"expected '{expected_job_id}'"
        )

    return payload
