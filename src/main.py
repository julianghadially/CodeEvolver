"""FastAPI application for CodeEvolver Agents.

Run locally:  uvicorn src.main:app --reload
Run on Modal: modal serve modal_app.py (or modal deploy modal_app.py)
"""

import os
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException

from .config import settings
from .db import get_database, lifespan
from .schemas import (
    ConnectGitRequest,
    ConnectGitResponse,
    ClientRecord,
    ChangeRequest,
    ChangeResponse,
    OptimizeRequest,
    OptimizeResponse,
    JobStatusResponse,
    JobStatusUpdateRequest,
    JobProgressUpdateRequest,
    CancelCheckResponse,
    JobRecord,
    JobStatus,
)
from .services import GitService, GitHubAppService
from .services.jwt_service import mint_job_token, validate_job_token
from modal_app import run_optimization

# Check if running on Modal (sandbox execution enabled)
USE_SANDBOX = os.getenv("CODEEVOLVER_USE_SANDBOX", "false").lower() == "true"

app = FastAPI(
    title="CodeEvolver Agents",
    description="Remote service for executing evolutionary code changes",
    version="0.2.0",
    lifespan=lifespan,
)


@app.post("/connect-git", response_model=ConnectGitResponse)
async def connect_git(request: ConnectGitRequest) -> ConnectGitResponse:
    """
    Register a client repository for evolutionary optimization.

    Clones the repository to the server and returns a client_id for future requests.

    For private repositories, provide installation_id from your GitHub App installation.
    """
    client_id = GitService.generate_client_id()

    try:
        workspace_path = GitService.clone_repository(
            request.repo_url,
            client_id,
            installation_id=request.installation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    record = ClientRecord(
        client_id=client_id,
        repo_url=request.repo_url,
        workspace_path=str(workspace_path),
    )

    db = get_database()
    await db.clients.insert_one(record.model_dump())

    return ConnectGitResponse(client_id=client_id, status="connected")



@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(request: OptimizeRequest) -> OptimizeResponse:
    """Start a GEPA optimization job.

    Clones the repository, creates a job record in MongoDB, and spawns
    a Modal function to run the GEPA optimization loop asynchronously.
    Returns a job_id for polling status via GET /job/{job_id}.
    """
    from uuid import uuid4

    job_id = f"job_{uuid4().hex[:12]}"
    client_id = GitService.generate_client_id()

    # Resolve installation_id: use request value or fall back to environment variable
    installation_id = request.installation_id
    if installation_id is None:
        env_installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
        if env_installation_id:
            try:
                installation_id = int(env_installation_id)
            except ValueError:
                pass  # Invalid env var, leave as None

    # Pre-generate GitHub installation token for the worker/sandbox
    # Security: The private key stays in FastAPI, only the short-lived token goes to sandbox
    github_token = None
    if installation_id:
        try:
            github_token = GitHubAppService.get_installation_token(installation_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"GitHub authentication failed: {e}")

    # Clone repository
    try:
        workspace_path = GitService.clone_repository(
            request.repo_url,
            client_id,
            installation_id=installation_id,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Store client record
    client_record = ClientRecord(
        client_id=client_id,
        repo_url=request.repo_url,
        workspace_path=str(workspace_path),
    )
    db = get_database()
    await db.clients.insert_one(client_record.model_dump())

    # Create job record
    job_record = JobRecord(
        job_id=job_id,
        client_id=client_id,
        repo_url=request.repo_url,
        config=request.model_dump(),
    )
    await db.jobs.insert_one(job_record.model_dump())

    # Mint a job-scoped JWT for sandbox→API callbacks
    # JWT TTL matches the job timeout (12 hours) + 5 min buffer
    jwt_token = ""
    callback_url = settings.callback_url
    if settings.jwt_secret:
        jwt_token = mint_job_token(
            job_id,
            ttl_seconds=settings.gepa_job_timeout + 300,
        )

    # Spawn Modal function for optimization (fire-and-forget)
    try:
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")


        run_optimization.spawn(
            job_id=job_id,
            repo_url=request.repo_url,
            program=request.program,
            metric=request.metric,
            saved_program_json_path=request.saved_program_json_path,
            trainset_json=request.trainset,
            trainset_path=request.trainset_path,
            valset_json=request.valset,
            valset_path=request.valset_path,
            program_lm=request.program_lm,
            reflection_lm=request.reflection_lm,
            max_metric_calls=request.max_metric_calls,
            github_token=github_token,
            input_keys=request.input_keys,
            num_threads=request.num_threads,
            seed=request.seed,
            callback_url=callback_url,
            jwt_token=jwt_token,
            additional_instructions=request.additional_instructions,
            initial=request.initial,
            decay_rate=request.decay_rate,
            decay_factor=request.decay_factor,
            code_cutoff_step=request.code_cutoff_step,
            initial_branch=request.initial_branch,
        )
    except ImportError:
        # Not running on Modal — update job status to failed
        await db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": JobStatus.FAILED.value,
                "error": "Optimization requires Modal deployment. Use 'modal serve modal_app.py'.",
            }},
        )
        return OptimizeResponse(job_id=job_id, status="failed")

    return OptimizeResponse(job_id=job_id, status="pending")


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    """Get optimization job status, progress, and current best candidate."""
    db = get_database()
    job = await db.jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        current_iteration=job.get("current_iteration"),
        total_metric_calls=job.get("total_metric_calls"),
        num_candidates=job.get("num_candidates"),
        best_candidate=job.get("best_candidate"),
        best_score=job.get("best_score"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        updated_at=job.get("updated_at"),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.3.0"}


@app.get("/debug_secrets")
async def debug_secrets():
    """Debug endpoint to check secret configuration (does not expose full secrets)."""
    import os
    from .config import settings
    
    # Get raw env var for comparison
    raw_key = os.getenv("CODEEVOLVER_GITHUB_APP_PRIVATE_KEY", "")
    processed_key = settings.github_app_private_key or ""
    
    return {
        "github_app_id": settings.github_app_id,
        "anthropic_api_key_set": bool(settings.anthropic_api_key),
        "openai_api_key_set": bool(settings.openai_api_key),
        "github_app_private_key": {
            "raw_env_length": len(raw_key),
            "raw_env_first_50": raw_key[:50] if raw_key else None,
            "raw_env_last_30": raw_key[-30:] if raw_key else None,
            "raw_contains_begin": "BEGIN" in raw_key,
            "raw_contains_newlines": "\n" in raw_key,
            "raw_newline_count": raw_key.count("\n"),
            "processed_length": len(processed_key),
            "processed_first_50": processed_key[:50] if processed_key else None,
            "processed_last_30": processed_key[-30:] if processed_key else None,
            "processed_contains_begin": "BEGIN" in processed_key,
            "processed_contains_newlines": "\n" in processed_key,
            "processed_newline_count": processed_key.count("\n"),
            "is_valid_pem_format": (
                processed_key.strip().startswith("-----BEGIN") 
                if processed_key else False
            ),
        },
        "env_vars_present": {
            "CODEEVOLVER_GITHUB_APP_PRIVATE_KEY": bool(raw_key),
            "GITHUB_TEST_INSTALLATION_ID": os.getenv("GITHUB_TEST_INSTALLATION_ID"),
            "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY"))
        }
    }


@app.post("/change_request", response_model=ChangeResponse)
async def change_request(request: ChangeRequest) -> ChangeResponse:
    """Execute a code change via the Claude coding agent.

    Creates a sandbox, clones the repository, applies the code change
    using Claude Agent SDK, and optionally pushes to remote.
    """
    try:
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")

        from modal_app import execute_change_request

        # Resolve installation_id: use request value or fall back to environment variable
        installation_id = request.installation_id
        if installation_id is None:
            env_installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
            if env_installation_id:
                try:
                    installation_id = int(env_installation_id)
                except ValueError:
                    pass  # Invalid env var, leave as None

        # Pre-generate GitHub installation token for the sandbox
        # Security: The private key stays in FastAPI, only the short-lived token goes to sandbox
        github_token = None
        if installation_id:
            try:
                github_token = GitHubAppService.get_installation_token(installation_id)
            except ValueError:
                pass  # Will attempt unauthenticated if token generation fails

        result = await execute_change_request.remote.aio(
            repo_url=request.repo_url,
            change_request=request.change_request,
            change_location=request.change_location,
            branch_name=request.branch_name,
            push_to_remote=request.push_to_remote,
            github_token=github_token,
            initial_branch=request.initial_branch,
        )

        return ChangeResponse(
            success=result.get("success", False),
            branch_name=result.get("branch_name"),
            error=result.get("error"),
            output=result.get("output"),
        )

    except ImportError as e:
        return ChangeResponse(
            success=False,
            error=f"Modal not available: {e}. Run with 'modal serve modal_app.py'.",
        )
    except Exception as e:
        return ChangeResponse(
            success=False,
            error=f"Change request failed: {e}",
        )


# Keep /execute_sandbox as deprecated alias (redirects to new implementation)
@app.post("/execute_sandbox", response_model=ChangeResponse, deprecated=True)
async def execute_sandbox_deprecated(request: ChangeRequest) -> ChangeResponse:
    """DEPRECATED: Use /change_request instead."""
    return await change_request(request)


# ---------------------------------------------------------------------------
# Internal callback endpoints (called by GEPA sandbox via JWT)
# ---------------------------------------------------------------------------

def _validate_internal_jwt(job_id: str, authorization: str | None) -> None:
    """Extract Bearer token and validate it against the given job_id.

    Raises HTTPException on any auth failure.
    """
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header")

    token = authorization.removeprefix("Bearer ").strip()
    try:
        validate_job_token(token, job_id)
    except ValueError as e:
        raise HTTPException(status_code=403, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@app.put("/internal/job/{job_id}/status")
async def internal_update_job_status(
    job_id: str,
    body: JobStatusUpdateRequest,
    authorization: str | None = Header(default=None),
):
    """Update job status (called by GEPA sandbox via callback)."""
    _validate_internal_jwt(job_id, authorization)

    update: dict = {"status": body.status, "updated_at": datetime.now(timezone.utc)}

    if body.status == "running":
        update["started_at"] = datetime.now(timezone.utc)
    if body.status in ("completed", "failed"):
        update["completed_at"] = datetime.now(timezone.utc)

    if body.best_candidate is not None:
        update["best_candidate"] = body.best_candidate
    if body.best_score is not None:
        update["best_score"] = body.best_score
    if body.total_metric_calls is not None:
        update["total_metric_calls"] = body.total_metric_calls
    if body.num_candidates is not None:
        update["num_candidates"] = body.num_candidates
    if body.error is not None:
        update["error"] = body.error

    db = get_database()
    await db.jobs.update_one({"job_id": job_id}, {"$set": update})
    return {"ok": True}


@app.put("/internal/job/{job_id}/progress")
async def internal_update_job_progress(
    job_id: str,
    body: JobProgressUpdateRequest,
    authorization: str | None = Header(default=None),
):
    """Update iteration progress (called each GEPA iteration)."""
    _validate_internal_jwt(job_id, authorization)

    update = {
        "current_iteration": body.current_iteration,
        "best_score": body.best_score,
        "best_candidate": body.best_candidate,
        "total_metric_calls": body.total_metric_calls,
        "num_candidates": body.num_candidates,
        "updated_at": datetime.now(timezone.utc),
    }

    db = get_database()
    await db.jobs.update_one({"job_id": job_id}, {"$set": update})
    return {"ok": True}


@app.get("/internal/job/{job_id}/check-cancelled", response_model=CancelCheckResponse)
async def internal_check_cancelled(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    """Check if a job has been cancelled (polled by GEPA progress tracker)."""
    _validate_internal_jwt(job_id, authorization)

    db = get_database()
    job = await db.jobs.find_one({"job_id": job_id}, {"status": 1})
    cancelled = bool(job and job.get("status") == "cancelled")
    return CancelCheckResponse(cancelled=cancelled)


@app.get("/internal/job/{job_id}/github-token")
async def internal_refresh_github_token(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    """Get a fresh GitHub installation token for the job.

    GitHub tokens expire after 1 hour. For long-running optimizations,
    the sandbox calls this endpoint to get a fresh token before git push.

    Returns:
        GitHubTokenResponse with fresh token or error message.
    """
    from .schemas.requests import GitHubTokenResponse

    _validate_internal_jwt(job_id, authorization)

    # Look up job to get installation_id from config
    db = get_database()
    job = await db.jobs.find_one({"job_id": job_id}, {"config": 1})
    if not job:
        return GitHubTokenResponse(error=f"Job not found: {job_id}")

    config = job.get("config", {})
    installation_id = config.get("installation_id")

    # Fall back to environment variable if not in job config
    if installation_id is None:
        env_installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
        if env_installation_id:
            try:
                installation_id = int(env_installation_id)
            except ValueError:
                pass

    if not installation_id:
        return GitHubTokenResponse(error="No installation_id configured for this job")

    try:
        token = GitHubAppService.get_installation_token(installation_id)
        return GitHubTokenResponse(token=token)
    except ValueError as e:
        return GitHubTokenResponse(error=f"Failed to generate token: {e}")
