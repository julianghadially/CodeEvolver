"""FastAPI application for CodeEvolver Agents.

Run locally:  uvicorn src.api.main:app --reload
Deploy:       gcloud run deploy codeevolver-api --source .
"""

import logging
import os
from datetime import datetime, timezone

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.db import get_database, lifespan
from src.db.db_schemas import JobRecord
from src.db.db_schemas.customer import CustomerRecord
from src.shared.enums import JobStatus
from src.api.schemas import (
    OptimizeRequest,
    OptimizeResponse,
    JobStatusResponse,
    JobDetailedStateResponse,
    JobStatusUpdateRequest,
    JobProgressUpdateRequest,
    ImportJobHistoryRequest,
    ImportJobHistoryResponse,
    CancelCheckResponse,
    CancelJobResponse,
    CleanupBranchesRequest,
    CleanupBranchesResponse,
    JobListResponse,
    IterationHistoryResponse,
    RepoResponse,
    RepoListResponse,
)
from src.shared.services.github_app import GitHubAppService, normalize_repo_url
from src.api.services.jwt_service import mint_job_token, validate_job_token
from src.launchers import spawn_optimization
from src.local.repo_lock import RepoLockedError
from src.db.job_writes import record_iteration_progress, record_job_status_update

# Auth
from src.api.auth.dependencies import get_current_user
from src.api.auth.clerk_webhook import router as clerk_router
from src.api.auth.routes import router as auth_router

# Local-mode-only routes (registered conditionally below)
from src.api.routes.local_repo import router as local_repo_router

# The optimizer keeps working past its own soft deadline: the loop stops
# `TIMEOUT_BUFFER_SECONDS` before the job timeout, and finalization (held-out
# evals, then the meta-reflection and run-integrity agents) runs after. The
# callback JWT has to outlive that tail — see `_mint_callback_token`.
# KEEP IN SYNC with optimizer_loop.TIMEOUT_BUFFER_SECONDS (not imported:
# src/api does not depend on src/engine).
FINALIZATION_RESERVE_SECONDS = 36000  # 10 hours

app = FastAPI(
    title="CodeEvolver Agents",
    description="Agentic optimization of AI system code against an evaluation dataset",
    version="0.3.0",
    lifespan=lifespan,
)

# CORS — allow frontend origins based on environment
_cors_origins = {
    "local": ["http://localhost:3000"],
    "dev": ["http://localhost:3000", "https://dev.codeevolver.ai"],
    "prod": ["https://codeevolver.ai", "https://www.codeevolver.ai"],
}
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins.get(settings.app_mode, _cors_origins["prod"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(auth_router)

# The Clerk webhook only means something on the hosted portal API; a
# single-operator local deploy has no identity provider and registers repos
# through /local/connect-repo instead.
if settings.sandbox_mode != "local":
    app.include_router(clerk_router)
else:
    app.include_router(local_repo_router)


@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(
    request: OptimizeRequest,
    user: CustomerRecord = Depends(get_current_user),
) -> OptimizeResponse:
    """Start a CodeEvolver optimization job.

    Creates a job record in MongoDB and launches the optimizer as a local
    subprocess (`src/launchers/local_launcher.py`). Returns a job_id for
    polling status via GET /job/{job_id}. Optimizations run in local mode
    only; the hosted portal API serves results imported via
    /import-job-history.
    """
    from uuid import uuid4

    if settings.sandbox_mode != "local":
        raise HTTPException(
            status_code=501,
            detail=(
                "Optimizations run locally or through the Docker runner "
                "(SANDBOX_MODE=local). Import finished runs with "
                "POST /import-job-history."
            ),
        )

    job_id = f"job_{uuid4().hex[:12]}"
    canonical_url = normalize_repo_url(request.repo_url)

    # Look up the connected repo (registered via POST /local/connect-repo). Accept
    # records saved with a `.git` suffix by matching both forms.
    db = get_database()
    existing_repo = await db.repos.find_one({
        "repo_url": {"$in": [canonical_url, f"{canonical_url}.git"]},
        "clerk_id": user.clerk_id,
    })
    if not existing_repo:
        raise HTTPException(
            status_code=404,
            detail="Repository not connected. Register it first with POST /local/connect-repo.",
        )

    # A repo record may carry a GitHub App installation_id; when it does, mint
    # a short-lived token so the optimizer can push its branches. Repos
    # registered via /local/connect-repo have none, and the optimizer's GitHub-token
    # codepaths (refresh_github_token, git_push_authenticated) no-op without it.
    installation_id = existing_repo.get("installation_id")
    github_token: str | None = None
    if installation_id:
        try:
            github_token = GitHubAppService.get_installation_token(installation_id)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"GitHub authentication failed: {e}")

    repo_id = existing_repo["repo_id"]

    # Create job record (inject installation_id so token refresh can find it)
    config = request.model_dump()
    config["installation_id"] = installation_id
    config["repo_url"] = canonical_url
    job_record = JobRecord(
        job_id=job_id,
        clerk_id=user.clerk_id,
        repo_id=repo_id,
        repo_url=canonical_url,
        config=config,
    )
    await db.jobs.insert_one(job_record.model_dump())

    # Mint a job-scoped JWT for sandbox→API callbacks
    jwt_token = ""
    callback_url = settings.callback_url
    if settings.jwt_secret:
        # The callback JWT must outlive the whole run or callbacks start
        # failing auth mid-job.
        job_timeout = settings.local_job_timeout
        # A token that expires during the finalization tail is not a graceful
        # degradation — `set_completed` 401s silently, the job never reaches a
        # terminal status, and the runner polls forever without exporting. So
        # the TTL carries a full extra finalization reserve on top of the job
        # timeout, which survives a finalization that overruns its budget.
        jwt_token = mint_job_token(
            job_id,
            ttl_seconds=job_timeout + FINALIZATION_RESERVE_SECONDS + 300,
        )

    # Launch the optimizer subprocess (fire-and-forget)
    try:
        await spawn_optimization(
            job_id=job_id,
            repo_url=canonical_url,
            program=request.program,
            metric=request.metric,
            trainset_json=request.trainset,
            trainset_path=request.trainset_path,
            testset_json=request.testset,
            testset_path=request.testset_path,
            reflection_lm=request.reflection_lm,
            coding_harness=request.coding_harness,
            coding_harness_model=request.coding_harness_model,
            max_metric_calls=request.max_metric_calls,
            github_token=github_token,
            input_keys=request.input_keys,
            num_threads=request.num_threads,
            seed=request.seed,
            callback_url=callback_url,
            jwt_token=jwt_token,
            additional_instructions=request.additional_instructions,
            additional_deny_paths=request.additional_deny_paths,
            subsample_size=request.subsample_size,
            initial_branch=request.initial_branch,
            comparison_set_size=(
                request.comparison_set_size
                if request.comparison_set_size is not None
                else settings.comparison_set_size
            ),
            comparison_credits_initial=request.comparison_credits_initial,
            comparison_credits_per_iteration=request.comparison_credits_per_iteration,
            debug_max_iterations=request.debug_max_iterations,
            asa_behavioral_contribution_constant=request.asa_behavioral_contribution_constant,
            asa_behavioral_contribution_threshold_cap=request.asa_behavioral_contribution_threshold_cap,
            asa_primary_stale_landed_iterations=request.asa_primary_stale_landed_iterations,
            subsample_eval_timeout=request.subsample_eval_timeout,
            comparison_eval_timeout=request.comparison_eval_timeout,
            test_samples=request.test_samples,
            seed_test_samples=request.seed_test_samples,
            variance_estimate_samples=request.variance_estimate_samples,
            python_version=request.python_version,
            enable_meta_reflection=request.enable_meta_reflection,
            enable_run_integrity_agent=request.enable_run_integrity_agent,
            enable_clean_slate_inspector=request.enable_clean_slate_inspector,
        )
    except RepoLockedError as e:
        # Documented in specs/experiment-docs/local-deployment.md but never implemented: a second
        # /optimize against a live local repo must be distinguishable from a real
        # failure, or an orchestrator retries into a permanently-failing job.
        await db.jobs.delete_one({"job_id": job_id})
        raise HTTPException(status_code=409, detail=str(e))
    except Exception as e:
        err_msg = f"Failed to spawn optimization: {e}"
        extras = [f"app_mode={os.getenv('APP_MODE', 'prod')}"]
        logger.error("%s (%s)", err_msg, ", ".join(extras), exc_info=True)
        await db.jobs.update_one(
            {"job_id": job_id},
            {"$set": {
                "status": JobStatus.FAILED.value,
                "error": err_msg,
            }},
        )
        return OptimizeResponse(job_id=job_id, status="failed")

    return OptimizeResponse(job_id=job_id, status="pending")


async def _get_owned_job(job_id: str, user: CustomerRecord) -> dict:
    """Load a job from DB and verify the current user owns it.

    Returns 404 (not 403) to prevent job ID enumeration.
    Legacy jobs without clerk_id are accessible to any authenticated user.
    """
    db = get_database()
    job = await db.jobs.find_one({"job_id": job_id})
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    if job.get("clerk_id") and job["clerk_id"] != user.clerk_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
    return job


@app.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    user: CustomerRecord = Depends(get_current_user),
) -> JobStatusResponse:
    """Get optimization job status, progress, and current best candidate."""
    job = await _get_owned_job(job_id, user)

    return JobStatusResponse(
        job_id=job["job_id"],
        status=job["status"],
        repo_url=job.get("repo_url"),
        config=job.get("config"),
        current_iteration=job.get("current_iteration"),
        total_metric_calls=job.get("total_metric_calls"),
        num_branches=job.get("num_branches"),
        final_candidate=job.get("final_candidate"),
        seed_train_score=job.get("seed_train_score"),
        final_train_score=job.get("final_train_score"),
        seed_test_score=job.get("seed_test_score"),
        final_test_score=job.get("final_test_score"),
        seed_test_scores=job.get("seed_test_scores"),
        final_test_scores=job.get("final_test_scores"),
        error=job.get("error"),
        created_at=job.get("created_at"),
        started_at=job.get("started_at"),
        completed_at=job.get("completed_at"),
        updated_at=job.get("updated_at"),
        name=job.get("name"),
        memory_md=job.get("memory_md"),
        meta_reflection_md=job.get("meta_reflection_md"),
        integrity_audit_md=job.get("integrity_audit_md"),
    )


@app.post("/job/{job_id}/cancel", response_model=CancelJobResponse)
async def cancel_job(
    job_id: str,
    user: CustomerRecord = Depends(get_current_user),
) -> CancelJobResponse:
    """Request a cooperative cancellation of a running job.

    Writes `status="cancelled"`, which the optimizer already polls every
    iteration via `GET /internal/job/{job_id}/check-cancelled`
    (`src/engine/services/callbacks.py`). The loop then exits at the next
    iteration boundary and still finalizes its artifacts -- unlike a SIGKILL,
    which loses the in-flight iteration. Terminal jobs are a no-op.
    """
    job = await _get_owned_job(job_id, user)

    if job["status"] in (
        JobStatus.COMPLETED.value,
        JobStatus.FAILED.value,
        JobStatus.CANCELLED.value,
    ):
        return CancelJobResponse(
            job_id=job_id, status=job["status"], already_terminal=True
        )

    await record_job_status_update(
        job_id,
        JobStatus.CANCELLED.value,
        {
            "status": JobStatus.CANCELLED.value,
            "updated_at": datetime.now(timezone.utc),
        },
    )
    logger.info("Job %s marked cancelled by %s", job_id, user.clerk_id)
    return CancelJobResponse(job_id=job_id, status=JobStatus.CANCELLED.value)


@app.get("/job/{job_id}/state", response_model=JobDetailedStateResponse)
async def get_job_detailed_state(
    job_id: str,
    user: CustomerRecord = Depends(get_current_user),
) -> JobDetailedStateResponse:
    """Get detailed CE state for a job, including all candidates and trajectories."""
    job = await _get_owned_job(job_id, user)

    ce_state = job.get("ce_state") or {}
    branches = job.get("branches") or ce_state.get("branches")
    primary_branch = job.get("primary_branch") or ce_state.get("primary_branch")

    return JobDetailedStateResponse(
        job_id=job["job_id"],
        status=job["status"],
        branches=branches,
        primary_branch=primary_branch,
        final_candidate=job.get("final_candidate"),
        seed_train_score=job.get("seed_train_score"),
        final_train_score=job.get("final_train_score"),
        seed_test_score=job.get("seed_test_score"),
        final_test_score=job.get("final_test_score"),
        seed_test_scores=job.get("seed_test_scores"),
        final_test_scores=job.get("final_test_scores"),
        num_branches=job.get("num_branches"),
        total_metric_calls=job.get("total_metric_calls"),
        num_iterations=ce_state.get("num_iterations"),
    )


@app.get("/jobs", response_model=JobListResponse)
async def list_jobs(
    skip: int = 0,
    limit: int = 20,
    user: CustomerRecord = Depends(get_current_user),
) -> JobListResponse:
    """List all optimization jobs for the authenticated user."""
    db = get_database()
    query = {"clerk_id": user.clerk_id}

    total = await db.jobs.count_documents(query)
    # The list never returns the history or the markdown reports, so don't
    # fetch them (a research job's history alone is hundreds of KB).
    projection = {
        "ce_state": 0, "ce_state_history": 0, "branches": 0,
        "memory_md": 0, "meta_reflection_md": 0, "integrity_audit_md": 0,
    }
    cursor = db.jobs.find(query, projection).sort("created_at", -1).skip(skip).limit(limit)
    jobs = []
    async for doc in cursor:
        jobs.append(JobStatusResponse(
            job_id=doc["job_id"],
            status=doc["status"],
            repo_url=doc.get("repo_url"),
            config=doc.get("config"),
            current_iteration=doc.get("current_iteration"),
            total_metric_calls=doc.get("total_metric_calls"),
            num_branches=doc.get("num_branches"),
            final_candidate=doc.get("final_candidate"),
            seed_train_score=doc.get("seed_train_score"),
            final_train_score=doc.get("final_train_score"),
            seed_test_score=doc.get("seed_test_score"),
            final_test_score=doc.get("final_test_score"),
            seed_test_scores=doc.get("seed_test_scores"),
            final_test_scores=doc.get("final_test_scores"),
            error=doc.get("error"),
            created_at=doc.get("created_at"),
            started_at=doc.get("started_at"),
            completed_at=doc.get("completed_at"),
            updated_at=doc.get("updated_at"),
            # Markdown reports are omitted from the list; see GET /job/{id}.
            name=doc.get("name"),
        ))

    return JobListResponse(jobs=jobs, total=total, skip=skip, limit=limit)


@app.get("/repos", response_model=RepoListResponse)
async def list_repos(
    skip: int = 0,
    limit: int = 50,
    user: CustomerRecord = Depends(get_current_user),
) -> RepoListResponse:
    """List all connected repositories for the authenticated user."""
    db = get_database()
    query = {"clerk_id": user.clerk_id, "status": "connected"}

    total = await db.repos.count_documents(query)
    cursor = db.repos.find(query).sort("created_at", -1).skip(skip).limit(limit)
    repos = []
    async for doc in cursor:
        repos.append(RepoResponse(
            repo_id=doc["repo_id"],
            repo_url=doc["repo_url"],
            github_owner=doc.get("github_owner"),
            github_repo_name=doc.get("github_repo_name"),
            installation_id=doc.get("installation_id"),
            status=doc["status"],
            created_at=doc.get("created_at"),
        ))

    return RepoListResponse(repos=repos, total=total, skip=skip, limit=limit)


@app.get("/job/{job_id}/iterations", response_model=IterationHistoryResponse)
async def get_job_iterations(
    job_id: str,
    user: CustomerRecord = Depends(get_current_user),
) -> IterationHistoryResponse:
    """Get iteration history for a job.

    Returns all iterations (including rejected candidates) as a dict
    keyed by iteration number. Each value is an IterationHistoryRecord.
    """
    job = await _get_owned_job(job_id, user)

    ce_state = job.get("ce_state") or {}
    iteration_history = ce_state.get("iteration_history") or {}

    # Sort by iteration number (keys are strings in MongoDB)
    sorted_history = dict(
        sorted(iteration_history.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else -1)
    )

    return IterationHistoryResponse(
        job_id=job["job_id"],
        status=job["status"],
        iterations=sorted_history,
        total_iterations=len(sorted_history),
    )


@app.post("/import-job-history", response_model=ImportJobHistoryResponse)
async def import_job_history(
    body: ImportJobHistoryRequest,
    user: CustomerRecord = Depends(get_current_user),
) -> ImportJobHistoryResponse:
    """Upload a locally-run job's iteration history into the cloud DB.

    API-key authenticated, so a researcher can push results with their own
    account credentials instead of the raw Mongo key. Upserts the job under
    the caller's clerk_id; if a job with this id already exists and is owned
    by someone else, returns 404 (mirrors `_get_owned_job`'s enumeration
    guard). The history is written in both shapes the portal reads:
    `ce_state_history` (idx-keyed) and `ce_state.iteration_history`. Also
    accepts an optional display `name`, the held-out `seed_test_score` /
    `final_test_score`, and the `memory_md`, `meta_reflection_md` and
    `integrity_audit_md` reports.
    """
    db = get_database()
    existing = await db.jobs.find_one({"job_id": body.job_id})
    if existing and existing.get("clerk_id") and existing["clerk_id"] != user.clerk_id:
        raise HTTPException(status_code=404, detail=f"Job not found: {body.job_id}")

    iteration_history = {str(k): v for k, v in body.iteration_history.items()}
    now = datetime.now(timezone.utc)
    prior_ce_state = (existing or {}).get("ce_state") or {}

    set_fields: dict = {
        "clerk_id": user.clerk_id,
        "ce_state_history": iteration_history,
        "ce_state": {**prior_ce_state, "iteration_history": iteration_history},
        "updated_at": now,
    }
    if body.status:
        set_fields["status"] = body.status
    elif not existing:
        set_fields["status"] = JobStatus.COMPLETED.value
    if body.repo_url:
        set_fields["repo_url"] = body.repo_url
    if body.final_candidate is not None:
        set_fields["final_candidate"] = body.final_candidate
    if body.num_branches is not None:
        set_fields["num_branches"] = body.num_branches
    # Optional fields: skip on None so a re-import that omits one keeps the
    # stored value.
    for field in (
        "memory_md",
        "name",
        "seed_test_score",
        "final_test_score",
        "meta_reflection_md",
        "integrity_audit_md",
    ):
        value = getattr(body, field)
        if value is not None:
            set_fields[field] = value

    result = await db.jobs.update_one(
        {"job_id": body.job_id},
        {"$set": set_fields, "$setOnInsert": {"created_at": now, "imported": True}},
        upsert=True,
    )
    return ImportJobHistoryResponse(
        job_id=body.job_id,
        created=result.upserted_id is not None,
        total_iterations=len(iteration_history),
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.3.0"}


@app.post("/cleanup-branches", response_model=CleanupBranchesResponse)
async def cleanup_branches(
    request: CleanupBranchesRequest,
    user: CustomerRecord = Depends(get_current_user),
) -> CleanupBranchesResponse:
    """Delete all remote 'codeevolver-{timestamp}-*' branches for a given run timestamp."""
    import httpx

    stripped = request.repo_url.replace("https://github.com/", "").rstrip("/").replace(".git", "")
    parts = stripped.split("/")
    if len(parts) < 2:
        raise HTTPException(status_code=400, detail=f"Cannot parse owner/repo from URL: {request.repo_url}")
    owner, repo = parts[0], parts[1]

    installation_id = None
    env_installation_id = os.getenv("GITHUB_TEST_INSTALLATION_ID")
    if env_installation_id:
        try:
            installation_id = int(env_installation_id)
        except ValueError:
            pass

    if not installation_id:
        raise HTTPException(status_code=400, detail="No GitHub installation ID configured. Set GITHUB_TEST_INSTALLATION_ID.")

    try:
        token = GitHubAppService.get_installation_token(installation_id)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"GitHub authentication failed: {e}")

    if not token:
        raise HTTPException(status_code=500, detail="Failed to obtain GitHub installation token")

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    if request.date == "none":
        prefix = "codeevolver-"
    else:
        prefix = f"codeevolver-{request.date}"
    deleted: list[str] = []
    skipped: list[str] = []
    errors: list[str] = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        page = 1
        matching_branches: list[str] = []
        while True:
            resp = await client.get(
                f"https://api.github.com/repos/{owner}/{repo}/branches",
                headers=headers,
                params={"per_page": 100, "page": page},
            )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=resp.status_code,
                    detail=f"GitHub API error listing branches: {resp.text}",
                )
            branches = resp.json()
            if not branches:
                break
            for b in branches:
                name = b["name"]
                if name.startswith(prefix):
                    matching_branches.append(name)
            page += 1

        for branch_name in matching_branches:
            if not branch_name.startswith("codeevolver-"):
                errors.append(f"{branch_name}: SAFETY - refused to delete non-codeevolver branch")
                continue

            if branch_name in request.except_branches:
                skipped.append(branch_name)
                continue

            ref = f"heads/{branch_name}"
            del_resp = await client.delete(
                f"https://api.github.com/repos/{owner}/{repo}/git/refs/{ref}",
                headers=headers,
            )
            if del_resp.status_code in (204, 200):
                deleted.append(branch_name)
            else:
                errors.append(f"{branch_name}: HTTP {del_resp.status_code} - {del_resp.text}")

    return CleanupBranchesResponse(deleted=deleted, skipped=skipped, errors=errors)


# ---------------------------------------------------------------------------
# Internal callback endpoints (called by optimizer sandbox via JWT)
# These use the existing HS256 job-scoped JWT, NOT Clerk auth.
# ---------------------------------------------------------------------------

def _validate_internal_jwt(job_id: str, authorization: str | None) -> None:
    """Extract Bearer token and validate it against the given job_id."""
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
    """Update job status (called by optimizer sandbox via callback)."""
    _validate_internal_jwt(job_id, authorization)

    print(
        f"[STATUS-API] Received status update for {job_id}: status={body.status}, "
        f"has_best_candidate={body.final_candidate is not None}, "
        f"has_ce_state={body.ce_state is not None}",
        flush=True,
    )

    update: dict = {"status": body.status, "updated_at": datetime.now(timezone.utc)}

    if body.status == "running":
        update["started_at"] = datetime.now(timezone.utc)
    if body.status in ("completed", "failed"):
        update["completed_at"] = datetime.now(timezone.utc)

    if body.final_candidate is not None:
        update["final_candidate"] = body.final_candidate
    if body.seed_train_score is not None:
        update["seed_train_score"] = body.seed_train_score
    if body.final_train_score is not None:
        update["final_train_score"] = body.final_train_score
    if body.seed_test_score is not None:
        update["seed_test_score"] = body.seed_test_score
    if body.final_test_score is not None:
        update["final_test_score"] = body.final_test_score
    if body.seed_test_scores is not None:
        update["seed_test_scores"] = body.seed_test_scores
    if body.final_test_scores is not None:
        update["final_test_scores"] = body.final_test_scores
    if body.total_metric_calls is not None:
        update["total_metric_calls"] = body.total_metric_calls
    if body.num_branches is not None:
        update["num_branches"] = body.num_branches
    if body.error is not None:
        update["error"] = body.error

    if body.ce_state is not None:
        update["ce_state"] = body.ce_state
        update["branches"] = body.ce_state.get("branches")
        update["primary_branch"] = body.ce_state.get("primary_branch")

        if body.status in ("completed", "failed"):
            # Engine 3: the loop already ships a ready-to-persist iteration history
            # under `ce_state["iteration_history"]`. We mirror it into
            # `ce_state_history` so the portal UI keeps its existing shape
            # without the v1 conversion path.
            raw_history = body.ce_state.get("iteration_history") or {}
            if isinstance(raw_history, dict):
                update["ce_state_history"] = {
                    str(k): v for k, v in raw_history.items()
                }

    await record_job_status_update(job_id, body.status, update)
    return {"ok": True}


@app.put("/internal/job/{job_id}/progress")
async def internal_update_job_progress(
    job_id: str,
    body: JobProgressUpdateRequest,
    authorization: str | None = Header(default=None),
):
    """Update iteration progress (called each iteration)."""
    _validate_internal_jwt(job_id, authorization)

    print(
        f"[PROGRESS-API] Received progress for {job_id}: "
        f"iteration={body.current_iteration}, "
        f"num_branches={body.num_branches}, total_metric_calls={body.total_metric_calls}, "
        f"has_ce_state={body.ce_state is not None}",
        flush=True,
    )

    update = {
        "current_iteration": body.current_iteration,
        "final_candidate": body.final_candidate,
        "total_metric_calls": body.total_metric_calls,
        "num_branches": body.num_branches,
        "updated_at": datetime.now(timezone.utc),
    }
    if body.memory_md is not None:
        # Rolling overwrite — skip on transient None so a failed sandbox
        # read doesn't wipe the previously stored snapshot.
        update["memory_md"] = body.memory_md

    # Engine 3: the optimizer sends an already-structured iteration history in
    # `ce_state["iteration_history"]` (iteration number -> record). The
    # write helper persists `update` and upserts each iteration under a
    # dotted key so out-of-order callbacks never clobber earlier entries.
    raw_history: dict | None = None
    if body.ce_state is not None:
        candidate = body.ce_state.get("iteration_history") or {}
        if isinstance(candidate, dict):
            raw_history = candidate

    await record_iteration_progress(job_id, update, raw_history)
    return {"ok": True}


@app.get("/internal/job/{job_id}/check-cancelled", response_model=CancelCheckResponse)
async def internal_check_cancelled(
    job_id: str,
    authorization: str | None = Header(default=None),
):
    """Check if a job has been cancelled (polled by progress tracker)."""
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
    """Get a fresh GitHub installation token for the job."""
    from src.api.schemas.requests import GitHubTokenResponse

    _validate_internal_jwt(job_id, authorization)

    db = get_database()
    job = await db.jobs.find_one({"job_id": job_id}, {"config": 1})
    if not job:
        return GitHubTokenResponse(error=f"Job not found: {job_id}")

    config = job.get("config", {})
    installation_id = config.get("installation_id")

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
