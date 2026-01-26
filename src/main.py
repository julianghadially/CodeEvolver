"""FastAPI application for CodeEvolver Agents.

Run locally:  uvicorn src.main:app --reload
Run on Modal: modal serve modal_app.py (or modal deploy modal_app.py)
"""

import os

from fastapi import FastAPI, HTTPException

from .db import get_database, lifespan
from .schemas import (
    ConnectGitRequest,
    ConnectGitResponse,
    ClientRecord,
    ExecuteStepRequest,
    ExecuteStepResponse,
    ExecuteSandboxRequest,
    ExecuteSandboxResponse,
    PipelineOutput,
    ProgramRecord,
    ProgramStatus,
    MutationType,
    GetProgramResponse,
    OptimizeRequest,
    OptimizeResponse,
    JobStatusResponse,
    JobRecord,
    JobStatus,
)
from .services import GitService
from .core import (
    apply_prompt_mutation,
    load_program_json,
    save_program_json,
    run_program,
)

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


@app.post("/execute_step", response_model=ExecuteStepResponse)
async def execute_step(request: ExecuteStepRequest) -> ExecuteStepResponse:
    """
    Execute one optimization step.

    Applies a mutation (prompt or code), runs the program, and returns output
    for GEPA reward calculation.
    """
    db = get_database()

    # Verify client exists
    client = await db.clients.find_one({"client_id": request.client_id})
    if not client:
        raise HTTPException(status_code=404, detail=f"Client not found: {request.client_id}")

    # Determine parent branch
    parent_branch = "main"
    if request.parent_program_id:
        parent_program = await db.programs.find_one({"program_id": request.parent_program_id})
        if parent_program and parent_program.get("branch_name"):
            parent_branch = parent_program["branch_name"]

    # Create worktree for this mutation
    try:
        worktree_path, branch_name = GitService.create_worktree(
            request.client_id,
            request.program_id,
            parent_branch,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create initial program record
    program_record = ProgramRecord(
        client_id=request.client_id,
        program_id=request.program_id,
        parent_program_id=request.parent_program_id,
        program_json={},
        branch_name=branch_name,
        status=ProgramStatus.IN_PROGRESS,
    )
    await db.programs.insert_one(program_record.model_dump())

    try:
        # Load program.json
        program_json = load_program_json(
            worktree_path,
            request.program_json_path,
        )

        # Apply mutation
        if request.mutation_type == MutationType.PROMPT:
            if not request.candidate:
                raise HTTPException(
                    status_code=400,
                    detail="candidate is required for prompt mutations",
                )
            program_json = apply_prompt_mutation(
                program_json,
                request.candidate,
            )
            # Save mutated program.json
            save_program_json(
                worktree_path,
                request.program_json_path,
                program_json,
            )
            # Commit changes
            GitService.commit_changes(
                worktree_path,
                f"Apply prompt mutation for program {request.program_id}",
            )

        elif request.mutation_type == MutationType.CODE:
            if not request.change_request:
                raise HTTPException(
                    status_code=400,
                    detail="change_request is required for code mutations",
                )
            # Code mutations require sandbox execution
            # For now, return 501 until sandbox integration is complete
            raise HTTPException(
                status_code=501,
                detail="Code mutations require sandbox execution. Use Modal deployment.",
            )

        # Run program on test examples
        run_result = await run_program(
            worktree_path,
            request.program_json_path,
            request.entry_point,
            request.test_examples,
            request.capture_traces,
        )

        # Update program record with success
        await db.programs.update_one(
            {"program_id": request.program_id},
            {
                "$set": {
                    "program_json": program_json,
                    "status": ProgramStatus.COMPLETED.value,
                }
            },
        )

        return ExecuteStepResponse(
            program_id=request.program_id,
            status="success",
            pipeline_outputs=[
                PipelineOutput(example_id=o.example_id, output=o.output)
                for o in run_result.outputs
            ],
            traces=run_result.traces,
            branch_name=branch_name,
            program_json=program_json,
        )

    except FileNotFoundError as e:
        await db.programs.update_one(
            {"program_id": request.program_id},
            {"$set": {"status": ProgramStatus.FAILED.value}},
        )
        return ExecuteStepResponse(
            program_id=request.program_id,
            status="failed",
            error=str(e),
            branch_name=branch_name,
        )

    except KeyError as e:
        await db.programs.update_one(
            {"program_id": request.program_id},
            {"$set": {"status": ProgramStatus.FAILED.value}},
        )
        return ExecuteStepResponse(
            program_id=request.program_id,
            status="failed",
            error=f"Mutation error: {e}",
            branch_name=branch_name,
        )

    except Exception as e:
        await db.programs.update_one(
            {"program_id": request.program_id},
            {"$set": {"status": ProgramStatus.FAILED.value}},
        )
        return ExecuteStepResponse(
            program_id=request.program_id,
            status="failed",
            error=f"Unexpected error: {e}",
            branch_name=branch_name,
        )

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

    # Clone repository
    try:
        workspace_path = GitService.clone_repository(
            request.repo_url,
            client_id,
            installation_id=request.installation_id,
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

    # Spawn Modal function for optimization (fire-and-forget)
    try:
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        from modal_app import run_optimization

        run_optimization.spawn(
            job_id=job_id,
            repo_url=request.repo_url,
            program_json_path=request.program_json_path,
            entry_point=request.entry_point,
            metric_path=request.metric_path,
            metric_fn_name=request.metric_fn_name,
            trainset_json=request.trainset,
            valset_json=request.valset,
            task_lm=request.task_lm,
            reflection_lm=request.reflection_lm,
            max_metric_calls=request.max_metric_calls,
            installation_id=request.installation_id,
            input_keys=request.input_keys,
            num_threads=request.num_threads,
            seed=request.seed,
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


@app.post("/execute_sandbox", response_model=ExecuteSandboxResponse)
async def execute_sandbox(request: ExecuteSandboxRequest) -> ExecuteSandboxResponse:
    """
    Execute a mutation in the Modal sandbox.
    
    This endpoint directly invokes the Modal sandbox function for code mutations.
    It's useful for testing and for cases where you want to bypass the /execute_step
    workflow.
    
    The sandbox will:
    1. Clone the repository
    2. Create a branch
    3. Apply the code mutation (using Claude Agent)
    4. Optionally push to remote
    5. Return the result
    """
    try:
        # Import the execute_in_sandbox function from modal_app
        # This works because we're running inside the Modal context
        import sys
        if "/app" not in sys.path:
            sys.path.insert(0, "/app")
        
        from modal_app import execute_in_sandbox
        
        # Call the sandbox function remotely
        result = await execute_in_sandbox.remote.aio(
            client_id=request.client_id,
            program_id=request.program_id,
            repo_url=request.repo_url,
            mutation_type=request.mutation_type,
            program_json_path=request.program_json_path,
            entry_point=request.entry_point,
            candidate=request.candidate,
            change_request=request.change_request,
            change_location=request.change_location,
            test_examples=request.test_examples,
            capture_traces=request.capture_traces,
            installation_id=request.installation_id,
            skip_program_run=request.skip_program_run,
            branch_name=request.branch_name,
            push_to_remote=request.push_to_remote,
        )
        
        return ExecuteSandboxResponse(
            status=result.get("status", "failed"),
            program_id=result.get("program_id"),
            program_json=result.get("program_json"),
            pipeline_outputs=result.get("pipeline_outputs"),
            traces=result.get("traces"),
            branch_name=result.get("branch_name"),
            error=result.get("error"),
        )
        
    except ImportError as e:
        # Not running on Modal - provide helpful error
        return ExecuteSandboxResponse(
            status="failed",
            error=(
                f"Cannot import modal_app: {e}. "
                "This endpoint only works when running on Modal (modal serve or modal deploy)."
            ),
        )
    except Exception as e:
        return ExecuteSandboxResponse(
            status="failed",
            error=f"Sandbox execution failed: {e}",
        )
