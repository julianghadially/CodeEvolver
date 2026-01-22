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
    PipelineOutput,
    ProgramRecord,
    ProgramStatus,
    MutationType,
    GetProgramResponse,
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


@app.get("/program/{program_id}", response_model=GetProgramResponse)
async def get_program(program_id: str) -> GetProgramResponse:
    """
    Retrieve program details and program_json.
    """
    db = get_database()

    program = await db.programs.find_one({"program_id": program_id})
    if not program:
        raise HTTPException(status_code=404, detail=f"Program not found: {program_id}")

    return GetProgramResponse(
        program_id=program["program_id"],
        client_id=program["client_id"],
        parent_program_id=program["parent_program_id"],
        program_json=program["program_json"],
        branch_name=program["branch_name"],
        status=program["status"],
        created_at=program["created_at"],
    )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "version": "0.2.0"}
