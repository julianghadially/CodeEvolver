"""Modal application for CodeEvolver Agents.

This is the main entry point for deploying to Modal.

Run on Modal: modal serve modal_app.py  # Runs on Modal cloud with local code mounts
Deploy:       modal deploy modal_app.py

The architecture follows the modal-vibe pattern:
- FastAPI serves the API endpoints
- SandboxApp manages isolated sandbox execution for mutations
- Core logic lives in src/core/
"""

import modal
from modal import FilePatternMatcher

# Create or lookup the Modal app
app = modal.App("codeevolver")

# Shared volume for git workspaces (persists across function calls)
workspaces_volume = modal.Volume.from_name(
    "codeevolver-workspaces",
    create_if_missing=True,
)

# Base image for the FastAPI web server
web_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "fastapi>=0.115.0",
        "motor>=3.6.0",
        "pydantic-settings>=2.6.0",
        "gitpython>=3.1.0",
        "httpx>=0.27.0",
        "pyjwt[cryptography]>=2.8.0",
        "cryptography>=41.0.0",
        "certifi",
    )
    .add_local_dir(".", remote_path="/app", ignore=FilePatternMatcher.from_file(".modalignore"),)
)

# Base image for sandbox execution (Claude Agent SDK + DSPy)
# This is used by SandboxApp.create() from src/core/sandbox.py
# Includes httpx and pyjwt for GitHubAppService authentication
#
# IMPORTANT: The Claude Agent SDK requires the Claude Code CLI (Node.js)
# to be installed. The Python SDK is just a wrapper that spawns the CLI.
sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential", "nodejs", "npm")
    .run_commands(
        # Install Claude Code CLI globally - required for claude-agent-sdk
        "npm install -g @anthropic-ai/claude-code",
    )
    .pip_install(
        "gitpython>=3.1.0",
        "dspy>=2.5.0",
        "claude-agent-sdk>=0.1.21",
        "httpx>=0.27.0",
        "pyjwt[cryptography]>=2.8.0",
        "pydantic-settings>=2.6.0",
    )
    .add_local_dir(".", remote_path="/app", ignore=FilePatternMatcher.from_file(".modalignore"),)
)


@app.function(
    image=web_image,
    volumes={"/workspaces": workspaces_volume},
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
    min_containers=1,
    timeout=900,  # Allow enough time for change_request calls (600s) plus overhead
)
@modal.concurrent(max_inputs=100)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI web application running on Modal.

    Serves the API endpoints and orchestrates sandbox execution.
    """
    import os
    import sys

    os.chdir("/app")
    if "/app" not in sys.path:
        sys.path.insert(0, "/app")

    os.environ["CODEEVOLVER_WORKSPACE_ROOT"] = "/workspaces"

    from src.main import app as fastapi_instance

    return fastapi_instance


@app.function(
    image=sandbox_image,  # Has claude-agent-sdk, git, httpx, pyjwt
    timeout=600,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
)
def execute_change_request(
    repo_url: str,
    change_request: str,
    change_location: str | None = None,
    branch_name: str | None = None,
    push_to_remote: bool = False,
    github_token: str | None = None,
) -> dict:
    """Execute a code change using GEPASandbox and Claude coding agent.

    Returns:
        Dict with 'success', 'branch_name', 'error', 'output' keys.
    """
    import os
    import sys
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from src.gepa.gepa_sandbox import GEPASandbox
    from src.services.git_sandbox import SandboxGitService

    # Create sandbox with pre-generated token
    sandbox = GEPASandbox(
        app=app,
        repo_url=repo_url,
        github_token=github_token,
        timeout=600,
    )

    try:
        # Start with venv isolation
        sandbox.start(use_venv=True)

        # Create branch if specified
        if branch_name:
            git = SandboxGitService(sandbox._sandbox, sandbox._workspace)
            git.configure_user()
            git.checkout(branch_name, create=True)

        # Execute agent
        result = sandbox.exec_agent(
            change_request=change_request,
            change_location=change_location,
            commit_changes=True,
        )

        # Push if requested (use authenticated push for fresh token)
        if result["success"] and push_to_remote and branch_name:
            push_result = sandbox.push_authenticated(branch_name)
            if not push_result.get("success"):
                return {
                    "success": False,
                    "branch_name": branch_name,
                    "error": f"Push failed: {push_result.get('stderr')}",
                    "output": result.get("output"),
                }

        return {
            "success": result["success"],
            "branch_name": branch_name,
            "error": result.get("error"),
            "output": result.get("output"),
        }

    finally:
        sandbox.stop()


# DEPRECATED: Keep execute_in_sandbox for backwards compatibility
# Will be removed in a future version - use execute_change_request instead
@app.function(
    image=sandbox_image,
    timeout=600,
    cpu=2,
    memory=4096,
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
)
async def execute_in_sandbox(
    client_id: str,
    program_id: str,
    repo_url: str,
    mutation_type: str,
    program_json_path: str,
    entry_point: str,
    candidate: dict | None = None,
    change_request: str | None = None,
    change_location: str | None = None,
    test_examples: list | None = None,
    capture_traces: bool = False,
    installation_id: int | None = None,
    skip_program_run: bool = False,
    branch_name: str | None = None,
    push_to_remote: bool = False,
) -> dict:
    """
    DEPRECATED: Use execute_change_request instead.

    Execute a mutation inside an isolated Modal Sandbox.
    """
    import os
    import sys

    sys.path.insert(0, "/app")

    from src.core import execute_mutation

    # Build secrets dict from environment
    secrets = {}
    if os.getenv("ANTHROPIC_API_KEY"):
        secrets["ANTHROPIC_API_KEY"] = os.getenv("ANTHROPIC_API_KEY")
    if os.getenv("OPENAI_KEY"):
        secrets["OPENAI_KEY"] = os.getenv("OPENAI_KEY")

    result = await execute_mutation(
        app=app,
        client_id=client_id,
        program_id=program_id,
        repo_url=repo_url,
        mutation_type=mutation_type,
        program_json_path=program_json_path,
        entry_point=entry_point,
        candidate=candidate,
        change_request=change_request,
        change_location=change_location,
        test_examples=test_examples,
        capture_traces=capture_traces,
        secrets=secrets,
        installation_id=installation_id,
        skip_program_run=skip_program_run,
        branch_name=branch_name,
        push_to_remote=push_to_remote,
    )

    # Convert dataclass to dict for serialization
    return {
        "status": result.status,
        "program_id": result.program_id,
        "program_json": result.program_json,
        "pipeline_outputs": result.pipeline_outputs,
        "traces": result.traces,
        "branch_name": result.branch_name,
        "error": result.error,
    }


# Image for GEPA optimization orchestrator (litellm, gepa — NO dspy).
# DSPy and client deps are installed inside the eval sandbox instead.
# DB drivers (pymongo/motor) are NOT needed here — progress is reported via HTTP callbacks.
gepa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pydantic-settings>=2.6.0",
        "gitpython>=3.1.0",
        "litellm>=1.64.0",
        "httpx>=0.27.0",
        "pyjwt[cryptography]>=2.8.0",
        "tqdm>=4.66.1",
        "gepa>=0.0.26",
    )
    .add_local_dir(".", remote_path="/app", ignore=FilePatternMatcher.from_file(".modalignore"),)
)


# NOTE: timeout must match settings.gepa_job_timeout in src/config.py
# Decorator values are evaluated at import time, so we can't use settings here
@app.function(
    image=gepa_image,
    volumes={"/workspaces": workspaces_volume},
    secrets=[modal.Secret.from_name("codeevolver-worker-secrets")],
    timeout=43200,  # 12 hours - KEEP IN SYNC with src/config.py gepa_job_timeout
    cpu=4,
    memory=8192,
    nonpreemptible=True,  # Prevent mid-optimization restarts (3x CPU/memory cost)
)
def run_optimization(
    job_id: str,
    repo_url: str,
    program: str,
    metric: str,
    saved_program_json_path: str | None = None,
    trainset_json: list | None = None,
    trainset_path: str | None = None,
    valset_json: list | None = None,
    valset_path: str | None = None,
    program_lm: str = "openai/gpt-5-mini",
    reflection_lm: str = "openai/gpt-5-mini",
    max_metric_calls: int = 1000,
    github_token: str | None = None,
    input_keys: list[str] | None = None,
    num_threads: int = 1,
    seed: int = 0,
    callback_url: str = "",
    jwt_token: str = "",
    python_version: str = "3.11",
    additional_instructions: str | None = None,
    initial: int | None = None,
    decay_rate: int = 25,
    decay_factor: int = 2,
    code_cutoff_step: int | None = None,
    initial_branch: str = "main",
) -> dict:
    """Run GEPA optimization in a dedicated Modal function.

    This is a long-running synchronous function. GEPA's optimize() blocks
    until completion, cancellation, or budget exhaustion.

    Client code (DSPy programs, metrics) runs in a separate Modal Sandbox
    to avoid dependency conflicts with the orchestrator.
    """
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")

    from src.services.github_app import GitHubAppService
    from src.gepa.gepa_sandbox import GEPASandbox
    from src.gepa.optimizer import run_gepa_optimization

    # Clone repo to workspace (for dataset file access by the orchestrator)
    workspace_path = f"/workspaces/gepa_{job_id}/main"
    os.makedirs(f"/workspaces/gepa_{job_id}", exist_ok=True)

    # Handle private repo authentication using pre-generated token
    authenticated_url = repo_url
    if github_token:
        authenticated_url = GitHubAppService.get_authenticated_repo_url(
            repo_url, github_token
        )

    # Clone (orchestrator needs local copy for dataset file reading)
    subprocess.run(
        ["git", "clone", authenticated_url, workspace_path],
        check=True,
    )

    # Create and start the GEPA sandbox (client deps installed there, not here)
    # Pass github_token and callback info for token refresh (tokens expire after 1 hour)
    # Import settings here (runtime) since decorator can't use imports
    from src.config import settings
    sandbox = GEPASandbox(
        app=app,
        repo_url=repo_url,
        github_token=github_token,
        timeout=settings.gepa_job_timeout,  # Must match function timeout
        callback_url=callback_url,
        jwt_token=jwt_token,
        job_id=job_id,
    )
    sandbox.start(python_version=python_version, branch=initial_branch)

    try:
        result = run_gepa_optimization(
            job_id=job_id,
            callback_url=callback_url,
            jwt_token=jwt_token,
            workspace_path=workspace_path,
            program=program,
            metric=metric,
            reflection_lm=reflection_lm,
            max_metric_calls=max_metric_calls,
            sandbox_manager=sandbox,
            saved_program_json_path=saved_program_json_path,
            trainset_json=trainset_json,
            trainset_path=trainset_path,
            valset_json=valset_json,
            valset_path=valset_path,
            input_keys=input_keys,
            num_threads=num_threads,
            seed=seed,
            program_lm=program_lm,
            additional_instructions=additional_instructions,
            initial=initial,
            decay_rate=decay_rate,
            decay_factor=decay_factor,
            code_cutoff_step=code_cutoff_step,
            initial_branch=initial_branch,
        )
        return result
    finally:
        sandbox.stop()


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("CodeEvolver Agents Modal App")
    print("----------------------------")
    print("Commands:")
    print("  modal serve modal_app.py    # Run on Modal with hot reload")
    print("  modal deploy modal_app.py   # Deploy to Modal cloud")
    print()
    print("Once deployed, the API will be available at:")
    print("  https://<your-modal-username>--codeevolver-fastapi-app.modal.run")
