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
    )
    .add_local_dir(".", remote_path="/app")
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
    .add_local_dir(".", remote_path="/app")
)


@app.function(
    image=web_image,
    volumes={"/workspaces": workspaces_volume},
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
    min_containers=1,
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
    Execute a mutation inside an isolated Modal Sandbox.

    This function is called remotely from the FastAPI endpoints.
    It creates a SandboxApp instance and runs the mutation workflow.
    Uses GitHubAppService for private repository authentication.

    Args:
        client_id: Client identifier
        program_id: Program identifier
        repo_url: Git repository URL to clone
        mutation_type: "prompt" or "code"
        program_json_path: Path to program.json
        entry_point: DSPy module class
        candidate: For prompt mutations
        change_request: For code mutations
        change_location: Optional hint for code mutations
        test_examples: Examples to run
        capture_traces: Whether to capture traces
        installation_id: Optional GitHub App installation ID for private repos
        skip_program_run: If True, skip running the DSPy program (code-only mode)
        branch_name: Optional branch name to use (otherwise auto-generated)
        push_to_remote: If True, push changes to remote after mutation

    Returns:
        Execution result dict
    """
    import os

    # Import core module
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


# Image for GEPA optimization (DSPy, litellm, pymongo, gepa)
# Uses pymongo (sync) because GEPA's optimize() loop is synchronous.
gepa_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "pymongo>=4.6.0",
        "pydantic-settings>=2.6.0",
        "gitpython>=3.1.0",
        "dspy>=2.5.0",
        "litellm>=1.64.0",
        "httpx>=0.27.0",
        "pyjwt[cryptography]>=2.8.0",
        "tqdm>=4.66.1",
    )
    # Development: install GEPA from local source
    # Production: replace with .pip_install("gepa @ git+https://github.com/<org>/GEPA-CodeEvolver.git")
    .add_local_dir(
        "/Users/julianghadially/Documents/0. Fine Tuning as a Service/2. Product/GEPA-CodeEvolver",
        remote_path="/gepa-src",
    )
    .run_commands("pip install -e /gepa-src")
    .add_local_dir(".", remote_path="/app")
)


@app.function(
    image=gepa_image,
    volumes={"/workspaces": workspaces_volume},
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
    timeout=3600,
    cpu=4,
    memory=8192,
)
def run_optimization(
    job_id: str,
    repo_url: str,
    program_json_path: str,
    entry_point: str,
    metric_path: str,
    metric_fn_name: str,
    trainset_json: list,
    valset_json: list | None = None,
    task_lm: str = "openai/gpt-5-mini",
    reflection_lm: str = "openai/gpt-5-mini",
    max_metric_calls: int = 1000,
    installation_id: int | None = None,
    input_keys: list[str] | None = None,
    num_threads: int = 1,
    seed: int = 0,
) -> dict:
    """Run GEPA optimization in a dedicated Modal function.

    This is a long-running synchronous function. GEPA's optimize() blocks
    until completion, cancellation, or budget exhaustion.
    """
    import os
    import subprocess
    import sys

    sys.path.insert(0, "/app")
    os.chdir("/app")

    from src.config import settings
    from src.services.github_app import GitHubAppService

    # Clone repo to workspace
    workspace_path = f"/workspaces/gepa_{job_id}/main"
    os.makedirs(f"/workspaces/gepa_{job_id}", exist_ok=True)

    # Handle private repo authentication
    authenticated_url = repo_url
    if installation_id:
        token = GitHubAppService.get_installation_token(installation_id)
        if token:
            authenticated_url = GitHubAppService.get_authenticated_repo_url(
                repo_url, token
            )

    # Clone
    subprocess.run(
        ["git", "clone", authenticated_url, workspace_path],
        check=True,
    )

    # Install user's requirements.txt if present
    req_path = os.path.join(workspace_path, "requirements.txt")
    if os.path.exists(req_path):
        subprocess.run(
            ["pip", "install", "-r", req_path],
            check=True,
        )

    # API keys are already available via Modal secrets in the environment
    from src.gepa.optimizer import run_gepa_optimization

    mongodb_url = os.getenv("CODEEVOLVER_MONGODB_URL", settings.mongodb_url)

    result = run_gepa_optimization(
        job_id=job_id,
        workspace_path=workspace_path,
        program_json_path=program_json_path,
        entry_point=entry_point,
        metric_path=metric_path,
        metric_fn_name=metric_fn_name,
        trainset_json=trainset_json,
        valset_json=valset_json,
        task_lm=task_lm,
        reflection_lm=reflection_lm,
        max_metric_calls=max_metric_calls,
        mongodb_url=mongodb_url,
        database_name=settings.database_name,
        input_keys=input_keys,
        num_threads=num_threads,
        seed=seed,
    )

    return result


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
