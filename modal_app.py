"""Modal application for CodeEvolver Agents.

This is the main entry point for deploying to Modal.

Run locally: modal serve modal_app.py
Deploy:      modal deploy modal_app.py
"""

import modal

# Create or lookup the Modal app
app = modal.App("codeevolver-agents")

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
    )
)

# Base image for sandbox execution (Claude Agent SDK + DSPy)
sandbox_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "gitpython>=3.1.0",
        "dspy>=2.5.0",
        # "claude-agent-sdk",  # Uncomment when available
    )
)


@app.function(
    image=web_image,
    volumes={"/workspaces": workspaces_volume},
    secrets=[modal.Secret.from_name("codeevolver-secrets", required=False)],
    keep_warm=1,  # Keep one instance warm to minimize cold starts
    allow_concurrent_inputs=100,
)
@modal.asgi_app()
def fastapi_app():
    """
    FastAPI web application running on Modal.
    
    This serves the API endpoints and orchestrates sandbox execution.
    """
    import os
    
    # Override workspace root to use Modal volume
    os.environ["CODEEVOLVER_WORKSPACE_ROOT"] = "/workspaces"
    
    # Import after setting env vars
    from src.main import app as fastapi_instance
    
    return fastapi_instance


@app.function(
    image=sandbox_image,
    timeout=600,  # 10 minute max per sandbox execution
    cpu=2,
    memory=4096,
)
async def execute_in_sandbox(
    repo_url: str,
    workspace_path: str,
    mutation: dict,
    client_secrets: dict | None = None,
) -> dict:
    """
    Execute a mutation inside an isolated Modal Sandbox.
    
    The Claude Agent SDK runs INSIDE the sandbox, so its native tools
    (Bash, Grep, Glob, Read, Edit) work via subprocess.
    
    Args:
        repo_url: Git repository URL to clone
        workspace_path: Path where repo should be cloned
        mutation: Mutation configuration dict
        client_secrets: Optional secrets to inject as env vars
    
    Returns:
        Execution result dict
    """
    import json
    import subprocess
    
    result = {"status": "success", "output": None, "error": None}
    
    try:
        # Clone the repository
        subprocess.run(
            ["git", "clone", repo_url, workspace_path],
            check=True,
            capture_output=True,
            text=True,
        )
        
        # Install dependencies if requirements.txt exists
        requirements_path = f"{workspace_path}/requirements.txt"
        import os
        if os.path.exists(requirements_path):
            subprocess.run(
                ["pip", "install", "-r", requirements_path],
                check=True,
                capture_output=True,
                text=True,
            )
        
        # Inject secrets as environment variables
        if client_secrets:
            for key, value in client_secrets.items():
                os.environ[key] = value
        
        # For code mutations, run Claude Agent SDK
        if mutation.get("mutation_type") == "code":
            # TODO: Implement Claude Agent SDK execution
            # The agent runs in this process, so its native tools work
            #
            # from claude_agent_sdk import query, ClaudeAgentOptions
            # 
            # for message in query(
            #     prompt=mutation["change_request"],
            #     options=ClaudeAgentOptions(
            #         cwd=workspace_path,
            #         allowed_tools=["Bash", "Read", "Edit", "Glob", "Grep"],
            #         permission_mode="acceptEdits",
            #     )
            # ):
            #     pass
            
            result["error"] = "Code mutations not yet implemented"
            result["status"] = "failed"
        
        # For prompt mutations, apply directly
        elif mutation.get("mutation_type") == "prompt":
            program_json_path = f"{workspace_path}/{mutation['program_json_path']}"
            
            # Load program.json
            with open(program_json_path) as f:
                program_json = json.load(f)
            
            # Apply mutation
            candidate = mutation.get("candidate", {})
            for component_name, new_instruction in candidate.items():
                if component_name in program_json:
                    program_json[component_name]["signature"]["instructions"] = new_instruction
            
            # Save modified program.json
            with open(program_json_path, "w") as f:
                json.dump(program_json, f, indent=2)
            
            # Commit changes
            subprocess.run(
                ["git", "-C", workspace_path, "add", "-A"],
                check=True,
            )
            subprocess.run(
                ["git", "-C", workspace_path, "commit", "-m", 
                 f"Apply prompt mutation: {mutation.get('program_id', 'unknown')}"],
                check=True,
            )
            
            result["program_json"] = program_json
        
        # TODO: Run DSPy program and capture outputs
        # This would load the entry_point module and run it on test_examples
        
    except subprocess.CalledProcessError as e:
        result["status"] = "failed"
        result["error"] = f"Command failed: {e.cmd}\nstderr: {e.stderr}"
    except Exception as e:
        result["status"] = "failed"
        result["error"] = str(e)
    
    return result


@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("CodeEvolver Agents Modal App")
    print("----------------------------")
    print("Commands:")
    print("  modal serve modal_app.py    # Run locally with hot reload")
    print("  modal deploy modal_app.py   # Deploy to Modal cloud")
    print()
    print("Once deployed, the API will be available at:")
    print("  https://<your-modal-username>--codeevolver-agents-fastapi-app.modal.run")
