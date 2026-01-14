"""Service for executing mutations in Modal Sandbox.

The Claude Agent SDK runs INSIDE the sandbox, so its native tools
(Bash, Grep, Glob, Read, Edit) work via subprocess.
"""

import json
import os
from pathlib import Path
from typing import Any

import modal

from ..config import settings


# Base image for sandbox execution
SANDBOX_IMAGE = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "curl", "build-essential")
    .pip_install(
        "gitpython>=3.1.0",
        "dspy>=2.5.0",
        # "claude-agent-sdk",  # Uncomment when available
    )
)


class SandboxExecutor:
    """Execute mutations in isolated Modal Sandbox environments."""
    
    def __init__(self):
        """Initialize the sandbox executor."""
        self.app = modal.App.lookup(
            settings.modal_app_name, 
            create_if_missing=True
        )
    
    async def execute_mutation(
        self,
        repo_url: str,
        client_id: str,
        program_id: str,
        mutation_type: str,
        program_json_path: str,
        entry_point: str,
        candidate: dict[str, str] | None = None,
        change_request: str | None = None,
        change_location: str | None = None,
        test_examples: list[dict[str, Any]] | None = None,
        capture_traces: bool = False,
        client_secrets: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Execute a mutation in an isolated Modal Sandbox.
        
        Args:
            repo_url: Git repository URL
            client_id: Client identifier
            program_id: Program identifier for this mutation
            mutation_type: "prompt" or "code"
            program_json_path: Path to program.json from repo root
            entry_point: DSPy module class to run
            candidate: For prompt mutations - component -> instruction mapping
            change_request: For code mutations - natural language description
            change_location: For code mutations - optional hint
            test_examples: Examples to run after mutation
            capture_traces: Whether to capture execution traces
            client_secrets: Secrets to inject as env vars in sandbox
        
        Returns:
            Execution result with status, outputs, traces, etc.
        """
        # Build secrets dict (from config + client-specific)
        secrets = {}
        if settings.anthropic_api_key:
            secrets["ANTHROPIC_API_KEY"] = settings.anthropic_api_key
        if settings.openai_api_key:
            secrets["OPENAI_API_KEY"] = settings.openai_api_key
        if client_secrets:
            secrets.update(client_secrets)
        
        # Create the sandbox
        sandbox = modal.Sandbox.create(
            app=self.app,
            image=SANDBOX_IMAGE,
            timeout=settings.sandbox_timeout,
            cpu=settings.sandbox_cpu,
            memory=settings.sandbox_memory,
        )
        
        workspace = "/workspace"
        
        try:
            # 1. Clone the repository
            p = sandbox.exec("git", "clone", repo_url, workspace)
            p.wait()
            if p.returncode != 0:
                return {
                    "status": "failed",
                    "error": f"Git clone failed: {p.stderr.read()}",
                }
            
            # 2. Install dependencies if requirements.txt exists
            p = sandbox.exec(
                "bash", "-c",
                f"if [ -f {workspace}/requirements.txt ]; then "
                f"pip install -r {workspace}/requirements.txt; fi"
            )
            p.wait()
            
            # 3. Inject secrets as environment file
            if secrets:
                env_content = "\n".join(f"export {k}='{v}'" for k, v in secrets.items())
                sandbox.exec(
                    "bash", "-c", 
                    f"echo '{env_content}' > {workspace}/.env"
                ).wait()
            
            # 4. Apply mutation
            if mutation_type == "prompt":
                result = await self._apply_prompt_mutation(
                    sandbox, workspace, program_json_path, candidate, program_id
                )
            elif mutation_type == "code":
                result = await self._apply_code_mutation(
                    sandbox, workspace, change_request, change_location, program_id
                )
            else:
                return {"status": "failed", "error": f"Unknown mutation type: {mutation_type}"}
            
            if result["status"] == "failed":
                return result
            
            # 5. Run the program on test examples
            outputs, traces = await self._run_program(
                sandbox, workspace, program_json_path, entry_point,
                test_examples or [], capture_traces
            )
            
            result["pipeline_outputs"] = outputs
            result["traces"] = traces
            
            return result
            
        except Exception as e:
            return {"status": "failed", "error": str(e)}
        
        finally:
            sandbox.terminate()
    
    async def _apply_prompt_mutation(
        self,
        sandbox: modal.Sandbox,
        workspace: str,
        program_json_path: str,
        candidate: dict[str, str] | None,
        program_id: str,
    ) -> dict[str, Any]:
        """Apply a prompt mutation to program.json."""
        if not candidate:
            return {"status": "failed", "error": "candidate required for prompt mutation"}
        
        full_path = f"{workspace}/{program_json_path}"
        
        # Read program.json
        p = sandbox.exec("cat", full_path)
        p.wait()
        if p.returncode != 0:
            return {"status": "failed", "error": f"program.json not found: {full_path}"}
        
        program_json = json.loads(p.stdout.read())
        
        # Apply mutations
        for component_name, new_instruction in candidate.items():
            if component_name not in program_json:
                return {"status": "failed", "error": f"Component not found: {component_name}"}
            if "signature" not in program_json[component_name]:
                return {"status": "failed", "error": f"No signature in component: {component_name}"}
            program_json[component_name]["signature"]["instructions"] = new_instruction
        
        # Write modified program.json
        modified_json = json.dumps(program_json, indent=2)
        sandbox.exec(
            "bash", "-c",
            f"cat > {full_path} << 'EOFPROGRAMJSON'\n{modified_json}\nEOFPROGRAMJSON"
        ).wait()
        
        # Commit changes
        sandbox.exec("git", "-C", workspace, "add", "-A").wait()
        sandbox.exec(
            "git", "-C", workspace, "commit", "-m",
            f"Apply prompt mutation for program {program_id}"
        ).wait()
        
        return {"status": "success", "program_json": program_json}
    
    async def _apply_code_mutation(
        self,
        sandbox: modal.Sandbox,
        workspace: str,
        change_request: str | None,
        change_location: str | None,
        program_id: str,
    ) -> dict[str, Any]:
        """
        Apply a code mutation using Claude Agent SDK.
        
        The agent runs INSIDE the sandbox, so native tools work.
        """
        if not change_request:
            return {"status": "failed", "error": "change_request required for code mutation"}
        
        # Build the agent script that will run inside the sandbox
        # The Claude Agent SDK's native tools (Bash, Grep, etc.) will work
        # because they use subprocess which runs in the sandbox
        agent_script = f'''
import os
import sys

# Source environment file for secrets
if os.path.exists("{workspace}/.env"):
    with open("{workspace}/.env") as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                key, _, value = line.strip().partition("=")
                key = key.replace("export ", "")
                value = value.strip("'\"")
                os.environ[key] = value

try:
    from claude_agent_sdk import query, ClaudeAgentOptions
    
    change_request = """{change_request}"""
    change_location = """{change_location or ''}"""
    
    prompt = change_request
    if change_location:
        prompt = f"Focus on {{change_location}}. " + prompt
    
    for message in query(
        prompt=prompt,
        options=ClaudeAgentOptions(
            cwd="{workspace}",
            allowed_tools=["Bash", "Read", "Edit", "Glob", "Grep"],
            permission_mode="acceptEdits",
        )
    ):
        pass
    
    print("SUCCESS")
    
except ImportError:
    print("ERROR: claude-agent-sdk not installed")
    sys.exit(1)
except Exception as e:
    print(f"ERROR: {{e}}")
    sys.exit(1)
'''
        
        # Write and run the agent script
        sandbox.exec(
            "bash", "-c",
            f"cat > /tmp/agent_script.py << 'EOFAGENT'\n{agent_script}\nEOFAGENT"
        ).wait()
        
        p = sandbox.exec("python", "/tmp/agent_script.py")
        p.wait()
        output = p.stdout.read()
        
        if p.returncode != 0 or "ERROR" in output:
            return {
                "status": "failed",
                "error": f"Code mutation failed: {output}\n{p.stderr.read()}"
            }
        
        # Commit any changes made by the agent
        sandbox.exec("git", "-C", workspace, "add", "-A").wait()
        sandbox.exec(
            "git", "-C", workspace, "commit", "-m",
            f"Apply code mutation for program {program_id}: {change_request[:50]}..."
        ).wait()
        
        return {"status": "success"}
    
    async def _run_program(
        self,
        sandbox: modal.Sandbox,
        workspace: str,
        program_json_path: str,
        entry_point: str,
        test_examples: list[dict[str, Any]],
        capture_traces: bool,
    ) -> tuple[list[dict[str, Any]], list[Any] | None]:
        """
        Run the DSPy program on test examples inside the sandbox.
        
        TODO: Full DSPy runtime integration
        """
        # Placeholder - return mock outputs
        # Full implementation would:
        # 1. Import the entry_point module
        # 2. Load program state from program.json
        # 3. Run program.forward() on each example
        # 4. Capture traces if requested
        
        outputs = []
        for i, example in enumerate(test_examples):
            outputs.append({
                "example_id": i,
                "output": {
                    "_placeholder": True,
                    "_note": "DSPy runtime integration pending",
                    "input": example,
                },
            })
        
        traces = [] if capture_traces else None
        
        return outputs, traces
