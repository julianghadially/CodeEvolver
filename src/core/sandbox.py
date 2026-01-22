"""Modal Sandbox management for CodeEvolver.

Inspired by modal-vibe's SandboxApp pattern, but adapted for
autonomous code mutations and DSPy program execution.

Uses GitHubAppService for private repository authentication.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

import modal

from .agent import generate_agent_script, parse_agent_output
from .program_runner import (
    ProgramRunResult,
    apply_prompt_mutation,
    generate_runner_script,
    load_program_json,
    parse_runner_output,
    save_program_json,
)
from ..services.github_app import GitHubAppService


class SandboxStatus(str, Enum):
    """Status of a sandbox execution."""

    CREATED = "created"
    CLONING = "cloning"
    INSTALLING = "installing"
    MUTATING = "mutating"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TERMINATED = "terminated"


@dataclass
class SandboxMetadata:
    """Metadata for a sandbox execution."""

    sandbox_id: str
    client_id: str
    program_id: str
    status: SandboxStatus
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    error: str | None = None


@dataclass
class MutationResult:
    """Result from applying a mutation."""

    success: bool
    program_json: dict[str, Any] | None = None
    error: str | None = None


@dataclass
class ExecutionResult:
    """Complete result from sandbox execution."""

    status: str
    program_id: str
    program_json: dict[str, Any] | None = None
    pipeline_outputs: list[dict[str, Any]] | None = None
    traces: list[Any] | None = None
    branch_name: str | None = None
    error: str | None = None


# Base image for sandbox execution
def get_sandbox_image() -> modal.Image:
    """Get the Modal image for sandbox execution."""
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "build-essential")
        .pip_install(
            "gitpython>=3.1.0",
            "dspy>=2.5.0",
            "claude-agent-sdk>=0.1.21",
        )
    )


class SandboxApp:
    """
    Manages a Modal sandbox for executing code mutations.

    Each SandboxApp instance handles:
    1. Cloning the repository
    2. Installing dependencies
    3. Applying mutations (prompt or code)
    4. Running the DSPy program
    5. Returning results
    """

    def __init__(
        self,
        sandbox: modal.Sandbox,
        metadata: SandboxMetadata,
    ):
        self.sandbox = sandbox
        self.metadata = metadata
        self._workspace = "/workspace"

    @property
    def id(self) -> str:
        """Get the sandbox ID."""
        return self.metadata.sandbox_id

    @staticmethod
    async def create(
        app: modal.App,
        client_id: str,
        program_id: str,
        repo_url: str,
        installation_id: int | None = None,
        secrets: dict[str, str] | None = None,
        timeout: int = 600,
        cpu: int = 2,
        memory: int = 4096,
    ) -> "SandboxApp":
        """
        Create a new sandbox and clone the repository.

        Uses GitHubAppService for private repository authentication when
        installation_id is provided.

        Args:
            app: Modal App instance
            client_id: Client identifier
            program_id: Program identifier for this mutation
            repo_url: Git repository URL
            installation_id: Optional GitHub App installation ID for private repos
            secrets: Optional secrets to inject as env vars
            timeout: Sandbox timeout in seconds
            cpu: CPU allocation
            memory: Memory allocation in MB

        Returns:
            SandboxApp instance ready for mutation

        Raises:
            ValueError: If private repo authentication fails
        """
        # Handle authentication for private repositories
        authenticated_url = repo_url
        if installation_id:
            token = GitHubAppService.get_installation_token(installation_id)
            if not token:
                raise ValueError(
                    f"Failed to get installation token for installation {installation_id}"
                )
            authenticated_url = GitHubAppService.get_authenticated_repo_url(
                repo_url, token
            )

        sandbox = modal.Sandbox.create(
            app=app,
            image=get_sandbox_image(),
            timeout=timeout,
            cpu=cpu,
            memory=memory,
        )

        metadata = SandboxMetadata(
            sandbox_id=sandbox.object_id,
            client_id=client_id,
            program_id=program_id,
            status=SandboxStatus.CREATED,
        )

        sandbox_app = SandboxApp(sandbox, metadata)

        # Clone repository (uses authenticated URL if private)
        await sandbox_app._clone_repo(authenticated_url)

        # Install dependencies
        await sandbox_app._install_deps()

        # Inject secrets
        if secrets:
            await sandbox_app._inject_secrets(secrets)

        return sandbox_app

    async def _clone_repo(self, repo_url: str) -> None:
        """Clone the repository into the sandbox."""
        self.metadata.status = SandboxStatus.CLONING
        self.metadata.updated_at = datetime.now()

        p = self.sandbox.exec("git", "clone", repo_url, self._workspace)
        p.wait()

        if p.returncode != 0:
            self.metadata.status = SandboxStatus.FAILED
            self.metadata.error = f"Git clone failed: {p.stderr.read()}"
            raise RuntimeError(self.metadata.error)

    async def _install_deps(self) -> None:
        """Install dependencies if requirements.txt exists."""
        self.metadata.status = SandboxStatus.INSTALLING
        self.metadata.updated_at = datetime.now()

        p = self.sandbox.exec(
            "bash",
            "-c",
            f"if [ -f {self._workspace}/requirements.txt ]; then "
            f"pip install -r {self._workspace}/requirements.txt; fi",
        )
        p.wait()

    async def _inject_secrets(self, secrets: dict[str, str]) -> None:
        """Inject secrets as environment file."""
        env_content = "\n".join(f"export {k}='{v}'" for k, v in secrets.items())
        self.sandbox.exec(
            "bash",
            "-c",
            f"cat > {self._workspace}/.env << 'EOFENV'\n{env_content}\nEOFENV",
        ).wait()

    async def apply_prompt_mutation(
        self,
        program_json_path: str,
        candidate: dict[str, str],
    ) -> MutationResult:
        """
        Apply a prompt mutation to program.json.

        Args:
            program_json_path: Relative path to program.json
            candidate: Dict mapping component_name -> new instruction

        Returns:
            MutationResult with updated program_json
        """
        self.metadata.status = SandboxStatus.MUTATING
        self.metadata.updated_at = datetime.now()

        full_path = f"{self._workspace}/{program_json_path}"

        # Read current program.json
        p = self.sandbox.exec("cat", full_path)
        p.wait()

        if p.returncode != 0:
            return MutationResult(
                success=False,
                error=f"program.json not found: {full_path}",
            )

        import json

        try:
            program_json = json.loads(p.stdout.read())
        except json.JSONDecodeError as e:
            return MutationResult(
                success=False,
                error=f"Invalid JSON in program.json: {e}",
            )

        # Apply mutations
        try:
            program_json = apply_prompt_mutation(program_json, candidate)
        except KeyError as e:
            return MutationResult(success=False, error=str(e))

        # Write modified program.json
        modified_json = json.dumps(program_json, indent=2)
        self.sandbox.exec(
            "bash",
            "-c",
            f"cat > {full_path} << 'EOFPROGRAMJSON'\n{modified_json}\nEOFPROGRAMJSON",
        ).wait()

        # Commit changes
        self.sandbox.exec("git", "-C", self._workspace, "add", "-A").wait()
        self.sandbox.exec(
            "git",
            "-C",
            self._workspace,
            "commit",
            "-m",
            f"Apply prompt mutation for program {self.metadata.program_id}",
        ).wait()

        return MutationResult(success=True, program_json=program_json)

    async def apply_code_mutation(
        self,
        change_request: str,
        change_location: str | None = None,
    ) -> MutationResult:
        """
        Apply a code mutation using the Claude agent.

        Args:
            change_request: Natural language description of code change
            change_location: Optional module path hint

        Returns:
            MutationResult with status
        """
        self.metadata.status = SandboxStatus.MUTATING
        self.metadata.updated_at = datetime.now()

        # Generate and write the agent script
        script = generate_agent_script(
            workspace_path=self._workspace,
            change_request=change_request,
            change_location=change_location,
        )

        self.sandbox.exec(
            "bash",
            "-c",
            f"cat > /tmp/agent_script.py << 'EOFAGENT'\n{script}\nEOFAGENT",
        ).wait()

        # Run the agent
        p = self.sandbox.exec("python", "/tmp/agent_script.py")
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()
        result = parse_agent_output(stdout, stderr, p.returncode)

        if not result.success:
            return MutationResult(success=False, error=result.error)

        # Commit changes made by the agent
        self.sandbox.exec("git", "-C", self._workspace, "add", "-A").wait()
        self.sandbox.exec(
            "git",
            "-C",
            self._workspace,
            "commit",
            "-m",
            f"Apply code mutation for program {self.metadata.program_id}: {change_request[:50]}...",
        ).wait()

        return MutationResult(success=True)

    async def run_program(
        self,
        program_json_path: str,
        entry_point: str,
        test_examples: list[dict[str, Any]],
        capture_traces: bool = False,
    ) -> ProgramRunResult:
        """
        Run the DSPy program on test examples.

        Args:
            program_json_path: Relative path to program.json
            entry_point: DSPy module class (e.g., 'fire.FIREJudge')
            test_examples: Examples to run
            capture_traces: Whether to capture execution traces

        Returns:
            ProgramRunResult with outputs
        """
        self.metadata.status = SandboxStatus.RUNNING
        self.metadata.updated_at = datetime.now()

        # Generate and write the runner script
        script = generate_runner_script(
            workspace_path=self._workspace,
            program_json_path=program_json_path,
            entry_point=entry_point,
            test_examples=test_examples,
            capture_traces=capture_traces,
        )

        self.sandbox.exec(
            "bash",
            "-c",
            f"cat > /tmp/runner_script.py << 'EOFRUNNER'\n{script}\nEOFRUNNER",
        ).wait()

        # Run the program
        p = self.sandbox.exec("python", "/tmp/runner_script.py")
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()
        result = parse_runner_output(stdout, stderr, p.returncode)

        if result.success:
            self.metadata.status = SandboxStatus.COMPLETED
        else:
            self.metadata.status = SandboxStatus.FAILED
            self.metadata.error = result.error

        self.metadata.updated_at = datetime.now()

        return result

    def terminate(self) -> None:
        """Terminate the sandbox."""
        try:
            self.sandbox.terminate()
            self.metadata.status = SandboxStatus.TERMINATED
            self.metadata.updated_at = datetime.now()
        except Exception as e:
            self.metadata.error = f"Failed to terminate: {e}"


async def execute_mutation(
    app: modal.App,
    client_id: str,
    program_id: str,
    repo_url: str,
    mutation_type: str,
    program_json_path: str,
    entry_point: str,
    candidate: dict[str, str] | None = None,
    change_request: str | None = None,
    change_location: str | None = None,
    test_examples: list[dict[str, Any]] | None = None,
    capture_traces: bool = False,
    secrets: dict[str, str] | None = None,
    installation_id: int | None = None,
) -> ExecutionResult:
    """
    Execute a complete mutation workflow in a sandbox.

    This is the main entry point for mutation execution. Uses GitHubAppService
    for private repository authentication when installation_id is provided.

    Args:
        app: Modal App instance
        client_id: Client identifier
        program_id: Program identifier
        repo_url: Git repository URL
        mutation_type: "prompt" or "code"
        program_json_path: Path to program.json
        entry_point: DSPy module class
        candidate: For prompt mutations
        change_request: For code mutations
        change_location: Optional hint for code mutations
        test_examples: Examples to run
        capture_traces: Whether to capture traces
        secrets: Secrets to inject
        installation_id: Optional GitHub App installation ID for private repos

    Returns:
        ExecutionResult with all outputs
    """
    sandbox_app = None

    try:
        # Create sandbox and clone repo (handles private repo auth via GitHubAppService)
        sandbox_app = await SandboxApp.create(
            app=app,
            client_id=client_id,
            program_id=program_id,
            repo_url=repo_url,
            installation_id=installation_id,
            secrets=secrets,
        )

        # Apply mutation
        if mutation_type == "prompt":
            if not candidate:
                return ExecutionResult(
                    status="failed",
                    program_id=program_id,
                    error="candidate required for prompt mutation",
                )
            mutation_result = await sandbox_app.apply_prompt_mutation(
                program_json_path, candidate
            )
        elif mutation_type == "code":
            if not change_request:
                return ExecutionResult(
                    status="failed",
                    program_id=program_id,
                    error="change_request required for code mutation",
                )
            mutation_result = await sandbox_app.apply_code_mutation(
                change_request, change_location
            )
        else:
            return ExecutionResult(
                status="failed",
                program_id=program_id,
                error=f"Unknown mutation type: {mutation_type}",
            )

        if not mutation_result.success:
            return ExecutionResult(
                status="failed",
                program_id=program_id,
                error=mutation_result.error,
            )

        # Run program on test examples
        run_result = await sandbox_app.run_program(
            program_json_path=program_json_path,
            entry_point=entry_point,
            test_examples=test_examples or [],
            capture_traces=capture_traces,
        )

        return ExecutionResult(
            status="success" if run_result.success else "failed",
            program_id=program_id,
            program_json=mutation_result.program_json,
            pipeline_outputs=[
                {"example_id": o.example_id, "output": o.output, "error": o.error}
                for o in run_result.outputs
            ],
            traces=run_result.traces,
            error=run_result.error,
        )

    except Exception as e:
        return ExecutionResult(
            status="failed",
            program_id=program_id,
            error=str(e),
        )

    finally:
        if sandbox_app:
            sandbox_app.terminate()
