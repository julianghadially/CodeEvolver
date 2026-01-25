"""Modal Sandbox management for CodeEvolver.

Inspired by modal-vibe's SandboxApp pattern, but adapted for
autonomous code mutations and DSPy program execution.

Uses GitHubAppService for private repository authentication.
Uses SandboxGitService for git operations within the sandbox.
"""

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
from ..services.git_sandbox import SandboxGitService, clone_repository


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
    """Get the Modal image for sandbox execution.

    IMPORTANT: The Claude Agent SDK requires the Claude Code CLI (Node.js)
    to be installed. The Python SDK is just a wrapper that spawns the CLI.
    """
    return (
        modal.Image.debian_slim(python_version="3.11")
        .apt_install("git", "curl", "build-essential", "nodejs", "npm")
        .run_commands(
            # Install Claude Code CLI globally
            "npm install -g @anthropic-ai/claude-code",
        )
        .pip_install(
            "gitpython>=3.1.0",
            "dspy>=2.5.0",
            "claude-agent-sdk>=0.1.21",
            "anyio>=4.0.0",
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

    Uses SandboxGitService for all git operations.
    """

    def __init__(
        self,
        sandbox: modal.Sandbox,
        metadata: SandboxMetadata,
    ):
        self.sandbox = sandbox
        self.metadata = metadata
        self._workspace = "/workspace"
        # Git service initialized after clone (needs workspace to exist)
        self._git: SandboxGitService | None = None

    @property
    def id(self) -> str:
        """Get the sandbox ID."""
        return self.metadata.sandbox_id

    @property
    def git(self) -> SandboxGitService:
        """Get the git service for this sandbox."""
        if self._git is None:
            self._git = SandboxGitService(self.sandbox, self._workspace)
        return self._git

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
        branch_name: str | None = None,
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
            branch_name: Optional branch name to checkout/create after cloning

        Returns:
            SandboxApp instance ready for mutation

        Raises:
            ValueError: If private repo authentication fails
        """
        # Handle authentication for private repositories
        print(f"[SANDBOX.create] Starting sandbox creation...")
        authenticated_url = repo_url
        if installation_id:
            print(f"[SANDBOX.create] Getting installation token for {installation_id}...")
            token = GitHubAppService.get_installation_token(installation_id)
            if not token:
                raise ValueError(
                    f"Failed to get installation token for installation {installation_id}"
                )
            authenticated_url = GitHubAppService.get_authenticated_repo_url(
                repo_url, token
            )
            print(f"[SANDBOX.create] Got authenticated URL")

        print(f"[SANDBOX.create] Creating Modal sandbox...")
        sandbox = modal.Sandbox.create(
            app=app,
            image=get_sandbox_image(),
            timeout=timeout,
            cpu=cpu,
            memory=memory,
        )
        print(f"[SANDBOX.create] Modal sandbox created")

        metadata = SandboxMetadata(
            sandbox_id=sandbox.object_id,
            client_id=client_id,
            program_id=program_id,
            status=SandboxStatus.CREATED,
        )

        sandbox_app = SandboxApp(sandbox, metadata)

        # Clone repository (uses authenticated URL if private)
        print(f"[SANDBOX.create] Cloning repository...")
        await sandbox_app._clone_repo(authenticated_url)
        print(f"[SANDBOX.create] Repository cloned")

        # Checkout/create branch if specified
        if branch_name:
            print(f"[SANDBOX.create] Checking out branch: {branch_name}")
            await sandbox_app.checkout_branch(branch_name, create=True)
            print(f"[SANDBOX.create] Branch checked out")

        # Install dependencies
        print(f"[SANDBOX.create] Installing dependencies...")
        await sandbox_app._install_deps()
        print(f"[SANDBOX.create] Dependencies installed")

        # Inject secrets
        if secrets:
            print(f"[SANDBOX.create] Injecting secrets...")
            await sandbox_app._inject_secrets(secrets)

        print(f"[SANDBOX.create] Sandbox setup complete")
        return sandbox_app

    async def _clone_repo(self, repo_url: str) -> None:
        """Clone the repository into the sandbox."""
        self.metadata.status = SandboxStatus.CLONING
        self.metadata.updated_at = datetime.now()

        # Clone using the standalone function (workspace doesn't exist yet)
        result = clone_repository(self.sandbox, repo_url, self._workspace)
        if not result.success:
            self.metadata.status = SandboxStatus.FAILED
            self.metadata.error = f"Git clone failed: {result.stderr}"
            raise RuntimeError(self.metadata.error)

        # Configure git user for commits (required for git commit to work)
        config_result = self.git.configure_user()
        if not config_result.success:
            print(f"[_clone_repo] Warning: Failed to configure git user: {config_result.stderr}")

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

        # Commit changes using git service
        commit_result = self.git.stage_and_commit(
            f"Apply prompt mutation for program {self.metadata.program_id}"
        )
        if not commit_result.success:
            return MutationResult(
                success=False,
                error=f"Git commit failed: {commit_result.stderr}",
                program_json=program_json,
            )

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
        print(f"[apply_code_mutation] Starting code mutation...")
        self.metadata.status = SandboxStatus.MUTATING
        self.metadata.updated_at = datetime.now()

        # Generate and write the agent script
        print(f"[apply_code_mutation] Generating agent script...")
        script = generate_agent_script(
            workspace_path=self._workspace,
            change_request=change_request,
            change_location=change_location,
        )

        print(f"[apply_code_mutation] Writing agent script to sandbox...")
        self.sandbox.exec(
            "bash",
            "-c",
            f"cat > /tmp/agent_script.py << 'EOFAGENT'\n{script}\nEOFAGENT",
        ).wait()

        # Run the agent
        print(f"[apply_code_mutation] Running Claude agent (this may take a few minutes)...")
        p = self.sandbox.exec("python", "/tmp/agent_script.py")
        p.wait()

        stdout = p.stdout.read()
        stderr = p.stderr.read()
        print(f"[apply_code_mutation] Agent finished. Return code: {p.returncode}")

        # Log agent output for debugging
        if stdout:
            print(f"[apply_code_mutation] Agent output:")
            for line in stdout.split("\n"):
                if line.strip():
                    print(f"  {line}")
        if stderr:
            print(f"[apply_code_mutation] Stderr: {stderr[:1000]}")

        result = parse_agent_output(stdout, stderr, p.returncode)

        if not result.success:
            return MutationResult(success=False, error=result.error)

        # Commit changes made by the agent using git service
        print(f"[apply_code_mutation] Committing changes via git service...")
        commit_msg = f"Apply code mutation for program {self.metadata.program_id}: {change_request[:50]}..."
        commit_result = self.git.stage_and_commit(commit_msg)

        if not commit_result.success:
            return MutationResult(
                success=False,
                error=f"Git commit failed: {commit_result.stderr}",
            )

        # Check if there were actually changes (stage_and_commit returns success even if no changes)
        if "no changes" in commit_result.operation:
            print(f"[apply_code_mutation] WARNING: No changes to commit!")
            return MutationResult(success=True, error="No changes were made by the agent")

        # Verify commit was created
        self.git.log_recent(1)

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

    async def checkout_branch(self, branch_name: str, create: bool = True) -> None:
        """
        Checkout or create a branch.

        Args:
            branch_name: Name of the branch to checkout
            create: If True, create the branch if it doesn't exist
        """
        result = self.git.checkout(branch_name, create=create)
        if not result.success:
            raise RuntimeError(f"Failed to checkout branch {branch_name}: {result.stderr}")

    async def push_to_remote(self, branch_name: str) -> None:
        """
        Push the current branch to remote.

        Args:
            branch_name: Name of the branch to push
        """
        # Show what we're about to push
        print(f"[push_to_remote] Preparing to push to origin/{branch_name}...")
        self.git.log_recent(3)
        self.git.status_short()

        # Push
        result = self.git.push(branch_name)
        if not result.success:
            raise RuntimeError(f"Failed to push to remote: {result.stderr}")


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
    skip_program_run: bool = False,
    branch_name: str | None = None,
    push_to_remote: bool = False,
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
        skip_program_run: If True, skip running the DSPy program (code-only mode)
        branch_name: Optional branch name to use (otherwise auto-generated)
        push_to_remote: If True, push changes to remote after mutation

    Returns:
        ExecutionResult with all outputs
    """
    sandbox_app = None

    try:
        print(f"[SANDBOX] Creating sandbox for {repo_url}...")
        print(f"[SANDBOX] Branch: {branch_name}, Installation ID: {installation_id}")

        # Create sandbox and clone repo (handles private repo auth via GitHubAppService)
        sandbox_app = await SandboxApp.create(
            app=app,
            client_id=client_id,
            program_id=program_id,
            repo_url=repo_url,
            installation_id=installation_id,
            secrets=secrets,
            branch_name=branch_name,
        )
        print(f"[SANDBOX] Sandbox created successfully")

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
            print(f"[SANDBOX] Applying code mutation...")
            mutation_result = await sandbox_app.apply_code_mutation(
                change_request, change_location
            )
            print(f"[SANDBOX] Code mutation result: success={mutation_result.success}")
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

        # Push to remote if requested
        actual_branch_name = branch_name or f"program_{program_id}"
        if push_to_remote:
            print(f"[SANDBOX] Pushing to remote branch: {actual_branch_name}...")
            await sandbox_app.push_to_remote(actual_branch_name)
            print(f"[SANDBOX] Push completed")

        # Skip program run if requested (code-only mode)
        if skip_program_run:
            return ExecutionResult(
                status="success",
                program_id=program_id,
                program_json=mutation_result.program_json,
                branch_name=actual_branch_name,
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
            branch_name=actual_branch_name,
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
