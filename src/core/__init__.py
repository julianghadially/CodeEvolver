"""Core module for CodeEvolver Agents.

Contains the main sandbox execution logic inspired by Modal-vibe's patterns.
"""

from .sandbox import (
    ExecutionResult,
    MutationResult,
    SandboxApp,
    SandboxMetadata,
    SandboxStatus,
    execute_mutation,
    get_sandbox_image,
)
from .client_sandbox import ClientSandbox
from .deploy_agent import (
    AgentResult,
    build_agent_config,
    build_reflection_config,
    parse_agent_output,
    parse_reflection_output,
    ReflectionResult,
    run_code_mutation_agent,
)
from .program_runner import (
    ProgramOutput,
    ProgramRunResult,
    apply_prompt_mutation,
    generate_runner_script,
    load_program_json,
    parse_runner_output,
    save_program_json,
    run_program,
)
from .system_prompt import (
    get_code_mutation_prompt,
    get_prompt_mutation_prompt,
    get_program_execution_prompt,
)

__all__ = [
    # Sandbox
    "SandboxApp",
    "SandboxMetadata",
    "SandboxStatus",
    "ExecutionResult",
    "MutationResult",
    "execute_mutation",
    "get_sandbox_image",
    # Client Sandbox (base class for optimizer sandboxes)
    "ClientSandbox",
    # Agent
    "AgentResult",
    "ReflectionResult",
    "run_code_mutation_agent",
    "build_agent_config",
    "build_reflection_config",
    "parse_agent_output",
    "parse_reflection_output",
    # Program Runner
    "ProgramOutput",
    "ProgramRunResult",
    "run_program",
    "load_program_json",
    "save_program_json",
    "apply_prompt_mutation",
    "generate_runner_script",
    "parse_runner_output",
    # Prompts
    "get_code_mutation_prompt",
    "get_prompt_mutation_prompt",
    "get_program_execution_prompt",
]
