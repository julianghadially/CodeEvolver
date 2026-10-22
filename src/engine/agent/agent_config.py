"""Agent config builder for the sandbox coding agent subprocess."""

from typing import Any

from .system_prompt import get_code_mutation_prompt


def build_agent_config(
    workspace_path: str,
    change_request: str,
    change_location: str | None = None,
    max_turns: int = 50,
    program_path: str | None = None,
    model: str | None = None,
    deny_paths: list[str] | None = None,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = None,
) -> dict[str, Any]:
    """Build a config dict for the sandbox agent script.

    The config is written as JSON to the sandbox and passed to
    agent/mounted/coding_agent.py via --config.

    When program_path is provided, the change_request is wrapped with the
    full system prompt from get_code_mutation_prompt() (including constraints,
    instructions about codeevolver.md, etc.).

    `model` is the bare Anthropic model id (e.g. "claude-opus-4-7") the SDK
    session runs on; omitted from the config when None so the agent falls back
    to the Claude Agent SDK default.

    `deny_paths` is a list of path fragments the coding agent must never read,
    open, or edit (the valset and the held-out test set). Enforced by a
    PreToolUse fence inside `coding_agent.py`. Omitted from the config when
    empty.
    """
    if program_path:
        full_prompt = get_code_mutation_prompt(
            change_request=change_request,
            change_location=change_location,
            workspace_path=workspace_path,
            program_path=program_path,
        )
        config: dict[str, Any] = {
            "workspace_path": workspace_path,
            "change_request": full_prompt,
            "max_turns": max_turns,
            "coding_harness": coding_harness,
            "coding_harness_model": coding_harness_model,
        }
    else:
        config = {
            "workspace_path": workspace_path,
            "change_request": change_request,
            "max_turns": max_turns,
            "coding_harness": coding_harness,
            "coding_harness_model": coding_harness_model,
        }
        if change_location:
            config["change_location"] = change_location
    if model:
        config["model"] = model
    if deny_paths:
        config["deny_paths"] = list(deny_paths)
    return config
