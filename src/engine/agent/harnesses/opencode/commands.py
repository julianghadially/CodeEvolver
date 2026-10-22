"""OpenCode session specs and command construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..base import AgentRole


@dataclass(frozen=True)
class OpenCodeSessionSpec:
    cwd: str
    prompt: str
    model: str | None
    role: AgentRole
    skill_dir: Path
    migration_fallback_skill_dir: Path
    uses_migration_fallback: bool
    bootstrap_commands: list[str]
    finalization_commands: list[str]
    autonomous_permissions: bool
    # Per-session OpenCode data/state dir (XDG_DATA_HOME / XDG_STATE_HOME;
    # the cache stays shared). Set for inspection sessions only; None means the
    # session uses OpenCode's global store under $HOME. See
    # `harness._prepare_opencode_data_dir`.
    data_dir: Path | None = None


def build_opencode_prompt(spec: OpenCodeSessionSpec) -> str:
    parts = [
        *spec.bootstrap_commands,
        spec.prompt,
        *spec.finalization_commands,
    ]
    return "\n\n".join(part for part in parts if part)


def build_opencode_run_command(spec: OpenCodeSessionSpec) -> list[str]:
    prompt = build_opencode_prompt(spec)
    cmd = [
        "opencode",
        "run",
        "--dir",
        spec.cwd,
        "--format",
        "json",
    ]
    if spec.model:
        cmd.extend(["--model", spec.model])
    if spec.autonomous_permissions:
        cmd.append("--dangerously-skip-permissions")
    cmd.append(prompt)
    return cmd


def build_opencode_architect_command(
    spec: OpenCodeSessionSpec,
    *,
    resume_session_id: str | None = None,
) -> list[str]:
    cmd = build_opencode_run_command(spec)
    if resume_session_id:
        cmd[-1:-1] = ["--session", resume_session_id]
    return cmd
