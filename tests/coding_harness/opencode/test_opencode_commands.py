from __future__ import annotations

from pathlib import Path


def test_build_opencode_run_command_includes_model_permissions_and_prompt():
    from src.engine.agent.harnesses import AgentRole
    from src.engine.agent.harnesses.opencode.commands import (
        OpenCodeSessionSpec,
        build_opencode_run_command,
    )

    spec = OpenCodeSessionSpec(
        cwd="/workspace",
        prompt="change code",
        model="openai/gpt-5.3-codex-spark",
        role=AgentRole.CODING,
        skill_dir=Path("/workspace/.opencode/skills"),
        migration_fallback_skill_dir=Path("/workspace/.claude/skills"),
        uses_migration_fallback=False,
        bootstrap_commands=["/s"],
        finalization_commands=["/update-specs"],
        autonomous_permissions=True,
    )

    cmd = build_opencode_run_command(spec)

    assert cmd[:2] == ["opencode", "run"]
    assert cmd[cmd.index("--dir") + 1] == "/workspace"
    assert cmd[cmd.index("--model") + 1] == "openai/gpt-5.3-codex-spark"
    assert "--dangerously-skip-permissions" in cmd
    assert cmd[-1] == "/s\n\nchange code\n\n/update-specs"


def test_build_opencode_run_command_omits_model_when_none():
    from src.engine.agent.harnesses import AgentRole
    from src.engine.agent.harnesses.opencode.commands import (
        OpenCodeSessionSpec,
        build_opencode_run_command,
    )

    spec = OpenCodeSessionSpec(
        cwd="/workspace",
        prompt="inspect",
        model=None,
        role=AgentRole.INSPECTION,
        skill_dir=Path("/workspace/.opencode/skills"),
        migration_fallback_skill_dir=Path("/workspace/.claude/skills"),
        uses_migration_fallback=False,
        bootstrap_commands=[],
        finalization_commands=[],
        autonomous_permissions=False,
    )

    cmd = build_opencode_run_command(spec)

    assert "--model" not in cmd
    assert "--dangerously-skip-permissions" not in cmd
    assert cmd[-1] == "inspect"
