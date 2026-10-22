"""Tests for harness-aware CodeEvolver skill staging."""

from __future__ import annotations


def test_skill_staging_uses_optimizer_skills_as_canonical_source():
    from src.engine.agent.harnesses.skills import CANONICAL_CODEEVOLVER_SKILLS_DIR
    from src.engine.optimizer.v3.architect_helpers import SKILLS_DIR

    assert CANONICAL_CODEEVOLVER_SKILLS_DIR == SKILLS_DIR
    # Engine 3 skill set: close_iteration + exploration replaced the old
    # outcome_reflection skill.
    assert (CANONICAL_CODEEVOLVER_SKILLS_DIR / "close_iteration" / "SKILL.md").exists()
    assert (CANONICAL_CODEEVOLVER_SKILLS_DIR / "exploration" / "SKILL.md").exists()
    assert not (CANONICAL_CODEEVOLVER_SKILLS_DIR / "outcome_reflection").exists()


def test_claude_skill_staging_keeps_existing_project_location(tmp_path):
    from src.engine.agent.harnesses.skills import (
        CANONICAL_CODEEVOLVER_SKILLS_DIR,
        stage_codeevolver_skills,
    )
    from src.engine.paths import history_dir, memory_dir

    staged = stage_codeevolver_skills(tmp_path, harness="claude")

    assert staged == tmp_path / ".claude" / "skills"
    # Placeholder-free skills stage byte-identical to the canonical source.
    assert (staged / "tracing" / "SKILL.md").read_text() == (
        CANONICAL_CODEEVOLVER_SKILLS_DIR / "tracing" / "SKILL.md"
    ).read_text()
    # Skills with path placeholders are rendered at staging time.
    close_text = (staged / "close_iteration" / "SKILL.md").read_text()
    assert "{memory_dir}" not in close_text
    assert "{history_dir}" not in close_text
    assert memory_dir() in close_text
    assert history_dir() in close_text
    # The {workspace_path} placeholder renders to the staged workspace.
    exploration_text = (staged / "exploration" / "SKILL.md").read_text()
    assert "{workspace_path}" not in exploration_text
    assert str(tmp_path) in exploration_text


def test_opencode_skill_staging_uses_opencode_primary_location(tmp_path):
    from src.engine.agent.harnesses.skills import (
        CANONICAL_CODEEVOLVER_SKILLS_DIR,
        stage_codeevolver_skills,
    )

    staged = stage_codeevolver_skills(tmp_path, harness="opencode")

    assert staged == tmp_path / ".opencode" / "skills"
    assert not (tmp_path / ".claude" / "skills").exists()
    assert (staged / "tracing" / "SKILL.md").read_text() == (
        CANONICAL_CODEEVOLVER_SKILLS_DIR / "tracing" / "SKILL.md"
    ).read_text()
    close_text = (staged / "close_iteration" / "SKILL.md").read_text()
    assert "{memory_dir}" not in close_text


def test_skill_staging_respects_skills_selection(tmp_path):
    from src.engine.agent.harnesses.skills import stage_codeevolver_skills

    staged = stage_codeevolver_skills(
        tmp_path, harness="opencode", skills=("exploration",)
    )

    staged_names = sorted(p.name for p in staged.iterdir() if p.is_dir())
    assert staged_names == ["exploration"]
    assert (staged / "exploration" / "SKILL.md").exists()


def test_opencode_session_spec_stages_skills_without_claude_dependency(tmp_path):
    from src.engine.agent.harnesses.skills import CANONICAL_CODEEVOLVER_SKILLS_DIR
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    harness = OpenCodeHarness(model="openai/gpt-5.3-codex-spark")

    spec = harness.build_session_spec(
        HarnessRequest(
            role=AgentRole.CODING,
            workspace=tmp_path,
            prompt="make a change",
            raw_provider_metadata={"commit_changes": True},
        )
    )

    assert spec.cwd == str(tmp_path)
    assert spec.model == "openai/gpt-5.3-codex-spark"
    assert spec.skill_dir.name == "skills"
    assert spec.skill_dir.parent != tmp_path / ".opencode"
    assert (spec.skill_dir / "tracing" / "SKILL.md").read_text() == (
        CANONICAL_CODEEVOLVER_SKILLS_DIR / "tracing" / "SKILL.md"
    ).read_text()
    # Placeholders are rendered in the staged copy (workspace comes from the
    # harness request).
    close_text = (spec.skill_dir / "close_iteration" / "SKILL.md").read_text()
    assert "{memory_dir}" not in close_text
    exploration_text = (spec.skill_dir / "exploration" / "SKILL.md").read_text()
    assert "{workspace_path}" not in exploration_text
    assert str(tmp_path) in exploration_text
    assert spec.migration_fallback_skill_dir == tmp_path / ".claude" / "skills"
    assert spec.uses_migration_fallback is False
    assert spec.bootstrap_commands == ["/s"]
    assert spec.finalization_commands == ["/update-specs"]
    assert not (tmp_path / ".opencode").exists()


def test_opencode_finalization_command_only_applies_to_code_changes(tmp_path):
    from src.engine.agent.harnesses import AgentRole, HarnessRequest
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    harness = OpenCodeHarness()

    spec = harness.build_session_spec(
        HarnessRequest(
            role=AgentRole.INSPECTION,
            workspace=tmp_path,
            prompt="inspect traces",
            raw_provider_metadata={"commit_changes": False},
        )
    )

    assert spec.bootstrap_commands == ["/s"]
    assert spec.finalization_commands == []


def test_opencode_coding_service_stages_skills_without_claude_subprocess(
    monkeypatch, tmp_path
):
    import subprocess

    import src.engine.agent.harnesses.opencode.harness as opencode_module
    from src.engine.services import subprocess_service as module

    monkeypatch.setattr(opencode_module.shutil, "which", lambda name: "/usr/bin/opencode")
    monkeypatch.setattr(
        opencode_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="partial output",
            stderr="provider failed",
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_claude_coding_agent_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCode mode must not launch the Claude coding subprocess")
        ),
    )

    result = module.run_coding_agent(
        str(tmp_path),
        change_request="make a change",
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result["success"] is False
    assert "OpenCode exited with code 1" in result["error"]
    assert not (tmp_path / ".opencode").exists()
    assert not (tmp_path / ".claude" / "skills").exists()


def test_opencode_inspection_service_stages_skills_without_claude_subprocess(
    monkeypatch, tmp_path
):
    import subprocess

    import src.engine.agent.harnesses.opencode.harness as opencode_module
    from src.engine.services import subprocess_service as module

    monkeypatch.setattr(opencode_module.shutil, "which", lambda name: "/usr/bin/opencode")
    monkeypatch.setattr(
        opencode_module.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args[0],
            returncode=1,
            stdout="partial output",
            stderr="provider failed",
        ),
    )
    monkeypatch.setattr(
        module,
        "_run_claude_inspection_agent_subprocess",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenCode mode must not launch the Claude inspection subprocess")
        ),
    )

    result = module.run_inspection_agent(
        str(tmp_path),
        prompt="inspect traces",
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert result["success"] is False
    assert "OpenCode exited with code 1" in result["error"]
    assert not (tmp_path / ".opencode").exists()
    assert not (tmp_path / ".claude" / "skills").exists()
