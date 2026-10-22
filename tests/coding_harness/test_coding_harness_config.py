"""Focused tests for run-level coding harness config."""

from __future__ import annotations

import pytest


def test_missing_coding_harness_defaults_to_opencode(monkeypatch):
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")

    config = normalize_coding_harness_config()

    assert config.coding_harness == "opencode"
    assert config.coding_harness_model is None


def test_accepts_supported_coding_harness_values(monkeypatch):
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")

    opencode_config = normalize_coding_harness_config(coding_harness="opencode")
    claude_config = normalize_coding_harness_config(coding_harness="claude")

    assert opencode_config.coding_harness == "opencode"
    assert claude_config.coding_harness == "claude"


def test_rejects_unknown_coding_harness():
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    with pytest.raises(ValueError, match="Unknown coding_harness 'cursor'"):
        normalize_coding_harness_config(coding_harness="cursor")
    with pytest.raises(ValueError, match="Unknown coding_harness ''"):
        normalize_coding_harness_config(coding_harness="")


def test_passes_through_coding_harness_model_for_opencode(monkeypatch):
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/opencode")

    config = normalize_coding_harness_config(
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert config.coding_harness_model == "openai/gpt-5.3-codex-spark"


def test_ignores_coding_harness_model_for_claude():
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    config = normalize_coding_harness_config(
        coding_harness="claude",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert config.coding_harness == "claude"
    assert config.coding_harness_model is None


def test_opencode_mode_fails_clearly_when_binary_is_missing(monkeypatch):
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )

    monkeypatch.setattr("shutil.which", lambda name: None)

    with pytest.raises(RuntimeError, match="opencode binary was not found"):
        normalize_coding_harness_config(coding_harness="opencode")


def test_agent_model_resolution_allows_openai_reflection_lm_for_opencode():
    from src.engine.optimizer.v3.lm_config import resolve_agent_model_for_harness

    assert (
        resolve_agent_model_for_harness(
            reflection_lm="openai/gpt-4.1-mini",
            coding_harness="opencode",
        )
        is None
    )


def test_agent_model_resolution_requires_anthropic_reflection_lm_for_claude():
    from src.engine.optimizer.v3.lm_config import resolve_agent_model_for_harness

    assert (
        resolve_agent_model_for_harness(
            reflection_lm="anthropic/claude-opus-4-7",
            coding_harness="claude",
        )
        == "claude-opus-4-7"
    )
    with pytest.raises(ValueError, match="reflection_lm must be an Anthropic"):
        resolve_agent_model_for_harness(
            reflection_lm="openai/gpt-4.1-mini",
            coding_harness="claude",
        )


def test_optimizer_run_config_checks_missing_opencode_binary(monkeypatch, tmp_path):
    from src.engine.optimizer.v3.optimization_config import OptimizationConfig
    from src.engine.optimizer.v3.optimizer_loop import run_optimization
    from src.engine.optimizer.v3.strategy import OptimizerStrategy

    monkeypatch.setattr("shutil.which", lambda name: None)
    strategy = OptimizerStrategy(
        method="test",
        prompt_builder=lambda **kwargs: "",
    )

    config = OptimizationConfig(
        workspace=str(tmp_path),
        program="pkg.program:Program",
        metric="pkg.metric:metric",
        coding_harness="opencode",
    )
    with pytest.raises(RuntimeError, match="opencode binary was not found"):
        run_optimization(config=config, strategy=strategy)


def test_local_config_generation_defaults_to_opencode():
    from src.launchers.local_launcher import LocalLauncher

    config = LocalLauncher()._build_optimization_config(
        workspace="/repo",
        kwargs={
            "job_id": "job-1",
            "repo_url": "https://github.com/acme/repo",
            "program": "pkg.program:Program",
            "metric": "pkg.metric:metric",
            "reflection_lm": "anthropic/claude-opus-4-7",
            "coding_harness_model": "openai/gpt-5.3-codex-spark",
        },
    )

    assert config["coding_harness"] == "opencode"
    assert config["coding_harness_model"] == "openai/gpt-5.3-codex-spark"
    assert "program_lm" not in config
    assert config["reflection_lm"] == "anthropic/claude-opus-4-7"
