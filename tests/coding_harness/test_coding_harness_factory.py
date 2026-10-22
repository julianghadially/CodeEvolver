from __future__ import annotations


def test_factory_creates_opencode_harness_without_verifying_binary():
    from src.engine.agent.harnesses.factory import create_coding_harness
    from src.engine.agent.harnesses.opencode import OpenCodeHarness

    harness = create_coding_harness(
        coding_harness="opencode",
        coding_harness_model="openai/gpt-5.3-codex-spark",
    )

    assert isinstance(harness, OpenCodeHarness)
    assert harness.model == "openai/gpt-5.3-codex-spark"


def test_factory_creates_claude_harness_with_legacy_runners():
    from src.engine.agent.harnesses.claude import ClaudeCodeHarness
    from src.engine.agent.harnesses.factory import create_coding_harness

    def coding_runner(*args, **kwargs):
        return {"success": True, "output": "coded", "error": None}

    def inspection_runner(*args, **kwargs):
        return {"success": True, "proposed_change": "inspected", "error": None}

    harness = create_coding_harness(
        coding_harness="claude",
        coding_runner=coding_runner,
        inspection_runner=inspection_runner,
    )

    assert isinstance(harness, ClaudeCodeHarness)


def test_package_reexports_factory_function():
    from src.engine.agent.harnesses import create_coding_harness
    from src.engine.agent.harnesses.factory import (
        create_coding_harness as factory_create_coding_harness,
    )

    assert create_coding_harness is factory_create_coding_harness
