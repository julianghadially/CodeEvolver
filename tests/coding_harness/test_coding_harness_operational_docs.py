"""Provider-independent checks for coding harness operational docs."""

from __future__ import annotations

from pathlib import Path


def test_local_deployment_docs_explain_harness_smoke_commands() -> None:
    text = Path("specs/experiment-docs/local-deployment.md").read_text()

    required_snippets = [
        "coding_harness=opencode",
        "coding_harness=claude",
        "coding_harness_model",
        "ignored when `coding_harness=claude`",
        "gpt-5.3-codex-spark",
        "opencode --version",
        "CODEEVOLVER_LIVE_OPENCODE_SMOKE=1",
        "CODEEVOLVER_LIVE_CLAUDE_SMOKE=1",
        "OpenCode is an environment prerequisite",
        "never installs it during a run",
        "A missing `coding_harness` resolves to `opencode`",
        "Set `coding_harness=claude` explicitly",
        "missing `opencode` binary",
        "provider auth/API key setup",
        "permissions/autonomous mode",
        "session/resume behavior",
        "MCP/tool registration issues",
    ]

    missing = [snippet for snippet in required_snippets if snippet not in text]

    assert missing == []
