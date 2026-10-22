"""Copyright © 2026 440 Labs LLC

Logging configuration and diagnostic helpers for the in-sandbox optimizer.

Stdlib-only by design — this module must be importable from anywhere in
`src.engine` without pulling in optimizer, evaluator, or agent dependencies.
Callers pass primitive values (iteration index, workspace path), not engine
objects.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path


DEFAULT_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"

EXPECTED_ENV_KEYS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GITHUB_TOKEN")


class _LitellmBotocoreFilter(logging.Filter):
    """Drop litellm's import-time botocore-missing warning.

    litellm tries to pre-load AWS Bedrock / SageMaker event-stream shapes at
    import time. When ``botocore`` is not installed it logs a WARNING per
    shape. The warning is harmless for non-AWS providers (it only disables
    Bedrock/SageMaker streaming), and since CodeEvolver targets Anthropic
    via litellm, the message is pure noise.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return True
        return not ("could not pre-load" in msg and "botocore" in msg)


def configure_logging() -> None:
    """Configure the root logger for the optimizer process.

    Level is controlled by the `LOG_LEVEL` env var (default `INFO`). Output
    goes to stdout so the launcher captures it.
    """
    level_name = os.environ.get("LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    logging.basicConfig(
        level=level,
        format=DEFAULT_FORMAT,
        stream=sys.stdout,
        force=True,
    )
    # litellm emits two copies of every warning: its own colored
    # ``LiteLLM:WARNING`` line (via a handler attached directly to the
    # ``LiteLLM`` / ``litellm`` loggers) and the propagated copy that hits
    # our root handler. A logger-level filter drops the record before
    # either handler sees it, so we attach it to both litellm loggers.
    botocore_filter = _LitellmBotocoreFilter()
    for logger_name in ("LiteLLM", "litellm"):
        logging.getLogger(logger_name).addFilter(botocore_filter)


_DETAILED_AGENT_LOGGER_NAME = "codeevolver.detailed_agent"


def get_detailed_agent_logger() -> logging.Logger:
    """Return the per-job detailed-agent logger (verbose, file-only).

    Routes the OpenCode architect's tool calls and reasoning text to a
    dedicated file (``paths.detailed_agent_log_path()``) so the verbose stream
    stays **out** of the main stdout log. ``propagate=False`` guarantees nothing
    reaches the root handler.

    Configured once and idempotent — repeated calls reuse the same handler.
    Best-effort: if the log file cannot be opened, the logger falls back to a
    ``NullHandler`` so callers never crash on observability.
    """
    logger = logging.getLogger(_DETAILED_AGENT_LOGGER_NAME)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    try:
        # Imported lazily so logger.py stays importable even if paths' env
        # assumptions are not yet set up at module import time.
        from src.engine.paths import detailed_agent_log_path

        path = Path(detailed_agent_log_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        handler: logging.Handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(DEFAULT_FORMAT))
    except Exception:  # noqa: BLE001 - observability must never crash a run
        handler = logging.NullHandler()
    logger.addHandler(handler)
    return logger


def log_session_debug_context(
    *,
    logger: logging.Logger,
    iteration_idx: int,
    workspace: str,
) -> None:
    """Emit DEBUG lines useful for diagnosing SDK-subprocess boot failures.

    The Claude Agent SDK spawns the `claude` CLI as a subprocess and
    surfaces its exit code without a copy of the CLI's stderr in the
    exception. When that subprocess fails (exit code 1, "Check stderr
    output for details"), these lines narrow it down: workspace layout,
    CLI presence, skill staging, and which expected env vars are set
    (names only — no values logged).
    """
    if not logger.isEnabledFor(logging.DEBUG):
        return
    logger.debug("iteration %d: cwd=%s exists=%s", iteration_idx, workspace, os.path.isdir(workspace))
    logger.debug("iteration %d: claude CLI on PATH=%s", iteration_idx, shutil.which("claude"))
    staged_skills = Path(workspace) / ".claude" / "skills"
    logger.debug(
        "iteration %d: staged skills dir=%s exists=%s contents=%s",
        iteration_idx,
        staged_skills,
        staged_skills.exists(),
        [p.name for p in staged_skills.iterdir()] if staged_skills.exists() else None,
    )
    present = [k for k in EXPECTED_ENV_KEYS if os.environ.get(k)]
    missing = [k for k in EXPECTED_ENV_KEYS if not os.environ.get(k)]
    logger.debug("iteration %d: env present=%s missing=%s", iteration_idx, present, missing)


__all__ = [
    "configure_logging",
    "get_detailed_agent_logger",
    "log_session_debug_context",
]
