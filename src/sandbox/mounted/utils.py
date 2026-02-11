"""Utilities for sandbox scripts.

Provides logging that can be returned to the user in results.
Includes timer utilities for performance profiling.
"""

import os
import subprocess
import sys
import time
import traceback
from typing import Any


# ============================================================================
# Timer utilities for performance profiling
# ============================================================================

# Global state for timer_printer
_last_timer_call: float | None = None


def timer_printer(label: str = "") -> float:
    """Print elapsed time since the last timer_printer call.

    On first call, initializes the timer and prints "[TIMER] Timer started".
    On subsequent calls, prints elapsed time since the last call.

    Args:
        label: Optional label to include in the output (e.g., "Agent started").

    Returns:
        The elapsed time in seconds since the last call (0.0 on first call).

    Example output:
        [TIMER] Timer started
        [TIMER] +2.34s - Agent started
        [TIMER] +15.67s - Tool: EnterPlanMode
    """
    global _last_timer_call

    now = time.time()

    if _last_timer_call is None:
        _last_timer_call = now
        print(f"[TIMER] Timer started{f' - {label}' if label else ''}", flush=True)
        return 0.0

    elapsed = now - _last_timer_call
    _last_timer_call = now

    label_suffix = f" - {label}" if label else ""
    print(f"[TIMER] +{elapsed:.2f}s{label_suffix}", flush=True)
    return elapsed


def timer_reset() -> None:
    """Reset the timer to None (next timer_printer call will start fresh)."""
    global _last_timer_call
    _last_timer_call = None


class TimerContext:
    """Context manager for timing a block of code.

    Usage:
        with TimerContext("evaluate") as t:
            result = adapter.evaluate(batch, candidate)
        # Prints: [TIMER] evaluate took 12.34s

    Attributes:
        elapsed: Time elapsed in seconds (available after exiting context).
    """

    def __init__(self, label: str):
        self.label = label
        self.start_time: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "TimerContext":
        self.start_time = time.time()
        print(f"[TIMER] Starting: {self.label}", flush=True)
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.time() - self.start_time
        print(f"[TIMER] {self.label} took {self.elapsed:.2f}s", flush=True)


# ============================================================================
# Logging utilities
# ============================================================================


class SandboxLogger:
    """Collects log messages during sandbox script execution.

    Logs are collected in memory and can be included in the result dict.
    Also prints to stderr for immediate visibility in Modal logs.
    """

    def __init__(self, name: str = "sandbox"):
        self.name = name
        self.logs: list[str] = []

    def _log(self, level: str, msg: str) -> None:
        """Add a log entry."""
        entry = f"[{self.name}:{level}] {msg}"
        self.logs.append(entry)
        print(entry, file=sys.stderr, flush=True)

    def debug(self, msg: str) -> None:
        self._log("DEBUG", msg)

    def info(self, msg: str) -> None:
        self._log("INFO", msg)

    def warning(self, msg: str) -> None:
        self._log("WARN", msg)

    def error(self, msg: str) -> None:
        self._log("ERROR", msg)

    def exception(self, msg: str) -> None:
        """Log an error with full traceback."""
        tb = traceback.format_exc()
        self._log("ERROR", f"{msg}\n{tb}")

    def get_logs(self) -> list[str]:
        """Return all collected logs."""
        return self.logs.copy()

    def clear(self) -> None:
        """Clear collected logs."""
        self.logs.clear()


# Global logger instance for convenience
_logger = SandboxLogger()


def get_logger(name: str | None = None) -> SandboxLogger:
    """Get a logger instance.

    Args:
        name: Optional name for a new logger. If None, returns global logger.

    Returns:
        SandboxLogger instance.
    """
    if name is None:
        return _logger
    return SandboxLogger(name)


def make_error_result(error: Exception, logs: list[str] | None = None) -> dict[str, Any]:
    """Create a standardized error result dict.

    Args:
        error: The exception that occurred.
        logs: Optional list of log messages.

    Returns:
        Dict with success=False, error message, traceback, and logs.
    """
    return {
        "success": False,
        "error": f"{type(error).__name__}: {error}",
        "traceback": traceback.format_exc(),
        "logs": logs or [],
    }


def verify_changes_with_git(workspace: str) -> tuple[bool, list[str]]:
    """Run git status and return (has_changes, changed_files).

    Args:
        workspace: Path to the git workspace directory.

    Returns:
        Tuple of (has_changes, list_of_changed_files).
    """
    os.chdir(workspace)
    git_status = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
    )
    changed_files = [line for line in git_status.stdout.strip().split("\n") if line]
    return bool(changed_files), changed_files


def make_success_result(data: dict[str, Any], logs: list[str] | None = None) -> dict[str, Any]:
    """Create a standardized success result dict.

    Args:
        data: The result data to include.
        logs: Optional list of log messages.

    Returns:
        Dict with success=True, data fields, and logs.
    """
    result = {"success": True, "logs": logs or []}
    result.update(data)
    return result
