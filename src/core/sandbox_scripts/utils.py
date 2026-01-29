"""Utilities for sandbox scripts.

Provides logging that can be returned to the user in results.
"""

import sys
import traceback
from typing import Any


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
