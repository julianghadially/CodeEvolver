"""Optimization job schemas for CodeEvolver integration."""

from datetime import datetime
from typing import Any
from dataclasses import dataclass

from .enums import JobStatus  # noqa: F401 — re-exported for backwards compatibility

@dataclass
class OptimizationResult:
    """Container for optimization results used across runs and tests."""
    final_status: dict[str, Any]
    score_history: list[float]
    job_id: str
    elapsed_seconds: int