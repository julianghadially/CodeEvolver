"""Optimization job schemas for GEPA integration."""

from datetime import datetime
from enum import Enum
from typing import Any
from dataclasses import dataclass


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class OptimizationResult:
    """Container for optimization results used across runs and tests."""
    final_status: dict[str, Any]
    score_history: list[float]
    job_id: str
    elapsed_seconds: int