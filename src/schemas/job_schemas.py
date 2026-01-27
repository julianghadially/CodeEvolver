"""Optimization job schemas for GEPA integration."""

from datetime import datetime
from enum import Enum
from typing import Any



class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
