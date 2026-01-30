"""Pydantic models for request/response and database records."""


from .program_schemas import (
    ProgramStatus,
    MutationType,
    PipelineOutput,
)
from .requests import (
    ExecuteStepRequest,
    ConnectGitRequest,
    ChangeRequest,
    ExecuteSandboxRequest,  # Deprecated alias for backwards compatibility
    OptimizeRequest,
    JobStatusUpdateRequest,
    JobProgressUpdateRequest,
)
from .responses import (
    ExecuteStepResponse,
    ConnectGitResponse,
    GetProgramResponse,
    ChangeResponse,
    ExecuteSandboxResponse,  # Deprecated alias for backwards compatibility
    OptimizeResponse,
    JobStatusResponse,
    CancelCheckResponse,
)
from .db_schemas import (
    ClientRecord,
    ProgramRecord,
    JobRecord,
)
from .job_schemas import (
    JobStatus,
)

__all__ = [
    "ConnectGitRequest",
    "ConnectGitResponse",
    "ClientRecord",
    "MutationType",
    "ProgramStatus",
    "ExecuteStepRequest",
    "ExecuteStepResponse",
    "ChangeRequest",
    "ChangeResponse",
    "ExecuteSandboxRequest",  # Deprecated alias
    "ExecuteSandboxResponse",  # Deprecated alias
    "PipelineOutput",
    "ProgramRecord",
    "GetProgramResponse",
    "JobStatus",
    "OptimizeRequest",
    "OptimizeResponse",
    "JobStatusResponse",
    "JobStatusUpdateRequest",
    "JobProgressUpdateRequest",
    "CancelCheckResponse",
    "JobRecord",
]
