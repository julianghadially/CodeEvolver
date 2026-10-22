"""Agent scripts that execute as subprocesses of the optimizer.

These scripts run the coding and reflection agents for code mutations and
analysis.
"""

from .environment_setup import (
    load_env_file,
    check_api_key,
    setup_sandbox_env,
    verify_claude_cli,
    create_keep_stream_hook,
    create_prompt_stream,
)
from .user_proxy import create_user_proxy

__all__ = [
    "load_env_file",
    "check_api_key",
    "setup_sandbox_env",
    "verify_claude_cli",
    "create_keep_stream_hook",
    "create_prompt_stream",
    "create_user_proxy",
]
