"""Sandbox scripts that run inside client Modal sandboxes.

These scripts are copied into the sandbox image and executed
via sandbox.exec(). They have access to the client's full
Python environment (dspy, their code, etc.).
"""
from .debug_env import (
    get_debug_python_command,
    get_debug_env_info,
    get_dspy_import_diagnostic,
    _log_environment_debug,
)
from .utils import (
    timer_printer,
    timer_reset,
    TimerContext,
    SandboxLogger,
    get_logger,
    make_error_result,
    make_success_result,
    verify_changes_with_git,
)
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
    # debug_env
    "get_debug_python_command",
    "get_debug_env_info",
    "get_dspy_import_diagnostic",
    "_log_environment_debug",
    # utils - timer
    "timer_printer",
    "timer_reset",
    "TimerContext",
    # utils - logging
    "SandboxLogger",
    "get_logger",
    "make_error_result",
    "make_success_result",
    "verify_changes_with_git",
    # environment_setup
    "load_env_file",
    "check_api_key",
    "setup_sandbox_env",
    "verify_claude_cli",
    "create_keep_stream_hook",
    "create_prompt_stream",
    # user_proxy
    "create_user_proxy",
]
