from sandbox.mounted.utils import timer_printer
# ============================================================================
# Agent environment setup utilities
# ============================================================================

def load_env_file(workspace_path: str, prefix: str = "AGENT") -> bool:
    """Load environment variables from .env file.

    Args:
        workspace_path: Path to the workspace containing .env
        prefix: Log prefix (e.g., "AGENT" or "REFLECT")

    Returns:
        True if .env was found and loaded, False otherwise.
    """
    import os

    env_file = f"{workspace_path}/.env"
    timer_printer("Loading .env")
    print(f"[{prefix}] Looking for .env at: {env_file}")

    if os.path.exists(env_file):
        print(f"[{prefix}] Found .env file, loading...")
        with open(env_file) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, _, value = line.partition("=")
                    key = key.replace("export ", "").strip()
                    value = value.strip().strip("'").strip('"')
                    os.environ[key] = value
                    if key == "ANTHROPIC_API_KEY":
                        print(f"[{prefix}] Loaded ANTHROPIC_API_KEY ({len(value)} chars)")
        return True
    else:
        print(f"[{prefix}] WARNING: No .env file found!")
        return False


def check_api_key(prefix: str = "AGENT") -> bool:
    """Check that ANTHROPIC_API_KEY is set.

    Args:
        prefix: Log prefix (e.g., "AGENT" or "REFLECT")

    Returns:
        True if API key is set, False otherwise (also exits with code 1).
    """
    import os
    import sys

    timer_printer("Checking API key")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"{prefix}_ERROR: ANTHROPIC_API_KEY not set in environment")
        sys.exit(1)
    else:
        print(f"[{prefix}] ANTHROPIC_API_KEY is set ({len(os.environ['ANTHROPIC_API_KEY'])} chars)")
        return True


def setup_sandbox_env(workspace_path: str, prefix: str = "AGENT", add_venv_to_path: bool = False) -> None:
    """Complete sandbox environment setup.

    Loads .env, checks API key, sets IS_SANDBOX, optionally adds venv to PATH.

    Args:
        workspace_path: Path to the workspace
        prefix: Log prefix (e.g., "AGENT" or "REFLECT")
        add_venv_to_path: If True, add workspace/.venv/bin to PATH
    """
    import os

    load_env_file(workspace_path, prefix)
    check_api_key(prefix)

    # Enable bypassPermissions mode for root user in sandbox environments
    os.environ["IS_SANDBOX"] = "1"
    print(f"[{prefix}] Set IS_SANDBOX=1 for bypassPermissions mode")

    # Add venv to PATH if requested (for client code execution via agent's Bash tool)
    if add_venv_to_path:
        venv_bin = f"{workspace_path}/.venv/bin"
        if os.path.exists(venv_bin):
            os.environ["PATH"] = venv_bin + ":" + os.environ.get("PATH", "")
            print(f"[{prefix}] Added venv to PATH: {venv_bin}")


def verify_claude_cli(prefix: str = "AGENT") -> bool:
    """Verify Claude Code CLI is installed.

    Args:
        prefix: Log prefix (e.g., "AGENT" or "REFLECT")

    Returns:
        True if CLI is found, exits with code 1 otherwise.
    """
    import subprocess
    import sys

    timer_printer("Checking Claude CLI")
    try:
        result = subprocess.run(["claude", "--version"], capture_output=True, text=True)
        print(f"[{prefix}] Claude Code CLI version: {result.stdout.strip()}")
        if result.returncode != 0:
            print(f"[{prefix}] CLI stderr: {result.stderr}")
        return True
    except FileNotFoundError:
        print(f"{prefix}_ERROR: Claude Code CLI not found. Install with: npm install -g @anthropic-ai/claude-code")
        sys.exit(1)


def create_keep_stream_hook():
    """Create the keep_stream_open hook for can_use_tool.

    Returns:
        Async function that returns {"continue_": True}
    """
    async def keep_stream_open(input_data, tool_use_id, context):
        return {"continue_": True}
    return keep_stream_open


def create_prompt_stream(prompt_text: str, timeout: int = 900):
    """Create an async generator that yields the prompt and stays alive.

    The generator must stay alive for the entire session so the control channel
    between the Python SDK and CLI subprocess remains open.

    Args:
        prompt_text: The prompt to send to the agent
        timeout: How long to keep the stream alive (default 900s)

    Returns:
        Tuple of (async generator function, anyio.Event). Signal the event
        after the query loop completes so the generator exits immediately
        instead of blocking for the full timeout.
    """
    import anyio
    _done_event = anyio.Event()

    async def prompt_stream():
        yield {
            "type": "user",
            "message": {
                "role": "user",
                "content": prompt_text
            }
        }
        # Keep stream alive for can_use_tool/hook callbacks.
        # The done_event is signaled by the caller when the query loop ends,
        # so this unblocks immediately. The move_on_after is a safety fallback.
        try:
            with anyio.move_on_after(timeout):
                await _done_event.wait()
        except BaseException:
            pass

    return prompt_stream, _done_event
