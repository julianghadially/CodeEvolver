def get_debug_python_command(workspace):
    """Return bash command string to check venv/dspy existence before running Python."""
    return (
        f"echo '[VENV DEBUG] Checking venv before master_script.py...' && "
        f"ls -la {workspace}/.venv/bin/python 2>&1 || echo 'VENV PYTHON MISSING!' && "
        f"ls -la {workspace}/.venv/lib/python*/site-packages/dspy 2>&1 | head -1 || echo 'DSPY PACKAGE MISSING!' && "
        f"which python && "
    )


def get_dspy_import_diagnostic(import_error: Exception) -> str:
    """Build detailed diagnostic string for DSPy import failures.

    Call this in an except block when `import dspy` fails to get
    comprehensive environment info for debugging.

    Args:
        import_error: The ImportError that was raised

    Returns:
        Formatted diagnostic string with environment details
    """
    import os
    import sys

    debug_info = get_debug_env_info()
    venv_site = "/workspace/.venv/lib/python3.11/site-packages"
    dspy_exists = os.path.exists(os.path.join(venv_site, "dspy"))
    site_exists = os.path.exists(venv_site)

    diag_lines = [
        f"DSPy import failed: {import_error}",
        f"Python executable: {debug_info['python_executable']}",
        f"sys.path[0:3]: {debug_info['sys_path_first_3']}",
        f"VIRTUAL_ENV: {debug_info['venv_env']}",
        f"PATH (first 150): {debug_info['path_env_start']}",
        f"Venv site-packages exists: {site_exists}",
        f"dspy in venv site-packages: {dspy_exists}",
    ]

    # Check if venv site-packages should be in sys.path but isn't
    if site_exists and venv_site not in sys.path:
        diag_lines.append(f"WARNING: {venv_site} is NOT in sys.path!")

    # List packages in site-packages if it exists but dspy doesn't
    if site_exists and not dspy_exists:
        try:
            packages = sorted(os.listdir(venv_site))[:15]
            diag_lines.append(f"Site-packages contents (first 15): {packages}")
        except Exception as e:
            diag_lines.append(f"Could not list site-packages: {e}")

    return "\n".join(diag_lines)


def get_debug_env_info():
    # Debug: Log environment before attempting dspy import
    # This helps diagnose iteration 7-8 failures where dspy suddenly becomes unavailable
    import os
    import sys
    _debug_info = {
        "python_executable": sys.executable,
        "sys_path_first_3": sys.path[:3],
        "venv_env": os.environ.get("VIRTUAL_ENV", "NOT SET"),
        "path_env_start": os.environ.get("PATH", "NOT SET")[:150],
    }
    return _debug_info

def _log_environment_debug(logger):
    import sys
    import os
    """Log environment info to help debug import issues."""
    logger.info(f"Python executable: {sys.executable}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"sys.path: {sys.path[:5]}...")  # First 5 entries
    logger.info(f"PATH env: {os.environ.get('PATH', 'NOT SET')[:200]}...")
    logger.info(f"VIRTUAL_ENV: {os.environ.get('VIRTUAL_ENV', 'NOT SET')}")
    logger.info(f"CWD: {os.getcwd()}")

    # Check if venv site-packages exists and contains dspy
    venv_site = "/workspace/.venv/lib/python3.11/site-packages"
    if os.path.exists(venv_site):
        logger.info(f"Venv site-packages exists: {venv_site}")
        dspy_path = os.path.join(venv_site, "dspy")
        if os.path.exists(dspy_path):
            logger.info(f"dspy package found at: {dspy_path}")
        else:
            logger.warn(f"dspy package NOT found at: {dspy_path}")
            # List what's in site-packages
            try:
                packages = sorted(os.listdir(venv_site))[:20]
                logger.info(f"Site-packages contents (first 20): {packages}")
            except Exception as e:
                logger.warn(f"Could not list site-packages: {e}")
    else:
        logger.warn(f"Venv site-packages NOT found: {venv_site}")

