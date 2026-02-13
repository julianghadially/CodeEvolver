"""Git command utilities for sandbox operations.

Provides reusable git operations for prebuilt commands like build_seed_candidate and evaluate.
"""

import subprocess
import sys
from typing import Any


def checkout_branch_if_needed(
    workspace: str,
    git_branch: str,
    logger: Any,
    clear_modules: bool = True,
) -> bool:
    """Checkout a git branch if not already on it.

    This function checks the current branch and only performs checkout
    if the target branch is different, avoiding unnecessary git operations
    and module cache invalidation.

    Args:
        workspace: Path to the git repository
        git_branch: Target branch name to checkout
        logger: Logger instance with info() and warn() methods
        clear_modules: If True, clear workspace modules from sys.modules after checkout

    Returns:
        True if checkout was performed, False if already on target branch

    Raises:
        subprocess.CalledProcessError: If git commands fail
    """
    # Check current branch
    current_branch_result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=True,
    )
    current_branch = current_branch_result.stdout.strip()

    # Skip checkout if already on target branch
    if current_branch == git_branch:
        logger.info(f"Already on branch {git_branch}, skipping checkout")
        return False

    # Perform checkout
    logger.info(f"Checking out branch: {git_branch} (current: {current_branch})")
    checkout_result = subprocess.run(
        ["git", "checkout", git_branch],
        cwd=workspace,
        capture_output=True,
        text=True,
    )

    if checkout_result.returncode != 0:
        logger.warn(f"Failed to checkout branch {git_branch}: {checkout_result.stderr}")
        return False

    logger.info(f"Successfully checked out branch: {git_branch}")

    # Clear workspace modules from sys.modules to force reimport from new branch
    if clear_modules:
        _clear_workspace_modules(logger)

    return True


def _clear_workspace_modules(logger: Any) -> None:
    """Clear workspace modules from sys.modules cache.

    Only clears user code modules (src.*), NOT external packages like dspy.
    Note: We intentionally do NOT call importlib.invalidate_caches() as it
    corrupts dspy's module locators over repeated invocations.

    Args:
        logger: Logger instance with info() method
    """
    workspace_modules = [
        name for name in list(sys.modules.keys())
        if name.startswith("src.") or name == "src"
    ]
    for mod_name in workspace_modules:
        del sys.modules[mod_name]
        logger.info(f"Cleared cached module: {mod_name}")
