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

    Clears all modules whose source file lives under /workspace/ but NOT
    under /workspace/.venv/, which covers user code (e.g., src.*,
    langProBe.*) while preserving installed packages (dspy, numpy, etc.).

    The venv site-packages lives at /workspace/.venv/lib/python3.11/site-packages/,
    so a naive "/workspace/" check would incorrectly match dspy and other
    installed packages, destroying their configuration state (e.g., dspy.configure).

    Note: We intentionally do NOT call importlib.invalidate_caches() as it
    corrupts dspy's module locators over repeated invocations.

    Args:
        logger: Logger instance with info() method
    """
    workspace_modules = []
    for name, mod in list(sys.modules.items()):
        if mod is None:
            continue
        mod_file = getattr(mod, "__file__", None) or ""
        mod_path = getattr(mod, "__path__", None)
        # Check if module's source file is under /workspace/ but NOT in the venv
        if "/workspace/" in mod_file and "/workspace/.venv/" not in mod_file:
            workspace_modules.append(name)
        # Also check package __path__ for namespace packages without __file__
        elif mod_path and any(
            "/workspace/" in str(p) and "/workspace/.venv/" not in str(p)
            for p in mod_path
        ):
            workspace_modules.append(name)

    for mod_name in workspace_modules:
        del sys.modules[mod_name]
    if workspace_modules:
        logger.info(f"Cleared {len(workspace_modules)} cached workspace modules")
