"""Modal functions for component testing.

These functions support the /test/ API endpoints without cluttering modal_app.py.
They use GEPASandbox to provide isolated test execution that mirrors production.
"""

import sys
from typing import Any

# ---------------------------------------------------------------------------
# Modal Test Functions
# ---------------------------------------------------------------------------

def test_build_seed_candidate_impl(
    repo_url: str,
    program: str,
    initial_branch: str = "main",
    saved_program_json_path: str | None = None,
    installation_id: int | None = None,
    refactor_files: dict[str, str] | None = None,
    refactor_classes: dict[str, str] | None = None,
    app_for_sandbox: Any | None = None,
) -> dict:
    """Implementation for test_build_seed_candidate Modal function.

    Uses GEPASandbox to run build_seed_candidate in the client's environment,
    exactly like in production. This avoids import conflicts between CodeEvolver
    and client repositories.

    Args:
        repo_url: Git repository URL
        program: Dotted import path to DSPy module
        initial_branch: Branch to clone
        saved_program_json_path: Optional path to program.json
        installation_id: GitHub App installation ID for private repos
        refactor_files: Optional dict of file renames {'old_path': 'new_path'}
        refactor_classes: Optional dict of class renames {'OldClass': 'NewClass'}
        app_for_sandbox: Modal app instance for sandbox creation

    Returns:
        Dict with 'success', 'candidate', 'num_predictors', 'predictor_names', 'error', 'logs'
    """
    import os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    # Import GEPASandbox and services (these are CodeEvolver modules)
    from src.optimizer.gepa_sandbox import GEPASandbox
    from src.services.github_app import GitHubAppService

    logs = ["Using GEPASandbox for test (production-like environment)"]
    sandbox = None

    try:
        # Get GitHub token if installation_id provided
        github_token = None
        if installation_id:
            github_token = GitHubAppService.get_installation_token(installation_id)
            logs.append("✓ GitHub authentication successful")

        # Create GEPASandbox (just like in production)
        sandbox = GEPASandbox(
            app=app_for_sandbox,
            repo_url=repo_url,
            github_token=github_token,
            timeout=600,
        )
        logs.append(f"✓ Created GEPASandbox for {repo_url}")

        # Start sandbox with client's environment
        sandbox.start(use_venv=True, branch=initial_branch)
        logs.append(f"✓ Sandbox started (branch: {initial_branch})")

        # Apply file refactoring if requested (run directly in sandbox)
        if refactor_files:
            logs.append(f"Applying file renames: {len(refactor_files)} files...")
            for old_path, new_path in refactor_files.items():
                result = sandbox.exec_bash(f"test -f {old_path} && mkdir -p $(dirname {new_path}) && mv {old_path} {new_path} || echo 'File not found: {old_path}'")
                if "File not found" in result["stdout"]:
                    logs.append(f"  ⚠ File not found: {old_path}")
                else:
                    logs.append(f"  ✓ Renamed {old_path} → {new_path}")

                # Also update import statements that reference the old module name
                # e.g. renaming research_agent_module.py → research_agent_module_test.py
                # needs to update "from .research_agent_module import" → "from .research_agent_module_test import"
                old_module = os.path.splitext(os.path.basename(old_path))[0]  # e.g. "research_agent_module"
                new_module = os.path.splitext(os.path.basename(new_path))[0]  # e.g. "research_agent_module_test"
                if old_module != new_module:
                    # Update all Python files in the same directory and parent directories
                    parent_dir = os.path.dirname(old_path) or "src"
                    result = sandbox.exec_bash(
                        f"find {parent_dir} -name '*.py' -exec sed -i 's/{old_module}/{new_module}/g' {{}} +"
                    )
                    logs.append(f"  ✓ Updated module imports: {old_module} → {new_module}")

        # Apply class refactoring if requested (run directly in sandbox)
        if refactor_classes:
            logs.append(f"Applying class renames: {len(refactor_classes)} classes...")
            for old_class, new_class in refactor_classes.items():
                # Use sed to replace class names in all Python files
                result = sandbox.exec_bash(f"find src/ -name '*.py' -exec sed -i 's/{old_class}/{new_class}/g' {{}} +")
                logs.append(f"  ✓ Renamed class {old_class} → {new_class}")

        # Call build_seed_candidate via exec_prebuilt (just like in production)
        logs.append(f"Building seed candidate for {program}...")
        result = sandbox.exec_prebuilt({
            "command": "build_seed_candidate",
            "program": program,
            "saved_program_json_path": saved_program_json_path,
        })

        if not result.get("success"):
            return {
                "success": False,
                "error": result.get("error", "build_seed_candidate failed"),
                "logs": logs + result.get("logs", []),
            }

        candidate = result.get("candidate", {})
        predictor_names = list(candidate.keys())

        logs.append(f"✓ Found {len(predictor_names)} predictors: {predictor_names}")

        return {
            "success": True,
            "candidate": candidate,
            "num_predictors": len(predictor_names),
            "predictor_names": predictor_names,
            "logs": logs,
        }

    except Exception as e:
        import traceback
        return {
            "success": False,
            "error": f"Test failed: {e}",
            "logs": logs + [traceback.format_exc()],
        }
    finally:
        # Stop sandbox (clean up)
        if sandbox:
            try:
                sandbox.stop()
                logs.append("✓ Sandbox stopped")
            except Exception as e:
                logs.append(f"⚠ Error stopping sandbox: {e}")
