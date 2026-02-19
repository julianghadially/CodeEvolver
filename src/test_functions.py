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


def test_evaluate_impl(
    repo_url: str,
    program: str,
    metric: str,
    batch: list[dict] | None = None,
    batch_path: str | None = None,
    candidate: dict[str, str] | None = None,
    git_branch: str = "main",
    saved_program_json_path: str | None = None,
    program_lm: str = "openai/gpt-5-mini",
    num_threads: int = 1,
    input_keys: list[str] | None = None,
    max_rows: int = 20,
    installation_id: int | None = None,
    capture_traces: bool = False,
    app_for_sandbox: Any | None = None,
) -> dict:
    """Implementation for test_evaluate Modal function.

    Runs evaluation on a dataset batch using exec_prebuilt, exactly like
    production. If no candidate is provided, builds the seed candidate first
    to extract default prompts.

    Args:
        repo_url: Git repository URL.
        program: Dotted import path to DSPy module.
        metric: Dotted import path to metric function.
        batch: Inline dataset (list of dicts). Either this or batch_path required.
        batch_path: Path to a data file in the repo (JSON/JSONL/CSV).
        candidate: Explicit prompt texts. If None, seed candidate is built.
        git_branch: Branch to evaluate (default "main").
        saved_program_json_path: Optional path to program.json.
        program_lm: LM for evaluation (default "openai/gpt-5-mini").
        num_threads: Parallelism (default 1).
        input_keys: Explicit input field names.
        max_rows: Cap on dataset size (default 20).
        installation_id: GitHub App installation ID for private repos.
        capture_traces: Whether to capture DSPy traces (default False).
        app_for_sandbox: Modal app instance for sandbox creation.

    Returns:
        Dict with 'success', 'scores', 'mean_score', 'outputs',
        'num_examples', 'error', 'error_details', 'logs'.
    """
    import json as json_mod
    import os
    sys.path.insert(0, "/app")
    os.chdir("/app")

    from src.optimizer.gepa_sandbox import GEPASandbox
    from src.services.github_app import GitHubAppService

    logs: list[str] = ["Using GEPASandbox for evaluate test"]
    sandbox = None

    try:
        # --- Validate inputs ---
        if batch is None and batch_path is None:
            return {
                "success": False,
                "error": "Either 'batch' or 'batch_path' must be provided.",
                "logs": logs,
            }

        # --- GitHub auth ---
        github_token = None
        if installation_id:
            github_token = GitHubAppService.get_installation_token(installation_id)
            logs.append("GitHub authentication successful")

        # --- Create and start sandbox ---
        sandbox = GEPASandbox(
            app=app_for_sandbox,
            repo_url=repo_url,
            github_token=github_token,
            timeout=1500,
        )
        logs.append(f"Created GEPASandbox for {repo_url}")

        sandbox.start(use_venv=True, branch=git_branch)
        logs.append(f"Sandbox started (branch: {git_branch})")

        # --- Load dataset ---
        if batch is not None:
            dataset = batch
            logs.append(f"Using inline batch ({len(dataset)} rows)")
        else:
            logs.append(f"Loading dataset from {batch_path}...")
            cat_result = sandbox.exec_bash(f"cat {batch_path}")
            raw = cat_result.get("stdout", "")
            if not raw.strip():
                return {
                    "success": False,
                    "error": f"batch_path '{batch_path}' is empty or not found.",
                    "logs": logs,
                }

            # Detect format and parse
            stripped = raw.strip()
            if stripped.startswith("["):
                # JSON array
                dataset = json_mod.loads(stripped)
            elif stripped.startswith("{"):
                # JSONL (one JSON object per line)
                dataset = [json_mod.loads(line) for line in stripped.splitlines() if line.strip()]
            else:
                # CSV – parse with csv module
                import csv
                import io
                reader = csv.DictReader(io.StringIO(raw))
                dataset = [dict(row) for row in reader]

            logs.append(f"Loaded {len(dataset)} rows from {batch_path}")

        # Apply max_rows cap
        if len(dataset) > max_rows:
            dataset = dataset[:max_rows]
            logs.append(f"Capped dataset to {max_rows} rows")

        # --- Optionally build seed candidate ---
        if candidate is None:
            logs.append("No candidate provided; building seed candidate...")
            seed_result = sandbox.exec_prebuilt({
                "command": "build_seed_candidate",
                "program": program,
                "saved_program_json_path": saved_program_json_path,
            })
            if not seed_result.get("success"):
                return {
                    "success": False,
                    "error": f"build_seed_candidate failed: {seed_result.get('error', 'unknown')}",
                    "logs": logs + seed_result.get("logs", []),
                }
            candidate = seed_result.get("candidate", {})
            logs.append(f"Seed candidate built ({len(candidate)} predictors)")

        # --- Run evaluation ---
        logs.append(f"Running evaluation on {len(dataset)} examples...")
        eval_result = sandbox.exec_prebuilt({
            "command": "evaluate",
            "program": program,
            "metric": metric,
            "saved_program_json_path": saved_program_json_path,
            "candidate": candidate,
            "batch": dataset,
            "capture_traces": capture_traces,
            "num_threads": num_threads,
            "input_keys": input_keys or [],
            "failure_score": 0.0,
            "program_lm": program_lm,
            "git_branch": git_branch,
        })

        if not eval_result.get("success", False):
            error_msg = eval_result.get("error", "evaluate failed")
            if eval_result.get("traceback"):
                error_msg += f"\n\nTraceback:\n{eval_result['traceback']}"
            return {
                "success": False,
                "error": error_msg,
                "logs": logs + eval_result.get("logs", []),
            }

        # --- Compute summary stats ---
        scores = eval_result.get("scores", [])
        outputs = eval_result.get("outputs", [])
        mean_score = sum(scores) / len(scores) if scores else None
        error_count = sum(1 for s in scores if s == 0.0)

        logs.append(
            f"Evaluation complete: mean_score={mean_score:.4f}, "
            f"{error_count}/{len(scores)} zero-score examples"
        )

        return {
            "success": True,
            "scores": scores,
            "mean_score": mean_score,
            "outputs": outputs,
            "num_examples": len(scores),
            "error_details": None,
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
        if sandbox:
            try:
                sandbox.stop()
                logs.append("Sandbox stopped")
            except Exception as e:
                logs.append(f"Error stopping sandbox: {e}")
