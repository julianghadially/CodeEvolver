"""Utility functions for GEPA optimization.

Pure file I/O and import utilities. No dspy dependency.
"""

import csv
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Callable

import litellm


# Reserved key for code component (underscore prefix distinguishes from DSPy predictor names)
# git_branch is stored INSIDE _code to prevent GEPA from treating it as a mutable component
CODE_COMPONENT_KEY = "_code"


def load_import_path(workspace_path: str, dotted_path: str) -> Any:
    """Import an attribute from a dotted path relative to the workspace.

    The last component is the attribute name, everything before is the module.
    e.g., "eval.evaluate.metric" → from eval.evaluate import metric

    Args:
        workspace_path: Path to the cloned repo root (added to sys.path).
        dotted_path: Dotted import path (e.g., "src.module.ClassName").

    Returns:
        The imported attribute (class, function, etc.).
    """
    ws = str(workspace_path)
    if ws not in sys.path:
        sys.path.insert(0, ws)

    module_path, attr_name = dotted_path.rsplit(".", 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, attr_name)


def load_dataset_from_file(file_path: Path) -> list[dict[str, Any]]:
    """Load a dataset from a file. Supports .json, .jsonl, and .csv.

    Args:
        file_path: Absolute path to the data file.

    Returns:
        List of dicts, one per example.
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix == ".json":
        with open(file_path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        raise ValueError(f"JSON file must contain a list of objects, got {type(data).__name__}")

    if suffix == ".jsonl":
        items = []
        with open(file_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if suffix == ".csv":
        items = []
        with open(file_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                items.append(dict(row))
        return items

    raise ValueError(f"Unsupported dataset format '{suffix}'. Use .json, .jsonl, or .csv")


def get_reflection_lm_callable(model_name: str) -> Callable[[str], str]:
    """Create a callable that invokes an LM via LiteLLM.

    GEPA's InstructionProposalSignature.run() expects an LM callable,
    not a model name string. This function wraps the model name into
    a callable using LiteLLM.

    Args:
        model_name: LiteLLM model identifier (e.g., "openai/gpt-5-mini").

    Returns:
        A function that takes a prompt string and returns a response string.
    """
    def lm_fn(prompt: str) -> str:
        response = litellm.completion(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content

    return lm_fn


def get_git_branch_from_candidate(
    candidate: dict[str, str],
    fallback_branch: str = "main",
) -> str:
    """Extract git_branch from the _code component of a candidate.

    Args:
        candidate: Candidate dict with _code component.
        fallback_branch: Branch to return if not found in _code.

    Returns:
        The git branch name, or fallback_branch if not found.
    """
    code_data = json.loads(candidate.get(CODE_COMPONENT_KEY, "{}"))
    return code_data.get("git_branch", fallback_branch)


def extract_run_timestamp_from_branch(branch_name: str) -> str | None:
    """Extract the run timestamp from a CodeEvolver branch name.

    Branch names follow the pattern: codeevolver-{YYYYMMDDHHmmss}-{suffix}
    where suffix is either "main" or a uuid.

    Args:
        branch_name: Branch name (e.g., "codeevolver-20260203143000-main").

    Returns:
        The timestamp string (e.g., "20260203143000"), or None if not found.
    """
    if not branch_name.startswith("codeevolver-"):
        return None

    # Remove "codeevolver-" prefix and split on "-"
    parts = branch_name[len("codeevolver-"):].split("-", 1)
    if parts:
        return parts[0]
    return None


def create_ce_main_branch(
    sandbox: Any,
    initial_branch: str,
    ce_main_branch: str,
) -> None:
    """Create the run's main branch from the initial branch.

    Args:
        sandbox: GEPASandbox instance with exec_bash method.
        initial_branch: Branch to create from (e.g., "main").
        ce_main_branch: Name of the new branch to create.

    Raises:
        RuntimeError: If branch creation fails.
    """
    # Checkout initial branch (usually "main")
    checkout_result = sandbox.exec_bash(f"git checkout {initial_branch}")
    if checkout_result.get("returncode") != 0:
        raise RuntimeError(
            f"Failed to checkout initial branch {initial_branch}: "
            f"{checkout_result.get('stderr')}"
        )

    # Create and checkout run's main branch
    create_result = sandbox.exec_bash(f"git checkout -b {ce_main_branch}")
    if create_result.get("returncode") != 0:
        raise RuntimeError(
            f"Failed to create run main branch {ce_main_branch}: "
            f"{create_result.get('stderr')}"
        )

    print(f"[UTILS] Created run main branch {ce_main_branch} from {initial_branch}", flush=True)


def save_file_to_sandbox(
    sandbox: Any,
    content: str,
    path: str,
    push: bool = True,
    commit_message: str | None = None,
    branch: str | None = None,
) -> bool:
    """Save a string to a file within the client sandbox.

    Args:
        sandbox: GEPASandbox instance with exec_bash and push_authenticated methods.
        content: String content to write.
        path: Relative path within workspace (e.g., "codeevolver.md").
        push: If True (default), push to remote after committing.
        commit_message: Commit message. If None, no commit is made.
        branch: Branch to push to. Required if push=True.

    Returns:
        True if successful, False otherwise.

    Raises:
        RuntimeError: If push is requested but fails.
    """
    # Write content to file using heredoc
    write_result = sandbox.exec_bash(
        f"cat > {path} << 'CODEEVOLVER_EOF'\n{content}\nCODEEVOLVER_EOF"
    )
    if write_result.get("returncode") != 0:
        print(f"[UTILS] Warning: Failed to write {path}: {write_result.get('stderr')}", flush=True)
        return False

    # Commit if message provided
    if commit_message:
        sandbox.exec_bash("git config user.email 'codeevolver@codeevolver.ai'")
        sandbox.exec_bash("git config user.name 'CodeEvolver'")
        sandbox.exec_bash(f"git add {path}")

        commit_result = sandbox.exec_bash(f'git commit -m "{commit_message}"')
        if commit_result.get("returncode") != 0:
            print(f"[UTILS] Warning: Failed to commit {path}: {commit_result.get('stderr')}", flush=True)
            return False

        print(f"[UTILS] Committed {path}", flush=True)

    # Push if requested
    if push:
        if not branch:
            raise ValueError("branch is required when push=True")
        push_result = sandbox.push_authenticated(branch)
        if not push_result.get("success"):
            raise RuntimeError(
                f"Failed to push {branch}: {push_result.get('stderr')}"
            )
        print(f"[UTILS] Pushed {branch} to origin", flush=True)

    return True
