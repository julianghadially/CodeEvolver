"""Utility functions for GEPA optimization.

Pure file I/O and import utilities. No dspy dependency.
"""

import csv
import importlib
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable

import litellm


# Reserved key for code component (underscore prefix distinguishes from DSPy predictor names)
# git_branch is stored INSIDE _code to prevent GEPA from treating it as a mutable component
CODE_COMPONENT_KEY = "_code"

# Keys for codeevolver.md header fields
PARENT_MODULE_PATH_KEY = "PARENT_MODULE_PATH"
METRIC_MODULE_PATH_KEY = "METRIC_MODULE_PATH"


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


def ensure_gitignore_committed(
    sandbox: Any,
    branch: str,
    entries: list[str] | None = None,
) -> bool:
    """Ensure .gitignore has sandbox entries and commit/push to branch.

    This should be called early in the optimization workflow (after creating
    the codeevolver main branch) so all mutation branches inherit the proper
    .gitignore. Prevents .venv from being committed during code mutations.

    Args:
        sandbox: GEPASandbox instance with exec_bash and push_authenticated methods.
        branch: Branch to commit and push to (e.g., codeevolver-{timestamp}-main).
        entries: Entries to add to .gitignore. Defaults to [".venv", ".env"].

    Returns:
        True if successful, False otherwise.
    """
    if entries is None:
        entries = [".venv", ".env"]

    print(f"[UTILS] Ensuring .gitignore has entries: {entries}", flush=True)

    # Build script to add entries to .gitignore if not present
    gitignore_path = ".gitignore"
    script_lines = [
        f"touch {gitignore_path}",
        # Ensure file ends with newline before appending
        f"[ -s {gitignore_path} ] && [ -n \"$(tail -c1 {gitignore_path})\" ] && echo '' >> {gitignore_path}",
    ]
    for entry in entries:
        # Add entry if not already present (exact line match)
        script_lines.append(
            f"grep -qxF '{entry}' {gitignore_path} || echo '{entry}' >> {gitignore_path}"
        )

    script = " && ".join(script_lines)
    result = sandbox.exec_bash(script)
    if result.get("returncode") != 0:
        print(f"[UTILS] Warning: Failed to update .gitignore: {result.get('stderr')}", flush=True)
        return False

    # Configure git user and commit
    sandbox.exec_bash("git config user.email 'codeevolver@codeevolver.ai'")
    sandbox.exec_bash("git config user.name 'CodeEvolver'")
    sandbox.exec_bash(f"git add {gitignore_path}")

    # Check if there are changes to commit
    status_result = sandbox.exec_bash("git diff --cached --quiet")
    if status_result.get("returncode") == 0:
        print(f"[UTILS] .gitignore already up to date, no commit needed", flush=True)
    else:
        commit_result = sandbox.exec_bash('git commit -m "Add sandbox artifacts to .gitignore"')
        if commit_result.get("returncode") != 0:
            print(f"[UTILS] Warning: Failed to commit .gitignore: {commit_result.get('stderr')}", flush=True)
            return False
        print(f"[UTILS] Committed .gitignore", flush=True)

    # Push to remote
    push_result = sandbox.push_authenticated(branch)
    if not push_result.get("success"):
        print(f"[UTILS] Warning: Failed to push .gitignore: {push_result.get('stderr')}", flush=True)
        return False

    print(f"[UTILS] Pushed .gitignore to {branch}", flush=True)
    return True


def parse_codeevolver_md(content: str) -> dict[str, str | None]:
    """Parse PARENT_MODULE_PATH and METRIC_MODULE_PATH from codeevolver.md content.

    The file is expected to have these fields at the top in the format:
        PARENT_MODULE_PATH: src.module.ClassName
        METRIC_MODULE_PATH: eval.metric

    Args:
        content: The full content of codeevolver.md.

    Returns:
        Dict with 'parent_module_path' and 'metric_module_path' keys.
        Values are None if not found.
    """
    result = {
        "parent_module_path": None,
        "metric_module_path": None,
    }

    for line in content.split("\n"):
        line = line.strip()
        if line.startswith(f"{PARENT_MODULE_PATH_KEY}:"):
            value = line[len(f"{PARENT_MODULE_PATH_KEY}:"):].strip()
            # Remove any surrounding quotes
            value = value.strip('"').strip("'")
            if value:
                result["parent_module_path"] = value
        elif line.startswith(f"{METRIC_MODULE_PATH_KEY}:"):
            value = line[len(f"{METRIC_MODULE_PATH_KEY}:"):].strip()
            value = value.strip('"').strip("'")
            if value:
                result["metric_module_path"] = value

        # Stop parsing after we find both or hit the architecture content
        if result["parent_module_path"] and result["metric_module_path"]:
            break
        # Stop if we hit a markdown header (architecture section started)
        if line.startswith("#") and result["parent_module_path"]:
            break

    return result


def read_codeevolver_md_from_sandbox(sandbox: Any) -> str | None:
    """Read codeevolver.md content from the sandbox workspace.

    Args:
        sandbox: GEPASandbox instance with exec_bash method.

    Returns:
        Content of codeevolver.md, or None if file doesn't exist.
    """
    result = sandbox.exec_bash("cat codeevolver.md 2>/dev/null")
    if result.get("returncode") == 0:
        return result.get("stdout", "")
    return None


def subsample_validation_set(
    valset: list[dict[str, Any]] | None,
    max_valset_size: int | None,
    seed: int = 0,
) -> list[dict[str, Any]] | None:
    """Randomly subsample validation set if max_valset_size is specified.

    This function deterministically subsamples the validation set using the provided
    seed, ensuring the same subsample is used throughout the optimization run.
    This significantly speeds up evaluation time without cluttering the main code.

    Args:
        valset: Full validation dataset (list of dicts), or None.
        max_valset_size: Maximum number of validation examples to use. If None or
            greater than len(valset), the full validation set is used.
        seed: Random seed for reproducible subsampling (default: 0).

    Returns:
        Subsampled validation set, full validation set, or None if valset is None.

    Example:
        >>> valset = [{"x": i} for i in range(300)]
        >>> subsampled = subsample_validation_set(valset, max_valset_size=100, seed=42)
        >>> len(subsampled)
        100
    """
    if valset is None:
        return None

    if max_valset_size is None or max_valset_size >= len(valset):
        print(
            f"[UTILS] Using full validation set ({len(valset)} examples)",
            flush=True
        )
        return valset

    # Deterministically subsample using the provided seed
    rng = random.Random(seed)
    subsampled = rng.sample(valset, max_valset_size)

    print(
        f"[UTILS] Subsampled validation set: {len(subsampled)} examples "
        f"(from {len(valset)} total, seed={seed})",
        flush=True
    )

    return subsampled
