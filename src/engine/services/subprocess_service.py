"""Copyright © 2026 440 Labs LLC

Subprocess-based service layer for operations that run inside the sandbox.

Replaces the v1 sandbox.exec_* RPC adapter. Every function here runs as a
direct subprocess call inside the sandbox Python process — git operations,
evaluator launch, coding/inspection agent launch, bash, file I/O. Used by
the EE optimizer, seed step, evaluator client, and environment validator.

Assumes:
  - /workspace (or the configured workspace path) contains the cloned repo
  - Client deps are in /workspace/.venv
  - Git, Python, Node.js, Claude Code CLI are installed
  - ANTHROPIC_API_KEY is set in the environment

No database imports, no sandbox object dependency.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Any

from src.engine.agent.harnesses import (
    AgentRole,
    HarnessPermissionPolicy,
    HarnessRequest,
)
from src.engine.agent.harnesses.factory import create_coding_harness

logger = logging.getLogger(__name__)


def _parse_usage_marker(stdout: str, prefix: str) -> dict:
    """Parse a `<PREFIX>{json}` token-usage marker emitted by a subprocess agent.

    Returns the parsed dict (token counts plus a `model` key) or `{}` when no
    valid marker is present. Scans for the last matching line so a degraded
    run that prints nothing simply yields no usage rather than erroring.
    """
    if not stdout or prefix not in stdout:
        return {}
    for line in reversed(stdout.split("\n")):
        if line.startswith(prefix):
            try:
                data = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                return data
    return {}


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

from src.engine.paths import (
    agent_user,
    engine_root,
    is_local,
    repo_venv_bin,
    repo_venv_dirname,
)


def _agent_user_kwargs() -> dict[str, str]:
    """``{"user": <name>}`` when CODEEVOLVER_AGENT_USER is set, else ``{}``.

    Spread into agent-process spawns only (never the evaluator or git
    helpers): inside the Docker runner the agent session runs as the
    unprivileged user and cannot read the root-owned mode-600 holdout.
    Empty everywhere else — local runs are unchanged. No fallback on
    failure; see ``paths.agent_user()``.
    """
    user = agent_user()
    return {"user": user} if user else {}


def _engine_script(*parts: str) -> str:
    """Resolve a script path under the engine root (e.g., master_script)."""
    return os.path.join(engine_root(), *parts)


def _build_env(workspace: str) -> dict[str, str]:
    """Build subprocess environment with venv and PYTHONPATH set.

    Engine source root is auto-detected from ``src.engine.paths`` (walks up
    from ``paths.py``'s own ``__file__`` — works in a container and any local
    clone). Venv defaults switch on ``SANDBOX_MODE``: non-local → ``.venv``,
    local → ``env``. ``REPO_VENV_DIR`` env var overrides if the operator
    needs a non-default venv name.
    """
    env = os.environ.copy()
    venv_bin = f"{workspace}/{repo_venv_dirname()}/{repo_venv_bin()}"
    if os.path.exists(venv_bin):
        env["PATH"] = venv_bin + ":" + env.get("PATH", "")
    env["PYTHONPATH"] = engine_root() + ":" + env.get("PYTHONPATH", "")
    # CODEEVOLVER_WORKSPACE_SRC: literal-string write of a cross-process
    # env var (registry in src/engine/paths.py::CROSS_PROCESS_ENV_VARS).
    # Read by src/__init__.py at import time to merge the client repo's
    # src/ into the `src` package __path__.
    env.setdefault("CODEEVOLVER_WORKSPACE_SRC", os.path.join(workspace, "src"))
    return env


# ---------------------------------------------------------------------------
# Git operations
# ---------------------------------------------------------------------------

def _git_run(
    workspace: str, *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess:
    """Run a git command in the workspace. Returns CompletedProcess.

    `env` overlays the inherited environment (used by `git_tree_signature`
    to redirect `GIT_INDEX_FILE` at a throwaway index).
    """
    return subprocess.run(
        ["git", *args],
        cwd=workspace,
        capture_output=True,
        text=True,
        env={**os.environ, **env} if env else None,
    )


def git_capture(workspace: str, *args: str) -> str:
    """Run a read-only git command and return its stdout, "" on failure.

    A genuinely non-raising accessor for git output that is *compared* rather
    than acted on (the clean-slate inspector's before/after workspace
    snapshot). Both a non-zero exit and an OSError — a missing or non-repo
    workspace — degrade to an empty string, which compares equal across the
    pair rather than manufacturing a spurious mismatch.
    """
    try:
        result = _git_run(workspace, *args)
    except OSError:
        return ""
    return result.stdout if result.returncode == 0 else ""


def git_checkout(workspace: str, branch: str, *, force: bool = False) -> None:
    """Check out a branch in the workspace.

    `force=True` runs `checkout -f`, discarding local modifications that
    would otherwise block the switch. Only finalization uses it — anything
    uncommitted at that point is leftovers from the last architect session,
    and the run's committed state is what gets scored. Mid-run callers keep
    the default so a blocked checkout still fails loudly.
    """
    args = ("checkout", "-f", branch) if force else ("checkout", branch)
    result = _git_run(workspace, *args)
    if result.returncode != 0:
        raise RuntimeError(f"git checkout {branch} failed: {result.stderr}")


def clean_python_bytecode(workspace: str) -> None:
    """Remove Python bytecode artifacts without touching runtime dirs like env."""
    for root, dirs, files in os.walk(workspace):
        if ".git" in dirs:
            dirs.remove(".git")
        for dirname in list(dirs):
            if dirname == "__pycache__":
                shutil.rmtree(os.path.join(root, dirname), ignore_errors=True)
                dirs.remove(dirname)
        for filename in files:
            if filename.endswith((".pyc", ".pyo")):
                try:
                    os.remove(os.path.join(root, filename))
                except FileNotFoundError:
                    pass


def git_create_branch(workspace: str, new_branch: str, source: str) -> None:
    """Create and check out a new branch from source."""
    result = _git_run(workspace, "checkout", "-b", new_branch, source)
    if result.returncode != 0:
        raise RuntimeError(
            f"git checkout -b {new_branch} {source} failed: {result.stderr}"
        )


def git_rename_branch(workspace: str, old: str, new: str) -> None:
    """Rename local branch `old` to `new` (`git branch -m`), list-args only."""
    result = _git_run(workspace, "branch", "-m", old, new)
    if result.returncode != 0:
        raise RuntimeError(f"git branch -m {old} {new} failed: {result.stderr}")


def git_create_branch_simple(workspace: str, branch: str) -> None:
    """Create and check out a new branch from current HEAD."""
    result = _git_run(workspace, "checkout", "-b", branch)
    if result.returncode != 0:
        raise RuntimeError(f"git checkout -b {branch} failed: {result.stderr}")


def git_get_sha(workspace: str, ref: str = "HEAD") -> str:
    """Return the SHA for a ref (branch name or HEAD)."""
    result = _git_run(workspace, "rev-parse", ref)
    if result.returncode != 0:
        raise RuntimeError(f"git rev-parse {ref} failed: {result.stderr}")
    return result.stdout.strip()


def git_is_ancestor(workspace: str, sha: str, branch: str) -> bool:
    """Return True iff sha is an ancestor of branch HEAD."""
    result = _git_run(workspace, "merge-base", "--is-ancestor", sha, branch)
    return result.returncode == 0


def git_merge_base(workspace: str, a: str, b: str) -> str | None:
    """Return the merge-base sha of `a` and `b`, or None when there is none
    (disjoint histories) or on any git failure. Non-mutating."""
    result = _git_run(workspace, "merge-base", a, b)
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None


def git_branch_exists(workspace: str, branch: str) -> bool:
    """Return True iff `branch` exists as a local branch. Non-mutating."""
    result = _git_run(workspace, "rev-parse", "--verify", "--quiet", f"refs/heads/{branch}")
    return result.returncode == 0


def git_get_current_branch(workspace: str) -> str | None:
    """Return the current branch name, or None if HEAD is detached."""
    result = _git_run(workspace, "symbolic-ref", "--short", "-q", "HEAD")
    if result.returncode != 0:
        return None
    name = result.stdout.strip()
    return name or None


def git_is_dirty(workspace: str) -> bool:
    """Return True iff the working tree has uncommitted changes (staged,
    unstaged, or untracked). Non-mutating.

    Returns False on a git failure — an unreadable tree is treated as 'no
    detectable change' so callers (e.g. the no-change watchdog) never fire
    spuriously on a git error. `--porcelain` includes untracked files by
    default, so an uncommitted `Write` of a new file counts as a change."""
    result = _git_run(workspace, "status", "--porcelain")
    if result.returncode != 0:
        return False
    return bool(result.stdout.strip())


def git_dirty_paths(workspace: str, max_paths: int = 20) -> list[str]:
    """Repo-relative paths with uncommitted changes, capped at `max_paths`.

    `git_is_dirty` answers the yes/no; this answers "which files", which is
    what makes a dirty-tree baseline diagnosable after the fact rather than
    just flagged. `[]` on a git failure, matching `git_is_dirty`'s fail-soft
    contract.
    """
    try:
        result = _git_run(workspace, "status", "--porcelain")
    except Exception:  # noqa: BLE001 - observability, never control flow
        return []
    if result.returncode != 0:
        return []
    paths: list[str] = []
    for line in result.stdout.splitlines():
        # Porcelain v1: two status chars, a space, then the path (or
        # "old -> new" for renames).
        entry = line[3:].strip() if len(line) > 3 else ""
        if not entry:
            continue
        paths.append(entry.split(" -> ")[-1])
        if len(paths) >= max_paths:
            break
    return paths


def git_tree_signature(workspace: str, ref: str | None = None) -> str | None:
    """Return the git *tree* hash identifying repo CONTENT.

    With `ref`, this is `git rev-parse <ref>^{tree}` — the content of that
    commit. With `ref=None` it is the content of the actual working tree
    (tracked edits plus untracked, non-ignored files), computed
    non-mutatingly via a throwaway index::

        GIT_INDEX_FILE=<tmp> git add -A
        GIT_INDEX_FILE=<tmp> git write-tree

    The point is COMMIT-INVARIANCE. `git_get_sha` reports HEAD, which is the
    *parent* commit whenever the working tree is dirty — so an eval of
    uncommitted edits gets recorded under the wrong sha, and the sha changes
    the moment the architect commits. The tree hash of a dirty working tree
    is byte-identical to `git_tree_signature(ws, <new sha>)` after
    `git add -A && git commit` of that same content, which is what lets
    "did we already evaluate this code?" survive the commit boundary.

    The real index, HEAD, refs and the working tree are never touched. The
    only side effect is unreferenced loose objects in the object DB, which
    the architect's own subsequent commit reuses and `git gc` reclaims.

    Returns None on ANY failure (not a repo, conflicted index, git missing)
    — fail-soft, matching `git_is_dirty`. None is deliberately not a
    self-equal sentinel like `"?"`: callers match with
    `bool(a) and bool(b) and a == b`, so a failed read degrades to sha-only
    behaviour instead of matching everything.
    """
    if ref:
        try:
            result = _git_run(
                workspace, "rev-parse", "--verify", "--quiet", f"{ref}^{{tree}}"
            )
        except Exception:  # noqa: BLE001 - e.g. workspace path does not exist
            return None
        if result.returncode != 0:
            return None
        return result.stdout.strip() or None

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp(prefix="codeevolver-tree-")
        # The temp index lives outside the repo so a concurrent `git status`
        # can never pick it up and cleanup can never touch repo state.
        index_path = os.path.join(tmpdir, "index")
        # Seed from the real index so `add -A` reuses git's stat cache
        # instead of re-hashing every file. Purely an optimisation; when
        # `.git` is a file (worktrees/submodules) this is skipped.
        real_index = os.path.join(workspace, ".git", "index")
        if os.path.isfile(real_index):
            try:
                shutil.copy2(real_index, index_path)
            except OSError:
                pass
        env = {"GIT_INDEX_FILE": index_path}
        # `add -A` honours .gitignore, matching what a real commit would take.
        if _git_run(workspace, "add", "-A", env=env).returncode != 0:
            return None
        tree = _git_run(workspace, "write-tree", env=env)
        if tree.returncode != 0:
            return None
        return tree.stdout.strip() or None
    except Exception:  # noqa: BLE001 - content identity is best-effort
        return None
    finally:
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)


def git_commit_trees(
    workspace: str,
    ref: str = "HEAD",
    *,
    since: str | None = None,
    max_count: int = 100,
) -> list[tuple[str, str]]:
    """Return `[(commit_sha, tree_sha), ...]` reachable from `ref`, NEWEST FIRST.

    With `since`, restricted to `since..ref` — the commits this iteration
    created. Feeds `IterationContext.reconcile_eval_shas`, which uses the
    newest-first order to decide which commit owns a tree when an in-iteration
    revert reproduces an earlier one. Returns `[]` on any git failure.
    """
    rev = f"{since}..{ref}" if since else ref
    try:
        result = _git_run(
            workspace, "log", f"--max-count={max_count}", "--format=%H %T", rev
        )
        if result.returncode != 0 and since:
            # `since` may have been rewritten away (rebase, reset); fall back
            # to the full history rather than losing reconciliation entirely.
            result = _git_run(
                workspace, "log", f"--max-count={max_count}", "--format=%H %T", ref
            )
    except Exception:  # noqa: BLE001 - e.g. workspace path does not exist
        return []
    if result.returncode != 0:
        return []
    out: list[tuple[str, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out.append((parts[0], parts[1]))
    return out


def git_diff_tree_names(workspace: str, a: str, b: str) -> list[str] | None:
    """Return the repo-relative paths that differ between two trees/commits.

    `a` and `b` may be tree shas, commit shas, or any rev git resolves. Returns
    `[]` when the two are content-identical, and `None` on any git failure —
    callers must treat `None` as "unknown" and degrade to exact sha/tree
    equality rather than assuming equivalence.
    """
    if not a or not b:
        return None
    if a == b:
        return []
    try:
        result = _git_run(workspace, "diff-tree", "-r", "--name-only", "--no-commit-id", a, b)
    except Exception:  # noqa: BLE001 - e.g. workspace path does not exist
        return None
    if result.returncode != 0:
        return None
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def git_update_ref(workspace: str, ref: str, sha: str) -> bool:
    """Point `ref` at `sha`. Returns True on success.

    Used to anchor per-iteration end states under `refs/codeevolver/`, which
    makes them reachable (so `git gc` cannot prune them) even after the
    architect resets the branch they were on. Never raises.
    """
    if not ref or not sha:
        return False
    try:
        return _git_run(workspace, "update-ref", ref, sha).returncode == 0
    except Exception:  # noqa: BLE001 - anchoring is best-effort
        return False


def git_delete_ref(workspace: str, ref: str) -> bool:
    """Delete `ref`. Returns True on success. Never raises."""
    if not ref:
        return False
    try:
        return _git_run(workspace, "update-ref", "-d", ref).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def git_set_identity(
    workspace: str,
    *,
    email: str = "ce@codeevolver.ai",
    name: str = "CodeEvolver",
) -> None:
    """Set a repo-local git identity so the AGENT's own `git commit` works.

    The engine's helpers pass `-c user.email/user.name` inline, but the
    coding agent commits with bare `git commit` and dies with "Author
    identity unknown" on a fresh container.
    Repo-local (`git config`, not `--global`) — never leaks outside the
    workspace. Raises on failure like the other git helpers.
    """
    for key, value in (("user.email", email), ("user.name", name)):
        result = _git_run(workspace, "config", key, value)
        if result.returncode != 0:
            raise RuntimeError(f"git config {key} failed: {result.stderr}")


def git_commit_tree(
    workspace: str, tree: str, *, parent: str | None = None, message: str = ""
) -> str | None:
    """Wrap a bare tree object in a commit so it can be diffed and checked out.

    `git_tree_signature` writes real tree objects for every eval, including
    evals of an uncommitted working tree that was later reverted. Those trees
    are unreachable and therefore prunable. Wrapping one in a commit (and
    pointing a ref at it) both preserves it and makes `git diff` / `git
    checkout` work on it naturally. Returns the new commit sha, or None on
    failure.
    """
    if not tree:
        return None
    args = ["commit-tree", tree]
    if parent:
        args += ["-p", parent]
    args += ["-m", message or "codeevolver eval snapshot"]
    try:
        result = _git_run(
            workspace,
            "-c", "user.email=ce@codeevolver.ai",
            "-c", "user.name=CodeEvolver",
            *args,
        )
    except Exception:  # noqa: BLE001
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def git_list_branches(workspace: str, pattern: str = "codeevolver-*") -> dict[str, str]:
    """Return `{branch_name: tip_sha}` for local branches matching `pattern`.

    The close record only ever learns branches the architect *registered*;
    experiment branches it created and abandoned are invisible to the state.
    This sweep records every branch that exists at close. `{}` on failure.
    """
    try:
        result = _git_run(
            workspace,
            "for-each-ref",
            "--format=%(refname:short) %(objectname)",
            f"refs/heads/{pattern}",
        )
    except Exception:  # noqa: BLE001
        return {}
    if result.returncode != 0:
        return {}
    out: dict[str, str] = {}
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def git_add_and_commit(workspace: str, message: str, paths: list[str] | None = None) -> bool:
    """Stage files and commit. Returns True if a commit was made."""
    if paths:
        add_result = subprocess.run(
            ["git", "add"] + paths,
            cwd=workspace,
            capture_output=True,
            text=True,
        )
        if add_result.returncode != 0:
            raise RuntimeError(f"git add failed: {add_result.stderr}")
    else:
        add_result = subprocess.run(
            ["git", "add", "-A"],
            cwd=workspace,
            capture_output=True,
            text=True,
        )

    # Check if there's anything to commit
    diff_result = subprocess.run(
        ["git", "diff", "--cached", "--quiet"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if diff_result.returncode == 0:
        return False  # nothing staged

    commit_result = subprocess.run(
        ["git", "-c", "user.email=ce@codeevolver.ai",
         "-c", "user.name=CodeEvolver",
         "commit", "-m", message],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    if commit_result.returncode != 0:
        raise RuntimeError(f"git commit failed: {commit_result.stderr}")
    return True


def git_fetch(workspace: str, remote: str = "origin") -> None:
    """Fetch from a git remote.

    Skipped in local mode: the local launcher operates in-place on the
    operator's already-cloned repo, so refreshing from origin is both
    unnecessary and may surprise an operator with unpushed local commits.
    """
    if is_local():
        return
    result = _git_run(workspace, "fetch", remote)
    if result.returncode != 0:
        raise RuntimeError(f"git fetch {remote} failed: {result.stderr}")


def _build_authenticated_repo_url(repo_url: str, token: str) -> str:
    """Insert a GitHub token into an https://github.com/... repo URL.

    Mirrors main:src/shared/services/github_app.py::GitHubAppService.get_authenticated_repo_url.
    Inlined here because src/shared/ is not mounted into the sandbox.
    """
    if not repo_url.startswith("https://github.com/"):
        return repo_url
    parts = repo_url.replace("https://github.com/", "").rstrip("/").split("/")
    if len(parts) < 2:
        return repo_url
    owner = parts[0]
    repo = parts[1].replace(".git", "")
    return f"https://x-access-token:{token}@github.com/{owner}/{repo}.git"


def refresh_github_token(
    callback_url: str,
    job_id: str,
    jwt_token: str,
    *,
    timeout: float = 60.0,
    max_retries: int = 3,
) -> str | None:
    """Call the API's /internal/job/{job_id}/github-token endpoint for a fresh token.

    GitHub installation tokens expire after 1 hour. Mirrors
    main:src/engine/sandbox/client_sandbox.py::refresh_github_token —
    bearer-auth with the job-scoped JWT, retries with linear backoff, and
    returns None on any non-transient failure so callers can decide whether
    to proceed with a stale token.

    Skipped in local mode.
    """
    if is_local():
        return None

    if not (callback_url and job_id and jwt_token):
        logger.warning(
            "refresh_github_token: missing callback_url/job_id/jwt_token — skipping"
        )
        return None

    try:
        import httpx
    except ImportError:
        logger.warning("refresh_github_token: httpx not available — skipping")
        return None

    url = f"{callback_url}/internal/job/{job_id}/github-token"
    for attempt in range(max_retries):
        try:
            with httpx.Client(timeout=timeout) as client:
                response = client.get(
                    url, headers={"Authorization": f"Bearer {jwt_token}"}
                )
                response.raise_for_status()
                data = response.json()
            if data.get("error"):
                logger.warning("refresh_github_token: %s", data["error"])
                return None
            token = data.get("token")
            if token:
                logger.info("refresh_github_token: refreshed successfully")
                return token
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt + 1 < max_retries:
                import time
                wait = 5 * (attempt + 1)
                logger.warning(
                    "refresh_github_token attempt %d/%d failed: %s (retrying in %ds)",
                    attempt + 1, max_retries, exc, wait,
                )
                time.sleep(wait)
            else:
                logger.warning(
                    "refresh_github_token: giving up after %d attempts: %s",
                    max_retries, exc,
                )
    return None


def git_push_authenticated(
    workspace: str,
    branch: str,
    repo_url: str,
    *,
    github_token: str | None,
    callback_url: str | None = None,
    job_id: str | None = None,
    jwt_token: str | None = None,
    set_upstream: bool = True,
) -> dict:
    """Push `branch` to `origin` with a fresh GitHub token when available.

    Refresh policy: if all of `callback_url`, `job_id`, `jwt_token` are
    present, we try to mint a fresh token first. Falls back to the token the
    caller passed in, then to an unauthenticated push.

    Branch guard: refuses any branch that does not start with `codeevolver-`,
    the same guard the v1 push-authenticated helper used.

    Returns a dict with keys `success`, `stdout`, `stderr`, `returncode`.

    Skipped in local mode: the local launcher operates in-place on the
    operator's already-cloned repo.
    """
    if is_local():
        return {"success": True, "stdout": "", "stderr": "", "returncode": 0, "skipped": True}

    if not branch.startswith("codeevolver-"):
        msg = (
            f"Branch guard: refused to push to '{branch}'. "
            "Only codeevolver-* branches are allowed."
        )
        logger.error(msg)
        return {"success": False, "stdout": "", "stderr": msg, "returncode": -1}

    token = github_token
    if callback_url and job_id and jwt_token:
        fresh = refresh_github_token(callback_url, job_id, jwt_token)
        if fresh:
            token = fresh

    def _push() -> dict:
        cmd = ["git", "push"]
        if set_upstream:
            cmd.extend(["-u", "origin", branch])
        else:
            cmd.extend(["origin", branch])
        result = subprocess.run(cmd, cwd=workspace, capture_output=True, text=True)
        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
        }

    if not token:
        logger.warning("git_push_authenticated: no github_token — attempting unauthenticated push")
        return _push()

    auth_url = _build_authenticated_repo_url(repo_url, token)

    set_auth = _git_run(workspace, "remote", "set-url", "origin", auth_url)
    if set_auth.returncode != 0:
        logger.warning(
            "git_push_authenticated: remote set-url failed (%s) — unauthenticated push",
            set_auth.stderr,
        )
        return _push()

    try:
        return _push()
    finally:
        # Reset remote URL so the token is never visible in `git remote -v`.
        reset = _git_run(workspace, "remote", "set-url", "origin", repo_url)
        if reset.returncode != 0:
            logger.warning(
                "git_push_authenticated: failed to reset remote URL: %s",
                reset.stderr,
            )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def run_evaluation(
    workspace: str,
    command: dict,
    timeout: int | None = None,
) -> dict:
    """Run the evaluator via master_script.py subprocess.

    Args:
        workspace: Path to the client repo workspace.
        command: The evaluation command dict (same shape as exec_prebuilt).
        timeout: Optional hard timeout in seconds.

    Returns:
        Parsed EVAL_RESULT dict from stdout, or an error dict.
    """
    # One command file per call: a fixed path would let two concurrent
    # `run_evaluation` calls on one host clobber each other's command — the
    # offline best-of-population audit (scripts/best_of_population) evaluates many
    # candidates in parallel worktrees on one machine.
    fd, config_path = tempfile.mkstemp(prefix="prebuilt_eval_command_", suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(command, f)

    cmd = [
        "python",
        _engine_script("src", "engine", "sandbox", "mounted", "master_script.py"),
        "--workspace", workspace,
        "--command-file", config_path,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=workspace,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_build_env(workspace),
        )
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"Evaluation timed out after {timeout} seconds"}
    finally:
        try:
            os.unlink(config_path)
        except OSError:
            pass

    if result.returncode != 0:
        print(
            f"[EXEC] master_script.py failed (rc={result.returncode}): "
            f"{result.stderr[:1000]}",
            flush=True,
        )
        return {
            "success": False,
            "error": f"Prebuilt script exit code {result.returncode}: {result.stderr[:2000]}",
        }

    for line in result.stdout.split("\n"):
        if line.startswith("EVAL_RESULT:"):
            payload = line[len("EVAL_RESULT:"):]
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError as e:
                return {"success": False, "error": f"Failed to parse script output: {e}"}
            _persist_eval_stderr(parsed, result.stderr)
            return parsed

    return {
        "success": False,
        "error": f"Prebuilt script produced no EVAL_RESULT. stderr: {result.stderr[:2000]}",
    }


# The eval subprocess's stderr is captured, but surfacing it only when the
# script FAILS would discard anything the client program writes to stderr
# during a successful eval -- e.g. provider-fallback warnings -- which is
# exactly the case worth seeing. Persist it
# next to the row traces of the eval it belongs to; the trace directory is
# already exported with the run.
EVAL_STDERR_FILENAME = "eval_stderr.log"
EVAL_STDERR_MAX_BYTES = 2 * 1024 * 1024


def _persist_eval_stderr(parsed: dict, stderr: str) -> None:
    """Write a successful eval's stderr to ``<trace_dir>/eval_stderr.log``.

    Best-effort and non-fatal: a missing/unwritable trace dir or an empty
    stderr writes nothing. Truncated to the LAST ``EVAL_STDERR_MAX_BYTES`` so a
    runaway warning loop cannot fill the disk while the tail (where a late
    provider failure lands) is kept.
    """
    if not stderr:
        return
    trace_dir = parsed.get("trace_dir") if isinstance(parsed, dict) else None
    if not trace_dir or not os.path.isdir(trace_dir):
        return
    text = stderr
    if len(text) > EVAL_STDERR_MAX_BYTES:
        text = (
            f"[truncated: kept the last {EVAL_STDERR_MAX_BYTES} of {len(stderr)} chars]\n"
            + text[-EVAL_STDERR_MAX_BYTES:]
        )
    try:
        with open(os.path.join(trace_dir, EVAL_STDERR_FILENAME), "w") as f:
            f.write(text)
    except OSError as e:
        logger.warning("could not persist eval stderr to %s: %s", trace_dir, e)


# ---------------------------------------------------------------------------
# Coding agent
# ---------------------------------------------------------------------------

def run_coding_agent(
    workspace: str,
    change_request: str,
    *,
    change_location: str | None = None,
    max_turns: int = 50,
    commit_changes: bool = True,
    program_path: str | None = None,
    model: str | None = None,
    deny_paths: list[str] | None = None,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = None,
) -> dict:
    """Spawn the coding agent as a subprocess.

    Args:
        workspace: Path to the client repo workspace.
        change_request: Natural language change description.
        change_location: Optional module path hint.
        max_turns: Max conversation turns.
        commit_changes: If True, stage and commit after success.
        program_path: Dotted import path to parent module class.
        model: Bare Anthropic model id (e.g. "claude-opus-4-7") for the SDK
            session. None falls back to the Claude Agent SDK default.
        deny_paths: Path fragments the coding agent must never read, open, or
            edit (the valset and the held-out test set). Enforced by a
            PreToolUse fence inside the agent subprocess.

    Returns:
        Dict with 'success', 'error', and 'output' keys.
    """
    try:
        raw_provider_metadata = {
            "change_location": change_location,
            "commit_changes": commit_changes,
            "program_path": program_path,
        }
        if model:
            raw_provider_metadata["agent_model"] = model
        harness = create_coding_harness(
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
            coding_runner=_run_claude_coding_agent_subprocess,
        )
        harness_result = harness.run_coding(
            HarnessRequest(
                role=AgentRole.CODING,
                workspace=workspace,
                prompt=change_request,
                permissions=HarnessPermissionPolicy.autonomous_policy(),
                max_turns=max_turns,
                raw_provider_metadata=raw_provider_metadata,
            )
        )
    except (NotImplementedError, RuntimeError) as exc:
        return {"success": False, "error": str(exc), "output": ""}

    if (
        harness_result.success
        and commit_changes
        and not harness_result.raw_provider_metadata.get("commit_handled")
    ):
        try:
            committed = git_add_and_commit(workspace, "Code mutation")
        except RuntimeError as exc:
            return {
                "success": False,
                "error": f"Git commit failed: {exc}",
                "output": harness_result.output,
                "session_id": harness_result.session_id,
                "raw_provider_metadata": harness_result.raw_provider_metadata,
            }
        if not committed:
            return {
                "success": True,
                "error": "Agent completed but no file changes were made",
                "output": harness_result.output,
                "session_id": harness_result.session_id,
                "raw_provider_metadata": harness_result.raw_provider_metadata,
                "usage": harness_result.raw_provider_metadata.get("usage") or {},
            }

    return {
        "success": harness_result.success,
        "error": harness_result.error,
        "output": harness_result.output,
        "session_id": harness_result.session_id,
        "raw_provider_metadata": harness_result.raw_provider_metadata,
        "usage": harness_result.raw_provider_metadata.get("usage") or {},
    }


def _run_claude_coding_agent_subprocess(
    workspace: str,
    change_request: str,
    *,
    change_location: str | None = None,
    max_turns: int = 50,
    commit_changes: bool = True,
    program_path: str | None = None,
    model: str | None = None,
    deny_paths: list[str] | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    from src.engine.agent.agent_config import build_agent_config

    config = build_agent_config(
        workspace_path=workspace,
        change_request=change_request,
        change_location=change_location,
        max_turns=max_turns,
        program_path=program_path,
        model=model,
        deny_paths=deny_paths,
        coding_harness=coding_harness,
        coding_harness_model=coding_harness_model,
    )

    config_path = "/tmp/nested_agent_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    print(f"[EXEC] Launching coding agent for: {change_request[:100]}...", flush=True)
    # Use sys.executable (system Python, where claude-agent-sdk is installed)
    # rather than `python` (which would resolve to venv Python via _build_env's
    # PATH prepend — and venv Python does not have the SDK). PATH still points
    # at the venv first, so Bash-tool invocations inside the SDK session still
    # resolve client code installed into /workspace/.venv.
    result = subprocess.run(
        [
            sys.executable,
            _engine_script("src", "engine", "agent", "mounted", "coding_agent.py"),
            "--config", config_path,
        ],
        capture_output=True,
        text=True,
        env=_build_env(workspace),
        **_agent_user_kwargs(),
    )

    print(f"[EXEC] Coding agent exit code: {result.returncode}", flush=True)
    if result.stdout:
        print(f"[EXEC] Coding agent stdout:\n{result.stdout[-2000:]}", flush=True)
    if result.stderr:
        print(f"[EXEC] Coding agent stderr:\n{result.stderr[-1000:]}", flush=True)

    usage = _parse_usage_marker(result.stdout, "AGENT_USAGE:")

    success = "AGENT_SUCCESS" in result.stdout
    error = None
    if result.returncode != 0:
        error_lines = [
            l for l in (result.stdout + "\n" + result.stderr).split("\n")
            if "AGENT_ERROR" in l
        ]
        error = error_lines[0] if error_lines else result.stderr[:500]
        success = False
    elif "AGENT_NO_CHANGES" in result.stdout:
        success = True
        error = "Agent completed but no file changes were made"

    if success and commit_changes:
        commit_result = subprocess.run(
            ["bash", "-c",
             f"cd {workspace} && "
             "git add -A && "
             "git diff --cached --quiet || git commit -m 'Code mutation'"],
            capture_output=True,
            text=True,
        )
        if commit_result.returncode != 0 and "no changes" not in commit_result.stderr:
            return {
                "success": False,
                "error": f"Git commit failed: {commit_result.stderr}",
                "output": result.stdout,
                "usage": usage,
            }

    return {
        "success": success,
        "error": error,
        "output": result.stdout,
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Reflection agent
# ---------------------------------------------------------------------------

def run_inspection_agent(
    workspace: str,
    prompt: str,
    *,
    max_turns: int = 20,
    iteration_num: int | None = None,
    writable: bool = False,
    deny_paths: list[str] | None = None,
    model: str | None = None,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = None,
    run_as_agent_user: bool = True,
    allow_bash: bool = False,
    timeout_seconds: int | None = None,
) -> dict:
    """Spawn the inspection sub-agent as a subprocess.

    A restricted Claude Agent SDK session: Read/Grep/Glob only (+ Write/Edit
    when ``writable=True``, + Bash when ``allow_bash=True``). Used by the
    architect's ``inspect_traces`` tool (read-only) and the valset reflection
    hook (writable memory edits).
    Not a general SDK harness — see ``run_coding_agent`` for the coding path.

    Args:
        workspace: Path to the client repo workspace.
        prompt: Inspection prompt.
        max_turns: Max conversation turns.
        iteration_num: Current iteration number (drives runtime diagnostics).
        writable: If True, the agent gets Write/Edit tools.
        deny_paths: Optional list of path fragments (substrings) that must
            not appear in any tool call. Used by the valset reflection agent
            to block Read/Grep of validation-set files and traces. Matched by
            substring on both absolute and relative path fragments and bash
            commands — this is a policy fence, not cryptographic.
        model: Bare Anthropic model id (e.g. "claude-opus-4-7") for the SDK
            session. None falls back to the Claude Agent SDK default.
        run_as_agent_user: When the Docker runner's CODEEVOLVER_AGENT_USER is
            set, run the agent process as that unprivileged user (the
            default). False is the privileged exception for the run-integrity
            inspector, which must read the root-owned holdout. No effect when
            the env var is unset (local runs).
        allow_bash: Grant the Bash tool. On for the auditors that need more
            than Read/Grep/Glob: the clean-slate inspector (git history —
            `git log` / `branch -a` / `reflog` / `fsck`; it proves the session
            mutated nothing via `clean_slate.py::_snapshot_repo`) and the
            post-run meta-reflection + run-integrity agents (read-only
            inspection after finalization, no guard — nothing they could
            commit changes a recorded sha). Off for the analysis agent and
            `inspect_traces`, which run mid-iteration.
        timeout_seconds: Hard wall-clock ceiling, overriding the turn-based
            timeout when lower. Only the OpenCode harness currently enforces
            this (with a whole-process-group kill — see
            `OpenCodeHarness._run_command`'s `hard_kill`); other harnesses
            ignore it. None (default) leaves the turn-based timeout as-is.

    Returns:
        Dict with 'success', 'proposed_change', and 'error' keys.
    """
    try:
        raw_provider_metadata = {
            "iteration_num": iteration_num,
            "writable": writable,
            "deny_paths": deny_paths,
            "run_as_agent_user": run_as_agent_user,
            "allow_bash": allow_bash,
        }
        if timeout_seconds is not None:
            raw_provider_metadata["timeout_seconds"] = timeout_seconds
        if model:
            raw_provider_metadata["agent_model"] = model
        harness = create_coding_harness(
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
            inspection_runner=_run_claude_inspection_agent_subprocess,
        )
        harness_result = harness.run_inspection(
            HarnessRequest(
                role=AgentRole.INSPECTION,
                workspace=workspace,
                prompt=prompt,
                permissions=HarnessPermissionPolicy.autonomous_policy(),
                max_turns=max_turns,
                raw_provider_metadata=raw_provider_metadata,
            )
        )
    except (NotImplementedError, RuntimeError) as exc:
        return {"success": False, "proposed_change": "", "error": str(exc)}

    return {
        "success": harness_result.success,
        "proposed_change": harness_result.output,
        "error": harness_result.error,
        "output": harness_result.output,
        "session_id": harness_result.session_id,
        "raw_provider_metadata": harness_result.raw_provider_metadata,
        "usage": harness_result.raw_provider_metadata.get("usage") or {},
    }


def _run_claude_inspection_agent_subprocess(
    workspace: str,
    prompt: str,
    *,
    max_turns: int = 20,
    iteration_num: int | None = None,
    writable: bool = False,
    deny_paths: list[str] | None = None,
    model: str | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    run_as_agent_user: bool = True,
    allow_bash: bool = False,
) -> dict:
    config: dict[str, Any] = {
        "workspace_path": workspace,
        "prompt": prompt,
        "max_turns": max_turns,
        "coding_harness": coding_harness,
        "coding_harness_model": coding_harness_model,
    }
    if iteration_num is not None:
        config["iteration_num"] = iteration_num
    if writable:
        config["writable"] = True
    if allow_bash:
        config["allow_bash"] = True
    if deny_paths:
        config["deny_paths"] = list(deny_paths)
    if model:
        config["model"] = model

    config_path = "/tmp/nested_inspection_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f)

    print("[EXEC] Launching inspection agent...", flush=True)
    # See run_coding_agent comment: system Python hosts claude-agent-sdk;
    # PATH still starts with the client venv so Bash-tool shells resolve
    # client-installed binaries.
    result = subprocess.run(
        [
            sys.executable,
            _engine_script("src", "engine", "agent", "mounted", "inspection_agent.py"),
            "--config", config_path,
        ],
        capture_output=True,
        text=True,
        env=_build_env(workspace),
        # run_as_agent_user=False is the run-integrity inspector's privileged
        # exception (it must read the testset).
        **(_agent_user_kwargs() if run_as_agent_user else {}),
    )

    stdout = result.stdout
    stderr = result.stderr

    usage = _parse_usage_marker(stdout, "INSPECT_USAGE:")

    # Read-only path (e.g. inspect_traces): the sub-agent emits
    # `INSPECT_ANALYSIS:{"text": "..."}` on its final turn. Parse that line
    # and return the analysis text as `proposed_change` so the architect tool
    # proxy surfaces it as the `summary` field instead of falling back to a
    # stdout tail full of [TIMER]/[AGENT] plumbing.
    for line in stdout.split("\n"):
        if line.startswith("INSPECT_ANALYSIS:"):
            try:
                data = json.loads(line[len("INSPECT_ANALYSIS:"):])
            except json.JSONDecodeError:
                continue
            text = data.get("text", "") if isinstance(data, dict) else ""
            if text:
                return {
                    "success": True,
                    "proposed_change": text,
                    "error": None,
                    "usage": usage,
                }

    if writable:
        success = "INSPECT_WRITE_SUCCESS" in stdout
        error = None
        if not success:
            error_lines = [l for l in stdout.split("\n") if "INSPECT_ERROR" in l]
            error = error_lines[0] if error_lines else "Writable inspection produced no writes"
        return {
            "success": success,
            "proposed_change": "Memory updated" if success else "No change proposed",
            "error": error,
            "usage": usage,
        }

    return {
        "success": result.returncode == 0,
        "proposed_change": stdout[-4000:] if stdout else "",
        "error": None if result.returncode == 0 else (stderr[:500] or "agent failed"),
        "usage": usage,
    }


# ---------------------------------------------------------------------------
# Bash and file helpers
# ---------------------------------------------------------------------------

def exec_bash(workspace: str, command: str) -> dict:
    """Execute a bash command in the workspace context.

    Returns dict with 'stdout', 'stderr', 'returncode'.
    """
    result = subprocess.run(
        ["bash", "-c", f"cd {workspace} && {command}"],
        capture_output=True,
        text=True,
    )
    return {
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
    }


def write_file(path: str, content: str) -> None:
    """Write content to a file, creating parent directories as needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def read_file(path: str) -> str | None:
    """Read a file, returning None if it doesn't exist."""
    try:
        with open(path, "r") as f:
            return f.read()
    except FileNotFoundError:
        return None


__all__ = [
    "exec_bash",
    "git_add_and_commit",
    "git_checkout",
    "git_commit_tree",
    "git_commit_trees",
    "git_create_branch",
    "git_create_branch_simple",
    "git_delete_ref",
    "git_diff_tree_names",
    "git_capture",
    "git_dirty_paths",
    "git_get_sha",
    "git_is_ancestor",
    "git_is_dirty",
    "git_list_branches",
    "git_set_identity",
    "git_tree_signature",
    "git_update_ref",
    "read_file",
    "run_coding_agent",
    "run_evaluation",
    "run_inspection_agent",
    "write_file",
]
