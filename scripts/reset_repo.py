"""Save research results to the server and reset the local experiment workspace.

After a client repo is optimized locally, the workspace accumulates artifacts
that would contaminate the *next* experiment: many `codeevolver-*` git branches
(good programs that bias idea generation), local memory files, and local
iteration history. Before a fresh run we want a clean slate — but we first
**save** the valuable results to a durable server so nothing is lost, then
**delete** the local contaminants.

Local research runs (`APP_MODE=local`) write job/repo records to a fragile
single-node local Community Mongo (`mongodb://localhost:27017`, db
`codeevolver_dev`) and mirror every iteration to
`~/.codeevolver/history/{job_id}.jsonl` (+ `{job_id}.final.json`) as a
crash-proof backup. The "server" we save to is the remote dev Atlas
(`codeevolver_dev`) — the durable record — plus pushing the best program
branches to the client repo's `origin`.

Two independent, combinable flags (no flags → does nothing):

    -S / --save      Push results to the server (history upload + top-3 branches).
    -D / --delete    Clear local contaminants (codeevolver-* branches + files,
                     the `origin` remote, and any loose files left at the root
                     of the memory folder — see "Memory root" below).

**Memory root.** Per-job artifacts live under `<traces|memory|runtime_tests|
architect_traces>/{job_id}` and are removed per job. But the architect sometimes writes memory
files (e.g. `memory.md`, `iteration_change_summaries.md`) directly to the *root* of the
memory folder when no `CODEEVOLVER_JOB_ID` is set — these are not job-scoped, so
the per-job sweep never catches them and they contaminate the next run. A full
`-D` reset (no `--job`) also clears those loose root files.

When both are given, the script saves first, then deletes.

**Origin removal (contamination guard).** `-D` also removes the `origin` remote
from each workspace. In local research the agent runs in-place with a Bash tool,
so a configured `origin` lets it `git fetch`/`git ls-remote` the very
`codeevolver-*` branches prior `-S` runs pushed to GitHub — cross-run
contamination. Deleting only the *local* branches doesn't close that hole; the
remote does. Because `-S` itself needs `origin` to push saves, `-D` first stashes
the URL into `git config codeevolver.origin-url` before removing the remote, and
`push_top_branches` falls back to that stashed URL when `origin` is absent — so
saves keep working across cycles. Restore manually any time with:
`git remote add origin "$(git config codeevolver.origin-url)"`.

Saving history goes through the dev API only. Each job's history is POSTed to
the dev API's `/import-job-history` endpoint, authenticated with a client API
key (`CODEEVOLVER_API_KEY`, falling back to `TEST_API_KEY`). Researchers use
this — no Mongo key needed; the dev API writes to the cloud DB on their behalf,
under whatever account owns the API key. The legacy direct-Mongo-write path has
been removed; `--via mongo` and `--via api` are both accepted and both save via
the API. Saving always targets dev — there is no path to prod.

The top-3 program branches per job are pushed to the repo's `origin`.

Usage:
    APP_MODE=local CODEEVOLVER_API_KEY=... python scripts/reset_repo.py -S
    APP_MODE=local CODEEVOLVER_API_KEY=... python scripts/reset_repo.py -S -D
    APP_MODE=local TEST_API_KEY=... python scripts/reset_repo.py -S
    APP_MODE=local python scripts/reset_repo.py -D
    python scripts/reset_repo.py -S -D --dry-run
    python scripts/reset_repo.py -S --job <job_id>
    python scripts/reset_repo.py -S -D --exclude <job_id> <job_id>
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

# Reset scripts target the local research deploy: default to local mode so
# `settings` resolves the localhost Mongo and the ~/.codeevolver state tree.
os.environ.setdefault("APP_MODE", "local")
os.environ.setdefault("SANDBOX_MODE", "local")
# We enumerate every job, so the per-job subdir suffix in paths.py must be off
# (it would otherwise pin traces/memory dirs to a single job).
os.environ.pop("CODEEVOLVER_JOB_ID", None)

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.export_history import reconstruct
from scripts.utils import normalize_iteration_history
from src.config import API_URLS, settings
from src.engine.paths import (
    architect_traces_dir,
    memory_dir,
    runtime_tests_dir,
    traces_dir,
)

DEV_DB_NAME = "codeevolver_dev"
BRANCH_PREFIX = "codeevolver-"
TOP_N = 3
# git config key where `-D` stashes the origin URL before removing the remote,
# so `-S` can still push saves after the contamination guard has run.
STASHED_ORIGIN_KEY = "codeevolver.origin-url"


# ---------------------------------------------------------------------------
# Mongo connections
# ---------------------------------------------------------------------------

def get_local_db():
    """Connect to the local Community Mongo the research run wrote to."""
    from pymongo import MongoClient

    uri = os.getenv("LOCAL_MONGO_URI", "mongodb://localhost:27017")
    return MongoClient(uri)[DEV_DB_NAME]


# ---------------------------------------------------------------------------
# Job enumeration & repo resolution
# ---------------------------------------------------------------------------

def enumerate_save_jobs(
    history_root: Path, only_job: str | None, exclude: set[str]
) -> list[str]:
    """Jobs to SAVE: the `{job_id}.jsonl` stems in the local history mirror.

    The history folder is the save manifest. To exclude a partial job from a
    save, just delete its `{job_id}.jsonl` file — it won't be enumerated here.
    `exclude` drops named jobs without touching their files.
    """
    if only_job:
        return [] if only_job in exclude else [only_job]
    if not history_root.exists():
        return []
    return sorted(
        p.stem
        for p in history_root.glob("*.jsonl")
        if not p.name.startswith(".") and p.stem not in exclude
    )


def discover_local_jobs(
    history_root: Path, only_job: str | None, exclude: set[str]
) -> list[str]:
    """Jobs to DELETE: the union of job ids found across every local artifact.

    Unlike the save manifest, deletion is a full sweep — it does not depend on
    the history file existing. This means removing a partial job's history file
    (to drop it from a save) does not also strand its traces/memory/etc.; a
    later `-D` still cleans them up. `exclude` keeps named jobs out of the
    sweep entirely (artifacts kept, branches untouched).
    """
    if only_job:
        return [] if only_job in exclude else [only_job]

    job_ids: set[str] = set()

    # JSONL + final snapshots in the history dir.
    if history_root.exists():
        for p in history_root.glob("*.jsonl"):
            if not p.name.startswith("."):
                job_ids.add(p.stem)
        for p in history_root.glob("*.final.json"):
            if not p.name.startswith("."):
                job_ids.add(p.name[: -len(".final.json")])

    # Per-job subdirectories under traces / memory / runtime_tests / architect_traces.
    for root in (traces_dir(), memory_dir(), runtime_tests_dir(), architect_traces_dir()):
        root_path = Path(root)
        if root_path.exists():
            for child in root_path.iterdir():
                if child.is_dir() and not child.name.startswith("."):
                    job_ids.add(child.name)

    # Per-job log files.
    if settings.local_log_root.exists():
        for p in settings.local_log_root.glob("*.log"):
            if not p.name.startswith("."):
                job_ids.add(p.stem)

    return sorted(job_ids - exclude)


def resolve_workspace_path(local_db, job_id: str) -> str | None:
    """job -> repo_url -> repos.workspace_path (mirrors local_launcher)."""
    job = local_db.jobs.find_one({"job_id": job_id})
    if not job:
        print(f"  [warn] no local job record for {job_id}; git steps skipped.")
        return None
    repo_url = job.get("repo_url")
    if not repo_url:
        print(f"  [warn] job {job_id} has no repo_url; git steps skipped.")
        return None
    repo = local_db.repos.find_one({"repo_url": repo_url})
    if not repo or not repo.get("workspace_path"):
        print(
            f"  [warn] no repo/workspace_path for repo_url={repo_url!r} "
            f"(job {job_id}); git steps skipped."
        )
        return None
    return repo["workspace_path"]


# ---------------------------------------------------------------------------
# Save path
# ---------------------------------------------------------------------------

def _top_branches(iteration_history: dict) -> list[str]:
    """Top-N branches ranked strictly by each record's `score` key.

    We rank only on `score` — the legacy mirror of `valset_score`. Records
    without a `score` are not ranked (a rejected/skipped iteration has no
    valset aggregate to rank on).
    """
    scored = [
        rec
        for rec in iteration_history.values()
        if isinstance(rec, dict)
        and rec.get("git_branch")
        and rec.get("score") is not None
    ]

    scored.sort(key=lambda rec: float(rec["score"]), reverse=True)

    branches: list[str] = []
    for rec in scored:
        branch = rec["git_branch"]
        if branch not in branches:
            branches.append(branch)
        if len(branches) >= TOP_N:
            break
    return branches


def save_via_api(
    api_url: str, api_key: str, job_id: str, blob: dict, repo_url: str | None, dry_run: bool
) -> None:
    """POST the reconstructed history to the dev API's /import-job-history.

    Uses the client API key — the dev API writes to the cloud DB, so the
    caller never needs the Mongo credentials.
    """
    import httpx

    iteration_history = blob.get("iteration_history", {}) or {}
    summary = blob.get("summary", {}) or {}

    # Rewrite local repo paths in the history to GitHub links, and backfill a
    # missing repo_url from those paths — the portal builds branch links as
    # `{repo_url}/tree/{git_branch}` and locally-mirrored jobs often lack it.
    iteration_history, detected_repo_url = normalize_iteration_history(iteration_history)
    if not repo_url and detected_repo_url:
        repo_url = detected_repo_url
        print(f"  [backfill] set missing repo_url -> {repo_url}")

    n_iters = len(iteration_history)
    endpoint = f"{api_url}/import-job-history"

    if dry_run:
        print(f"  [dry-run] would POST {n_iters} iteration record(s) for {job_id} to {endpoint}")
        return

    status = blob.get("status")
    payload = {
        "job_id": job_id,
        "iteration_history": iteration_history,
        "status": status if status and status != "unknown" else None,
        "repo_url": repo_url,
        "best_candidate": summary.get("best_candidate"),
        "best_score": summary.get("best_score"),
        "num_candidates": summary.get("num_candidates"),
    }
    headers = {"Authorization": f"Bearer {api_key}", "Authorization-Type": "api_key"}
    try:
        resp = httpx.post(endpoint, json=payload, headers=headers, timeout=120.0)
    except httpx.RequestError as exc:
        print(f"  [warn] failed to POST history for {job_id}: {exc}")
        return
    if resp.status_code == 200:
        data = resp.json()
        print(
            f"  imported {job_id} via API "
            f"({data.get('total_iterations')} iters, created={data.get('created')})"
        )
    else:
        print(f"  [warn] import failed for {job_id}: HTTP {resp.status_code} {resp.text[:200]}")


def _git_config_get(workspace_path: str, key: str) -> str | None:
    """Read a single git config value, or None when unset."""
    result = subprocess.run(
        ["git", "config", "--get", key],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _resolve_push_target(workspace_path: str) -> str | None:
    """Return the push target for saves: the `origin` remote if configured,
    else the URL stashed by a prior `-D` (`codeevolver.origin-url`).

    `-D` removes `origin` as a contamination guard, so on every cycle after the
    first the remote is gone and we push to the stashed URL directly
    (`git push <url> <branch>` needs no named remote). Returns None when neither
    is available.
    """
    if _git_config_get(workspace_path, "remote.origin.url"):
        return "origin"
    return _git_config_get(workspace_path, STASHED_ORIGIN_KEY)


def push_top_branches(workspace_path: str, branches: list[str], dry_run: bool) -> None:
    if not branches:
        print("  no admitted candidate branches to push.")
        return
    target = _resolve_push_target(workspace_path)
    if target is None:
        print(
            "  [warn] no origin remote and no stashed "
            f"{STASHED_ORIGIN_KEY}; cannot push saves. "
            "Re-add origin to enable saving."
        )
        return
    shown = "origin" if target == "origin" else f"{target} (stashed)"
    for branch in branches:
        if dry_run:
            print(f"  [dry-run] would: git push {shown} {branch}  (cwd={workspace_path})")
            continue
        result = subprocess.run(
            ["git", "push", target, branch],
            cwd=workspace_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  pushed {branch}")
        else:
            print(
                f"  [warn] failed to push {branch}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )


def do_save(args, local_db) -> None:
    print("\n=== SAVE ===")

    # Saving always goes through the dev API. `--via mongo`/`--via api` are both
    # accepted (back-compat) and both take this path; the direct-Mongo write is
    # gone. The API key resolves the owning account on the dev side.
    api_url = (
        args.api_url or os.getenv("CODEEVOLVER_API_URL") or API_URLS["dev"]
    ).rstrip("/")
    api_key = args.api_key or os.getenv("CODEEVOLVER_API_KEY") or os.getenv("TEST_API_KEY")
    if not api_key and not args.dry_run:
        print(
            "ERROR: saving needs a client API key. "
            "Set CODEEVOLVER_API_KEY (or TEST_API_KEY) or pass --api-key."
        )
        sys.exit(1)
    print(f"  target: {api_url}/import-job-history (client API key)")

    history_root: Path = args.history_root
    job_ids = enumerate_save_jobs(history_root, args.job, set(args.exclude))
    if not job_ids:
        print(f"No jobs found under {history_root}.")
        return

    for job_id in job_ids:
        print(f"\njob {job_id}:")
        blob = reconstruct(job_id, history_root)
        job = local_db.jobs.find_one({"job_id": job_id})
        repo_url = job.get("repo_url") if job else None

        save_via_api(api_url, api_key, job_id, blob, repo_url, args.dry_run)

        workspace_path = resolve_workspace_path(local_db, job_id)
        if workspace_path:
            branches = _top_branches(blob.get("iteration_history", {}) or {})
            print(f"  top {len(branches)} branch(es): {branches or '(none)'}")
            push_top_branches(workspace_path, branches, args.dry_run)


# ---------------------------------------------------------------------------
# Delete path
# ---------------------------------------------------------------------------

def _current_branch(workspace_path: str) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def delete_local_branches(workspace_path: str, dry_run: bool) -> None:
    result = subprocess.run(
        ["git", "branch", "--list", f"{BRANCH_PREFIX}*"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(
            f"  [warn] could not list branches in {workspace_path}: "
            f"{result.stderr.strip()}"
        )
        return
    branches = [
        line.lstrip("* ").strip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    branches = [b for b in branches if b.startswith(BRANCH_PREFIX)]
    if not branches:
        print(f"  no {BRANCH_PREFIX}* branches in {workspace_path}.")
        return

    current = _current_branch(workspace_path)
    for branch in branches:
        if branch == current:
            print(f"  [skip] {branch} is the current branch — checkout another first.")
            continue
        if dry_run:
            print(f"  [dry-run] would: git branch -D {branch}  (cwd={workspace_path})")
            continue
        result = subprocess.run(
            ["git", "branch", "-D", branch],
            cwd=workspace_path,
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print(f"  deleted local branch {branch}")
        else:
            print(f"  [warn] failed to delete {branch}: {result.stderr.strip()}")


def remove_origin(workspace_path: str, dry_run: bool) -> None:
    """Remove the `origin` remote, stashing its URL first.

    Contamination guard: with no `origin`, the in-place research agent can no
    longer `git fetch`/`git ls-remote` the `codeevolver-*` branches that prior
    `-S` runs pushed to GitHub. The URL is saved to `git config
    codeevolver.origin-url` so `-S` can keep pushing saves (see
    `_resolve_push_target`) and the operator can restore the remote manually.
    """
    url = _git_config_get(workspace_path, "remote.origin.url")
    if url is None:
        print("  no origin remote to remove.")
        return
    if dry_run:
        print(
            f"  [dry-run] would: stash origin url ({url}) to {STASHED_ORIGIN_KEY} "
            f"and git remote remove origin  (cwd={workspace_path})"
        )
        return
    stash = subprocess.run(
        ["git", "config", STASHED_ORIGIN_KEY, url],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if stash.returncode != 0:
        print(
            f"  [warn] failed to stash origin url ({stash.stderr.strip()}); "
            "leaving origin in place."
        )
        return
    result = subprocess.run(
        ["git", "remote", "remove", "origin"],
        cwd=workspace_path,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print(f"  removed origin remote (stashed url as {STASHED_ORIGIN_KEY})")
    else:
        print(f"  [warn] failed to remove origin: {result.stderr.strip()}")


def delete_job_artifacts(job_id: str, dry_run: bool) -> None:
    """Remove every local per-job artifact for `job_id`."""
    targets: list[Path] = [
        Path(traces_dir()) / job_id,
        Path(memory_dir()) / job_id,
        Path(runtime_tests_dir()) / job_id,
        Path(architect_traces_dir()) / job_id,
        settings.local_history_root / f"{job_id}.jsonl",
        settings.local_history_root / f"{job_id}.final.json",
        settings.local_log_root / f"{job_id}.log",
        settings.local_log_root / f"{job_id}.pid",
    ]
    for target in targets:
        if not target.exists():
            continue
        if dry_run:
            print(f"  [dry-run] would remove {target}")
            continue
        if target.is_dir():
            shutil.rmtree(target)
        else:
            target.unlink()
        print(f"  removed {target}")


def clear_memory_root_files(dry_run: bool) -> None:
    """Remove loose files left at the *root* of the memory folder.

    Per-job memory lives under ``memory_dir()/{job_id}`` and is removed by
    ``delete_job_artifacts``. But the architect sometimes writes memory files
    (e.g. ``memory.md``, ``iteration_change_summaries.md``) directly to the memory
    root when no ``CODEEVOLVER_JOB_ID`` is set — these are not job-scoped, so
    no per-job sweep catches them, and they contaminate the next run's memory.
    A full ``-D`` reset must clear them too.

    Only loose files are removed; per-job subdirectories are left to
    ``delete_job_artifacts`` (so an ``--exclude``'d job's dir survives). Hidden
    files (``.DS_Store``) are skipped, matching the rest of the sweep.
    """
    root = Path(memory_dir())
    if not root.exists():
        return
    loose = [p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")]
    if not loose:
        return
    print(f"\nmemory root {root}:")
    for p in loose:
        if dry_run:
            print(f"  [dry-run] would remove {p}")
            continue
        p.unlink()
        print(f"  removed {p}")


def do_delete(args, local_db) -> None:
    print("\n=== DELETE ===")
    job_ids = discover_local_jobs(args.history_root, args.job, set(args.exclude))
    if not job_ids:
        print("No local job artifacts found.")
        # Loose memory-root files outlive every job (they're written without a
        # CODEEVOLVER_JOB_ID), so a full reset still has work to do here even
        # when no job-scoped artifacts remain.
        if not args.job:
            clear_memory_root_files(args.dry_run)
        return

    # Resolve unique workspaces up front.
    workspaces: list[str] = []
    for job_id in job_ids:
        workspace_path = resolve_workspace_path(local_db, job_id)
        if workspace_path and workspace_path not in workspaces:
            workspaces.append(workspace_path)

    # Guard: refuse to delete anything if a workspace is currently checked out
    # on a codeevolver-* branch — git can't delete the current branch, and the
    # operator would be left on a branch this script is meant to remove.
    for workspace_path in workspaces:
        current = _current_branch(workspace_path)
        if current and current.startswith(BRANCH_PREFIX):
            print(
                f"ERROR: {workspace_path} is on branch '{current}', which has the "
                f"'{BRANCH_PREFIX}' prefix. Please check out a branch that does not "
                f"include the prefix, then re-run."
            )
            sys.exit(1)

    # Delete codeevolver-* branches and remove origin once per unique workspace.
    for workspace_path in workspaces:
        print(f"\nworkspace {workspace_path}:")
        delete_local_branches(workspace_path, args.dry_run)
        remove_origin(workspace_path, args.dry_run)

    # Delete local file storage per job.
    for job_id in job_ids:
        print(f"\njob {job_id} artifacts:")
        delete_job_artifacts(job_id, args.dry_run)

    # Clear non-job-scoped files left at the memory root. Skipped for a
    # targeted `--job` run, where these shared files aren't attributable to
    # the single job being reset and should be left alone.
    if not args.job:
        clear_memory_root_files(args.dry_run)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Save research results to the server and/or reset the local workspace.",
    )
    parser.add_argument("-S", "--save", action="store_true", help="Push results to the server.")
    parser.add_argument("-D", "--delete", action="store_true", help="Clear local contaminants.")
    parser.add_argument(
        "--via",
        choices=["api", "mongo"],
        default="api",
        help="Back-compat only. Both 'api' and 'mongo' now save via the dev API "
        "(/import-job-history) using the client API key; the direct Atlas-write "
        "path has been removed.",
    )
    parser.add_argument(
        "--api-url",
        default=None,
        help="Base API URL for saving (default: $CODEEVOLVER_API_URL or the dev cloud API).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Client API key for saving (default: $CODEEVOLVER_API_KEY, then $TEST_API_KEY).",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print actions without performing them."
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=settings.local_history_root,
        help="Directory holding {job_id}.jsonl (default: ~/.codeevolver/history).",
    )
    parser.add_argument("--job", default=None, help="Limit to a single job_id.")
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=[],
        metavar="JOB_ID",
        help="Job id(s) to exclude from both save and delete (space-separated).",
    )
    args = parser.parse_args()

    if not args.save and not args.delete:
        print("Nothing to do: pass -S to save and/or -D to delete.\n")
        parser.print_help()
        return 0

    local_db = get_local_db()

    if args.save:
        do_save(args, local_db)
    if args.delete:
        do_delete(args, local_db)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
