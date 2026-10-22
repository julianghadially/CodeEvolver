#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Pull one run's artifacts back to the controller computer, into two places.

1. `~/.codeevolver/{memory,architect_traces,history,logs,runtime_tests}/{job_id}/`
   Merged straight into the existing local-deployment tree. `paths.py` already
   scopes every one of those directories by `{job_id}`, so merging 36 runs is
   collision-free by construction and the existing tools --
   `scripts/export_history.py`, `query_iteration_history.py`, `view.py` --
   work on fleet runs with no changes.

2. `$CE_RUNS_DIR/<run_id>/` (default `~/ce-runs`) — the run-level record (status, settings, egress
   report, squid log, candidates.jsonl, repo.bundle, and `workspace.tar.gz`:
   the final program's whole working tree, gitignored files included, which is
   the only copy of anything the run wrote outside git).

The bundle is then fetched into a per-benchmark bare repo under
`refs/runs/<run_id>/*`. Every run of a benchmark descends from the same squashed
seed commit, so git dedupes objects across all twelve runs instead of storing
twelve copies of the history -- and the later per-candidate analysis can use
`git worktree add` to check out many candidates at once rather than serializing
on branch switches in a single clone.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

CE_HOME = Path.home() / ".codeevolver"
RUNS_HOME = Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()
REPOS_HOME = RUNS_HOME / "repos"

# Scoped-by-job_id trees that merge safely into the operator's own ~/.codeevolver.
#
# `traces` is deliberately absent. It holds a record of every evaluated ROW --
# each LM input and output, for up to 25,000 metric calls on ragqa -- and is by
# far the largest artifact class the fleet produces. Nothing downstream reads
# it: the `inspect_traces` tool and meta-reflection consume it INSIDE the
# container, and the only controller-side reader of `traces_dir()` is
# scripts/reset_repo.py, a local-workspace cleaner. `architect_traces` -- the
# per-session record of what the architect actually did -- is kept.
JOB_TREES = ("memory", "architect_traces", "history", "logs", "runtime_tests")

# Caches the harnesses drop under HOME. Never worth 100s of MB of transfer --
# except opencode's log, which is the ground truth for MCP server health.
#
# `.codeevolver/traces` is excluded at the RSYNC layer too, not just left out of
# JOB_TREES above: dropping it from the merge alone would still pay to transfer
# it. Consequence to know -- hosts are wiped by `ce recycle` and droplets are
# destroyed, so per-row traces are gone permanently once a run is collected.
RSYNC_EXCLUDES = (
    ".cache", ".npm", ".local/share/opencode/storage", "node_modules",
    ".mongo", "mongod.log", ".codeevolver/traces",
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def rsync(remote: str, dest: Path) -> bool:
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["rsync", "-az", "--partial", "--timeout=120"]
    for pat in RSYNC_EXCLUDES:
        cmd += ["--exclude", pat]
    cmd += [remote, str(dest)]
    res = run(cmd)
    if res.returncode != 0:
        print(f"rsync failed: {res.stderr.strip()[:300]}", file=sys.stderr)
    return res.returncode == 0


def merge_job_trees(staged: Path) -> str | None:
    """Move the job-scoped subtrees into ~/.codeevolver, returning the job_id."""
    src_root = staged / ".codeevolver"
    if not src_root.is_dir():
        return None

    job_id = None
    for tree in JOB_TREES:
        src = src_root / tree
        if not src.is_dir():
            continue
        dest = CE_HOME / tree
        dest.mkdir(parents=True, exist_ok=True)
        for child in src.iterdir():
            # history/logs are file-per-job ({job_id}.jsonl, {job_id}.log);
            # the rest are directory-per-job.
            job_id = job_id or (child.name if child.is_dir() else child.name.split(".")[0])
            run(["rsync", "-a", f"{child}/" if child.is_dir() else str(child),
                 str(dest / child.name) + ("/" if child.is_dir() else "")])
    return job_id


def bundle_refspecs(run_id: str, job_id: str | None) -> list[str]:
    """Refspecs that fold one run's bundle into the shared bare repo.

    `refs/runs/<run_id>/*` holds the branch heads for the tools that read
    them. The bundle is also fetched under
    `refs/jobs/<job_id>/{heads,codeevolver}/*`: `job_id` is unique where
    `run_id` is reused across attempts (a re-dispatched run would otherwise
    overwrite the previous attempt's refs), and `refs/codeevolver/*` holds
    the per-iteration close anchors and the wrapped evaluated trees that the
    offline best-of-population audit checks out.
    """
    specs = [f"refs/heads/*:refs/runs/{run_id}/*"]
    if job_id:
        specs += [
            f"refs/heads/*:refs/jobs/{job_id}/heads/*",
            f"refs/codeevolver/*:refs/jobs/{job_id}/codeevolver/*",
        ]
    return specs


def read_job_id(staged: Path) -> str | None:
    status = staged / "status.json"
    if not status.exists():
        return None
    try:
        return json.loads(status.read_text()).get("job_id") or None
    except (json.JSONDecodeError, OSError):
        return None


def fetch_bundle(
    bench: str, run_id: str, bundle: Path, job_id: str | None = None
) -> None:
    """Fold a run's branches AND its `refs/codeevolver/*` anchors into the
    benchmark's shared bare repo (see `bundle_refspecs`)."""
    if not bundle.exists():
        return
    repo = REPOS_HOME / f"{bench}.git"
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "init", "--bare", str(repo)], check=True)

    res = run(["git", "-C", str(repo), "fetch", "--force", str(bundle),
               *bundle_refspecs(run_id, job_id)])
    if res.returncode != 0:
        print(f"bundle fetch failed: {res.stderr.strip()[:300]}", file=sys.stderr)
        return
    refs = run(["git", "-C", str(repo), "for-each-ref", "--format=%(refname)",
                f"refs/runs/{run_id}"]).stdout.split()
    job_refs = (
        run(["git", "-C", str(repo), "for-each-ref", "--format=%(refname)",
             f"refs/jobs/{job_id}"]).stdout.split()
        if job_id else []
    )
    size = run(["du", "-sh", str(repo)]).stdout.split("\t")[0]
    print(f"  bundle -> {repo.name}: {len(refs)} refs under refs/runs/{run_id}/, "
          f"{len(job_refs)} under refs/jobs/{job_id or '?'}/ (repo now {size})")


def main() -> int:
    ap = argparse.ArgumentParser(description="Collect one run's artifacts")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--bench", required=True)
    ap.add_argument("--remote", required=True, help="user@host:/data/runs/<run_id>/")
    ap.add_argument("--final", action="store_true",
                    help="also merge job trees and fetch the repo bundle")
    args = ap.parse_args()

    staged = RUNS_HOME / args.run_id
    if not rsync(args.remote, staged):
        return 1

    if not args.final:
        return 0

    job_id = merge_job_trees(staged)
    if job_id:
        print(f"  merged job trees into {CE_HOME} (job_id={job_id})")

    fetch_bundle(args.bench, args.run_id, staged / "repo.bundle",
                 job_id=job_id or read_job_id(staged))

    status = staged / "status.json"
    if status.exists():
        try:
            blob = json.loads(status.read_text())
            print(f"  {args.run_id}: {blob.get('status')} "
                  f"final_test={blob.get('final_test_score')} "
                  f"{blob.get('final_test_scores')} "
                  f"seed_test={blob.get('seed_test_score')} "
                  f"{blob.get('seed_test_scores')} "
                  f"iters={blob.get('current_iteration')}")
        except json.JSONDecodeError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
