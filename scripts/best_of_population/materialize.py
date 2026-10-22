"""Copyright © 2026 440 Labs LLC

Git materialization: bare repo per benchmark, anchor refspecs, worktree pool.

The analysis owns `<out>/repos/<bench>.git` rather than reusing
`~/ce-runs/repos/<bench>.git`: that repo's refs are keyed by a reusable
`run_id` and may not hold `refs/codeevolver/*`. Each run's
`repo.bundle` is fetched under `refs/jobs/<job_id>/{heads,codeevolver}/*`.
Candidates are checked out DETACHED into a small pool of worktrees (one per
worker), the way `finalization._eval_branch` checks out a sha; untracked
staged files — the held-out testset (amended out of every bundle), gitignored
benchmark assets, the `env` venv symlink — survive `git checkout`.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.engine.services.subprocess_service import clean_python_bytecode

from .config import AnalysisConfig, BenchSpec, analysis_paths
from .environment import link_venv


def _git(repo: Path | str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    res = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if check and res.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {res.stderr.strip()[:400]}")
    return res


def ensure_bare_repo(path: Path) -> Path:
    if not (path / "HEAD").exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "init", "--bare", "-q", str(path)], check=True)
    return path


def job_refspecs(job_id: str) -> list[str]:
    return [
        f"refs/heads/*:refs/jobs/{job_id}/heads/*",
        f"refs/codeevolver/*:refs/jobs/{job_id}/codeevolver/*",
    ]


def fetch_run(bare: Path, bundle: Path, job_id: str) -> int:
    """Fetch one run's bundle; returns the number of refs now under the job."""
    _git(bare, "fetch", "--force", "-q", str(bundle), *job_refspecs(job_id))
    refs = _git(bare, "for-each-ref", "--format=%(refname)", f"refs/jobs/{job_id}").stdout.split()
    return len(refs)


def anchor_ref(job_id: str, pointer: str) -> str:
    """Map a record's `refs/codeevolver/...` pointer (or a sha) to the
    fetched ref name; shas pass through."""
    if pointer.startswith("refs/codeevolver/"):
        return f"refs/jobs/{job_id}/" + pointer[len("refs/"):]
    return pointer


def resolve_sha(bare: Path, ref: str) -> str:
    return _git(bare, "rev-parse", "--verify", f"{ref}^{{commit}}").stdout.strip()


def resolve_tree(bare: Path, ref: str) -> str:
    return _git(bare, "rev-parse", "--verify", f"{ref}^{{tree}}").stdout.strip()


def file_at(bare: Path, ref: str, path: str) -> str | None:
    res = _git(bare, "show", f"{ref}:{path}", check=False)
    return res.stdout if res.returncode == 0 else None


def blob_sha256(bare: Path, ref: str, path: str) -> str | None:
    res = subprocess.run(["git", "-C", str(bare), "show", f"{ref}:{path}"], capture_output=True)
    if res.returncode != 0:
        return None
    return hashlib.sha256(res.stdout).hexdigest()


def eval_anchor_refs(bare: Path, job_id: str) -> list[tuple[str, str]]:
    """`(refname, sha)` for every `refs/codeevolver/evals/<N>/<i>` of a job."""
    out = _git(bare, "for-each-ref", "--format=%(refname) %(objectname)",
               f"refs/jobs/{job_id}/codeevolver/evals").stdout.split("\n")
    return [tuple(line.split()) for line in out if line.strip()]  # type: ignore[misc]


def sha256_file(path: Path) -> str | None:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class Slot:
    index: int
    path: Path
    sha: str | None = None
    venv: Path | None = None


class WorktreePool:
    """`n` detached worktrees off the bare repo. `checkout(k, sha)` moves
    slot k; staged untracked files stay put."""

    def __init__(self, bare: Path, root: Path, spec: BenchSpec, n: int) -> None:
        self.bare, self.root, self.spec = bare, root, spec
        self.slots = [Slot(i, root / f"w{i}") for i in range(max(1, n))]

    def ensure(self, initial_sha: str) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for slot in self.slots:
            if not (slot.path / ".git").exists():
                if slot.path.exists():
                    shutil.rmtree(slot.path)
                _git(self.bare, "worktree", "prune")
                _git(self.bare, "worktree", "add", "--detach", "-q", str(slot.path), initial_sha)
            self.stage_untracked(slot)

    def stage_untracked(self, slot: Slot) -> None:
        """Held-out testset + gitignored assets from the researcher's checkout."""
        repo = Path(self.spec.repo_path).expanduser()
        if self.spec.testset_path:
            src = repo / self.spec.testset_path
            dst = slot.path / self.spec.testset_path
            if not src.exists():
                raise FileNotFoundError(
                    f"held-out testset {src} not found in the client checkout "
                    f"({self.spec.repo_path}); it is amended out of every run bundle"
                )
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or sha256_file(dst) != sha256_file(src):
                shutil.copyfile(src, dst)
        for rel in self.spec.assets:
            src = repo / rel
            dst = slot.path / rel
            if not src.exists():
                raise FileNotFoundError(f"benchmark asset {src} missing in the client checkout")
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.is_symlink() or dst.exists():
                continue
            dst.symlink_to(src)

    def checkout(self, slot: Slot, sha: str, venv: Path | None = None) -> None:
        if venv is not None and slot.venv != venv:
            link_venv(slot.path, venv)
            slot.venv = venv
        if slot.sha == sha:
            return
        try:
            clean_python_bytecode(str(slot.path))
        except Exception:  # noqa: BLE001
            pass
        _git(slot.path, "checkout", "--detach", "--force", "-q", sha)
        slot.sha = sha
        self.stage_untracked(slot)

    def testset_sha256(self, slot: Slot) -> str | None:
        return sha256_file(slot.path / self.spec.testset_path) if self.spec.testset_path else None

    def trainset_sha256(self, slot: Slot) -> str | None:
        return sha256_file(slot.path / self.spec.trainset_path)

    def teardown(self) -> None:
        for slot in self.slots:
            if slot.path.exists():
                _git(self.bare, "worktree", "remove", "--force", str(slot.path), check=False)
                shutil.rmtree(slot.path, ignore_errors=True)
        _git(self.bare, "worktree", "prune", check=False)


def load_materialized(cfg: AnalysisConfig, bench: str) -> dict[str, Any]:
    path = analysis_paths(cfg, bench)["materialize"]
    if not path.exists():
        raise FileNotFoundError(f"{path} not found — run `materialize --bench {bench}` first")
    return json.loads(path.read_text())


__all__ = [
    "WorktreePool", "anchor_ref", "blob_sha256", "ensure_bare_repo", "eval_anchor_refs", "fetch_run",
    "file_at", "job_refspecs", "load_materialized", "resolve_sha", "resolve_tree", "sha256_file",
]
