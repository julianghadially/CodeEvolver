"""Unit tests for scripts/best_of_population/materialize.py against a real
tiny git repo with a codeevolver eval anchor, bundled with --all."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts.best_of_population.config import BenchSpec
from scripts.best_of_population.materialize import (
    WorktreePool,
    anchor_ref,
    ensure_bare_repo,
    eval_anchor_refs,
    fetch_run,
    resolve_sha,
    resolve_tree,
)

pytestmark = pytest.mark.unit


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout.strip()


def _make_repo(root: Path) -> tuple[Path, dict[str, str]]:
    repo = root / "client"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "seed")
    _git(repo, "config", "user.email", "t@t"); _git(repo, "config", "user.name", "t")
    (repo / "data").mkdir()
    (repo / "data" / "train.json").write_text(json.dumps([{"q": "a"}, {"q": "b"}]))
    (repo / "prog.py").write_text("X = 1\n")
    (repo / "requirements.txt").write_text("")
    _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "seed")
    seed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "-q", "-b", "codeevolver-x-main")
    (repo / "prog.py").write_text("X = 2\n")
    _git(repo, "commit", "-q", "-am", "iter1")
    close1 = _git(repo, "rev-parse", "HEAD")
    # an evaluated-but-never-committed tree, anchored like close_effects does
    (repo / "prog.py").write_text("X = 3\n")
    tree = _git(repo, "write-tree")  # index unchanged -> use a temp index
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    tree = _git(repo, "write-tree")
    _git(repo, "reset", "-q", "--hard", "HEAD")
    snap = _git(repo, "commit-tree", tree, "-p", close1, "-m", "eval snapshot")
    _git(repo, "update-ref", "refs/codeevolver/evals/1/0", snap)
    _git(repo, "update-ref", "refs/codeevolver/iterations/1", close1)
    # a second commit with the same tree as close1 (doc-only revert), to test dedupe
    (repo / "README.md").write_text("x\n"); _git(repo, "add", "-A"); _git(repo, "commit", "-q", "-m", "doc")
    _git(repo, "rm", "-q", "README.md"); _git(repo, "commit", "-q", "-m", "undoc")
    close2 = _git(repo, "rev-parse", "HEAD")
    bundle = root / "repo.bundle"
    _git(repo, "bundle", "create", str(bundle), "--all")
    return bundle, {"seed": seed, "close1": close1, "close2": close2, "snap": snap, "snap_tree": tree}


def test_fetch_brings_anchors_and_resolves_trees(tmp_path):
    bundle, shas = _make_repo(tmp_path)
    bare = ensure_bare_repo(tmp_path / "repos" / "hover.git")
    n = fetch_run(bare, bundle, "job_1")
    assert n == 4  # heads: seed, codeevolver-x-main; codeevolver: evals/1/0, iterations/1
    anchors = eval_anchor_refs(bare, "job_1")
    assert anchors == [("refs/jobs/job_1/codeevolver/evals/1/0", shas["snap"])]
    ref = anchor_ref("job_1", "refs/codeevolver/evals/1/0")
    assert resolve_sha(bare, ref) == shas["snap"]
    assert resolve_tree(bare, ref) == shas["snap_tree"]
    assert resolve_tree(bare, shas["close1"]) == resolve_tree(bare, shas["close2"])  # dedupe key
    assert resolve_sha(bare, "refs/jobs/job_1/heads/seed") == shas["seed"]


def test_worktree_pool_checkout_keeps_staged_files(tmp_path):
    bundle, shas = _make_repo(tmp_path)
    bare = ensure_bare_repo(tmp_path / "repos" / "hover.git")
    fetch_run(bare, bundle, "job_1")
    client = tmp_path / "client"
    (client / "data" / "test.json").write_text(json.dumps([{"q": "t"}]))
    spec = BenchSpec(slug="hover", experiment_dir="hover", repo_path=str(client), assets=(),
                     program_factory="m:f", metric_fn="m:g", trainset_path="data/train.json",
                     testset_path="data/test.json", input_keys=("q",), num_threads=1,
                     comparison_set_size=0, subsample_eval_timeout=1, comparison_eval_timeout=1)
    pool = WorktreePool(bare, tmp_path / "wt", spec, 2)
    pool.ensure(shas["seed"])
    assert len(pool.slots) == 2
    slot = pool.slots[0]
    assert (slot.path / "data" / "test.json").exists()
    pool.checkout(slot, shas["snap"])
    assert (slot.path / "prog.py").read_text() == "X = 3\n"
    assert (slot.path / "data" / "test.json").exists()  # untracked file survived
    assert _git(slot.path, "rev-parse", "HEAD") == shas["snap"]
    assert pool.testset_sha256(slot) == pool.testset_sha256(pool.slots[1])
    pool.checkout(slot, shas["close1"])
    assert (slot.path / "prog.py").read_text() == "X = 2\n"
    pool.teardown()
    assert not slot.path.exists()
