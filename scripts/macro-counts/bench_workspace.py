"""The per-benchmark workspace the catalog sessions (discover and count) read.

One workspace per benchmark holds every run of that benchmark (by default the
three pareto_evolution and three freeform seeds), unblinded:

  RUNS.md                       the map: per run, its id, job id, repo clone, memory copy, final
                                commit, and an iteration table (parent, branch, closing commit,
                                landed yes/no, skipped)
  seed/<program file>           the seed program (the starting point of every run)
  repos/<run_id>/               a git clone of the run's repo.bundle: every branch, plus the
                                per-iteration eval snapshots under refs/codeevolver/evals/
  memory/<run_id>/              the run's memory files: memory.md,
                                iteration_change_summaries.md, iteration_<N>_change_summary.md

The change summaries themselves go into the prompt (``change_summaries``);
the session reaches everything else through the file system. Scores are not
hidden: memory.md carries them, and the prompts say scores are irrelevant.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

from runs import Run, build_pairs, discover_runs, ensure_bare, log, program_file_for, seed_source

MEMORY_FILES = ("memory.md", "iteration_change_summaries.md")  # plus iteration_<N>_change_summary.md
# Method inputs written by ``catalog.py fill``: inputs/<bench>/<run_id>/raw_change_summaries.md (the run's own
# file, verbatim) and processed_change_summaries.md (raw plus a section for every iteration it lacks).
INPUTS_DIR = Path(__file__).resolve().parent / "inputs"


def bench_runs(archive_root: Path, bench: str, arm1: str, arm2: str) -> tuple[list[Run], list[tuple[str, str]]]:
    """The runs of one benchmark that form complete (arm1, arm2) seed pairs, arm1 first then by seed.
    Returns (runs, [(run_id, why left out)]); the redo and void rules of ``build_pairs`` apply."""
    found = discover_runs(archive_root, None, {bench}, {arm1, arm2})
    pairs, unpaired = build_pairs(found, arm1, arm2)
    runs = [p.run1 for p in pairs] + [p.run2 for p in pairs]
    runs.sort(key=lambda r: (r.arm != arm1, r.seed))
    return runs, unpaired


def memory_dir(run: Run) -> Path | None:
    d = run.path / ".codeevolver" / "memory" / str(run.job_id)
    return d if d.is_dir() else None


def processed_summaries_path(run: Run) -> Path:
    return INPUTS_DIR / run.bench / run.run_id / "processed_change_summaries.md"


def change_summaries(run: Run) -> str:
    """The summaries the sessions see: the ``fill`` stage's processed file when it exists, else the raw file."""
    p = processed_summaries_path(run)
    if p.exists():
        return p.read_text(encoding="utf-8").strip()
    return raw_change_summaries(run)


def raw_change_summaries(run: Run) -> str:
    """The run's ``iteration_change_summaries.md``; rebuilt from history.json when the file is missing."""
    d = memory_dir(run)
    f = d / "iteration_change_summaries.md" if d else None
    if f and f.exists() and f.read_text(encoding="utf-8", errors="replace").strip():
        return f.read_text(encoding="utf-8", errors="replace").strip()
    parts = []
    for k in run.iteration_keys:
        s = (run.iters[k].get("change_summary") or "").strip()
        parts.append(f"# Change Description Iteration {k}\n{s or '(no change summary recorded)'}")
    return "\n\n".join(parts)


def _git(repo: Path, *args: str) -> str:
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    if p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {repo}: {p.stderr.strip()}")
    return p.stdout


def clone_run(run: Run, dest: Path) -> None:
    """A fresh non-bare clone of the run's bundle with every ref (branches and
    refs/codeevolver/*) as local refs, no remote, checked out at the final commit."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    _git(dest, "init", "--quiet")
    _git(dest, "fetch", "--quiet", "--update-head-ok", str(run.path / "repo.bundle"), "refs/*:refs/*")
    target = run.final_sha() or "refs/heads/main"
    _git(dest, "-c", "advice.detachedHead=false", "checkout", "--quiet", "--force", "--detach", target)


def copy_memory(run: Run, dest: Path) -> list[str]:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    d = memory_dir(run)
    if not d:
        return []
    names = [f.name for f in sorted(d.iterdir()) if f.is_file() and
             (f.name in MEMORY_FILES or (f.name.startswith("iteration_") and f.name.endswith("_change_summary.md")))]
    for n in names:
        shutil.copy2(d / n, dest / n)
    if processed_summaries_path(run).exists():  # the session reads the filled summaries, not the raw ones
        (dest / "iteration_change_summaries.md").write_text(change_summaries(run) + "\n", encoding="utf-8")
    return names


def iteration_rows(run: Run) -> list[dict]:
    rows = []
    for k in ["0"] + run.iteration_keys:
        rec = run.iters.get(k) or {}
        rows.append({"iteration": int(k), "parent_iteration": rec.get("parent_iteration"),
                     "git_branch": rec.get("git_branch") or "", "close_sha": rec.get("close_sha") or "",
                     "landed": bool(rec.get("changes_landed")), "skipped": bool(rec.get("skipped"))})
    return rows


def landed_iterations(run: Run) -> set[int]:
    """Iterations whose closing content landed (``changes_landed`` in history.json)."""
    return {int(k) for k in run.iteration_keys if run.iters[k].get("changes_landed")}


def runs_md(bench: str, runs: list[Run], prog: str, ws: Path) -> str:
    fname = Path(prog).name
    out = [f"# Runs of benchmark `{bench}`\n",
           f"Every run started from the same seed program, `seed/{fname}` (the repo path is `{prog}`). "
           "An iteration's code is its closing commit: `git -C repos/<run> show <close_sha>:" + prog + "`, "
           "or check it out with `git -C repos/<run> checkout --detach <close_sha>`. Diff an iteration against "
           "its parent with `git -C repos/<run> diff <parent close_sha> <close_sha> -- " + prog + "` (the "
           "program may also import sibling modules; `git diff --stat` shows which files changed). "
           "In some runs every iteration is on the same branch, so use the commit, not the branch name. "
           "`landed = no` means the iteration's change was reverted or never committed; its closing commit "
           "is its parent's content.\n",
           f"Workspace root: `{ws}`\n"]
    for r in runs:
        mem = sorted(p.name for p in (ws / "memory" / r.run_id).iterdir()) if (ws / "memory" / r.run_id).exists() else []
        out.append(f"## {r.run_id}\n")
        out.append(f"- job id: `{r.job_id}`")
        out.append(f"- repo clone: `repos/{r.run_id}/` (checked out at the final commit `{r.final_sha()}`)")
        out.append(f"- memory: `memory/{r.run_id}/` ({', '.join(f for f in mem if not f.endswith('_change_summary.md')) or 'none'}, "
                   f"plus {sum(f.startswith('iteration_') and f.endswith('_change_summary.md') for f in mem)} "
                   "per-iteration change summaries `iteration_<N>_change_summary.md`)")
        out.append(f"- original run directory (read-only): `{r.path}`")
        out.append(f"- iterations: {len(r.iteration_keys)} ({len(landed_iterations(r))} landed)\n")
        out.append("| iteration | parent | landed | skipped | branch | close_sha |\n|---|---|---|---|---|---|")
        for row in iteration_rows(r):
            out.append(f"| {row['iteration']} | {'-' if row['parent_iteration'] is None else row['parent_iteration']} | "
                       f"{'seed' if row['iteration'] == 0 else ('yes' if row['landed'] else 'no')} | "
                       f"{'yes' if row['skipped'] else ''} | `{row['git_branch']}` | `{row['close_sha']}` |")
        out.append("")
    return "\n".join(out) + "\n"


def materialize_bench(bench: str, runs: list[Run], out: Path) -> tuple[Path, dict]:
    """Build ``<out>/workspaces/<bench>/``. Returns (workspace, stats)."""
    ws = out / "workspaces" / bench
    ws.mkdir(parents=True, exist_ok=True)
    first_git = ensure_bare(runs[0], out / "bundles")
    prog = program_file_for(runs[0], first_git)
    seeds = {r.run_id: seed_source(r, ensure_bare(r, out / "bundles"), prog) for r in runs}
    if len(set(seeds.values())) != 1:
        log(f"[{bench}] WARNING: the runs' seed programs differ; showing {runs[0].run_id}'s")
    seed_path = ws / "seed" / Path(prog).name
    seed_path.parent.mkdir(parents=True, exist_ok=True)
    seed_path.write_text(seeds[runs[0].run_id], encoding="utf-8")
    stats: dict = {"bench": bench, "program_file": prog, "runs": {}}
    for r in runs:
        clone_run(r, ws / "repos" / r.run_id)
        names = copy_memory(r, ws / "memory" / r.run_id)
        stats["runs"][r.run_id] = {"arm": r.arm, "seed": r.seed, "job_id": r.job_id,
                                   "iterations": len(r.iteration_keys), "landed": len(landed_iterations(r)),
                                   "memory_files": len(names), "summary_chars": len(change_summaries(r))}
    (ws / "RUNS.md").write_text(runs_md(bench, runs, prog, ws), encoding="utf-8")
    (ws / "workspace.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
    return ws, stats


def input_sha256(ws: Path, runs: list[Run]) -> str:
    """Content hash of what a session sees: RUNS.md, seed, memory copies, and each run's bundle.
    The repo clones are left out (a session may move their checkout); their bundles stand in."""
    h = hashlib.sha256()
    files = [ws / "RUNS.md"] + sorted((ws / "seed").rglob("*")) + sorted((ws / "memory").rglob("*"))
    for f in files:
        if f.is_file():
            h.update(str(f.relative_to(ws)).encode() + b"\0" + hashlib.sha256(f.read_bytes()).hexdigest().encode())
    for r in runs:
        h.update(r.run_id.encode() + b"\0" + hashlib.sha256((r.path / "repo.bundle").read_bytes()).hexdigest().encode())
    return h.hexdigest()
