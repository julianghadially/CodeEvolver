"""Run records shared by the macro-counts scripts: discovery, pairing, and git access.

A run is a directory named ``<arm>-<bench>-s<seed>[-redo]`` holding ``history.json``
and ``repo.bundle`` (``status.json`` and ``.codeevolver/memory/`` are used when
present). ``macro_counts.py`` (v4, pairwise ranking) and ``catalog.py`` (v5,
catalog discovery and counting) both read runs through this module.
Read-only against the run archives.
"""
from __future__ import annotations

import json
import re
import subprocess
import threading
from pathlib import Path

# An optional "-redo" suffix marks a re-run of the same (arm, bench, seed); when both exist the
# redo supersedes the original.
RUN_ID_RE = re.compile(r"^(?P<arm>[a-z][a-z0-9_]*)-(?P<bench>[a-z][a-z0-9_]*)-s(?P<seed>\d+)(?P<redo>-redo)?$")


BENCHES = ["hover", "phantomwiki", "ragqa", "paperreview", "arcagi2", "appworld", "alfworld"]  # display order; other benches are accepted and appended
# Main program file per bench. Unknown benches: pass --program-file BENCH=PATH, or the
# script looks for codeevolver_benchmarks/*<bench>*program*.py in the seed commit.
PROGRAM_FILE = {
    "hover": "langProBe/hover/hover_program.py",
    "phantomwiki": "src/program/phantomwiki_module.py",
    "ragqa": "langProBe/RAGQAArenaTech/RAGQAArenaTech_program.py",
    "paperreview": "src/program/review_pipeline.py",
    "arcagi2": "src/program/arc_pipeline.py",
    "appworld": "codeevolver_benchmarks/appworld_program.py",
    "alfworld": "codeevolver_benchmarks/alfworld_vision_program.py",
}
# Runs left out of pairing (the unsuffixed pareto_evolution runs used an earlier
# pruning rule). Override with --void REGEX ('' = nothing is void).
VOID_RUN_RE: re.Pattern | None = re.compile(r"^pareto_evolution-[a-z0-9]+-s[123]$")
# Paths excluded from per-iteration diffs (engine bookkeeping, tests, data).
DIFF_EXCLUDES = [
    ":!codeevolver.md",
    ":!.codeevolver",
    ":!.codeevolver/**",
    ":!.codeevolver.lock",
    ":!.gitignore",
    ":!CODEEVOLVER_SETUP.md",
    ":!tests/**",
    ":!data/**",
    ":!*.md",
    ":!*.lock",
]
AUTO_COMMIT_RE = re.compile(r"^iter \d+: session end \(auto-committed\)$")


_print_lock = threading.Lock()


def log(msg: str) -> None:
    with _print_lock:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Run discovery and pairing
# ---------------------------------------------------------------------------

class Run:
    def __init__(self, path: Path):
        self.path = path
        self.run_id = path.name
        m = RUN_ID_RE.match(self.run_id)
        if not m:
            raise ValueError(f"not a run dir: {path}")
        self.arm = m["arm"]
        self.bench = m["bench"]
        self.seed = int(m["seed"])
        self.redo = bool(m["redo"])
        self.history = json.loads((path / "history.json").read_text(encoding="utf-8"))
        self.iters: dict[str, dict] = self.history["iteration_history"]
        self.summary = self.history.get("summary") or {}
        st = path / "status.json"
        self.status = json.loads(st.read_text(encoding="utf-8")) if st.exists() else {}
        self.job_id = self.history.get("job_id") or self.status.get("job_id")

    @property
    def void(self) -> bool:
        return bool(VOID_RUN_RE and VOID_RUN_RE.match(self.run_id))

    @property
    def iteration_keys(self) -> list[str]:
        return sorted((k for k in self.iters if k.isdigit() and int(k) > 0), key=int)

    def score(self, key: str):
        v = self.summary.get(key)
        if v is None:
            v = self.status.get(key)
        return v

    def seed_sha(self) -> str | None:
        z = self.iters.get("0") or {}
        return z.get("close_sha") or (self.iters.get("1") or {}).get("parent_sha")

    def final_sha(self) -> str | None:
        fc = self.summary.get("final_candidate") or self.status.get("final_candidate") or {}
        return fc.get("git_sha")


class Pair:
    """arm1's run and arm2's run on the same bench and seed."""

    def __init__(self, run1: Run, run2: Run):
        self.run1, self.run2 = run1, run2
        self.bench, self.seed = run1.bench, run1.seed
        self.pair_id = f"{self.bench}-s{self.seed}"

    def programs(self, order: str) -> dict[str, Run]:
        """Program letter -> run for one presentation order."""
        return {"A": self.run1, "B": self.run2} if order == "ab" else {"A": self.run2, "B": self.run1}


def key_of(pair: Pair, order: str) -> str:
    return f"{pair.pair_id}__{order}"


def discover_runs(archive_root: Path, pattern: str | None,
                  benches: set[str] | None, arms: set[str] | None) -> list[Run]:
    """Find run dirs (need history.json + repo.bundle) under any of:
    <root>/ce-run*-archive/ce-runs/<run>, <root>/ce-runs/<run>, <root>/<run>."""
    candidates: list[Path] = []
    for archive in sorted(archive_root.glob("ce-run*-archive")):
        candidates += sorted((archive / "ce-runs").glob("*-s[0-9]*"))
    candidates += sorted((archive_root / "ce-runs").glob("*-s[0-9]*"))
    candidates += sorted(archive_root.glob("*-s[0-9]*"))
    runs, seen = [], set()
    for d in candidates:
        if d.name in seen or not (d / "history.json").exists() or not (d / "repo.bundle").exists():
            continue
        m = RUN_ID_RE.match(d.name)
        if not m or (benches and m["bench"] not in benches) or (arms and m["arm"] not in arms):
            continue
        if pattern and not re.search(pattern, d.name):
            continue
        seen.add(d.name)
        runs.append(Run(d))
    return runs


def build_pairs(runs: list[Run], arm1: str, arm2: str) -> tuple[list[Pair], list[tuple[str, str]]]:
    """Pair arm1 and arm2 runs by (bench, seed). Returns (pairs, [(run_id, why unpaired)])."""
    by_key: dict[tuple[str, int], dict[str, Run]] = {}
    unpaired: list[tuple[str, str]] = []
    for r in runs:
        if r.void:
            unpaired.append((r.run_id, f"void (matched --void `{VOID_RUN_RE.pattern}`)"))
            continue
        slot = by_key.setdefault((r.bench, r.seed), {})
        prev = slot.get(r.arm)
        if prev is not None and prev.redo != r.redo:
            keep, drop = (r, prev) if r.redo else (prev, r)
            slot[r.arm] = keep
            unpaired.append((drop.run_id, f"superseded by {keep.run_id}"))
        else:
            slot[r.arm] = r
    pairs = []
    for (bench, seed), arms in sorted(by_key.items()):
        if arm1 in arms and arm2 in arms:
            pairs.append(Pair(arms[arm1], arms[arm2]))
        else:
            missing = arm2 if arm1 in arms else arm1
            unpaired += [(r.run_id, f"no valid {missing} run for {bench} s{seed}") for r in arms.values()]
    return pairs, unpaired


# ---------------------------------------------------------------------------
# Git helpers (bare clone per run, cached)
# ---------------------------------------------------------------------------

def git(git_dir: Path, *args: str, check: bool = True) -> str:
    p = subprocess.run(["git", "--git-dir", str(git_dir), *args], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if check and p.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed in {git_dir}: {p.stderr.strip()}")
    return p.stdout


def ensure_bare(run: Run, bundles_dir: Path) -> Path:
    git_dir = bundles_dir / f"{run.run_id}.git"
    if not (git_dir / "HEAD").exists():
        bundles_dir.mkdir(parents=True, exist_ok=True)
        p = subprocess.run(["git", "clone", "--bare", "--quiet", str(run.path / "repo.bundle"), str(git_dir)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if p.returncode != 0:
            raise RuntimeError(f"clone failed for {run.run_id}: {p.stderr.strip()}")
    # `clone --bare` only copies refs/heads; the per-iteration eval snapshots live
    # under refs/codeevolver/evals/<iter>/<n> and are needed for reverted iterations.
    if not git(git_dir, "for-each-ref", "refs/codeevolver/", check=False).strip():
        subprocess.run(["git", "--git-dir", str(git_dir), "fetch", "--quiet", str(run.path / "repo.bundle"),
                        "refs/codeevolver/*:refs/codeevolver/*"], capture_output=True, text=True)
    return git_dir


def show_file(git_dir: Path, sha: str, path: str) -> str | None:
    p = subprocess.run(["git", "--git-dir", str(git_dir), "show", f"{sha}:{path}"], capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    return p.stdout if p.returncode == 0 else None


def diff_between(git_dir: Path, a: str, b: str) -> str:
    return git(git_dir, "diff", "--no-color", f"{a}..{b}", "--", ".", *DIFF_EXCLUDES)


def commit_messages(git_dir: Path, a: str, b: str) -> list[str]:
    out = git(git_dir, "log", "--format=%B%x00", f"{a}..{b}", check=False)
    msgs = [m.strip() for m in out.split("\x00")]
    msgs = [m for m in msgs if m and not AUTO_COMMIT_RE.match(m)]
    return list(reversed(msgs))  # chronological


def eval_snapshot_shas(git_dir: Path, iteration: str) -> list[str]:
    """Eval-snapshot commits of one iteration, latest first."""
    out = git(git_dir, "for-each-ref", "--format=%(refname) %(objectname)",
              f"refs/codeevolver/evals/{iteration}/", check=False)
    refs = []
    for line in out.splitlines():
        name, sha = line.split()
        n = name.rsplit("/", 1)[-1]
        refs.append((int(n) if n.isdigit() else -1, sha))
    return [sha for _, sha in sorted(refs, reverse=True)]


def memory_summary(run: Run, iteration: str) -> str | None:
    for f in sorted((run.path / ".codeevolver" / "memory").glob(f"*/iteration_{iteration}_change_summary.md")):
        txt = f.read_text(encoding="utf-8", errors="replace").strip()
        if txt:
            return txt
    return None


def program_file_for(run: Run, git_dir: Path) -> str:
    if run.bench in PROGRAM_FILE:
        return PROGRAM_FILE[run.bench]
    ref = run.seed_sha() or "refs/heads/seed"
    names = git(git_dir, "ls-tree", "-r", "--name-only", ref, check=False).split()
    hits = [n for n in names if n.startswith("codeevolver_benchmarks/")
            and run.bench in n and "program" in n and n.endswith(".py")]
    if len(hits) != 1:
        raise SystemExit(f"[{run.run_id}] cannot pick program file for bench '{run.bench}' "
                         f"(candidates: {hits or 'none'}); pass --program-file {run.bench}=PATH")
    PROGRAM_FILE[run.bench] = hits[0]
    log(f"[{run.run_id}] program file for {run.bench}: {hits[0]}")
    return hits[0]


def seed_source(run: Run, git_dir: Path, prog: str) -> str:
    seed_sha = run.seed_sha()
    src = show_file(git_dir, seed_sha, prog) if seed_sha else None
    if src is None:
        src = show_file(git_dir, "refs/heads/seed", prog) or "(seed program not found)"
    return src
