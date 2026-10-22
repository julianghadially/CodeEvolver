#!/usr/bin/env python3
"""Lineage audit: how often does an iteration build on something OTHER than
the immediately previous candidate?

Engine-3 arms where the architect owns git (freeform, ee, asa) can move HEAD
anywhere in history. For a "trajectory" figure (each candidate builds on the
last) we need to know how often that assumption breaks. This script answers it
from collected run artifacts only (`history.json`, `repo.bundle`,
`.codeevolver/architect_traces/`), so the number is replicable.

Per iteration i, with `pred` = the close of the previous closed iteration
(iteration 0 = seed), four independent signals:

  1. topo     — git ancestry. Recomputes the engine's `derive_iteration_parent`
                rule (newest past close that is an ancestor of close_i) from the
                bundle; non-latest when that is not `pred`, with the fork
                point (merge-base with pred) mapped to the iteration that made
                it. Also reports the engine's recorded `parent_iteration` and
                flags disagreement (its `merge_base` fallback credits pred).
  2. rollback — content. The program tree at close_i (all files except the
                loop-written `codeevolver.md`) differs from pred's but equals an
                OLDER candidate's. Catches restores that keep git linear
                (`git revert`, `git checkout <old> -- .`).
  3. partial  — content, per file. Some file at close_i differs from pred's
                version but is byte-identical to that file at an older close
                (a component restored from history; includes undoing pred's own
                change to that file). Deletions are not counted.
  4. trace    — behavior. Whole-tree moves (`git reset --hard <ref>`,
                `git checkout <ref>`, `git switch`) and file restores
                (`git checkout <ref> -- <paths>`) in the architect's bash calls,
                with <ref> resolved in the bundle and classified as the
                iteration's start, a commit made this iteration, or HISTORICAL
                (an ancestor of pred / another lineage). Only sha refs are
                classified: relative refs (`HEAD~1`) and branch names (which
                resolve to their tip at run END) are counted apart.
                A historical checkout that was later abandoned shows here but
                not in 1–3.

Headline: an iteration is "non-latest" when topo OR rollback fires.

Usage:
    python scripts/lineage_audit.py                       # ~/ce-runs/freeform-*
    python scripts/lineage_audit.py ~/ce-runs/ee-hover-s1 ~/ce-runs/asa-*
    python scripts/lineage_audit.py --out ~/ce-analysis/lineage
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

LOOP_WRITTEN = {"codeevolver.md"}
DEFAULT_RUNS = str(Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser() / "freeform-*")
DEFAULT_OUT = "~/ce-analysis/lineage"

GIT_CMD_RE = re.compile(r"\bgit\s+(?:-C\s+\S+\s+)?(reset|checkout|switch|revert)\b([^;&|\n]*)")
HEX_RE = re.compile(r"^[0-9a-f]{7,40}$")


# ---------------------------------------------------------------------------
# git helpers (all against a bare mirror of repo.bundle)


class Repo:
    def __init__(self, path: Path):
        self.path = path
        self._fp: dict[str, dict[str, str]] = {}
        self._anc: dict[tuple[str, str], bool] = {}

    def git(self, *args: str, check: bool = True) -> str:
        r = subprocess.run(
            ["git", "-C", str(self.path), *args],
            capture_output=True, text=True,
        )
        if check and r.returncode != 0:
            raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()}")
        return r.stdout.strip() if r.returncode == 0 else ""

    def resolve(self, ref: str) -> str | None:
        out = self.git("rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}", check=False)
        return out or None

    def is_ancestor(self, a: str, d: str) -> bool:
        if a == d:
            return True
        key = (a, d)
        if key not in self._anc:
            r = subprocess.run(
                ["git", "-C", str(self.path), "merge-base", "--is-ancestor", a, d],
                capture_output=True,
            )
            self._anc[key] = r.returncode == 0
        return self._anc[key]

    def files(self, sha: str) -> dict[str, str]:
        """path -> blob sha, excluding loop-written files."""
        if sha not in self._fp:
            out = self.git("ls-tree", "-r", sha)
            fp = {}
            for line in out.splitlines():
                meta, path = line.split("\t", 1)
                if path not in LOOP_WRITTEN:
                    fp[path] = meta.split()[2]
            self._fp[sha] = fp
        return self._fp[sha]


def open_mirror(run_dir: Path, cache: Path) -> Repo:
    bundle = run_dir / "repo.bundle"
    dest = cache / "mirrors" / run_dir.name
    if not dest.exists() or bundle.stat().st_mtime > dest.stat().st_mtime:
        subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "-q", "--mirror", str(bundle), str(dest)],
            check=True, capture_output=True,
        )
    return Repo(dest)


# ---------------------------------------------------------------------------
# trace scan


def bash_commands(trace_dir: Path) -> list[str]:
    cmds = []
    for turn in sorted(trace_dir.glob("turn_*.json")):
        try:
            data = json.loads(turn.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        for block in data.get("blocks") or []:
            if block.get("kind") == "tool_use" and block.get("name") in ("bash", "Bash"):
                cmd = (block.get("input") or {}).get("command")
                if isinstance(cmd, str):  # malformed calls carry a dict
                    cmds.append(cmd)
    return cmds


def git_moves(cmd: str) -> list[tuple[str, str]]:
    """(op, ref) for each history-addressing git command in a bash string.

    op: "tree" (whole-tree move), "file" (`checkout <ref> -- paths`),
    "revert". Working-tree discards (`checkout -- f`, `checkout .`,
    `reset --hard` with no ref) are not moves and are skipped.
    """
    moves = []
    for verb, rest in GIT_CMD_RE.findall(cmd):
        try:
            toks = shlex.split(rest)
        except ValueError:
            toks = rest.split()
        file_mode = "--" in toks
        args = [t for t in (toks[: toks.index("--")] if file_mode else toks) if not t.startswith("-")]
        if verb == "revert":
            moves += [("revert", a) for a in args[:1]]
            continue
        if verb == "reset" and "--hard" not in toks:
            continue
        if not args or args[0] in (".", "HEAD"):
            continue
        ref = args[0]
        if verb == "checkout" and not file_mode and "/" in ref and not HEX_RE.match(ref):
            continue  # `git checkout path/to/file` (discard), not a ref
        moves.append(("file" if file_mode else "tree", ref))
    return moves


# ---------------------------------------------------------------------------
# per-run analysis


def load_closes(run_dir: Path) -> list[dict]:
    hist = json.loads((run_dir / "history.json").read_text())["iteration_history"]
    recs = sorted(((int(k), v) for k, v in hist.items()), key=lambda kv: kv[0])
    return [dict(v, iteration=k) for k, v in recs if v.get("close_sha")]


def trace_dir_for(run_dir: Path, it: int) -> Path | None:
    hits = glob.glob(str(run_dir / ".codeevolver/architect_traces/*" / f"iteration_{it}_architect_trace"))
    return Path(hits[0]) if hits else None


def classify_ref(repo: Repo, sha: str, pred: str, close: str, closes: list[dict]) -> str:
    if sha == pred:
        return "start"
    if repo.is_ancestor(pred, sha):
        return "this_iteration"
    for c in closes:  # oldest first: first close that contains the commit
        if repo.is_ancestor(sha, c["close_sha"]):
            return f"historical:iter{c['iteration']}"
    return "historical:off_lineage"


def analyze_run(run_dir: Path, cache: Path) -> dict:
    closes = load_closes(run_dir)
    repo = open_mirror(run_dir, cache)
    rows = []
    for idx in range(1, len(closes)):
        rec, pred_rec = closes[idx], closes[idx - 1]
        it, close, pred = rec["iteration"], rec["close_sha"], pred_rec["close_sha"]
        older = closes[: idx - 1]  # strictly before pred

        # 1. topo
        git_parent = 0
        for past in reversed(closes[:idx]):
            if repo.is_ancestor(past["close_sha"], close):
                git_parent = past["iteration"]
                break
        recorded = rec.get("parent_iteration")
        topo = git_parent != pred_rec["iteration"]
        # Off pred's line: where did close_i fork from? (The engine's
        # `merge_base` derivation credits this to pred; the fork point says
        # which older candidate's content it actually built on.)
        fork = None
        if topo:
            base = repo.git("merge-base", close, pred, check=False)
            if base:
                fork = classify_ref(repo, base, pred, close, closes[:idx])

        # 2. full rollback / 3. partial restores
        fp, fp_pred = repo.files(close), repo.files(pred)
        rollback_to = None
        if fp != fp_pred:
            for past in reversed(older):
                if repo.files(past["close_sha"]) == fp:
                    rollback_to = past["iteration"]
                    break
        partial = []
        for path, blob in fp.items():
            if fp_pred.get(path) == blob:
                continue
            for past in reversed(older):
                if repo.files(past["close_sha"]).get(path) == blob:
                    partial.append({"path": path, "from_iter": past["iteration"]})
                    break

        # 4. trace
        trace = {"tree_historical": [], "file_historical": [], "revert": [],
                 "relative_refs": 0, "named_refs": [], "unresolved": 0, "in_iteration_moves": 0}
        tdir = trace_dir_for(run_dir, it)
        for cmd in bash_commands(tdir) if tdir else []:
            for op, ref in git_moves(cmd):
                if op == "revert":
                    trace["revert"].append(ref)
                    continue
                if "~" in ref or "^" in ref or ref.startswith("HEAD") or ref.startswith("@"):
                    trace["relative_refs"] += 1
                    continue
                if not HEX_RE.match(ref):
                    # A branch name resolves to its tip at RUN END, not at the
                    # time of the command — unclassifiable post hoc.
                    trace["named_refs"].append(ref)
                    continue
                sha = repo.resolve(ref)
                if not sha:
                    trace["unresolved"] += 1
                    continue
                cls = classify_ref(repo, sha, pred, close, closes)
                if cls.startswith("historical"):
                    trace[f"{op}_historical"].append({"ref": ref, "class": cls})
                else:
                    trace["in_iteration_moves"] += 1

        rows.append({
            "iteration": it,
            "pred_iteration": pred_rec["iteration"],
            "git_parent_iteration": git_parent,
            "recorded_parent_iteration": recorded,
            "recorded_disagrees": recorded is not None and int(recorded) != git_parent,
            "topo_non_latest": topo,
            "fork_point": fork,
            "rollback_to_iteration": rollback_to,
            "partial_restores": partial,
            "trace": trace,
            "trace_found": tdir is not None,
            "non_latest": topo or rollback_to is not None,
        })
    return {"run": run_dir.name, "iterations": rows}


def summarize(run: dict) -> dict:
    rows = run["iterations"]
    n = len(rows)
    return {
        "run": run["run"],
        "n": n,
        "non_latest": sum(r["non_latest"] for r in rows),
        "topo": sum(r["topo_non_latest"] for r in rows),
        "rollback": sum(r["rollback_to_iteration"] is not None for r in rows),
        "partial": sum(bool(r["partial_restores"]) for r in rows),
        "trace_hist": sum(bool(r["trace"]["tree_historical"] or r["trace"]["file_historical"]) for r in rows),
        "revert": sum(bool(r["trace"]["revert"]) for r in rows),
        "recorded_disagrees": sum(r["recorded_disagrees"] for r in rows),
        "no_trace": sum(not r["trace_found"] for r in rows),
    }


def pct(k: int, n: int) -> str:
    return f"{k:3d} ({100 * k / n:4.1f}%)" if n else f"{k:3d}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help=f"collected run dirs (default {DEFAULT_RUNS})")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for lineage.json + bundle mirrors")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in (args.runs or sorted(glob.glob(os.path.expanduser(DEFAULT_RUNS))))]
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)

    runs, sums = [], []
    for rd in run_dirs:
        if not (rd / "history.json").exists() or not (rd / "repo.bundle").exists():
            print(f"skip {rd.name}: missing history.json or repo.bundle", file=sys.stderr)
            continue
        run = analyze_run(rd, out)
        runs.append(run)
        sums.append(summarize(run))

    cols = ["non_latest", "topo", "rollback", "partial", "trace_hist", "revert"]
    print(f"{'run':34s} {'n':>4s} " + " ".join(f"{c:>13s}" for c in cols) + "  flags")
    for s in sums:
        flags = []
        if s["recorded_disagrees"]:
            flags.append(f"recorded!=git:{s['recorded_disagrees']}")
        if s["no_trace"]:
            flags.append(f"no_trace:{s['no_trace']}")
        print(f"{s['run']:34s} {s['n']:4d} " + " ".join(f"{pct(s[c], s['n']):>13s}" for c in cols) + "  " + " ".join(flags))
    total = {c: sum(s[c] for s in sums) for c in cols}
    n = sum(s["n"] for s in sums)
    print(f"{'TOTAL':34s} {n:4d} " + " ".join(f"{pct(total[c], n):>13s}" for c in cols))

    for run in runs:
        for r in run["iterations"]:
            t = r["trace"]
            if r["non_latest"] or r["partial_restores"] or t["tree_historical"] or t["file_historical"]:
                bits = []
                if r["topo_non_latest"]:
                    bits.append(f"git-parent=iter{r['git_parent_iteration']}, forked at {r['fork_point']}")
                if r["rollback_to_iteration"] is not None:
                    bits.append(f"rollback-to=iter{r['rollback_to_iteration']}")
                if r["partial_restores"]:
                    bits.append("restored " + ", ".join(f"{p['path']}<-iter{p['from_iter']}" for p in r["partial_restores"]))
                for h in t["tree_historical"]:
                    bits.append(f"trace tree-move {h['ref'][:10]} ({h['class']})")
                for h in t["file_historical"]:
                    bits.append(f"trace file-restore {h['ref'][:10]} ({h['class']})")
                print(f"  {run['run']} it{r['iteration']}: " + "; ".join(bits))

    (out / "lineage.json").write_text(json.dumps({"summary": sums, "runs": runs}, indent=1))
    print(f"\nwrote {out / 'lineage.json'}")


if __name__ == "__main__":
    main()
