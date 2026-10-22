#!/usr/bin/env python3
"""Transplant audit: how often does a pareto_evolution child discard the
loop-sampled parent and adopt another candidate's program instead?

In `pareto_evolution` the LOOP picks the parent (Pareto sample / exploit-best)
and forks the child branch from its close sha; the architect must close on
that branch. Git ancestry therefore always says "built on the parent" — but
the architect reads memory.md (which names other iterations and their shas)
and can `git checkout <other> -- <files>`, `git apply` another iteration's
attempted diff, or retype another program. This script measures that from the
closed program CONTENT, so it is independent of how the code got there.

Read-only against the run data (history.json, repo.bundle, population.json).

Program. Per run, the program is every file that at least one closed
iteration changed relative to the seed, excluding the loop-written
`codeevolver.md`, tests (`tests/…`, `test_*.py`) and non-`.py` files.
A candidate's program is the set of its normalized lines across those files
(whitespace collapsed; lines < 12 chars or without a 3-letter word dropped).
Files are pooled, so keeping the parent's version of one file while copying
another counts as keeping the parent.

Candidates are the seed plus every iteration with `close_tree`,
`changes_landed` and no constraint violation. Lineage follows
`selected_parent_iteration`. An iteration is ELIGIBLE when at least one earlier
candidate is off its lineage (not an ancestor).

Rates use EVERY iteration the run executed (1..N) as the denominator, so
"x% of iterations were transplants" reads literally; an iteration that could
not transplant (no off-lineage candidate yet, no change landed, constraint
revert) simply counts as a non-transplant. The eligible-only rate is kept as
a secondary column.

Transplant. Eligible iteration i whose program is >= 0.85 Jaccard-similar to
some off-lineage candidate (the "source", the most similar one) AND at least
0.10 more similar to it than to the sampled parent. Also reported per
transplant: how many of the parent's distinctive lines (in parent, not in
source) the child kept, and whether the source is iteration i-1.

Gap. For each transplant, source minus parent comparison-set score from
`population.json` (same fixed rows for both). "Large" = beyond 2 standard
errors of a difference between two independent evals on the comparison set,
using the run's `run_variance_estimate` run-to-run std on the train subsample
scaled by sqrt(n_train / n_comparison). A source that was never admitted has
no comparison score and is reported as unscored.

Git cross-check. The content rule is validated against explicit copy
operations in the architect's bash calls (`.codeevolver/architect_traces/`):
`git checkout <ref> -- <program files>`, `git show <ref>:<f> > <program
file>` (also via a temp file then `cp`/`mv`), `git restore --source`,
`git apply` of another iteration's `iteration_N_attempted.diff`,
`git cherry-pick` and `git merge`. A ref is mapped to the iteration whose
close sha it is (else the oldest close containing it; `codeevolver-*-iterN`
branch names map to N). Only copies from a candidate OFF the child's lineage
count. Each iteration with a copy op or a content flag is classified:
  flagged_and_git      content rule and a git copy agree
  content_only         flagged with no git copy (rewritten via edit tools)
  git_no_change        git copy, but the iteration landed no change
  git_parent_similar   git copy, closing program >= 0.85 to the source but the
                       parent was already within 0.10 of it (margin rule)
  git_edited_after     git copy then substantial new code: 0.50-0.85 to source
  git_not_retained     git copy, closing program < 0.50 to the source
`git_edited_after` iterations are transplants the content rule misses; the
summary reports the rate with them added as an upper bound. Branch names
resolve to their END-of-run tip, and copies retyped by hand are invisible to
this check, which is why content stays the primary measure.

Outputs:
  results/transplant_rates.csv     benchmark x seed transplant counts/rates
                                   (+ per-benchmark mean and the median run)
  results/transplant_by_benchmark.csv
                                   per benchmark (seeds pooled): rate over all
                                   iterations and over iterations 1-10 / 1-20,
                                   the eligible-only rate, and archive integrity
                                   (admitted candidates / transplant parents
                                   missing from the final population.json)
  results/transplant_by_run.csv    the same columns, one row per run
  results/transplants.csv          one row per transplant, with the gap
  results/transplant_git_crossref.csv
                                   one row per iteration with a git copy op or
                                   a content flag, with its class
  <out>/transplant.json            every eligible iteration's metrics

Usage:
    python scripts/transplant_audit.py
    python scripts/transplant_audit.py ~/ce-runs/pareto_evolution-hover-s1-redo
    python scripts/transplant_audit.py --results-dir results --out ~/ce-analysis/transplant
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import shlex
import statistics as st
import subprocess
import sys
from pathlib import Path

# Only the -redo cells count: they are the reported pareto_evolution runs
# (orchestrator/matrix.evo.yaml).
DEFAULT_RUNS = (str(Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser() / "pareto_evolution-*-redo"),)
DEFAULT_OUT = "~/ce-analysis/transplant"
REPO_ROOT = Path(__file__).resolve().parent.parent

SIM_THRESHOLD = 0.85
EDITED_FLOOR = 0.50  # git copy then edits: still >= this similar to the source
SIM_MARGIN = 0.10
MIN_DISTINCTIVE = 10  # parent-distinctive lines needed to report "kept"
LOOP_WRITTEN = {"codeevolver.md"}
WORD_RE = re.compile(r"[A-Za-z]{3}")
RUN_RE = re.compile(r"pareto_evolution-(?P<bench>[a-z0-9]+)-s(?P<seed>\d+)")


# ---------------------------------------------------------------------------
# git (against a bare mirror of repo.bundle)


def git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else ""


def open_mirror(run_dir: Path, cache: Path) -> Path:
    bundle = run_dir / "repo.bundle"
    dest = cache / "mirrors" / run_dir.name
    if not dest.exists() or bundle.stat().st_mtime > dest.stat().st_mtime:
        subprocess.run(["rm", "-rf", str(dest)], check=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "-q", "--mirror", str(bundle), str(dest)],
                       check=True, capture_output=True)
    return dest


def is_program_file(path: str) -> bool:
    name = os.path.basename(path)
    return (path.endswith(".py") and path not in LOOP_WRITTEN
            and not path.startswith("tests/") and "/tests/" not in path
            and not name.startswith("test_"))


def norm_lines(text: str) -> set[str]:
    out = set()
    for ln in text.splitlines():
        s = " ".join(ln.split())
        if len(s) >= 12 and WORD_RE.search(s):
            out.add(s)
    return out


def jaccard(a: set, b: set) -> float:
    return len(a & b) / max(1, len(a | b))


# ---------------------------------------------------------------------------
# git cross-check (architect bash calls)

ATTEMPTED_DIFF_RE = re.compile(r"iteration_(\d+)_attempted\.diff")
ITER_BRANCH_RE = re.compile(r"codeevolver-\d+-iter(\d+)")
SHOW_REDIRECT_RE = re.compile(r"git show\s+(\S+?):(\S+)\s*>\s*(\S+)")


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
                if isinstance(cmd, str):
                    cmds.append(cmd)
    return cmds


def targets_program(path: str, files: set[str]) -> bool:
    p = path.replace("/workspace/", "")
    p = p[2:] if p.startswith("./") else p
    return p in (".", "") or any(p == f or f.endswith("/" + p) for f in files)


def copy_ops(cmd: str, files: set[str]) -> list[tuple[str, str | int]]:
    """(kind, ref or iteration) for each copy of other code INTO program files."""
    ops, temp = [], {}
    for seg in re.split(r"&&|\|\||;|\n", cmd):
        seg = seg.strip()
        try:
            toks = shlex.split(seg)
        except ValueError:
            toks = seg.split()
        if len(toks) >= 2 and toks[0] == "git":
            verb, rest = toks[1], toks[2:]
            if verb == "checkout" and "--" in rest:
                k = rest.index("--")
                refs = [t for t in rest[:k] if not t.startswith("-")]
                if refs and any(targets_program(f, files) for f in rest[k + 1:]):
                    ops.append(("checkout", refs[0]))
            elif verb == "restore":
                for k, t in enumerate(rest):
                    if t.startswith("--source"):
                        ref = t.split("=", 1)[1] if "=" in t else (rest[k + 1] if k + 1 < len(rest) else None)
                        if ref:
                            ops.append(("restore", ref))
            elif verb == "show" and ">" in seg:
                m = SHOW_REDIRECT_RE.search(seg)
                if m and targets_program(m.group(3), files):
                    ops.append(("show_redirect", m.group(1)))
                elif m:
                    temp[m.group(3)] = m.group(1)
            elif verb == "apply" and "--check" not in rest:
                m = ATTEMPTED_DIFF_RE.search(seg)
                if m:
                    ops.append(("apply_attempted_diff", int(m.group(1))))
            elif verb in ("cherry-pick", "merge"):
                refs = [t for t in rest if not t.startswith("-")]
                if refs:
                    ops.append((verb, refs[0]))
        elif len(toks) >= 3 and toks[0] in ("cp", "mv") and targets_program(toks[-1], files):
            if toks[-2] in temp:
                ops.append(("show_temp_copy", temp[toks[-2]]))
    return ops


def ref_iteration(repo: Path, ref: str | int, closes: dict[int, str]) -> int | None:
    if isinstance(ref, int):
        return ref
    m = ITER_BRANCH_RE.search(ref)
    if m:
        return int(m.group(1))
    sha = git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()
    if not sha:
        return None
    for it, c in sorted(closes.items()):
        if c == sha:
            return it
    for it, c in sorted(closes.items()):  # oldest close containing the commit
        if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", sha, c],
                          capture_output=True).returncode == 0:
            return it
    return None


def git_crosscheck(run_dir: Path, repo: Path, recs: dict[int, dict], seed_sha: str, seed_tree: str,
                   files: set[str], lines: dict[int, set], iters: list[dict]) -> list[dict]:
    closes = {0: seed_sha, **{i: r["close_sha"] for i, r in recs.items() if i and r.get("close_sha")}}
    tree_of = {0: seed_tree, **{i: r.get("close_tree") for i, r in recs.items() if i}}
    all_parent = {i: r.get("selected_parent_iteration") for i, r in recs.items() if i}
    flagged = {x["iteration"] for x in iters if x["transplant"]}
    cache = dict(lines)

    def program(i: int) -> set[str]:
        if i not in cache:
            t = tree_of.get(i)
            cache[i] = set().union(*(norm_lines(git(repo, "show", f"{t}:{f}")) for f in sorted(files))) if t else set()
        return cache[i]

    rows = []
    for i in sorted(k for k in recs if k > 0):
        lineage, p = {0, all_parent.get(i) or 0}, all_parent.get(i)
        while p:
            lineage.add(p)
            p = all_parent.get(p)
        hits = glob.glob(str(run_dir / f".codeevolver/architect_traces/*/iteration_{i}_architect_trace"))
        sources: dict[int, set[str]] = {}
        for cmd in bash_commands(Path(hits[0])) if hits else []:
            for kind, ref in copy_ops(cmd, files):
                src = ref_iteration(repo, ref, closes)
                if src is not None and src != i and src not in lineage:
                    sources.setdefault(src, set()).add(kind)
        content = i in flagged
        if not sources and not content:
            continue
        row = dict(iteration=i, has_trace=bool(hits), content_flag=content,
                   git_sources=sorted(sources), git_ops=sorted({k for v in sources.values() for k in v}))
        landed = bool(recs[i].get("changes_landed") and recs[i].get("close_tree"))
        if content:
            row["class"] = "flagged_and_git" if sources else "content_only"
        elif not landed or recs[i].get("constraint_violation"):
            row["class"] = "git_no_change"
        else:
            child, par = program(i), program(all_parent.get(i) or 0)
            src = max(sources, key=lambda s: jaccard(child, program(s)))
            sim_src, sim_par = jaccard(child, program(src)), jaccard(child, par)
            row.update(source=src, sim_source=round(sim_src, 4), sim_parent=round(sim_par, 4),
                       source_kept=round(len(child & program(src)) / max(1, len(program(src))), 4))
            row["class"] = ("git_parent_similar" if sim_src >= SIM_THRESHOLD
                            else "git_edited_after" if sim_src >= EDITED_FLOOR else "git_not_retained")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# per-run analysis


def load_population(run_dir: Path) -> dict[int, dict]:
    hits = glob.glob(str(run_dir / ".codeevolver/history/*/population.json"))
    if not hits:
        return {}
    cands = json.loads(Path(hits[0]).read_text()).get("candidates", {})
    return {int(k): v for k, v in cands.items()}


def noise_threshold(recs: dict[int, dict], n_comp: int | None) -> tuple[float | None, dict]:
    ve = next((r["variance_estimate"] for r in recs.values() if r.get("variance_estimate")), None)
    if not ve or not n_comp or ve.get("run_to_run_std") is None:
        return None, {}
    n_train = ve.get("n_rows") or ve.get("subsample_size")
    sigma_comp = ve["run_to_run_std"] * math.sqrt(n_train / n_comp)
    return 2 * math.sqrt(2) * sigma_comp, dict(run_to_run_std=ve["run_to_run_std"], n_train=n_train,
                                                n_comparison=n_comp, sigma_comparison=sigma_comp)


def analyze_run(run_dir: Path, cache: Path) -> dict:
    m = RUN_RE.search(run_dir.name)
    recs = {int(k): v for k, v in
            json.loads((run_dir / "history.json").read_text())["iteration_history"].items()}
    repo = open_mirror(run_dir, cache)
    seed_sha = next(r["parent_sha"] for r in recs.values()
                    if r.get("selected_parent_iteration") == 0 and r.get("parent_sha"))
    trees = {0: git(repo, "rev-parse", f"{seed_sha}^{{tree}}").strip()}
    for i, r in recs.items():
        if i and r.get("close_tree") and r.get("changes_landed") and not r.get("constraint_violation"):
            trees[i] = r["close_tree"]
    files = set()
    for i, t in trees.items():
        if i:
            files |= {f for f in git(repo, "diff", "--name-only", trees[0], t).split() if is_program_file(f)}
    lines = {i: set().union(*(norm_lines(git(repo, "show", f"{t}:{f}")) for f in sorted(files)))
             for i, t in trees.items()}
    parent = {i: recs[i].get("selected_parent_iteration") for i in trees if i}

    def ancestors(i: int) -> set[int]:
        out, p = {0}, parent.get(i)
        while p:
            out.add(p)
            p = parent.get(p)
        return out

    pop = load_population(run_dir)
    n_comp = next((len(c["row_scores"]) for c in pop.values() if c.get("row_scores")), None)
    threshold, noise = noise_threshold(recs, n_comp)

    iters = []
    for i in sorted(trees):
        p = parent.get(i)
        if not i or p not in lines:
            continue
        anc = ancestors(i)
        off = [j for j in trees if 0 < j < i and j not in anc]
        if not off:
            continue
        lineage_lines = set().union(*(lines[a] for a in anc if a in lines))
        added = lines[i] - lineage_lines
        off_pool = set().union(*(lines[j] - lineage_lines for j in off))
        sims = {j: jaccard(lines[i], lines[j]) for j in trees if j < i}
        src = max(off, key=sims.get)
        transplant = sims[src] >= SIM_THRESHOLD and sims[src] > sims[p] + SIM_MARGIN
        distinctive = lines[p] - lines[src]
        row = dict(iteration=i, parent=p, selection_policy=recs[i].get("selection_policy"),
                   n_off_lineage=len(off), added_lines=len(added),
                   copied_frac=round(len(added & off_pool) / len(added), 4) if added else 0.0,
                   sim_parent=round(sims[p], 4), source=src, sim_source=round(sims[src], 4),
                   transplant=transplant, source_is_previous=src == i - 1,
                   parent_distinctive_lines=len(distinctive),
                   parent_kept=(round(len(distinctive & lines[i]) / len(distinctive), 4)
                                if len(distinctive) >= MIN_DISTINCTIVE else None))
        if transplant:
            cs = (pop.get(src) or {}).get("comparison_set_score")
            cp = (pop.get(p) or {}).get("comparison_set_score")
            gap = cs - cp if cs is not None and cp is not None else None
            row.update(source_comparison=cs, parent_comparison=cp,
                       gap=round(gap, 4) if gap is not None else None, gap_threshold=threshold,
                       gap_class=("unscored" if gap is None else "source_not_better" if gap <= 0
                                  else "large" if threshold is not None and gap > threshold
                                  else "within_noise"))
        iters.append(row)

    # Archive integrity: the population only grows, so every admitted
    # candidate (and every transplant's parent) must still be in it.
    admitted = {i for i, r in recs.items() if r.get("candidate_admitted")} | {0}
    tx = [x for x in iters if x["transplant"]]
    archive = dict(
        admitted=len(admitted),
        admitted_missing=sorted(admitted - set(pop)),
        transplant_parents_missing=sorted({x["parent"] for x in tx if x["parent"] not in pop}),
        transplant_parents_resampled=sum(
            any(r.get("selected_parent_iteration") == x["parent"] for j, r in recs.items() if j > x["iteration"])
            for x in tx),
    )
    crosscheck = git_crosscheck(run_dir, repo, recs, seed_sha, trees[0], files, lines, iters)
    return dict(run=run_dir.name, bench=m["bench"], seed=int(m["seed"]), program_files=sorted(files),
                noise=noise, archive=archive, iterations_run=sorted(i for i in recs if i > 0),
                iterations=iters, git_crosscheck=crosscheck)


# ---------------------------------------------------------------------------
# output


def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)


def pct(x: float | None) -> str:
    return "" if x is None else f"{x:.1%}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="collected run dirs (default: every pareto_evolution -redo run)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="output dir for transplant.json + bundle mirrors")
    ap.add_argument("--results-dir", default=str(REPO_ROOT / "results"), help="where the CSV tables go")
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.runs] if args.runs else \
        [Path(p) for pat in DEFAULT_RUNS for p in sorted(glob.glob(os.path.expanduser(pat)))]
    out = Path(os.path.expanduser(args.out))
    out.mkdir(parents=True, exist_ok=True)
    results = Path(os.path.expanduser(args.results_dir))
    results.mkdir(parents=True, exist_ok=True)

    runs = []
    for rd in run_dirs:
        if not (rd / "history.json").exists() or not (rd / "repo.bundle").exists():
            print(f"skip {rd.name}: missing history.json or repo.bundle", file=sys.stderr)
            continue
        runs.append(analyze_run(rd, out))

    # benchmark x seed table
    by_bench: dict[str, dict[int, dict]] = {}
    for r in runs:
        n = len(r["iterations_run"])
        t = sum(x["transplant"] for x in r["iterations"])
        by_bench.setdefault(r["bench"], {})[r["seed"]] = dict(n=n, t=t, rate=t / n if n else None)
    seeds = sorted({s for b in by_bench.values() for s in b})
    rate_rows, bench_means, run_rates = [], {}, []
    print(f"{'benchmark':12s} " + " ".join(f"{'s' + str(s):>15s}" for s in seeds) + f" {'mean':>8s}")
    for b in sorted(by_bench):
        cells, rates = [], []
        for s in seeds:
            c = by_bench[b].get(s)
            if c is None or c["rate"] is None:
                cells.append(("", "", ""))
                continue
            cells.append((c["t"], c["n"], pct(c["rate"])))
            rates.append(c["rate"])
            run_rates.append(c["rate"])
        bench_means[b] = st.mean(rates) if rates else None
        rate_rows.append([b] + [v for c in cells for v in c] + [pct(bench_means[b])])
        print(f"{b:12s} " + " ".join(f"{f'{c[0]}/{c[1]} ({c[2]})' if c[1] != '' else '':>15s}" for c in cells)
              + f" {pct(bench_means[b]):>8s}")
    q = st.quantiles(run_rates, n=4) if len(run_rates) >= 2 else [None, None, None]
    summary = dict(median_run=st.median(run_rates), iqr=(q[0], q[2]),
                   median_bench_mean=st.median([v for v in bench_means.values() if v is not None]),
                   pooled_transplants=sum(x["transplant"] for r in runs for x in r["iterations"]),
                   pooled_iterations=sum(len(r["iterations_run"]) for r in runs),
                   pooled_eligible=sum(len(r["iterations"]) for r in runs), n_runs=len(run_rates))
    rate_rows.append(["MEDIAN RUN"] + [""] * (3 * len(seeds)) + [pct(summary["median_run"])])
    rate_rows.append(["IQR RUNS"] + [""] * (3 * len(seeds)) + [f"{pct(q[0])}-{pct(q[2])}"])
    rate_rows.append(["POOLED"] + [""] * (3 * len(seeds)) +
                     [f"{summary['pooled_transplants']}/{summary['pooled_iterations']}"])
    write_csv(results / "transplant_rates.csv",
              ["benchmark"] + [f"s{s}_{k}" for s in seeds for k in ("transplants", "iterations", "rate")]
              + ["bench_mean_rate"], rate_rows)
    print(f"\nmedian run {pct(summary['median_run'])} (IQR {pct(q[0])}-{pct(q[2])}, {len(run_rates)} runs); "
          f"median of benchmark means {pct(summary['median_bench_mean'])}; "
          f"pooled {summary['pooled_transplants']}/{summary['pooled_iterations']} "
          f"(eligible-only {summary['pooled_transplants']}/{summary['pooled_eligible']})")

    # per-benchmark table: transplant rate over the whole run and over the
    # first 10 / 20 iterations (pooled over seeds), plus archive integrity
    def window(rs: list[dict], limit: int | None) -> tuple[int, int]:
        t = sum(x["transplant"] for r in rs for x in r["iterations"] if limit is None or x["iteration"] <= limit)
        n = sum(1 for r in rs for i in r["iterations_run"] if limit is None or i <= limit)
        return t, n

    def cell(t: int, n: int) -> str:
        return f"{t}/{n} ({t / n:.1%})" if n else ""

    def table_rows(groups: list[tuple[str, list[dict]]]) -> list[list]:
        rows = []
        for b, rs in groups:
            windows = [window(rs, lim) for lim in (None, 10, 20)]
            arch = [r["archive"] for r in rs]
            n_tx = windows[0][0]
            rows.append([
                b, len(rs), max(i for r in rs for i in r["iterations_run"]),
                *(cell(*w) for w in windows),
                cell(n_tx, sum(len(r["iterations"]) for r in rs)),
                f"{sum(len(a['admitted_missing']) for a in arch)}/{sum(a['admitted'] for a in arch)}",
                f"{sum(len(a['transplant_parents_missing']) for a in arch)}/{n_tx}",
                f"{sum(a['transplant_parents_resampled'] for a in arch)}/{n_tx}",
            ])
        return rows

    header = ["benchmark", "runs", "max_iteration", "transplant_rate_all", "transplant_rate_iter_1_10",
              "transplant_rate_iter_1_20", "transplant_rate_eligible_only", "admitted_missing_from_final_population",
              "transplant_parents_missing_from_final_population", "transplant_parents_resampled_later"]
    bench_rows = table_rows([(b, [r for r in runs if r["bench"] == b]) for b in sorted(by_bench)] + [("ALL", runs)])
    write_csv(results / "transplant_by_benchmark.csv", header, bench_rows)
    run_order = sorted(runs, key=lambda r: (r["bench"], r["seed"]))
    write_csv(results / "transplant_by_run.csv", ["run"] + header[2:],
              [row[:1] + row[2:] for row in table_rows([(r["run"], [r]) for r in run_order] + [("ALL", runs)])])
    print()
    for row in [header] + bench_rows:
        print("  ".join(f"{str(v):>16s}" if k else f"{str(v):12s}" for k, v in enumerate(row)))

    # per-transplant gap table
    tx = [(r, x) for r in runs for x in r["iterations"] if x["transplant"]]
    write_csv(results / "transplants.csv",
              ["run", "iteration", "parent", "source", "source_is_previous", "sim_parent", "sim_source",
               "parent_kept", "parent_comparison", "source_comparison", "gap", "gap_threshold", "gap_class"],
              [[r["run"], x["iteration"], x["parent"], x["source"], x["source_is_previous"], x["sim_parent"],
                x["sim_source"], x["parent_kept"], x["parent_comparison"], x["source_comparison"], x["gap"],
                None if x["gap_threshold"] is None else round(x["gap_threshold"], 4), x["gap_class"]]
               for r, x in tx])
    classes = ("large", "within_noise", "source_not_better", "unscored")
    count = {c: sum(x["gap_class"] == c for _, x in tx) for c in classes}
    print(f"\ntransplants: {len(tx)}; gap source-parent on the comparison set: "
          + ", ".join(f"{c} {count[c]} ({count[c] / max(1, len(tx)):.0%})" for c in classes))
    per_bench = {}
    for r, x in tx:
        per_bench.setdefault(r["bench"], []).append(x["gap_class"])
    for b, cl in sorted(per_bench.items()):
        print(f"  {b:12s} n={len(cl):2d} " + " ".join(f"{c}={cl.count(c)}" for c in classes))
    summary["gap_classes"] = count

    # git cross-check of the content rule
    xc = [(r, x) for r in runs for x in r["git_crosscheck"]]
    write_csv(results / "transplant_git_crossref.csv",
              ["run", "iteration", "class", "content_flag", "git_sources", "git_ops", "source", "sim_source",
               "sim_parent", "source_kept", "has_trace"],
              [[r["run"], x["iteration"], x["class"], x["content_flag"], " ".join(map(str, x["git_sources"])),
                " ".join(x["git_ops"]), x.get("source"), x.get("sim_source"), x.get("sim_parent"),
                x.get("source_kept"), x["has_trace"]] for r, x in xc])
    order = ("flagged_and_git", "content_only", "git_no_change", "git_parent_similar", "git_edited_after",
             "git_not_retained")
    classes = {c: sum(x["class"] == c for _, x in xc) for c in order}
    n_iter = summary["pooled_iterations"]
    missed = classes["git_edited_after"]
    summary["git_crosscheck"] = dict(classes, upper_bound_transplants=summary["pooled_transplants"] + missed)
    print("\ngit cross-check: " + ", ".join(f"{c} {n}" for c, n in classes.items()))
    print(f"  with a git copy op: {sum(bool(x['git_sources']) for _, x in xc)}; "
          f"rate with git_edited_after added (upper bound): "
          f"{summary['pooled_transplants'] + missed}/{n_iter} ({(summary['pooled_transplants'] + missed) / n_iter:.1%})")
    for r, x in xc:
        if x["class"] in ("git_parent_similar", "git_edited_after", "git_not_retained"):
            print(f"    {r['run']} it{x['iteration']} {x['class']}: src {x['source']} sim_src {x['sim_source']} "
                  f"sim_parent {x['sim_parent']} source_kept {x['source_kept']} ({' '.join(x['git_ops'])})")

    (out / "transplant.json").write_text(json.dumps({"summary": summary, "runs": runs}, indent=1))
    print(f"\nwrote transplant_rates.csv, transplant_by_benchmark.csv, transplant_by_run.csv, transplants.csv, "
          f"transplant_git_crossref.csv in {results}; {out / 'transplant.json'}")


if __name__ == "__main__":
    main()
