#!/usr/bin/env python3
"""Build the raw CSV tables behind the main results figure from the collected runs.

Strictly read-only against the run data: every file under the runs root is
opened for reading only, and the script refuses an --out-dir that lives
inside the runs root.

Runs are read from `--runs-root` (default `$CE_RUNS_DIR`, else `~/ce-runs`).
AppWorld and ALFWorld runs may instead live in a second root
(`--rk-runs-root`); when one is given, a run directory whose benchmark belongs
to the other root is ignored with a warning, so a stray copy never
double-counts.

Per run directory `<root>/<arm>-<bench>-s<seed>[-<label>]/` we read
`history.json` (the mirrored job history; its `summary` block carries
`seed_test_score(s)` / `final_test_scores`). If a collected run has a
`status.json` but no `history.json`, `status.json` supplies the same fields.
A run that has not completed (no history file yet, or status != completed)
contributes blanks.

Replacements. Some cells were deliberately re-run under a `-redo` suffix and
the original run directory is VOID: it is kept on disk as a record only and
is never read for a score, even when the redo has not finished (the cell is
blank instead). The full replacement table:

  cell(s)                         reads from                    replaces (void)
  ------------------------------  ----------------------------  ----------------------------
  pareto_evolution, every         pareto_evolution-<b>-s<n>-redo  pareto_evolution-<b>-s<n>
    benchmark, every seed         (re-run with the corrected pruning rule + freeform
                                  prompt; the unsuffixed pareto_evolution runs are void)
  ee / hover / seed 3             ee-hover-s3-redo              ee-hover-s3
                                  (re-run after the original failed)

That is the whole table (`ARM_REPLACEMENTS` + `CELL_REPLACEMENTS` below). A
labelled directory that is not in it is ignored with a warning, never used.

Outputs (CSV):

  final_scores.csv  one row per (benchmark, arm); 9 score columns =
                    3 held-out test scores x 3 rounds (seeds 1-3):
                    r1_test1 r1_test2 r1_test3 r2_test1 ... r3_test3
  seed_scores.csv   one row per benchmark; 15 columns = 5 arms x 3 rounds:
                    ee_r1 ee_r2 ee_r3 freeform_r1 ... asa_r3
                    (the seed program's held-out test score for that run)
  replacements.csv  one row per replaced cell: which run directory the cell
                    reads from, which void original it replaces, whether each
                    exists on disk, and the replacement's status. The same
                    table is printed first on stdout.

Ablations (`--label L`). A campaign that re-runs arms under a changed
variable (e.g. a coder-model ablation, run_suffix `-g41m`) is
read with `--label L`: EVERY cell then reads only `<arm>-<bench>-s<n>-L`, the
replacement table is bypassed, and the output goes to results/ablation-L/
unless --out-dir says otherwise. It adds

  ablation_2x2.csv  per (benchmark, model, arm): the per-seed mean held-out
                    score, mean and SD over seeds; per (benchmark, model) the
                    pareto_evolution - freeform gap; and the paired-by-seed
                    interaction (gap under L minus gap under the baseline
                    cells), with the baseline read exactly as without --label.

Usage:
    python scripts/results_tables.py                      # -> results/
    python scripts/results_tables.py --label g41m --benchmarks hover,phantomwiki --arms freeform,pareto_evolution
    python scripts/results_tables.py --out-dir ~/ce-results
    python scripts/results_tables.py --runs-root ~/ce-runs --rk-runs-root ~/ce-runs-appworld
    python scripts/results_tables.py --benchmarks hover,ragqa
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

# Matrix order (orchestrator/matrix.main.yaml, benchmarks.yaml).
ARMS = ["ee", "freeform", "pareto_evolution", "simple_loop", "asa"]
BENCHMARKS = ["hover", "phantomwiki", "ragqa", "arcagi2", "paperreview", "appworld", "alfworld"]
# Benchmarks whose runs may be collected into a separate root (--rk-runs-root).
RK_BENCHMARKS = {"appworld", "alfworld"}
DEFAULT_RUNS_ROOT = Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()
ROUNDS = [1, 2, 3]
TESTS_PER_RUN = 3

# `<arm>-<bench>-s<seed>[-<label>]`: the optional label is a matrix `run_suffix`
# (orchestrator/controller.py::build_runs), used when a cell is deliberately
# re-run beside its kept original (e.g. the pareto_evolution `-redo` cells).
RUN_ID_RE = re.compile(
    r"^(?P<arm>[a-z][a-z0-9_]*)-(?P<bench>[a-z][a-z0-9]*)-s(?P<seed>\d+)"
    r"(?:-(?P<label>[a-z0-9]+))?$"
)

# Replacement table. A cell listed here reads ONLY from the run directory
# carrying the given label; the unlabelled original is void and never read,
# not even as a fallback while the redo is still running.
#   ARM_REPLACEMENTS  : every cell of the arm (all benchmarks, all seeds).
#   CELL_REPLACEMENTS : one (bench, arm, seed) cell.
ARM_REPLACEMENTS: dict[str, str] = {
    # Re-run with the corrected pruning rule + freeform prompt; every
    # unsuffixed pareto_evolution run is void.
    "pareto_evolution": "redo",
}
CELL_REPLACEMENTS: dict[tuple[str, str, int], str] = {
    # One-cell re-run: the original failed.
    ("hover", "ee", 3): "redo",
}


def required_label(bench: str, arm: str, seed: int) -> str | None:
    """The label a run directory must carry to be read for this cell, or None
    when the cell reads its plain `<arm>-<bench>-s<seed>` directory."""
    return CELL_REPLACEMENTS.get((bench, arm, seed)) or ARM_REPLACEMENTS.get(arm)


def replacement_run_id(bench: str, arm: str, seed: int) -> str | None:
    label = required_label(bench, arm, seed)
    return f"{arm}-{bench}-s{seed}-{label}" if label else None


def _read_json(path: Path) -> dict | None:
    try:
        with path.open("r") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"[WARN] {path}: unreadable ({exc})", file=sys.stderr)
        return None


def _fmt(x) -> str:
    if x is None:
        return ""
    return repr(float(x))


def load_run(run_dir: Path) -> dict | None:
    """Return {'arm','bench','seed','status','seed_test_score','final_test_scores'}
    for one run directory, or None if it is not a run directory."""
    m = RUN_ID_RE.match(run_dir.name)
    if not m or not run_dir.is_dir():
        return None
    arm, bench, seed = m.group("arm"), m.group("bench"), int(m.group("seed"))

    summary: dict = {}
    status = "in_progress"
    source = None
    history = run_dir / "history.json"
    status_file = run_dir / "status.json"
    if history.exists():
        blob = _read_json(history) or {}
        summary = blob.get("summary") or {}
        status = str(blob.get("status") or summary.get("status") or "unknown")
        source = history.name
    elif status_file.exists():
        blob = _read_json(status_file) or {}
        summary = blob
        status = str(blob.get("status") or "unknown")
        source = status_file.name

    seed_score = summary.get("seed_test_score")
    if seed_score is None:
        seeds = summary.get("seed_test_scores") or []
        seed_score = seeds[0] if seeds else None

    finals = list(summary.get("final_test_scores") or [])
    if not finals and summary.get("final_test_score") is not None:
        finals = [summary["final_test_score"]]

    completed = status == "completed"
    return {
        "run_id": run_dir.name,
        "arm": arm,
        "bench": bench,
        "seed": seed,
        "label": m.group("label"),
        "status": status,
        "source": source,
        # Both tables only report completed runs; an unfinished run is blank.
        "seed_test_score": seed_score if completed else None,
        "final_test_scores": finals if completed else [],
    }


def scan_runs(roots: dict[Path, set[str] | None], label: str | None = None) -> dict[tuple[str, str, int], dict]:
    """Scan each root; `roots` maps a directory to the benchmarks it owns
    (None = every benchmark not owned by another root).

    A cell in the replacement table is read only from its labelled run
    directory; its unlabelled original is skipped as void. A labelled
    directory for a cell that is NOT in the table is skipped with a warning
    (add it to the table if it is a real replacement).

    With `label`, every cell reads only its `-<label>` directory instead
    (an ablation campaign; the replacement table does not apply)."""
    owned = {b for bs in roots.values() if bs for b in bs}
    runs: dict[tuple[str, str, int], dict] = {}
    for root, benches in roots.items():
        if not root.is_dir():
            print(f"[WARN] runs root not found, skipping: {root}", file=sys.stderr)
            continue
        for d in sorted(root.iterdir()):
            r = load_run(d)
            if r is None:
                continue
            belongs = r["bench"] in benches if benches else r["bench"] not in owned
            if not belongs:
                print(f"[WARN] ignoring {root / d.name}: {r['bench']} runs belong in another root", file=sys.stderr)
                continue
            key = (r["bench"], r["arm"], r["seed"])
            want = label or required_label(*key)
            if r["label"] != want:
                if label:
                    continue  # another campaign's run; not this ablation
                if want and not r["label"]:
                    print(f"[INFO] skipping void original {r['run_id']}: cell reads {replacement_run_id(*key)}",
                          file=sys.stderr)
                elif want:
                    print(f"[WARN] ignoring {r['run_id']}: cell reads label '{want}', not '{r['label']}'",
                          file=sys.stderr)
                else:
                    print(f"[WARN] ignoring labelled run {r['run_id']}: not in the replacement table",
                          file=sys.stderr)
                continue
            prev = runs.get(key)
            if prev is not None:
                print(f"[WARN] duplicate run {r['run_id']} in {root}; keeping {prev['run_id']}", file=sys.stderr)
                continue
            r["root"] = root
            runs[key] = r
    return runs


def replacement_rows(roots: list[Path], runs, benchmarks, arms, rounds) -> tuple[list[str], list[list[str]]]:
    """One row per replaced cell inside the requested matrix: what it reads
    from, what that replaces, and whether each directory exists on disk."""
    header = ["benchmark", "arm", "seed", "reads_from", "replaces", "replacement_on_disk",
              "original_on_disk", "replacement_status"]
    rows = []
    for bench in benchmarks:
        for arm in arms:
            for seed in rounds:
                redo_id = replacement_run_id(bench, arm, seed)
                if redo_id is None:
                    continue
                orig_id = f"{arm}-{bench}-s{seed}"
                redo_on_disk = any((root / redo_id).is_dir() for root in roots)
                orig_on_disk = any((root / orig_id).is_dir() for root in roots)
                r = runs.get((bench, arm, seed))
                status = r["status"] if r else "missing"
                rows.append([bench, arm, str(seed), redo_id, orig_id,
                             "yes" if redo_on_disk else "no",
                             "yes" if orig_on_disk else "no", status])
    return header, rows


def print_table(header: list[str], rows: list[list[str]]) -> None:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(header)]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)))


def final_scores_rows(runs, benchmarks, arms, rounds) -> tuple[list[str], list[list[str]]]:
    header = ["benchmark", "arm"] + [
        f"r{r}_test{t}" for r in rounds for t in range(1, TESTS_PER_RUN + 1)
    ]
    rows = []
    for bench in benchmarks:
        for arm in arms:
            row = [bench, arm]
            for r in rounds:
                scores = (runs.get((bench, arm, r)) or {}).get("final_test_scores", [])
                if len(scores) > TESTS_PER_RUN:
                    print(
                        f"[WARN] {arm}-{bench}-s{r}: {len(scores)} final test scores, keeping first {TESTS_PER_RUN}",
                        file=sys.stderr,
                    )
                padded = list(scores[:TESTS_PER_RUN]) + [None] * (TESTS_PER_RUN - len(scores[:TESTS_PER_RUN]))
                row.extend(_fmt(s) for s in padded)
            rows.append(row)
    return header, rows


def seed_scores_rows(runs, benchmarks, arms, rounds) -> tuple[list[str], list[list[str]]]:
    header = ["benchmark"] + [f"{arm}_r{r}" for arm in arms for r in rounds]
    rows = []
    for bench in benchmarks:
        row = [bench]
        for arm in arms:
            for r in rounds:
                row.append(_fmt((runs.get((bench, arm, r)) or {}).get("seed_test_score")))
        rows.append(row)
    return header, rows


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def _sd(xs: list[float]) -> float | None:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5


def ablation_rows(base, abl, label, benchmarks, arms, rounds) -> tuple[list[str], list[list[str]]]:
    """The model x arm 2x2 per benchmark. A run's score is the mean of its
    held-out draws; SD is across seeds. Gap = pareto_evolution - freeform per
    seed; the interaction pairs seeds: (gap under `label`) - (gap under the
    baseline cells), so it is only computed on seeds complete in all four."""
    def run_mean(runs, bench, arm, seed):
        return _mean((runs.get((bench, arm, seed)) or {}).get("final_test_scores", []))

    header = ["benchmark", "model", "arm"] + [f"s{r}" for r in rounds] + ["mean", "sd", "n"]
    rows = []
    evo, ff = "pareto_evolution", "freeform"
    for bench in benchmarks:
        gaps: dict[str, dict[int, float]] = {}
        for model, runs in (("baseline", base), (label, abl)):
            for arm in arms:
                per_seed = {r: run_mean(runs, bench, arm, r) for r in rounds}
                vals = [v for v in per_seed.values() if v is not None]
                rows.append([bench, model, arm] + [_fmt(per_seed[r]) for r in rounds]
                            + [_fmt(_mean(vals)), _fmt(_sd(vals)), str(len(vals))])
            if evo in arms and ff in arms:
                g = {r: run_mean(runs, bench, evo, r) - run_mean(runs, bench, ff, r)
                     for r in rounds
                     if run_mean(runs, bench, evo, r) is not None and run_mean(runs, bench, ff, r) is not None}
                gaps[model] = g
                vals = list(g.values())
                rows.append([bench, model, f"gap({evo}-{ff})"] + [_fmt(g.get(r)) for r in rounds]
                            + [_fmt(_mean(vals)), _fmt(_sd(vals)), str(len(vals))])
        if len(gaps) == 2:
            inter = {r: gaps[label][r] - gaps["baseline"][r]
                     for r in rounds if r in gaps[label] and r in gaps["baseline"]}
            vals = list(inter.values())
            rows.append([bench, f"{label}-baseline", "interaction(gap diff, paired)"]
                        + [_fmt(inter.get(r)) for r in rounds]
                        + [_fmt(_mean(vals)), _fmt(_sd(vals)), str(len(vals))])
    return header, rows


def write_csv(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", type=Path, default=DEFAULT_RUNS_ROOT,
                    help="root for every benchmark except the RK ones")
    ap.add_argument("--rk-runs-root", type=Path, default=None,
                    help=f"separate root for {', '.join(sorted(RK_BENCHMARKS))} runs "
                         "(default: the same root as every other benchmark)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="default results/, or results/ablation-<label>/ with --label")
    ap.add_argument("--label", default=None,
                    help="ablation mode: read only <arm>-<bench>-s<n>-<label> runs (see module doc)")
    ap.add_argument("--benchmarks", default=",".join(BENCHMARKS), help="comma-separated row order")
    ap.add_argument("--arms", default=",".join(ARMS), help="comma-separated arm order")
    ap.add_argument("--rounds", default=",".join(map(str, ROUNDS)), help="comma-separated seeds/rounds")
    ap.add_argument("--only-present", action="store_true", help="drop benchmarks with no run directory at all")
    args = ap.parse_args(argv)

    runs_root = args.runs_root.expanduser().resolve()
    rk_root = (args.rk_runs_root or args.runs_root).expanduser().resolve()
    results_dir = Path(__file__).resolve().parent.parent / "results"
    if args.out_dir is None:
        args.out_dir = results_dir / f"ablation-{args.label}" if args.label else results_dir
    out_dir = args.out_dir.expanduser().resolve()
    if not runs_root.is_dir() and not rk_root.is_dir():
        print(f"no runs root found: {runs_root}, {rk_root}", file=sys.stderr)
        return 2
    for root in (runs_root, rk_root):
        if out_dir == root or root in out_dir.parents:
            print(f"refusing to write inside a runs root: {out_dir}", file=sys.stderr)
            return 2

    benchmarks = [b for b in args.benchmarks.split(",") if b]
    arms = [a for a in args.arms.split(",") if a]
    rounds = [int(r) for r in args.rounds.split(",") if r]

    roots = {runs_root: None} if rk_root == runs_root else {runs_root: None, rk_root: set(RK_BENCHMARKS)}
    runs = scan_runs(roots, label=args.label)
    for (bench, arm, seed), r in runs.items():
        if bench not in benchmarks or arm not in arms or seed not in rounds:
            print(f"[INFO] ignoring {r['run_id']} (outside the requested matrix)", file=sys.stderr)
    if args.only_present:
        present = {b for (b, _, _) in runs}
        benchmarks = [b for b in benchmarks if b in present]

    out_dir.mkdir(parents=True, exist_ok=True)
    rh, rr = replacement_rows(list(roots), runs, benchmarks, arms, rounds)
    repl_path = out_dir / "replacements.csv"
    write_csv(repl_path, rh, rr)
    print("replacements (void originals are never read, even when the redo is unfinished):")
    print_table(rh, rr)
    print()
    fh, fr = final_scores_rows(runs, benchmarks, arms, rounds)
    sh, sr = seed_scores_rows(runs, benchmarks, arms, rounds)
    final_path = out_dir / "final_scores.csv"
    seed_path = out_dir / "seed_scores.csv"
    write_csv(final_path, fh, fr)
    write_csv(seed_path, sh, sr)

    n_expected = len(benchmarks) * len(arms) * len(rounds)
    n_done = sum(
        1 for (b, a, s), r in runs.items()
        if b in benchmarks and a in arms and s in rounds and r["status"] == "completed"
    )
    n_seen = sum(1 for (b, a, s) in runs if b in benchmarks and a in arms and s in rounds)
    print(f"runs root      : {runs_root}")
    print(f"RK runs root   : {rk_root}  ({', '.join(sorted(RK_BENCHMARKS))})")
    print(f"run dirs found : {n_seen} / {n_expected} cells   completed: {n_done}")
    for (b, a, s), r in sorted(runs.items()):
        if b in benchmarks and a in arms and s in rounds and r["status"] != "completed":
            print(f"  blank (status={r['status']}): {r['run_id']}")
    print(f"wrote {repl_path}")
    print(f"wrote {final_path}")
    print(f"wrote {seed_path}")
    if args.label:
        base = scan_runs(roots)
        ah, ar = ablation_rows(base, runs, args.label, benchmarks, arms, rounds)
        abl_path = out_dir / "ablation_2x2.csv"
        write_csv(abl_path, ah, ar)
        print()
        print_table(ah, ar)
        print(f"wrote {abl_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
