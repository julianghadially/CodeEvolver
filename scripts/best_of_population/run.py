#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Best-of-population audit CLI (`scripts/best_of_population/README.md`).

    python scripts/best_of_population/run.py plan        --out R --bench B [--workers N]
    python scripts/best_of_population/run.py select-run  --out R --bench B          # selection/<bench>.json
    python scripts/best_of_population/run.py difficult   --out R --bench B          # difficult/<bench>.json
    python scripts/best_of_population/run.py materialize --out R --bench B          # candidates/, materialize/
    python scripts/best_of_population/run.py screen      --out R --bench B [--dry-run]   # -> portfolio/
    python scripts/best_of_population/run.py select      --out R --bench B [--dry-run]   # -> winners/
    python scripts/best_of_population/run.py confirm     --out R --bench B [--dry-run]
    python scripts/best_of_population/run.py report      --out R [--bench B]
    python scripts/best_of_population/run.py all         --out R --bench B --workers N
    python scripts/best_of_population/run.py aggregate   --inputs R1 R2 ... --out MERGED

Every stage is resumable: frozen artifacts (selection, difficult set,
candidates, portfolio, winners) are loaded when present and only recomputed
with `--refreeze`; eval stages consult the row store first and never convert
a cached observation into a new repetition. Eval stages are BILLABLE — they
run the client program with real provider keys.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("SANDBOX_MODE", "local")
os.environ.setdefault("APP_MODE", "local")

from scripts.best_of_population import planner as P  # noqa: E402
from scripts.best_of_population.candidates import load_candidates, materialize_bench  # noqa: E402
from scripts.best_of_population.catalog import candidates_from_run, RunRecord  # noqa: E402
from scripts.best_of_population.config import (  # noqa: E402
    DEFAULT_OUT, AuditConfig, CONFIG_FILENAME, arm_path, audit_paths, env_preamble, load_bench_spec,
)
from scripts.best_of_population.difficult import build_difficult, load_difficult, write_difficult  # noqa: E402
from scripts.best_of_population.environment import missing_provider_keys  # noqa: E402
from scripts.best_of_population.estimate import estimate_bench, format_estimate  # noqa: E402
from scripts.best_of_population.execute import cross_phase_guard, run_phase  # noqa: E402
from scripts.best_of_population.materialize import load_materialized  # noqa: E402
from scripts.best_of_population.portfolio import build_portfolio, load_portfolio, write_portfolio  # noqa: E402
from scripts.best_of_population.records import read_json  # noqa: E402
from scripts.best_of_population.report import (  # noqa: E402
    build_results, load_results, summary_rows, write_results, write_summary,
)
from scripts.best_of_population.select_run import load_selection, select_runs, selection_table, write_selection  # noqa: E402
from scripts.best_of_population.store import AuditStore  # noqa: E402
from scripts.best_of_population.winners import build_winners, load_winners, write_winners  # noqa: E402

EXIT_PREFLIGHT = 2
EXIT_INCOMPLETE = 3


def log(*args: object) -> None:
    print(*args, flush=True)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cfg(args: argparse.Namespace) -> AuditConfig:
    out = str(Path(args.out).expanduser())
    cfg = AuditConfig.load(out) if (Path(out) / CONFIG_FILENAME).exists() else AuditConfig(out=out)
    overrides = {k: getattr(args, k, None) for k in ("workers", "dotenv", "config_name", "reps_select", "reps_confirm",
                                                       "top_k", "threshold", "eval_timeout", "ce_home")}
    if getattr(args, "runs_root", None):
        overrides["runs_roots"] = [str(Path(r).expanduser()) for r in args.runs_root]
    if getattr(args, "arms", None):
        overrides["arms"] = [a.strip() for a in args.arms.split(",") if a.strip()]
    for flag in ("include_void_seed",):
        if getattr(args, flag, None):
            overrides[flag] = True
    cfg = cfg.with_overrides(**overrides)
    cfg.save()
    env_preamble(cfg)
    return cfg


def _spec(cfg: AuditConfig, bench: str):
    return load_bench_spec(bench, config_name=cfg.config_name)


def _store(cfg: AuditConfig, bench: str) -> AuditStore:
    p = audit_paths(cfg, bench)
    return AuditStore(p["rows"], p["evals"], p["attempts"])


def _arms_with_runs(sel: dict) -> list[str]:
    return [a for a, e in sel["arms"].items() if e.get("run_id")]


def _preflight(cfg: AuditConfig, bench: str, spec, sel: dict) -> int:
    from scripts.smoke_eval import preflight

    problems = preflight(bench, spec, "test")
    for arm in _arms_with_runs(sel):
        settings = read_json(Path(sel["arms"][arm]["run_dir"]) / "settings.json") or {}
        missing = missing_provider_keys(settings)
        if missing:
            problems.append(f"{arm}: provider keys the run had are missing from the environment: {missing}")
    if problems:
        for pr in problems:
            log(f"PREFLIGHT {bench}: {pr}")
        return EXIT_PREFLIGHT
    return 0


def _frozen(path: Path, refreeze: bool) -> bool:
    return path.exists() and not refreeze


def _not_applicable(diff: dict) -> bool:
    return bool(diff.get("not_applicable"))


# ---------------------------------------------------------------------------
# stages
# ---------------------------------------------------------------------------

def cmd_select_run(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    path = audit_paths(cfg, args.bench)["selection"]
    if _frozen(path, args.refreeze):
        sel = load_selection(cfg, args.bench)
        log(f"(frozen) {path}")
    else:
        sel = select_runs(cfg, args.bench, log=log)
        write_selection(cfg, args.bench, sel)
    log(selection_table(sel))
    return 0 if _arms_with_runs(sel) else 1


def cmd_difficult(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    path = audit_paths(cfg, args.bench)["difficult"]
    if _frozen(path, args.refreeze):
        d = load_difficult(cfg, args.bench)
        log(f"(frozen) {path}: |H| = {d.get('n_H')} ({d.get('label')}), {d.get('n_seed_observations')} seed observations")
    else:
        d = build_difficult(cfg, args.bench, log=log)
        write_difficult(cfg, args.bench, d)
    if _not_applicable(d):
        log(f"{args.bench}: NOT APPLICABLE — H is empty ({d.get('reason', 'no difficult rows')})")
    return 0


def cmd_materialize(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    sel, diff = load_selection(cfg, args.bench), load_difficult(cfg, args.bench)
    if _not_applicable(diff):
        log(f"{args.bench}: not applicable (empty H) — nothing to materialize")
        return 0
    spec = _spec(cfg, args.bench)
    explicit = None
    if args.candidates_file:
        explicit = json.loads(Path(args.candidates_file).read_text())
    mat = materialize_bench(cfg, args.bench, spec, sel, diff, explicit=explicit, log=log)
    log(f"materialized {len(mat['candidates'])} run(s), {len(mat['venvs'])} venv(s), {len(mat['worktrees'])} worktree slot(s)")
    return 0


def _eval_prep(args: argparse.Namespace):
    cfg = _cfg(args)
    sel, diff = load_selection(cfg, args.bench), load_difficult(cfg, args.bench)
    if _not_applicable(diff):
        log(f"{args.bench}: not applicable (empty H)")
        return cfg, sel, diff, None, None, None, 0
    spec = _spec(cfg, args.bench)
    mat = load_materialized(cfg, args.bench)
    store = _store(cfg, args.bench)
    rc = 0 if args.dry_run else _preflight(cfg, args.bench, spec, sel)
    return cfg, sel, diff, spec, mat, store, rc


def cmd_screen(args: argparse.Namespace) -> int:
    cfg, sel, diff, spec, mat, store, rc = _eval_prep(args)
    if spec is None or rc:
        return rc
    H = list(diff["H"])
    arms = [a for a in _arms_with_runs(sel) if not _frozen(arm_path(cfg, "portfolio", args.bench, a), args.refreeze)]
    if not arms:
        log("(frozen) every portfolio exists; nothing to screen")
        return 0
    docs = {a: load_candidates(cfg, args.bench, a) for a in arms}
    all_cands = [c for d in docs.values() for c in d["candidates"]]
    rep = run_phase(cfg, args.bench, spec, mat, lambda st: P.plan_screen(all_cands, H, st), store, P.PHASE_SCREEN,
                    dry_run=args.dry_run, frozen_testset_sha256=diff.get("testset_sha256"), log=log)
    if args.dry_run:
        return 0
    for a in arms:
        doc = build_portfolio(cfg, args.bench, a, docs[a], H, store, log=log)
        doc["phase_report"] = {"errors": rep.errors, "unresolved": rep.unresolved, "filled_rows": rep.filled_rows,
                               "still_failed": rep.still_failed, "identical_pairs": rep.identical_pairs}
        write_portfolio(cfg, args.bench, a, doc)
    return 0


def cmd_select(args: argparse.Namespace) -> int:
    cfg, sel, diff, spec, mat, store, rc = _eval_prep(args)
    if spec is None or rc:
        return rc
    H = list(diff["H"])
    arms = [a for a in _arms_with_runs(sel) if not _frozen(arm_path(cfg, "winners", args.bench, a), args.refreeze)]
    if not arms:
        log("(frozen) every winners file exists; nothing to select")
        return 0
    ports = {a: load_portfolio(cfg, args.bench, a) for a in arms}
    members = [c for p in ports.values() for c in p["portfolio"]]
    rep = run_phase(cfg, args.bench, spec, mat, lambda st: P.plan_select(members, H, st, cfg.reps_select), store,
                    P.PHASE_SELECT, dry_run=args.dry_run, frozen_testset_sha256=diff.get("testset_sha256"), log=log)
    if args.dry_run:
        return 0
    if rep.unresolved:
        log(f"{args.bench}: {len(rep.unresolved)} selection call(s) still missing rows — rerun `select` to resume")
        return EXIT_INCOMPLETE
    guard = cross_phase_guard(store, [(c["job_id"], c["tree"], P.PHASE_SCREEN, 0, P.PHASE_SELECT, 1) for c in members],
                              log=log)
    for a in arms:
        doc = build_winners(cfg, args.bench, a, ports[a], store, log=log)
        doc["cache_guard"] = {"within_phase_identical_pairs": rep.identical_pairs, "cross_phase": guard}
        doc["phase_report"] = {"errors": rep.errors, "filled_rows": rep.filled_rows, "still_failed": rep.still_failed}
        write_winners(cfg, args.bench, a, doc)
    return 0


def cmd_confirm(args: argparse.Namespace) -> int:
    cfg, sel, diff, spec, mat, store, rc = _eval_prep(args)
    if spec is None or rc:
        return rc
    H = list(diff["H"])
    arms = _arms_with_runs(sel)
    wins = {a: load_winners(cfg, args.bench, a) for a in arms}

    def plan(st: AuditStore):
        tasks = []
        for a in arms:
            w = wins[a]
            by_tree = {c["tree"]: c for c in w["portfolio"]}
            final = by_tree[w["final_tree"]]
            tasks += P.plan_confirm(by_tree, w["rows_by_winner"], st, cfg.reps_select, cfg.reps_confirm)
            tasks += P.plan_final(final, H, w["rows_by_winner"].get(w["final_tree"], []), st, cfg.reps_select, cfg.reps_confirm)
        return tasks

    # confirm and final are one execution (they share draw ids); the report reads them by phase
    rep = run_phase(cfg, args.bench, spec, mat, plan, store, P.PHASE_CONFIRM, dry_run=args.dry_run,
                    frozen_testset_sha256=diff.get("testset_sha256"), log=log)
    if args.dry_run:
        return 0
    # the fill pass in run_phase covered `confirm`; run it for `final` too
    from scripts.best_of_population.execute import _fill_failed_rows, eval_spec

    class _R:  # minimal report shim for the final-phase fill
        filled_rows = 0
        still_failed: dict = {}
    _fill_failed_rows(cfg, cfg.evaluator_cfg(), args.bench, eval_spec(cfg, spec), mat, plan, store, P.PHASE_FINAL, _R(), log)
    if plan(store):
        log(f"{args.bench}: confirmation is incomplete after 2 attempts — rerun `confirm` to resume (exit {EXIT_INCOMPLETE})")
        return EXIT_INCOMPLETE
    d6 = cfg.reps_select + 1
    pairs = [(c["job_id"], c["tree"], P.PHASE_SELECT, cfg.reps_select, P.PHASE_CONFIRM, d6)
             for w in wins.values() for c in w["portfolio"] if c["tree"] in w["rows_by_winner"]]
    guard = cross_phase_guard(store, pairs, log=log)
    for a in arms:
        wins[a].setdefault("cache_guard", {})["confirm_cross_phase"] = guard
        wins[a]["confirm_phase_report"] = {"errors": rep.errors, "filled_rows": rep.filled_rows, "still_failed": rep.still_failed}
        write_winners(cfg, args.bench, a, wins[a])
    log(f"{args.bench}: confirmation complete")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    benches = [args.bench] if args.bench else sorted(p.stem for p in (cfg.out_path / "difficult").glob("*.json"))
    not_applicable: dict[str, str] = {}
    for bench in benches:
        diff = load_difficult(cfg, bench)
        if _not_applicable(diff):
            not_applicable[bench] = diff.get("reason", "empty difficult set")
            continue
        sel = load_selection(cfg, bench)
        store = _store(cfg, bench)
        for arm in _arms_with_runs(sel):
            wpath = arm_path(cfg, "winners", bench, arm)
            if not wpath.exists():
                log(f"{bench}/{arm}: no winners yet — skipped")
                continue
            w = load_winners(cfg, bench, arm)
            port = load_portfolio(cfg, bench, arm)
            try:
                res = build_results(cfg, bench, arm, sel, diff, port, w, store, log=log)
            except ValueError as exc:
                log(f"{bench}/{arm}: confirmation incomplete — {exc}")
                continue
            paths = write_results(cfg, bench, arm, res, diff, w, store)
            log(f"  wrote {paths[0]} and {paths[1]}")
    rows = summary_rows(load_results(cfg), not_applicable)
    out = write_summary(cfg, rows)
    log(f"summary: {out['md']}")
    log(Path(out["md"]).read_text())
    return 0


def cmd_plan(args: argparse.Namespace) -> int:
    cfg = _cfg(args)
    bench = args.bench
    p = audit_paths(cfg, bench)
    diff = load_difficult(cfg, bench) if p["difficult"].exists() else None
    sel = load_selection(cfg, bench) if p["selection"].exists() else select_runs(cfg, bench, log=lambda *a: None)
    per_arm: dict[str, int] = {}
    seed_score = None
    testset_size = (diff or {}).get("testset_size")
    for arm in _arms_with_runs(sel):
        e = sel["arms"][arm]
        seed_score = seed_score or e.get("seed_test_score")
        cpath = arm_path(cfg, "candidates", bench, arm)
        if cpath.exists():
            per_arm[arm] = json.loads(cpath.read_text())["n_candidates"]
        else:
            run = RunRecord(**e["run"])
            per_arm[arm] = len({c.sha for c in candidates_from_run(cfg, run)})  # distinct shas (trees unknown yet)
        if not testset_size:
            man = read_json(Path(e["history_dir"]) / "datasets.json") if e.get("history_dir") else None
            testset_size = (man or {}).get("testset_size")
    n_final_won = {}
    for arm in per_arm:
        wpath = arm_path(cfg, "winners", bench, arm)
        if wpath.exists():
            n_final_won[arm] = json.loads(wpath.read_text()).get("n_rows_final_is_winner", 0)
    store = _store(cfg, bench) if p["rows"].exists() else None
    spec_threads = 1
    try:
        spec_threads = _spec(cfg, bench).num_threads
    except Exception:  # noqa: BLE001
        pass
    est = estimate_bench(cfg, bench, diff, per_arm, store, seed_score, testset_size, n_final_won, num_threads=spec_threads)
    log(format_estimate(est))
    (p["root"] / "plan").mkdir(parents=True, exist_ok=True)
    (p["root"] / "plan" / f"{bench}.json").write_text(json.dumps(est, indent=2))
    return 0


def cmd_all(args: argparse.Namespace) -> int:
    for name, fn in (("select-run", cmd_select_run), ("difficult", cmd_difficult), ("materialize", cmd_materialize),
                     ("plan", cmd_plan), ("screen", cmd_screen), ("select", cmd_select), ("confirm", cmd_confirm),
                     ("report", cmd_report)):
        log(f"=== {args.bench}: {name} ===")
        rc = fn(args)
        if rc:
            log(f"{args.bench}: stage {name} exited {rc}; stopping")
            return rc
    return 0


def cmd_aggregate(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cfg = AuditConfig(out=str(out))
    cfg.save()
    for src in args.inputs:
        src = Path(src).expanduser()
        for sub in ("selection", "difficult", "candidates", "portfolio", "winners", "results", "plan"):
            d = src / sub
            if not d.exists():
                continue
            for f in sorted(d.iterdir()):
                if f.name.startswith("summary"):
                    continue
                dst = out / sub / f.name
                if dst.exists() and dst.read_bytes() != f.read_bytes():
                    log(f"CONFLICT: {dst} already exists with different content (from another input)")
                    return 1
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(f, dst)
    not_applicable = {}
    for p in sorted((out / "difficult").glob("*.json")) if (out / "difficult").exists() else []:
        d = json.loads(p.read_text())
        if d.get("not_applicable"):
            not_applicable[d["bench"]] = d.get("reason", "empty difficult set")
    rows = summary_rows(load_results(cfg), not_applicable)
    paths = write_summary(cfg, rows)
    log(Path(paths["md"]).read_text())
    return 0


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

def _common(p: argparse.ArgumentParser, *, bench_required: bool = True, evalstage: bool = False) -> None:
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--bench", required=bench_required)
    p.add_argument("--runs-root", action="append", help="collected runs root (repeatable; default $CE_RUNS_DIR or ~/ce-runs)")
    p.add_argument("--ce-home")
    p.add_argument("--arms", help="comma-separated (default freeform,pareto_evolution)")
    p.add_argument("--workers", type=int)
    p.add_argument("--dotenv")
    p.add_argument("--config-name", dest="config_name")
    p.add_argument("--reps-select", type=int, dest="reps_select")
    p.add_argument("--reps-confirm", type=int, dest="reps_confirm")
    p.add_argument("--top-k", type=int, dest="top_k")
    p.add_argument("--threshold", type=float)
    p.add_argument("--eval-timeout", type=int, dest="eval_timeout", help="seconds per run_eval call (default: the bench's comparison_eval_timeout)")
    p.add_argument("--include-void-seed", action="store_true", dest="include_void_seed")
    p.add_argument("--refreeze", action="store_true", help="recompute a frozen artifact (selection/difficult/portfolio/winners)")
    p.add_argument("--candidates-file", dest="candidates_file", help='JSON {"<arm>": ["<sha>", ...]} explicit candidate lists')
    if evalstage:
        p.add_argument("--dry-run", action="store_true", dest="dry_run")
    else:
        p.set_defaults(dry_run=False)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, ev in (("plan", cmd_plan, False), ("select-run", cmd_select_run, False), ("difficult", cmd_difficult, False),
                         ("materialize", cmd_materialize, False), ("screen", cmd_screen, True), ("select", cmd_select, True),
                         ("confirm", cmd_confirm, True), ("all", cmd_all, True)):
        p = sub.add_parser(name)
        _common(p, evalstage=ev)
        p.set_defaults(fn=fn)
    p = sub.add_parser("report")
    _common(p, bench_required=False)
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("aggregate")
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(fn=cmd_aggregate)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except FileNotFoundError as exc:
        log(f"ERROR: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
