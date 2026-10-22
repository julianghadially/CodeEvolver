#!/usr/bin/env python3
"""Export the v5 count results (``catalog.py count``) to CSV files.

Reads ``<results>/count/<bench>/<run_id>.json`` and writes two files to ``--out-dir``
(default ``scripts/macro-counts/output/``, gitignored):

  macro_counts_runs.csv       one row per run: bench, arm, seed, count, counted ids, dropped ids,
                              uncatalogued count, iterations, test scores, judge model, tokens
  macro_counts_approaches.csv one row per approach a run was judged to have: every catalog hit
                              (status ``counted``, or ``dropped`` when it cites no landed iteration)
                              and every ``uncatalogued`` approach, with iterations and evidence

The landed check is the same deterministic one the report applies (``apply_landed_check``).

  python scripts/macro-counts/export_csv.py --archive-root ~/ce-analysis/macro-counts/catalog-v5/runs-root
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from catalog import (BENCHES, DEFAULT_ARCHIVE_ROOT, DEFAULT_OUT, apply_landed_check, count_dir,  # noqa: E402
                     find_run, landed_iterations, load_counts)

DEFAULT_EXPORT_DIR = HERE / "output"

RUN_FIELDS = ["bench", "run_id", "arm", "seed", "count", "counted_ids", "dropped_ids", "n_uncatalogued",
              "catalog_size", "iterations", "landed_iterations", "seed_test", "final_test", "judge_model",
              "catalog_sha256", "judged_at", "session_id", "n_tool_calls", "retries", "total_tokens"]
APPROACH_FIELDS = ["bench", "run_id", "arm", "seed", "status", "id", "name", "iterations_cited",
                   "landed_iterations_cited", "evidence", "why_macro"]


def ints(v) -> str:
    return " ".join(str(x) for x in v) if isinstance(v, list) else str(v or "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", type=Path, default=DEFAULT_OUT, help="catalog.py --out directory")
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    ap.add_argument("--bench", action="append", help="only these benchmarks (default: every counted one)")
    args = ap.parse_args()

    root = args.results / "count"
    found = sorted(d.name for d in root.iterdir() if d.is_dir()) if root.is_dir() else []
    benches = [b for b in BENCHES if b in found] + sorted(set(found) - set(BENCHES))
    if args.bench:
        benches = [b for b in benches if b in args.bench]
    if not benches:
        raise SystemExit(f"no count results under {root}")

    run_rows, approach_rows = [], []
    for bench in benches:
        jds = load_counts(args.results, bench)
        for rid, jd in sorted(jds.items()):
            run = find_run(args.archive_root, rid)
            if run is None:
                raise SystemExit(f"run dir for {rid} not found under {args.archive_root}")
            names = {e["id"]: e["name"] for e in jd["catalog"]["macro"]}
            ch = apply_landed_check(jd["result"], run)
            base = {"bench": bench, "run_id": rid, "arm": run.arm, "seed": run.seed}
            run_rows.append({**base, "count": len(ch["counted"]),
                             "counted_ids": " ".join(h["id"] for h in ch["counted"]),
                             "dropped_ids": " ".join(h["id"] for h in ch["dropped"]),
                             "n_uncatalogued": len(ch["uncatalogued"]), "catalog_size": len(jd["catalog"]["macro"]),
                             "iterations": len(run.iteration_keys), "landed_iterations": len(landed_iterations(run)),
                             "seed_test": run.score("seed_test_score"), "final_test": run.score("final_test_score"),
                             "judge_model": jd["model"], "catalog_sha256": jd["catalog_sha256"][:12],
                             "judged_at": jd["judged_at"], "session_id": jd["session_id"],
                             "n_tool_calls": jd["n_tool_calls"], "retries": jd["retries"],
                             "total_tokens": (jd.get("usage") or {}).get("total")})
            for status, hits in (("counted", ch["counted"]), ("dropped", ch["dropped"])):
                for h in hits:
                    approach_rows.append({**base, "status": status, "id": h["id"], "name": names.get(h["id"], ""),
                                          "iterations_cited": ints(h.get("iterations")),
                                          "landed_iterations_cited": ints(h.get("landed_iterations", [])),
                                          "evidence": h.get("evidence", ""), "why_macro": ""})
            for u in ch["uncatalogued"]:
                approach_rows.append({**base, "status": "uncatalogued", "id": "", "name": u.get("name", ""),
                                      "iterations_cited": ints(u.get("iterations")), "landed_iterations_cited": "",
                                      "evidence": u.get("description", ""), "why_macro": u.get("why_macro", "")})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, fields, rows in (("macro_counts_runs.csv", RUN_FIELDS, run_rows),
                               ("macro_counts_approaches.csv", APPROACH_FIELDS, approach_rows)):
        with (args.out_dir / name).open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    print(f"wrote {args.out_dir / 'macro_counts_runs.csv'} ({len(run_rows)} runs) and "
          f"{args.out_dir / 'macro_counts_approaches.csv'} ({len(approach_rows)} rows) from {len(benches)} benches")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
