#!/usr/bin/env python3
"""Ranked macro-approach count for pairs of CodeEvolver runs, judged by an OpenCode session.

Secondary result for the CoRL workshop paper: how much each arm explores,
measured head to head. A macro approach is a distinct, non-incremental change
in the solution for a task relative to the previously attempted solutions. For
each (benchmark, seed) the judge sees the pareto_evolution run and the freeform
run (blinded as Program A / Program B), lists each program's distinct
approaches (consolidated within the run, ranked by departure from the seed,
ending at the first change it considers incremental, at most 15 each), then ranks
the approaches of both runs in one combined order, leaving out any approach both
runs implemented. The count for a run is how many of its approaches are in that
combined order (--top-k 0, the default; --top-k N counts only the top N). Every pair is
judged twice, once with each arm as Program A, and the headline averages over
seeds x orders. Definition, examples and instructions: ``judge_prompt.md``.

HOW TO RUN (from the repo root; needs ``opencode`` >= 1.17.11 and ``rg`` on PATH)

  # 1. Preview: find runs and pairs, materialize the judge workspaces, print sizes. No sessions.
  python scripts/macro-counts/macro_counts.py --archive-root ~/ce-runs --dry-run

  # 2. Judge every pair in both orders (one OpenCode session per judgment; 2 in parallel).
  #    Judgments made on an identical workspace with the current prompt are reused from judge/.
  python scripts/macro-counts/macro_counts.py --archive-root ~/ce-runs --jobs 2

  # A subset / other records / other arms:
  python scripts/macro-counts/macro_counts.py --runs 'alfworld-s2$'
  python scripts/macro-counts/macro_counts.py --archive-root /path/records --out /path/records/macro-out
  python scripts/macro-counts/macro_counts.py --arm1 ee --arm2 freeform --void ''

  Needs DEEPINFRA_API_KEY in the environment or in <repo>/.env (for the default
  deepinfra/zai-org/GLM-5.2 judge), and git on PATH. Default --archive-root is
  the parent of the repo; default --out is ~/ce-analysis/macro-counts
  (outside the repo: results are never committed).

WHERE THE RUNS COME FROM
  A run is a directory named <arm>-<bench>-s<seed>[-redo] holding history.json and repo.bundle
  (status.json and .codeevolver/memory/ are used when present), found under
  <root>/ce-run*-archive/ce-runs/, <root>/ce-runs/, or <root>/ directly. The fleet
  controller collects runs to ~/ce-runs/<run_id>/ (specs/experiment-docs/experiment-fleet.md),
  so ``--archive-root ~/ce-runs`` is the usual choice on a controller machine.
  A pair is <arm1>-<bench>-sN with <arm2>-<bench>-sN; a -redo run supersedes the original of the
  same arm, bench and seed; void runs (--void) and runs
  without a partner are reported as unpaired and not judged.

WHAT THE JUDGE SEES (workspaces/<pair>__<order>/, blinded, materialized on disk)
  INDEX.md, seed/<program file>, and for A/ and B/: iterations/<NN>/{meta.md, summary.md,
  diff.patch | attempted.patch, <program file>} and final/<program file>. Summaries and
  commit messages have the arms' search vocabulary redacted (BLIND_WORDS); code and
  patches are never altered. No scores, no arm name, run id, branch name, improvement
  mode, lineage or architecture title. Order ``ab`` puts arm1 as Program A, ``ba`` puts
  arm2 as Program A. The judge session (judge.py) reads the workspace with read/grep/
  glob/list only: no bash, no edit, no CodeEvolver skills, no memory.

WHAT COMES OUT (--out)
  judge/<pair>__<order>.json            raw judgment + model, session id, usage, tool calls, A/B -> arm
                                        mapping, sha256 of the workspace and of the prompt
  judge/<pair>__<order>.transcript.json the exported OpenCode session (best effort)
  review/<pair>.md                      human review file: top-k counts per order, both programs' ranked
                                        lists, combined ranking ("1. A: ...") with the left-out (shared)
                                        ids, iteration summaries
  macro_counts.csv                      one row per (pair, order)
  MACRO_COUNTS.md                       per-bench tables; the headline is each arm's mean top-k count
  workspaces/, bundles/, opencode/      what the judge saw; bare git clones (cache); engine scratch

Stages (``--stage``): build | judge | report | all (default).
The judge itself (prompt loading, the OpenCode session, fenced-JSON parsing, validation)
is ``judge.py`` next to this file; the definition and instructions are ``judge_prompt.md``.
Read-only against the run archives.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from judge import (DEFAULT_MAX_PER_PROGRAM, DEFAULT_MAX_TURNS, DEFAULT_MODEL, DEFAULT_RETRIES,  # noqa: E402
                   DEFAULT_TIMEOUT_SECONDS, LIST_KEY, PROGRAMS, PROMPT_PATH, JudgeError, judge_workspace,
                   load_env_key, load_judgment, prepare_engine_env, prompt_sha256, provider_key, shared_ids,
                   top_k_counts, unique_ids, workspace_sha256)
import runs as runs_mod  # noqa: E402
from runs import (BENCHES, PROGRAM_FILE, Pair, Run, build_pairs, commit_messages, diff_between,  # noqa: E402
                  discover_runs, ensure_bare, eval_snapshot_shas, key_of, log, memory_summary,
                  program_file_for, seed_source, show_file)

REPO_ROOT = HERE.parent.parent
DEFAULT_ARCHIVE_ROOT = REPO_ROOT.parent  # sibling ce-run*-archive folders; pass --archive-root ~/ce-runs for fleet-collected runs
DEFAULT_OUT = Path.home() / "ce-analysis" / "macro-counts"
DEFAULT_ARM1 = "pareto_evolution"
DEFAULT_ARM2 = "freeform"
DEFAULT_TOP_K = 0  # 0 = whole combined ranking (the judge ends each list at the incremental cutoff)
ORDERS = ("ab", "ba")  # ab: arm1 is Program A; ba: arm2 is Program A

# Blinding: the judge must not see which arm produced a run. The arm name and
# arm-specific metadata are left out of the evidence, and these words are
# redacted from the run-written prose (iteration summaries, commit messages;
# never from code or diffs). They are the arms' own vocabulary for their
# search strategy.
BLIND_WORDS = re.compile(
    r"\b(macro|micro|explor\w*|exploit\w*|pareto|lineages?|ee|asa|freeform|simple[_ -]?loop|"
    r"evolution\w*|evolv\w*)\b", re.I)
BLIND_MARK = "[redacted]"


def count_label(top_k: int) -> str:
    """Column label for the count: the top-k cutoff, or the whole ranking (incremental cutoff) when k <= 0."""
    return f"top {top_k}" if top_k > 0 else "above cutoff"


def blind(text: str) -> str:
    return BLIND_WORDS.sub(BLIND_MARK, text)


MAX_PROGRAM_CHARS = 120_000   # warn (never cut) when a program file is larger than this


# ---------------------------------------------------------------------------
# Judge workspace (materialized files the session reads)
# ---------------------------------------------------------------------------

def fmt_score(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def materialize_run(run: Run, git_dir: Path, prog: str, label: str, dest: Path) -> tuple[list[str], dict]:
    """Write Program <label>'s iteration files and final program under ``dest``.
    Returns (INDEX lines, stats). Arm-neutral: no run id, arm, branch, scores."""
    stats = {"iterations": 0, "landed": 0, "summary_source": {}, "empty_diffs": [], "attempted_diffs": [],
             "chars": 0}
    fname = Path(prog).name
    index: list[str] = [f"\n## Program {label} (`{label}/`)\n"]
    for k in run.iteration_keys:
        rec = run.iters[k]
        stats["iterations"] += 1
        landed = bool(rec.get("changes_landed"))
        stats["landed"] += landed
        it_dir = dest / "iterations" / f"{int(k):02d}"
        # Only arm-neutral metadata: branch names, improvement_mode, new_lineage,
        # architecture_title and every score are left out.
        meta = [f"- iteration: {k}", f"- parent_iteration: {rec.get('parent_iteration')}",
                f"- changes_landed: {landed}"]
        if rec.get("skipped"):
            meta.append("- skipped: True")
        _write(it_dir / "meta.md", f"# Program {label}, iteration {k}\n\n" + "\n".join(meta) + "\n")

        # Summary: history change_summary -> memory file -> commit messages. Blinded.
        summary = (rec.get("change_summary") or "").strip()
        source = "history.change_summary"
        if not summary:
            summary = memory_summary(run, k) or ""
            source = "memory_file"
        p_sha, c_sha = rec.get("parent_sha"), rec.get("close_sha")
        commits = [blind(m) for m in (commit_messages(git_dir, p_sha, c_sha) if p_sha and c_sha else [])]
        if not summary:
            source = "commit_messages" if commits else "none"
            summary = "\n\n".join(f"(commit) {m}" for m in commits) if commits else "(no summary or commit message available)"
        summary = blind(summary)
        stats["summary_source"][k] = source
        text = f"# Program {label}, iteration {k}: summary (source: {source})\n\n{summary}\n"
        if commits and source != "commit_messages":
            text += "\n## Commit messages\n\n" + "\n".join(f"- {m.splitlines()[0]}" for m in commits) + "\n"
        _write(it_dir / "summary.md", text)

        diff = diff_between(git_dir, p_sha, c_sha) if p_sha and c_sha else ""
        files = {"meta.md": "metadata", "summary.md": "summary"}
        if diff.strip():
            _write(it_dir / "diff.patch", diff)
            files["diff.patch"] = "program diff, parent commit -> closing commit"
        else:
            stats["empty_diffs"].append(k)
            attempted = ""
            for snap in (eval_snapshot_shas(git_dir, k) if p_sha else []):
                attempted = diff_between(git_dir, p_sha, snap)
                if attempted.strip():
                    break
            if attempted.strip():
                _write(it_dir / "attempted.patch", attempted)
                _write(it_dir / "diff.patch",
                       "(no program change landed in this iteration; the attempted change is in attempted.patch)\n")
                files["diff.patch"] = "no landed change (see attempted.patch)"
                files["attempted.patch"] = ("ATTEMPTED program diff, parent commit -> last eval snapshot of this "
                                            "iteration; this change did NOT land")
                stats["attempted_diffs"].append(k)
            else:
                _write(it_dir / "diff.patch", "(no program change in this iteration)\n")
                files["diff.patch"] = "no program change"
        if c_sha:
            src = show_file(git_dir, c_sha, prog)
            if src is not None:
                _write(it_dir / fname, src)
                files[fname] = "full program source at this iteration's closing commit"
        for f, what in files.items():
            index.append(f"- `{label}/iterations/{int(k):02d}/{f}`: iteration {k} {what} "
                         f"(parent iteration {rec.get('parent_iteration')}, {'landed' if landed else 'NOT landed'})")

    final_sha = run.final_sha()
    final_src = show_file(git_dir, final_sha, prog) if final_sha else None
    if final_src is None:
        final_src = show_file(git_dir, "refs/heads/main", prog) or "(final program not found)"
    if len(final_src) > MAX_PROGRAM_CHARS:
        log(f"[{run.run_id}] WARNING: final program is {len(final_src):,} chars")
    _write(dest / "final" / fname, final_src)
    index.append(f"- `{label}/final/{fname}`: full program source at the final candidate")
    stats["chars"] = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    return index, stats


def materialize_pair_workspace(pair: Pair, order: str, git_dirs: dict[str, Path], ws: Path) -> dict:
    """Rebuild the judge workspace for one pair in one presentation order. Returns stats."""
    progs = pair.programs(order)
    prog = program_file_for(pair.run1, git_dirs[pair.run1.run_id])
    fname = Path(prog).name
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    seeds = {r.run_id: seed_source(r, git_dirs[r.run_id], prog) for r in (pair.run1, pair.run2)}
    stats: dict = {"order": order, "seed_programs_match": len(set(seeds.values())) == 1}
    if not stats["seed_programs_match"]:
        log(f"[{pair.pair_id}] WARNING: the two runs' seed programs differ; showing Program A's")
    _write(ws / "seed" / fname, seeds[progs["A"].run_id])

    index = [f"# Judge workspace: {pair.bench} seed {pair.seed}\n",
             f"- benchmark: {pair.bench}", f"- seed: {pair.seed}",
             f"- main program file: `{prog}` (materialized as `{fname}` in each directory below)"]
    for p in PROGRAMS:
        index.append(f"- Program {p}: {len(progs[p].iteration_keys)} iterations")
    index.append(f"- note: some words in summaries and commit messages are replaced by `{BLIND_MARK}` to blind "
                 f"this audit; judge from the code and the rest of the text. No scores are provided.\n")
    index.append("## Files\n")
    index.append(f"- `seed/{fname}`: full program source at iteration 0 (the starting point of both programs)")
    for p in PROGRAMS:
        lines, st = materialize_run(progs[p], git_dirs[progs[p].run_id], prog, p, ws / p)
        index += lines
        stats[p] = st
    _write(ws / "INDEX.md", "\n".join(index) + "\n")
    stats["files"] = sum(1 for f in ws.rglob("*") if f.is_file())
    stats["chars"] = sum(f.stat().st_size for f in ws.rglob("*") if f.is_file())
    stats["est_tokens"] = stats["chars"] // 4
    return stats


def src_counts(st: dict) -> dict:
    vals = list(st["summary_source"].values())
    return {s: vals.count(s) for s in sorted(set(vals))}


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def iteration_summary_text(run: Run, git_dir: Path, k: str) -> tuple[str, str]:
    rec = run.iters[k]
    s = (rec.get("change_summary") or "").strip()
    if s:
        return s, "summary"
    s = memory_summary(run, k)
    if s:
        return s, "summary(memory)"
    p_sha, c_sha = rec.get("parent_sha"), rec.get("close_sha")
    commits = commit_messages(git_dir, p_sha, c_sha) if p_sha and c_sha else []
    if commits:
        return "\n\n".join(commits), "commits"
    return "(no summary or commit message available)", "none"


def cell(v) -> str:
    """Make a value safe inside a one-line markdown table cell."""
    return str(v).replace("\r", "").replace("\n", " ").replace("|", "\\|")


def order_metrics(pair: Pair, order: str, jd: dict, top_k: int) -> dict:
    """Per-arm numbers for one judgment, keyed by arm slot (1 = arm1, 2 = arm2)."""
    j = jd["judgment"]
    progs = pair.programs(order)
    top = top_k_counts(j, top_k)
    uniq, shared = unique_ids(j), shared_ids(j)
    cap = jd.get("max_per_program", DEFAULT_MAX_PER_PROGRAM)
    slot_of = {"A": 1 if order == "ab" else 2, "B": 2 if order == "ab" else 1}
    m: dict = {"n_combined": len(j["combined_ranking"]), "program_of_arm1": "A" if order == "ab" else "B",
               "session_id": jd.get("session_id") or "", "n_tool_calls": jd.get("n_tool_calls", 0),
               "retries": jd.get("retries", 0), "transcript": jd.get("transcript") or ""}
    for p in PROGRAMS:
        s = slot_of[p]
        lst = j[LIST_KEY[p]]
        m[f"arm{s}_run"] = progs[p].run_id
        m[f"arm{s}_topk"] = top[p]
        m[f"arm{s}_unique"] = len(uniq[p])
        m[f"arm{s}_shared"] = len(shared[p])
        m[f"arm{s}_listed"] = len(lst)
        m[f"arm{s}_listed_in_final"] = sum(1 for a in lst if a["in_final_program"])
        m[f"arm{s}_capped"] = int(len(lst) >= cap)
    return m


def write_review(pair: Pair, judged: dict[str, dict], git_dirs: dict[str, Path], out_dir: Path,
                 top_k: int) -> None:
    arm1, arm2 = pair.run1.arm, pair.run2.arm
    lines: list[str] = [f"# {pair.pair_id}: {pair.run1.run_id} vs {pair.run2.run_id} (ranked macro approaches)\n"]
    lines.append(f"| order | Program A | Program B | {arm1} {count_label(top_k)} | {arm2} {count_label(top_k)} | "
                 f"unique ({arm1} / {arm2}) | left out as shared ({arm1} / {arm2}) | judge | tool calls | retries | transcript |\n"
                 "|---|---|---|---|---|---|---|---|---|---|---|")
    ms = {}
    for order in ORDERS:
        jd = judged.get(order)
        if not jd:
            lines.append(f"| {order} | - | - | not judged | not judged | - | - | - | - | - | - |")
            continue
        m = ms[order] = order_metrics(pair, order, jd, top_k)
        progs = pair.programs(order)
        tr = f"[{m['transcript']}](../judge/{m['transcript']})" if m["transcript"] else "-"
        lines.append(f"| {order} | {progs['A'].arm} | {progs['B'].arm} | **{m['arm1_topk']}** | **{m['arm2_topk']}** | "
                     f"{m['arm1_unique']} / {m['arm2_unique']} | {m['arm1_shared']} / {m['arm2_shared']} | {jd['model']} | "
                     f"{m['n_tool_calls']} | {m['retries']} | {tr} |")
    if len(ms) == 2:
        a1 = sum(m["arm1_topk"] for m in ms.values()) / 2
        a2 = sum(m["arm2_topk"] for m in ms.values()) / 2
        lines.append(f"| **mean** | | | **{a1:.1f}** | **{a2:.1f}** | | | | | | |")
    lines.append(f"\nHeld-out score seed → final (reviewer only; the judge never sees scores): "
                 f"{arm1} {fmt_score(pair.run1.score('seed_test_score'))} → "
                 f"{fmt_score(pair.run1.score('final_test_score'))}; {arm2} {fmt_score(pair.run2.score('seed_test_score'))} → "
                 f"{fmt_score(pair.run2.score('final_test_score'))}. "
                 f"Workspaces the judge read: `workspaces/{pair.pair_id}__ab/`, `workspaces/{pair.pair_id}__ba/`.\n")

    for order in ORDERS:
        jd = judged.get(order)
        if not jd:
            continue
        j = jd["judgment"]
        progs = pair.programs(order)
        by_id = {a["id"]: a for p in PROGRAMS for a in j[LIST_KEY[p]]}
        lines.append(f"## Order `{order}`: Program A = {progs['A'].run_id}, Program B = {progs['B'].run_id}\n")
        for p in PROGRAMS:
            lines.append(f"### Program {p} ({progs[p].arm}): ranked approaches\n")
            lst = j[LIST_KEY[p]]
            if not lst:
                lines.append("(none listed)\n")
                continue
            lines.append("| id | name | iterations | status | in final | mechanism | evidence |\n|---|---|---|---|---|---|---|")
            for a in lst:
                lines.append(f"| {a['id']} | {cell(a['name'])} | {', '.join(str(i) for i in a['iterations'])} | "
                             f"{a['status']} | {'yes' if a['in_final_program'] else 'no'} | {cell(a['description'])} | "
                             f"{cell(a['evidence'])} |")
            lines.append("")
        lines.append("### Combined ranking of unique approaches (by departure from the seed)\n")
        for rank, c in enumerate(j["combined_ranking"], 1):
            name = by_id.get(c["id"], {}).get("name", "?")
            lines.append(f"{rank}. {c['program']}: {name} ({c['id']}) — {c['departure_note']}")
            if rank == top_k:
                lines.append(f"\n*— top {top_k} cutoff —*\n")
        lines.append("")
        left_out = shared_ids(j)
        if any(left_out.values()):
            lines.append("Left out of the combined ranking (the judge treated these as implemented by both runs): "
                         + "; ".join(f"{i} {cell(by_id[i]['name'])}" for p in PROGRAMS for i in left_out[p]) + "\n")
        lines.append("### Judge rationale\n")
        lines.append(j["rationale"].strip() + "\n")
        flags = list(j["review_flags"]) + [f"(validator) {w}" for w in jd.get("warnings", [])]
        if flags:
            lines.append("**Review flags:**\n")
            lines += [f"- {f}" for f in flags]
            lines.append("")

    lines.append("## Iteration summaries (the runs' own words, for checking the lists)\n")
    for run in (pair.run1, pair.run2):
        lines.append(f"### {run.run_id}\n")
        for k in run.iteration_keys:
            rec = run.iters[k]
            summary, src = iteration_summary_text(run, git_dirs[run.run_id], k)
            lines.append(f"#### Iter {k} · parent {rec.get('parent_iteration')} · "
                         f"{'landed' if rec.get('changes_landed') else 'NOT landed'} · "
                         f"sub {fmt_score(rec.get('best_subsample_score'))} (parent {fmt_score(rec.get('parent_subsample_score'))})\n")
            lines.append(f"> **{src}:** " + summary.replace("\n", "\n> ") + "\n")
    (out_dir / f"{pair.pair_id}.md").write_text("\n".join(lines), encoding="utf-8")


COUNT_FIELDS = ["pair_id", "bench", "seed", "order", "arm1", "arm2", "program_of_arm1",
                "arm1_run", "arm2_run", "top_k", "arm1_topk", "arm2_topk", "n_combined",
                "arm1_unique", "arm2_unique", "arm1_shared", "arm2_shared", "arm1_listed", "arm2_listed",
                "arm1_listed_in_final", "arm2_listed_in_final", "arm1_capped", "arm2_capped",
                "arm1_seed_test", "arm1_final_test", "arm2_seed_test", "arm2_final_test",
                "judge_model", "session_id", "n_tool_calls", "retries", "judged_at", "input_tokens", "output_tokens",
                "total_tokens"]
COUNT_INTS = ["seed", "top_k", "arm1_topk", "arm2_topk", "n_combined", "arm1_unique",
              "arm2_unique", "arm1_shared", "arm2_shared", "arm1_listed", "arm2_listed",
              "arm1_listed_in_final", "arm2_listed_in_final", "arm1_capped", "arm2_capped"]


def read_csv_rows(path: Path) -> list[dict]:
    """Read a CSV we wrote earlier; tolerate a BOM / Excel re-save."""
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def write_csv_rows(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def write_aggregate(pairs: list[Pair], judged: dict[str, dict], out: Path, top_k: int,
                    unpaired: list[tuple[str, str]]) -> None:
    """Rewrite macro_counts.csv and MACRO_COUNTS.md.

    Rows for (pair, order) judgments loaded in this invocation replace their old
    rows; rows for every other judgment already in the CSV are kept, so a
    filtered invocation (``--runs``, ``--bench``) never discards other pairs'
    counts. The tables are built from the merged rows."""
    counts_csv = out / "macro_counts.csv"
    current = set(judged)
    rows: list[dict] = [r for r in read_csv_rows(counts_csv) if f"{r['pair_id']}__{r['order']}" not in current]
    for r in rows:  # old rows come back as strings
        for k in COUNT_INTS:
            r[k] = int(r[k]) if r.get(k) not in (None, "") else 0
    for pair in pairs:
        for order in ORDERS:
            jd = judged.get(key_of(pair, order))
            if not jd:
                continue
            rows.append({
                "pair_id": pair.pair_id, "bench": pair.bench, "seed": pair.seed, "order": order,
                "arm1": pair.run1.arm, "arm2": pair.run2.arm, "top_k": top_k,
                **order_metrics(pair, order, jd, top_k),
                "arm1_seed_test": pair.run1.score("seed_test_score"), "arm1_final_test": pair.run1.score("final_test_score"),
                "arm2_seed_test": pair.run2.score("seed_test_score"), "arm2_final_test": pair.run2.score("final_test_score"),
                "judge_model": jd["model"], "judged_at": jd["judged_at"],
                # With prompt caching, input_tokens is only the uncached slice; total (= input + cache
                # creation + cache read + output, as OpenCode reports it) is the number to cost.
                "input_tokens": jd["usage"].get("input_tokens"), "output_tokens": jd["usage"].get("output_tokens"),
                "total_tokens": jd["usage"].get("total"),
            })
    benches = BENCHES + sorted({r["bench"] for r in rows} - set(BENCHES))
    rows.sort(key=lambda r: (benches.index(r["bench"]), r["seed"], r["order"]))
    write_csv_rows(counts_csv, rows, COUNT_FIELDS)

    stale_k = sorted({r["pair_id"] for r in rows if r["top_k"] != top_k})
    md: list[str] = [
        "# Ranked macro approaches (LLM judge, pairwise)\n",
        f"Generated {dt.datetime.now().isoformat(timespec='minutes')}. "
        f"Judge model: {', '.join(sorted({str(r['judge_model']) for r in rows}))}. "
        "Definition and prompt: `scripts/macro-counts/judge_prompt.md`. For each bench and seed the judge sees both "
        "runs, blinded as Program A / Program B, lists each program's distinct approaches (consolidated within the "
        "run, capped by --max-per-program, each list ending where the changes become incremental), then ranks them together by departure from the seed, leaving out the approaches both runs implemented. "
        "The judge is a read-only OpenCode session over a materialized workspace (`workspaces/`); it sees no scores. "
        + (f"**top {top_k}** = how many of a run's unique approaches are in the top {top_k} of that combined ranking. "
           if top_k > 0 else "**above cutoff** = how many of a run's unique approaches are in that combined ranking "
           "(every listed approach is above the judge's incremental cutoff). ")
        + 
        "Every pair is judged in both orders (`ab`: first arm is Program A; `ba`: second arm is Program A); "
        "the headline mean and sd are over seeds × orders, complete pairs (both orders judged) only.\n"]
    if stale_k:
        md.append(f"**Note:** rows for {', '.join(stale_k)} were computed with a different --top-k; "
                  "re-run `--stage report` over them to recompute.\n")
    for bench in benches:
        brows = [r for r in rows if r["bench"] == bench]
        if not brows:
            continue
        by_pair: dict[str, dict[str, dict]] = {}
        for r in brows:
            by_pair.setdefault(r["pair_id"], {})[r["order"]] = r
        complete = {pid: o for pid, o in by_pair.items() if set(o) == set(ORDERS)}
        arm1, arm2 = brows[0]["arm1"], brows[0]["arm2"]
        md.append(f"## {bench}\n")
        md.append(f"| arm | judgments (seeds × orders) | {count_label(top_k)} mean ± sd | as Program A (mean) | as Program B (mean) | "
                  "unique (mean) | listed (mean) | listed & in final (mean) |\n|---|---|---|---|---|---|---|---|")
        crow = [r for o in complete.values() for r in o.values()]
        for s, arm in ((1, arm1), (2, arm2)):
            m, sd = mean_sd([r[f"arm{s}_topk"] for r in crow])
            as_a = [r[f"arm{s}_topk"] for r in crow if (r["program_of_arm1"] == "A") == (s == 1)]
            as_b = [r[f"arm{s}_topk"] for r in crow if (r["program_of_arm1"] == "A") != (s == 1)]
            md.append(f"| {arm} | {len(crow)} ({len(complete)} × 2) | **{m:.2f} ± {sd:.2f}** | {mean_sd(as_a)[0]:.2f} | "
                      f"{mean_sd(as_b)[0]:.2f} | {mean_sd([r[f'arm{s}_unique'] for r in crow])[0]:.2f} | "
                      f"{mean_sd([r[f'arm{s}_listed'] for r in crow])[0]:.2f} | "
                      f"{mean_sd([r[f'arm{s}_listed_in_final'] for r in crow])[0]:.2f} |")
        md.append("")
        md.append(f"| seed | {arm1} ab | {arm1} ba | {arm1} mean | {arm2} ab | {arm2} ba | {arm2} mean | "
                  f"left out as shared, {arm1}/{arm2} (ab; ba) | review |\n|---|---|---|---|---|---|---|---|---|")
        disagree = 0
        for pid, o in sorted(by_pair.items(), key=lambda kv: next(iter(kv[1].values()))["seed"]):
            ab, ba = o.get("ab"), o.get("ba")
            v = lambda r, f: "-" if r is None else str(r[f])
            mean = lambda f: f"{(ab[f] + ba[f]) / 2:.1f}" if ab and ba else "incomplete"
            if ab and ba:
                sign = lambda r: (r["arm1_topk"] > r["arm2_topk"]) - (r["arm1_topk"] < r["arm2_topk"])
                disagree += sign(ab) != sign(ba)
            seed = (ab or ba)["seed"]
            md.append(f"| s{seed} | {v(ab, 'arm1_topk')} | {v(ba, 'arm1_topk')} | {mean('arm1_topk')} | "
                      f"{v(ab, 'arm2_topk')} | {v(ba, 'arm2_topk')} | {mean('arm2_topk')} | "
                      f"{v(ab, 'arm1_shared')}/{v(ab, 'arm2_shared')}; {v(ba, 'arm1_shared')}/{v(ba, 'arm2_shared')} | "
                      f"[review/{pid}.md](review/{pid}.md) |")
        md.append(f"\nPosition check: the two orders disagree on which arm has more {count_label(top_k)} approaches "
                  f"(or on a tie) in {disagree} of {len(complete)} complete pairs.\n")
    if unpaired:
        md.append("## Unpaired runs (not judged)\n")
        md.append("| run | why |\n|---|---|")
        md += [f"| {rid} | {why} |" for rid, why in unpaired]
        md.append("")
    (out / "MACRO_COUNTS.md").write_text("\n".join(md), encoding="utf-8")
    log(f"wrote {out / 'MACRO_COUNTS.md'} ({len(rows)} judgments, {len(current)} from this invocation)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--runs", help="regex on run id; both runs of a pair must match, e.g. 'alfworld-s2$'")
    ap.add_argument("--bench", action="append", help="benchmark name(s) to include (default: all found)")
    ap.add_argument("--arm1", default=DEFAULT_ARM1, help=f"first arm of each pair (default {DEFAULT_ARM1})")
    ap.add_argument("--arm2", default=DEFAULT_ARM2, help=f"second arm of each pair (default {DEFAULT_ARM2})")
    ap.add_argument("--program-file", action="append", default=[], metavar="BENCH=PATH",
                    help="main program file for a bench, e.g. webshop=codeevolver_benchmarks/webshop_program.py")
    ap.add_argument("--void", default=None, metavar="REGEX",
                    help="run ids left out of pairing (default: pareto_evolution s1-s3; '' = none)")
    ap.add_argument("--top-k", type=int, default=DEFAULT_TOP_K, help="combined-ranking cutoff for the count (0 = whole ranking, i.e. the judge's incremental cutoff)")
    ap.add_argument("--max-per-program", type=int, default=DEFAULT_MAX_PER_PROGRAM,
                    help="cap on each program's approach list")
    ap.add_argument("--stage", choices=["build", "judge", "report", "all"], default="all")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenCode provider/model id for the judge session")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS, help="turn budget per session")
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per session (hard kill)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="fresh sessions to try per judgment")
    ap.add_argument("--jobs", type=int, default=2, help="concurrent judge sessions")
    ap.add_argument("--force", action="store_true", help="re-judge pairs that already have a judgment")
    ap.add_argument("--dry-run", action="store_true", help="materialize workspaces only and print size estimates")
    args = ap.parse_args()

    if not PROMPT_PATH.exists():
        print(f"missing prompt: {PROMPT_PATH}", file=sys.stderr)
        return 2
    out: Path = args.out
    for sub in ("workspaces", "judge", "review", "bundles", "opencode"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    if args.void is not None:
        runs_mod.VOID_RUN_RE = re.compile(args.void) if args.void else None
    for spec in args.program_file:
        bench, _, path = spec.partition("=")
        if not path:
            print(f"--program-file expects BENCH=PATH, got {spec!r}", file=sys.stderr)
            return 2
        PROGRAM_FILE[bench] = path
    runs = discover_runs(args.archive_root, args.runs, set(args.bench or []), {args.arm1, args.arm2})
    pairs, unpaired = build_pairs(runs, args.arm1, args.arm2)
    log(f"{len(runs)} runs, {len(pairs)} pairs ({args.arm1} vs {args.arm2}): "
        f"{', '.join(p.pair_id for p in pairs) or '-'}")
    for rid, why in unpaired:
        log(f"  unpaired: {rid} ({why})")
    if not pairs:
        print("no pairs matched", file=sys.stderr)
        return 1

    stage = "build" if args.dry_run else args.stage
    git_dirs: dict[str, Path] = {}
    for pair in pairs:
        for run in (pair.run1, pair.run2):
            git_dirs[run.run_id] = ensure_bare(run, out / "bundles")
    tasks = [(pair, order) for pair in pairs for order in ORDERS]

    # Materialize workspaces (two per pair: ab and ba).
    stats_all: dict[str, dict] = {}
    for pair, order in tasks:
        key = key_of(pair, order)
        ws, st_path = out / "workspaces" / key, out / "workspaces" / f"{key}.stats.json"
        if stage in ("build", "all", "judge") or not (ws / "INDEX.md").exists():
            stats = materialize_pair_workspace(pair, order, git_dirs, ws)
            st_path.write_text(json.dumps(stats, indent=1), encoding="utf-8")
        else:
            stats = json.loads(st_path.read_text(encoding="utf-8"))
        stats_all[key] = stats
        if order == "ab":
            log(f"[{pair.pair_id}] workspace {stats['files']} files, {stats['chars']:,} chars (~{stats['est_tokens']:,} tok "
                f"if read in full) per order; "
                + "; ".join(f"{p}={pair.programs(order)[p].run_id}: {stats[p]['iterations']} iters, summaries "
                            f"{src_counts(stats[p])}, attempted diffs {stats[p]['attempted_diffs'] or '-'}"
                            for p in PROGRAMS))
    if stage == "build":
        total = sum(s["est_tokens"] for s in stats_all.values())
        log(f"total ~{total:,} tokens of workspace text across {len(tasks)} judgments ({len(pairs)} pairs x 2 orders)")
        return 0

    judged: dict[str, dict] = {}
    failures: list[str] = []
    if stage in ("judge", "all"):
        prepare_engine_env(out / "opencode", model=args.model)
        load_env_key(REPO_ROOT, HERE)
        key_var = provider_key(args.model)
        if key_var and not os.environ.get(key_var):
            print(f"{key_var} not set (checked env, <repo>/.env, scripts/macro-counts/.env)", file=sys.stderr)
            return 2
        if not shutil.which("opencode"):
            print("opencode not found on PATH", file=sys.stderr)
            return 2
        todo, stale = [], 0
        for pair, order in tasks:
            key = key_of(pair, order)
            jp = out / "judge" / f"{key}.json"
            if not jp.exists() or args.force:
                todo.append((pair, order))
                continue
            try:
                jd = load_judgment(jp)
            except ValueError as e:
                stale += 1
                log(f"[{key}] cached judgment fails validation; re-judging ({e})")
                todo.append((pair, order))
                continue
            if jd.get("workspace_sha256") and jd["workspace_sha256"] != workspace_sha256(out / "workspaces" / key):
                why = "a different workspace"
            elif jd.get("prompt_sha256", "") != prompt_sha256():
                why = "a different judge prompt"
            elif jd.get("max_per_program") != args.max_per_program:
                why = "a different --max-per-program"
            elif jd.get("model") != args.model:
                why = f"a different model ({jd.get('model')})"
            else:
                judged[key] = jd
                continue
            stale += 1
            log(f"[{key}] cached judgment was made on {why}; re-judging")
            todo.append((pair, order))
        log(f"judging {len(todo)} of {len(tasks)} judgments with {args.model} "
            f"(max_turns={args.max_turns}, timeout={args.timeout}s; {len(judged)} cached, {stale} stale)")

        def _one(pair: Pair, order: str) -> tuple[str, dict]:
            key = key_of(pair, order)
            t0 = time.time()
            res = judge_workspace(out / "workspaces" / key, pair_id=key, model=args.model,
                                  max_turns=args.max_turns, timeout_seconds=args.timeout,
                                  max_per_program=args.max_per_program, retries=args.retries,
                                  transcript_path=out / "judge" / f"{key}.transcript.json", log=log)
            progs = pair.programs(order)
            res["order"] = order
            res["arm_of"] = {p: progs[p].arm for p in PROGRAMS}
            res["run_of"] = {p: progs[p].run_id for p in PROGRAMS}
            res["workspace_stats"] = stats_all[key]
            (out / "judge" / f"{key}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
            top = top_k_counts(res["judgment"], args.top_k)
            sh = shared_ids(res["judgment"])
            u = res["usage"]
            log(f"[{key}] {count_label(args.top_k)} A({progs['A'].arm})={top['A']} B({progs['B'].arm})={top['B']} "
                f"left out as shared A={len(sh['A'])} B={len(sh['B'])} tool_calls={res['n_tool_calls']} "
                f"retries={res['retries']} in={u.get('input_tokens')} out={u.get('output_tokens')} "
                f"{time.time() - t0:.0f}s")
            return key, res

        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futs = {ex.submit(_one, p, o): key_of(p, o) for p, o in todo}
            for fut in as_completed(futs):
                try:
                    key, res = fut.result()
                    judged[key] = res
                except JudgeError as e:
                    failures.append(futs[fut])
                    log(f"[{futs[fut]}] FAILED (not retryable): {e}")
                except Exception as e:  # noqa: BLE001
                    failures.append(futs[fut])
                    log(f"[{futs[fut]}] FAILED: {e}")
        if failures:
            log(f"{len(failures)} judgments failed: {failures}")
    else:
        for pair, order in tasks:
            key = key_of(pair, order)
            jp = out / "judge" / f"{key}.json"
            if not jp.exists():
                continue
            try:
                judged[key] = load_judgment(jp)
            except ValueError as e:
                raise SystemExit(str(e))
            if judged[key].get("workspace_sha256") not in (None, "", workspace_sha256(out / "workspaces" / key)):
                log(f"[{key}] WARNING: judgment was made on a different workspace than workspaces/ now holds")
            if judged[key].get("prompt_sha256", "") != prompt_sha256():
                log(f"[{key}] WARNING: judgment was made under a different judge prompt than judge_prompt.md now holds")
        log(f"report: {len(judged)} of {len(tasks)} judgments present")

    # Report.
    for pair in pairs:
        per_order = {o: judged[key_of(pair, o)] for o in ORDERS if key_of(pair, o) in judged}
        if per_order:
            write_review(pair, per_order, git_dirs, out / "review", args.top_k)
    if judged:
        write_aggregate(pairs, judged, out, args.top_k, unpaired)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
