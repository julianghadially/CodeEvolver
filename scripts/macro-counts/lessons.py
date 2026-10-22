#!/usr/bin/env python3
"""Prompt-lesson catalog discovery (the catalog method of ``catalog.py``, applied to prompt lessons).

Measures accumulated experience at the prompt layer: how many distinct lessons (task-specific beliefs or
decision rules in LM-visible text, ``lesson_definition.md``) each run wrote into its program, over the
whole run and in the final program.

  discover  One OpenCode session reads every paired run of a benchmark (same runs, workspace layout,
            provenance rules and judge as macro counts v5) from mechanically extracted prompt text
            (prompt_text.py) and proposes a theme-grouped lesson list with no provenance. -> drafts/<bench>.md
  (curate)  A human edits the draft into lesson_catalogs/<bench>.md.
  count     One BLIND session per distinct landed program (every landed iteration's closing commit, plus
            the seed and the final program): it sees that program's prompt-like text inline, no run name,
            and gives a present/absent verdict with a verbatim quote for EVERY catalog lesson, theme by
            theme. Programs with identical text are judged once. Per run: whole-run lessons = lessons present
            in any landed program; final lessons = lessons present in the final program; lessons the seed
            already has never count. -> LESSONS.md, lesson_counts.csv, lesson_matrix.csv

  python scripts/macro-counts/lessons.py discover --bench phantomwiki
  python scripts/macro-counts/lessons.py count --bench phantomwiki --finals-only   # 6 finals + seed: a cheap pilot
  python scripts/macro-counts/lessons.py count --bench phantomwiki
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import catalog  # noqa: E402
import runs as runs_mod  # noqa: E402
from bench_workspace import _git, input_sha256, landed_iterations  # noqa: E402
from catalog import (SOFT_PROVENANCE, _cached, _provenance_hit, _session_env, _strings, mean_sd, prepare,  # noqa: E402
                     provenance_patterns, sha256_text, summaries_block)
from prompt_text import program_text, seed_sha_of, write_prompt_text  # noqa: E402
from judge import DEFAULT_RETRIES, DEFAULT_TIMEOUT_SECONDS, JudgeError, run_json_session  # noqa: E402
from runs import PROGRAM_FILE, log  # noqa: E402

DEFAULT_OUT = Path.home() / "ce-analysis" / "prompt-lessons"
DEFINITION_PATH = HERE / "lesson_definition.md"
DISCOVER_PROMPT_PATH = HERE / "lesson_discover_prompt.md"
EXTRACT_PROMPT_PATH = HERE / "lesson_extract_prompt.md"
CONSOLIDATE_PROMPT_PATH = HERE / "lesson_consolidate_prompt.md"
COUNT_PROMPT_PATH = HERE / "lesson_count_prompt.md"
DEFAULT_CATALOG_DIR = HERE / "lesson_catalogs"
LABELS = ("Task", "Domain", "Tool", "Procedure", "Decision", "Output")

DISCOVER_SHAPE = """{
  "themes": [
    {"name": "<short benchmark-specific theme>",
     "lessons": [
       {"name": "<short name of the lesson>",
        "label": "<one of: Task, Domain, Tool, Procedure, Decision, Output>",
        "description": "<one or two sentences: the belief or rule itself, in generic words>"}
     ]}
  ],
  "notes": ["<anything the reviewer should know about the boundary or the merges; no provenance>"]
}"""


def discover_prompt(bench: str, runs: list) -> str:
    head = DISCOVER_PROMPT_PATH.read_text(encoding="utf-8").replace(
        "{definition}", DEFINITION_PATH.read_text(encoding="utf-8").strip()).rstrip()
    task = (f"## Your task now\n\nBenchmark: `{bench}`. Your working directory is its workspace; start with "
            "`RUNS.md` and `prompt_text/seed.md`, then for EVERY run its `prompt_text/<run>/final.md` and its whole "
            "`prompt_text/<run>/iterations.md`. List every prompt lesson you found, grouped into themes. Remember: nothing about where a "
            "lesson came from. End your reply with EXACTLY ONE fenced ```json block (and nothing after it) of "
            f"this shape:\n\n```json\n{DISCOVER_SHAPE}\n```\n")
    return head + "\n\n" + summaries_block(runs) + "\n" + task


def validate_discovery(d: dict, runs: list) -> list[str]:
    """Fatal: missing/empty fields, an unknown label, or provenance in a theme or lesson. A provenance hit
    in a reviewer note drops the note with a warning. Warnings: soft provenance wording, no lessons."""
    problems: list[str] = []
    warnings: list[str] = []
    themes = d.get("themes")
    if not isinstance(themes, list):
        raise ValueError("invalid discovery: 'themes' missing or not a list")
    pats = provenance_patterns(runs)
    n = 0
    for i, t in enumerate(themes):
        if not isinstance(t, dict) or not str(t.get("name") or "").strip():
            problems.append(f"themes[{i}]: empty 'name'")
            continue
        if (hit := _provenance_hit(t["name"], pats)):
            problems.append(f"theme {t['name']!r}: {hit}")
        lessons = t.get("lessons")
        if not isinstance(lessons, list) or not lessons:
            problems.append(f"theme {t['name']!r}: no lessons")
            continue
        for j, e in enumerate(lessons):
            n += 1
            where = f"theme {t['name']!r} lesson {j}"
            for f in ("name", "label", "description"):
                if not isinstance(e, dict) or not str(e.get(f) or "").strip():
                    problems.append(f"{where}: empty '{f}'")
            if isinstance(e, dict) and e.get("label") not in LABELS:
                problems.append(f"{where}: label {e.get('label')!r} is not one of {', '.join(LABELS)}")
            if (hit := next((h for s in _strings(e) if (h := _provenance_hit(s, pats))), None)):
                problems.append(f"{where}: {hit}")
    if n == 0:
        warnings.append("no lessons proposed")
    kept = []
    for note in d.get("notes") or []:
        if (hit := _provenance_hit(str(note), pats)):
            warnings.append(f"a reviewer note was removed: {hit}")
        else:
            kept.append(note)
    d["notes"] = kept
    for s in _strings({k: d.get(k) for k in ("themes", "notes")}):
        if (m := SOFT_PROVENANCE.search(s)):
            warnings.append(f"possible provenance wording '{m.group(0)}' in: {s[:120]}")
    if problems:
        raise ValueError("invalid discovery: " + "; ".join(problems))
    return warnings


def render_draft(bench: str, d: dict, header: str, warnings: list[str]) -> str:
    lessons = [(t["name"], e) for t in d["themes"] for e in t["lessons"]]
    by_label = {lab: sum(1 for _, e in lessons if e["label"] == lab) for lab in LABELS}
    lines = [f"# Prompt-lesson catalog: {bench}\n", header.rstrip() + "\n",
             f"{len(lessons)} lessons in {len(d['themes'])} themes. By label: "
             + ", ".join(f"{k} {v}" for k, v in by_label.items()) + ".\n", "## Lessons\n"]
    n = 0
    for ti, t in enumerate(d["themes"], 1):
        lines.append(f"### T{ti}: {t['name']}\n")
        for e in t["lessons"]:
            n += 1
            lines += [f"#### L{n}: {e['name']}", f"Label: {e['label']}", f"Description: {e['description']}", ""]
    notes = list(d.get("notes") or []) + [f"(validator) {w}" for w in warnings]
    if notes:
        lines.append("## Notes from discovery (ignored by the parser)\n")
        lines += [f"- {x}" for x in notes] + [""]
    return "\n".join(lines)


def run_discover(args, bench: str) -> None:
    runs, ws, stats = prepare(args, bench)
    sizes = write_prompt_text(ws, runs, stats["program_file"])
    log(f"[{bench}] prompt_text/: " + "; ".join(f"{r} {n:,} chars of iteration diffs" for r, n in sizes.items()))
    prompt = discover_prompt(bench, runs)
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)
    (args.out / "prompts" / f"{bench}.discover.md").write_text(prompt, encoding="utf-8")
    log(f"[{bench}] discover prompt {len(prompt):,} chars (~{len(prompt) // 4:,} tok) -> prompts/{bench}.discover.md")
    if args.dry_run:
        return
    want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "input_sha256": input_sha256(ws, runs)}
    jpath = args.out / "drafts" / f"{bench}.json"
    jd = _cached(jpath, want, args.force)
    if jd is None:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jd = run_json_session(ws, prompt, required_key="themes", label=f"{bench} lessons discover",
                              validate=lambda d: validate_discovery(d, runs), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=True, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"bench": bench, "stage": "discover", **want, "runs": [r.run_id for r in runs], **jd}
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
    header = (f"Draft from the discovery session ({jd['model']}, {jd['judged_at']}). Curate it: merge, split, "
              "delete, relabel, move between themes and renumber freely.")
    md = render_draft(bench, jd["result"], header, jd.get("warnings") or [])
    (args.out / "drafts" / f"{bench}.md").write_text(md, encoding="utf-8")
    u = jd.get("usage") or {}
    n = sum(len(t["lessons"]) for t in jd["result"]["themes"])
    log(f"[{bench}] draft: {n} lessons in {len(jd['result']['themes'])} themes -> drafts/{bench}.md "
        f"(tool_calls={jd['n_tool_calls']} retries={jd['retries']} in={u.get('input_tokens')} out={u.get('output_tokens')})")


# ---------------------------------------------------------------------------
# Two-stage discovery: one extraction session per run, then one blind consolidation session
# ---------------------------------------------------------------------------

EXTRACT_SHAPE = """{
  "lessons": [
    {"name": "<short name of the lesson>",
     "label": "<one of: Task, Domain, Tool, Procedure, Decision, Output>",
     "description": "<one or two sentences: the belief or rule itself, in generic words>"}
  ],
  "notes": ["<anything about the boundary or the merges; no provenance>"]
}"""


def extract_prompt(bench: str, run, ws: Path) -> str:
    """The run's whole prompt history inline: the session needs no tool, so it cannot skim with grep or sed."""
    head = (EXTRACT_PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{definition}", DEFINITION_PATH.read_text(encoding="utf-8").strip()).rstrip())
    pt = ws / "prompt_text"
    body = "\n\n".join([
        "## Text 1: the seed program\n\n" + (pt / "seed.md").read_text(encoding="utf-8"),
        "## Text 2: the final program\n\n" + (pt / run.run_id / "final.md").read_text(encoding="utf-8"),
        "## Text 3: every landed iteration's changes\n\n" + (pt / run.run_id / "iterations.md").read_text(encoding="utf-8"),
    ])
    # Run names inside the texts' own headings would leak provenance into the lesson wording; drop them.
    body = body.replace(run.run_id, "this run")
    return (head + f"\n\n## Your task now\n\nBenchmark: `{bench}`. Read all three texts below in full, then list "
            "every prompt lesson of this run. You do not need any tool. End your reply with EXACTLY ONE fenced ```json "
            f"block (and nothing after it) of this shape:\n\n```json\n{EXTRACT_SHAPE}\n```\n\n" + body + "\n")


def validate_extract(d: dict, runs: list) -> list[str]:
    """Fatal: missing fields, unknown label, provenance in a lesson. A note with provenance is dropped."""
    lst = d.get("lessons")
    if not isinstance(lst, list):
        raise ValueError("invalid extraction: 'lessons' missing or not a list")
    return validate_discovery({"themes": [{"name": "all", "lessons": lst}] if lst else [],
                               "notes": d.get("notes")}, runs) if lst else ["no lessons extracted"]


def consolidate_prompt(bench: str, lists: list[list[dict]]) -> str:
    head = (CONSOLIDATE_PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{definition}", DEFINITION_PATH.read_text(encoding="utf-8").strip()).rstrip())
    parts = [head, "\n\n## The lists\n"]
    for i, lst in enumerate(lists, 1):
        parts.append(f"### List {i}\n")
        parts += [f"- **{e['name']}** ({e['label']}). {e['description']}" for e in lst]
        parts.append("")
    parts.append(f"## Your task now\n\nBenchmark: `{bench}`. Merge the lists into one catalog grouped into themes. End "
                 "your reply with EXACTLY ONE fenced ```json block (and nothing after it) of this shape:\n\n"
                 f"```json\n{DISCOVER_SHAPE}\n```\n")
    return "\n".join(parts)


def run_discover_two_stage(args, bench: str) -> None:
    import random
    runs, ws, stats = prepare(args, bench)
    sizes = write_prompt_text(ws, runs, stats["program_file"])
    log(f"[{bench}] prompt_text/: " + "; ".join(f"{r} {n:,} chars of iteration diffs" for r, n in sizes.items()))
    edir = args.out / "extract" / bench
    edir.mkdir(parents=True, exist_ok=True)
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)
    jobs = []
    for run in runs:
        prompt = extract_prompt(bench, run, ws)
        (args.out / "prompts" / f"{bench}.extract.{run.run_id}.md").write_text(prompt, encoding="utf-8")
        want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "input_sha256": input_sha256(ws, [run])}
        jobs.append((run, prompt, want))
    if args.dry_run:
        log(f"[{bench}] {len(jobs)} extraction prompts written; consolidation follows them")
        return

    def _one(run, prompt, want) -> dict:
        jpath = edir / f"{run.run_id}.json"
        jd = _cached(jpath, want, args.force)
        if jd is None:
            blind = args.out / "extract-blind"
            blind.mkdir(parents=True, exist_ok=True)
            jd = run_json_session(blind, prompt, required_key="lessons", label=f"{bench} extract {run.run_id}",
                                  validate=lambda d: validate_extract(d, runs), model=args.model,
                                  max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                                  allow_bash=False, transcript_path=jpath.with_suffix(".transcript.json"),
                                  rejected_dir=args.out / "rejected", log=log)
            jd = {"bench": bench, "stage": "extract", "run_id": run.run_id, **want, **jd}
            jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
            log(f"[{bench}] {run.run_id}: {len(jd['result']['lessons'])} lessons (tool_calls={jd['n_tool_calls']} "
                f"retries={jd['retries']})")
        return jd

    extracted: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, *j): j[0].run_id for j in jobs}
        for fut in as_completed(futs):
            extracted[futs[fut]] = fut.result()
    # Shuffle with a seed fixed per benchmark so the consolidation prompt (and its cache key) is reproducible.
    order = sorted(extracted)
    random.Random(f"lessons-{bench}").shuffle(order)
    lists = [extracted[rid]["result"]["lessons"] for rid in order]
    prompt = consolidate_prompt(bench, lists)
    (args.out / "prompts" / f"{bench}.consolidate.md").write_text(prompt, encoding="utf-8")
    want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "input_sha256": input_sha256(ws, runs)}
    jpath = args.out / "drafts" / f"{bench}.json"
    jd = _cached(jpath, want, args.force)
    if jd is None:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        blind = args.out / "consolidate-blind"
        blind.mkdir(parents=True, exist_ok=True)
        jd = run_json_session(blind, prompt, required_key="themes", label=f"{bench} consolidate",
                              validate=lambda d: validate_discovery(d, runs), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=False, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"bench": bench, "stage": "discover", "method": "two-stage", "list_order": order, **want,
              "runs": [r.run_id for r in runs], **jd}
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
    header = (f"Draft from two-stage discovery ({jd['model']}, {jd['judged_at']}): one extraction session per run "
              f"({', '.join(str(len(l)) for l in lists)} lessons, lists shuffled), then one blind consolidation "
              "session. Curate it: merge, split, delete, relabel, move between themes and renumber freely.")
    (args.out / "drafts" / f"{bench}.md").write_text(render_draft(bench, jd["result"], header, jd.get("warnings") or []),
                                                    encoding="utf-8")
    n = sum(len(t["lessons"]) for t in jd["result"]["themes"])
    log(f"[{bench}] draft: {n} lessons in {len(jd['result']['themes'])} themes -> drafts/{bench}.md")


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------

THEME_RE = re.compile(r"^###\s+(T\d+)\s*[:.\-]\s*(.+?)\s*$")
LESSON_RE = re.compile(r"^####\s+(L\d+)\s*[:.\-]\s*(.+?)\s*$")
LFIELD_RE = re.compile(r"^(Label|Description)\s*:\s*(.*)$", re.I)


def parse_catalog(text: str) -> dict:
    """``## Lessons`` -> ``### T<n>: theme`` -> ``#### L<n>: name`` with ``Label:`` and ``Description:``
    (the description may continue on following lines). Other ``##`` sections are ignored. Raises
    ValueError on a malformed catalog."""
    themes: list[dict] = []
    problems: list[str] = []
    in_lessons = False
    theme = lesson = field = None
    for n, line in enumerate(text.splitlines(), 1):
        if line.startswith("## ") and not line.startswith("###"):
            in_lessons = line[3:].strip().lower() == "lessons"
            theme = lesson = field = None
            continue
        if not in_lessons:
            continue
        if (m := THEME_RE.match(line)):
            theme = {"id": m.group(1), "name": m.group(2), "lessons": []}
            themes.append(theme)
            lesson = field = None
        elif (m := LESSON_RE.match(line)):
            if theme is None:
                problems.append(f"line {n}: lesson {m.group(1)} outside a theme")
                continue
            lesson = {"id": m.group(1), "name": m.group(2), "label": "", "description": ""}
            theme["lessons"].append(lesson)
            field = None
        elif lesson is not None and (m := LFIELD_RE.match(line.strip())):
            field = m.group(1).lower()
            lesson[field] = m.group(2).strip()
        elif lesson is not None and field == "description" and line.strip():
            lesson["description"] = (lesson["description"] + " " + line.strip()).strip()
    lessons = [e for t in themes for e in t["lessons"]]
    for kind, ids in (("theme", [t["id"] for t in themes]), ("lesson", [e["id"] for e in lessons])):
        for i in sorted({i for i in ids if ids.count(i) > 1}):
            problems.append(f"duplicate {kind} id {i}")
    for e in lessons:
        if not e["description"]:
            problems.append(f"{e['id']}: empty Description")
        if e["label"] and e["label"] not in LABELS:
            problems.append(f"{e['id']}: label {e['label']!r} is not one of {', '.join(LABELS)}")
    for t in themes:
        if not t["lessons"]:
            problems.append(f"{t['id']}: no lessons")
    if not lessons:
        problems.append("no lessons under '## Lessons'")
    if problems:
        raise ValueError("invalid lesson catalog: " + "; ".join(problems))
    return {"themes": themes}


def load_catalog(catalog_dir: Path, bench: str) -> dict:
    path = catalog_dir / f"{bench}.md"
    if not path.exists():
        raise SystemExit(f"no lesson catalog for {bench}: {path}")
    return parse_catalog(path.read_text(encoding="utf-8"))


def catalog_lessons(cat: dict) -> list[dict]:
    return [{**e, "theme": t["id"]} for t in cat["themes"] for e in t["lessons"]]


def render_catalog_for_prompt(cat: dict) -> str:
    lines = []
    for t in cat["themes"]:
        lines.append(f"### {t['id']}: {t['name']}\n")
        lines += [f"- **{e['id']}: {e['name']}.** {e['description']}" for e in t["lessons"]]
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Count: one blind session per distinct program
# ---------------------------------------------------------------------------

COUNT_SHAPE = """{
  "themes": [
    {"id": "T<n>",
     "lessons": [
       {"id": "L<n>", "present": <true|false>, "quote": "<verbatim quote when present, else empty>"}
     ]}
  ],
  "uncatalogued": [
    {"name": "<short name>", "description": "<one sentence: the belief or rule>", "quote": "<verbatim quote>"}
  ]
}"""


def count_prompt(cat: dict, text: str) -> str:
    head = (COUNT_PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{definition}", DEFINITION_PATH.read_text(encoding="utf-8").strip())
            .replace("{catalog}", render_catalog_for_prompt(cat).rstrip())
            .replace("{program_text}", text.rstrip()).rstrip())
    task = ("\n\n## Your task now\n\nGive a verdict for every catalog lesson, theme by theme in catalog order. "
            "Answer from the text above; you do not need any tool. End your reply with EXACTLY ONE fenced ```json "
            f"block (and nothing after it) of this shape:\n\n```json\n{COUNT_SHAPE}\n```\n")
    return head + task


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\"'`*‘’“”]", "", s)).strip().lower()


def quote_in(quote: str, text: str) -> bool:
    q = _norm(quote).rstrip(".").strip()
    return bool(q) and q in _norm(text)


def validate_count(c: dict, cat: dict, text: str) -> list[str]:
    """Fatal: a catalog lesson without exactly one verdict, an unknown id, a non-boolean verdict, or a
    present verdict without a quote. Warning: a quote not found verbatim in the text."""
    want = [e["id"] for e in catalog_lessons(cat)]
    got: list[str] = []
    problems: list[str] = []
    warnings: list[str] = []
    themes = c.get("themes")
    if not isinstance(themes, list):
        raise ValueError("invalid count: 'themes' missing or not a list")
    for t in themes:
        for v in (t.get("lessons") if isinstance(t, dict) else None) or []:
            if not isinstance(v, dict):
                problems.append("a verdict is not an object")
                continue
            lid = str(v.get("id"))
            got.append(lid)
            if not isinstance(v.get("present"), bool):
                problems.append(f"{lid}: 'present' is not true/false")
            elif v["present"]:
                if not str(v.get("quote") or "").strip():
                    problems.append(f"{lid}: present without a quote")
                elif not quote_in(v["quote"], text):
                    warnings.append(f"{lid}: quote not found verbatim")
    missing = [i for i in want if i not in got]
    unknown = sorted(set(got) - set(want))
    dup = sorted({i for i in got if got.count(i) > 1})
    if missing:
        problems.append(f"no verdict for {', '.join(missing)}")
    if unknown:
        problems.append(f"unknown ids {', '.join(unknown)}")
    if dup:
        problems.append(f"more than one verdict for {', '.join(dup)}")
    if problems:
        raise ValueError("invalid count: " + "; ".join(problems))
    return warnings


def verdicts(result: dict) -> dict[str, dict]:
    return {str(v["id"]): v for t in result["themes"] for v in t["lessons"]}


SESSIONS_ROOT = Path.home() / ".codeevolver" / "opencode-sessions"


def _drop_session_dir(data_dir: str | None) -> None:
    """Delete a finished count session's OpenCode data dir once its verdict (and exported transcript) is saved.
    Hundreds of count sessions otherwise pile up their databases and logs. Only inspection dirs under
    ~/.codeevolver/opencode-sessions are touched."""
    import shutil
    if not data_dir:
        return
    d = Path(data_dir).resolve()
    if d.parent == SESSIONS_ROOT.resolve() and d.name.startswith("inspection-"):
        shutil.rmtree(d, ignore_errors=True)


def _sweep_config_dirs(tmp: Path, older_than_s: float) -> None:
    """The OpenCode harness makes a ~60MB temp config dir per session (`_prepare_opencode_config_dir`, with a
    node_modules plugin install) and never deletes it; hundreds of judge sessions fill the disk. ``main`` points
    this process's tempdir at ``tmp``, and every finished session sweeps the config dirs there older than the
    longest a session can run, so in-flight sessions are never touched."""
    import shutil
    import time
    cutoff = time.time() - older_than_s
    for d in tmp.glob("codeevolver-opencode-*"):
        try:
            if d.is_dir() and d.stat().st_mtime < cutoff:
                shutil.rmtree(d, ignore_errors=True)
        except FileNotFoundError:
            pass


def run_programs(ws: Path, run, prog: str, finals_only: bool) -> list[dict]:
    """The programs of one run to judge: the seed, every landed iteration's closing commit, the final
    program. Each with its text hash."""
    repo = ws / "repos" / run.run_id
    seed = seed_sha_of(repo, run)
    final = (run.status.get("final_candidate") or {}).get("git_sha") or _git(repo, "rev-parse", "HEAD").strip()
    items = [{"key": "seed", "iteration": 0, "sha": seed, "parent_sha": None, "parent_key": None}]
    if not finals_only:
        for n in sorted(landed_iterations(run)):
            rec = run.iters[str(n)]
            if rec.get("close_sha"):
                # The parent by iteration, not by parent_sha: in pareto runs parent_sha is a later commit on the
                # parent's branch (same program text, different sha), so it never matches a closing commit.
                p = rec.get("parent_iteration")
                items.append({"key": f"it{n}", "iteration": n, "sha": rec["close_sha"],
                              "parent_sha": rec.get("parent_sha"),
                              "parent_key": None if p is None else ("seed" if int(p) == 0 else f"it{int(p)}")})
    items.append({"key": "final", "iteration": None, "sha": final, "parent_sha": None, "parent_key": None})
    for it in items:
        it["text"] = program_text(repo, it["sha"], seed, prog)
        it["hash"] = sha256_text(it["text"])
    return items


def run_count(args, bench: str) -> list[str]:
    cat = load_catalog(args.catalog_dir, bench)
    catalog_sha = sha256_text(render_catalog_for_prompt(cat))
    runs, ws, stats = prepare(args, bench)
    prog = stats["program_file"]
    cdir = args.out / "count" / bench
    pdir = cdir / "programs"
    pdir.mkdir(parents=True, exist_ok=True)
    per_run = {r.run_id: run_programs(ws, r, prog, args.finals_only) for r in runs}
    texts = {it["hash"]: it["text"] for items in per_run.values() for it in items}
    todo = []
    for h, text in texts.items():
        (pdir / f"{h[:16]}.md").write_text(text + "\n", encoding="utf-8")
        prompt = count_prompt(cat, text)
        want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "catalog_sha256": catalog_sha}
        if _cached(pdir / f"{h[:16]}.json", want, args.force) is None:
            todo.append((h, prompt, want))
    n_items = sum(len(v) for v in per_run.values())
    log(f"[{bench}] {len(catalog_lessons(cat))} lessons in {len(cat['themes'])} themes; {n_items} programs across "
        f"{len(runs)} runs, {len(texts)} distinct texts, {len(todo)} to judge")
    if args.dry_run:
        if todo:
            (args.out / "prompts").mkdir(parents=True, exist_ok=True)
            (args.out / "prompts" / f"{bench}.count.example.md").write_text(todo[0][1], encoding="utf-8")
        return []
    blind = cdir / "blind"  # the sessions' working directory: holds nothing that names a run
    blind.mkdir(parents=True, exist_ok=True)

    def _one(h: str, prompt: str, want: dict) -> None:
        jpath = pdir / f"{h[:16]}.json"
        jd = run_json_session(blind, prompt, required_key="themes", label=f"{bench} lessons {h[:12]}",
                              validate=lambda c: validate_count(c, cat, texts[h]), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=False, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"bench": bench, "stage": "count", "text_sha256": h, **want, **jd}
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
        _drop_session_dir(jd.get("data_dir"))
        _sweep_config_dirs(args.tmp, args.timeout + 600)
        n = sum(1 for v in verdicts(jd["result"]).values() if v["present"])
        log(f"[{bench}] {h[:12]}: {n} present, {len(jd.get('warnings') or [])} warnings, retries={jd['retries']}")

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, *t): t[0][:12] for t in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(futs[fut])
                log(f"[{bench}] {futs[fut]}: FAILED: {e}")
    manifest = {rid: [{k: it[k] for k in ("key", "iteration", "sha", "parent_sha", "parent_key", "hash")} for it in items]
                for rid, items in per_run.items()}
    (cdir / "manifest.json").write_text(json.dumps({"finals_only": args.finals_only, "runs": manifest}, indent=1),
                                        encoding="utf-8")
    return failed


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def assemble_run(items: list[dict], judged: dict[str, dict], texts: dict[str, str], cat: dict,
                 reconcile: bool = True) -> dict:
    """Per-run lesson sets from the per-program verdicts. Seed lessons never count.

    Flicker (raw verdicts): a lesson present in a parent program and absent in its child although the parent's
    quote is still in the child's text; the judge's miss rate. With ``reconcile`` (the headline), a lesson is
    present in a program when the judge marked it OR any quote a judge gave for it in this run appears verbatim in
    the program's text: identical text gets identical verdicts, the same in both arms."""
    raw = {it["key"]: {i for i, v in verdicts(judged[it["hash"]]["result"]).items() if v["present"]} for it in items}
    quotes: dict[str, set[str]] = {}
    for it in items:
        for lid, v in verdicts(judged[it["hash"]]["result"]).items():
            if v["present"] and quote_in(v.get("quote") or "", texts[it["hash"]]):
                quotes.setdefault(lid, set()).add(v["quote"])
    present = {k: set(v) for k, v in raw.items()}
    if reconcile:
        for it in items:
            present[it["key"]] |= {lid for lid, qs in quotes.items() if any(quote_in(q, texts[it["hash"]]) for q in qs)}
    seed = present["seed"]
    landed = [it for it in items if it["key"].startswith("it")]
    whole = (set().union(*(present[it["key"]] for it in landed)) if landed else set()) | present["final"]
    whole -= seed
    final = present["final"] - seed
    by_key = {it["key"]: it for it in items}
    flicker = []
    for it in landed:
        par = by_key.get(it.get("parent_key") or "")
        if not par:  # a parent that did not land has no verdict of its own
            continue
        pv = verdicts(judged[par["hash"]]["result"])
        for lid in raw[par["key"]] - raw[it["key"]]:
            if quote_in(pv[lid].get("quote") or "", texts[it["hash"]]):
                flicker.append({"lesson": lid, "parent": par["key"], "child": it["key"]})
    theme_of = {e["id"]: e["theme"] for e in catalog_lessons(cat)}
    order = lambda xs: sorted(xs, key=lambda x: int(x[1:]))  # noqa: E731
    return {"whole": order(whole), "final": order(final), "lost": order(whole - final), "seed": sorted(seed),
            "themes_whole": sorted({theme_of[i] for i in whole}), "themes_final": sorted({theme_of[i] for i in final}),
            "present": {k: order(v) for k, v in present.items()}, "flicker": flicker}


def run_report(args, bench: str) -> None:
    cdir = args.out / "count" / bench
    man = json.loads((cdir / "manifest.json").read_text(encoding="utf-8"))
    cat = load_catalog(args.catalog_dir, bench)
    lessons = catalog_lessons(cat)
    judged, texts = {}, {}
    for items in man["runs"].values():
        for it in items:
            h = it["hash"]
            if h not in judged:
                judged[h] = json.loads((cdir / "programs" / f"{h[:16]}.json").read_text(encoding="utf-8"))
                texts[h] = (cdir / "programs" / f"{h[:16]}.md").read_text(encoding="utf-8")
    runs = {r.run_id: r for r in bench_runs_for(args, bench)}
    res = {rid: assemble_run(items, judged, texts, cat) for rid, items in man["runs"].items()}
    raw = {rid: assemble_run(items, judged, texts, cat, reconcile=False) for rid, items in man["runs"].items()}
    arms = [args.arm1, args.arm2]
    order = sorted(res, key=lambda rid: (arms.index(runs[rid].arm) if runs[rid].arm in arms else 9, runs[rid].seed))
    warn = sum(len(j.get("warnings") or []) for j in judged.values())
    md = [f"# Prompt lessons per run: {bench}\n",
          f"Generated {dt.datetime.now().isoformat(timespec='minutes')}. Catalog `lesson_catalogs/{bench}.md` "
          f"({len(lessons)} lessons, {len(cat['themes'])} themes). Judge {next(iter(judged.values()))['model']}, one blind "
          f"session per distinct program ({len(judged)} judged). "
          + ("**Finals only: whole-run numbers are the final program's.** " if man["finals_only"] else "")
          + "Whole run = lessons in any landed program; final = lessons in the final program; lost = whole - final. "
          "Seed lessons excluded. Counts are quote-reconciled: a lesson is present in a program when the judge marked "
          "it or a quote a judge gave for it in the same run appears verbatim in the program's text. Flicker counts "
          "the raw verdicts (present in a parent, absent in the child, parent's quote still in the child). "
          f"Quotes not found verbatim: {warn}.\n",
          "| arm | runs | whole-run lessons | final lessons | lost | whole-run themes | final themes | flicker |",
          "|---|---|---|---|---|---|---|---|"]
    for arm in arms:
        ar = [rid for rid in order if runs[rid].arm == arm]
        if not ar:
            continue
        def ms(key):
            m, sd = mean_sd([len(res[r][key]) for r in ar])
            return f"{m:.1f} ± {sd:.1f}"
        md.append(f"| {arm} | {len(ar)} | {ms('whole')} | {ms('final')} | {ms('lost')} | {ms('themes_whole')} | "
                  f"{ms('themes_final')} | {sum(len(res[r]['flicker']) for r in ar)} |")
    md.append("\n| run | whole | final | lost | themes whole | themes final | flicker | raw whole | raw final |\n"
              "|---|---|---|---|---|---|---|---|---|")
    for rid in order:
        x = res[rid]
        md.append(f"| {rid} | {len(x['whole'])} | {len(x['final'])} | {len(x['lost'])} | {len(x['themes_whole'])} | "
                  f"{len(x['themes_final'])} | {len(x['flicker'])} | {len(raw[rid]['whole'])} | {len(raw[rid]['final'])} |")
    md.append("\n## Lesson x run (W = whole run only, F = in the final program)\n")
    md.append("| lesson | " + " | ".join(f"{runs[r].arm[:4]} s{runs[r].seed}" for r in order) + " |\n|---|"
              + "---|" * len(order))
    for e in lessons:
        cells = ["F" if e["id"] in res[r]["final"] else ("W" if e["id"] in res[r]["whole"] else "") for r in order]
        md.append(f"| {e['id']} {e['name']} | " + " | ".join(cells) + " |")
    fl = [(rid, f) for rid in order for f in res[rid]["flicker"]]
    if fl:
        md.append("\n## Flicker (present in parent, absent in child, parent's quote still in the child)\n")
        md += [f"- {rid}: {f['lesson']} {f['parent']} -> {f['child']}" for rid, f in fl]
    (args.out / f"LESSONS_{bench}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with (args.out / f"lesson_counts_{bench}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bench", "run_id", "arm", "seed", "whole", "final", "lost", "themes_whole", "themes_final",
                    "flicker", "raw_whole", "raw_final", "whole_ids", "final_ids", "seed_test", "final_test"])
        for rid in order:
            x, r = res[rid], runs[rid]
            w.writerow([bench, rid, r.arm, r.seed, len(x["whole"]), len(x["final"]), len(x["lost"]),
                        len(x["themes_whole"]), len(x["themes_final"]), len(x["flicker"]), len(raw[rid]["whole"]),
                        len(raw[rid]["final"]), " ".join(x["whole"]),
                        " ".join(x["final"]), r.score("seed_test_score"), r.score("final_test_score")])
    with (args.out / f"lesson_matrix_{bench}.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "program", "iteration", "sha", "lesson", "present", "quote"])
        for rid in order:
            for it in man["runs"][rid]:
                for lid, v in verdicts(judged[it["hash"]]["result"]).items():
                    w.writerow([rid, it["key"], it["iteration"], it["sha"], lid, v["present"], v.get("quote") or ""])
    (args.out / f"lesson_runs_{bench}.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    log(f"[{bench}] wrote LESSONS_{bench}.md, lesson_counts_{bench}.csv, lesson_matrix_{bench}.csv")


def bench_runs_for(args, bench: str):
    from bench_workspace import bench_runs
    return bench_runs(args.archive_root, bench, args.arm1, args.arm2)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["discover", "check", "count", "report"])
    ap.add_argument("--bench", action="append", required=True)
    ap.add_argument("--archive-root", type=Path, default=catalog.DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--arm1", default=catalog.DEFAULT_ARM1)
    ap.add_argument("--arm2", default=catalog.DEFAULT_ARM2)
    ap.add_argument("--void", default=None, metavar="REGEX",
                    help="run ids left out (default: pareto_evolution s1-s3, superseded by the -redo runs; '' = none)")
    ap.add_argument("--program-file", action="append", default=[], metavar="BENCH=PATH")
    ap.add_argument("--model", default=catalog.DEFAULT_MODEL, help="OpenCode provider/model id")
    ap.add_argument("--max-turns", type=int, default=catalog.DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per session (hard kill)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="fresh sessions to try")
    ap.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    ap.add_argument("--single-session", action="store_true",
                    help="discover: the original one-session discovery (used for phantomwiki) instead of two-stage")
    ap.add_argument("--finals-only", action="store_true", help="count: judge only the seed and the final programs")
    ap.add_argument("--jobs", type=int, default=4, help="count: parallel sessions")
    ap.add_argument("--force", action="store_true", help="re-run even when a matching result is cached")
    ap.add_argument("--dry-run", action="store_true", help="build the workspace and prompt only; no session")
    args = ap.parse_args()
    if args.void is not None:
        runs_mod.VOID_RUN_RE = re.compile(args.void) if args.void else None
    for spec in args.program_file:
        bench, _, path = spec.partition("=")
        if not path:
            ap.error(f"--program-file expects BENCH=PATH, got {spec!r}")
        PROGRAM_FILE[bench] = path
    if args.stage == "check":
        for bench in args.bench:
            cat = load_catalog(args.catalog_dir, bench)
            print(f"{bench}: OK, {len(catalog_lessons(cat))} lessons in {len(cat['themes'])} themes")
        return 0
    if args.stage == "report":
        for bench in args.bench:
            run_report(args, bench)
        return 0
    # A private tempdir for the harness's per-session config dirs, which it never deletes (see _sweep_config_dirs).
    import tempfile
    args.tmp = args.out / "tmp"
    args.tmp.mkdir(parents=True, exist_ok=True)
    tempfile.tempdir = str(args.tmp)
    os.environ["TMPDIR"] = str(args.tmp)
    if not args.dry_run:
        _session_env(args)
    if args.stage == "discover":
        failed = []
        for bench in args.bench:
            try:
                (run_discover if args.single_session else run_discover_two_stage)(args, bench)
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(bench)
                log(f"[{bench}] discover FAILED: {e}")
        return 1 if failed else 0
    failed: list[str] = []
    for bench in args.bench:
        bench_failed = run_count(args, bench)
        failed += [f"{bench}/{h}" for h in bench_failed]
        if not args.dry_run and not bench_failed:
            run_report(args, bench)
    if failed:
        log(f"{len(failed)} program(s) failed: {failed}; re-run `count` (judged programs are cached)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
