#!/usr/bin/env python3
"""Macro-approach count against a curated per-benchmark catalog (macro counts v5).

Secondary result for the CoRL workshop paper: how many distinct macro
approaches each run landed. A macro approach is a distinct, non-incremental
change in how the program solves the task (``macro_definition.md``). Three steps
per benchmark:

  1. discover  One OpenCode session reads all runs of the benchmark (the change
               summaries in the prompt; memory, repo clones and the seed on disk)
               and proposes macro approaches, each with why it is macro. The
               list names no run, arm,
               seed or iteration, so the reviewer curates it without knowing
               where an approach came from. -> drafts/<bench>.md
  2. curate    A human edits the draft into catalogs/<bench>.md (in this folder,
               a method input like the prompts). ``check`` validates it.
  3. count     One OpenCode session PER RUN marks which catalog approaches (M<n>)
               that run landed, citing landed iterations and code. Each approach
               counts at most once per run; approaches outside the catalog go in
               a separate, uncounted ``uncatalogued`` list. The code keeps only
               hits that cite an iteration that landed (changes_landed in
               history.json). -> count/<bench>/<run_id>.json
  report       COUNTS.md, counts.csv, review/<bench>.md from the count judgments.

A run's count = the number of catalog macro approaches it landed. The headline
is each arm's mean +- sd over its seeds.

HOW TO RUN (from the repo root; needs ``opencode`` >= 1.17.11, ``rg`` and ``git`` on PATH)

  python scripts/macro-counts/catalog.py discover --bench arcagi2 --dry-run   # workspace + prompt only
  python scripts/macro-counts/catalog.py discover --bench arcagi2             # one session
  cp ~/ce-analysis/macro-counts/catalog-v5/drafts/arcagi2.md scripts/macro-counts/catalogs/arcagi2.md
  #   ... edit the catalog ...
  python scripts/macro-counts/catalog.py check --bench arcagi2
  python scripts/macro-counts/catalog.py count --bench arcagi2               # one session per run, then report
  python scripts/macro-counts/catalog.py report                              # all counted benches

Needs ANTHROPIC_API_KEY (default judge anthropic/claude-sonnet-5-5) in the
environment or a .env at the repo root / this folder. Sessions are not
blinded: they may read scores in the run memories, and the prompts say scores
are irrelevant. Read-only against the run archives.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import runs as runs_mod  # noqa: E402
from bench_workspace import (_git, bench_runs, change_summaries, input_sha256, landed_iterations,  # noqa: E402
                             materialize_bench, memory_dir, processed_summaries_path, raw_change_summaries)
from judge import (DEFAULT_RETRIES, DEFAULT_TIMEOUT_SECONDS, JudgeError,  # noqa: E402
                   load_env_key, prepare_engine_env, provider_key, run_json_session)
from runs import BENCHES, PROGRAM_FILE, Run, log  # noqa: E402

REPO_ROOT = HERE.parent.parent
DEFAULT_ARCHIVE_ROOT = Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser()
DEFAULT_OUT = Path.home() / "ce-analysis" / "macro-counts" / "catalog-v5"
DEFAULT_CATALOG_DIR = HERE / "catalogs"
DEFAULT_ARM1 = "pareto_evolution"
DEFAULT_ARM2 = "freeform"
DEFAULT_MAX_TURNS = 200
# v5 judge (discover and count). v4 (macro_counts.py) keeps judge.DEFAULT_MODEL (GLM 5.2) for reproducibility.
DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"
DEFINITION_PATH = HERE / "macro_definition.md"
DISCOVER_PROMPT_PATH = HERE / "discover_prompt.md"
COUNT_PROMPT_PATH = HERE / "count_prompt.md"
FILL_PROMPT_PATH = HERE / "fill_prompt.md"
MAX_DIFF_CHARS = 40_000  # per missing iteration in the fill prompt; the session can read the rest with git
# The generic macro patterns help discovery find candidates; the count session may only match catalog entries,
# so its copy of the definition stops before that section.
MACRO_PATTERNS_HEADING = "## Macro approaches (generic patterns)"

DISCOVER_SHAPE = """{
  "macro_approaches": [
    {"name": "<short generic name of the mechanism>",
     "description": "<one or two sentences: the mechanism and how it changes the way the task is solved>",
     "why_macro": "<one or two sentences: why this is non-incremental relative to the seed>"}
  ],
  "notes": ["<anything the reviewer should know about the boundary; no provenance>"]
}"""

FILL_SHAPE = """{
  "summaries": [
    {"iteration": <N>, "summary": "<three to eight sentences: what the change does to how the program works>"}
  ]
}"""

COUNT_SHAPE = """{
  "catalog_hits": [
    {"id": "M<n>", "iterations": [<every landed iteration of this run that has the mechanism>],
     "evidence": "<function names and a short quoted snippet from a cited iteration's closing commit>"}
  ],
  "uncatalogued": [
    {"name": "<short name>", "description": "<one sentence: the mechanism>",
     "iterations": [<landed iterations>], "why_macro": "<one sentence>"}
  ],
  "review_flags": ["<anything a human should double-check>"]
}"""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Catalog file: parse / render
# ---------------------------------------------------------------------------

SECTION_RE = re.compile(r"^##\s+(.+?)\s*$")
ENTRY_RE = re.compile(r"^###\s+([A-Z]\d+)\s*[:.\-]\s*(.+?)\s*$")
FIELD_RE = re.compile(r"^(Description|Why macro)\s*:\s*(.*)$", re.I)
MACRO_SECTION = "macro approaches"


def parse_catalog(text: str) -> dict:
    """Parse a catalog markdown file into {"macro": [...]}.

    Entries are ``### M<n>: name`` under ``## Macro approaches``, each followed by a
    ``Description:`` field (it may continue on the following lines). An optional
    ``Why macro:`` field is kept for the reviewer but never shown to the count
    session. Other sections (notes, reviewer comments, a "## Rejected" section
    holding entries taken out of the catalog) are ignored. There are no
    per-benchmark incremental examples: the generic incremental table in
    ``macro_definition.md`` is the only boundary, so an ``## Incremental examples``
    section or an ``I<n>`` entry is an error rather than silently ignored.
    Raises ValueError on a malformed catalog."""
    macro: list[dict] = []
    problems: list[str] = []
    in_macro = False
    entry = None
    field = None
    for n, line in enumerate(text.splitlines(), 1):
        m = SECTION_RE.match(line)
        if m and not line.startswith("###"):
            name = m.group(1).strip().lower()
            if name == "incremental examples":
                problems.append(f"line {n}: '## Incremental examples' is no longer part of the catalog; "
                                "delete it or rename it to a note section")
            in_macro = name == MACRO_SECTION
            entry = field = None
            continue
        m = ENTRY_RE.match(line)
        if m:
            if not in_macro:  # parked in another section (e.g. "## Rejected"): ignored
                entry = None
                continue
            if not m.group(1).startswith("M"):
                problems.append(f"line {n}: {m.group(1)} is not a macro id (M<n>)")
            entry = {"id": m.group(1), "name": m.group(2), "description": ""}
            macro.append(entry)
            field = None
            continue
        if entry is None:
            continue
        m = FIELD_RE.match(line.strip())
        if m:
            field = "description" if m.group(1).lower() == "description" else "why_macro"
            entry[field] = m.group(2).strip()
        elif field and line.strip():
            entry[field] = (entry[field] + " " + line.strip()).strip()
    ids = [e["id"] for e in macro]
    for i in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"duplicate id {i}")
    for e in macro:
        if not e["description"]:
            problems.append(f"{e['id']}: empty 'description'")
    if not re.search(r"^##\s+Macro approaches\s*$", text, re.I | re.M):
        problems.append("no '## Macro approaches' section (an empty section is allowed: every run then counts 0)")
    if problems:
        raise ValueError("invalid catalog: " + "; ".join(problems))
    return {"macro": macro}


def render_catalog_md(bench: str, macro: list[dict], notes: list[str] | None = None, header: str = "") -> str:
    """The catalog file format (also the draft the reviewer edits)."""
    lines = [f"# Macro-approach catalog: {bench}\n"]
    if header:
        lines.append(header.rstrip() + "\n")
    lines.append("## Macro approaches\n")
    for e in macro:
        lines += [f"### {e['id']}: {e['name']}", f"Description: {e['description']}"]
        if e.get("why_macro"):
            lines.append(f"Why macro: {e['why_macro']}")
        lines.append("")
    if notes:
        lines.append("## Notes from discovery (ignored by the parser)\n")
        lines += [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def render_catalog_for_prompt(cat: dict) -> str:
    """The parsed catalog as the count prompt shows it: names and descriptions only (no "why", no other
    sections)."""
    lines = ["### Macro approaches (the only approaches you may count)\n"]
    lines += [f"- **{e['id']}: {e['name']}.** {e['description']}" for e in cat["macro"]] or \
        ["- (none: this catalog is empty, so every run counts 0; still report `uncatalogued` approaches)"]
    return "\n".join(lines) + "\n"


def load_catalog(catalog_dir: Path, bench: str) -> dict:
    path = catalog_dir / f"{bench}.md"
    if not path.exists():
        raise SystemExit(f"no catalog for {bench}: {path} (run `discover`, then curate the draft into that file)")
    return parse_catalog(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def summaries_block(runs: list[Run]) -> str:
    parts = ["## Change summaries of every run\n",
             "Each run's `iteration_change_summaries.md`, verbatim (also in `memory/<run>/`).\n"]
    for r in runs:
        parts.append(f"### Run `{r.run_id}`\n\n{change_summaries(r)}\n")
    return "\n".join(parts)


def definition(patterns: bool = True) -> str:
    """``macro_definition.md``; without the generic macro-pattern table when ``patterns`` is False (count)."""
    text = DEFINITION_PATH.read_text(encoding="utf-8")
    if not patterns:
        text = text.split(MACRO_PATTERNS_HEADING, 1)[0]
    return text.strip()


SUMMARY_ITERATION_RE = re.compile(r"^#+ .*?\bIteration (\d+)\b", re.M)


def summarized_iterations(run: Run, text: str | None = None) -> set[int]:
    """Iteration numbers with a heading in the run's change summaries (the processed file when it exists)."""
    return {int(k) for k in SUMMARY_ITERATION_RE.findall(change_summaries(run) if text is None else text)}


def split_summary_blocks(text: str) -> tuple[str, list[tuple[int, str]]]:
    """(preamble, [(iteration, block)]) with each block starting at its ``... Iteration N`` heading."""
    starts = [(m.start(), int(m.group(1))) for m in SUMMARY_ITERATION_RE.finditer(text)]
    if not starts:
        return text.strip(), []
    blocks = [(n, text[a:b].strip()) for (a, n), (b, _) in zip(starts, starts[1:] + [(len(text), 0)])]
    return text[:starts[0][0]].strip(), blocks


def iteration_diff(ws: Path, run: Run, n: int) -> tuple[str, str, str]:
    """(parent sha, close sha, diff text) for iteration n, the diff truncated to MAX_DIFF_CHARS."""
    rec = run.iters[str(n)]
    parent, close = rec.get("parent_sha") or "", rec.get("close_sha") or ""
    if not parent or not close:
        return parent, close, "(no parent or closing commit recorded)"
    repo = ws / "repos" / run.run_id
    stat = _git(repo, "diff", "--stat", parent, close).strip()
    diff = _git(repo, "diff", parent, close).strip()
    if not diff:
        return parent, close, "(empty diff: the closing commit has the same content as the parent)"
    if len(diff) > MAX_DIFF_CHARS:
        diff = diff[:MAX_DIFF_CHARS] + f"\n... [truncated at {MAX_DIFF_CHARS:,} of {len(diff):,} chars]"
    return parent, close, f"{stat}\n\n{diff}"


def discover_prompt(bench: str, runs: list[Run]) -> str:
    head = DISCOVER_PROMPT_PATH.read_text(encoding="utf-8").replace("{definition}", definition()).rstrip()
    task = (f"## Your task now\n\nBenchmark: `{bench}`. Your working directory is its workspace; start with "
            "`RUNS.md` and the seed program, then the change summaries above, then confirm every candidate in the "
            "code. List every macro approach you confirmed. Remember: nothing about where an approach came from. End your reply with EXACTLY ONE fenced "
            f"```json block (and nothing after it) of this shape:\n\n```json\n{DISCOVER_SHAPE}\n```\n")
    return head + "\n\n" + summaries_block(runs) + "\n" + task


def count_prompt(bench: str, run: Run, cat: dict) -> str:
    """The count prompt for ONE run: each run is judged in its own session against the fixed catalog."""
    total = len(run.iteration_keys)
    missing = sorted(set(range(1, total + 1)) - summarized_iterations(run))
    head = (COUNT_PROMPT_PATH.read_text(encoding="utf-8").replace("{definition}", definition(patterns=False))
            .replace("{catalog}", render_catalog_for_prompt(cat))
            .replace("{total_iterations}", str(total))
            .replace("{missing_iterations}", ", ".join(map(str, missing)) or "none").rstrip())
    task = (f"## Your task now\n\nBenchmark: `{bench}`. The run you judge is `{run.run_id}` (its section of "
            f"`RUNS.md`, `repos/{run.run_id}/`, `memory/{run.run_id}/`); ignore the other runs in the workspace. "
            "Start with its iteration table in `RUNS.md` and the seed program. Find every catalog macro approach "
            "(M<n>) the run landed, citing its landed iterations and code evidence, and list its landed "
            "non-incremental approaches that match no catalog entry under `uncatalogued`. End your reply with "
            "EXACTLY ONE fenced ```json block (and nothing after it) of this shape:\n\n"
            f"```json\n{COUNT_SHAPE}\n```\n")
    return head + "\n\n" + summaries_block([run]) + "\n" + task


def fill_prompt(bench: str, run: Run, raw: str, need: dict[int, tuple[str, str, str]]) -> str:
    head = FILL_PROMPT_PATH.read_text(encoding="utf-8").rstrip()
    parts = [head, f"\n## Existing change summaries of `{run.run_id}`\n", raw, "\n## Iterations without a summary\n"]
    for n, (parent, close, diff) in sorted(need.items()):
        landed = "yes" if run.iters[str(n)].get("changes_landed") else "no"
        parts.append(f"### Iteration {n} (parent `{parent}`, closing commit `{close}`, landed = {landed})\n\n"
                     f"```diff\n{diff}\n```\n")
    parts.append(f"## Your task now\n\nBenchmark: `{bench}`, run `{run.run_id}`. Write a change summary for "
                 f"each of iterations {', '.join(map(str, sorted(need)))} and no others. End your reply with EXACTLY "
                 f"ONE fenced ```json block (and nothing after it) of this shape:\n\n```json\n{FILL_SHAPE}\n```\n")
    return "\n".join(parts)


def validate_fill(d: dict, need: set[int]) -> list[str]:
    """Fatal: not exactly one non-empty summary for each requested iteration."""
    lst = d.get("summaries")
    if not isinstance(lst, list):
        raise ValueError("invalid fill: 'summaries' missing or not a list")
    got: dict[int, int] = {}
    problems = []
    for i, e in enumerate(lst):
        try:
            n = int(e["iteration"])
        except (KeyError, TypeError, ValueError):
            problems.append(f"summaries[{i}] has no integer 'iteration'")
            continue
        got[n] = got.get(n, 0) + 1
        if not str(e.get("summary") or "").strip():
            problems.append(f"iteration {n}: empty summary")
    missing, extra = sorted(need - set(got)), sorted(set(got) - need)
    dup = sorted(n for n, c in got.items() if c > 1)
    if missing:
        problems.append(f"no summary for iteration(s) {missing}")
    if extra:
        problems.append(f"iteration(s) {extra} were not asked for")
    if dup:
        problems.append(f"iteration(s) {dup} summarized more than once")
    if problems:
        raise ValueError("invalid fill: " + "; ".join(problems))
    return []


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def provenance_patterns(runs: list[Run]) -> list[tuple[re.Pattern, str]]:
    """Patterns that reveal where an approach came from. Any hit invalidates a discovery answer, so they
    are kept narrow enough not to fire on mechanism prose ("the solver runs a program", "seed program")."""
    I = re.I
    pats = [
        (r"\b(pareto[_ ]?evolution|pareto|freeform|free-form)\b", I, "arm name"),
        (r"\bjob_[0-9a-f]{6,}\b", I, "job id"),
        (r"\bcodeevolver-\d{8,}", I, "branch name"),
        (r"\b[0-9a-f]{12,40}\b", 0, "commit sha"),
        (r"\b[sS][1-9]\b", 0, "seed label"),
        (r"\b(iter|iters|iteration|iterations)\s*#?\s*\d", I, "iteration number"),
        (r"\b[Rr]un\s*#\s*\d+|\bRun\s+(\d+|[A-F])\b|\bR[1-6]\b", 0, "run label"),
        (r"\b(one|two|three|four|five|six|several|multiple|many|most|some|all|both|other|a few|\d+)\s+"
         r"(of the\s+)?runs\b", I, "run frequency"),
        # "run" as a noun only: "sub-goals that each run their own loop" (a mechanism, and the wording of
        # macro_definition.md itself) is not provenance.
        (r"\b(one|two|three|several|multiple|every|each|another|a single|only one)\s+(of the\s+)?run\b"
         r"(?!\s+(their|its|in|on|over|through|with|a|an|the|as|once|twice|for|per|until|after|before)\b)", I,
         "run frequency"),
    ]
    out = [(re.compile(p, f), why) for p, f, why in pats]
    for r in runs:
        out.append((re.compile(re.escape(r.run_id), re.I), "run id"))
        if r.job_id:
            out.append((re.compile(re.escape(str(r.job_id).removeprefix("job_")), re.I), "job id"))
    return out


# Wording that may hint at provenance but also appears in mechanism prose: a warning in the draft, not a retry.
SOFT_PROVENANCE = re.compile(r"\b(reverted|landed|later|earlier|first attempt|re-?implemented|abandoned|"
                             r"(this|that|the|a) run|seeds?\s*[1-3]\b)", re.I)


def _strings(obj) -> list[str]:
    if isinstance(obj, str):
        return [obj]
    if isinstance(obj, dict):
        return [s for v in obj.values() for s in _strings(v)]
    if isinstance(obj, list):
        return [s for v in obj for s in _strings(v)]
    return []


def _provenance_hit(text: str, pats: list[tuple[re.Pattern, str]]) -> str | None:
    for pat, why in pats:
        m = pat.search(text)
        if m:
            return f"provenance ({why}): '{m.group(0)}'"
    return None


def validate_discovery(d: dict, runs: list[Run]) -> list[str]:
    """Fatal (ValueError): a missing list, an entry without its fields, or any
    provenance (run/arm/seed/job/branch/commit/iteration, or how many runs used an approach) in an
    approach entry.
    A reviewer note with provenance is removed from ``d["notes"]`` (in place) with a warning instead:
    notes are optional, and one stray "across all runs" should not cost a fresh session.
    Warning: an empty macro list and soft provenance words (reverted, landed, later, ...)."""
    problems: list[str] = []
    warnings: list[str] = []
    lst = d.get("macro_approaches")
    if not isinstance(lst, list):
        problems.append("'macro_approaches' missing or not a list")
    else:
        for i, e in enumerate(lst):
            for f in ("name", "description", "why_macro"):
                if not isinstance(e, dict) or not str(e.get(f) or "").strip():
                    problems.append(f"macro_approaches[{i}]: empty '{f}'")
        if not lst:
            # A legitimate finding (e.g. every change is a guard on one loop); the reviewer decides.
            warnings.append("no macro approaches proposed: every candidate was judged incremental")
    pats = provenance_patterns(runs)
    for i, e in enumerate(d.get("macro_approaches") or []):
        hit = next((h for t in _strings(e) if (h := _provenance_hit(t, pats))), None)
        if hit:
            problems.append(f"macro_approaches[{i}]: {hit}")
    kept = []
    for n in d.get("notes") or []:
        hit = _provenance_hit(str(n), pats)
        if hit:
            warnings.append(f"a reviewer note was removed: {hit}")
        else:
            kept.append(n)
    d["notes"] = kept
    for t in _strings({k: d.get(k) for k in ("macro_approaches", "notes")}):
        m = SOFT_PROVENANCE.search(t)
        if m:
            warnings.append(f"possible provenance wording '{m.group(0)}' in: {t[:120]}")
    if problems:
        raise ValueError("invalid discovery: " + "; ".join(problems))
    return warnings


def _int_list(v) -> list[int] | None:
    if not isinstance(v, list):
        return None
    try:
        return [int(x) for x in v]
    except (TypeError, ValueError):
        return None


def validate_count(c: dict, run: Run, cat: dict) -> list[str]:
    """One run's count. Fatal (ValueError): 'catalog_hits' missing, a hit id not in the catalog, an id repeated, a hit without an integer iteration list, an
    uncatalogued entry without a name. Warnings: hits citing no landed iteration (dropped from the
    count; see ``apply_landed_check``)."""
    problems: list[str] = []
    macro_ids = {e["id"] for e in cat["macro"]}
    hits = c.get("catalog_hits")
    if not isinstance(hits, list):
        raise ValueError(f"invalid count for {run.run_id}: 'catalog_hits' missing or not a list")
    ids = [str(h.get("id")) if isinstance(h, dict) else "?" for h in hits]
    for h, hid in zip(hits, ids):
        if hid not in macro_ids:
            problems.append(f"{hid!r} is not a catalog macro id")
        if not isinstance(h, dict) or _int_list(h.get("iterations")) is None:
            problems.append(f"hit {hid} has no integer 'iterations' list")
    for d in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{d} counted more than once")
    unc = c.get("uncatalogued", [])
    if not isinstance(unc, list):
        problems.append("'uncatalogued' is not a list")
    else:
        for i, u in enumerate(unc):
            if not isinstance(u, dict) or not str(u.get("name") or "").strip():
                problems.append(f"uncatalogued[{i}] has no name")
    if problems:
        raise ValueError(f"invalid count for {run.run_id}: " + "; ".join(problems))
    landed = landed_iterations(run)
    return [f"{h['id']} cites no landed iteration ({_int_list(h['iterations'])}); not counted"
            for h in hits if not set(_int_list(h["iterations"]) or []) & landed]


def apply_landed_check(c: dict, run: Run) -> dict:
    """One run: counted hits (cite at least one landed iteration), dropped hits, and notes on
    non-landed iterations a counted hit also cited. Deterministic from history.json."""
    landed = landed_iterations(run)
    counted, dropped, extra = [], [], []
    for h in c["catalog_hits"]:
        its = _int_list(h["iterations"]) or []
        ok = sorted(set(its) & landed)
        if ok:
            counted.append({**h, "landed_iterations": ok})
            bad = sorted(set(its) - landed)
            if bad:
                extra.append(f"{h['id']} also cites non-landed iteration(s) {bad}")
        else:
            dropped.append(h)
    return {"counted": counted, "dropped": dropped, "notes": extra, "uncatalogued": c.get("uncatalogued") or []}


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def _session_env(args) -> None:
    prepare_engine_env(args.out / "opencode", model=args.model)
    load_env_key(REPO_ROOT, HERE)
    key_var = provider_key(args.model)
    if key_var and not os.environ.get(key_var):
        raise SystemExit(f"{key_var} not set (checked env, <repo>/.env, scripts/macro-counts/.env)")
    if not shutil.which("opencode"):
        raise SystemExit("opencode not found on PATH")


def _cached(path: Path, want: dict, force: bool) -> dict | None:
    """The stored session wrapper when every hash in ``want`` matches, else None."""
    if force or not path.exists():
        return None
    jd = json.loads(path.read_text(encoding="utf-8"))
    stale = [k for k, v in want.items() if jd.get(k) != v]
    if stale:
        log(f"[{path.stem}] cached result is stale ({', '.join(stale)} changed); re-running")
        return None
    return jd


def prepare(args, bench: str) -> tuple[list[Run], Path, dict]:
    runs, unpaired = bench_runs(args.archive_root, bench, args.arm1, args.arm2)
    for rid, why in unpaired:
        log(f"[{bench}] left out: {rid} ({why})")
    if not runs:
        raise SystemExit(f"[{bench}] no complete {args.arm1}/{args.arm2} seed pairs under {args.archive_root}")
    ws, stats = materialize_bench(bench, runs, args.out)
    log(f"[{bench}] workspace {ws}: " + "; ".join(
        f"{rid} {s['iterations']} iters ({s['landed']} landed), summaries {s['summary_chars']:,} chars"
        for rid, s in stats["runs"].items()))
    return runs, ws, stats


def session_replies(session_dir: Path) -> tuple[str | None, str | None, list[str]]:
    """(workspace dir, session id, assistant text parts in order) from a finished OpenCode
    inspection session's data dir (``~/.codeevolver/opencode-sessions/inspection-*``)."""
    import sqlite3
    db = Path(session_dir) / "opencode" / "opencode.db"
    if not db.exists():
        return None, None, []
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute("select id, directory from session order by time_created limit 1").fetchone()
        parts = [json.loads(r[0]) for r in con.execute("select data from part order by time_created, id")]
    finally:
        con.close()
    texts = [p.get("text", "") for p in parts if p.get("type") == "text" and p.get("text")]
    return (row[1], row[0], texts) if row else (None, None, texts)


def salvage_discovery(args, bench: str, ws: Path, runs: list[Run], want: dict, jpath: Path) -> dict | None:
    """Re-validate the stored replies of earlier discovery sessions for this benchmark (newest first)
    and keep the first one that passes. No model call. Used when a reply was rejected by a validator
    that has since been fixed."""
    from judge import parse_fenced_json
    for sdir in sorted(args.salvage, key=lambda d: d.name, reverse=True):
        directory, sid, texts = session_replies(sdir)
        if not directory or Path(directory).resolve() != ws.resolve():
            continue
        parsed = next((j for t in reversed(texts) if (j := parse_fenced_json(t, "macro_approaches"))), None)
        if parsed is None:
            log(f"[{bench}] salvage {sdir.name}: no fenced JSON reply")
            continue
        try:
            warnings = validate_discovery(parsed, runs)
        except ValueError as e:
            log(f"[{bench}] salvage {sdir.name}: still invalid ({e})")
            continue
        log(f"[{bench}] salvaged the reply of {sdir.name} (session {sid})")
        jd = {"bench": bench, "stage": "discover", **want, "runs": [r.run_id for r in runs], "harness": "opencode",
              "model": args.model, "judged_at": dt.datetime.fromtimestamp(sdir.stat().st_mtime, dt.timezone.utc)
              .isoformat(timespec="seconds"), "session_id": sid, "data_dir": str(sdir), "salvaged": True,
              "usage": {}, "tool_calls": [], "n_tool_calls": 0, "retries": 0, "attempt_errors": [],
              "warnings": warnings, "transcript": None, "result": parsed}
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
        return jd
    return None


def run_discover(args, bench: str) -> None:
    runs, ws, _ = prepare(args, bench)
    prompt = discover_prompt(bench, runs)
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)
    (args.out / "prompts" / f"{bench}.discover.md").write_text(prompt, encoding="utf-8")
    log(f"[{bench}] discover prompt {len(prompt):,} chars (~{len(prompt) // 4:,} tok) -> prompts/{bench}.discover.md")
    if args.dry_run:
        return
    want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "input_sha256": input_sha256(ws, runs)}
    jpath = args.out / "drafts" / f"{bench}.json"
    jd = _cached(jpath, want, args.force)
    if jd is None and args.salvage:
        jd = salvage_discovery(args, bench, ws, runs, want, jpath)
    if jd is None:
        jpath.parent.mkdir(parents=True, exist_ok=True)
        jd = run_json_session(ws, prompt, required_key="macro_approaches", label=f"{bench} discover",
                              validate=lambda d: validate_discovery(d, runs), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=True, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"bench": bench, "stage": "discover", **want, "runs": [r.run_id for r in runs], **jd}
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
    d = jd["result"]
    macro = [{"id": f"M{i}", **{k: e[k] for k in ("name", "description", "why_macro")}}
             for i, e in enumerate(d["macro_approaches"], 1)]
    header = (f"Draft from the discovery session ({jd['model']}, {jd['judged_at']}). Curate it into "
              f"`scripts/macro-counts/catalogs/{bench}.md`: merge, split, delete, promote or demote entries and "
              "renumber freely. Only the macro section is parsed; `Why macro` is for you and never reaches the count "
              "session.")
    md = render_catalog_md(bench, macro, notes=list(d.get("notes") or []) + [f"(validator) {w}" for w in jd["warnings"]],
                           header=header)
    (args.out / "drafts" / f"{bench}.md").write_text(md, encoding="utf-8")
    u = jd.get("usage") or {}
    log(f"[{bench}] draft: {len(macro)} macro approaches -> drafts/{bench}.md "
        f"(tool_calls={jd['n_tool_calls']} retries={jd['retries']} in={u.get('input_tokens')} out={u.get('output_tokens')})")


def run_fill(args, bench: str) -> list[str]:
    """Write inputs/<bench>/<run>/raw_change_summaries.md and processed_change_summaries.md for every
    paired run. An iteration without a heading in the raw file gets its ``iteration_<N>_change_summary.md``
    when the run has one, else a summary written from its diff by ONE session for the whole run (no
    session when nothing is left to write). Runs whose processed file exists are skipped unless --force.
    Returns the run ids that failed."""
    runs, ws, _ = prepare(args, bench)
    if args.seed:
        runs = [r for r in runs if r.seed in args.seed]
    todo = []
    for run in runs:
        out_path = processed_summaries_path(run)
        if out_path.exists() and not args.force:
            log(f"[{bench}] {run.run_id}: processed summaries exist; skipped")
            continue
        out_path.parent.mkdir(parents=True, exist_ok=True)
        raw = raw_change_summaries(run)
        (out_path.parent / "raw_change_summaries.md").write_text(raw + "\n", encoding="utf-8")
        missing = sorted(set(range(1, len(run.iteration_keys) + 1)) - summarized_iterations(run, raw))
        added: dict[int, str] = {}
        need: dict[int, tuple[str, str, str]] = {}
        md = memory_dir(run)
        for n in missing:
            f = md / f"iteration_{n}_change_summary.md" if md else None
            text = f.read_text(encoding="utf-8", errors="replace").strip() if f and f.exists() else ""
            if text:
                added[n] = (f"# Change Description Iteration {n} (from `iteration_{n}_change_summary.md`; missing "
                            f"from the run's combined file)\n{text}")
            else:
                need[n] = iteration_diff(ws, run, n)
        todo.append((run, out_path, raw, added, need))
    if args.dry_run:
        for run, _, _, added, need in todo:
            log(f"[{bench}] {run.run_id}: {len(added)} from per-iteration files, {len(need)} to write from diffs "
                f"{sorted(need) or ''}")
        return []

    def _one(run: Run, out_path: Path, raw: str, added: dict[int, str], need: dict) -> None:
        meta = {"bench": bench, "run_id": run.run_id, "from_iteration_files": sorted(added),
                "from_diffs": sorted(need), "model": None}
        if need:
            prompt = fill_prompt(bench, run, raw, need)
            (out_path.parent / "fill_prompt.md").write_text(prompt, encoding="utf-8")
            jd = run_json_session(ws, prompt, required_key="summaries", label=f"{bench} fill {run.run_id}",
                                  validate=lambda d: validate_fill(d, set(need)), model=args.model,
                                  max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                                  allow_bash=True, transcript_path=out_path.parent / "fill.transcript.json",
                                  rejected_dir=args.out / "rejected", log=log)
            for e in jd["result"]["summaries"]:
                n = int(e["iteration"])
                parent, close, _ = need[n]
                added[n] = (f"# Change Description Iteration {n} (written by the fill stage from the diff "
                            f"{parent[:8]}..{close[:8]}; the run recorded no summary)\n{e['summary'].strip()}")
            meta.update({"model": args.model, "session_id": jd.get("session_id"), "usage": jd.get("usage"),
                         "retries": jd.get("retries"), "judged_at": jd.get("judged_at")})
        preamble, blocks = split_summary_blocks(raw)
        blocks += list(added.items())
        body = "\n\n".join(b for _, b in sorted(blocks, key=lambda x: x[0]))
        out_path.write_text(((preamble + "\n\n") if preamble else "") + body + "\n", encoding="utf-8")
        (out_path.parent / "fill.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
        log(f"[{bench}] {run.run_id}: processed summaries written ({len(meta['from_iteration_files'])} from "
            f"per-iteration files, {len(meta['from_diffs'])} from diffs)")

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, *t): t[0].run_id for t in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(futs[fut])
                log(f"[{bench}] {futs[fut]}: fill FAILED: {e}")
    return failed


def count_dir(out: Path, bench: str) -> Path:
    return out / "count" / bench


def run_count(args, bench: str) -> list[str]:
    """One session per run (``--jobs`` in parallel). Returns the run ids that failed."""
    cat = load_catalog(args.catalog_dir, bench)
    catalog_sha = sha256_text(render_catalog_for_prompt(cat))
    runs, ws, _ = prepare(args, bench)
    if args.seed:  # the workspace keeps every paired run; only these seeds are counted
        runs = [r for r in runs if r.seed in args.seed]
    unfilled = [r.run_id for r in runs if not processed_summaries_path(r).exists()]
    if unfilled:
        raise SystemExit(f"[{bench}] no processed change summaries for {unfilled}: run `catalog.py fill --bench {bench}` first")
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)
    cdir = count_dir(args.out, bench)
    cdir.mkdir(parents=True, exist_ok=True)
    todo: list[tuple[Run, str, dict]] = []
    for run in runs:
        prompt = count_prompt(bench, run, cat)
        (args.out / "prompts" / f"{bench}.count.{run.run_id}.md").write_text(prompt, encoding="utf-8")
        want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "input_sha256": input_sha256(ws, [run]),
                "catalog_sha256": catalog_sha}
        if not args.dry_run and _cached(cdir / f"{run.run_id}.json", want, args.force) is not None:
            log(f"[{bench}] {run.run_id}: cached")
            continue
        todo.append((run, prompt, want))
    sizes = ", ".join(f"{r.run_id} ~{len(p) // 4:,} tok" for r, p, _ in todo)
    log(f"[{bench}] catalog {len(cat['macro'])} macro approaches; "
        f"{len(todo)} of {len(runs)} runs to count ({sizes or '-'}) -> prompts/{bench}.count.<run>.md")
    if args.dry_run:
        return []

    def _one(run: Run, prompt: str, want: dict) -> None:
        jpath = cdir / f"{run.run_id}.json"
        jd = run_json_session(ws, prompt, required_key="catalog_hits", label=f"{bench} count {run.run_id}",
                              validate=lambda c: validate_count(c, run, cat), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=True, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"bench": bench, "stage": "count", "run_id": run.run_id, **want, "catalog": cat, **jd}
        jpath.write_text(json.dumps(jd, indent=1), encoding="utf-8")
        ch = apply_landed_check(jd["result"], run)
        log(f"[{bench}] {run.run_id}: {len(ch['counted'])} counted ({' '.join(h['id'] for h in ch['counted']) or '-'}), "
            f"{len(ch['dropped'])} dropped, {len(ch['uncatalogued'])} uncatalogued "
            f"(tool_calls={jd['n_tool_calls']} retries={jd['retries']})")

    failed: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, r, p, w): r.run_id for r, p, w in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(futs[fut])
                log(f"[{bench}] {futs[fut]}: FAILED: {e}")
    return failed


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def mean_sd(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return 0.0, 0.0
    m = sum(xs) / len(xs)
    sd = (sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5 if len(xs) > 1 else 0.0
    return m, sd


def cell(v) -> str:
    return str(v).replace("\r", "").replace("\n", " ").replace("|", "\\|")


def fmt_score(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def find_run(archive_root: Path, run_id: str) -> Run | None:
    hits = runs_mod.discover_runs(archive_root, f"^{re.escape(run_id)}$", None, None)
    return hits[0] if hits else None


CSV_FIELDS = ["bench", "run_id", "arm", "seed", "count", "counted_ids", "dropped_ids", "n_uncatalogued",
              "catalog_size", "iterations", "landed_iterations", "seed_test", "final_test", "judge_model",
              "catalog_sha256", "judged_at", "session_id", "n_tool_calls", "retries", "total_tokens"]


def write_review(bench: str, cat: dict, jds: dict[str, dict], runs: dict[str, Run],
                 checked: dict[str, dict], out: Path) -> None:
    names = {e["id"]: e["name"] for e in cat["macro"]}
    lines = [f"# {bench}: catalog macro approaches per run\n",
             "Each run was judged in its own session against the fixed catalog. A hit counts only if it cites an "
             "iteration that landed (history.json `changes_landed`).\n"]
    lines.append("## Catalog\n")
    lines += [f"- **{e['id']}: {e['name']}.** {e['description']}" for e in cat["macro"]]
    lines.append("")
    for rid, run in runs.items():
        ch, jd = checked[rid], jds[rid]
        lines.append(f"## {rid}: {len(ch['counted'])} counted\n")
        lines.append(f"Judge {jd['model']}, {jd['judged_at']}, session `{jd['session_id']}`, {jd['n_tool_calls']} tool "
                     f"calls, {jd['retries']} retries"
                     + (f", transcript [{jd['transcript']}](../count/{bench}/{jd['transcript']})" if jd.get("transcript") else "")
                     + (f"; catalog sha {jd['catalog_sha256'][:12]}" if jd["catalog_sha256"] != sha256_text(render_catalog_for_prompt(cat)) else "")
                     + f". Held-out score seed → final (reviewer only): {fmt_score(run.score('seed_test_score'))} → "
                     f"{fmt_score(run.score('final_test_score'))}. Landed iterations: "
                     f"{sorted(landed_iterations(run)) or 'none'}.\n")
        if ch["counted"]:
            lines.append("| id | approach | landed iterations cited | evidence |\n|---|---|---|---|")
            lines += [f"| {h['id']} | {cell(names.get(h['id'], '?'))} | {', '.join(map(str, h['landed_iterations']))} | "
                      f"{cell(h.get('evidence', ''))} |" for h in ch["counted"]]
            lines.append("")
        if ch["dropped"]:
            lines.append("**Dropped (no landed iteration cited):** " + "; ".join(
                f"{h['id']} {cell(names.get(h['id'], '?'))} (cited {h.get('iterations')})" for h in ch["dropped"]) + "\n")
        if ch["notes"]:
            lines += [f"- note: {n}" for n in ch["notes"]] + [""]
        if ch["uncatalogued"]:
            lines.append("**Uncatalogued (not counted; candidates for the catalog):**\n")
            lines += [f"- {cell(u.get('name'))} (iterations {u.get('iterations')}): {cell(u.get('description', ''))} "
                      f"*Why macro:* {cell(u.get('why_macro', ''))}" for u in ch["uncatalogued"]]
            lines.append("")
        flags = list(jd["result"].get("review_flags") or []) + [f"(validator) {w}" for w in jd.get("warnings", [])]
        if flags:
            lines.append("**Review flags:**\n")
            lines += [f"- {cell(f)}" for f in flags] + [""]
    (out / "review").mkdir(parents=True, exist_ok=True)
    (out / "review" / f"{bench}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_counts(out: Path, bench: str) -> dict[str, dict]:
    """run_id -> stored per-run count result, for every run counted under count/<bench>/."""
    d = count_dir(out, bench)
    files = sorted(f for f in d.glob("*.json") if not f.name.endswith(".transcript.json")) if d.is_dir() else []
    return {json.loads(f.read_text(encoding="utf-8"))["run_id"]: json.loads(f.read_text(encoding="utf-8")) for f in files}


def run_report(args) -> None:
    root = args.out / "count"
    found = sorted(d.name for d in root.iterdir() if d.is_dir() and load_counts(args.out, d.name)) if root.is_dir() else []
    if args.bench:
        found = [b for b in found if b in args.bench]
    if not found:
        raise SystemExit(f"no count results in {root}")
    rows: list[dict] = []
    md = ["# Macro approaches per run, counted against a curated catalog (v5)\n",
          f"Generated {dt.datetime.now().isoformat(timespec='minutes')}. Definition: "
          "`scripts/macro-counts/macro_definition.md`; catalogs: `scripts/macro-counts/catalogs/<bench>.md`. "
          "Each run is judged in its own session. A run's **count** is the number of catalog macro approaches it "
          "landed (each at most once, citing an iteration whose change landed). Approaches outside the catalog are "
          "listed as uncatalogued and not counted. Mean ± sd is over the arm's seeds.\n"]
    benches = [b for b in BENCHES if b in found] + sorted(set(found) - set(BENCHES))
    for bench in benches:
        jds = load_counts(args.out, bench)
        runs = {rid: find_run(args.archive_root, rid) for rid in jds}
        missing = [rid for rid, r in runs.items() if r is None]
        if missing:
            raise SystemExit(f"[{bench}] run dirs not found under {args.archive_root}: {missing}")
        # The catalog the newest result used is the reference; results under another catalog are flagged.
        latest = max(jds.values(), key=lambda j: j["judged_at"])
        cat, cat_sha = latest["catalog"], latest["catalog_sha256"]
        stale = sorted(rid for rid, j in jds.items() if j["catalog_sha256"] != cat_sha)
        checked = {rid: apply_landed_check(jds[rid]["result"], runs[rid]) for rid in jds}
        write_review(bench, cat, jds, runs, checked, args.out)
        for rid, r in runs.items():
            ch, jd = checked[rid], jds[rid]
            rows.append({"bench": bench, "run_id": rid, "arm": r.arm, "seed": r.seed, "count": len(ch["counted"]),
                         "counted_ids": " ".join(h["id"] for h in ch["counted"]),
                         "dropped_ids": " ".join(h["id"] for h in ch["dropped"]),
                         "n_uncatalogued": len(ch["uncatalogued"]), "catalog_size": len(jd["catalog"]["macro"]),
                         "iterations": len(r.iteration_keys), "landed_iterations": len(landed_iterations(r)),
                         "seed_test": r.score("seed_test_score"), "final_test": r.score("final_test_score"),
                         "judge_model": jd["model"], "catalog_sha256": jd["catalog_sha256"][:12],
                         "judged_at": jd["judged_at"], "session_id": jd["session_id"],
                         "n_tool_calls": jd["n_tool_calls"], "retries": jd["retries"],
                         "total_tokens": (jd.get("usage") or {}).get("total")})
        brows = [x for x in rows if x["bench"] == bench]
        arms = list(dict.fromkeys(x["arm"] for x in sorted(brows, key=lambda x: x["arm"] != args.arm1)))
        md.append(f"## {bench}\n")
        md.append(f"Catalog: {len(cat['macro'])} macro approaches "
                  f"(sha {cat_sha[:12]}). Judge {latest['model']}. [review/{bench}.md](review/{bench}.md)\n")
        if stale:
            md.append(f"**Warning:** {', '.join(stale)} were counted under a different catalog; re-run `count` for them.\n")
        md.append("| arm | runs | count mean ± sd | per seed | uncatalogued (mean) |\n|---|---|---|---|---|")
        for arm in arms:
            ar = sorted((x for x in brows if x["arm"] == arm), key=lambda x: x["seed"])
            m, sd = mean_sd([x["count"] for x in ar])
            md.append(f"| {arm} | {len(ar)} | **{m:.2f} ± {sd:.2f}** | "
                      + ", ".join(f"s{x['seed']}={x['count']}" for x in ar)
                      + f" | {mean_sd([x['n_uncatalogued'] for x in ar])[0]:.2f} |")
        md.append("")
        order = sorted(runs, key=lambda rid: (arms.index(runs[rid].arm), runs[rid].seed))
        md.append("| id | approach | " + " | ".join(f"{runs[r].arm} s{runs[r].seed}" for r in order) + " |\n"
                  "|---|---|" + "---|" * len(order))
        for e in cat["macro"]:
            cells = []
            for rid in order:
                hit = next((h for h in checked[rid]["counted"] if h["id"] == e["id"]), None)
                cells.append("✓ " + ",".join(map(str, hit["landed_iterations"])) if hit else "")
            md.append(f"| {e['id']} | {cell(e['name'])} | " + " | ".join(cells) + " |")
        md.append("| | **count** | " + " | ".join(f"**{len(checked[r]['counted'])}**" for r in order) + " |\n")
        dropped = sum(len(v["dropped"]) for v in checked.values())
        if dropped:
            md.append(f"{dropped} hit(s) dropped for citing no landed iteration; see the review file.\n")
    csv_path = args.out / "counts.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    (args.out / "COUNTS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log(f"wrote {args.out / 'COUNTS.md'}, {csv_path} ({len(rows)} runs, {len(benches)} benches)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["fill", "discover", "check", "count", "report"])
    ap.add_argument("--bench", action="append", help="benchmark(s); required for fill / discover / check / count")
    ap.add_argument("--archive-root", type=Path, default=DEFAULT_ARCHIVE_ROOT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--catalog-dir", type=Path, default=DEFAULT_CATALOG_DIR)
    ap.add_argument("--seed", type=int, action="append", help="fill / count: only these seeds (repeatable; default all)")
    ap.add_argument("--arm1", default=DEFAULT_ARM1)
    ap.add_argument("--arm2", default=DEFAULT_ARM2)
    ap.add_argument("--void", default=None, metavar="REGEX",
                    help="run ids left out (default: pareto_evolution s1-s3, superseded by the -redo runs; '' = none)")
    ap.add_argument("--program-file", action="append", default=[], metavar="BENCH=PATH")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenCode provider/model id")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per session (hard kill)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES, help="fresh sessions to try")
    ap.add_argument("--jobs", type=int, default=2, help="parallel sessions: benchmarks (discover) or runs (count)")
    ap.add_argument("--force", action="store_true", help="re-run even when a matching result is cached")
    ap.add_argument("--dry-run", action="store_true", help="build the workspace and prompt only; no session")
    ap.add_argument("--no-report", action="store_true", help="count: skip the report afterwards")
    ap.add_argument("--salvage", type=Path, nargs="+", default=None, metavar="SESSION_DIR",
                    help="discover: before starting a session, re-validate these finished sessions' stored replies "
                         "(~/.codeevolver/opencode-sessions/inspection-*) and keep the newest one that passes")
    args = ap.parse_args()

    if args.void is not None:
        runs_mod.VOID_RUN_RE = re.compile(args.void) if args.void else None
    for spec in args.program_file:
        bench, _, path = spec.partition("=")
        if not path:
            ap.error(f"--program-file expects BENCH=PATH, got {spec!r}")
        PROGRAM_FILE[bench] = path
    if args.stage == "report":
        run_report(args)
        return 0
    if not args.bench:
        ap.error(f"{args.stage} needs --bench")
    if args.stage == "check":
        for bench in args.bench:
            cat = load_catalog(args.catalog_dir, bench)
            print(f"{bench}: OK, {len(cat['macro'])} macro approaches")
        return 0
    if not args.dry_run:
        _session_env(args)
    failures = []
    if args.stage == "discover":  # one session per benchmark; --jobs benchmarks in parallel
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
            futs = {ex.submit(run_discover, args, b): b for b in args.bench}
            for fut in as_completed(futs):
                try:
                    fut.result()
                except (JudgeError, RuntimeError, ValueError, SystemExit) as e:
                    failures.append(futs[fut])
                    log(f"[{futs[fut]}] FAILED: {e}")
        return 1 if failures else 0
    if args.stage == "fill":  # at most one session per run; --jobs runs in parallel within a benchmark
        for bench in args.bench:
            try:
                failures += [f"{bench}/{r}" for r in run_fill(args, bench)]
            except (JudgeError, RuntimeError, ValueError, SystemExit) as e:
                failures.append(bench)
                log(f"[{bench}] FAILED: {e}")
        if failures:
            log(f"{len(failures)} failed: {failures}")
        return 1 if failures else 0
    for bench in args.bench:  # count: one session per run; --jobs runs in parallel within a benchmark
        try:
            failures += [f"{bench}/{r}" for r in run_count(args, bench)]
        except (JudgeError, RuntimeError, ValueError, SystemExit) as e:
            failures.append(bench)
            log(f"[{bench}] FAILED: {e}")
    if failures:
        log(f"{len(failures)} failed: {failures}")
    if not args.dry_run and not args.no_report and (args.out / "count").is_dir():
        args.bench = None  # the report covers every counted benchmark, not just this invocation's
        run_report(args)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
