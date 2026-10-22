#!/usr/bin/env python3
"""fm_count.py -- shared helpers for fm_v3.py.

This module started as the fm-v2 failure-mode counter. fm-v2 is retired: its
prompt and definition files are no longer in the repository, so its CLI stages
(fill, build, count, report) do not run. fm_v3.py imports the helpers kept here
(text wrapping and numbering, git diffs, harness routing, permutation tests,
session environment and caching, run-level shares). Use fm_v3.py.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import hashlib
import itertools
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "macro-counts"))
import fm_runs as F  # noqa: E402
from anthropic_api import cost as usd  # noqa: E402  (USD from a usage dict; the price table F1 used)
from judge import JudgeError, load_env_key, prepare_engine_env, provider_key, run_json_session, workspace_sha256  # noqa: E402

REPO_ROOT = HERE.parent.parent
DEFAULT_OUT = Path.home() / "ce-analysis" / "fm-v2"
DEFAULT_RESULTS = Path.home() / "ce-analysis" / "failure-modes" / "results" / "F2"
DEFAULT_MODEL = "anthropic/claude-sonnet-5-5"   # the judge of F1 and of macro counts v5
DEFAULT_MAX_TURNS = 200
DEFINITIONS_PATH = HERE / "fm_definitions.md"
COUNT_PROMPT_PATH = HERE / "fm_count_prompt.md"
FILL_PROMPT_PATH = HERE / "fm_fill_prompt.md"

MODES = ("MS", "WR", "KR", "BO", "BC", "FN", "UR", "RP", "UL")
BINARY = {"KR", "BO", "BC", "UL"}
SOURCES = ("assistant", "tool_call", "memory")
MAX_QUOTE = 300
MAX_PROMPT_CHARS = 30_000  # the engine passes the prompt on the command line (Windows limit 32,767 characters)
SECTION_CAP = 20_000      # characters of one file's section of a diff shown in a workspace (head + tail)
RESULT_CAP = 12_000       # characters of one tool result shown in transcript.md (head + tail)
WRAP = 1_900              # OpenCode's read tool cuts lines at 2,000 characters
SUMMARY_CAP = 1_200       # characters of one change summary in earlier_iterations.md
SCORE_RE = re.compile(r"(?<![\w.])0?\.\d{2,}(?!\d)|\b\d{1,3}(?:\.\d+)?\s?%")   # score-like numbers, masked in summaries

FILL_SHAPE = """{
  "summaries": [
    {"iteration": <N>, "summary": "<three to eight sentences: what the change does to how the program works>"}
  ]
}"""

COUNT_SHAPE = """{
  "modes": {
    "MS": [{"turn": <n or null>, "source": "assistant|tool_call|memory", "quote": "<verbatim, <= 300 chars>",
            "cites": ["T<n>", "EVAL#<n>", "..."], "explanation": "<= 400 chars: the clause met and the numbers"}],
    "WR": [], "KR": [], "BO": [], "BC": [], "FN": [], "UR": [], "RP": [], "UL": []
  },
  "review_flags": ["<anything a human should double-check>"]
}"""

_lock = threading.Lock()


def log(msg: str) -> None:
    with _lock:
        print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


# ---------------------------------------------------------------------------------------------------------------
# runs and units
# ---------------------------------------------------------------------------------------------------------------
class Runs:
    """RunData of the 24 core runs, opened lazily (one bare git repository each, fm-private/git)."""

    def __init__(self, runs_root: Path):
        self.root = runs_root
        self.specs = F.core_runs()
        self._rd: dict[str, F.RunData] = {}
        self._mx = threading.Lock()

    def rd(self, run: str) -> F.RunData:
        with self._mx:
            if run not in self._rd:
                self._rd[run] = F.RunData(self.root, F.spec_of(run))
            return self._rd[run]

    def units(self) -> list[tuple[F.RunSpec, int, str]]:
        out = [(s, it, F.unit_id(s.run, it)) for s in self.specs for it in self.rd(s.run).iterations if it >= 1]
        assert len(out) == 351, len(out)
        assert len({u for *_, u in out}) == 351
        return out


def select_units(runs: Runs, only: str | None, limit: int | None) -> list[tuple[F.RunSpec, int, str]]:
    units = runs.units()
    if only:
        want = {x.strip() for x in only.split(",") if x.strip()}
        units = [u for u in units if u[2] in want or f"{u[0].run}:{u[1]}" in want]
        missing = want - {u[2] for u in units} - {f"{u[0].run}:{u[1]}" for u in units}
        if missing:
            raise SystemExit(f"unknown units: {sorted(missing)}")
    return units[:limit] if limit else units


# ---------------------------------------------------------------------------------------------------------------
# git
# ---------------------------------------------------------------------------------------------------------------
def git_diff(rd: F.RunData, a: str, b: str) -> str:
    """`git diff a b` in the run's bare repository, without the files the code identity ignores (fm_runs.EXC)."""
    rd.ensure_repo()
    r = subprocess.run(["git", "--git-dir", str(rd.bare), "diff", "--no-color", a, b], capture_output=True)
    if r.returncode != 0:
        return f"(git diff failed: {r.stderr.decode('utf-8', 'replace')[:200]})"
    text = r.stdout.decode("utf-8", "replace")
    parts = re.split(r"(?m)^(?=diff --git )", text)
    keep = [p for p in parts if p.strip() and not (
        (m := re.match(r"diff --git a/(\S+) b/", p)) and F.EXC.search(m.group(1)))]
    return "".join(keep).strip()


def cap_sections(diff: str) -> str:
    """Cap each file's section of a diff at SECTION_CAP characters (head + tail, the cut marked), so a rewritten data file
    (e.g. a 4.8 MB train JSON) does not swamp the workspace."""
    out = []
    for part in re.split(r"(?m)^(?=diff --git )", diff):
        if len(part) > SECTION_CAP:
            head, tail = part[:SECTION_CAP * 3 // 4], part[-SECTION_CAP // 4:]
            part = (f"{head}\n[... {len(part) - len(head) - len(tail):,} characters of this file's diff left out ...]\n"
                    f"{tail}")
        out.append(part)
    return "".join(out)


def attempted_diff(rd: F.RunData, it: int) -> str:
    p = rd.hdir / f"iteration_{it}_attempted.diff"
    return p.read_text(encoding="utf-8", errors="replace").strip() if p.exists() else ""


def changed(rd: F.RunData, it: int) -> bool | None:
    rec = rd.rec(it)
    a, b = rd.repo.sig(rec.get("parent_sha") or ""), rd.repo.sig(rec.get("close_sha") or "")
    return None if a is None or b is None else a != b


# ---------------------------------------------------------------------------------------------------------------
# fill
# ---------------------------------------------------------------------------------------------------------------
def summaries_path(out: Path, run: str) -> Path:
    return out / "summaries" / f"{run}.json"


def own_summary(rd: F.RunData, it: int) -> tuple[str, str] | None:
    t = str(rd.rec(it).get("change_summary") or "").strip()
    if t:
        return t, "history"
    for h in sorted(glob.glob(str(rd.root / ".codeevolver" / "memory" / "*" / f"iteration_{it}_change_summary.md"))):
        t = Path(h).read_text(encoding="utf-8", errors="replace").strip()
        if t:
            return t, "file"
    return None


def validate_fill(d: dict, need: set[int]) -> list[str]:
    """Fatal: not exactly one non-empty summary for each requested iteration (catalog.validate_fill)."""
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


def run_fill(args, runs: Runs) -> None:
    todo = []
    for spec in runs.specs:
        path = summaries_path(args.out, spec.run)
        if path.exists() and not args.force:
            log(f"[fill] {spec.run}: summaries exist; skipped")
            continue
        rd = runs.rd(spec.run)
        have, need = {}, []
        for it in rd.iterations:
            if it < 1:
                continue
            s = own_summary(rd, it)
            if s:
                have[it] = {"text": s[0], "source": s[1]}
            else:
                need.append(it)
        todo.append((spec, rd, have, need))
        log(f"[fill] {spec.run}: {len(have)} own summaries, {len(need)} to write {need or ''}")
    if args.dry_run:
        return

    def _one(spec: F.RunSpec, rd: F.RunData, have: dict, need: list[int]) -> None:
        meta = {"run": spec.run, "filled": need, "model": None}
        if need:
            ws = args.out / "fill" / spec.run
            if ws.exists():
                shutil.rmtree(ws)
            (ws / "diffs").mkdir(parents=True)
            # The material goes in workspace files, not in the prompt: the engine passes the prompt on the command line,
            # which Windows caps at 32,767 characters.
            blocks = []
            for it in need:
                rec = rd.rec(it)
                d = git_diff(rd, rec["parent_sha"], rec["close_sha"])
                att = attempted_diff(rd, it) if not d else ""
                (ws / "diffs" / f"iteration_{it}.patch").write_text(
                    (d or "(empty: the closing code has the same content as the start code)") + "\n", encoding="utf-8")
                b = f"- iteration {it}: `diffs/iteration_{it}.patch` ({len(d.splitlines())} lines" + (")" if d else ", empty)")
                if att:
                    (ws / "diffs" / f"iteration_{it}_attempted.diff").write_text(att + "\n", encoding="utf-8")
                    b += f" and `diffs/iteration_{it}_attempted.diff` ({len(att.splitlines())} lines; the change before the revert)"
                blocks.append(b)
            existing = "\n\n".join(f"# Change Description Iteration {k}\n{v['text']}" for k, v in sorted(have.items()))
            (ws / "existing_summaries.md").write_text((existing or "(none)") + "\n", encoding="utf-8")
            prompt = "\n".join([
                FILL_PROMPT_PATH.read_text(encoding="utf-8").rstrip(),
                "\n## Iterations without a summary, and their files\n", *blocks, "",
                f"## Your task now\n\nWrite a change summary for each of iterations {', '.join(map(str, need))} and no "
                "others. End your reply with EXACTLY ONE fenced ```json block (and nothing after it) of this shape:\n\n"
                f"```json\n{FILL_SHAPE}\n```\n"])
            (args.out / "fill" / f"{spec.run}.prompt.md").write_text(prompt, encoding="utf-8")
            assert len(prompt) < MAX_PROMPT_CHARS, (spec.run, len(prompt))
            jd = run_json_session(ws, prompt, required_key="summaries", label=f"fill {spec.run}",
                                  validate=lambda d: validate_fill(d, set(need)), model=args.model,
                                  max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                                  allow_bash=False, transcript_path=args.out / "fill" / f"{spec.run}.transcript.json",
                                  rejected_dir=args.out / "rejected", log=log)
            for e in jd["result"]["summaries"]:
                have[int(e["iteration"])] = {"text": e["summary"].strip(), "source": "filled"}
            meta.update({k: jd.get(k) for k in ("model", "session_id", "usage", "retries", "judged_at", "elapsed_seconds")})
            meta["prompt_sha256"] = sha256_text(prompt)
            meta["workspace_sha256"] = workspace_sha256(ws)
        path = summaries_path(args.out, spec.run)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"meta": meta, "summaries": {str(k): v for k, v in sorted(have.items())}},
                                   indent=1, ensure_ascii=False), encoding="utf-8")
        log(f"[fill] {spec.run}: written ({len(need)} filled)")

    failed = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, *t): t[0].run for t in todo}
        for fut in as_completed(futs):
            try:
                fut.result()
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(futs[fut])
                log(f"[fill] {futs[fut]}: FAILED: {e}")
    if failed:
        raise SystemExit(f"fill failed for {failed}")


def load_summaries(out: Path, run: str) -> dict[int, dict]:
    p = summaries_path(out, run)
    if not p.exists():
        raise SystemExit(f"no change summaries for {run}: run `fm_count.py fill` first")
    return {int(k): v for k, v in json.loads(p.read_text(encoding="utf-8"))["summaries"].items()}


# ---------------------------------------------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------------------------------------------
def wrap(text: str) -> str:
    """Hard-wrap lines longer than WRAP at a space (OpenCode's read tool cuts long lines). Whitespace only, so a quote
    check that ignores whitespace is unaffected."""
    out = []
    for line in str(text).split("\n"):
        while len(line) > WRAP:
            cut = line.rfind(" ", 0, WRAP)
            cut = cut if cut > WRAP // 2 else WRAP
            out.append(line[:cut])
            line = line[cut:].lstrip(" ") if line[cut:cut + 1] == " " else line[cut:]
        out.append(line)
    return "\n".join(out)


def cap_result(text: str) -> str:
    if len(text) <= RESULT_CAP:
        return text
    head, tail = text[:RESULT_CAP * 2 // 3], text[-RESULT_CAP // 3:]
    return f"{head}\n[... {len(text) - len(head) - len(tail):,} characters of this tool result left out ...]\n{tail}"


def call_fields(inp) -> list[tuple[str, str]]:
    """(field, text) per input field of a tool call: strings raw, anything else as JSON."""
    if not isinstance(inp, dict):
        return [("input", json.dumps(inp, ensure_ascii=False))]
    return [(k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in inp.items()]


def render_transcript(steps: list[dict]) -> tuple[str, dict[int, list[str]]]:
    """transcript.md (thinking left out) and the quotable texts per turn: the agent's text blocks and its tool-call input
    fields (each field, and the whole input as JSON)."""
    L = ["# Transcript", "",
         "Turns are numbered T<n>. `### T<n> assistant` holds the agent's text and its tool calls (`CALL <tool>`, then one "
         "`<field>:` block per input field). `### T<n> tool results` holds what the tools returned (not the agent's words; "
         "long results are shortened in the middle). `### T<n> harness message` is text from the harness. The agent's "
         "private reasoning is not shown.", ""]
    quotable: dict[int, list[str]] = {}
    for s in steps:
        k, t = s["kind"], s["turn"]
        if k == "prompt":
            L += [f"### T{t} task prompt (harness)", "", s["text"], ""]
        elif k == "notice":
            L += [f"### T{t} harness message", "", s["text"], ""]
        elif k == "assistant":
            q = quotable.setdefault(t, [])
            L += [f"### T{t} assistant", ""]
            for tx in s["texts"]:
                L += [tx, ""]
                q.append(tx)
            for c in s["calls"]:
                L.append(f"CALL {c['name']}")
                for field, val in call_fields(c["input"]):
                    if "\n" in val or len(val) > 200:
                        L += [f"{field}:", val]
                    else:
                        L.append(f"{field}: {val}")
                    q.append(val)
                q.append(json.dumps(c["input"], ensure_ascii=False))
                L.append("")
            for p in s["patches"]:
                L += [f"(harness note) {p}", ""]
        elif k == "results":
            L += [f"### T{t} tool results", ""]
            for r in s["results"]:
                ct = f"T{r['call_turn']}" if r.get("call_turn") is not None else "?"
                L += [f"RESULT of {r['name']} (called at {ct})" + (" [error]" if r["is_error"] else ""),
                      cap_result(r["content"]), ""]
            for e in s["extra"]:
                L += ["(harness text)", e, ""]
    return wrap("\n".join(L)) + "\n", quotable


def fmt(x) -> str:
    return "(no score)" if x is None else f"{x:.4f}"


def numbered(text: str) -> str:
    return "\n".join(f"L{i}: {ln}" for i, ln in enumerate(text.split("\n"), 1)) if text else "(empty)"


def build_unit(out: Path, runs: Runs, spec: F.RunSpec, it: int, uid: str) -> dict:
    """Write <out>/ws/<uid>/ and <out>/meta/<uid>.json. Returns the meta record."""
    rd = runs.rd(spec.run)
    R = rd.repo
    rec = rd.rec(it)
    bench = spec.bench
    sd = {w: F.tau_of(bench, w) / 2 ** 0.5 for w in ("train", "comparison")}
    summaries = load_summaries(out, spec.run)
    start_sig, close_sig = R.sig(rec["parent_sha"]), R.sig(rec["close_sha"])
    is_changed = None if start_sig is None or close_sig is None else start_sig != close_sig
    gate = bool(rec.get("constraint_violation"))

    # code labels: C0 = start code, then in order of first eval
    labels: dict[str, str] = {}
    if start_sig:
        labels[start_sig] = "C0"

    def lab(sig):
        if sig is None:
            return "?"
        if sig not in labels:
            labels[sig] = f"C{len(labels)}"
        return labels[sig]

    steps = rd.steps(it)
    shown: dict[str, int] = {}
    for s in steps:
        if s["kind"] == "results":
            for r in s["results"]:
                for e in rd.evals_of(it):
                    if e["eval_id"] not in shown and e["eval_id"] in r["content"]:
                        shown[e["eval_id"]] = s["turn"]
    samples: dict[str, str] = {}
    ev_rows = []
    for n, e in enumerate(rd.evals_of(it), 1):
        sk = e.get("sample_key")
        if e.get("eval_sample") != "comparison" and sk:
            samples.setdefault(sk, f"S{len(samples) + 1}")
        sig = R.sig(e["tree"]) if e.get("tree") else None
        ev_rows.append(dict(n=n, set=e.get("eval_sample"), sample=samples.get(sk, "-") if e.get("eval_sample") != "comparison" else "-",
                            code=lab(sig), sig=sig, rows=e.get("n_rows"), score=e.get("score"),
                            by="harness" if e.get("by_loop") else "agent",
                            shown=f"T{shown[e['eval_id']]}" if e["eval_id"] in shown else "-", eval_id=e["eval_id"]))
    close_label = lab(close_sig)

    L = [f"# Evals of this iteration", "",
         "Every eval logged in this iteration, in time order. `code`: C0 = the start code; the same label = identical code "
         f"(file contents). The closing code is {close_label}. `sample`: train samples with the same alias have the same "
         "rows. `run by`: agent = requested by the agent; harness = run by the harness itself. `shown at`: the turn whose "
         "tool result contains the eval (- = not found in the transcript).", "",
         "| EVAL# | eval set | sample | code | rows | score | run by | shown at |", "|---|---|---|---|---|---|---|---|"]
    L += [f"| {r['n']} | {r['set']} | {r['sample']} | {r['code']}{' (closing)' if r['sig'] and r['sig'] == close_sig else ''} | "
          f"{r['rows']} | {fmt(r['score'])} | {r['by']} | {r['shown']} |" for r in ev_rows]
    if not ev_rows:
        L.append("| - | (no evals in this iteration) | | | | | | |")
    evals_md = "\n".join(L) + "\n"

    # comparison history up to this iteration
    close_of: dict[str, int] = {}
    for k in rd.iterations:
        if 0 <= k < it:
            s = R.sig(rd.rec(k).get("close_sha") or "")
            if s and s not in close_of:
                close_of[s] = k
    cmp_rows, ref = [], []
    for c in rd.comps:
        if int(c.get("iteration") or 0) > it:
            continue
        s = R.sig(c["tree"]) if c.get("tree") else None
        if int(c.get("iteration") or 0) == it and s in labels:
            what = f"{labels[s]} of this iteration"
        elif s and s == start_sig:
            what = "the start code (C0)"
        elif s in close_of:
            what = "the seed program (iteration 0)" if close_of[s] == 0 else f"closing code of iteration {close_of[s]}"
        elif s and int(c.get("iteration") or 0) < it:
            what = "a code version of that iteration"
        else:
            what = "other code"
        cmp_rows.append(dict(n=len(cmp_rows) + 1, it=c.get("iteration"), what=what, rows=c.get("n_rows"), score=c.get("score")))
        if s and s == start_sig and int(c.get("iteration") or 0) < it and c.get("score") is not None:
            ref.append((c["score"], f"CMP#{len(cmp_rows)}"))
    L = ["# Comparison-set history of the run", "",
         "Every comparison-set (held-out) eval of the run up to the end of this iteration, in time order, with the code it "
         "scored. Comparison scores of different iterations are comparable.", "",
         "| CMP# | iteration | code | rows | score |", "|---|---|---|---|---|"]
    L += [f"| {r['n']} | {r['it']} | {r['what']} | {r['rows']} | {fmt(r['score'])} |" for r in cmp_rows]
    if not cmp_rows:
        L.append("| - | (none) | | | |")
    cmp_md = "\n".join(L) + "\n"
    parent_it = rec.get("parent_iteration")
    if not ref and parent_it is not None and str(parent_it) in rd.history["iteration_history"]:
        pcs = rd.rec(int(parent_it)).get("comparison_set_score")
        if pcs is not None and rd.repo.sig(rd.rec(int(parent_it)).get("close_sha") or "") == start_sig:
            ref.append((pcs, f"the comparison score the harness recorded for iteration {parent_it}"))

    # earlier iterations
    L = ["# Earlier iterations", "",
         "One row per earlier iteration (cite PRIOR:<n>), as the harness logged it. Train scores are on that iteration's own "
         "50-row sample (never comparable across iterations); comparison scores are on the fixed held-out set. The change "
         "summary was written after the iteration by a separate model: CONTEXT ONLY (what the code change was), not "
         "evidence, never the agent's words; score-like numbers are masked as [n].", ""]
    if it == 1:
        L.append("(this is iteration 1: there is no earlier iteration)")
    else:
        L += ["| iter | started from iter | code changed vs its start | harness outcome | train: start-code eval -> child "
              "evals (best, last) | comparison evals (agent / harness) | change summary (context only) |",
              "|---|---|---|---|---|---|---|"]
        for k in rd.iterations:
            if not 1 <= k < it:
                continue
            rk, evk = rd.rec(k), rd.evals_of(k)
            par = [e["score"] for e in evk if e.get("eval_sample") == "train" and e.get("suffix") == "parent" and e.get("score") is not None]
            chd = [e["score"] for e in evk if e.get("eval_sample") == "train" and e.get("suffix") == "child" and e.get("score") is not None]
            tr = (f"{par[0]:.3f}" if par else "n/a") + " -> " + (f"{max(chd):.3f}, {chd[-1]:.3f} (n={len(chd)})" if chd else "no child eval")
            cm = ", ".join(f"{e['score']:.3f}" + (" (harness)" if e.get("by_loop") else "")
                           for e in evk if e.get("eval_sample") == "comparison" and e.get("score") is not None) or "none"
            ch = changed(rd, k)
            h = []
            if rk.get("constraint_violation"):
                h.append("constraint checker reverted the code")
            if rk.get("candidate_admitted") is not None:
                h.append("admitted to the candidate pool" if rk["candidate_admitted"] else "not admitted to the candidate pool")
            sm = summaries.get(k, {}).get("text") or "(none)"
            sm = SCORE_RE.sub("[n]", sm)
            sm = sm if len(sm) <= SUMMARY_CAP else sm[:SUMMARY_CAP] + " [...]"
            L.append(f"| {k} | {rk.get('parent_iteration', '?')} | {'unknown' if ch is None else ('yes' if ch else 'no')} | "
                     f"{'; '.join(h) or '-'} | {tr} | {cm} | {norm(sm).replace('|', '/')} |")
    prior_md = "\n".join(L) + "\n"

    mem_before = rd.snaps.get(it - 1, "") if it > 1 else ""
    mem_after = rd.snaps.get(it, "")
    diff = git_diff(rd, rec["parent_sha"], rec["close_sha"])
    att = attempted_diff(rd, it)
    credits = rec.get("comparison_credits_after")

    L = [f"# Iteration {it}", "",
         "Facts the harness recorded for this iteration (cite OUTCOME). They are not the agent's statements.", "",
         f"- benchmark: {bench}; comparison (held-out) set {F.SET_ROWS[bench]} rows; train samples {F.TRAIN_ROWS} rows",
         f"- start code: the closing code of iteration {parent_it if parent_it is not None else '?'} (iteration 0 = the seed "
         "program); code label C0",
         f"- closing code: code label {close_label}; identical to the start code: "
         f"{'unknown' if is_changed is None else ('no' if is_changed else 'yes')}"]
    if gate:
        L += ["- constraint checker: FIRED. The harness found that the change broke a rule of the task and reverted the code "
              "to the start code (so the closing code equals the start code).",
              f"  - rule text: {norm(str(rec.get('constraint_violation')))[:1500]}"]
    else:
        L.append("- constraint checker: did not fire")
    L.append(f"- comparison credit left at the end of the iteration: {credits if credits is not None else 'not recorded'}")
    L.append("- the start code's comparison scores on record before this session: "
             + ("; ".join(f"{fmt(s)} ({src})" for s, src in ref) if ref else "none"))
    L.append(f"- noise: one SD of one eval of identical code: train {sd['train']:.4f}, comparison {sd['comparison']:.4f}")
    L.append(f"- evals in this iteration: {len(ev_rows)} (evals.md)")
    L.append(f"- memory_before.md: {len(mem_before.splitlines())} lines; memory_after.md: {len(mem_after.splitlines())} lines "
             "(each line is prefixed with its number L<n>: in these files; the prefix is not part of the memory text)")
    L.append(f"- code_diff.patch: {'empty (no code change from the start code)' if not diff else str(len(diff.splitlines())) + ' lines'}"
             + ("; attempted.diff: the change the agent made before it was reverted" if att and not diff else ""))
    L += ["", "## Files", "",
          "| file | what it is |", "|---|---|",
          "| transcript.md | the session, turn by turn (the agent's text and tool calls, tool results, harness messages) |",
          "| evals.md | every eval of this iteration (EVAL#n) |",
          "| comparison_history.md | every comparison-set eval of the run up to the end of this iteration (CMP#n) |",
          "| earlier_iterations.md | one row per earlier iteration (PRIOR:n) |",
          "| memory_before.md / memory_after.md | the agent's memory file at the start / end of this iteration |",
          "| code_diff.patch | code change from the start code to the closing code |"]
    if att and not diff:
        L.append("| attempted.diff | the change the agent made before it was reverted |")
    iteration_md = "\n".join(L) + "\n"

    transcript, quotable = render_transcript(steps)
    rd.drop_steps(it)
    ws = out / "ws" / uid
    if ws.exists():
        shutil.rmtree(ws)
    ws.mkdir(parents=True)
    files = {"ITERATION.md": iteration_md, "transcript.md": transcript, "evals.md": evals_md,
             "comparison_history.md": cmp_md, "earlier_iterations.md": prior_md,
             "memory_before.md": wrap(numbered(mem_before)) + "\n", "memory_after.md": wrap(numbered(mem_after)) + "\n",
             "code_diff.patch": (cap_sections(diff) or "(empty: the closing code has the same content as the start code)") + "\n"}
    if att and not diff:
        files["attempted.diff"] = cap_sections(att) + "\n"
    for name, text in files.items():
        (ws / name).write_text(text, encoding="utf-8")
    meta = {"unit_id": uid, "run": spec.run, "model": spec.model, "arm": spec.arm, "bench": bench, "seed": spec.seed,
            "iteration": it, "workspace_sha256": workspace_sha256(ws),
            "sizes": {n: len(t) for n, t in files.items()},
            "quotable": {str(k): v for k, v in quotable.items()}, "memory_after": mem_after.split("\n"),
            "facts": {"changed": is_changed, "gate": gate, "credits_after": credits, "close_label": close_label,
                      "n_evals": len(ev_rows), "ref": [s for s, _ in ref]}}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / f"{uid}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def run_build(args, runs: Runs) -> None:
    units = select_units(runs, args.only, args.limit)
    for spec in {u[0].run: u[0] for u in units}.values():
        load_summaries(args.out, spec.run)
    sizes = []
    for i, (spec, it, uid) in enumerate(units, 1):
        m = build_unit(args.out, runs, spec, it, uid)
        sizes.append(sum(m["sizes"].values()))
        if i % 25 == 0 or i == len(units):
            log(f"[build] {i}/{len(units)} workspaces")
    sizes.sort()
    log(f"[build] workspace chars: median {sizes[len(sizes) // 2]:,}, max {sizes[-1]:,}, total {sum(sizes):,} "
        f"(~{sum(sizes) // 4:,} tokens if every file were read once)")


# ---------------------------------------------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------------------------------------------
def count_prompt(it: int) -> str:
    head = (COUNT_PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{definitions}", DEFINITIONS_PATH.read_text(encoding="utf-8").strip())
            .replace("{shape}", COUNT_SHAPE).rstrip())
    return head + (f"\n\n## Your task now\n\nYour working directory is the workspace of iteration {it}. Read the files, "
                   "decide all nine modes, and end your reply with the one fenced ```json block.\n")


def quote_found(inst: dict, meta: dict) -> str | None:
    """None when the quote is found verbatim (whitespace aside) where the instance says; else the problem."""
    q = norm(inst.get("quote") or "").strip('"“”')
    if not q:
        return "empty quote"
    if inst.get("source") == "memory":
        return None if any(q in norm(ln) for ln in meta["memory_after"]) else "not found in one line of memory_after.md"
    t = inst.get("turn")
    try:
        t = int(str(t).lstrip("T"))
    except (TypeError, ValueError):
        return f"turn {t!r} is not a turn number"
    texts = meta["quotable"].get(str(t))
    if texts is None:
        return f"T{t} is not an assistant turn"
    if any(q in norm(x) for x in texts):
        return None
    near = [k for k, xs in meta["quotable"].items() if any(q in norm(x) for x in xs)]
    return f"not found in T{t}" + (f" (found in T{', T'.join(near[:3])})" if near else "")


def validate_count(d: dict, meta: dict) -> list[str]:
    """Fatal (a fresh session replaces the reply): a mode missing or not a list, an unknown mode, more than one instance of
    a binary mode, a bad source, or a quote that is empty, too long or not found verbatim. Warnings: none yet."""
    problems = []
    modes = d.get("modes")
    if not isinstance(modes, dict):
        raise ValueError("'modes' missing or not an object")
    for m in MODES:
        if not isinstance(modes.get(m), list):
            problems.append(f"mode {m} missing or not a list")
    for m in set(modes) - set(MODES):
        problems.append(f"unknown mode {m}")
    for m in MODES:
        lst = modes.get(m) if isinstance(modes.get(m), list) else []
        if m in BINARY and len(lst) > 1:
            problems.append(f"{m} has {len(lst)} instances (at most 1)")
        for i, inst in enumerate(lst):
            if not isinstance(inst, dict):
                problems.append(f"{m}[{i}] is not an object")
                continue
            if inst.get("source") not in SOURCES:
                problems.append(f"{m}[{i}] source {inst.get('source')!r}")
                continue
            if len(norm(inst.get("quote") or "")) > MAX_QUOTE:
                problems.append(f"{m}[{i}] quote longer than {MAX_QUOTE} characters")
            why = quote_found(inst, meta)
            if why:
                problems.append(f"{m}[{i}] quote {why}: {norm(inst.get('quote') or '')[:80]!r}")
    if problems:
        raise ValueError("invalid count: " + "; ".join(problems))
    return []


def _windows_opencode_exe() -> None:
    """On Windows the npm `opencode` on PATH is a .cmd shim, which the engine's Popen(["opencode", ...]) cannot start.
    Put the directory of the package's native opencode.exe first on PATH (this process and its children only)."""
    if os.name != "nt" or shutil.which("opencode.exe"):
        return
    shim = shutil.which("opencode")
    exe = Path(shim).parent / "node_modules" / "opencode-ai" / "bin" / "opencode.exe" if shim else None
    if not exe or not exe.exists():
        raise SystemExit("opencode.exe not found (npm's opencode.cmd cannot be started by the engine harness)")
    os.environ["PATH"] = str(exe.parent) + os.pathsep + os.environ.get("PATH", "")


def _session_env(args) -> None:
    _windows_opencode_exe()
    # The Anthropic Python SDK reads ANTHROPIC_BASE_URL without /v1; OpenCode's AI-SDK provider reads it WITH /v1 and gets
    # 404 Not Found on the bare host. When it is just the default host, drop it (OpenCode then uses its default, the same
    # host with /v1). Any other value is left alone.
    if os.environ.get("ANTHROPIC_BASE_URL", "").rstrip("/") == "https://api.anthropic.com":
        os.environ.pop("ANTHROPIC_BASE_URL")
    prepare_engine_env(args.out / "opencode", model=args.model)
    load_env_key(REPO_ROOT, HERE.parent / "macro-counts")
    key_var = provider_key(args.model)
    if key_var and not os.environ.get(key_var):
        raise SystemExit(f"{key_var} not set (checked env, <repo>/.env, scripts/macro-counts/.env)")
    if not shutil.which("opencode"):
        raise SystemExit("opencode not found on PATH")


def _cached(path: Path, want: dict, force: bool) -> dict | None:
    if force or not path.exists():
        return None
    jd = json.loads(path.read_text(encoding="utf-8"))
    stale = [k for k, v in want.items() if jd.get(k) != v]
    if stale:
        log(f"[{path.stem}] cached result is stale ({', '.join(stale)} changed); re-running")
        return None
    return jd


def run_count(args, runs: Runs) -> None:
    units = select_units(runs, args.only, args.limit)
    cdir = args.out / "count"
    cdir.mkdir(parents=True, exist_ok=True)
    (args.out / "prompts").mkdir(parents=True, exist_ok=True)
    todo = []
    for spec, it, uid in units:
        mpath = args.out / "meta" / f"{uid}.json"
        if not mpath.exists():
            raise SystemExit(f"no workspace for {uid} ({spec.run}:{it}): run `fm_count.py build` first")
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        ws = args.out / "ws" / uid
        if workspace_sha256(ws) != meta["workspace_sha256"]:
            raise SystemExit(f"{uid}: workspace changed since build; rebuild it")
        prompt = count_prompt(it)
        want = {"model": args.model, "prompt_sha256": sha256_text(prompt), "workspace_sha256": meta["workspace_sha256"]}
        if _cached(cdir / f"{uid}.json", want, args.force) is not None:
            continue
        todo.append((uid, ws, meta, prompt, want))
    p0 = count_prompt(1)
    assert len(count_prompt(99)) < MAX_PROMPT_CHARS, len(count_prompt(99))
    (args.out / "prompts" / "count_prompt.iteration_1.md").write_text(p0, encoding="utf-8")
    log(f"[count] {len(todo)} of {len(units)} iterations to count; prompt ~{len(p0) // 4:,} tokens "
        f"(prompts/count_prompt.iteration_1.md)")
    if args.dry_run or not todo:
        return
    _session_env(args)

    def _one(uid: str, ws: Path, meta: dict, prompt: str, want: dict) -> dict:
        jpath = cdir / f"{uid}.json"
        jd = run_json_session(ws, prompt, required_key="modes", label=f"count {uid}",
                              validate=lambda d: validate_count(d, meta), model=args.model,
                              max_turns=args.max_turns, timeout_seconds=args.timeout, retries=args.retries,
                              allow_bash=False, transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=log)
        jd = {"unit_id": uid, "run": meta["run"], "iteration": meta["iteration"], **want, **jd}
        jpath.write_text(json.dumps(jd, indent=1, ensure_ascii=False), encoding="utf-8")
        return jd

    failed, done, spent = [], 0, 0.0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as ex:
        futs = {ex.submit(_one, *t): t[0] for t in todo}
        for fut in as_completed(futs):
            uid = futs[fut]
            try:
                jd = fut.result()
                done += 1
                c = usd(jd.get("usage"))
                spent += c
                hits = " ".join(f"{m}{len(v)}" for m, v in jd["result"]["modes"].items() if v) or "-"
                log(f"[count] {done}/{len(todo)} {uid}: {hits} (tool_calls={jd['n_tool_calls']} retries={jd['retries']} "
                    f"{jd['elapsed_seconds']}s ${c:.2f}; accepted sessions so far ${spent:.2f})")
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(uid)
                log(f"[count] {uid}: FAILED: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} iteration(s) failed: {','.join(failed)} (re-run `count`; finished ones are cached)")


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
CELLS = [("Qwen3", "freeform"), ("Qwen3", "pareto"), ("GLM-5.2", "freeform"), ("GLM-5.2", "pareto")]
STOPPED = {"rule_revert", "rejected", "unused"}
CODE = {"MS", "WR", "KR", "BC", "RP"}
HEADLINE = ("MS", "WR", "KR", "BO", "BC")
NAMES = {"MS": "Misread score", "WR": "Wrong-reference comparison", "KR": "Kept a measured regression (> 1 SD)",
         "BO": "Blind memory overwrite", "BC": "Broke a constraint", "FN": "False number written to memory",
         "UR": "Measured failure never recorded (> 1 SD)", "RP": "Repeated an earlier failed change",
         "UL": "Unmeasured landing"}
ROWS = [("MS", False), ("WR", False), ("KR", False), ("BO", False), ("BC", False), ("MS or WR", False), ("ANY", False),
        ("FN", True), ("UR", True), ("RP", True), ("UL", True)]
ROW_NAME = {"MS or WR": "**Misread score or wrong reference**", "ANY": "**Any of the five modes above**"}
ROUTE_TEXT = {"passed_on": "got through", "no_new_code": "got through (agent closed on its parent's code)",
              "elsewhere": "got through (next iteration restarted from a merge base)",
              "rule_revert": "stopped: reverted by the rule checker", "rejected": "stopped: not admitted",
              "unused": "stopped: admitted, never built on or chosen"}


def route(rd: F.RunData, spec: F.RunSpec, it: int) -> str:
    """What became of the code iteration `it` closed on (the rules of results/F1/damage/seed_and_routing.py, from
    history.json). First match wins: rule_revert, no_new_code, (freeform) passed_on / elsewhere, (pareto) rejected,
    passed_on (a later iteration started from it, or it is the final program), unused."""
    rec = rd.rec(it)
    if rec.get("constraint_violation"):
        return "rule_revert"
    if changed(rd, it) is False:
        return "no_new_code"
    its = rd.iterations
    if spec.arm == "freeform":
        if it == max(its):
            return "passed_on"
        return "passed_on" if rd.rec(it + 1).get("parent_sha") == rec.get("close_sha") else "elsewhere"
    if not rec.get("candidate_admitted"):
        return "rejected"
    kids = sum(1 for k in its if k > it and rd.rec(k).get("parent_iteration") == it)
    fin = (rd.history.get("summary") or {}).get("final_candidate") or {}
    is_final = fin.get("git_branch") == rec.get("git_branch") or fin.get("git_sha") == rec.get("close_sha")
    return "passed_on" if kids > 0 or is_final else "unused"


def raw(u, m):
    if m == "MS or WR":
        return raw(u, "MS") or raw(u, "WR")
    if m == "ANY":
        return any(raw(u, x) for x in HEADLINE)
    return u[m] > 0


def eff(u, m):
    if m == "MS or WR":
        return eff(u, "MS") or eff(u, "WR")
    if m == "ANY":
        return any(eff(u, x) for x in HEADLINE)
    return raw(u, m) and not (m in CODE and u["route"] in STOPPED)


def run_shares(units, m, a, code, f):
    runs = sorted({(u["bench"], u["seed"]) for u in units if (u["model"], u["arm"]) == (m, a)})
    assert len(runs) == 6, (m, a, runs)
    return [sum(f(u, code) for u in us) / len(us) for b, s in runs
            for us in [[u for u in units if (u["model"], u["arm"], u["bench"], u["seed"]) == (m, a, b, s)]]]


def perm(x, y):
    """Exact two-sided permutation test on the difference of means (6 runs vs 6: 924 splits)."""
    pool, obs = x + y, abs(sum(x) / len(x) - sum(y) / len(y))
    hits = tot = 0
    for idx in itertools.combinations(range(len(pool)), len(x)):
        a = [pool[i] for i in idx]
        b = [pool[i] for i in range(len(pool)) if i not in idx]
        tot += 1
        hits += abs(sum(a) / len(a) - sum(b) / len(b)) >= obs - 1e-12
    return hits / tot


def fmt_p(p):
    return f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def definition_of(mode: str) -> str:
    text = DEFINITIONS_PATH.read_text(encoding="utf-8")
    m = re.search(rf"^## {mode} .*?\n\nDefinition\. (.*?)\n\n", text, re.S | re.M)
    return norm(m.group(1)) if m else "?"


def run_report(args, runs: Runs) -> None:
    units, insts, missing = [], [], []
    prompt_sha = {sha256_text(count_prompt(it)) for it in range(1, 200)}
    for spec, it, uid in runs.units():
        p = args.out / "count" / f"{uid}.json"
        if not p.exists():
            missing.append(uid)
            continue
        jd = json.loads(p.read_text(encoding="utf-8"))
        meta = json.loads((args.out / "meta" / f"{uid}.json").read_text(encoding="utf-8"))
        assert jd["prompt_sha256"] in prompt_sha and jd["workspace_sha256"] == meta["workspace_sha256"], uid
        validate_count(jd["result"], meta)   # re-check every quote against the raw turns
        rd = runs.rd(spec.run)
        u = {"unit_id": uid, "run": spec.run, "model": spec.model, "arm": spec.arm, "bench": spec.bench, "seed": spec.seed,
             "iteration": it, "route": route(rd, spec, it), "facts": meta["facts"]}
        for m in MODES:
            lst = jd["result"]["modes"][m]
            u[m] = len(lst)
            for i, inst in enumerate(lst):
                insts.append({"unit_id": uid, "model": spec.model, "arm": spec.arm, "bench": spec.bench, "seed": spec.seed,
                              "iteration": it, "route": u["route"], "mode": m, "i": i, "turn": inst.get("turn"),
                              "source": inst["source"], "quote": norm(inst["quote"]),
                              "cites": " ".join(map(str, inst.get("cites") or [])),
                              "explanation": norm(inst.get("explanation") or "")})
        u.update({"session_id": jd["session_id"], "judged_at": jd["judged_at"], "retries": jd["retries"],
                  "n_tool_calls": jd["n_tool_calls"], "review_flags": " | ".join(map(norm, jd["result"].get("review_flags") or []))})
        units.append(u)
    if missing:
        raise SystemExit(f"{len(missing)} of 351 iterations not counted yet (e.g. {missing[:5]}); run `count` first")
    res = args.results
    res.mkdir(parents=True, exist_ok=True)
    ufields = ["unit_id", "run", "model", "arm", "bench", "seed", "iteration", "route", *MODES, "session_id", "judged_at",
               "retries", "n_tool_calls", "review_flags"]
    with open(res / "units.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ufields, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(units)
    with open(res / "instances.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(insts[0]) if insts else ["unit_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(insts)

    n = {c: sum(1 for u in units if (u["model"], u["arm"]) == c) for c in CELLS}
    md = ["# Failure modes of Qwen3 and GLM-5.2, with and without pareto evolution", "",
          "*Generated by `scripts/failure-modes/fm_count.py report` (fm-v2: one OpenCode session per iteration, judge "
          f"{args.model}). Do not edit by hand; rerun the script.*", "",
          "## Table: iterations with an uncaught mistake", "",
          "**Main number:** iterations where the model made the mistake and the harness did not stop it. **[Brackets]:** every "
          "iteration where the model made the mistake, including ones the harness stopped.", "",
          "| Failure mode | " + " | ".join(f"{m} {a}" for m, a in CELLS) + " |", "|---|---|---|---|---|"]
    for code, low in ROWS:
        cells = []
        for c in CELLS:
            us = [u for u in units if (u["model"], u["arm"]) == c]
            e, r = sum(eff(u, code) for u in us), sum(raw(u, code) for u in us)
            cells.append(f"{e} of {len(us)} ({100 * e / len(us):.0f}%) [{r}]")
        md.append(f"| {'*Lower priority:* ' if low else ''}{ROW_NAME.get(code, NAMES.get(code))} | " + " | ".join(cells) + " |")
    md += ["", "**Run-level tests** (mean of the 6 run shares; exact permutation test, 6 runs vs 6, 924 splits, two-sided):", "",
           "| Comparison | Misread or wrong reference, uncaught | same, all | Any of the five, uncaught | same, all |",
           "|---|---|---|---|---|"]
    tests = {}
    for lab, c1, c2 in [("Qwen3: freeform vs pareto", CELLS[0], CELLS[1]), ("GLM-5.2: freeform vs pareto", CELLS[2], CELLS[3]),
                        ("Qwen3 vs GLM-5.2, freeform", CELLS[0], CELLS[2]), ("Qwen3 vs GLM-5.2, pareto", CELLS[1], CELLS[3])]:
        cells = []
        for code in ("MS or WR", "ANY"):
            for fname, f in (("eff", eff), ("raw", raw)):
                x, y = run_shares(units, *c1, code, f), run_shares(units, *c2, code, f)
                p = perm(x, y)
                tests[(lab, code, fname)] = (sum(x) / 6, sum(y) / 6, p)
                cells.append(f"{100 * sum(x) / 6:.0f}% vs {100 * sum(y) / 6:.0f}%, p = {fmt_p(p)}")
        md.append(f"| {lab} | " + " | ".join(cells) + " |")
    retried = sum(1 for u in units if u["retries"])
    md += ["", "**How the table was made**",
           f"- Every iteration of every run is counted (6 runs per column: HoVer and PhantomWiki x 3 seeds), {len(units)} in all. "
           "An iteration counts once per mode, however many times the mistake happens in it.",
           f"- **All nine modes are decided by AI** ({args.model} in OpenCode, read-only tools), one session per iteration. "
           "The session reads the iteration's transcript (the agent's private reasoning left out), its evals, the run's "
           "comparison-set history, the earlier iterations, memory before and after, and the code diff, and must back every "
           f"instance with a word-for-word quote of the agent's words. A program checked every quote against the raw logs; "
           f"a reply with a quote not found was replaced by a fresh session ({retried} iterations needed one or more).",
           "- Missing change summaries were first written from the code diff by a separate session per run (context only).",
           "- **Stopped by the harness** (from history.json): the rule checker reverted the code; or pareto did not admit the "
           "child; or pareto admitted it but no later iteration built on it and it is not the final program.",
           "- **Memory mistakes and unmeasured landings always count:** memory is one file per run, shared by every branch, so "
           "no harness choice undoes it; unmeasured landing is set by the arm.",
           "- **Iterations where the agent itself closed on its parent's code still count:** that is the agent, not the harness."]
    jds = [json.loads((args.out / "count" / f"{u['unit_id']}.json").read_text(encoding="utf-8")) for u in units]
    c_ok = sum(usd(j.get("usage")) for j in jds)
    c_rej = sum(usd(json.loads(p.read_text(encoding="utf-8")).get("usage")) for p in (args.out / "rejected").glob("count_*.json"))
    c_fill = sum(usd(json.loads(p.read_text(encoding="utf-8"))["meta"].get("usage")) for p in (args.out / "summaries").glob("*.json"))
    c_fill += sum(usd(json.loads(p.read_text(encoding="utf-8")).get("usage")) for p in (args.out / "rejected").glob("fill_*.json"))
    secs = sorted(float(j.get("elapsed_seconds") or 0) for j in jds)
    md.append(f"- Cost (list prices of `anthropic_api.PRICE`): ${c_ok:.2f} accepted count sessions + ${c_rej:.2f} rejected "
              f"count sessions + ${c_fill:.2f} summary fill = ${c_ok + c_rej + c_fill:.2f}. Session time: median "
              f"{secs[len(secs) // 2] / 60:.1f} min, longest {secs[-1] / 60:.1f} min.")

    # quotes: fixed rule
    def rank(x):
        ln = len(x["quote"])
        return (0 if 40 <= ln <= 250 else 1, x["unit_id"], x["i"])

    shown = []
    md += ["", "## What each mode means, with examples", "",
           "Examples are chosen by a fixed rule: for every model x arm with an instance, one iteration where the mistake got "
           "through and, for code modes, one the harness stopped when there is one; among candidates, quotes of 40-250 "
           "characters first, then by unit id. Quotes are verbatim agent text, tool calls or memory lines, checked word for "
           "word against the raw logs. **Why** is the AI's own explanation, verbatim. Cites: T<n> = turn, EVAL#n = n-th eval "
           "of the iteration, CMP#n = comparison-set history row, PRIOR:n = iteration n, MEM-BEFORE/MEM-AFTER:L<n> = memory "
           "line, OUTCOME = harness record, C0, C1 ... = code versions (C0 = start code), S1, S2 ... = train samples.", ""]
    for m in MODES:
        low = m in ("FN", "UR", "RP", "UL")
        md += [f"### {NAMES[m]}{' (lower priority)' if low else ''}", "", f"*Definition:* {definition_of(m)}", ""]
        for c in CELLS:
            cand = [x for x in insts if x["mode"] == m and (x["model"], x["arm"]) == c]
            if not cand:
                md.append(f"- *{c[0]} {c[1]}:* no instance in {n[c]} iterations.")
                continue
            for want_st in ((False, True) if m in CODE else (False,)):
                pool = sorted((x for x in cand if m not in CODE or (x["route"] in STOPPED) == want_st), key=rank)
                if not pool:
                    continue
                x = pool[0]
                status = ROUTE_TEXT[x["route"]] if m in CODE else "counts (cannot be stopped)"
                md += [f"- *{c[0]} {c[1]}, {status}* ({x['bench']} seed {x['seed']}, iteration {x['iteration']}, unit "
                       f"{x['unit_id']}, {x['source']}" + (f" T{x['turn']}" if x["turn"] not in (None, "") else "") + "):",
                       f"  > \"{x['quote']}\"", f"  **Why:** {x['explanation']}"]
                shown.append({**x, "example": "stopped" if want_st else "through"})
        md.append("")

    qf, qp = CELLS[0], CELLS[1]

    def cnt(c, code, f):
        return sum(f(u, code) for u in units if (u["model"], u["arm"]) == c)

    def passed(c, code=None):
        us = [u for u in units if (u["model"], u["arm"]) == c and (code is None or raw(u, code))]
        return sum(u["route"] == "passed_on" for u in us), len(us)

    t1, t2 = tests[("Qwen3: freeform vs pareto", "MS or WR", "eff")], tests[("Qwen3: freeform vs pareto", "ANY", "eff")]
    pb, nb = passed(qp)
    pd, nd = passed(qp, "MS or WR")
    pk, nk = passed(qp, "KR")
    md += ["## Numbers in short", "",
           f"- Qwen3 made a misread-score or wrong-reference mistake in {cnt(qf, 'MS or WR', raw)} freeform and "
           f"{cnt(qp, 'MS or WR', raw)} pareto iterations; {cnt(qf, 'MS or WR', eff)} of {n[qf]} and {cnt(qp, 'MS or WR', eff)} "
           f"of {n[qp]} got through (run means {100 * t1[0]:.0f}% vs {100 * t1[1]:.0f}%, p = {fmt_p(t1[2])}).",
           f"- Any of the five headline modes, uncaught: Qwen3 {cnt(qf, 'ANY', eff)} of {n[qf]} freeform vs {cnt(qp, 'ANY', eff)} "
           f"of {n[qp]} pareto (run means {100 * t2[0]:.0f}% vs {100 * t2[1]:.0f}%, p = {fmt_p(t2[2])}).",
           f"- Under pareto, {pb} of {nb} ({100 * pb / max(nb, 1):.0f}%) of all Qwen3 children were passed on (built on or "
           f"final), against {pd} of {nd} ({100 * pd / max(nd, 1):.0f}%) of those with a misread or wrong reference and "
           f"{pk} of {nk} of those with a kept regression.",
           f"- GLM-5.2 iterations with any of the five headline modes, before the harness: {cnt(CELLS[2], 'ANY', raw)} of "
           f"{n[CELLS[2]]} freeform and {cnt(CELLS[3], 'ANY', raw)} of {n[CELLS[3]]} pareto; Qwen3: {cnt(qf, 'ANY', raw)} of "
           f"{n[qf]} and {cnt(qp, 'ANY', raw)} of {n[qp]}."]
    (res / "TABLE_AND_QUOTES.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(res / "quotes_selected.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, list(shown[0]) if shown else ["unit_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(shown)
    log(f"[report] wrote {res / 'TABLE_AND_QUOTES.md'} ({len(units)} iterations, {len(insts)} instances, {len(shown)} quotes)")
    compare_f1(units, res)


F1_UNITS = DEFAULT_RESULTS.parent / "F1" / "damage" / "routing_by_unit.csv"


def compare_f1(units: list[dict], res: Path) -> None:
    """COMPARE_F1.md: per mode, iterations flagged by F1 (the earlier pack + Python pipeline) and by this run. Not for the paper;
    it shows how far the counts moved. Also checks that the harness routing is the same."""
    if not F1_UNITS.exists():
        return
    old = {r["unit_id"]: r for r in csv.DictReader(open(F1_UNITS, encoding="utf-8"))}
    assert all(old[u["unit_id"]]["route"] == u["route"] for u in units), "routing differs from F1"
    md = ["# fm-v2 (this run) against F1", "",
          "Per mode, iterations with at least one instance (raw, before the harness). Routing is identical to F1's for all "
          f"{len(units)} iterations.", "",
          "| Mode | " + " | ".join(f"{m} {a}: F1 -> fm-v2" for m, a in CELLS) + " | both | F1 only | fm-v2 only |",
          "|---|---|---|---|---|---|---|---|"]
    rows = []
    for m in MODES:
        cells = []
        for c in CELLS:
            us = [u for u in units if (u["model"], u["arm"]) == c]
            cells.append(f"{sum(int(old[u['unit_id']][m]) > 0 for u in us)} -> {sum(u[m] > 0 for u in us)}")
        both = sum(int(old[u["unit_id"]][m]) > 0 and u[m] > 0 for u in units)
        f1o = sum(int(old[u["unit_id"]][m]) > 0 and not u[m] for u in units)
        f2o = sum(not int(old[u["unit_id"]][m]) and u[m] > 0 for u in units)
        md.append(f"| {m} | " + " | ".join(cells) + f" | {both} | {f1o} | {f2o} |")
        rows += [{"unit_id": u["unit_id"], "mode": m, "f1": int(old[u["unit_id"]][m]) > 0, "fm_v2": u[m] > 0}
                 for u in units if (int(old[u["unit_id"]][m]) > 0) != (u[m] > 0)]
    # instances that contradict a harness fact their definition requires (ITERATION.md); flagged, not filtered
    need = {"BC": lambda f: f["gate"],
            "KR": lambda f: f["changed"] and not f["gate"],
            "UL": lambda f: f["changed"] and not f["gate"] and (f["credits_after"] or 0) > 0}
    bad = [f"{u['unit_id']} {m}" for u in units for m, ok in need.items() if u[m] and not ok(u["facts"])]
    md += ["", "## Instances that contradict a harness fact", "",
           "BC needs the constraint checker to have reverted the code; KR and UL need changed code that the checker did not "
           "revert; UL also needs comparison credit left at the end. "
           f"Contradictions: {len(bad)}" + (f" ({', '.join(bad)})." if bad else ".")]
    (res / "COMPARE_F1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(res / "compare_f1_disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["unit_id", "mode", "f1", "fm_v2"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    log(f"[report] wrote {res / 'COMPARE_F1.md'} ({len(rows)} unit-mode disagreements)")


# ---------------------------------------------------------------------------------------------------------------
def main() -> int:
    print("fm_count.py is a helper module for fm_v3.py; the fm-v2 counter is retired. Run fm_v3.py.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
