#!/usr/bin/env python3
"""fm_v3.py -- failure modes per iteration: Python facts + judge sessions over a filtered trace (fm-v3).

Design (the paragraph for the paper):
  Deterministic modes (KR kept a measured regression, UL unmeasured landing, BO blind memory overwrite) are computed in
  Python from the harness logs (fm_facts.py). For the rest, each iteration's architect trace passes through ONE general
  section filter: every block the agent wrote visibly, every result of a harness tool (ce_*) and every short result
  (<= 1,000 characters) is kept verbatim; other tool results and code-edit bodies become a one-line stub naming the turn,
  the tool, the size and a file with the full text. Pass A: the filtered trace goes INLINE in the prompt of one read-only
  OpenCode session per iteration, so the judge sees every visible word the agent wrote; it opens stubs (read) or searches
  the full turns (grep) when a verdict needs them. The judge lists every score claim with a verdict (ok / MS / WR / FN /
  corrected) and maps every score-like number of the agent's visible words (found by Python) to a claim or to "-"; a reply
  that skips a number, or quotes words not found verbatim in the agent's text, tool calls or memory, is replaced by a fresh
  session. UR (from Python's failed-version list) and BC (only when the constraint checker fired) are decided in the same
  session. The agent's private reasoning is not evidence in pass A.
  FN (false number written to memory) is judged in pass A but NOT reported: it overlaps with MS / WR.
  Pass B (only iterations with reasoning, i.e. GLM-5.2) was used ONLY to check whether adding the thinking flips pass-A
  verdicts; it is not part of the main counts. (Result, gpt-6-sol on 40 GLM-5.2 iterations: 1 of 23 WR and 0 of 4 FN
  verdicts flipped, 0 of 492 ok claims became errors, so the main table uses pass A alone.) A second session gets pass A's
  claims, the tables and every assistant turn with its reasoning and visible text inline, and re-judges each claim. Instances stay claims in the
  visible words; the reasoning can only change how a visible claim is read (which reference it used, a correction before
  acting). Comparing A and B answers whether seeing the reasoning changes the counts.

Stages:
  build   <out>/ws/<unit>/ (turns/, memory_before.md, memory_after.md, code_diff.patch, attempted.diff),
          <out>/tables|inline|reasoning/<unit>.md (the prompts' evidence) and <out>/meta/<unit>.json (facts, quotable text,
          number ids; outside the workspace). Unit ids are opaque (fm_runs.unit_id); no model, arm or run name is written.
  count   --pass A (default; all iterations) or --pass B (iterations with reasoning; needs A); one session per iteration
          (--jobs in parallel); cached by model, prompt hash and workspace hash. -> <out>/count/ and <out>/rethink/
  report  TABLE.md (pass A: uncaught [all] per model x arm, exact run-level permutation tests; pass B: the same with the
          re-judged verdicts and the A -> B changes), units.csv, instances.csv, claims.csv; with --f2 <dir holding fm-v2
          units.csv>, COMPARE_F2.md (unit-level agreement, recall).

HOW TO RUN (repo root, macOS/Linux: the prompt goes on the command line, up to ~600k characters):
  .venv/bin/python scripts/failure-modes/fm_v3.py build
  .venv/bin/python scripts/failure-modes/fm_v3.py count --dry-run
  .venv/bin/python scripts/failure-modes/fm_v3.py count --jobs 4 [--only U-xxxxxxxx,...]
  .venv/bin/python scripts/failure-modes/fm_v3.py count --pass B --jobs 4
  .venv/bin/python scripts/failure-modes/fm_v3.py report [--partial] [--f2 <dir>]
Read-only on the run folders (default $CE_RUNS_DIR, else ~/ce-runs).
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
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fm_count as C  # noqa: E402  (shared helpers of fm-v2: wrap, numbered, git_diff, route, perm, session env, cache)
import fm_facts as X  # noqa: E402
import fm_runs as F  # noqa: E402
from anthropic_api import cost as usd  # noqa: E402
from judge import JudgeError, run_json_session, workspace_sha256  # noqa: E402

DEFAULT_OUT = Path.home() / "ce-analysis" / "fm-v3"
DEFAULT_MODEL = C.DEFAULT_MODEL
DEFINITIONS_PATH = HERE / "fm_definitions_v3.md"   # NOTE: FN in this file was judged but is not reported (overlaps with MS / WR)
PROMPT_PATH = HERE / "fm_v3_prompt.md"
RETHINK_PROMPT_PATH = HERE / "fm_v3_rethink_prompt.md"
SHORT_RESULT = 1_000            # tool results up to this many characters are kept in full
MAX_PROMPT_BYTES = 800_000      # macOS ARG_MAX is 1 MiB for argv + environment
DEFAULT_MAX_TURNS = 100

JUDGE_MODES = ("MS", "WR", "FN", "UR", "BC")
PY_MODES = ("KR", "UL", "BO")
MODES = ("MS", "WR", "KR", "BO", "BC", "FN", "UR", "UL")
VERDICTS = ("ok", "MS", "WR", "FN", "corrected")
SOURCES = ("assistant", "tool_call", "memory")      # pass A: visible words only
PASS_DIR = {"A": "count", "B": "rethink"}


def pass_dir(out: Path, p: str, api_judge: str | None = None) -> Path:
    """Result folder of a pass. With an API judge (e.g. gpt-6-sol) both passes get their own folders, so a within-judge
    A -> B comparison never mixes judges."""
    return out / (PASS_DIR[p] + (f"_{api_judge}" if api_judge else ""))
NAMES = {"MS": "Misread score", "WR": "Wrong-reference comparison", "KR": "Kept a measured regression (> 1 SD)",
         "BO": "Blind memory overwrite", "BC": "Broke a constraint", "FN": "False number written to memory",
         "UR": "Measured failure never recorded (> 1 SD)", "UL": "Unmeasured landing",
         }
ROWS = ("MS", "WR", "KR", "BO", "BC", "MS or WR", "ANY", "UR", "UL")   # FN not reported: overlaps with MS / WR
ROW_NAME = {"MS or WR": "**Misread score or wrong reference**", "ANY": "**Any of the five modes above**"}

SHAPE = """{
  "claims": [
    {"id": "C1", "turn": <n or null>, "source": "assistant|tool_call|memory", "quote": "<verbatim, <= 300 chars>",
     "verdict": "ok|MS|WR|FN|corrected", "clause": "<e.g. MS-1, WR-b, FN-wrong-set; empty for ok>",
     "cites": ["T<n>", "EVAL#<n>", "..."], "explanation": "<= 400 chars; the numbers for any verdict other than ok>"}
  ],
  "numbers": {"N1": "C1", "N2": "-"},
  "UR": [{"turn": <n>, "source": "assistant|tool_call", "quote": "...", "cites": ["EVAL#<n>"], "explanation": "..."}],
  "BC": [],
  "review_flags": ["<anything a human should double-check>"]
}"""

SHAPE_B = """{
  "reviews": {
    "C1": {"verdict": "ok|MS|WR|FN|corrected", "turn": <n of the thinking block that changes it, or null>,
           "explanation": "<= 400 chars, your own words; empty when unchanged>"}
  },
  "thinking_only": [{"turn": <n>, "explanation": "<your own words: a mistake that never reached the output>"}],
  "review_flags": ["<anything a human should double-check>"]
}"""


class Runs(C.Runs):
    """fm_count.Runs with the scratch git folder under <out> (fm_runs' default is a Windows path)."""

    def __init__(self, runs_root: Path, private: Path):
        super().__init__(runs_root)
        self.private = private

    def rd(self, run: str) -> F.RunData:
        with self._mx:
            if run not in self._rd:
                self._rd[run] = F.RunData(self.root, F.spec_of(run), private=self.private)
            return self._rd[run]


# ---------------------------------------------------------------------------------------------------------------
# the filtered view
# ---------------------------------------------------------------------------------------------------------------
def brief(inp) -> str:
    """A short label of a tool call's input for a stub: the path, command or pattern, cut at 100 characters."""
    if not isinstance(inp, dict):
        return ""
    for k in ("filePath", "path", "command", "pattern", "name", "description"):
        if inp.get(k):
            v = " ".join(str(inp[k]).split())
            return f"{k}={v[:100]}{'…' if len(v) > 100 else ''}"
    return ""


def is_code_edit(call: dict) -> bool:
    inp = call["input"] if isinstance(call["input"], dict) else {}
    return call["name"] in ("edit", "write") and not F.is_memfile(inp.get("filePath") or inp.get("path") or "")


def render_call(c: dict, full: bool) -> list[str]:
    L = [f"CALL {c['name']}"]
    fields = X.call_fields(c["input"])
    if not full and is_code_edit(c):
        keep = [(k, v) for k, v in fields if k in ("filePath", "path")]
        body = [(k, v) for k, v in fields if k not in ("filePath", "path")]
        for k, v in keep:
            L.append(f"{k}: {v}")
        sizes = ", ".join(f"{k} {len(v):,} chars" for k, v in body)
        L.append(f"[code-edit body: {sizes} — turns/T{c['turn']}.md]")
        return L
    for k, v in fields:
        L += [f"{k}:", v] if "\n" in v or len(v) > 200 else [f"{k}: {v}"]
    return L


def render_turn(s: dict, full: bool) -> list[str]:
    """One step as markdown (thinking left out: pass A never sees it). full=False applies the section filter (stubs);
    full=True renders every visible block."""
    k, t = s["kind"], s["turn"]
    if k == "prompt":
        return [f"### T{t} task prompt (harness)", "", s["text"], ""]
    if k == "notice":
        return [f"### T{t} harness message", "", s["text"], ""]
    if k == "assistant":
        L = [f"### T{t} assistant", ""]
        for tx in s["texts"]:
            L += [tx, ""]
        for c in s["calls"]:
            L += render_call(c, full) + [""]
        for p in s["patches"]:
            L += [f"(harness note) {p}", ""]
        return L
    if k == "results":
        L = [f"### T{t} tool results", ""]
        for r in s["results"]:
            ct = f"T{r['call_turn']}" if r.get("call_turn") is not None else "?"
            head = f"RESULT of {r['name']} (called at {ct})" + (" [error]" if r["is_error"] else "")
            keep = full or str(r["name"]).startswith("ce_") or len(r["content"]) <= SHORT_RESULT
            if keep:
                L += [head, r["content"], ""]
            else:
                L += [f"[T{t} result of {r['name']}({brief(r['input'])}): {len(r['content']):,} chars — turns/T{t}.md]", ""]
        for e in s["extra"]:
            L += ["(harness text)", e, ""]
        return L
    return []


def render_view(steps: list[dict]) -> tuple[str, dict[str, str], dict]:
    """(view.md, {relative path: text} of turns/, quotable texts per turn by source). The thinking texts go into
    `quotable` (outside the workspace) for pass B's quote check only."""
    L = ["## Transcript (filtered)", "",
         "Turns are numbered T<n>. `### T<n> assistant` holds the agent's text and its tool calls (`CALL <tool>`, then one "
         "`<field>:` block per input field). `### T<n> tool results` holds what the tools returned (not the agent's words). "
         "`### T<n> harness message` is text from the harness. Stubs `[... — <file>]` name the file with the full text.", ""]
    files: dict[str, str] = {}
    quotable: dict[str, dict[str, list[str]]] = {}
    for s in steps:
        if s["kind"] == "meta":
            continue
        L += render_turn(s, full=False)
        t = s["turn"]
        if s["kind"] in ("assistant", "results"):
            files[f"turns/T{t}.md"] = C.wrap("\n".join(render_turn(s, full=True))) + "\n"
        if s["kind"] == "assistant":
            q = quotable.setdefault(str(t), {"assistant": [], "tool_call": [], "thinking": []})
            q["assistant"] += s["texts"]
            q["thinking"] += s["thinks"]
            for c in s["calls"]:
                q["tool_call"] += [v for _, v in X.call_fields(c["input"])] + [json.dumps(c["input"], ensure_ascii=False)]
    return "\n".join(L) + "\n", files, quotable


def render_reasoning(steps: list[dict]) -> str:
    """Pass B's transcript: every assistant turn with its reasoning and visible text in full, tool calls as one line each,
    tool results as one line naming the file with the full text. Empty when the agent had no reasoning."""
    if not any(s["kind"] == "assistant" and s["thinks"] for s in steps):
        return ""
    L = ["## Transcript with the agent's thinking blocks", "",
         "`### T<n> assistant` holds the agent's `(thinking)` blocks, then its output text, then its tool calls in one line "
         "each. Tool results are one line naming `turns/T<n>.md`; the "
         "eval tables above give every score and the turn that showed it.", ""]
    for s in steps:
        k, t = s["kind"], s["turn"]
        if k in ("prompt", "notice"):
            L += render_turn(s, full=False)
        elif k == "assistant":
            L += [f"### T{t} assistant", ""]
            for th in s["thinks"]:
                L += ["(thinking)", th, ""]
            for tx in s["texts"]:
                L += [tx, ""]
            L += [f"CALL {c['name']}({brief(c['input'])})" for c in s["calls"]] + [""]
        elif k == "results":
            L += [f"[T{t} results of {', '.join(str(r['name']) for r in s['results'])} — turns/T{t}.md]", ""]
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------------------------------------------
def fmt(x) -> str:
    return "(no score)" if x is None else f"{x:.4f}"


def comparison_history(rd: F.RunData, it: int, labels: dict, start_sig) -> str:
    R = rd.repo
    close_of: dict[str, int] = {}
    for k in rd.iterations:
        if 0 <= k < it:
            s = R.sig(rd.rec(k).get("close_sha") or "")
            if s and s not in close_of:
                close_of[s] = k
    L = ["## CMP: comparison-set history of the run", "",
         "Every comparison-set (held-out) eval of the run up to the end of this iteration, in time order, with the code it "
         "scored. Comparison scores of different iterations are comparable.", "",
         "| CMP# | iteration | code | rows | score |", "|---|---|---|---|---|"]
    n = 0
    for c in rd.comps:
        ci = int(c.get("iteration") or 0)
        if ci > it:
            continue
        n += 1
        s = R.sig(c["tree"]) if c.get("tree") else None
        if ci == it and s in labels:
            what = f"{labels[s]} of this iteration"
        elif s and s == start_sig:
            what = "the start code (C0)"
        elif s in close_of:
            what = "the seed program (iteration 0)" if close_of[s] == 0 else f"closing code of iteration {close_of[s]}"
        elif s and ci < it:
            what = "a code version of that iteration"
        else:
            what = "other code"
        L.append(f"| {n} | {c.get('iteration')} | {what} | {c.get('n_rows')} | {fmt(c.get('score'))} |")
    if n == 0:
        L.append("| - | (none) | | | |")
    return "\n".join(L) + "\n"


def earlier_iterations(rd: F.RunData, it: int) -> str:
    """PRIOR:n -- the logged scores of every earlier iteration, so memory numbers about earlier iterations are
    verifiable. Scores only: no change
    summaries (they were context for RP, which v3 does not count) and no admission outcome (it would reveal the arm)."""
    L = ["## PRIOR: earlier iterations as the harness logged them", "",
         "One row per earlier iteration (cite PRIOR:<n>). Train scores are on that iteration's own 50-row sample (never "
         "like-for-like across iterations); comparison scores are on the fixed held-out set.", ""]
    if it == 1:
        return "\n".join(L + ["(this is iteration 1: there is no earlier iteration)"]) + "\n"
    L += ["| iter | started from iter | code changed vs its start | constraint checker | train: start-code eval -> child evals "
          "(best, last) | comparison evals (agent / harness) |", "|---|---|---|---|---|---|"]
    for k in rd.iterations:
        if not 1 <= k < it:
            continue
        rk, evk = rd.rec(k), rd.evals_of(k)
        par = [e["score"] for e in evk if e.get("eval_sample") == "train" and e.get("suffix") == "parent" and e.get("score") is not None]
        chd = [e["score"] for e in evk if e.get("eval_sample") == "train" and e.get("suffix") == "child" and e.get("score") is not None]
        tr = (f"{par[0]:.4f}" if par else "n/a") + " -> " + (f"{max(chd):.4f}, {chd[-1]:.4f} (n={len(chd)})" if chd else "no child eval")
        cm = ", ".join(f"{e['score']:.4f}" + (" (harness)" if e.get("by_loop") else "")
                       for e in evk if e.get("eval_sample") == "comparison" and e.get("score") is not None) or "none"
        ch = C.changed(rd, k)
        L.append(f"| {k} | {rk.get('parent_iteration', '?')} | {'unknown' if ch is None else ('yes' if ch else 'no')} | "
                 f"{'reverted the code' if rk.get('constraint_violation') else '-'} | {tr} | {cm} |")
    return "\n".join(L) + "\n"


def evidence_tables(spec: F.RunSpec, fx: dict, cmp_md: str, prior_md: str) -> str:
    """OUTCOME, EVAL, CMP, PRIOR and the memory lines of this session: the tables both passes get inline."""
    t = fx["table"]
    L = ["## OUTCOME: facts the harness recorded for this iteration (not the agent's statements)", "",
         f"- benchmark: {spec.bench}; comparison (held-out) set {F.SET_ROWS[spec.bench]} rows; train samples {F.TRAIN_ROWS} rows",
         f"- start code: the closing code of iteration {fx['parent_iteration']} (iteration 0 = the seed program); code label C0",
         f"- closing code: code label {t['close_label']}; identical to the start code: "
         f"{'unknown' if t['changed'] is None else ('no' if t['changed'] else 'yes')}"]
    if fx["gate"]:
        L += ["- constraint checker: FIRED. The harness found that the change broke a rule of the task and reverted the code "
              "to the start code (attempted.diff holds the change).", f"  - rule text: {fx['rule_text'][:1500]}"]
    else:
        L.append("- constraint checker: did not fire")
    L.append(f"- comparison credit left at the end of the iteration: "
             f"{fx['credits_after'] if fx['credits_after'] is not None else 'not recorded'}")
    L.append("- the start code's comparison scores on record before this session: "
             + ("; ".join(f"{fmt(s)} ({src})" for s, src in fx["ref"]) if fx["ref"] else "none"))
    L.append(f"- noise: one SD of one eval of identical code: train {fx['sd_train']:.4f}, comparison {fx['sd_comparison']:.4f}")
    L += ["", "## EVAL: evals of this iteration", "",
          f"In time order. `code`: C0 = the start code; the same label = identical code. The closing code is "
          f"{t['close_label']}. `sample`: train samples with the same alias have the same rows. `run by`: agent or harness. "
          "`shown at`: the turn whose tool result showed the eval to the agent (- = never shown).", "",
          "| EVAL# | eval set | sample | code | rows | score | run by | shown at |", "|---|---|---|---|---|---|---|---|"]
    for r in t["rows"]:
        cl = " (closing)" if r["sig"] and r["sig"] == t["close_sig"] else ""
        L.append(f"| {r['n']} | {r['set']} | {r['sample']} | {r['code']}{cl} | {r['rows']} | {fmt(r['score'])} | {r['by']} | "
                 f"{'T' + str(r['shown']) if r['shown'] is not None else '-'} |")
    if not t["rows"]:
        L.append("| - | (no evals in this iteration) | | | | | | |")
    L += ["", cmp_md, prior_md, "## Memory lines written in this session (memory_after.md lines not in memory_before.md)", ""]
    L += [f"MEM-AFTER:L{i}: {ln}" for i, ln in fx["new_lines"]] or ["(none)"]
    L += ["", f"memory_before.md has {len(fx['mem_before'].splitlines())} lines and memory_after.md "
              f"{len(fx['mem_after'].splitlines())} lines (files in your working directory).", ""]
    return "\n".join(L) + "\n"


def inline_evidence(tables: str, fx: dict, view: str) -> str:
    """Pass A's evidence: the tables, the UR candidates, the numbers to account for, and the filtered transcript."""
    L = [tables, "## Failed versions found by the program (for UR)", ""]
    if fx["ur_candidates"]:
        L += [f"- {u['eval']}: {u['code']} on {u['set']}" + (f" {u['sample']}" if u["set"] != "comparison" else "")
              + f" scored {fmt(u['score'])}, start code {fmt(u['start_mean'])} ({u['gap_sd']:.1f} SD below), shown at T{u['shown']}"
              for u in fx["ur_candidates"]]
    else:
        L.append("(none: UR is empty)")
    L += ["", "## Numbers to account for (every score-like number in the agent's visible words and new memory lines)", ""]
    L += [f"{n['id']}: {n['where']} {n['number']}" for n in fx["numbers"]] or ["(none)"]
    L += ["", view]
    return "\n".join(L) + "\n"


def build_unit(out: Path, runs: Runs, spec: F.RunSpec, it: int, uid: str) -> dict:
    rd = runs.rd(spec.run)
    fx = X.facts_of(rd, it)
    steps = rd.steps(it)
    view, files, quotable = render_view(steps)
    reasoning = render_reasoning(steps)
    rd.drop_steps(it)
    rec = rd.rec(it)
    diff = C.git_diff(rd, rec["parent_sha"], rec["close_sha"])
    att = C.attempted_diff(rd, it)
    files["memory_before.md"] = C.wrap(C.numbered(fx["mem_before"])) + "\n"
    files["memory_after.md"] = C.wrap(C.numbered(fx["mem_after"])) + "\n"
    files["code_diff.patch"] = (diff or "(empty: the closing code has the same content as the start code)") + "\n"
    if att:
        files["attempted.diff"] = att + "\n"
    ws = out / "ws" / uid
    if ws.exists():
        shutil.rmtree(ws)
    for name, text in files.items():
        p = ws / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
    cmp_md = comparison_history(rd, it, fx["table"]["labels"], fx["table"]["start_sig"])
    tables = evidence_tables(spec, fx, cmp_md, earlier_iterations(rd, it))
    inline = inline_evidence(tables, fx, view)
    for sub, text in (("tables", tables), ("inline", inline), ("reasoning", reasoning)):
        (out / sub).mkdir(parents=True, exist_ok=True)
        path = out / sub / f"{uid}.md"
        if text:
            path.write_text(text, encoding="utf-8")
        elif path.exists():
            path.unlink()
    t = fx["table"]
    meta = {"unit_id": uid, "run": spec.run, "model": spec.model, "arm": spec.arm, "bench": spec.bench, "seed": spec.seed,
            "iteration": it, "workspace_sha256": workspace_sha256(ws),
            "sizes": {"inline": len(inline), "view": len(view), "reasoning": len(reasoning),
                      "workspace": sum(len(v) for v in files.values())},
            "quotable": quotable, "memory_after": fx["mem_after"].split("\n"),
            "numbers": fx["numbers"], "has_reasoning": bool(reasoning),
            "python_modes": fx["python_modes"], "ur_candidates": fx["ur_candidates"],
            "facts": {"changed": t["changed"], "gate": fx["gate"], "credits_after": fx["credits_after"],
                      "close_label": t["close_label"], "n_evals": len(t["rows"])}}
    (out / "meta").mkdir(parents=True, exist_ok=True)
    (out / "meta" / f"{uid}.json").write_text(json.dumps(meta, ensure_ascii=False, default=str), encoding="utf-8")
    return meta


def run_build(args, runs: Runs) -> None:
    units = C.select_units(runs, args.only, args.limit)
    sizes = []
    for i, (spec, it, uid) in enumerate(units, 1):
        m = build_unit(args.out, runs, spec, it, uid)
        sizes.append((m["sizes"]["inline"], len(m["numbers"]), spec.model))
        if i % 25 == 0 or i == len(units):
            C.log(f"[build] {i}/{len(units)} workspaces")
    for model in sorted({s[2] for s in sizes}):
        xs = sorted(s[0] for s in sizes if s[2] == model)
        ns = sorted(s[1] for s in sizes if s[2] == model)
        C.log(f"[build] {model}: inline evidence chars median {xs[len(xs) // 2]:,}, max {xs[-1]:,}; numbers to account "
              f"for median {ns[len(ns) // 2]}, max {ns[-1]}")


# ---------------------------------------------------------------------------------------------------------------
# count
# ---------------------------------------------------------------------------------------------------------------
def prompt_of(out: Path, uid: str, it: int) -> str:
    head = (PROMPT_PATH.read_text(encoding="utf-8")
            .replace("{definitions}", DEFINITIONS_PATH.read_text(encoding="utf-8").strip())
            .replace("{shape}", SHAPE).rstrip())
    inline = (out / "inline" / f"{uid}.md").read_text(encoding="utf-8")
    return (head + f"\n\n# The evidence of iteration {it}\n\nYour working directory holds this iteration's files (turns/, "
            "memory_before.md, memory_after.md, code_diff.patch, and attempted.diff when a change was reverted).\n\n"
            + inline + "\n## Your task now\n\nList every score claim with its verdict, map every N<k>, decide UR and BC, "
            "and end your reply with the one fenced ```json block.\n")


def prompt_of_api(out: Path, uid: str, it: int) -> str:
    """Pass A for a single API call (no tools): the same prompt, with the tool instructions replaced."""
    p = prompt_of(out, uid, it)
    for old, new in (
            ("2. Open a stub file when a verdict depends on it, for example a tool result that may hold a number the agent cites (WR d).\n"
             "   Use read and grep in your working directory as often as you need.",
             "2. There are no files or tools in this run: a stub cannot be opened, so judge from what is shown (a number the "
             "agent cites from a stubbed result is not unsupported unless the shown evidence contradicts it)."),
            ("Your working directory holds this iteration's files (turns/, memory_before.md, memory_after.md, code_diff.patch, "
             "and attempted.diff when a change was reverted).\n\n", "")):
        assert old in p, old[:60]
        p = p.replace(old, new)
    return p


def prompt_of_b(out: Path, uid: str, it: int, a: dict) -> str:
    """Pass B: the definitions (adapted), the first-pass claim list, the tables, and the transcript with the reasoning.
    Pass B was used only to compare whether adding the thinking flipped pass-A verdicts (it did not, materially: see the
    module docstring); the main counts come from pass A alone."""
    defs = DEFINITIONS_PATH.read_text(encoding="utf-8").strip()
    for old, new in (
            ("You decide five modes: MS, WR and FN (through the claim list), UR and BC.",
             "In this re-check you decide only the verdicts (MS, WR, FN) of the claims the first pass listed."),
            ("The agent's private reasoning, if it had any, is not part of the evidence.",
             "In this re-check the agent's thinking blocks are in the transcript (see below how they change the rules)."),
            ):
        assert old in defs, old[:60]
        defs = defs.replace(old, new)
    # pass B has no number list: drop the paragraph that asks for one (it runs to the next blank line)
    defs = re.sub(r"Then account for every number N<k>.*?\n\n", "", defs, count=1, flags=re.S)
    head = (RETHINK_PROMPT_PATH.read_text(encoding="utf-8").replace("{definitions}", defs)
            .replace("{shape}", SHAPE_B).rstrip())
    L = ["## The first pass's claims (visible words only)", ""]
    for c in a["claims"]:
        L.append(f"- {c['id']} [{c['verdict']}{' ' + c['clause'] if c.get('clause') else ''}] "
                 + (f"T{c['turn']} " if c.get("turn") is not None else "") + f"{c['source']}: \"{C.norm(c['quote'])}\""
                 + (f" -- {C.norm(c.get('explanation') or '')}" if c["verdict"] != "ok" and c.get("explanation") else ""))
    if not a["claims"]:
        L.append("(none)")
    tables = (out / "tables" / f"{uid}.md").read_text(encoding="utf-8")
    reasoning = (out / "reasoning" / f"{uid}.md").read_text(encoding="utf-8")
    return (head + f"\n\n# The evidence of iteration {it}\n\n" + "\n".join(L) + "\n\n" + tables + "\n" + reasoning
            + "\n## Your task now\n\nGive every first-pass claim a verdict with the thinking blocks in view, list any "
            "thinking-only mistakes, and end your reply with the one fenced ```json block.\n")


def qn(x: str) -> str:
    """Quote-check normaliser: whitespace collapsed (fm_count.norm) and markdown emphasis marks (* _ `) dropped on both
    sides, so a quote of '*below*' as 'below' matches. Formatting only: no word can be added or changed."""
    return re.sub(r"[*_`]", "", C.norm(x))


def mem_window(lines: list[str], i: int, radius: int = 2) -> str:
    """Lines i-radius..i+radius (0-based) joined: memory sentences are often hard-wrapped over two or three lines."""
    return " ".join(lines[max(0, i - radius): i + radius + 1])


def mem_window_hit(q: str, lines: list[str]) -> int | None:
    """0-based index of the first memory line whose +-2-line window contains the quote, else None."""
    for i in range(len(lines)):
        if q in qn(mem_window(lines, i)):
            return i
    return None


def quote_found(inst: dict, meta: dict, allowed=SOURCES) -> str | None:
    q = qn(inst.get("quote") or "").strip('"“”')
    if not q:
        return "empty quote"
    if len(q) > C.MAX_QUOTE:
        return f"quote longer than {C.MAX_QUOTE} characters"
    src = inst.get("source")
    if src not in allowed:
        return f"source {src!r}"
    if src == "memory":
        return None if mem_window_hit(q, meta["memory_after"]) is not None else "not found in memory_after.md"
    try:
        t = int(str(inst.get("turn")).lstrip("T"))
    except (TypeError, ValueError):
        return f"turn {inst.get('turn')!r} is not a turn number"
    turn_q = meta["quotable"].get(str(t)) or {}
    if any(q in qn(x) for x in turn_q.get(src) or []):
        return None
    other = [s for s in allowed if s not in (src, "memory") and any(q in qn(x) for x in turn_q.get(s) or [])]
    if other:             # right turn, wrong source label (assistant text tagged as a tool call or the reverse): fix it
        inst["source"], inst["source_fixed_from"] = other[0], src
        return None
    near = [k for k, d in meta["quotable"].items() if any(q in qn(x) for x in d.get(src, []))]
    close = [k for k in near if abs(int(k) - t) <= 2]
    if len(close) == 1:   # an off-by-one turn citation: the quote is verbatim, so fix the turn and keep the instance
        inst["turn"], inst["turn_fixed_from"] = int(close[0]), t
        return None
    return f"not found in {src} of T{t}" + (f" (found in T{', T'.join(near[:3])})" if near else "")


def validate(d: dict, meta: dict) -> list[str]:
    """Fatal problems (a fresh session replaces the reply): bad shape, an unknown verdict or source, a duplicate claim id, a
    quote not found verbatim, an FN claim not sourced from memory, a number N<k> missing or mapped to an unknown claim,
    more than one BC. Warnings: BC although the checker did not fire, UR although Python found no failed version."""
    problems, warnings = [], []
    claims = d.get("claims")
    if not isinstance(claims, list) or not isinstance(d.get("numbers"), dict):
        raise ValueError("'claims' must be a list and 'numbers' an object")
    ids = set()
    for i, c in enumerate(claims):
        if not isinstance(c, dict):
            problems.append(f"claims[{i}] is not an object")
            continue
        cid = c.get("id")
        if not cid or cid in ids:
            problems.append(f"claims[{i}] id {cid!r} missing or duplicate")
        ids.add(cid)
        if c.get("verdict") not in VERDICTS:
            problems.append(f"{cid} verdict {c.get('verdict')!r}")
        if c.get("verdict") == "FN" and c.get("source") != "memory":
            problems.append(f"{cid} is FN but not a memory line")
        why = quote_found(c, meta)
        if why:
            problems.append(f"{cid} quote {why}: {C.norm(c.get('quote') or '')[:80]!r}")
    want = {n["id"] for n in meta["numbers"]}
    got = d["numbers"]
    missing = sorted(want - set(got), key=lambda x: int(x[1:]))
    if missing:
        problems.append(f"{len(missing)} numbers not accounted for: {', '.join(missing[:20])}")
    bad = [k for k, v in got.items() if k in want and v != "-" and v not in ids]
    if bad:
        problems.append(f"numbers mapped to unknown claims: {', '.join(bad[:10])}")
    # A number in a memory line of this session belongs to that memory line's OWN claim (source memory,
    # quote taken from that line), never to a transcript claim it repeats: FN judges what the memory line itself says.
    by_id = {c.get("id"): c for c in claims if isinstance(c, dict)}
    for n in meta["numbers"]:
        if not n["where"].startswith("MEM-AFTER:L"):
            continue
        cid = got.get(n["id"])
        if cid in (None, "-") or cid not in by_id:
            continue
        c = by_id[cid]
        window = mem_window(meta["memory_after"], int(n["where"].split(":L")[1]) - 1)
        if c.get("source") != "memory" or qn(c.get("quote") or "").strip('"“”') not in qn(window):
            problems.append(f"{n['id']} ({n['where']}) is mapped to {cid}, which is not that memory line's own claim")
    for m in ("UR", "BC"):
        lst = d.get(m)
        if not isinstance(lst, list):
            problems.append(f"{m} missing or not a list")
            continue
        for i, inst in enumerate(lst):
            why = quote_found(inst, meta) if isinstance(inst, dict) else "not an object"
            if why:
                problems.append(f"{m}[{i}] quote {why}")
    if isinstance(d.get("BC"), list) and len(d["BC"]) > 1:
        problems.append(f"BC has {len(d['BC'])} instances (at most 1)")
    if problems:
        raise ValueError("invalid count: " + "; ".join(problems))
    if d.get("BC") and not meta["facts"]["gate"]:
        warnings.append("BC although the constraint checker did not fire")
    if d.get("UR") and not meta["ur_candidates"]:
        warnings.append("UR although the program found no failed version")
    fixed = [x for x in list(claims) + list(d.get("UR") or []) + list(d.get("BC") or [])
             if isinstance(x, dict) and ("turn_fixed_from" in x or "source_fixed_from" in x)]
    if fixed:
        warnings.append(f"{len(fixed)} quote citation(s) fixed (turn within +-2 or source label of the same turn)")
    return warnings


def has_thinking(meta: dict, turn) -> bool:
    try:
        return bool((meta["quotable"].get(str(int(str(turn).lstrip("T")))) or {}).get("thinking"))
    except (TypeError, ValueError):
        return False


def validate_b(d: dict, meta: dict, a: dict) -> list[str]:
    """Fatal: a first-pass claim without a review, an unknown verdict, a changed verdict without an explanation or citing a
    turn that has no thinking block, a thinking-only item citing such a turn. (Pass B never quotes the thinking: the
    provider refuses requests to reproduce a model's reasoning verbatim, so changes are cited by turn and paraphrased.)"""
    problems = []
    rv = d.get("reviews")
    if not isinstance(rv, dict):
        raise ValueError("'reviews' must be an object")
    for c in a["claims"]:
        r = rv.get(c["id"])
        if not isinstance(r, dict):
            problems.append(f"{c['id']} has no review")
            continue
        if r.get("verdict") not in VERDICTS:
            problems.append(f"{c['id']} verdict {r.get('verdict')!r}")
        elif r["verdict"] != c["verdict"]:
            if not has_thinking(meta, r.get("turn")) or not C.norm(r.get("explanation") or ""):
                problems.append(f"{c['id']} changed {c['verdict']}->{r['verdict']} without an explanation and a turn "
                                f"that has a thinking block (turn {r.get('turn')!r})")
    for i, x in enumerate(d.get("thinking_only") or []):
        if not isinstance(x, dict) or not has_thinking(meta, x.get("turn")):
            problems.append(f"thinking_only[{i}] cites no turn with a thinking block")
    if problems:
        raise ValueError("invalid re-check: " + "; ".join(problems))
    return []


def run_api_session(prompt: str, *, validate, required_key: str, label: str, model: str, retries: int,
                    timeout_seconds: int, rejected_dir: Path) -> dict:
    """ONE plain API call (no tools, no agent harness), retried like judge.run_json_session. Used for pass B with Claude,
    and for both passes with an OpenAI judge (--api-judge, e.g. gpt-6-sol: chat completions).
    WHY not OpenCode: inside the OpenCode harness every pass-B request was refused by the provider's content filter
    (stop_reason refusal), while the identical prompt as a plain API call completes; pass B needs no tools (its evidence is
    inline). Same model as pass A (the provider/model prefix is dropped for the API)."""
    api_model = model.split("/", 1)[-1]
    openai = api_model.startswith("gpt")
    if openai:
        url, headers = "https://api.openai.com/v1/chat/completions", {
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "content-type": "application/json"}
        body = json.dumps({"model": api_model, "max_completion_tokens": 64000,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    else:
        url, headers = "https://api.anthropic.com/v1/messages", {
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01",
            "content-type": "application/json"}
        body = json.dumps({"model": api_model, "max_tokens": 32000,
                           "messages": [{"role": "user", "content": prompt}]}).encode("utf-8")
    attempts = []
    for attempt in range(1, retries + 1):
        t0 = time.time()
        req = urllib.request.Request(url, data=body, headers=headers)
        problem, r = None, {}
        try:
            r = json.load(urllib.request.urlopen(req, timeout=timeout_seconds))
        except urllib.error.HTTPError as e:
            problem = f"HTTP {e.code}: {e.read()[:300]!r}"
        except (urllib.error.URLError, TimeoutError) as e:
            problem = f"request failed: {e}"
        if openai:
            ch = (r.get("choices") or [{}])[0]
            text, stop = (ch.get("message") or {}).get("content") or "", ch.get("finish_reason")
            ok_stop = stop == "stop"
        else:
            text = "".join(c.get("text", "") for c in r.get("content", []) if c.get("type") == "text")
            stop = r.get("stop_reason")
            ok_stop = stop in ("end_turn", "stop_sequence")
        parsed = None
        if problem is None and stop in ("refusal", "content_filter"):
            # never retry a provider refusal: a retry only adds filter hits on the account
            raise RuntimeError(f"{label}: provider refused ({stop}); not retried")
        if problem is None:
            if not ok_stop:
                problem = f"stop_reason {stop}"
            else:
                from judge import parse_fenced_json
                parsed = parse_fenced_json(text, required_key)
                if parsed is None:
                    problem = "no fenced JSON block in the reply"
                else:
                    try:
                        warnings = validate(parsed)
                    except ValueError as e:
                        problem = str(e)
        elapsed = time.time() - t0
        if problem is None:
            return {"harness": "messages-api", "model": model,
                    "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                    "request_id": r.get("id"), "elapsed_seconds": round(elapsed), "usage": r.get("usage") or {},
                    "tool_calls": [], "n_tool_calls": 0, "retries": attempt - 1, "attempt_errors": attempts,
                    "warnings": warnings, "output": text, "result": parsed}
        attempts.append(f"attempt {attempt}: {problem}")
        rejected_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
        (rejected_dir / f"{label.replace(' ', '_')}.{stamp}.attempt{attempt}.json").write_text(json.dumps(
            {"label": label, "attempt": attempt, "problem": problem, "usage": r.get("usage"), "output": text}, indent=1),
            encoding="utf-8")
        C.log(f"[{label}] {attempts[-1]} ({elapsed:.0f}s)" + ("; retrying" if attempt < retries else ""))
    raise RuntimeError(f"session failed for {label} after {retries} attempts: " + " | ".join(attempts))


def run_count(args, runs: Runs) -> None:
    units = C.select_units(runs, args.only, args.limit)
    is_b = args.pass_ == "B"
    judge = args.api_judge
    cdir = pass_dir(args.out, args.pass_, judge)
    cdir.mkdir(parents=True, exist_ok=True)
    todo, sizes, skipped = [], [], 0
    for spec, it, uid in units:
        mpath = args.out / "meta" / f"{uid}.json"
        if not mpath.exists():
            raise SystemExit(f"no workspace for {uid} ({spec.run}:{it}): run `fm_v3.py build` first")
        meta = json.loads(mpath.read_text(encoding="utf-8"))
        ws = args.out / "ws" / uid
        if workspace_sha256(ws) != meta["workspace_sha256"]:
            raise SystemExit(f"{uid}: workspace changed since build; rebuild it")
        a = None
        if is_b:
            if not meta["has_reasoning"]:
                skipped += 1
                continue
            ap = pass_dir(args.out, "A", judge) / f"{uid}.json"
            if not ap.exists():
                raise SystemExit(f"{uid}: no pass-A result in {ap.parent.name}; run `count --pass A` first")
            a = json.loads(ap.read_text(encoding="utf-8"))["result"]
            prompt = prompt_of_b(args.out, uid, it, a)
        else:
            prompt = prompt_of_api(args.out, uid, it) if judge else prompt_of(args.out, uid, it)
        nb = len(prompt.encode("utf-8"))
        if nb > MAX_PROMPT_BYTES and not (judge or is_b):
            raise SystemExit(f"{uid}: prompt is {nb:,} bytes (> {MAX_PROMPT_BYTES:,}); too big for the command line")
        sizes.append(nb)
        want = {"model": judge or args.model, "prompt_sha256": C.sha256_text(prompt), "workspace_sha256": meta["workspace_sha256"]}
        if C._cached(cdir / f"{uid}.json", want, args.force) is not None:
            continue
        todo.append((uid, ws, meta, prompt, want, a))
    if not sizes:
        C.log(f"[count] pass {args.pass_}: nothing to do ({skipped} without reasoning)")
        return
    sizes.sort()
    C.log(f"[count] pass {args.pass_}: {len(todo)} of {len(units) - skipped} iterations to count; prompt bytes median {sizes[len(sizes) // 2]:,}, "
          f"max {sizes[-1]:,} (~{sum(sizes) // 4:,} prompt tokens in all, before tool reads)")
    if todo:
        (args.out / "prompts").mkdir(parents=True, exist_ok=True)
        (args.out / "prompts" / f"{todo[0][0]}.md").write_text(todo[0][3], encoding="utf-8")
    if args.dry_run or not todo:
        return
    if judge:
        C.load_env_key(C.REPO_ROOT, HERE.parent / "macro-counts")
    else:
        C._session_env(args)

    def _one(uid, ws, meta, prompt, want, a):
        jpath = cdir / f"{uid}.json"
        if is_b or judge:
            check = (lambda d: validate_b(d, meta, a)) if is_b else (lambda d: validate(d, meta))
            jd = run_api_session(prompt, validate=check, required_key="reviews" if is_b else "claims",
                                 label=f"{args.pass_} {uid}{'@' + judge if judge else ''}", model=judge or args.model,
                                 retries=args.retries, timeout_seconds=args.timeout, rejected_dir=args.out / "rejected")
            jd = {"unit_id": uid, "run": meta["run"], "iteration": meta["iteration"], **want, **jd}
            jpath.write_text(json.dumps(jd, indent=1, ensure_ascii=False), encoding="utf-8")
            return jd
        check = lambda d: validate(d, meta)  # noqa: E731
        jd = run_json_session(ws, prompt, required_key="claims", label=f"A {uid}",
                              validate=check, model=args.model, max_turns=args.max_turns,
                              timeout_seconds=args.timeout, retries=args.retries, allow_bash=args.allow_bash,
                              transcript_path=jpath.with_suffix(".transcript.json"),
                              rejected_dir=args.out / "rejected", log=C.log)
        jd = {"unit_id": uid, "run": meta["run"], "iteration": meta["iteration"], "allow_bash": args.allow_bash,
              **want, **jd}
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
                r = jd["result"]
                if is_b:
                    ch = [f"{x['verdict']}->{r['reviews'][x['id']]['verdict']}" for x in a_of(args.out, uid, judge)["claims"]
                          if r["reviews"][x["id"]]["verdict"] != x["verdict"]]
                    C.log(f"[count B] {done}/{len(todo)} {uid}: changed {len(ch)} {' '.join(ch)} thinking_only="
                          f"{len(r.get('thinking_only') or [])} (tool_calls={jd['n_tool_calls']} retries={jd['retries']} "
                          f"{jd['elapsed_seconds']}s ${c:.2f}; so far ${spent:.2f})")
                    continue
                hits = " ".join(f"{v}{sum(1 for x in r['claims'] if x['verdict'] == v)}" for v in ("MS", "WR", "FN", "corrected")
                                if any(x["verdict"] == v for x in r["claims"]))
                hits += f" UR{len(r['UR'])}" if r["UR"] else ""
                hits += f" BC{len(r['BC'])}" if r["BC"] else ""
                C.log(f"[count] {done}/{len(todo)} {uid}: claims={len(r['claims'])} {hits or '-'} "
                      f"(tool_calls={jd['n_tool_calls']} retries={jd['retries']} {jd['elapsed_seconds']}s ${c:.2f}; "
                      f"so far ${spent:.2f})")
            except (JudgeError, RuntimeError, ValueError) as e:
                failed.append(uid)
                C.log(f"[count] {uid}: FAILED: {e}")
    if failed:
        raise SystemExit(f"{len(failed)} iteration(s) failed: {','.join(failed)} (re-run `count`; finished ones are cached)")


# ---------------------------------------------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------------------------------------------
def a_of(out: Path, uid: str, judge: str | None = None) -> dict:
    return json.loads((pass_dir(out, "A", judge) / f"{uid}.json").read_text(encoding="utf-8"))["result"]


def unit_counts(meta: dict, r: dict) -> dict:
    u = {m: sum(1 for c in r["claims"] if c["verdict"] == m) for m in ("MS", "WR", "FN")}
    u["UR"], u["BC"] = len(r["UR"]), len(r["BC"])
    for m in PY_MODES:
        u[m] = 1 if meta["python_modes"].get(m) else 0
    u["corrected"] = sum(1 for c in r["claims"] if c["verdict"] == "corrected")
    u["claims"] = len(r["claims"])
    return u


def mode_table(units: list[dict], cells: list) -> list[str]:
    md = ["| Failure mode | " + " | ".join(f"{m} {a}" for m, a in cells) + " |", "|---" * (len(cells) + 1) + "|"]
    for code in ROWS:
        row = []
        for c in cells:
            us = [u for u in units if (u["model"], u["arm"]) == c]
            e, r = sum(C.eff(u, code) for u in us), sum(C.raw(u, code) for u in us)
            row.append(f"{e} of {len(us)} ({100 * e / max(len(us), 1):.0f}%) [{r}]")
        md.append(f"| {ROW_NAME.get(code, NAMES.get(code))} | " + " | ".join(row) + " |")
    md.append("| *claims the agent corrected itself (not counted)* | "
              + " | ".join(str(sum(u["corrected"] for u in units if (u["model"], u["arm"]) == c)) for c in cells) + " |")
    return md


def table_csv(units: list[dict], path: Path) -> None:
    """TABLE.csv: one row per mode x model x arm. Counts and pooled % over iterations, plus the mean and the SAMPLE standard
    deviation (n-1) of the per-run % across the runs of the cell (6 runs: 2 benchmarks x 3 seeds), the level at which the
    permutation tests are run. `uncaught` = the mistake got past the harness; `all` = every iteration with the mistake."""
    import statistics
    rows = []
    for code in ROWS:
        for m, a in C.CELLS:
            us = [u for u in units if (u["model"], u["arm"]) == (m, a)]
            if not us:
                continue
            runs = sorted({(u["bench"], u["seed"]) for u in us})
            row = {"mode": code, "mode_name": ROW_NAME.get(code, NAMES.get(code, code)).strip("*"), "model": m, "arm": a,
                   "iterations": len(us), "runs": len(runs)}
            for lab, f in (("uncaught", C.eff), ("all", C.raw)):
                n = sum(f(u, code) for u in us)
                shares = [100 * sum(f(u, code) for u in rr) / len(rr) for b, sd in runs
                          for rr in [[u for u in us if (u["bench"], u["seed"]) == (b, sd)]]]
                row[f"{lab}_count"] = n
                row[f"{lab}_pct"] = round(100 * n / len(us), 1)
                row[f"{lab}_run_mean_pct"] = round(statistics.mean(shares), 1)
                row[f"{lab}_run_sd_pct"] = round(statistics.stdev(shares), 1) if len(shares) > 1 else ""
            rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def tests_csv(units: list[dict], path: Path) -> None:
    """TESTS.csv: the run-level permutation tests, one row per comparison x mode x scope."""
    rows = []
    for lab, c1, c2 in [("Qwen3: freeform vs pareto", C.CELLS[0], C.CELLS[1]),
                        ("GLM-5.2: freeform vs pareto", C.CELLS[2], C.CELLS[3]),
                        ("Qwen3 vs GLM-5.2, freeform", C.CELLS[0], C.CELLS[2]),
                        ("Qwen3 vs GLM-5.2, pareto", C.CELLS[1], C.CELLS[3])]:
        for code in ("MS or WR", "ANY", "UR"):
            for scope, f in (("uncaught", C.eff), ("all", C.raw)):
                x, y = C.run_shares(units, *c1, code, f), C.run_shares(units, *c2, code, f)
                rows.append({"comparison": lab, "mode": code, "scope": scope,
                             "group1": f"{c1[0]} {c1[1]}", "group1_run_mean_pct": round(100 * sum(x) / len(x), 1),
                             "group2": f"{c2[0]} {c2[1]}", "group2_run_mean_pct": round(100 * sum(y) / len(y), 1),
                             "p_permutation": round(C.perm(x, y), 4)})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def tests_table(units: list[dict]) -> list[str]:
    md = ["| Comparison | " + " | ".join(f"{m}, uncaught" for m in ("MS or WR", "ANY", "UR")) + " |",
          "|---|---|---|---|"]
    for lab, c1, c2 in [("Qwen3: freeform vs pareto", C.CELLS[0], C.CELLS[1]),
                        ("GLM-5.2: freeform vs pareto", C.CELLS[2], C.CELLS[3]),
                        ("Qwen3 vs GLM-5.2, freeform", C.CELLS[0], C.CELLS[2]),
                        ("Qwen3 vs GLM-5.2, pareto", C.CELLS[1], C.CELLS[3])]:
        row = []
        for code in ("MS or WR", "ANY", "UR"):
            x, y = C.run_shares(units, *c1, code, C.eff), C.run_shares(units, *c2, code, C.eff)
            row.append(f"{100 * sum(x) / 6:.0f}% vs {100 * sum(y) / 6:.0f}%, p = {C.fmt_p(C.perm(x, y))}")
        md.append(f"| {lab} | " + " | ".join(row) + " |")
    return md


def rethink_section(args, units: list[dict], complete: bool) -> list[str]:
    """Pass B: the same table with the verdicts re-judged with the reasoning in view (GLM; Qwen3 has no reasoning, so its
    pass-A counts stand), the A -> B verdict changes, and the thinking-only mistakes."""
    units_b, changes, thinking_only, n_b = [], {}, {}, 0
    for u in units:
        bp = args.out / PASS_DIR["B"] / f"{u['unit_id']}.json"
        meta = json.loads((args.out / "meta" / f"{u['unit_id']}.json").read_text(encoding="utf-8"))
        if not meta["has_reasoning"]:
            units_b.append(u)
            continue
        if not bp.exists():
            continue
        a = a_of(args.out, u["unit_id"])
        bj = json.loads(bp.read_text(encoding="utf-8"))
        if bj["prompt_sha256"] != C.sha256_text(prompt_of_b(args.out, u["unit_id"], u["iteration"], a)):
            continue          # made against an older pass-A result: not comparable
        b = bj["result"]
        validate_b(b, meta, a)
        n_b += 1
        cell = (u["model"], u["arm"])
        vb = {c["id"]: b["reviews"][c["id"]]["verdict"] for c in a["claims"]}
        for c in a["claims"]:
            if vb[c["id"]] != c["verdict"]:
                changes.setdefault(cell, {}).setdefault(f"{c['verdict']} -> {vb[c['id']]}", 0)
                changes[cell][f"{c['verdict']} -> {vb[c['id']]}"] += 1
        thinking_only[cell] = thinking_only.get(cell, 0) + len(b.get("thinking_only") or [])
        ub = dict(u)
        for m in ("MS", "WR", "FN"):
            ub[m] = sum(1 for v in vb.values() if v == m)
        ub["corrected"] = sum(1 for v in vb.values() if v == "corrected")
        units_b.append(ub)
    n_reasoning = sum(1 for u in units if (u["model"], u["arm"]) in (C.CELLS[2], C.CELLS[3]))
    if not n_b or n_b < 0.9 * n_reasoning:   # a partial re-check (e.g. 2 iterations) is not a table; see judged_items.csv
        return ["", f"*Pass B (re-check with the reasoning) is not reported here: {n_b} of {n_reasoning} GLM-5.2 iterations "
                    "re-checked with this judge. The thinking comparison was run with gpt-6-sol on 40 GLM-5.2 iterations "
                    "(judged_items.csv, judge gpt-6-sol).*"]
    cells = [c for c in C.CELLS if any((u["model"], u["arm"]) == c for u in units_b)]
    md = ["", "## With the reasoning in view (pass B)", "",
          f"{n_b} iterations with reasoning re-judged (GLM-5.2); Qwen3 iterations have no reasoning, so their pass-A counts "
          "stand. Instances are still claims in the visible words.", ""] + mode_table(units_b, cells)
    md += ["", "**Verdict changes A -> B** (first-pass claims whose verdict the reasoning changed):", "",
           "| cell | changes | thinking-only mistakes (not counted) |", "|---|---|---|"]
    for c in cells:
        if c[0] == "Qwen3":
            continue
        ch = changes.get(c, {})
        md.append(f"| {c[0]} {c[1]} | " + (", ".join(f"{k}: {v}" for k, v in sorted(ch.items())) or "none")
                  + f" | {thinking_only.get(c, 0)} |")
    if complete and len(units_b) == len(units):
        md += ["", "**Run-level tests with pass-B verdicts:**", ""] + tests_table(units_b)
    return md


def run_report(args, runs: Runs) -> None:
    units, insts, claims_rows, missing = [], [], [], []
    for spec, it, uid in runs.units():
        p = args.out / "count" / f"{uid}.json"
        if not p.exists():
            missing.append(uid)
            continue
        jd = json.loads(p.read_text(encoding="utf-8"))
        meta = json.loads((args.out / "meta" / f"{uid}.json").read_text(encoding="utf-8"))
        assert jd["workspace_sha256"] == meta["workspace_sha256"], uid
        r = jd["result"]
        validate(r, meta)     # re-check every quote and the number ledger against the raw turns
        rd = runs.rd(spec.run)
        u = {"unit_id": uid, "run": spec.run, "model": spec.model, "arm": spec.arm, "bench": spec.bench, "seed": spec.seed,
             "iteration": it, "route": C.route(rd, spec, it), **unit_counts(meta, r), "retries": jd["retries"],
             "n_tool_calls": jd["n_tool_calls"], "usd": usd(jd.get("usage")),
             "review_flags": " | ".join(map(C.norm, r.get("review_flags") or []))}
        units.append(u)
        base = {k: u[k] for k in ("unit_id", "model", "arm", "bench", "seed", "iteration", "route")}
        for c in r["claims"]:
            claims_rows.append({**base, "id": c.get("id"), "verdict": c["verdict"], "clause": c.get("clause", ""),
                                "turn": c.get("turn"), "source": c["source"], "quote": C.norm(c["quote"]),
                                "explanation": C.norm(c.get("explanation") or "")})
            if c["verdict"] in ("MS", "WR"):
                insts.append({**base, "mode": c["verdict"], "turn": c.get("turn"), "source": c["source"],
                              "quote": C.norm(c["quote"]), "explanation": C.norm(c.get("explanation") or "")})
        for m in ("UR", "BC"):
            for x in r[m]:
                insts.append({**base, "mode": m, "turn": x.get("turn"), "source": x["source"], "quote": C.norm(x["quote"]),
                              "explanation": C.norm(x.get("explanation") or "")})
        for m in PY_MODES:
            if meta["python_modes"].get(m):
                insts.append({**base, "mode": m, "turn": meta["python_modes"][m].get("turn"), "source": "python",
                              "quote": "", "explanation": json.dumps(meta["python_modes"][m])})
    if missing and not args.partial:
        raise SystemExit(f"{len(missing)} of 351 iterations not counted yet (e.g. {missing[:5]}); run `count` or pass --partial")
    res = args.results
    res.mkdir(parents=True, exist_ok=True)
    for name, rows in (("units.csv", units), ("instances.csv", insts), ("claims.csv", claims_rows)):
        with open(res / name, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, list(rows[0]) if rows else ["unit_id"], lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
    cells = [c for c in C.CELLS if any((u["model"], u["arm"]) == c for u in units)]
    md = ["# Failure modes of Qwen3 and GLM-5.2, with and without pareto evolution (fm-v3)", "",
          f"*Generated by `scripts/failure-modes/fm_v3.py report` ({len(units)} iterations"
          + (f", {len(missing)} NOT YET COUNTED" if missing else "") + f", judge {args.model}). Do not edit by hand.*", "",
          "**Main number:** iterations where the model made the mistake and the harness did not stop it. **[Brackets]:** every "
          "iteration with the mistake. KR, UL, BO decided by Python; MS, WR, UR, BC by the judge from the visible "
          "words (pass A).", "", "*False number written to memory (FN) is not reported: it overlaps with MS / WR.*", ""] + mode_table(units, cells)
    if not missing:
        md += ["", "**Run-level tests** (mean of the 6 run shares; exact permutation test, 6 runs vs 6, two-sided):", ""]
        md += tests_table(units)
    md += rethink_section(args, units, not missing)
    md += ["", f"Cost: ${sum(u['usd'] for u in units):.2f} for accepted sessions; "
               f"{sum(1 for u in units if u['retries'])} iterations needed a fresh session."]
    (res / "TABLE.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    table_csv(units, res / "TABLE.csv")
    if not missing:
        tests_csv(units, res / "TESTS.csv")
    C.log(f"[report] wrote {res / 'TABLE.md'} ({len(units)} iterations, {len(insts)} instances)")
    if args.f2:
        compare_f2(units, args.f2, res)


def compare_f2(units: list[dict], f2: Path, res: Path) -> None:
    """COMPARE_F2.md: per mode and model x arm, iterations flagged by fm-v2 and fm-v3, both, only one; and the recall of
    fm-v2's flagged iterations (unit level). Qwen3 rows are the like-for-like check (no reasoning to add)."""
    old = {r["unit_id"]: r for r in csv.DictReader(open(f2 / "units.csv", encoding="utf-8"))}
    us = [u for u in units if u["unit_id"] in old]
    md = ["# fm-v3 against fm-v2 (F2), iterations with at least one instance", "",
          f"{len(us)} iterations in both.", "",
          "| Mode | cell | F2 | v3 | both | F2 only | v3 only | recall of F2 |", "|---|---|---|---|---|---|---|---|"]
    rows = []
    for m in MODES:
        for c in C.CELLS:
            cu = [u for u in us if (u["model"], u["arm"]) == c]
            a = {u["unit_id"] for u in cu if int(old[u["unit_id"]].get(m) or 0) > 0}
            b = {u["unit_id"] for u in cu if u[m] > 0}
            md.append(f"| {m} | {c[0]} {c[1]} | {len(a)} | {len(b)} | {len(a & b)} | {len(a - b)} | {len(b - a)} | "
                      + (f"{100 * len(a & b) / len(a):.0f}%" if a else "-") + " |")
            rows += [{"unit_id": x, "mode": m, "f2": x in a, "v3": x in b} for x in sorted(a ^ b)]
    (res / "COMPARE_F2.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    with open(res / "compare_f2_disagreements.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, ["unit_id", "mode", "f2", "v3"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    C.log(f"[report] wrote {res / 'COMPARE_F2.md'} ({len(rows)} unit-mode disagreements)")


CLAUSE_TEXT = {
    "MS-1": "direction word contradicts the two numbers it names",
    "MS-2": "stated difference does not match the gap between the numbers",
    "WR-a": "train score compared with a comparison-set score",
    "WR-b": "train scores from different samples compared",
    "WR-c": "number called baseline/start/parent/best is not that program's logged score",
    "WR-d": "number presented as a measured score matches nothing the agent had seen",
    "WR-e": "score of one code version attributed to a different version",
    "FN": "memory line records a score or comparison the logs show is false or mislabelled",
    "UR": "a version measured > 1 SD worse than the start code was dropped and never recorded as failed in memory",
    "BC": "change broke a stated task rule; the constraint checker reverted it",
}
DEF_TEXT = {"MS": "A score claim states a direction or difference that its own numbers contradict.",
            "WR": "A comparison or decision rests on two numbers that are not like-for-like, or on a number that is not what "
                  "the agent says it is, and the agent does not say so.",
            "FN": "A memory line written this iteration records a score or comparison that this iteration's logs show is "
                  "false or mislabelled.",
            "UR": "The agent measured a change as worse than the start code, dropped it, and did not record in memory that it "
                  "failed.",
            "BC": "The agent's code change broke one of the task's stated rules and the constraint checker reverted it.",
            "KR": "The agent scored its closing code on the comparison set, it was > 1 SD below the start code, and the "
                  "agent kept it anyway.",
            "UL": "The agent ended on changed code it never scored on the comparison set, although it had credit left.",
            "BO": "The agent overwrote its whole memory file without reading it, and at least half the old notes are gone."}


def clause_text(clause: str, mode: str) -> str:
    c = (clause or "").strip()
    key = c[:4] if c[:2] in ("MS", "WR") and len(c) >= 4 else mode
    return CLAUSE_TEXT.get(key, CLAUSE_TEXT.get(mode, "")) + (f" ({c})" if c and key == mode else "")


def python_why(mode: str, f: dict) -> str:
    if mode == "KR":
        return (f"Closing code's comparison mean {f['close_mean']:.4f} vs start code's {f['start_mean']:.4f}: "
                f"{f['gap_sd']:.1f} SD below ({', '.join(f['evals'])}); the agent closed on it.")
    if mode == "UL":
        return (f"Closing code {f['close_label']} changed from the start code but was never scored on the comparison set; "
                f"{f['credits_after']} comparison credit(s) left at close.")
    return (f"Full overwrite of memory.md at T{f['turn']} with no earlier read; {f['gone']} of {f['old_lines']} substantive "
            "old lines gone.")


def run_export(args, runs: Runs) -> None:
    """judged_items.csv: one row per LLM-judged item of every result on disk (all judges): each pass-A claim (with the
    pass-B verdict of the same judge when it exists), each UR and BC instance, each thinking-only note. `current` says
    whether the result was made with the prompt the code builds today (False = an older prompt version)."""
    specs = {uid: (spec, it) for spec, it, uid in runs.units()}
    rows = []
    for adir in sorted(p for p in args.out.iterdir() if p.is_dir() and p.name.split("_")[0] == "count"):
        judge = adir.name[len("count_"):] or None
        for f in sorted(adir.glob("U-*.json")):
            if f.name.endswith(".transcript.json"):
                continue
            jd = json.loads(f.read_text(encoding="utf-8"))
            uid = jd["unit_id"]
            spec, it = specs[uid]
            meta = json.loads((args.out / "meta" / f"{uid}.json").read_text(encoding="utf-8"))
            prompt = prompt_of_api(args.out, uid, it) if judge else prompt_of(args.out, uid, it)
            a = jd["result"]
            bf = pass_dir(args.out, "B", judge) / f.name
            b = json.loads(bf.read_text(encoding="utf-8")) if bf.exists() else None
            b_current = None
            if b:
                b_current = b["prompt_sha256"] == C.sha256_text(prompt_of_b(args.out, uid, it, a))
            nums: dict[str, list[str]] = {}
            for n in meta["numbers"]:
                cid = a["numbers"].get(n["id"])
                if cid and cid != "-":
                    nums.setdefault(cid, []).append(f"{n['id']}={n['where']}:{n['number']}")
            base = {"judge": judge or jd.get("model"), "unit_id": uid, "model": spec.model, "arm": spec.arm,
                    "bench": spec.bench, "seed": spec.seed, "iteration": it,
                    "a_current": jd["prompt_sha256"] == C.sha256_text(prompt), "b_current": b_current}

            def row(kind, x, verdict, **extra):
                rows.append({**base, "item": kind, "id": x.get("id", ""), "verdict_a": verdict, "clause": x.get("clause", ""),
                             "turn": x.get("turn"), "source": x.get("source", ""), "quote": C.norm(x.get("quote") or ""),
                             "explanation_a": C.norm(x.get("explanation") or ""), "cites": " ".join(map(str, x.get("cites") or [])),
                             **{"numbers": "", "verdict_b": "", "changed_b": "", "turn_b": "", "explanation_b": ""}, **extra})

            for c in a["claims"]:
                extra = {"numbers": " ".join(nums.get(c["id"], []))}
                if b:
                    rv = b["result"]["reviews"].get(c["id"], {})
                    extra.update(verdict_b=rv.get("verdict", ""), changed_b=rv.get("verdict") != c["verdict"],
                                 turn_b=rv.get("turn") if rv.get("verdict") != c["verdict"] else "",
                                 explanation_b=C.norm(rv.get("explanation") or ""))
                row("claim", c, c["verdict"], **extra)
            for m in ("UR", "BC"):
                for i, x in enumerate(a[m]):
                    row(m, dict(x, id=f"{m}{i + 1}"), m)
            for i, x in enumerate((b or {}).get("result", {}).get("thinking_only") or []):
                rows.append({**base, "item": "thinking_only", "id": f"T{i + 1}", "verdict_a": "", "clause": "",
                             "turn": x.get("turn"), "source": "thinking", "quote": "", "explanation_a": "", "cites": "",
                             "numbers": "", "verdict_b": "not counted", "changed_b": "", "turn_b": x.get("turn"),
                             "explanation_b": C.norm(x.get("explanation") or "")})
            for flag in a.get("review_flags") or []:
                rows.append({**base, "item": "review_flag_a", "id": "", "verdict_a": "", "clause": "", "turn": "",
                             "source": "", "quote": "", "explanation_a": C.norm(flag), "cites": "", "numbers": "",
                             "verdict_b": "", "changed_b": "", "turn_b": "", "explanation_b": ""})
    args.results.mkdir(parents=True, exist_ok=True)
    path = args.results / "judged_items.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(rows[0]) if rows else ["unit_id"], lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    C.log(f"[export] wrote {path} ({len(rows)} rows: every judged claim, ok ones included)")

    # counted_instances.csv: ONLY the counts, one row each, with the mode spelled out and the reason
    counted = []
    for r in rows:
        if not r["a_current"] or r["item"] not in ("claim", "UR", "BC"):
            continue
        mode = r["verdict_a"]
        if mode not in ("MS", "WR", "UR", "BC"):     # FN is judged but not reported (overlaps with MS / WR)
            continue
        overturned = r["item"] == "claim" and r["changed_b"] is True
        counted.append({
            "mode": mode, "mode_name": NAMES[mode], "decided_by": r["judge"],
            "counted": "no: overturned by pass B (thinking)" if overturned else "yes",
            "model": r["model"], "arm": r["arm"], "bench": r["bench"], "seed": r["seed"], "iteration": r["iteration"],
            "unit_id": r["unit_id"], "turn": r["turn"], "agent_words": r["quote"],
            "why_counted": r["explanation_a"], "clause": clause_text(r["clause"], mode), "evidence": r["cites"],
            "pass_b_with_thinking": (f"{r['verdict_b']}: {r['explanation_b']}" if overturned else
                                     ("unchanged" if r["verdict_b"] else "not run")),
            "definition": DEF_TEXT[mode]})
    for spec, it, uid in runs.units():
        mp = args.out / "meta" / f"{uid}.json"
        if not mp.exists():
            continue
        meta = json.loads(mp.read_text(encoding="utf-8"))
        for m in PY_MODES:
            f = meta["python_modes"].get(m)
            if f:
                counted.append({"mode": m, "mode_name": NAMES[m], "decided_by": "python (harness logs)", "counted": "yes",
                                "model": spec.model, "arm": spec.arm, "bench": spec.bench, "seed": spec.seed,
                                "iteration": it, "unit_id": uid, "turn": f.get("turn", ""), "agent_words": "",
                                "why_counted": python_why(m, f), "clause": "", "evidence": "harness logs",
                                "pass_b_with_thinking": "n/a", "definition": DEF_TEXT[m]})
    order = {m: i for i, m in enumerate(("MS", "WR", "UR", "BC", "KR", "UL", "BO"))}
    counted.sort(key=lambda x: (order[x["mode"]], x["decided_by"], x["model"], x["arm"], x["unit_id"]))
    cpath = args.results / "counted_instances.csv"
    with open(cpath, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, list(counted[0]) if counted else ["mode"], lineterminator="\n")
        w.writeheader()
        w.writerows(counted)
    C.log(f"[export] wrote {cpath} ({len(counted)} counted instances; current prompt only)")


# ---------------------------------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("stage", choices=["build", "count", "report", "export"])
    ap.add_argument("--runs-root", type=Path, default=Path(os.environ.get("CE_RUNS_DIR") or Path.home() / "ce-runs").expanduser())
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--results", type=Path, default=None, help="default <out>/results")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--only", help="comma-separated unit ids or run:iteration")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--allow-bash", action="store_true", help="give the judge bash (off by default: it can read outside "
                                                              "the workspace, which breaks blinding)")
    ap.add_argument("--pass", dest="pass_", choices=["A", "B"], default="A",
                    help="count: A = visible words (all iterations); B = re-check of A's claims with the reasoning (GLM)")
    ap.add_argument("--api-judge", help="run the judge as one plain API call to this model for BOTH passes (e.g. gpt-6-sol); "
                                         "results go to count_<model>/ and rethink_<model>/")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--partial", action="store_true", help="report: allow iterations not yet counted (pilot)")
    ap.add_argument("--f2", type=Path, help="report: folder with fm-v2 units.csv for COMPARE_F2.md")
    args = ap.parse_args()
    args.results = args.results or args.out / "results"
    runs = Runs(args.runs_root, args.out / "private")
    {"build": run_build, "count": run_count, "report": run_report, "export": run_export}[args.stage](args, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
