"""fm_facts.py -- the harness facts of one iteration, and the failure modes Python decides (fm-v3).

WHY: three modes need no judgement at all; they follow from the eval logs, the code identities and the memory tool calls
(KR kept a measured regression, UL unmeasured landing, BO blind memory overwrite). Python decides them, so they are exactly
reproducible. For the judge modes it computes candidates and context only (never verdicts): the failed versions UR is about,
the memory lines FN is about, the rule text BC is about, and the score-like numbers in the agent's words, which the judge
must account for one by one (fm_v3.py checks that none is skipped).

The rules are coded directly (they are the definitions of the Python-decided modes):
  KR  closing code changed, checker did not revert; the agent scored the closing code on the comparison set (result shown);
      the mean of those scores is more than 1 comparison SD below the start code's comparison mean (on record before the
      session, plus C0 comparison rows of this session).
  UL  closing code changed, checker did not revert; no comparison eval run by the agent on the closing code; comparison
      credit left at the end of the iteration.
  BO  iteration >= 2, memory_before not empty; a successful full-file write of memory.md before which the agent had not
      read memory.md (no read/grep/cat call on it, and no tool result or prompt that showed at least 3 of its substantive
      lines); at least half of the substantive (>= 25 characters, markers ignored) lines of memory_before are gone at the
      end of the iteration.

Read-only on the run folders (fm_runs.RunData). Standard library only.
"""
from __future__ import annotations

import json
import re

import fm_runs as F

SCORE_RE = re.compile(r"(?<![\w.])0?\.\d{2,}(?!\d)|\b\d{1,3}(?:\.\d+)?\s?%")   # same as fm_count.SCORE_RE
MARKER_RE = re.compile(r"^\s*(?:[-*+>]|#{1,6}|\d+[.)])\s+")


def mean(xs):
    return sum(xs) / len(xs) if xs else None


def sd_of(bench: str, which: str) -> float:
    """One noise SD of ONE eval of identical code (tau / sqrt 2)."""
    return F.tau_of(bench, which) / 2 ** 0.5


def substantive(text: str) -> set[str]:
    """Substantive memory lines: >= 25 characters after dropping list/heading markers, whitespace-normalised."""
    out = set()
    for ln in (text or "").split("\n"):
        s = F.norm_line(MARKER_RE.sub("", ln))
        if len(s) >= 25:
            out.add(s)
    return out


def norm_ws(s: str) -> str:
    return " ".join(str(s).split())


# ---------------------------------------------------------------------------------------------------------------
# the eval table of one iteration (code labels, sample aliases, the turn that showed each eval)
# ---------------------------------------------------------------------------------------------------------------
def eval_table(rd: F.RunData, it: int, steps: list[dict]) -> dict:
    """Code labels: C0 = the start code, then C1, C2 ... in order of first eval; the same label = identical file contents.
    Sample aliases S1, S2 ... for train/variance samples. `shown` = the first turn whose tool result contains the eval id."""
    R = rd.repo
    rec = rd.rec(it)
    start_sig, close_sig = R.sig(rec["parent_sha"]), R.sig(rec["close_sha"])
    labels: dict[str, str] = {}
    if start_sig:
        labels[start_sig] = "C0"

    def lab(sig):
        if sig is None:
            return "?"
        if sig not in labels:
            labels[sig] = f"C{len(labels)}"
        return labels[sig]

    evs = rd.evals_of(it)
    shown: dict[str, int] = {}
    for s in steps:
        if s["kind"] == "results":
            for r in s["results"]:
                for e in evs:
                    if e["eval_id"] not in shown and e["eval_id"] in r["content"]:
                        shown[e["eval_id"]] = s["turn"]
    samples: dict[str, str] = {}
    rows = []
    for n, e in enumerate(evs, 1):
        comp = e.get("eval_sample") == "comparison"
        sk = e.get("sample_key")
        if not comp and sk:
            samples.setdefault(sk, f"S{len(samples) + 1}")
        sig = R.sig(e["tree"]) if e.get("tree") else None
        rows.append(dict(n=n, set=e.get("eval_sample"), sample="-" if comp else samples.get(sk, "-"), code=lab(sig),
                         sig=sig, rows=e.get("n_rows"), score=e.get("score"), by="harness" if e.get("by_loop") else "agent",
                         shown=shown.get(e["eval_id"]), eval_id=e["eval_id"]))
    close_label = lab(close_sig)
    return dict(rows=rows, labels=labels, start_sig=start_sig, close_sig=close_sig, close_label=close_label,
                changed=None if start_sig is None or close_sig is None else start_sig != close_sig)


def start_comparison_refs(rd: F.RunData, it: int, start_sig) -> list[tuple[float, str]]:
    """The start code's comparison scores on record before this session: comparison-history rows of earlier iterations on the
    start code; else the comparison score the harness recorded for the parent iteration (when it closed on the start code)."""
    R, ref = rd.repo, []
    rows = [c for c in rd.comps if int(c.get("iteration") or 0) <= it]
    for n, c in enumerate(rows, 1):
        s = R.sig(c["tree"]) if c.get("tree") else None
        if s and s == start_sig and int(c.get("iteration") or 0) < it and c.get("score") is not None:
            ref.append((c["score"], f"CMP#{n}"))
    parent_it = rd.rec(it).get("parent_iteration")
    if not ref and parent_it is not None and str(parent_it) in rd.history["iteration_history"]:
        prec = rd.rec(int(parent_it))
        if prec.get("comparison_set_score") is not None and R.sig(prec.get("close_sha") or "") == start_sig:
            ref.append((prec["comparison_set_score"], f"recorded for iteration {parent_it}"))
    return ref


# ---------------------------------------------------------------------------------------------------------------
# the Python modes
# ---------------------------------------------------------------------------------------------------------------
def decide_kr(t: dict, ref: list, gate: bool, sd_c: float) -> dict | None:
    if not t["changed"] or gate:
        return None
    close = [r for r in t["rows"] if r["set"] == "comparison" and r["by"] == "agent" and r["code"] == t["close_label"]
             and r["score"] is not None and r["shown"] is not None]
    start = [s for s, _ in ref] + [r["score"] for r in t["rows"]
                                   if r["set"] == "comparison" and r["code"] == "C0" and r["score"] is not None]
    if not close or not start:
        return None
    mc, ms = mean([r["score"] for r in close]), mean(start)
    if ms - mc <= sd_c:
        return None
    return dict(close_mean=mc, start_mean=ms, gap_sd=(ms - mc) / sd_c, evals=[f"EVAL#{r['n']}" for r in close])


def decide_ul(t: dict, gate: bool, credits) -> dict | None:
    if not t["changed"] or gate or not credits or credits <= 0:
        return None
    if any(r["set"] == "comparison" and r["by"] == "agent" and r["code"] == t["close_label"] for r in t["rows"]):
        return None
    return dict(close_label=t["close_label"], credits_after=credits)


def decide_bo(it: int, steps: list[dict], mem_before: str, mem_after: str) -> dict | None:
    """Python BO. The replay events of fm_runs give the memory tool calls in order; a call counts as a read when fm_runs
    classifies it so, or when any earlier tool result or the task prompt showed >= 3 substantive lines of memory_before."""
    old = substantive(mem_before)
    if it < 2 or not old:
        return None
    seen_lines = 0
    shown_turn = None
    for s in steps:   # the first turn at which >= 3 old lines had been shown by a prompt or a tool result
        texts = [s.get("text") or ""] if s["kind"] in ("prompt", "notice") else \
            [r["content"] for r in s["results"]] if s["kind"] == "results" else []
        for tx in texts:
            seen_lines = len(old & substantive(tx))
            if seen_lines >= 3:
                shown_turn = s["turn"]
                break
        if shown_turn is not None:
            break
    events, _ = F.replay_iteration(mem_before, F.mem_items(steps))
    for ev in events:
        if ev["kind"] != "write_overwrite" or not ev["ok"]:
            continue
        if ev["read_before"] or (shown_turn is not None and shown_turn < ev["turn"]):
            return None    # the first full overwrite came after a read: not blind (later overwrites too)
        gone = old - substantive(mem_after)
        if len(gone) * 2 < len(old):
            return None
        return dict(turn=ev["turn"], old_lines=len(old), gone=len(gone))
    return None


# ---------------------------------------------------------------------------------------------------------------
# candidates for the judge (context, never verdicts)
# ---------------------------------------------------------------------------------------------------------------
def ur_candidates(t: dict, ref: list, sd_t: float, sd_c: float) -> list[dict]:
    """Failed versions: a non-C0, non-closing code version scored (result shown) more than 1 SD below the start code on the
    same eval set and sample (train/variance: C0 rows of the same sample alias; comparison: start code's comparison mean)."""
    out = []
    c0_train: dict[str, list[float]] = {}
    for r in t["rows"]:
        if r["code"] == "C0" and r["set"] != "comparison" and r["score"] is not None:
            c0_train.setdefault(r["sample"], []).append(r["score"])
    c0_comp = [s for s, _ in ref] + [r["score"] for r in t["rows"]
                                     if r["code"] == "C0" and r["set"] == "comparison" and r["score"] is not None]
    for r in t["rows"]:
        if r["code"] in ("C0", t["close_label"], "?") or r["score"] is None or r["shown"] is None:
            continue
        if r["set"] == "comparison":
            base, sd = mean(c0_comp), sd_c
        else:
            base, sd = mean(c0_train.get(r["sample"], [])), sd_t
        if base is not None and base - r["score"] > sd:
            out.append(dict(eval=f"EVAL#{r['n']}", code=r["code"], set=r["set"], sample=r["sample"], score=r["score"],
                            start_mean=base, gap_sd=(base - r["score"]) / sd, shown=r["shown"]))
    return out


def memory_new_lines(mem_before: str, mem_after: str) -> list[tuple[int, str]]:
    """Lines of memory_after (1-based number, text) that are not lines of memory_before (whitespace-normalised)."""
    before = {norm_ws(x) for x in (mem_before or "").split("\n")}
    return [(i, ln) for i, ln in enumerate((mem_after or "").split("\n"), 1) if ln.strip() and norm_ws(ln) not in before]


def number_ledger(steps: list[dict], new_lines: list[tuple[int, str]]) -> list[dict]:
    """Every score-like number in the agent's visible words that the judge must account for: the text of its assistant
    turns, the inputs of its calls to harness (ce_*) tools, and the memory lines of this session. One entry per distinct
    (place, number); ids N1, N2 ... in order."""
    out, seen = [], set()

    def add(where: str, text: str):
        for m in SCORE_RE.finditer(text or ""):
            num = norm_ws(m.group(0)).replace(" ", "")
            if (where, num) not in seen:
                seen.add((where, num))
                out.append(dict(id=f"N{len(out) + 1}", where=where, number=num))

    for s in steps:
        if s["kind"] != "assistant":
            continue
        for tx in s["texts"]:
            add(f"T{s['turn']}", tx)
        for c in s["calls"]:
            if str(c["name"]).startswith("ce_"):
                for _, v in call_fields(c["input"]):
                    add(f"T{s['turn']}", v)
    for i, ln in new_lines:
        add(f"MEM-AFTER:L{i}", ln)
    return out


def call_fields(inp) -> list[tuple[str, str]]:
    """(field, text) per input field of a tool call: strings raw, anything else as JSON (same as fm_count.call_fields)."""
    if not isinstance(inp, dict):
        return [("input", json.dumps(inp, ensure_ascii=False))]
    return [(k, v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)) for k, v in inp.items()]


# ---------------------------------------------------------------------------------------------------------------
def facts_of(rd: F.RunData, it: int) -> dict:
    """Everything Python knows about iteration `it` of a run: the eval table, the start code's references, the Python
    modes, and the judge candidates. Steps are left cached in `rd` for the caller (drop them with rd.drop_steps)."""
    spec, rec = rd.spec, rd.rec(it)
    steps = rd.steps(it)
    t = eval_table(rd, it, steps)
    ref = start_comparison_refs(rd, it, t["start_sig"])
    gate = bool(rec.get("constraint_violation"))
    credits = rec.get("comparison_credits_after")
    sd_t, sd_c = sd_of(spec.bench, "train"), sd_of(spec.bench, "comparison")
    mem_before = rd.snaps.get(it - 1, "") if it > 1 else ""
    mem_after = rd.snaps.get(it, "")
    new_lines = memory_new_lines(mem_before, mem_after)
    return dict(
        table=t, ref=ref, gate=gate, rule_text=norm_ws(str(rec.get("constraint_violation") or "")) if gate else None,
        credits_after=credits, sd_train=sd_t, sd_comparison=sd_c, parent_iteration=rec.get("parent_iteration"),
        mem_before=mem_before, mem_after=mem_after, new_lines=new_lines,
        python_modes={"KR": decide_kr(t, ref, gate, sd_c), "UL": decide_ul(t, gate, credits),
                      "BO": decide_bo(it, steps, mem_before, mem_after)},
        ur_candidates=ur_candidates(t, ref, sd_t, sd_c),
        numbers=number_ledger(steps, new_lines))
