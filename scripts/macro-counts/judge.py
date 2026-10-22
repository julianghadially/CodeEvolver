#!/usr/bin/env python3
"""Ranked macro-approach judge: one read-only OpenCode session per pair of runs.

The judge is the engine's OpenCode *inspection* session (the same harness the
post-run auditors use), pointed at a judge workspace that ``macro_counts.py``
materialized on disk: ``INDEX.md``, the seed program, and for Program A and
Program B every iteration's summary, diff and full program plus the final
program. The agent reads what it needs with read/grep/glob/list (no bash, no
edit, no CodeEvolver skills, no ``/s`` bootstrap, no memory) and ends its
reply with exactly one fenced ```json block, which is parsed the way
``src/engine/optimizer/meta/run_integrity.py`` parses its verdict.

The judge (see ``judge_prompt.md``) lists each program's distinct macro
approaches, consolidated within the run and ranked by departure from the seed
(at most ``max_per_program`` each), then ranks the approaches of both programs
in one combined order, leaving out any approach both programs implemented. An
id that is listed but absent from the combined ranking is therefore "shared".
Counts (how many of each program's approaches land in the top k) are computed
here from the ranking, never reported by the model: see ``top_k_counts``.

Repeatability: fresh OpenCode config dir and data dir per session, pinned
model, read-only tools, no skills, no memory. A judgment depends on the prompt,
the workspace files, model sampling, and the agent's own reading path (which
files it opened). ``tool_calls`` in the wrapper and the exported transcript
make that path inspectable.

Validation is deliberately light: only what the count needs. Ids must be well
formed and unique, and every id in the combined ranking must resolve to a
listed approach exactly once; otherwise ``top_k_counts`` would count the wrong
thing. Everything else (the cap, reordering) is a warning shown in the review.

Library use:
    from judge import judge_workspace, load_judgment, top_k_counts
    result = judge_workspace(Path("workspaces/alfworld-s2__ab"), pair_id="alfworld-s2__ab")
    top_k_counts(result["judgment"], 7)   # {"A": 4, "B": 3}

CLI use (one pair, one order):
    python judge.py workspaces/alfworld-s2__ab --out judge/alfworld-s2__ab.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
PROMPT_PATH = HERE / "judge_prompt.md"
DEFAULT_MODEL = "deepinfra/zai-org/GLM-5.2"  # OpenCode provider/model form
DEFAULT_MAX_TURNS = 80          # OpenCode has no native turn cap; the harness uses this for its timeout
DEFAULT_TIMEOUT_SECONDS = 7200  # hard wall-clock ceiling per session (process-group kill); GLM via DeepInfra is slow
DEFAULT_MAX_PER_PROGRAM = 15
DEFAULT_RETRIES = 3
EXPORT_TIMEOUT_SECONDS = 120
STATUSES = ["landed", "reverted", "superseded"]
PROGRAMS = ["A", "B"]
LIST_KEY = {"A": "program_a_approaches", "B": "program_b_approaches"}

# The shape the agent must end with. Shown verbatim in the task block.
JUDGMENT_SHAPE = """{
  "program_a_approaches": [
    {"id": "A1", "name": "<short name>", "description": "<one sentence: the mechanism>",
     "iterations": [<every iteration that touched it>], "status": "landed|reverted|superseded",
     "in_final_program": true|false, "evidence": "<function/section names from the diff, quoted summary phrases>"}
  ],
  "program_b_approaches": [ {"id": "B1", ...} ],
  "combined_ranking": [
    {"program": "A|B", "id": "<an id from that program's list>", "departure_note": "<one sentence>"}
  ],
  "rationale": "<150-300 words>",
  "review_flags": ["<anything a human should double-check>"]
}"""

_FENCED_JSON_RE = re.compile(r"```(?:json)?\s*\n(\{.*?\})\s*\n```", re.DOTALL)


# ---------------------------------------------------------------------------
# Environment / prompt
# ---------------------------------------------------------------------------

def prepare_engine_env(scratch: Path, *, model: str = DEFAULT_MODEL) -> None:
    """Env preamble BEFORE any engine import (same shape as
    scripts/run_meta_reflection.py): local mode, a job id, and memory/trace
    dirs under ``scratch`` because the inspection permission builder reads
    ``paths.memory_dir()`` / ``paths.traces_dir()``. The OpenCode small model
    defaults to an OpenAI model; pin it to the judge model so no second
    provider key is needed."""
    scratch = Path(scratch)
    (scratch / "memory").mkdir(parents=True, exist_ok=True)
    (scratch / "traces").mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("APP_MODE", "local")
    os.environ.setdefault("SANDBOX_MODE", "local")
    os.environ.setdefault("CODEEVOLVER_JOB_ID", "macro-counts")
    os.environ.setdefault("CODEEVOLVER_MEMORY_DIR", str(scratch / "memory"))
    os.environ.setdefault("CODEEVOLVER_TRACES_DIR", str(scratch / "traces"))
    os.environ.setdefault("CODEEVOLVER_OPENCODE_SMALL_MODEL", model)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


# OpenCode provider prefix -> the env var OpenCode reads that provider's key from.
PROVIDER_KEYS = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY", "deepinfra": "DEEPINFRA_API_KEY"}


def provider_key(model: str) -> str | None:
    return PROVIDER_KEYS.get(model.split("/", 1)[0])


def load_env_key(*search_dirs: Path) -> None:
    """Put any provider key missing from the environment in from the first ``.env`` that has it."""
    for var in PROVIDER_KEYS.values():
        if os.environ.get(var):
            continue
        for d in search_dirs or (HERE, HERE.parent, HERE.parent.parent, Path.cwd()):
            env = d / ".env"
            if not env.exists():
                continue
            for line in env.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith(f"{var}="):
                    os.environ[var] = line.split("=", 1)[1].strip().strip('"').strip("'")
            if os.environ.get(var):
                break


def system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def prompt_sha256() -> str:
    """Hash of the judge prompt; stored with every judgment so a cached judgment
    made under an earlier definition is detected as stale (see macro_counts.py)."""
    return hashlib.sha256(system_prompt().encode("utf-8")).hexdigest()


def task_block(max_per_program: int) -> str:
    return (
        "## Your task now\n\n"
        "Your working directory is the judge workspace for one pair of runs. Start by reading `INDEX.md`, "
        "which lists every file. Read every iteration's `summary.md` and `diff.patch` (or `attempted.patch`) "
        "for both programs, the seed program, and both final programs; open the full program file of an "
        "iteration whenever a diff alone is ambiguous. Do not sample or skim.\n\n"
        f"List at most {max_per_program} approaches per program. When you are done, end your reply with "
        "EXACTLY ONE fenced ```json block (and nothing after it) of this shape:\n\n"
        f"```json\n{JUDGMENT_SHAPE}\n```\n"
    )


def build_prompt(max_per_program: int = DEFAULT_MAX_PER_PROGRAM) -> str:
    return system_prompt().rstrip() + "\n\n" + task_block(max_per_program)


def workspace_sha256(ws: Path) -> str:
    """Content hash of a judge workspace: sorted (relative path, file sha256) pairs."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(ws).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(ws)).encode("utf-8"))
        h.update(b"\0")
        h.update(hashlib.sha256(f.read_bytes()).hexdigest().encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class JudgeError(RuntimeError):
    """A failure that a retry will not fix (opencode missing, bad workspace)."""


def run_session(workspace: Path, prompt: str, *, model: str, max_turns: int,
                timeout_seconds: int, allow_bash: bool = False) -> dict:
    """One read-only OpenCode inspection session over ``workspace``.

    Mirrors ``subprocess_service.run_inspection_agent`` but builds the harness
    directly so the CodeEvolver skills and the ``/s`` bootstrap are OFF
    (``stage_project_skills=False``; the entry function does not expose it).
    ``allow_bash`` turns on the bash tool (the clean-slate inspector's
    exception), which catalog sessions need for ``git show`` / ``git checkout``
    in the run clones; edit/write stay denied either way. Returns the dict
    shape run_inspection_agent returns."""
    from src.engine.agent.harnesses.base import (AgentRole, HarnessPermissionPolicy,  # noqa: E402
                                                 HarnessRequest)
    from src.engine.agent.harnesses.factory import create_coding_harness  # noqa: E402

    harness = create_coding_harness(
        coding_harness="opencode", coding_harness_model=model,
        architect_config=SimpleNamespace(stage_project_skills=False),
    )
    res = harness.run_inspection(HarnessRequest(
        role=AgentRole.INSPECTION, workspace=str(workspace), prompt=prompt,
        permissions=HarnessPermissionPolicy.autonomous_policy(), max_turns=max_turns,
        raw_provider_metadata={"writable": False, "deny_paths": [], "run_as_agent_user": False,
                               "allow_bash": allow_bash, "timeout_seconds": timeout_seconds},
    ))
    meta = dict(res.raw_provider_metadata or {})
    return {"success": res.success, "output": res.output or "", "error": res.error,
            "session_id": res.session_id, "data_dir": meta.get("data_dir"),
            "usage": meta.get("usage") or {}, "tool_calls": meta.get("tool_calls") or [],
            "error_events": meta.get("error_events") or [], "returncode": meta.get("returncode")}


def export_transcript(session_id: str, data_dir: str | None, out_path: Path) -> bool:
    """``opencode export <session_id>`` into ``out_path``. Best effort.

    Inspection sessions live in their own data dir (XDG_DATA_HOME /
    XDG_STATE_HOME), so the export must be pointed at it. stdout goes to a
    FILE, not a pipe: opencode exits without draining a >64KiB pipe."""
    env = dict(os.environ)
    if data_dir:
        env.update({"XDG_DATA_HOME": data_dir, "XDG_STATE_HOME": data_dir})
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            p = subprocess.run(["opencode", "export", session_id], stdout=f, stderr=subprocess.PIPE,
                               text=True, timeout=EXPORT_TIMEOUT_SECONDS, env=env)
        if p.returncode != 0 or out_path.stat().st_size == 0:
            out_path.unlink(missing_ok=True)
            return False
        json.loads(out_path.read_text(encoding="utf-8"))
        return True
    except Exception:  # noqa: BLE001 - never load-bearing
        out_path.unlink(missing_ok=True)
        return False


def parse_fenced_json(text: str | None, required_key: str = "combined_ranking") -> dict | None:
    """The last fenced JSON block that parses to a dict holding ``required_key``, else None."""
    if not text:
        return None
    for match in reversed(_FENCED_JSON_RE.findall(text)):
        try:
            data = json.loads(match)
        except (ValueError, TypeError):
            continue
        if isinstance(data, dict) and required_key in data:
            return data
    return None


def run_json_session(ws: Path, prompt: str, *, validate, required_key: str, label: str = "",
                     model: str = DEFAULT_MODEL, max_turns: int = DEFAULT_MAX_TURNS,
                     timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS, retries: int = DEFAULT_RETRIES,
                     allow_bash: bool = False, transcript_path: Path | None = None,
                     rejected_dir: Path | None = None, log=lambda msg: None) -> dict:
    """Run fresh sessions over ``ws`` until one ends with a valid fenced JSON block.

    ``validate(parsed) -> list[str]`` returns non-fatal warnings and raises
    ``ValueError`` for a result that a fresh session should replace. Retries
    when the session failed, produced no fenced JSON holding ``required_key``,
    or failed validation; a missing ``opencode`` binary raises ``JudgeError``
    at once. Each rejected attempt's reply is kept in ``rejected_dir`` (when
    given) with its session id and data dir, so it can be inspected or salvaged. Returns the session wrapper: model, session id, usage, tool calls,
    retries, warnings and the parsed JSON under ``result`` (callers add their
    own keys and hashes)."""
    attempts: list[str] = []
    for attempt in range(1, retries + 1):
        t0 = time.time()
        try:
            res = run_session(ws, prompt, model=model, max_turns=max_turns, timeout_seconds=timeout_seconds,
                              allow_bash=allow_bash)
        except RuntimeError as e:  # binary/version problems: not retryable
            raise JudgeError(f"{label}: {e}") from e
        elapsed = time.time() - t0
        parsed = parse_fenced_json(res["output"], required_key)
        problem = None
        if not res["success"]:
            problem = f"session failed: {res['error']}"
        elif parsed is None:
            problem = "no fenced JSON block in the reply"
        else:
            try:
                warnings = validate(parsed)
            except ValueError as e:
                problem = str(e)
        if problem is None:
            wrapper = {
                "harness": "opencode",
                "model": model,
                "max_turns": max_turns,
                "timeout_seconds": timeout_seconds,
                "judged_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
                "session_id": res["session_id"],
                "data_dir": res["data_dir"],
                "elapsed_seconds": round(elapsed),
                "usage": res["usage"],
                "tool_calls": res["tool_calls"][:1000],
                "n_tool_calls": len(res["tool_calls"]),
                "retries": attempt - 1,
                "attempt_errors": attempts,
                "warnings": warnings,
                "transcript": None,
                "result": parsed,
            }
            if transcript_path and res["session_id"]:
                if export_transcript(res["session_id"], res["data_dir"], transcript_path):
                    wrapper["transcript"] = transcript_path.name
                else:
                    log(f"[{label}] transcript export failed (non-fatal)")
            return wrapper
        attempts.append(f"attempt {attempt}: {problem}")
        if rejected_dir is not None:
            rejected_dir.mkdir(parents=True, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%dT%H%M%S")
            (rejected_dir / f"{label.replace(' ', '_')}.{stamp}.attempt{attempt}.json").write_text(json.dumps(
                {"label": label, "attempt": attempt, "problem": problem, "session_id": res["session_id"],
                 "data_dir": res["data_dir"], "usage": res["usage"], "output": res["output"]}, indent=1),
                encoding="utf-8")
        log(f"[{label}] {attempts[-1]} ({elapsed:.0f}s)" + ("; retrying" if attempt < retries else ""))
    raise RuntimeError(f"session failed for {label} after {retries} attempts: " + " | ".join(attempts))


def judge_workspace(ws: Path, *, pair_id: str = "", model: str = DEFAULT_MODEL,
                    max_turns: int = DEFAULT_MAX_TURNS, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
                    max_per_program: int = DEFAULT_MAX_PER_PROGRAM, retries: int = DEFAULT_RETRIES,
                    transcript_path: Path | None = None, log=lambda msg: None) -> dict:
    """Judge one workspace. Returns the wrapper dict stored as judge/<pair>__<order>.json."""
    ws = Path(ws).resolve()
    if not (ws / "INDEX.md").exists():
        raise JudgeError(f"{pair_id}: not a judge workspace (no INDEX.md): {ws}")
    res = run_json_session(
        ws, build_prompt(max_per_program), required_key="combined_ranking", label=pair_id,
        validate=lambda j: validate_judgment(j, max_per_program=max_per_program, where=pair_id or "judgment"),
        model=model, max_turns=max_turns, timeout_seconds=timeout_seconds, retries=retries,
        transcript_path=transcript_path, log=log)
    judgment = res.pop("result")
    return {"pair_id": pair_id, **res, "max_per_program": max_per_program,
            "workspace_sha256": workspace_sha256(ws), "prompt_sha256": prompt_sha256(), "judgment": judgment}


# ---------------------------------------------------------------------------
# Validation / counts
# ---------------------------------------------------------------------------

ID_RE = {"A": re.compile(r"A\d+"), "B": re.compile(r"B\d+")}


def unique_ids(j: dict) -> dict[str, list[str]]:
    """Per program, the listed ids that appear in the combined ranking (i.e. not shared), in rank order."""
    listed = {p: {a["id"] for a in j.get(LIST_KEY[p], [])} for p in PROGRAMS}
    return {p: [c["id"] for c in j.get("combined_ranking", []) if c.get("program") == p and c.get("id") in listed[p]]
            for p in PROGRAMS}


def shared_ids(j: dict) -> dict[str, list[str]]:
    """Per program, the listed ids the judge left out of the combined ranking (both runs implemented them)."""
    ranked = {c.get("id") for c in j.get("combined_ranking", [])}
    return {p: [a["id"] for a in j.get(LIST_KEY[p], []) if a["id"] not in ranked] for p in PROGRAMS}


def validate_judgment(j: dict, *, max_per_program: int = DEFAULT_MAX_PER_PROGRAM,
                      where: str = "judgment") -> list[str]:
    """Raise ValueError only when the count could not be computed correctly;
    return non-fatal warnings.

    Fatal: a required key is missing; a per-program id is not ``A<n>`` /
    ``B<n>`` or is duplicated; a combined-ranking entry names an id that is not
    in the matching program's list, or names it twice.
    Warning: a list exceeds ``max_per_program``; the combined ranking reorders a
    program's own step-1 order; an approach has an unknown status."""
    problems: list[str] = []
    warnings: list[str] = []
    for key in ("program_a_approaches", "program_b_approaches", "combined_ranking", "rationale", "review_flags"):
        if key not in j:
            problems.append(f"missing '{key}'")
    ids: dict[str, list[str]] = {}
    for p in PROGRAMS:
        lst = j.get(LIST_KEY[p], []) or []
        ids[p] = [str(a.get("id")) for a in lst]
        if len(lst) > max_per_program:
            warnings.append(f"program {p} lists {len(lst)} approaches (cap {max_per_program})")
        for a in lst:
            if not ID_RE[p].fullmatch(str(a.get("id"))):
                problems.append(f"program {p}: id {a.get('id')!r} is not of the form {p}<n>")
            if a.get("status") not in STATUSES:
                warnings.append(f"approach {a.get('id')}: status {a.get('status')!r} is not one of {STATUSES}")
        dups = sorted({i for i in ids[p] if ids[p].count(i) > 1})
        if dups:
            problems.append(f"program {p}: duplicate ids {dups}")
    combined = j.get("combined_ranking", []) or []
    got = [str(c.get("id")) for c in combined]
    for c in combined:
        cid, prog = str(c.get("id")), c.get("program")
        if prog not in PROGRAMS or cid not in ids[prog]:
            problems.append(f"combined ranking: id {cid!r} is not a listed approach of program {prog!r}")
    dup = sorted({i for i in got if got.count(i) > 1})
    if dup:
        problems.append(f"combined ranking repeats ids {dup}")
    if problems:
        raise ValueError(f"{where}: invalid judgment: " + "; ".join(problems))
    uniq = unique_ids(j)
    for p in PROGRAMS:
        own_order = [i for i in ids[p] if i in uniq[p]]
        if uniq[p] != own_order:
            warnings.append(f"combined ranking reorders program {p}'s approaches: {own_order} -> {uniq[p]}")
    return warnings


def top_k_counts(j: dict, k: int) -> dict[str, int]:
    """How many of each program's unshared approaches are in the top k of the combined ranking.
    k <= 0 counts the whole ranking, i.e. every approach above the incremental cutoff."""
    top = j["combined_ranking"][:k] if k > 0 else j["combined_ranking"]
    return {p: sum(1 for c in top if c["program"] == p) for p in PROGRAMS}


def load_judgment(path: Path) -> dict:
    """Read judge/<pair>__<order>.json (wrapper or bare judgment), fill wrapper defaults, validate."""
    jd = json.loads(path.read_text(encoding="utf-8"))
    if "judgment" not in jd and "combined_ranking" in jd:
        jd = {"judgment": jd}
    jd.setdefault("pair_id", path.stem)
    jd.setdefault("harness", "manual")
    jd.setdefault("model", "manual")
    jd.setdefault("max_per_program", DEFAULT_MAX_PER_PROGRAM)
    jd.setdefault("judged_at", dt.datetime.fromtimestamp(path.stat().st_mtime, dt.timezone.utc)
                  .isoformat(timespec="seconds"))
    jd.setdefault("usage", {})
    jd.setdefault("n_tool_calls", len(jd.get("tool_calls") or []))
    jd.setdefault("retries", 0)
    jd["warnings"] = validate_judgment(jd["judgment"], max_per_program=jd["max_per_program"], where=path.name)
    return jd


def main() -> int:
    ap = argparse.ArgumentParser(description="Judge one pair workspace with an OpenCode session; JSON to --out or stdout.")
    ap.add_argument("workspace", type=Path, help="a judge workspace dir (contains INDEX.md)")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="OpenCode provider/model id")
    ap.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_SECONDS, help="seconds per session")
    ap.add_argument("--max-per-program", type=int, default=DEFAULT_MAX_PER_PROGRAM)
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES)
    ap.add_argument("--pair-id", default=None, help="defaults to the workspace dir name")
    ap.add_argument("--out", type=Path, default=None, help="write the judgment JSON here; default is stdout")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="dir for the engine's memory/trace scratch (default: <workspace>/../../opencode)")
    args = ap.parse_args()
    scratch = args.scratch or (args.workspace.resolve().parent.parent / "opencode")
    prepare_engine_env(scratch, model=args.model)
    load_env_key(REPO_ROOT, HERE)
    pair_id = args.pair_id or args.workspace.resolve().name
    transcript = args.out.with_suffix(".transcript.json") if args.out else None
    res = judge_workspace(args.workspace, pair_id=pair_id, model=args.model, max_turns=args.max_turns,
                          timeout_seconds=args.timeout, max_per_program=args.max_per_program,
                          retries=args.retries, transcript_path=transcript,
                          log=lambda m: print(m, file=sys.stderr))
    if args.out:
        args.out.write_text(json.dumps(res, indent=1), encoding="utf-8")
    else:
        json.dump(res, sys.stdout, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
