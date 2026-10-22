"""Copyright © 2026 440 Labs LLC

Post-run meta-reflection.

This is CodeEvolver's answer to the HyperAgents "self-improving meta agent",
deliberately DECOUPLED to avoid that design's drawbacks (no controls / security
exposure from in-loop self-modification, doubled optimization cost, and weak
outcome→meta-agent attribution amid high between-cycle variance).

Instead of letting the meta-agent rewrite its own controller mid-run, a single
meta-reflection agent runs ONCE at the very end — after all iterations, after
held-out test scoring, and after the run has been posted to MongoDB. It is
strictly ADVISORY: it produces structured improvement proposals for CodeEvolver's
own meta-agent surfaces (architect prompt, skills, tool contracts, controls,
memory schema), grounded in evidence from this run. Nothing
is auto-applied at this point.

It reflects on four inputs:
  1. **CodeEvolver itself** — its specs (``{engine_root()}/specs/requirements.md``
     + ``engine3.md``) and the real meta-agent source under
     ``{engine_root()}/src/engine/optimizer/v3/`` (prompts, skills, tool
     contracts, controls). The agent reads these by absolute path so its
     proposals are grounded in how CodeEvolver actually works, not guesses.
     ``engine_root()`` resolves to the CodeEvolver repo root; when ``specs/``
     is not present in the environment the prompt says to skip it gracefully.
  2. Cross-iteration memory — ``{memory_dir()}/memory.md`` and the per-iteration
     architect session traces.
  3. The task (client) repository at ``workspace``.
  4. The IterationArchitect's own session traces —
     ``{architect_traces_dir()}/iteration_{N}_architect_trace/`` (thinking, tool
     calls, tool results; one ``turn_NNNN.json`` per turn plus ``index.jsonl``).
     These are captured only when ``enable_meta_reflection`` is on (see
     ``architect_trace``).

It reuses the read-only inspection sub-agent (``run_inspection_agent``) — the
same vehicle the architect's ``inspect_traces`` tool uses — because the architect
traces can be large and that agent already navigates them via Grep + Read.
Output is written to ``{memory_dir()}/meta_reflection.md`` and logged.

STRICTLY non-fatal: the run result and MongoDB completion are committed before
this runs, so any failure here only forfeits the advisory artifact.

Three pieces here are SHARED with the simple_loop arm's own meta-reflection
(``optimizer/meta/simple_meta_reflection.py``): ``format_score``,
``architect_trace_reading_guide``, and the prompt-agnostic driver
``run_reflection_agent``. The prompt PROSE is deliberately NOT shared — the two
arms expose different surfaces — but the mechanism and the trace-reading guide
must never drift apart.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any

from src.engine.paths import architect_traces_dir, engine_root, memory_dir

logger = logging.getLogger(__name__)


def format_score(score: float | None) -> str:
    """Render a score for a prompt: 4dp, or ``"N/A"`` when unmeasured."""
    return f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"


def architect_trace_reading_guide(iteration_indices: list[int]) -> str:
    """The architect-trace navigation block, shared by both arms' prompts.

    Returned WITHOUT a list marker: each caller prefixes its own number
    (``"2. "`` in engine 3, ``"1. "`` in simple_loop). This is load-bearing JSONL
    schema documentation plus the graceful-degradation sentence — the one
    prompt fragment that is genuinely identical for both arms and the most
    expensive to let drift.
    """
    arch = architect_traces_dir()
    idx_list = ", ".join(str(i) for i in iteration_indices) or "(none)"
    return f"""IterationArchitect SESSION traces (the meta-agent's own thinking + actions):
   - One directory per iteration: `{arch}/iteration_{{N}}_architect_trace/` \
for N in: {idx_list}. Start with `{arch}/README.md` — one row per iteration \
(train baseline → best, comparison best, changes landed, watchdog resumes, eval \
rows, turn count, tool errors, elapsed, links).
   - READ IN THIS ORDER:
     (1) `digest.md` for EVERY iteration — the whole session minus thinking: a \
header (scores, branch, close sha, eval rows, watchdogs, tool-call counts, \
elapsed), then every turn in order: the opening `Prompt` and each `Watchdog \
resume — <which>` verbatim, the architect's text in full, each tool call as \
`name(args)`, each tool result capped at 1,000 chars, `run_eval` results as one \
`run_eval <sample>/<suffix> score=… rows=… zero/failed=… sha=…` line, errors \
marked **ERROR**. ~40–110K chars each.
     (2) `transcript.md` for the 1–2 iterations that matter most — adds the \
architect's full thinking (the most important signal) and full tool inputs; \
tool results capped at 2,000 chars. Transcripts are 250–450K chars: read them \
in pages (Read with offset/limit — on OpenCode each read returns ~50 KB, so \
plan 5–9 reads per transcript) or `Grep(pattern, path=<dir>)` for the turn \
headings `### turn NNNN`.
     (3) `turn_NNNN.json` only when a payload was capped (`→ full: \
turn_NNNN.json`) — the pretty-printed turn: assistant `blocks` = \
{{"kind":"thinking"|"text"|"tool_use",…}}; user `blocks` = \
{{"kind":"tool_result","tool_use_id","is_error","content"}} or {{"kind":"text"}} \
(a prompt / watchdog re-prompt); the trailing `result` turn carries `usage` + \
`meta`. A value shaped {{"__chunked_string__": [...]}} is one long string split \
into pieces; read it as their concatenation. `index.jsonl` (one line per turn: \
`turn`, `role`, `ts`, `kinds`, `tools`, `is_error`, `chars`, `file`, `summary`) \
is a grep-able map if you need one.
   - You have a Bash tool for read-only inspection (`git log`, `grep`, \
`python3 -c` over JSONL). Do not modify the workspace. NEVER read every turn \
file blindly.
   - If a digest says `NOT CAPTURED` (or a turn has `"captured": false`), no \
rich trace exists for that iteration (session capture/export was unavailable) \
— fall back to memory + iteration history for it. On OpenCode, thinking blocks \
may be sparse or absent (some providers export reasoning as opaque encrypted \
content); lean on text / tool_use / tool_result blocks for those runs."""


def build_meta_reflection_prompt(
    *,
    method: str,
    num_iterations: int,
    seed_test_score: float | None,
    best_candidate_test_score: float | None,
    workspace: str,
    iteration_indices: list[int],
) -> str:
    """Build the meta-reflection prompt (structured-proposals format)."""
    mem = memory_dir()
    ce_root = engine_root()
    req_spec = os.path.join(ce_root, "specs", "requirements.md")
    eng_spec = os.path.join(ce_root, "specs", "engine3.md")
    v3_src = os.path.join(ce_root, "src", "engine", "optimizer", "v3")
    # specs/ may be absent from the environment (e.g. a runner image that
    # ships only src/). Tailor the instruction so the agent doesn't chase
    # missing files.
    specs_present = os.path.exists(req_spec) and os.path.exists(eng_spec)
    specs_note = (
        ""
        if specs_present
        else (
            " NOTE: the specs/ directory is NOT present in this environment "
            "(e.g. Modal mounts only src/engine) — skip the specs and rely on "
            "the source under src/engine plus the run artifacts."
        )
    )
    trace_guide = architect_trace_reading_guide(iteration_indices)

    return f"""You are CodeEvolver's META-REFLECTION agent. An optimization run just \
finished. Your job is ADVISORY ONLY: propose improvements to CodeEvolver's own \
meta-agent (the "IterationArchitect" and its surrounding machinery), grounded in \
evidence from THIS run. You do not modify any files — a human reviews your \
output. Report only; do not attempt to edit code, prompts, or memory.

You MUST NOT read or open any validation/holdout or held-out test data. Those are \
fenced. Reason only from CodeEvolver's own files, memory, the architect's own \
traces, and the client repo.

## FIRST — study CodeEvolver itself
Study specs/requirements.md and specs/engine3.md to get context on the project \
before doing anything, then proceed with the request. You CAN see the CodeEvolver \
repository — it lives at `{ce_root}`. Read these by absolute path (Read / Grep, \
on demand — do not ask for inline content):
- `{req_spec}` and `{eng_spec}` — the authoritative project specs.{specs_note}
- The meta-agent source you will propose changes to, under `{v3_src}/`:
  `prompts.py` (architect system prompts), `skills/` (close_iteration, \
exploration, tracing), `agent_tools.py` + `iteration_architect.py` (tool \
contracts + iteration controls), and `architect_hooks.py` (guards).
Grounding proposals in this REAL source — exact prompt language, real tool \
signatures, the actual control logic — is what makes them actionable. Do this \
before writing any proposal.

## Run summary (context)
- Optimizer method: {method}
- Iterations recorded: {num_iterations}
- Held-out test score — seed: {format_score(seed_test_score)}, best candidate: \
{format_score(best_candidate_test_score)}

## What to read (read on demand — files are large; do not ask for inline content)
1. Cross-iteration memory:
   - `{mem}/memory.md` (the architect's own rolling memory + candidate pool)
   - `{mem}/iteration_change_summaries.md` (one concise change summary per iteration)
2. {trace_guide}
3. The task (client) repo at `{workspace}` — read `codeevolver.md` and the \
prompts/code the architect was actually steering, so proposals are grounded in \
what the meta-agent was working with.

## Produce (markdown):

### 1. Narrative analysis
Where did the architect spend its turns? What recurring failure or inefficiency \
patterns appear ACROSS iterations — e.g. repeated dead-end hypotheses, tool \
misuse, ignored memory, wasted evals, thrashing between explore/exploit, weak \
hypotheses? Cite specific trace evidence (iteration N + a short tool_use / \
tool_result / thinking excerpt) for each pattern.

### 2. Structured proposals
One block PER SURFACE below, and ONLY where the evidence supports a change. For \
each, output exactly these fields:
- **Surface**
- **Observed problem** (with trace citation: iteration N + excerpt)
- **Proposed change** (concrete — e.g. exact prompt language, a new tool field, a \
new control)
- **Expected effect**
- **Confidence** (low / medium / high)

Surfaces:
- **A. Architect system prompt** — the phases/guidelines the architect follows.
- **B. Skills** — `close_iteration`, `exploration`, or a missing skill.
- **C. Tool contracts** — `run_eval` / `coding_agent` / `inspect_traces` / \
`close_iteration`: signatures, descriptions, error messages, defaults.
- **D. Iteration controls** — watchdogs, the baseline/admission gate, \
eval-budget guidance, max change rounds.
- **E. Memory schema** — `memory.md` / `iteration_history` fields the architect \
under-used or that were missing.

Ground EVERY proposal in a specific trace citation. For any surface with no \
evidence, write exactly: "No evidence-backed change." Do not speculate beyond \
what the traces and memory show.
"""


# Attempts for one reflection-agent session and the backoff between them.
# Cheap next to the per-session timeout (40 turns x 120s). `_sleep` is a
# module attribute so tests can stub the wait.
REFLECTION_AGENT_ATTEMPTS = 3
REFLECTION_AGENT_BACKOFF_SECONDS = (5.0, 15.0)
_sleep = time.sleep


def run_reflection_agent(
    *,
    workspace: str,
    prompt: str,
    label: str,
    num_iterations: int,
    agent_model: str | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
    max_turns: int = 40,
    out_filename: str = "meta_reflection.md",
    out_path: str | None = None,
    run_as_agent_user: bool = True,
    allow_bash: bool = False,
) -> str | None:
    """Spawn the read-only reflection agent, write its markdown, log it.

    Prompt-agnostic and shared by both arms — ``label`` and ``num_iterations``
    are for log lines only. Returns the markdown written, or ``None`` on any
    failure (logged). STRICTLY non-fatal.

    ``out_path`` overrides the default ``{memory_dir()}/{out_filename}``
    destination — used by the run-integrity inspector's safe mode, which must
    write OUTSIDE memory_dir while a run is live (the architect can reach
    that tree).

    ``run_as_agent_user=False`` is the run-integrity inspector's privileged
    exception: inside the Docker runner (CODEEVOLVER_AGENT_USER set) agent
    processes run as an unprivileged user that cannot read the root-owned
    holdout, but the integrity inspector is explicitly allowed to read the
    testset. No effect when the env var is unset.

    ``allow_bash=True`` grants a Bash tool. The post-run auditors (meta-
    reflection, run-integrity) use it for read-only inspection — ``git log``,
    ``grep``, ``python3 -c`` over JSONL traces. They run after finalization
    with ``writable=False``; no mutation guard is layered on top (a commit
    could not change any recorded sha at that point).
    """
    try:
        # Imported inside the body on purpose: the unit tests monkeypatch
        # `subprocess_service.run_inspection_agent` as a module attribute, and
        # a call-time import is what makes that patch take effect.
        from src.engine.services.subprocess_service import (
            run_inspection_agent,
            write_file,
        )

        # Engine 3: the meta-reflection agent runs AFTER finalization — the
        # run is over, so no deny-path fences are armed (spec §System-checks:
        # meta-reflection needs no deny paths and may read architect traces).
        deny_paths: list[str] = []

        logger.info(
            "meta-reflection: starting (method=%s, %d iterations)",
            label,
            num_iterations,
        )
        # Bounded retry on the one failure shape seen in production: the
        # harness exited non-zero and emitted NOTHING (a non-zero exit WITH
        # output is still accepted below, as before). The sessions are
        # read-only, so re-running is idempotent. This is insurance for
        # transient harness failures only.
        analysis = ""
        error: Any = None
        for attempt in range(1, REFLECTION_AGENT_ATTEMPTS + 1):
            result = run_inspection_agent(
                workspace=workspace,
                prompt=prompt,
                max_turns=max_turns,
                writable=False,
                deny_paths=deny_paths,
                model=agent_model,
                coding_harness=coding_harness,
                coding_harness_model=coding_harness_model,
                run_as_agent_user=run_as_agent_user,
                allow_bash=allow_bash,
            )
            analysis = (result or {}).get("proposed_change") or ""
            error = (result or {}).get("error")
            if analysis.strip():
                break
            if attempt < REFLECTION_AGENT_ATTEMPTS:
                delay = REFLECTION_AGENT_BACKOFF_SECONDS[
                    min(attempt, len(REFLECTION_AGENT_BACKOFF_SECONDS)) - 1
                ]
                logger.warning(
                    "meta-reflection: agent produced no output on attempt %d/%d "
                    "(method=%s): %s -- retrying in %.0fs",
                    attempt, REFLECTION_AGENT_ATTEMPTS, label, error, delay,
                )
                _sleep(delay)
        if not analysis.strip():
            logger.error(
                "meta-reflection: agent produced no output after %d attempts "
                "(method=%s): %s",
                REFLECTION_AGENT_ATTEMPTS, label, error,
            )
            return None

        out_path = out_path or os.path.join(memory_dir(), out_filename)
        write_file(out_path, analysis)
        logger.info("meta-reflection: wrote proposals to %s", out_path)
        logger.info("meta-reflection: proposals\n%s", analysis)
        return analysis
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta-reflection failed (non-fatal): %s", exc)
        return None


def run_meta_reflection(
    *,
    workspace: str,
    method: str,
    num_iterations: int,
    seed_test_score: float | None,
    best_candidate_test_score: float | None,
    iteration_indices: list[int],
    agent_model: str | None = None,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str | None:
    """Run the post-run meta-reflection agent once. STRICTLY non-fatal.

    Returns the markdown that was written, or ``None`` on any failure (logged).
    Writes the result to ``{memory_dir()}/meta_reflection.md`` (a
    CodeEvolver-owned dir, outside the client repo — never git-committed).
    """
    try:
        prompt = build_meta_reflection_prompt(
            method=method,
            num_iterations=num_iterations,
            seed_test_score=seed_test_score,
            best_candidate_test_score=best_candidate_test_score,
            workspace=workspace,
            iteration_indices=iteration_indices,
        )
        return run_reflection_agent(
            workspace=workspace,
            prompt=prompt,
            label=method,
            num_iterations=num_iterations,
            agent_model=agent_model,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
            allow_bash=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("meta-reflection failed (non-fatal): %s", exc)
        return None


__all__ = [
    "architect_trace_reading_guide",
    "build_meta_reflection_prompt",
    "format_score",
    "run_meta_reflection",
    "run_reflection_agent",
]
