"""Copyright © 2026 440 Labs LLC

Post-run meta-reflection for the simple loop (research Baseline 0).

Same mechanism as `meta.meta_reflection` — one read-only advisory agent, once,
after finalization — with a different prompt document, because the simple
loop does not expose the CodeEvolver surfaces (`memory.md`, skills,
`inspect_traces`, `close_iteration`, `codeevolver.md`). Pointing an agent at
files that never exist is the classic setup for confident fabrication.

Two properties are load-bearing and unit-tested:

1. Research purity. The simple loop is the study's control arm; the absence
   of memory / skills / trace tooling / a candidate pool IS the variable. The
   prompt says so, because otherwise a competent agent proposes exactly the
   treatment arm on every run.
2. Nothing readable during the run. Run context reaches the agent INLINE as a
   markdown table, never via a file, so it cannot exist while a session runs.
"""

from __future__ import annotations

import logging
import os

from src.engine.optimizer.meta.meta_reflection import (
    architect_trace_reading_guide,
    format_score,
    run_reflection_agent,
)
from src.engine.paths import engine_root, traces_dir

logger = logging.getLogger(__name__)


def _cell(value: object, limit: int = 40) -> str:
    """One markdown table cell: flattened, pipe-escaped, truncated."""
    if value is None or value == "":
        return "—"
    text = " ".join(str(value).split()).replace("|", "\\|")
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _run_table(history: dict[int, dict]) -> str:
    """Render the whole run as one markdown table (inline, never a file)."""
    header = (
        "| session | start sha | end sha | rows | baseline | best train | Δ | "
        "best comparison | changed | constraint violation | error |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    rows = []
    for i in sorted(history):
        r = history[i] or {}
        base, best = r.get("initial_subsample_score"), r.get("best_subsample_score")
        delta = (
            format_score(best - base)
            if isinstance(base, (int, float)) and isinstance(best, (int, float))
            else "—"
        )
        rows.append(
            f"| {i}{' (seed)' if i == 0 else ''} "
            f"| {_cell((r.get('parent_sha') or '')[:8])} "
            f"| {_cell((r.get('close_sha') or '')[:8])} "
            f"| {_cell(r.get('eval_rows_used'))} "
            f"| {format_score(base)} | {format_score(best)} | {delta} "
            f"| {format_score(r.get('comparison_best_score'))} "
            f"| {'yes' if r.get('changes_landed') else 'no'} "
            f"| {_cell(r.get('constraint_violation'), 60)} "
            f"| {_cell(r.get('subsample_error'), 60)} |"
        )
    return "\n".join([header, *rows]) if rows else "(no sessions recorded)"


def build_simple_loop_meta_reflection_prompt(
    *,
    workspace: str,
    history: dict[int, dict],
    iteration_indices: list[int],
    branch: str | None,
    program_path: str | None,
    seed_test_score: float | None,
    final_test_score: float | None,
    subsample_size: int | None,
    max_eval_rows: int | None,
    total_evals: int | None,
) -> str:
    """Build the simple-loop meta-reflection prompt (structured proposals)."""
    ce_root = engine_root()
    req_spec = os.path.join(ce_root, "specs", "requirements.md")
    loop_spec = os.path.join(ce_root, "specs", "v3_simple_loop.md")
    loop_src = os.path.join(ce_root, "src", "engine", "optimizer", "simple_loop")
    trc = traces_dir()
    specs_present = os.path.exists(req_spec) and os.path.exists(loop_spec)
    specs_note = (
        ""
        if specs_present
        else (
            " NOTE: the specs/ directory is NOT present in this environment — "
            "skip the specs and rely on the source under src/engine plus the "
            "run artifacts."
        )
    )
    trace_guide = architect_trace_reading_guide(iteration_indices)
    run_table = _run_table(history)
    budget_line = (
        f"{total_evals if total_evals is not None else 'N/A'} / "
        f"{max_eval_rows if max_eval_rows is not None else 'unbounded'}"
    )
    last_best = next(
        (history[i].get("best_subsample_score") for i in sorted(history, reverse=True)
         if (history[i] or {}).get("best_subsample_score") is not None),
        None,
    )

    return f"""You are CodeEvolver's META-REFLECTION agent. A SIMPLE-LOOP optimization run \
just finished. Your job is ADVISORY ONLY: propose improvements to the simple \
loop's own machinery, grounded in evidence from THIS run. You do not modify any \
files — a human reviews your output. Report only; do not attempt to edit code, \
prompts, or memory.

You MUST NOT read or open any held-out test data. It is fenced. Reason only from \
CodeEvolver's own files, the agent's own traces, the eval traces, and the client \
repo.

## READ THIS FIRST — what is in scope
This run is a simple loop, by design. The goal is NOT to change that strategy. \
What's in scope is suggesting fixes for actual failure modes and limitations in \
how the simple loop is *implemented* — bugs, true lack of clarity in prompt \
wording, a control that misfires, a tool contract that misleads the agent — that \
you find evidence of in this run. Adding capability (memory, skills, \
trace-inspection tooling, a candidate pool) is out of scope: that absence IS the \
point of this control arm, and proposing additions would muddy the comparison \
the simple loop exists to make.

## FIRST — study the simple loop itself
You CAN see the CodeEvolver repository — it lives at `{ce_root}`. Read these by \
absolute path (Read / Grep, on demand — do not ask for inline content):
- `{req_spec}` and `{loop_spec}` — the authoritative project spec and the \
authoritative spec for THIS arm.{specs_note}
- The loop source you will propose changes to, under `{loop_src}/`:
  - `prompt.md` — the ENTIRE system prompt the agent ran under, and \
`prompt.py` (`build_kickoff_message`) — the per-session kickoff message.
  - `session.py` — the only tool the agent had (`run_eval`); its \
implementation lives in `v3/run_eval_tool.py`.
  - `constraint_check.py` — the post-session constraint gate and its revert.
  - `loop.py` — budget and termination semantics: an eval-row budget only, no \
iteration cap, the session-end commit, the stall counter, the soft deadline.
Grounding proposals in this REAL source — exact prompt language, real tool \
signatures, the actual control logic — is what makes them actionable.

## Run summary (context)
- Optimizer method: simple_loop (one linear branch `{branch}`; each session \
starts from the previous session's final HEAD; no candidate pool, no admission \
gate, no validation set)
- Sessions recorded: {len(history)} (iteration 0 is the seed; no agent session \
ran for it)
- Eval rows spent: {budget_line}; subsample size: \
{subsample_size if subsample_size is not None else 'N/A'}
- Last recorded best training-subsample score: {format_score(last_best)}
  NOTE: this is IN-SAMPLE — the agent's own in-budget training subsample, NOT a \
holdout. Do not read it as generalization.
- Held-out test score — seed: {format_score(seed_test_score)}, final program: \
{format_score(final_test_score)}

## The run, session by session
{run_table}

Legend: `changed=no` means the session landed no code change at all (a very \
informative failure mode). `best train` is the best training-subsample reading \
of the session regardless of what was kept; `best comparison` is the best \
fixed-comparison-set reading. A session with a `constraint violation` was \
reverted by the post-session gate: its `best train` / `best comparison` are \
readings on the discarded tree and do not count as the run's score.

## What to read (read on demand — files are large; do not ask for inline content)
1. {trace_guide}
2. The EVAL traces the agent could have read itself: \
`{trc}/iteration_<N>_parent/` (baseline), `{trc}/iteration_<N>_child/` \
(post-change) and `{trc}/iteration_<N>_comparison/` — each an `index.jsonl` plus \
`rows/<example_id>/row.json` and `span_NNN.json` files, and a `provenance.json` \
naming the eval (id / sha / tree) that last wrote the directory — each directory \
is rewritten by every eval with that suffix, so check the stamp before assuming \
an older directory describes the program the agent inherited. This arm has no \
`inspect_traces` tool — `prompt.md` tells the agent these directories exist and to \
read them with Bash / Read / Grep. So there is a directly answerable question here: \
**did the agent ever actually read its own eval traces, and when it did, did that \
change what it did next?** Answer it by name.
3. Git history in `{workspace}` — `git log --oneline {branch}` plus per-session \
`git diff <start sha>..<end sha>`. The git DAG IS the cross-session record; the \
table above gives you every sha.
4. The client repo at `{workspace}` — the program under optimization is \
`{program_path}`. Read its current state and the diffs that produced it.

## Produce (markdown):

### 1. Narrative analysis
Where did the agent spend its turns and its eval budget? What recurring failure \
or inefficiency patterns appear ACROSS sessions — e.g. repeated dead-end \
hypotheses that a later session re-tried from scratch, regressions carried \
forward, eval rows burned on redundant measurements, or the agent ignoring its \
own before/after numbers? Cite specific trace evidence (session N + a short \
tool_use / tool_result / thinking excerpt) for each pattern.

### 2. Structured proposals
One block PER SURFACE below, and ONLY where the evidence supports a change. For \
each, output exactly these fields:
- **Surface**
- **Observed problem** (with trace citation: session N + excerpt)
- **Proposed change** (concrete — e.g. exact prompt language, a new tool field, a \
new control)
- **Expected effect**
- **Confidence** (low / medium / high)

Surfaces:
- **A. Simple-loop system prompt** — the exact language of `prompt.md` and the \
per-session kickoff message. Frame any proposal as fixing a defect or ambiguity, \
never as adding capability.
- **B. Tool contracts** — `run_eval`: signature, description, error messages, \
defaults. Specifically: did the agent understand that the FIRST train call locks \
the row ids and every later call silently reuses them, and that `comparison` is \
the fixed cross-session set?
- **C. Loop controls** — the session-end commit, the post-session constraint \
gate and its revert, the stall counter, the soft deadline.
- **D. Budget guidance** — the budget note's wording, `subsample_size` versus \
the training pool size, and whether rows were burned on redundant evals or the \
program was left under-measured.
- **E. Diagnostics / observability** — what an operator simply COULD NOT answer \
from this run's artifacts. This is the one surface where "add an artifact" is a \
legitimate proposal, because anything written at finalization is invisible to the \
run and therefore cannot change what the arm measures.

Ground EVERY proposal in a specific trace citation. For any surface with no \
evidence, write exactly: "No evidence-backed change." Do not speculate beyond \
what the traces and the run table show.
"""


def run_simple_loop_meta_reflection(
    *,
    workspace: str,
    history: dict[int, dict],
    branch: str | None = None,
    program_path: str | None = None,
    seed_test_score: float | None = None,
    final_test_score: float | None = None,
    subsample_size: int | None = None,
    max_eval_rows: int | None = None,
    total_evals: int | None = None,
    agent_model: str | None = None,
    coding_harness: str = "opencode",
    coding_harness_model: str | None = None,
) -> str | None:
    """Run the simple-loop post-run meta-reflection once. STRICTLY non-fatal.

    Returns the markdown written, or None on any failure (logged). Output
    lands at `{memory_dir()}/meta_reflection.md`. Iteration 0 runs no agent
    session, so it is excluded from the trace indices.
    """
    iteration_indices = [i for i in sorted(history) if i >= 1]
    prompt = build_simple_loop_meta_reflection_prompt(
        workspace=workspace,
        history=history,
        iteration_indices=iteration_indices,
        branch=branch,
        program_path=program_path,
        seed_test_score=seed_test_score,
        final_test_score=final_test_score,
        subsample_size=subsample_size,
        max_eval_rows=max_eval_rows,
        total_evals=total_evals,
    )
    return run_reflection_agent(
        workspace=workspace,
        prompt=prompt,
        label="simple_loop",
        num_iterations=len(history),
        agent_model=agent_model,
        coding_harness=coding_harness,
        coding_harness_model=coding_harness_model,
    )


__all__ = [
    "build_simple_loop_meta_reflection_prompt",
    "run_simple_loop_meta_reflection",
]
