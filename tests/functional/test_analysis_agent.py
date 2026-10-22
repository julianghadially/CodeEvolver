"""Integration test for the engine 3 per-iteration analysis agent.

Replaces ``test_valset_reflection_v2.py``: engine 3 removed the validation
set and the valset-reflection agent. The architect now maintains
``memory.md`` itself, and the post-iteration agent's single job is the
concise ``iteration_{N}_change_summary.md`` (used to build
``iteration_change_summaries.md``), driven by
``prompts.build_analysis_agent_prompt``.

Engine 3 removed the per-iteration ``iteration_history_{N}.md`` files — the
analysis agent receives the iteration's objective record INLINE in its prompt
(branch, parent/final shas, scores, change_request) and grounds deeper detail
in the child trace file. A golden-nugget token is planted only in the child
trace (subsample error entries). Runs the production analysis-agent path —
``subprocess_service.run_inspection_agent`` in ``writable=True`` mode with
``build_analysis_agent_prompt`` (mirroring the loop's ``_run_analysis_agent``
hook) — and asserts:

1. the agent returned ``success=True``;
2. the agent wrote ``/memory/iteration_5_change_summary.md`` and it carries
   the golden-nugget signal (the literal token or a faithful paraphrase of
   iteration 5's failure), proving it grounded in the child trace;
3. ``/memory/memory.md`` was NOT modified — the engine 3 prompt forbids the
   analysis agent from touching memory.md (the architect owns that file);
4. a GPT-5.4-mini judge confirms the change summary is coherent (skipped
   when ``OPENAI_API_KEY`` is unset).

To run::

    python local_test_app.py analysis_agent
"""

from __future__ import annotations

import json
import os
import textwrap

from src.engine.optimizer.v3.prompts import build_analysis_agent_prompt
from tests._test_env import SyntheticTestEnvironment, parse_marker, seed_eval_trace_dir


VERIFICATION_TOKEN = "CodeEvolverValueError"

# Light LLM judge for the per-iteration change summary. GPT-5.4-mini is cheap
# and adequate for a coherence check; it is independent of the harness model
# the analysis agent ran on.
CHANGE_SUMMARY_JUDGE_MODEL = "openai/gpt-5.4-mini"


def judge_change_summary_coherent(change_summary: str) -> tuple[bool | None, str]:
    """LLM-judge that the generated change summary is coherent and on-topic.

    Returns ``(verdict, reason)``. ``verdict`` is:
    - ``False`` when the summary is missing/empty (no generation happened) or
      the judge rules it incoherent;
    - ``True`` when the judge confirms a coherent, on-topic iteration summary;
    - ``None`` when the judge cannot run (``OPENAI_API_KEY`` unset) or its
      output cannot be parsed — the caller treats ``None`` as "skip", not
      "fail", so the assertion never goes flaky on infra.

    Mirrors the litellm call shape in ``test_constraint_checker._litellm_callable``.
    """
    if not change_summary or not change_summary.strip():
        return False, "iteration_5_change_summary.md was missing or empty"
    if not os.environ.get("OPENAI_API_KEY"):
        return None, "OPENAI_API_KEY unset — change-summary coherence judge skipped"

    prompt = (
        "You are evaluating an automatically generated \"change summary\" for "
        "one iteration of an AI-program optimization run. A good change summary "
        "is a concise, coherent, on-topic executive summary (roughly 10-200 "
        "words) of the code or prompt change attempted that iteration. It "
        "must read as meaningful English about a concrete change — not empty, a "
        "placeholder, a refusal, gibberish, or unrelated boilerplate.\n\n"
        "Ground truth of what the iteration actually did: iteration 5 added "
        "chain-of-thought reasoning to the evidence summarizer of a "
        "fact-checking pipeline; it failed because the CoT wrapper dropped the "
        "'stance' field the downstream judge depended on.\n\n"
        "Change summary to evaluate:\n"
        '"""\n'
        f"{change_summary.strip()}\n"
        '"""\n\n'
        "Respond with ONLY a JSON object: "
        '{"coherent": true|false, "reason": "<one sentence>"}. '
        "Set coherent=true if it is a coherent, on-topic summary of an "
        "iteration change (it need not mention every detail). Set "
        "coherent=false if it is empty, gibberish, a placeholder, a refusal, "
        "or clearly unrelated to a change in this program."
    )

    try:
        import litellm  # type: ignore

        resp = litellm.completion(
            model=CHANGE_SUMMARY_JUDGE_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp["choices"][0]["message"]["content"] or "").strip()
    except Exception as exc:  # noqa: BLE001 — infra hiccup must not fail the run
        return None, f"change-summary judge call failed: {exc}"

    # Tolerate ```json fences / surrounding prose by extracting the JSON object.
    raw = text
    if "{" in raw and "}" in raw:
        raw = raw[raw.index("{"): raw.rindex("}") + 1]
    try:
        verdict = json.loads(raw)
        coherent = bool(verdict.get("coherent"))
        reason = str(verdict.get("reason", "")).strip() or "(no reason given)"
        return coherent, reason
    except Exception:  # noqa: BLE001 — unparseable → skip, don't false-fail
        return None, f"could not parse judge output: {text[:200]!r}"


def summarize_result(result: dict) -> str:
    """Compact one-line summary of the inspection-agent result.

    The raw ``result['result']`` carries ``raw_provider_metadata.tool_calls``
    (every tool call's full inputs and outputs) — hundreds of lines. The
    entrypoints print this instead so the console stays readable.
    """
    r = result.get("result", {}) or {}
    meta = r.get("raw_provider_metadata", {}) or {}
    tool_calls = meta.get("tool_calls") or []
    return (
        f"success={r.get('success')} "
        f"proposed_change={r.get('proposed_change')!r} "
        f"error={r.get('error')!r} "
        f"provider={meta.get('provider')} "
        f"returncode={meta.get('returncode')} "
        f"tool_calls={len(tool_calls)} "
        f"session_id={r.get('session_id')}"
    )


def summary_surfaced_nugget(change_summary: str) -> bool:
    """True if the change summary proves the agent grounded in iteration 5's data.

    Accepts the literal verification token OR a faithful paraphrase of
    iteration 5's failure. A capable model writing a polished 10-200 word
    summary routinely paraphrases the exact exception name away, so the
    literal token alone is brittle. The CoT / chain-of-thought signal is the
    iteration-5-distinctive marker — it appears nowhere in the seed
    codeevolver.md — and pairing it with ``stance`` confirms the agent
    captured the actual failure (the CoT wrapper dropping the judge's
    required stance field), not just the architecture title.
    """
    if VERIFICATION_TOKEN in change_summary:
        return True
    m = change_summary.lower()
    return "stance" in m and ("chain-of-thought" in m or "cot" in m)


# Provider API keys forwarded into the driver process / .env. ANTHROPIC for the
# claude harness, OPENAI for the opencode pass. Only keys present in the
# environment are forwarded.
_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY")


def _provider_env_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in _PROVIDER_ENV_VARS if os.environ.get(k)}


# Realistic seed-step architecture doc. The analysis-agent prompt opens with
# "Read {workspace}/codeevolver.md to understand the system architecture", so
# a believable doc (not a self-announced stub) grounds the agent's summary in
# the actual module names that appear in the iteration history and traces
# (ClaimExtractor → ResearchAgent → FireJudge → Aggregator). Mirrors the
# FACTCHECKER_CODEEVOLVER_MD fixture in test_inspect_traces_architect.py and
# the real seed doc shape (build_codeevolver_md_fallback_body).
FACTCHECKER_CODEEVOLVER_MD = """\
PARENT_MODULE_PATH: src.factchecker.simple.factchecker_program:FactCheckerProgram
METRIC_MODULE_PATH: src.codeevolver.metric:metric

## ARCHITECTURE TITLE: Baseline FactChecker — extract, research, judge, aggregate
## ARCHITECTURE SUMMARY:
FactCheckerProgram verifies free-text statements end to end. It runs four
modules in sequence: ClaimExtractor isolates the checkable claim(s) from the
input statement; ResearchAgent retrieves supporting or refuting evidence for
each claim (dense top-k retrieval over the evidence corpus, then an evidence
summarizer that condenses the passages into a stance-bearing summary); FireJudge
weighs each claim against its evidence summary and emits a per-claim stance
("supported", "refuted", or "insufficient"); Aggregator combines the per-claim
stances into a single verdict. The program is scored by the `accuracy` metric
against the dataset's gold verdict labels.

## ARCHITECTURE DESCRIPTION:
The program is a four-stage retrieval-augmented fact-checking pipeline.

1. ClaimExtractor — takes the raw `statement` and returns the normalized
   claim(s) to verify. Output shape downstream stages depend on: a list of
   claim strings.
2. ResearchAgent — for each claim, retrieves the top-k evidence passages and
   runs an evidence summarizer that returns an explicit `stance` field plus a
   short rationale. This is the only stage that depends on data from outside
   the program, so the quality of the final verdict is bounded by the evidence
   this stage surfaces and how faithfully the summarizer preserves it.
3. FireJudge — given a claim and its evidence summary, decides whether the
   evidence supports, refutes, or is insufficient for the claim. Consumes the
   summarizer's `stance` field directly.
4. Aggregator — reduces the per-claim stances to a single verdict for the
   statement.

Data flow: statement → ClaimExtractor → ResearchAgent (retrieve + summarize)
→ FireJudge → Aggregator → verdict. The primary metric, `accuracy`, compares
the aggregated verdict against each row's gold label. Rows that come back
"insufficient" generally score poorly, so the evidence-gathering and summarizer
stages are the usual leverage points when accuracy is low. FireJudge reads the
per-claim `stance` field produced by the ResearchAgent summarizer.
"""


# Engine 3 memory.md seed — architect-maintained (Candidate Pool section, no
# checkpoint marker). The analysis agent must NOT modify this file; the test
# asserts it is byte-identical after the run.
INITIAL_MEMORY = """\
# CodeEvolver Memory

Run id: test-analysis-agent
Seed branch: codeevolver-20260419-main
Seed architecture: Baseline FactChecker Pipeline (ClaimExtractor → \
ResearchAgent → FireJudge → Aggregator)

## Candidate Pool
- codeevolver-20260419-111111 (PRIMARY): one-claim-per-line extractor, \
subsample 2.6
- codeevolver-20260419-main: seed program, subsample 2.4

## Overall Progress
- Iteration 0 (Seed Program — Baseline FactChecker Pipeline): subsample \
score 2.4000 (parent baseline).
- Iteration 1 (One-claim-per-line extractor): subsample 2.4000 → 2.6000 \
(exploitation, parented off the seed). Current primary.

## Observations

### What Works
1. **Splitting compound statements into one claim per line helps \
downstream**: Reworking ClaimExtractor to emit exactly one normalized claim \
per line gave the first gain (+2pp subsample over the seed). Improvement \
showed on 2 of 4 subsample rows (the multi-claim ones); single-claim rows \
were unchanged, which is the expected, non-overfit signature.
2. **'insufficient' is the dominant low-score verdict**: rows that \
aggregate to 'insufficient' score near zero. The leverage is almost always \
in evidence quality (retrieval + summarization), not in the judge or \
aggregator wording.

## Macro Approaches
### Iteration 1 — One-claim-per-line extraction (incremental, not a macro \
approach)
- Approach: Reworked ClaimExtractor to emit one normalized claim per line.
- Result: Accepted; subsample improved 2.4 → 2.6.
- Observation: Cleaner per-claim units helped downstream research/judging.
- Remaining opportunity: Retrieval re-ranking and evidence-summarization \
changes are the obvious next distinct directions.
"""


# Iteration record for iteration 5 — the one that carries the golden nugget.
# Engine 3 field set (closing
# metadata instead of candidate_idx / valset fields) — fed inline into the prompt.
DUMMY_RECORD_5 = {
    "git_branch": "codeevolver-20260419-a1b2c3",
    "selected_component": "_code",
    "change_request": (
        "Add chain-of-thought reasoning to the evidence summarizer so it "
        "produces an explicit stance field."
    ),
    "architecture_title": "CoT Evidence Summarizer",
    "architecture_summary": (
        "Added chain-of-thought reasoning before evidence stance classification."
    ),
    "parent_subsample_score": 2.8,
    "initial_subsample_score": 2.8,
    "subsample_score": 0.0,
    "subsample_error": (
        f"{VERIFICATION_TOKEN}: Evidence summarizer returned empty stance "
        "field on 4/4 subsample examples. The chain-of-thought wrapper "
        "overwrote the output dict with its reasoning trace and dropped the "
        "'stance' field the downstream judge depended on."
    ),
    "skipped": False,
    "final_sha": "deadbeef1234",
    "close_sha": "deadbeef1234",
    "parent_branch": "codeevolver-20260419-333333",
    "parent_sha": "a333",
    "parent_iteration": 3,
    "parent_is_primary": False,
    "parent_derivation": "ancestor",
    "new_lineage": False,
    "improvement_mode": "exploitation",
    "memory_updated": True,
    "primary_branch_at_start": "codeevolver-20260419-111111",
    "primary_branch_after": "codeevolver-20260419-111111",
    "closed_by_fallback": False,
    "metrics_by_name": {"accuracy": 0.0},
    "watchdog_fired": None,
}


def _build_dummy_child_trace_rows() -> list[dict]:
    """Rows for /traces/iteration_5_child/ with a matching failure pattern.

    The analysis-agent prompt names this file so the agent can grep it for
    context. We populate it with four failed rows whose trace arrays carry
    the same ValueError the iteration-history record mentions — the agent can
    safely cross-reference without needing the real dataset.
    """
    rows = []
    for i in range(4):
        rows.append(
            {
                "example_id": i,
                "score": 0.0,
                "feedback": None,
                "failed": True,
                "input": {"statement": f"test claim {i}"},
                "output": {"verdict": "error"},
                "trace": [
                    {
                        "signature_key": "EvidenceSummarizer",
                        "inputs": {"statement": f"test claim {i}"},
                        "output": {
                            "__failed__": True,
                            "error": (
                                f"{VERIFICATION_TOKEN}: Evidence summarizer "
                                "returned empty stance field."
                            ),
                        },
                    }
                ],
            }
        )
    return rows


# The analysis agent carrying the golden-nugget signal into the change summary
# is model-driven: a capable model may compress the summary hard enough to
# drop the failure detail. That makes a single run inherently not 100%
# reliable, so the entrypoints retry up to this many times: a single miss is
# tolerated, and we only flag a warning when it takes all the way to the final
# attempt. Mirrors INSPECT_MAX_ATTEMPTS in test_inspect_traces_architect.
ANALYSIS_MAX_ATTEMPTS = 3


def analysis_agent_failures(result: dict) -> list[str]:
    """Return the list of assertion failures for one analysis run (empty == pass)."""
    failures: list[str] = []
    if not result["write_success"]:
        failures.append(
            f"agent did not report success (error={result['result'].get('error')!r})"
        )
    if not result["summary_written"]:
        failures.append("iteration_5_change_summary.md was not written.")
    if result["summary_written"] and not result["has_token"]:
        failures.append(
            f"iteration_5_change_summary.md neither contains "
            f"{VERIFICATION_TOKEN!r} nor a clear paraphrase of iteration 5's "
            "failure (CoT summarizer dropped the 'stance' field) — no "
            "evidence the agent grounded in the iteration's inline record "
            "or child trace"
        )
    if result["memory_modified"]:
        failures.append(
            "memory.md was modified — the engine 3 analysis agent must NOT "
            "touch memory.md (the architect owns that file)."
        )
    # change_summary_coherent is None when the judge was skipped (no
    # OPENAI_API_KEY) or its output was unparseable — treat as non-blocking.
    # Only a hard False (missing/empty/incoherent summary) is a failure.
    if result.get("change_summary_coherent") is False:
        failures.append(
            "iteration_5_change_summary.md was not a coherent summary of the "
            f"iteration's change (judge: {result.get('change_summary_judge_reason')!r})"
        )
    return failures


def run_analysis_agent(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the analysis-agent integration test against a `SyntheticTestEnvironment`.

    ``coding_harness`` selects the provider harness the analysis (inspection)
    agent runs on (`claude` or `opencode`); the runner sweeps both.
    Returns a dict suitable for pass/fail reporting by the entrypoint in
    ``local_test_app.py``:

    - ``write_success``: agent returned ``success=True``.
    - ``summary_written``: ``iteration_5_change_summary.md`` exists, non-empty.
    - ``has_token``: the summary carries the golden-nugget signal.
    - ``memory_modified``: ``memory.md`` changed (must be False).
    - ``change_summary`` / ``memory_md``: final file contents.
    - ``result``: raw ``run_inspection_agent`` return dict.
    """
    provider_keys = _provider_env_keys()

    try:
        # Workspace + .env (load_env_file reads this; worker secrets also
        # inject provider keys, but writing the file keeps parity with
        # the production path).
        env.seed_file(
            f"{env.workspace_dir}/.env",
            "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
        )

        # Architecture doc — the analysis-agent prompt's first step reads
        # `{workspace}/codeevolver.md`; seed a realistic one so the agent
        # grounds its summary in the real module names.
        env.seed_file(
            f"{env.workspace_dir}/codeevolver.md", FACTCHECKER_CODEEVOLVER_MD
        )

        # Memory file (the analysis agent must leave it untouched).
        env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY)

        # Trace directories the analysis-agent prompt names. The prompt's
        # `child_trace_dir` parameter takes any absolute path; we point
        # it at the env's traces_dir to stay agnostic to the environment.
        child_trace_dir = f"{env.traces_dir}/iteration_5_child"
        parent_trace_dir = f"{env.traces_dir}/iteration_5_parent"
        seed_eval_trace_dir(env, child_trace_dir, _build_dummy_child_trace_rows())

        prompt = build_analysis_agent_prompt(
            iteration_num=5,
            workspace_path=env.workspace_dir,
            branch=DUMMY_RECORD_5["git_branch"],
            parent_sha=DUMMY_RECORD_5["parent_sha"],
            close_sha=DUMMY_RECORD_5["final_sha"],
            change_request=DUMMY_RECORD_5["change_request"],
            subsample_baseline=DUMMY_RECORD_5["initial_subsample_score"],
            subsample_best=DUMMY_RECORD_5["subsample_score"],
            child_trace_dir=child_trace_dir,
            parent_trace_dir=parent_trace_dir,
            subsample_size=4,
        )

        # Ship the prompt as a workspace file so the driver stays short
        # regardless of prompt length.
        prompt_path = f"{env.workspace_dir}/analysis_prompt.txt"
        env.seed_file(prompt_path, prompt)

        # Inline driver invokes the production `run_inspection_agent` path
        # (writable=True, as the loop's `_run_analysis_agent` hook does) and
        # prints ANALYSIS_RESULT:{...} on stdout for the harness to parse.
        driver = textwrap.dedent(
            f"""\
            import json, os, sys
            sys.path.insert(0, {env.engine_root!r})
            os.chdir({env.engine_root!r})
            from src.engine.services.subprocess_service import (
                read_file, run_inspection_agent,
            )

            with open({prompt_path!r}) as f:
                prompt = f.read()

            memory_before = read_file({env.memory_dir!r} + "/memory.md") or ""

            result = run_inspection_agent(
                workspace={env.workspace_dir!r},
                prompt=prompt,
                max_turns=25,
                iteration_num=5,
                writable=True,
                deny_paths=["_holdout"],
                coding_harness={coding_harness!r},
                coding_harness_model={coding_harness_model!r},
            )
            memory = read_file({env.memory_dir!r} + "/memory.md") or ""
            change_summary = read_file(
                {env.memory_dir!r} + "/iteration_5_change_summary.md"
            ) or ""
            print("ANALYSIS_RESULT:" + json.dumps({{
                "result": result,
                "memory_md": memory,
                "memory_modified": memory != memory_before,
                "change_summary": change_summary,
            }}))
            """
        )

        stdout_lines: list[str] = []
        for line in env.exec_python_driver(
            driver, env_vars=provider_keys
        ):
            # The ANALYSIS_RESULT marker line carries the entire payload (full
            # memory.md + the agent's tool-call metadata) on one line — echoing
            # it floods the console and duplicates what the caller prints in a
            # compact form. Collect it, but don't echo it.
            if not line.startswith("ANALYSIS_RESULT:"):
                print(line, flush=True)
            stdout_lines.append(line)

        payload = parse_marker(stdout_lines, "ANALYSIS_RESULT")

        if payload is None:
            return {
                "result": {
                    "success": False,
                    "error": "No ANALYSIS_RESULT marker on stdout.",
                },
                "memory_md": "",
                "change_summary": "",
                "write_success": False,
                "summary_written": False,
                "has_token": False,
                "memory_modified": False,
                "change_summary_coherent": False,
                "change_summary_judge_reason": "no ANALYSIS_RESULT marker on stdout",
            }

        memory_md = payload.get("memory_md", "") or ""
        change_summary = payload.get("change_summary", "") or ""
        result = payload.get("result", {}) or {}

        coherent, judge_reason = judge_change_summary_coherent(change_summary)

        return {
            "result": result,
            "memory_md": memory_md,
            "change_summary": change_summary,
            "write_success": bool(result.get("success")),
            "summary_written": bool(change_summary.strip()),
            "has_token": summary_surfaced_nugget(change_summary),
            "memory_modified": bool(payload.get("memory_modified")),
            "change_summary_coherent": coherent,
            "change_summary_judge_reason": judge_reason,
        }
    finally:
        env.cleanup()
