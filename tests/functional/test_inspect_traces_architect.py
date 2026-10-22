"""Integration test for the v3 ``inspect_traces`` architect tool (Group D).

Proves the architect actually reaches for ``inspect_traces`` when the
compact ``trace_meta`` delivered inline via ``run_eval`` is insufficient,
and that what the sub-agent reports back materially shapes the
``change_request`` sent to ``coding_agent``.

Setup:
- Seed ``/memory/memory.md`` with a specific, useful hint (a local
  research-proxy URL) that only pays off if the architect figures out the
  root cause.
- Seed ``/traces/iteration_{N}_child/`` with a hidden FireCrawl
  secret-missing error buried inside the ``ResearchAgent`` trace entry,
  whose compact ``index.jsonl`` does NOT expose the
  error — only low scores.
- Stub ``run_eval``, ``coding_agent``, and ``close_iteration`` via an
  ``IterationArchitect`` subclass so the test harness does not need git
  or a real evaluator (engine 3: the context is locked at construction;
  there is no ``select_candidate``). **``inspect_traces`` is left REAL** —
  the whole point is to exercise it end-to-end through
  ``run_inspection_agent``.

Assertions:
- ``inspect_traces`` was invoked at least once.
- Captured ``change_request`` contains ``"firecrawl"`` (from the trace)
  OR the proxy URL (from ``memory.md``) — preferably both.
- ``context.coding_agent_called`` is True.

Real Claude Agent SDK sessions drive both the architect and the
inspection sub-agent. Expensive; run manually.

To run::

    python local_test_app.py inspect_traces --coding-harness claude
"""

from __future__ import annotations

import json
import os
import textwrap

from tests._test_env import (
    seed_eval_trace_dir,
    SyntheticTestEnvironment,
    git_fixture_snippet,
    parse_marker,
)


# Provider API keys forwarded into the driver process / .env. ANTHROPIC for the
# claude harness, OPENAI for the opencode pass (e.g. an openai/gpt-5.* model).
# Only keys actually present in the environment are forwarded; the runner is
# responsible for exporting the right one for the harness under test.
_PROVIDER_ENV_VARS = ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY")


def _provider_env_keys() -> dict[str, str]:
    return {k: os.environ[k] for k in _PROVIDER_ENV_VARS if os.environ.get(k)}


# --- Fixtures -----------------------------------------------------------

ITERATION_NUM = 5
SEED_BRANCH = "codeevolver-20260419000000-seed"

# Two golden-nugget markers the architect's ``change_request`` can hit.
# Either one is sufficient; both is ideal.
VERIFICATION_URL = "http://localhost:8742/research"
FIRECRAWL_TOKEN = "FIRECRAWL_API_KEY"


# ``/memory/memory.md`` seed. The proxy URL is the architect's reward for
# actually reading memory — if the change_request names it, we know the
# architect combined memory with whatever inspect_traces reported.
INITIAL_MEMORY_MD = f"""\
# CodeEvolver Memory

## Candidate Pool
- codeevolver-20260419000000-seed (PRIMARY): baseline pipeline

## Environment Configuration
A local web research proxy is available at {VERIFICATION_URL} that returns
free web search results as JSON. It accepts POST requests with a
``{{"query": "..."}}`` body and returns
``{{"results": [{{"title": ..., "snippet": ...}}]}}``. This proxy should be
used by the ResearchAgent when {FIRECRAWL_TOKEN} is not available.

## Observations
- Iteration 0: Baseline pipeline scored 0.62. Pipeline works end-to-end.
- Iteration 2: Research module works when {FIRECRAWL_TOKEN} is configured.
"""


# codeevolver.md for the architect's Phase 1, step 2 read. This must read
# like a REAL seed-step architecture doc, not a self-announced test stub.
#
# Earlier versions said "Stub file for the D1 inspect_traces integration
# test ... the real program isn't here." The architect read that, correctly
# concluded there was no real program to improve, and exited after appending
# a doc note — never inspecting the traces (which hold the real bug). So the
# test failed for the wrong reason: the fixture told the architect there was
# nothing to fix.
#
# This version mirrors the canonical seed format
# (`build_codeevolver_md_fallback_body` / the seed reflection prompt) and
# describes a believable four-module FactChecker matching the trace's
# `signature_key`s (ClaimExtractor → ResearchAgent → FireJudge → Aggregator).
# ResearchAgent is framed as the evidence-dependent leverage point so the
# architect's attention lands there — but the *reason* it is failing (the
# missing FireCrawl secret) is deliberately NOT mentioned here. That cause
# lives only in the trace, so the architect must call inspect_traces to find
# it. (memory.md carries the proxy hint as the payoff for combining the two.)
FACTCHECKER_CODEEVOLVER_MD = """\
PARENT_MODULE_PATH: factchecker.pipeline:FactCheckPipeline
METRIC_MODULE_PATH: factchecker.metrics:verdict_accuracy

## ARCHITECTURE TITLE: Baseline FactChecker — extract, research, judge, aggregate
## ARCHITECTURE SUMMARY:
FactCheckPipeline verifies free-text statements end to end. It runs four
modules in sequence: ClaimExtractor isolates the checkable claim from the
input statement; ResearchAgent gathers supporting or refuting evidence for
each claim; FireJudge weighs the claim against the gathered evidence and emits
a per-claim judgment; Aggregator combines the judgments into a final verdict
("supported", "refuted", or "insufficient evidence"). The pipeline is scored
by verdict_accuracy against the dataset's gold labels.

## ARCHITECTURE DESCRIPTION:
The program is a four-stage fact-checking pipeline.

1. ClaimExtractor — takes the raw `statement` and returns the normalized
   claim(s) to verify.
2. ResearchAgent — retrieves external evidence for each claim. This is the
   only stage that depends on data from outside the program, so the quality
   of the final verdict is bounded by the evidence this stage returns.
3. FireJudge — given a claim and its evidence, decides whether the evidence
   supports, refutes, or is insufficient for the claim.
4. Aggregator — reduces the per-claim judgments to a single verdict for the
   statement.

Data flow: statement → ClaimExtractor → ResearchAgent → FireJudge →
Aggregator → verdict. The primary metric, verdict_accuracy, compares the
aggregated verdict against each row's gold label. Rows that come back
"insufficient evidence" generally score poorly, so the evidence-gathering and
judgment stages are the usual leverage points when accuracy is low.
"""


# The program codeevolver.md describes, committed into the workspace. Without
# it the architect finds an empty repo, concludes the iteration is a stub
# (Claude: "This is a stubbed test iteration") and closes without changing
# anything. ResearchAgent hard-fails without the FireCrawl key, the same
# failure the child trace records, so reading the code is a valid way to ground
# the change request too (see `inspect_traces_failures`).
FACTCHECKER_PROGRAM_FILES = {
    "factchecker/__init__.py": "",
    "factchecker/modules/__init__.py": "",
    "factchecker/pipeline.py": '''\
import dspy

from factchecker.modules.aggregator import Aggregator
from factchecker.modules.claim_extractor import ClaimExtractor
from factchecker.modules.fire_judge import FireJudge
from factchecker.modules.research_agent import ResearchAgent


class FactCheckPipeline(dspy.Module):
    def __init__(self):
        super().__init__()
        self.extract = ClaimExtractor()
        self.research = ResearchAgent()
        self.judge = FireJudge()
        self.aggregate = Aggregator()

    def forward(self, statement: str) -> dspy.Prediction:
        claims = self.extract(statement=statement).claims
        evidence = self.research(claims=claims)
        judgments = [
            self.judge(claim=c, evidence=evidence.get(c, "No evidence available")).judgment
            for c in claims
        ]
        return self.aggregate(judgments=judgments)
''',
    "factchecker/modules/claim_extractor.py": '''\
import dspy


class ExtractClaims(dspy.Signature):
    """Split a statement into the atomic factual claims it makes."""

    statement: str = dspy.InputField()
    claims: list[str] = dspy.OutputField()


class ClaimExtractor(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.Predict(ExtractClaims)

    def forward(self, statement: str):
        return self.predict(statement=statement)
''',
    "factchecker/modules/research_agent.py": '''\
import os

import dspy
import requests

FIRECRAWL_SEARCH_URL = "https://api.firecrawl.dev/v1/search"


class ResearchAgent(dspy.Module):
    """Gathers web evidence for each claim via FireCrawl search."""

    def __init__(self, max_results: int = 5):
        super().__init__()
        self.max_results = max_results

    def forward(self, claims: list[str]) -> dict[str, str]:
        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return {
                "__failed__": True,
                "error": (
                    "FireCrawl API secret not found in environment. "
                    "Set FIRECRAWL_API_KEY to enable web scraping. The ResearchAgent "
                    "requires FireCrawl to gather evidence from the web."
                ),
            }
        evidence = {}
        for claim in claims:
            resp = requests.post(
                FIRECRAWL_SEARCH_URL,
                headers={"Authorization": f"Bearer {api_key}"},
                json={"query": claim, "limit": self.max_results},
                timeout=30,
            )
            resp.raise_for_status()
            hits = resp.json().get("data", [])
            evidence[claim] = "\\n".join(
                f"{h.get('title', '')}: {h.get('description', '')}" for h in hits
            )
        return evidence
''',
    "factchecker/modules/fire_judge.py": '''\
import dspy


class JudgeClaim(dspy.Signature):
    """Decide whether the evidence supports, refutes, or is insufficient for the claim."""

    claim: str = dspy.InputField()
    evidence: str = dspy.InputField()
    judgment: str = dspy.OutputField(desc="supported | refuted | insufficient")


class FireJudge(dspy.Module):
    def __init__(self):
        super().__init__()
        self.predict = dspy.ChainOfThought(JudgeClaim)

    def forward(self, claim: str, evidence: str):
        return self.predict(claim=claim, evidence=evidence)
''',
    "factchecker/modules/aggregator.py": '''\
import dspy


class Aggregator(dspy.Module):
    def forward(self, judgments: list[str]) -> dspy.Prediction:
        if judgments and all(j == "supported" for j in judgments):
            verdict = "supported"
        elif any(j == "refuted" for j in judgments):
            verdict = "refuted"
        else:
            verdict = "insufficient evidence"
        return dspy.Prediction(verdict=verdict)
''',
    "factchecker/metrics.py": '''\
def verdict_accuracy(example, prediction, trace=None) -> float:
    return float(example.label.strip().lower() == prediction.verdict.strip().lower())
''',
}


def _build_child_trace_rows() -> list[dict]:
    """Rows of ``/traces/iteration_{N}_child/`` with the FireCrawl error buried in trace[]."""
    statements = [
        "The Earth orbits the Sun",
        "Water boils at 100 degrees Celsius at sea level",
        "The Great Wall of China is visible from space",
    ]
    rows = []
    for i, stmt in enumerate(statements):
        rows.append(
            {
                "example_id": i,
                "score": 0.1,
                "feedback": None,
                "failed": False,
                "input": {"statement": stmt},
                "output": {"verdict": "insufficient evidence"},
                "trace": [
                    {
                        "signature_key": "ClaimExtractor",
                        "inputs": {"statement": stmt},
                        "output": {"claims": [stmt]},
                    },
                    {
                        "signature_key": "ResearchAgent",
                        "inputs": {"claims": [stmt]},
                        "output": {
                            "__failed__": True,
                            "error": (
                                f"FireCrawl API secret not found in environment. "
                                f"Set {FIRECRAWL_TOKEN} to enable web scraping. "
                                "The ResearchAgent requires FireCrawl to gather "
                                "evidence from the web."
                            ),
                        },
                    },
                    {
                        "signature_key": "FireJudge",
                        "inputs": {
                            "claim": stmt,
                            "evidence": "No evidence available",
                        },
                        "output": {"judgment": "insufficient"},
                    },
                    {
                        "signature_key": "Aggregator",
                        "inputs": {"judgments": ["insufficient"]},
                        "output": {"verdict": "insufficient evidence"},
                    },
                ],
            }
        )
    return rows



# NOTE: file seeding is delegated to `env.seed_file` from `tests/_test_env.py`.


# --- Architect prompt additions ----------------------------------------
# Do not provide additional guidance about using the Inspect tool. 
# We are not only testing the Inspect tool capability but also 
# the reliability of the architect agent calling the Inspect tool according 
# to the prompt instructions.

_ADDITIONAL_INSTRUCTIONS = f"""\
Note a few irregularities for this run:

1. ``run_eval(eval_sample="train")`` — you will see
   ``primary_score=0.1`` across 3 rows.
2. Stay on the current branch; do not create new branches or commit
   anything yourself.
3. You do not need to update memory.md for this iteration; once you have
   proposed and applied your change, call
   ``close_iteration(register_branch={{"branch":
   "codeevolver-20260419000000-seed", "was_primary": true,
   "is_primary": true}}, improvement_mode="exploitation",
   memory_updated=false)`` and exit.
"""


# --- Driver (runs inside the sandbox) ----------------------------------

def _build_driver_script(
    *,
    engine_root: str,
    workspace: str,
    traces_dir: str,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> str:
    """Driver: runs one iteration with REAL inspect_traces, stubs elsewhere.

    `engine_root` / `workspace` / `traces_dir` are baked into the driver so
    the same body works against container paths (`/app`, `/workspace`,
    `/traces`) and local tmpdirs (auto-detected engine root,
    `<tmp>/workspace`, `<tmp>/traces`).

    `coding_harness` selects the provider harness the architect session runs
    on (`claude` or `opencode`); `coding_harness_model` is the provider/model
    string OpenCode requires (ignored in Claude mode).
    """
    git_fixture = git_fixture_snippet(
        branch=SEED_BRANCH,
        message="Initial FactChecker pipeline",
        stage_all=True,
        ignore=(".env",),
    )
    return textwrap.dedent(
        f"""\
        import json, os, subprocess, sys
        sys.path.insert(0, {engine_root!r})
        os.chdir({engine_root!r})
        # Claude CLI refuses `--dangerously-skip-permissions` under
        # root/sudo unless IS_SANDBOX=1 is set. Production's
        # optimization_entry.py sets this; our driver bypasses that path,
        # so we must replicate it here or the SDK session won't boot.
        os.environ.setdefault("IS_SANDBOX", "1")

        from src.engine.optimizer.v3.iteration_architect import (
            ArchitectConfig, IterationArchitect,
        )
        from src.engine.optimizer.v3.iteration_context import IterationContext
        from src.engine.optimizer.v3.agent_tools import inspect_traces_tool
        from src.engine.optimizer.v3.prompts import build_trace_inspection_prompt
        from src.engine.evaluator.config import EvaluatorConfig
        from src.engine.services.subprocess_service import read_file

        ITERATION_NUM = {ITERATION_NUM}
        ADDITIONAL_INSTRUCTIONS = {_ADDITIONAL_INSTRUCTIONS!r}
        SEED_BRANCH = {SEED_BRANCH!r}
        WORKSPACE = {workspace!r}

        # Real repo so the close-time git hooks have something to read.
        # `.env` holds live provider keys and stays gitignored.
        {git_fixture}

        class _StubState:
            primary_branch = SEED_BRANCH

        class _TestArchitect(IterationArchitect):
            \"\"\"Stubs 3 tools; inspect_traces stays REAL end-to-end.\"\"\"

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                # Captured side effects for the test to assert on.
                self.inspect_traces_calls = []
                self.inspect_traces_summaries = []

            def build_tool_registry(self, *, agent_harness):
                ctx = self.context
                workspace = self.workspace

                # --- run_eval: stub (no evaluator subprocess) ---
                def run_eval(eval_sample, filter=None):
                    if ctx.initial_subsample_score is None:
                        ctx.initial_subsample_score = 0.1
                        ctx.subsample_ids = [0, 1, 2]
                    ctx.record_eval(START_SHA, 0.1)
                    return {{
                        "scores_by_metric": {{"accuracy": 0.1}},
                        "primary_score": 0.1,
                        "trace_dir": (
                            f"{traces_dir}/iteration_{{ITERATION_NUM}}_child"
                        ),
                        "trace_index_file": (
                            f"{traces_dir}/iteration_{{ITERATION_NUM}}_child/index.jsonl"
                        ),
                        "trace_meta": [
                            {{"example_id": i, "score": 0.1,
                              "failed": False, "feedback": None,
                              "error_summary": None}}
                            for i in range(3)
                        ],
                    }}

                # --- coding_agent: stub (captures the change_request) ---
                def coding_agent(mode, change_request):
                    ctx.coding_agent_called = True
                    ctx.change_requests.append(change_request)
                    return {{
                        "ok": True,
                        "success": True,
                        "summary": (
                            "Applied the requested change to "
                            "factchecker/modules/research_agent.py and "
                            "committed it."
                        ),
                    }}

                # --- inspect_traces: REAL, wrapped with a call counter ---
                # This mirrors iteration_architect.build_tool_registry's
                # inspect_traces closure exactly, then records calls and
                # summaries on the test subclass for later assertions.
                tester = self

                def inspect_traces(query, trace="child"):
                    full_prompt = build_trace_inspection_prompt(
                        query=query,
                        iteration_num=ctx.iteration_idx,
                        suffix="child",
                    )
                    tester.inspect_traces_calls.append(query[:500])
                    result = inspect_traces_tool(
                        ctx=ctx,
                        query=full_prompt,
                        workspace=workspace,
                        agent_harness=agent_harness,
                    )
                    summary = result.get("summary") or ""
                    tester.inspect_traces_summaries.append(summary[:1500])
                    return result

                # --- close_iteration: stub (accepts anything; marks the
                # iteration closed so the close-missing watchdog does not
                # resume the session — the closing protocol is not under
                # test here).
                def close_iteration(register_branch, improvement_mode,
                                    memory_updated):
                    ctx.registered_branch = (
                        register_branch if isinstance(register_branch, dict)
                        else {{}}
                    )
                    ctx.claimed_improvement_mode = improvement_mode
                    ctx.close_called = True
                    ctx.child_branch = SEED_BRANCH
                    return {{"ok": True}}

                return {{
                    "run_eval": run_eval,
                    "coding_agent": coding_agent,
                    "inspect_traces": inspect_traces,
                    "close_iteration": close_iteration,
                }}

        # Engine 3 iteration-start snapshot (the loop locks the context at
        # construction; there is no select_candidate).
        ctx = IterationContext(iteration_idx=ITERATION_NUM)
        ctx.parent_locked = True
        ctx.start_branch = SEED_BRANCH
        ctx.start_sha = START_SHA
        ctx.primary_branch_at_start = SEED_BRANCH

        evaluator = EvaluatorConfig(
            program_factory="stub_pkg.stub_mod:stub_factory",
            metric_fns=["stub_pkg.stub_mod:stub_metric"],
            metric_names=["accuracy"],
            primary_metric="accuracy",
            trainset_path="{workspace}/.codeevolver/trainset.json",
        )

        cfg = ArchitectConfig(
            run_timestamp="20260419000000",
            additional_instructions=ADDITIONAL_INSTRUCTIONS,
            program_path=None,
            evaluator=evaluator,
            trainset_ids=[0, 1, 2],
            max_turns=40,
            coding_harness={coding_harness!r},
            coding_harness_model={coding_harness_model!r},
        )

        architect = _TestArchitect(
            context=ctx,
            state=_StubState(),
            workspace={workspace!r},
            config=cfg,
        )
        # Wrap architect.run() so the marker always prints even if the SDK
        # session raises. Without this, any exception (SDK boot failure,
        # max_turns hit, etc.) exits the driver with code 1 and the outer
        # harness reports "No D1_RESULT marker" with no clue why.
        import traceback as _traceback
        run_error = None
        try:
            architect.run()
        except Exception as exc:  # noqa: BLE001
            run_error = f"{{type(exc).__name__}}: {{exc}}"
            _traceback.print_exc()

        # Capture the last change_request (architect may have called
        # coding_agent more than once).
        last_change_request = (
            ctx.change_requests[-1] if ctx.change_requests else ""
        )
        # The architect may also fix the program itself (Edit / Bash) instead
        # of delegating to coding_agent. Collect the lines its edits ADDED
        # since the seed commit, committed or not; context lines are excluded
        # because the seed's research_agent.py already names the FireCrawl key.
        subprocess.run(["git", "add", "-A"], cwd=WORKSPACE, capture_output=True)
        _diff = subprocess.run(
            ["git", "diff", "--cached", START_SHA, "--", "factchecker"],
            cwd=WORKSPACE, capture_output=True, text=True,
        ).stdout
        direct_edit_added = "\\n".join(
            l[1:] for l in _diff.splitlines()
            if l.startswith("+") and not l.startswith("+++")
        )
        joined_change_requests = "\\n\\n".join(ctx.change_requests)
        grounding_text = joined_change_requests + "\\n" + direct_edit_added
        lower = grounding_text.lower()

        payload = {{
            "inspect_traces_call_count": len(architect.inspect_traces_calls),
            "inspect_traces_queries": architect.inspect_traces_calls,
            "inspect_traces_summaries": architect.inspect_traces_summaries,
            "coding_agent_called": bool(ctx.coding_agent_called),
            "change_requests": list(ctx.change_requests),
            "last_change_request": last_change_request,
            "direct_edit": bool(direct_edit_added.strip()),
            "mentions_firecrawl": "firecrawl" in lower,
            "mentions_proxy_url": "{VERIFICATION_URL}" in grounding_text,
            "mentions_research_agent": "researchagent" in lower,
            "run_error": run_error,
        }}
        print("D1_RESULT:" + json.dumps(payload))
        """
    )


# --- Pass/fail criteria (shared by both entrypoints) ------------------

# The architect's decision to reach for ``inspect_traces`` is model-driven and
# only nudged by the system prompt (the per-run instructions intentionally do
# NOT name the tool). That makes it inherently not 100% reliable, so the
# entrypoints retry up to this many times: a single skip is tolerated, and we
# only flag a warning when it takes all the way to the final attempt.
INSPECT_MAX_ATTEMPTS = 3


def inspect_traces_failures(result: dict) -> list[str]:
    """Return the list of assertion failures for one D1 run (empty == pass).

    Calling `inspect_traces` is NOT required: the trace directory is directly
    readable (`rows/<id>/span_NNN.json`), so an architect that grounds its
    change request by reading it itself has done the job. A skipped tool is a
    warning (`inspect_traces_warnings`), not a failure — what must hold is
    that the cycle completed and the change carries the trace signal.

    The architect may make the change itself (engine3.md §3.3: it can edit
    files directly), so a direct edit to the program completes the cycle as
    well as a `coding_agent` call; the grounding markers are then read from
    the lines the edit added.
    """
    failures: list[str] = []
    changed = result["coding_agent_called"] or result.get("direct_edit", False)
    if not changed:
        failures.append(
            "no change was made (no coding_agent call, no direct edit) — "
            "architect did not complete the cycle."
        )
    if changed and not (
        result["mentions_firecrawl"] or result["mentions_proxy_url"]
    ):
        failures.append(
            f"the change mentions neither {FIRECRAWL_TOKEN!r} (from the trace) "
            f"nor {VERIFICATION_URL!r} (from memory.md)."
        )
    return failures


def inspect_traces_warnings(result: dict) -> list[str]:
    """Non-fatal observations for a passing D1 run."""
    warnings: list[str] = []
    if result["inspect_traces_call_count"] < 1:
        warnings.append(
            "inspect_traces was never invoked — the architect grounded itself "
            "by reading the trace directory directly instead of the sub-agent."
        )
    if not result["mentions_research_agent"]:
        warnings.append(
            "change_request does not name ResearchAgent (the failing module)."
        )
    return warnings


# --- Runner -----------------------------------------------------------

def run_inspect_traces_architect(
    env: SyntheticTestEnvironment,
    *,
    coding_harness: str = "claude",
    coding_harness_model: str | None = None,
) -> dict:
    """Run the D1 integration test against a `SyntheticTestEnvironment`.

    ``coding_harness`` selects the provider harness (`claude` or `opencode`);
    the runner sweeps both. Returns a dict suitable for pass/fail reporting by
    the entrypoint in ``local_test_app.py``:

    - ``coding_agent_called`` or ``direct_edit``: one True required.
    - ``mentions_firecrawl`` / ``mentions_proxy_url``: at least one True
      required (proves the architect reflected the trace signal or the
      memory hint in its change_request or in the lines its edit added).
    - ``inspect_traces_call_count``: reported; 0 is a warning, not a
      failure (the architect may read the trace directory itself).
    - ``mentions_research_agent``: reported; False is a warning.
    """
    provider_keys = _provider_env_keys()

    try:
        env.seed_file(
            f"{env.workspace_dir}/.env",
            "".join(f"{k}={v}\n" for k, v in provider_keys.items()),
        )
        env.seed_file(
            f"{env.workspace_dir}/codeevolver.md",
            FACTCHECKER_CODEEVOLVER_MD,
        )
        for rel, content in FACTCHECKER_PROGRAM_FILES.items():
            env.seed_file(f"{env.workspace_dir}/{rel}", content)
        env.seed_file(f"{env.memory_dir}/memory.md", INITIAL_MEMORY_MD)
        seed_eval_trace_dir(
            env,
            f"{env.traces_dir}/iteration_{ITERATION_NUM}_child",
            _build_child_trace_rows(),
        )

        driver = _build_driver_script(
            engine_root=env.engine_root,
            workspace=env.workspace_dir,
            traces_dir=env.traces_dir,
            coding_harness=coding_harness,
            coding_harness_model=coding_harness_model,
        )
        stdout_lines: list[str] = []
        for line in env.exec_python_driver(
            driver,
            env_vars=provider_keys,
        ):
            print(line, flush=True)
            stdout_lines.append(line)

        payload = parse_marker(stdout_lines, "D1_RESULT")

        if payload is None:
            tail = "\n".join(stdout_lines[-40:])
            return {
                "inspect_traces_call_count": 0,
                "inspect_traces_queries": [],
                "inspect_traces_summaries": [],
                "coding_agent_called": False,
                "change_requests": [],
                "last_change_request": "",
                "mentions_firecrawl": False,
                "mentions_proxy_url": False,
                "mentions_research_agent": False,
                "error": (
                    f"No D1_RESULT marker on stdout. "
                    f"Last stdout lines:\n{tail}"
                ),
            }

        payload.setdefault("error", payload.get("run_error"))
        return payload
    finally:
        env.cleanup()
