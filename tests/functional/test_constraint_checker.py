"""Tests for the constraint checker module."""

import json
import os
from pathlib import Path

import pytest

from src.engine.agent.constraint_checker import check_diff_constraints


def _litellm_callable(model: str):
    """Build a `Callable[[str], str]` over a litellm model string.

    Replaces the deleted ``optimizer.utils.get_reflection_lm_callable``;
    mirrors the litellm call shape in
    ``src/engine/optimizer/v3/seed.py::_generate_codeevolver_md``.
    """

    def _call(prompt: str) -> str:
        import litellm  # type: ignore

        resp = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp["choices"][0]["message"]["content"] or ""

    return _call

# ---------------------------------------------------------------------------
# Diff-aware checker (check_diff_constraints) — mock LM plumbing
# ---------------------------------------------------------------------------

_SAMPLE_DIFF = """\
diff --git a/program.py b/program.py
index 1111111..2222222 100644
--- a/program.py
+++ b/program.py
@@ -1,3 +1,4 @@
 def forward(self, question):
-    return self.predict(question)
+    docs = self.web_search(question)
+    return self.predict(question, docs)
"""


def test_diff_pass_when_no_violation():
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "pass",
            "reason": "No constraint violated.",
            "violated_constraint": None,
        })

    result = check_diff_constraints(
        additional_instructions="Do not call external web services.",
        diff="diff --git a/x.py b/x.py\n+    self.cache = {}\n",
        lm_callable=mock_lm,
    )
    assert result.status == "pass"
    assert result.violated_constraint is None


def test_diff_fail_when_constraint_violated():
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "fail",
            "reason": "Adds a web search call.",
            "violated_constraint": "Do not call external web services.",
        })

    result = check_diff_constraints(
        additional_instructions="Do not call external web services.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "fail"
    assert result.violated_constraint == "Do not call external web services."


def test_diff_coerces_list_violated_constraint_preserving_fail():
    """A model that returns violated_constraint as a list must still fail.

    Without coercion the pydantic build raises and the checker fails open
    (downgrades the real 'fail' to 'pass') — the regression this guards.
    """
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "FAIL",
            "reason": "Two constraints violated.",
            "violated_constraint": ["No network calls.", "No new deps."],
        })

    result = check_diff_constraints(
        additional_instructions="No network calls. No new deps.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "fail"  # casing normalized too
    assert "No network calls." in result.violated_constraint
    assert "No new deps." in result.violated_constraint


def test_diff_reconciliation_guard_fail_without_violated_constraint():
    """A 'fail' with no named violated_constraint is downgraded to pass.

    A checker can emit status='fail' while its own reasoning concludes
    nothing was violated and leaves `violated_constraint` empty. Honoring
    that force-reverts a legitimate
    iteration, so the reconciliation guard treats a self-contradictory verdict
    as pass (consistent with the gate's fail-open contract).
    """
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "fail",
            "reason": (
                "The diff does not change the LM (gpt-5.4-nano remains), and no "
                "other hard constraints are violated."
            ),
            "violated_constraint": None,
        })

    result = check_diff_constraints(
        additional_instructions="Do not change the language model.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "pass"
    assert result.violated_constraint is None
    assert "self-contradictory" in result.reason.lower()


def test_diff_reconciliation_guard_keeps_named_fail():
    """A 'fail' that names the violated constraint is preserved (no downgrade)."""
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "fail",
            "reason": "Adds a web search call.",
            "violated_constraint": "Do not call external web services.",
        })

    result = check_diff_constraints(
        additional_instructions="Do not call external web services.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "fail"
    assert result.violated_constraint == "Do not call external web services."


def test_diff_reconciliation_guard_blank_violated_constraint_downgrades():
    """A 'fail' whose violated_constraint is whitespace-only also downgrades."""
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "fail",
            "reason": "Something seemed off about the implementation.",
            "violated_constraint": "   ",
        })

    result = check_diff_constraints(
        additional_instructions="Do not change the language model.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "pass"
    assert result.violated_constraint is None


def test_diff_defaults_to_pass_on_llm_exception():
    def mock_lm(prompt: str) -> str:
        raise RuntimeError("LLM service unavailable")

    result = check_diff_constraints(
        additional_instructions="Do not change the API.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert result.status == "pass"
    assert "error" in result.reason.lower()


# ---------------------------------------------------------------------------
# Retry-once on an unusable first verdict (empty/unparseable OR fail-without-
# constraint). Mirrors the real Sonnet flake that fail-opened a >21-doc
# violation to pass.
# ---------------------------------------------------------------------------


def _sequenced_lm(*responses: str):
    """Return a mock lm_callable that yields each response in turn (then repeats last)."""
    calls = {"n": 0}

    def _lm(prompt: str) -> str:
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    _lm.calls = calls  # type: ignore[attr-defined]
    return _lm


def test_diff_retries_once_on_empty_response_then_fails():
    """An empty first response is retried; the valid 'fail' second response wins.

    This is the exact over_doc_limit regression: Sonnet returned empty content,
    which json.loads rejected and fail-opened to pass. The retry recovers it.
    """
    lm = _sequenced_lm(
        "",  # empty -> unusable -> retry
        json.dumps({
            "status": "fail",
            "reason": "Returns up to 50 documents.",
            "violated_constraint": "CANNOT return more than 21 documents.",
        }),
    )
    result = check_diff_constraints(
        additional_instructions="CANNOT return more than 21 documents.",
        diff=_SAMPLE_DIFF,
        lm_callable=lm,
    )
    assert result.status == "fail"
    assert "21 documents" in result.violated_constraint
    assert lm.calls["n"] == 2  # one retry


def test_diff_retries_once_on_fail_without_constraint_then_keeps_named_fail():
    """A first 'fail' with no constraint is retried; a named 'fail' second wins."""
    lm = _sequenced_lm(
        json.dumps({"status": "fail", "reason": "seems off", "violated_constraint": None}),
        json.dumps({
            "status": "fail",
            "reason": "Adds a web search call.",
            "violated_constraint": "Do not call external web services.",
        }),
    )
    result = check_diff_constraints(
        additional_instructions="Do not call external web services.",
        diff=_SAMPLE_DIFF,
        lm_callable=lm,
    )
    assert result.status == "fail"
    assert result.violated_constraint == "Do not call external web services."
    assert lm.calls["n"] == 2


def test_diff_retry_exhausted_fail_without_constraint_downgrades_to_pass():
    """If both attempts are fail-without-constraint, reconciliation still downgrades."""
    bad = json.dumps({"status": "fail", "reason": "nothing concrete", "violated_constraint": None})
    lm = _sequenced_lm(bad, bad)
    result = check_diff_constraints(
        additional_instructions="Do not change the language model.",
        diff=_SAMPLE_DIFF,
        lm_callable=lm,
    )
    assert result.status == "pass"
    assert result.violated_constraint is None
    assert lm.calls["n"] == 2


def test_diff_retry_exhausted_empty_defaults_to_pass():
    """Two empty responses -> fail-open pass with an 'after retry' reason."""
    lm = _sequenced_lm("", "")
    result = check_diff_constraints(
        additional_instructions="Do not change the API.",
        diff=_SAMPLE_DIFF,
        lm_callable=lm,
    )
    assert result.status == "pass"
    assert "retry" in result.reason.lower()
    assert lm.calls["n"] == 2


def test_diff_no_retry_on_clean_verdict():
    """A usable first verdict (clean pass) is returned without a second call."""
    lm = _sequenced_lm(
        json.dumps({"status": "pass", "reason": "ok", "violated_constraint": None})
    )
    result = check_diff_constraints(
        additional_instructions="Do not change the API.",
        diff=_SAMPLE_DIFF,
        lm_callable=lm,
    )
    assert result.status == "pass"
    assert lm.calls["n"] == 1  # no retry


def test_diff_prompt_contains_diff_and_constraints_and_change_summary():
    captured_prompt = None

    def mock_lm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"status": "pass", "reason": "ok", "violated_constraint": None})

    check_diff_constraints(
        additional_instructions="Never add network calls.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
        change_requests=["Add web search to the retrieval step"],
    )
    assert captured_prompt is not None
    assert "Never add network calls." in captured_prompt
    assert "web_search" in captured_prompt  # the diff body is included
    assert "Add web search to the retrieval step" in captured_prompt  # change summary


def test_diff_prompt_includes_outcome_reflection_context():
    """The architect's outcome reflection is forwarded into the prompt.

    The reflection often reveals what a diff actually does (e.g. that a local
    file it reads is a corpus export) — context the checker needs to catch
    non-obvious violations. Verify it reaches the prompt alongside the change
    request.
    """
    captured_prompt = None

    def mock_lm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"status": "pass", "reason": "ok", "violated_constraint": None})

    check_diff_constraints(
        additional_instructions="Only use the approved retriever.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
        change_requests=["Add a local index lookup tool"],
        outcome_reflection="The index is a pre-built export of the full corpus.",
    )
    assert captured_prompt is not None
    assert "Add a local index lookup tool" in captured_prompt
    assert "pre-built export of the full corpus" in captured_prompt


def test_diff_no_change_context_section_when_absent():
    """With neither change_requests nor outcome_reflection, no context section."""
    captured_prompt = None

    def mock_lm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"status": "pass", "reason": "ok", "violated_constraint": None})

    check_diff_constraints(
        additional_instructions="Do not change the language model.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
    )
    assert captured_prompt is not None
    # The rendered context block's fences and labels must be absent (the phrase
    # "Architect's description of the change" also appears in the static prompt
    # guidance, so we key off the markers that only render with content).
    assert "===== BEGIN ARCHITECT'S DESCRIPTION" not in captured_prompt
    assert "[Change request(s) the architect issued]" not in captured_prompt
    assert "[Architect's outcome reflection]" not in captured_prompt


def test_diff_prompt_quarantines_architect_context_below_constraints():
    """The architect's description must render *below* the constraints fence.

    Guards against change-request-as-bait false positives. The architect
    writes its change requests as imperative "do NOT ..." prose; interpolated
    directly under the ``## Additional Instructions (constraints)`` heading at
    the same heading level, the judge reads it as a continuation of the
    constraints list and quotes the architect's self-imposed rules as violated
    constraints. The guard is structural, so assert the structure:
    constraints, then the end-of-constraints separator, then the fenced
    architect block.
    """
    captured_prompt = None

    def mock_lm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({"status": "pass", "reason": "ok", "violated_constraint": None})

    check_diff_constraints(
        additional_instructions="Do not change the language model.",
        diff=_SAMPLE_DIFF,
        lm_callable=mock_lm,
        change_requests=["Do NOT change the LM or k (10 per retrieve)."],
        outcome_reflection="Raised k to 15.",
    )
    assert captured_prompt is not None
    fence = captured_prompt.index("=== END OF CONSTRAINTS ===")
    begin = captured_prompt.index("===== BEGIN ARCHITECT'S DESCRIPTION")
    end = captured_prompt.index("===== END ARCHITECT'S DESCRIPTION")
    bait = captured_prompt.index("Do NOT change the LM or k (10 per retrieve).")
    # The constraints fence closes before any architect text appears, and the
    # architect's imperative prose sits inside the quarantine block.
    assert fence < begin < bait < end
    # The architect block also renders after the diff, not between the
    # constraints and the decision rules.
    assert captured_prompt.index("## Proposed Code Change (git diff)") < begin


# ---------------------------------------------------------------------------
# Real-LM eval setup
# ---------------------------------------------------------------------------

# Default checker model = the production constraint judge
# (`lm_config.CONSTRAINT_CHECK_LM`, Claude Sonnet 5 — hardcoded in the gate).
# Override with CONSTRAINT_CHECK_MODEL (a litellm model string) to eval a
# different judge.
CONSTRAINT_CHECK_MODEL = os.environ.get(
    "CONSTRAINT_CHECK_MODEL", "anthropic/claude-sonnet-5"
)


# ---------------------------------------------------------------------------
# Diff-driven eval tests (real LM) — real false-positive diffs from a run
# ---------------------------------------------------------------------------

DIFFS_DIR = Path(__file__).parent / "test_data" / "constraint_diffs"
DIFF_EXAMPLES_JSON = DIFFS_DIR / "constraint_diff_examples.json"


def _load_diff_examples() -> list[dict[str, str]]:
    with open(DIFF_EXAMPLES_JSON) as f:
        return json.load(f)


def _missing_key_for_model(model: str) -> str | None:
    """Name of the provider API key the model needs but is unset, else None.

    Keeps the diff-eval skip gate model-aware so the same test can be run with
    different `CONSTRAINT_CHECK_MODEL` values (e.g. `anthropic/claude-sonnet-5`
    vs a one-off `openai/...` cross-model check) and each run skips only when
    *its* provider key is missing — not the default provider's.
    """
    if model.startswith("anthropic/"):
        return None if os.environ.get("ANTHROPIC_API_KEY") else "ANTHROPIC_API_KEY"
    if model.startswith("openai/"):
        return None if os.environ.get("OPENAI_API_KEY") else "OPENAI_API_KEY"
    # Unknown / other provider: accept either key as evidence of a live env.
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return None
    return "ANTHROPIC_API_KEY or OPENAI_API_KEY"


_EVAL_MISSING_KEY = _missing_key_for_model(CONSTRAINT_CHECK_MODEL)


# The LLM judge is not perfectly deterministic: transient provider errors
# (e.g. Anthropic `overloaded_error`) fail open to 'pass' by design, and a
# borderline diff can occasionally flip. Rather than fail the whole suite on a
# single flaky verdict, the eval tolerates this many wrong verdicts across the
# example set (13 examples today ⇒ at least 12 must be judged correctly).
MAX_DIFF_EVAL_MISMATCHES = 1


@pytest.mark.skipif(
    _EVAL_MISSING_KEY is not None,
    reason=f"{_EVAL_MISSING_KEY} not set — skipping LLM diff eval "
    f"(CONSTRAINT_CHECK_MODEL={CONSTRAINT_CHECK_MODEL})",
)
def test_diff_constraint_checker_real_diffs():
    """Real optimization diffs must get the correct pass/fail verdict.

    Captured by SHA from the PhantomWiki and Hover repos. Covers:
    - false-positive LM-change reverts that must score ``pass`` (three
      PhantomWiki retrieval-only diffs; the phantom date-index diff under the
      LM-only constraint; the Hover hardcoded-pattern programs);
    - change-request-as-bait false positives that must score ``pass`` (two
      Hover cases whose architect change requests contain "constraints" absent
      from additional_instructions);
    - genuine ``fail`` cases (the date-index diff under a ColBERT-only DB constraint; the
      Hover program edited to return >21 documents);

    Each case carries its own ``additional_instructions`` and
    ``expected_status`` in ``constraint_diff_examples.json``.

    Aggregate assertion: the judge must get at least
    ``len(examples) - MAX_DIFF_EVAL_MISMATCHES`` verdicts right (12 of 13
    today) — one miss is tolerated as LLM-judge noise, two or more is a real
    regression.
    """
    examples = _load_diff_examples()
    lm_callable = _litellm_callable(CONSTRAINT_CHECK_MODEL)

    def _judge(example: dict[str, str]):
        diff = (DIFFS_DIR / example["diff_file"]).read_text()
        # Optional architect-authored context (the change request(s) + outcome
        # reflection the system produces each iteration). Mirrors what the live
        # gate passes; lets the checker catch violations the raw diff hides.
        return check_diff_constraints(
            additional_instructions=example["additional_instructions"],
            diff=diff,
            lm_callable=lm_callable,
            change_requests=example.get("change_requests"),
            outcome_reflection=example.get("outcome_reflection"),
        )

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(_judge, examples))

    mismatches: list[str] = []
    for example, result in zip(examples, results):
        if result.status == example["expected_status"]:
            continue
        mismatches.append(
            f"[{example['name']}] expected {example['expected_status']!r} but "
            f"got {result.status!r}. Reason: {result.reason}. "
            f"violated_constraint: {result.violated_constraint!r}. "
            f"Rationale: {example['why']}"
        )

    correct = len(examples) - len(mismatches)
    print(
        f"\n[constraint diff eval] {CONSTRAINT_CHECK_MODEL}: "
        f"{correct}/{len(examples)} verdicts correct"
    )
    for m in mismatches:
        print(f"  MISMATCH {m}")

    assert len(mismatches) <= MAX_DIFF_EVAL_MISMATCHES, (
        f"Constraint judge got {correct}/{len(examples)} verdicts right "
        f"(threshold: at least {len(examples) - MAX_DIFF_EVAL_MISMATCHES}). "
        "Mismatches:\n" + "\n".join(mismatches)
    )
