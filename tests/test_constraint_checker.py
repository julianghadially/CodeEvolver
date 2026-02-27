"""Tests for the constraint checker module."""

import csv
import json
import os
from pathlib import Path

import pytest

from src.agent.constraint_checker import check_constraints
from src.optimizer.utils import get_reflection_lm_callable

# ---------------------------------------------------------------------------
# Unit tests (mock LM — test the plumbing)
# ---------------------------------------------------------------------------


def test_pass_when_no_violation():
    """Constraint checker returns pass when no constraint is violated."""
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "pass",
            "reason": "The change request does not violate any constraints.",
            "violated_constraint": None,
        })

    result = check_constraints(
        additional_instructions="Do not modify the database schema.",
        change_request="Refactor the logging module to use structured logs.",
        lm_callable=mock_lm,
    )

    assert result.status == "pass"
    assert result.violated_constraint is None


def test_fail_when_constraint_violated():
    """Constraint checker returns fail when a constraint is violated."""
    def mock_lm(prompt: str) -> str:
        return json.dumps({
            "status": "fail",
            "reason": "The change request modifies the database schema.",
            "violated_constraint": "Do not modify the database schema.",
        })

    result = check_constraints(
        additional_instructions="Do not modify the database schema.",
        change_request="Add a new column 'status' to the users table.",
        lm_callable=mock_lm,
    )

    assert result.status == "fail"
    assert result.violated_constraint == "Do not modify the database schema."
    assert "database schema" in result.reason


def test_defaults_to_pass_on_malformed_json():
    """Constraint checker defaults to pass when LLM returns malformed JSON."""
    def mock_lm(prompt: str) -> str:
        return "This is not valid JSON at all"

    result = check_constraints(
        additional_instructions="Do not change the API.",
        change_request="Refactor internal helpers.",
        lm_callable=mock_lm,
    )

    assert result.status == "pass"
    assert "error" in result.reason.lower()


def test_defaults_to_pass_on_llm_exception():
    """Constraint checker defaults to pass when LLM callable raises."""
    def mock_lm(prompt: str) -> str:
        raise RuntimeError("LLM service unavailable")

    result = check_constraints(
        additional_instructions="Do not change the API.",
        change_request="Refactor internal helpers.",
        lm_callable=mock_lm,
    )

    assert result.status == "pass"
    assert "error" in result.reason.lower()


def test_handles_markdown_code_block_response():
    """Constraint checker extracts JSON from markdown code fences."""
    def mock_lm(prompt: str) -> str:
        return """Here is my analysis:

```json
{
    "status": "fail",
    "reason": "Violates the no-API-change constraint.",
    "violated_constraint": "Do not change the public API."
}
```
"""

    result = check_constraints(
        additional_instructions="Do not change the public API.",
        change_request="Rename the /users endpoint to /accounts.",
        lm_callable=mock_lm,
    )

    assert result.status == "fail"
    assert result.violated_constraint == "Do not change the public API."


def test_prompt_contains_both_inputs():
    """Constraint checker prompt includes both additional_instructions and change_request."""
    captured_prompt = None

    def mock_lm(prompt: str) -> str:
        nonlocal captured_prompt
        captured_prompt = prompt
        return json.dumps({
            "status": "pass",
            "reason": "No violations.",
            "violated_constraint": None,
        })

    check_constraints(
        additional_instructions="Never use eval().",
        change_request="Add dynamic code execution feature.",
        lm_callable=mock_lm,
    )

    assert captured_prompt is not None
    assert "Never use eval()" in captured_prompt
    assert "Add dynamic code execution feature." in captured_prompt


# ---------------------------------------------------------------------------
# CSV-driven eval tests (real LM — test the prompt + model judgment)
# ---------------------------------------------------------------------------

EXAMPLES_CSV = Path(__file__).parent / "test_data" / "constraint_checker_examples.csv"
CONSTRAINT_CHECK_MODEL = os.environ.get("CONSTRAINT_CHECK_MODEL", "openai/gpt-5-mini")


def _load_examples() -> list[dict[str, str]]:
    with open(EXAMPLES_CSV, newline="") as f:
        return list(csv.DictReader(f))


def _example_ids() -> list[str]:
    """Short test IDs from change_request first 60 chars."""
    rows = _load_examples()
    return [row["change_request"][:60].replace(" ", "_") for row in rows]


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OPENAI_API_KEY not set — skipping LLM eval tests",
)
@pytest.mark.parametrize("example", _load_examples(), ids=_example_ids())
def test_constraint_checker_examples(example: dict[str, str]):
    """Run constraint checker against CSV examples with a real LM."""
    lm_callable = get_reflection_lm_callable(CONSTRAINT_CHECK_MODEL)

    result = check_constraints(
        additional_instructions=example["additional_instructions"],
        change_request=example["change_request"],
        lm_callable=lm_callable,
    )

    assert result.status == example["expected_status"], (
        f"Expected {example['expected_status']!r} but got {result.status!r}. "
        f"Reason: {result.reason}"
    )
