"""Constraint checker for validating change requests against client constraints.

Lightweight LLM-based validation gate between the reflection agent's output
and the coding agent's execution.
"""

import json
import re
from typing import Callable

from src.schemas.lm_output_schemas import ConstraintCheckResult

_CONSTRAINT_CHECK_PROMPT = """\
You are a constraint compliance checker. Your job is to determine whether a \
proposed code change request violates any constraints specified in the \
additional instructions below.

## Additional Instructions (constraints)
{additional_instructions}

## Proposed Change Request
{change_request}

## Task
Review the additional instructions for any constraints identified. If the \
change request violates any of the constraints, respond with "fail". \
Otherwise respond with "pass".

Respond with a JSON object with these fields:
- "status": "pass" or "fail"
- "reason": brief explanation
- "violated_constraint": the specific constraint text that was violated (null if pass)

Respond ONLY with the JSON object, no other text.
"""


def _extract_json(text: str) -> str:
    """Extract JSON from text that may contain markdown code fences."""
    # Try to find JSON in a code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()


def check_constraints(
    additional_instructions: str,
    change_request: str,
    lm_callable: Callable[[str], str],
) -> ConstraintCheckResult:
    """Validate a change request against client-provided constraints.

    Args:
        additional_instructions: Client constraints text.
        change_request: The proposed change from the reflection agent.
        lm_callable: LiteLLM callable (same as used for reflection).

    Returns:
        ConstraintCheckResult with pass/fail status.
        Defaults to "pass" on any error to avoid blocking the optimization loop.
    """
    try:
        prompt = _CONSTRAINT_CHECK_PROMPT.format(
            additional_instructions=additional_instructions,
            change_request=change_request,
        )

        raw_response = lm_callable(prompt)
        json_str = _extract_json(raw_response)
        parsed = json.loads(json_str)

        return ConstraintCheckResult(**parsed)

    except Exception as e:
        print(f"[CONSTRAINT_CHECKER] Error during constraint check, defaulting to pass: {e}", flush=True)
        return ConstraintCheckResult(
            status="pass",
            reason=f"Constraint check failed with error: {e}",
            violated_constraint=None,
        )
