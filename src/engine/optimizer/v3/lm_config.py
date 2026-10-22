"""Copyright © 2026 440 Labs LLC

reflection_lm → Claude agent-model bridge.

`reflection_lm` is a litellm model string (`provider/model`). In Claude-harness
mode it selects the model that Claude Agent SDK agents run on — the
IterationArchitect plus the coding / inspection / valset-reflection subprocess
agents and the seed-step `codeevolver.md` repo-analysis agent session
(`seed.py::_generate_codeevolver_md`, which runs through
`run_inspection_agent`, not a bare litellm completion). It still names the
litellm model for any non-agentic reflection summaries.

The Claude Agent SDK is Anthropic-only and takes a **bare** model id
(e.g. `"claude-opus-4-7"`), while litellm wants the `"anthropic/..."`-prefixed
form. So `reflection_lm` must be an `anthropic/` litellm string; we validate
that at startup and strip the prefix when handing the model to the SDK.

OpenCode-harness mode does not use the Claude Agent SDK. There, the coding
agent model (and the seed `codeevolver.md` analysis session) comes from
`coding_harness_model`, and `reflection_lm` is only the LiteLLM model for any
non-agentic reflection summaries.

litellm forwards unmapped Anthropic model names straight through to the
Messages API, so the newest Opus models (e.g. `claude-opus-4-7`,
`claude-opus-4-8`) work even before they land in litellm's static cost map —
cost tracking degrades gracefully but the call succeeds.
"""

from __future__ import annotations

import logging
import os
from typing import Callable

logger = logging.getLogger(__name__)

ANTHROPIC_PREFIX = "anthropic/"

# Hardcoded constraint-gate judge. The gate is a plain litellm call inside the
# optimizer process, so this is harness-independent (claude and opencode).
# Hardcoded — not a config knob — because the judge's false-positive discipline
# is validated against the fixture suite in
# `tests/functional/test_constraint_checker.py`; an arbitrary `reflection_lm`
# (e.g. GLM 5.2 in opencode mode) is unvetted for this role and has produced
# false-positive rejections.
CONSTRAINT_CHECK_LM = "anthropic/claude-sonnet-5"


def build_reflection_lm_callable(reflection_lm: str) -> Callable[[str], str]:
    """Wrap `reflection_lm` into a `Callable[[str], str]` over litellm.

    Used for non-agentic reflection summaries — e.g. the constraint-gate
    checker (`agent.check_diff_constraints`). The returned closure is cheap to
    build (no API call until invoked) and works for both Claude and OpenCode
    harness modes, since `reflection_lm` is always a valid litellm model
    string. Mirrors the litellm call shape used by the constraint-checker
    tests and the legacy seed-step summary call.

    Token accounting: each call accumulates its token usage onto the returned
    callable's `token_ledger` attribute (a `ModelTokenLedger`), keyed by the
    model litellm reports. The `Callable[[str], str]` contract is unchanged for
    every existing caller; the optimizer loop reads `token_ledger` to attribute
    constraint-checker tokens per iteration.
    """
    from src.engine.agent.token_usage import ModelTokenLedger, TokenUsage

    ledger = ModelTokenLedger()

    def _call(prompt: str) -> str:
        import litellm  # type: ignore

        resp = litellm.completion(
            model=reflection_lm,
            messages=[{"role": "user", "content": prompt}],
        )
        try:
            usage = resp.get("usage") if isinstance(resp, dict) else getattr(resp, "usage", None)
            model = (
                resp.get("model") if isinstance(resp, dict) else getattr(resp, "model", None)
            ) or reflection_lm
            ledger.add(model, TokenUsage.from_litellm_usage(usage))
        except Exception:  # noqa: BLE001 - usage capture is best-effort
            pass
        return resp["choices"][0]["message"]["content"] or ""

    _call.token_ledger = ledger  # type: ignore[attr-defined]
    return _call


def build_constraint_lm_callable(reflection_lm: str) -> Callable[[str], str]:
    """Build the constraint-gate judge callable (hardcoded `CONSTRAINT_CHECK_LM`).

    Falls back to `reflection_lm` (with a warning) when `ANTHROPIC_API_KEY` is
    absent — e.g. a local opencode-only setup — so constraints stay enforced in
    every environment rather than silently no-oping. The returned callable
    carries the same `token_ledger` contract as `build_reflection_lm_callable`.
    """
    if os.environ.get("ANTHROPIC_API_KEY"):
        return build_reflection_lm_callable(CONSTRAINT_CHECK_LM)
    logger.warning(
        "ANTHROPIC_API_KEY unset; constraint gate falling back to "
        "reflection_lm=%s instead of %s",
        reflection_lm,
        CONSTRAINT_CHECK_LM,
    )
    return build_reflection_lm_callable(reflection_lm)


def validate_reflection_lm(reflection_lm: str) -> None:
    """Raise ValueError unless `reflection_lm` is an `anthropic/...` model.

    Hard-error at startup: `reflection_lm` now also selects the Claude Agent
    SDK model for all agents, and the SDK cannot run non-Anthropic models, so
    a non-Anthropic value is unusable rather than a degraded fallback.
    """
    if not isinstance(reflection_lm, str) or not reflection_lm.startswith(
        ANTHROPIC_PREFIX
    ):
        raise ValueError(
            "reflection_lm must be an Anthropic litellm model string "
            "(e.g. 'anthropic/claude-opus-4-7' or 'anthropic/claude-opus-4-8'). "
            "In engine 3 it also selects the model for the Claude Agent SDK agents "
            "(architect, coding, inspection), which are Anthropic-only. "
            f"Got {reflection_lm!r}."
        )


def resolve_agent_model(reflection_lm: str) -> str:
    """Validate `reflection_lm` and return the bare Anthropic model id.

    `"anthropic/claude-opus-4-7"` → `"claude-opus-4-7"`. The bare id is what
    `claude_agent_sdk.ClaudeAgentOptions(model=...)` expects.
    """
    validate_reflection_lm(reflection_lm)
    return reflection_lm[len(ANTHROPIC_PREFIX):]


def resolve_agent_model_for_harness(
    *,
    reflection_lm: str,
    coding_harness: str,
) -> str | None:
    """Return the provider agent model for the selected coding harness.

    Claude needs a bare Anthropic model id. OpenCode receives its model through
    `coding_harness_model`, so no Claude SDK model is resolved or validated.
    """
    if coding_harness == "claude":
        return resolve_agent_model(reflection_lm)
    return None


__all__ = [
    "ANTHROPIC_PREFIX",
    "CONSTRAINT_CHECK_LM",
    "build_constraint_lm_callable",
    "build_reflection_lm_callable",
    "resolve_agent_model",
    "resolve_agent_model_for_harness",
    "validate_reflection_lm",
]
