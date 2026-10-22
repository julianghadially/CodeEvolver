"""Copyright © 2026 440 Labs LLC

Token-usage accounting for CodeEvolver's own agents and LLM calls.

CodeEvolver runs many CodeEvolver-owned model calls per optimization run — the
in-process IterationArchitect session, the ``coding_agent`` / ``inspect_traces``
subprocesses, the valset-reflection / seed / meta-reflection inspection agents,
and the direct ``litellm`` constraint checker. The architect session's
``total_cost_usd`` alone undercounts the real spend (every subprocess / litellm
call is a *separate* API call whose usage never appears in the architect
session's usage).

This module is the single, dependency-free place that:
  - normalizes the two usage shapes we see (Claude Agent SDK ``ResultMessage.usage``
    and ``litellm`` completion ``usage``) into a :class:`TokenUsage` value type, and
  - accumulates token totals **keyed by model id** in :class:`ModelTokenLedger`,
    so runs that mix models (e.g. an OpenCode ``coding_harness_model`` alongside
    the Claude architect model) track each model's tokens separately.

We count **tokens**, not dollars. It lives under ``src/engine/agent/`` so the
harnesses, the mounted subprocess agents, the optimizer, and the subprocess
service can all import it without an import cycle. (``src/engine/telemetry/`` is
OTel *client-program* tracing run inside the client venv — a different concern.)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

ANTHROPIC_PREFIX = "anthropic/"

_UNKNOWN_MODEL = "unknown"


def normalize_model_id(model: str | None) -> str:
    """Collapse provider-prefixed and bare model ids onto one key.

    The Claude Agent SDK takes a *bare* id (``"claude-opus-4-8"``) while litellm
    uses the ``"anthropic/claude-opus-4-8"`` form. Stripping the ``anthropic/``
    prefix makes the same underlying model aggregate under one ledger key. An
    empty/None model maps to ``"unknown"`` so usage is never silently dropped.
    """
    if not model:
        return _UNKNOWN_MODEL
    model = model.strip()
    if model.startswith(ANTHROPIC_PREFIX):
        model = model[len(ANTHROPIC_PREFIX):]
    return model or _UNKNOWN_MODEL


def _as_int(value: Any) -> int:
    """Coerce a usage field to a non-negative int, tolerating None/garbage."""
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True)
class TokenUsage:
    """Immutable token counts for a single model call (or an aggregate)."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @property
    def total(self) -> int:
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_creation_input_tokens
            + self.cache_read_input_tokens
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=self.input_tokens + other.input_tokens,
            output_tokens=self.output_tokens + other.output_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
        )

    def __sub__(self, other: "TokenUsage") -> "TokenUsage":
        """Field-wise difference, clamped at zero (for snapshot deltas)."""
        if not isinstance(other, TokenUsage):
            return NotImplemented
        return TokenUsage(
            input_tokens=max(0, self.input_tokens - other.input_tokens),
            output_tokens=max(0, self.output_tokens - other.output_tokens),
            cache_creation_input_tokens=max(
                0, self.cache_creation_input_tokens - other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=max(
                0, self.cache_read_input_tokens - other.cache_read_input_tokens
            ),
        )

    @classmethod
    def zero(cls) -> "TokenUsage":
        return cls()

    @classmethod
    def from_sdk_usage(cls, usage: Mapping[str, Any] | None) -> "TokenUsage":
        """Parse a Claude Agent SDK ``ResultMessage.usage`` mapping.

        The SDK keys map 1:1 to our fields. Missing keys / a None mapping yield
        zeros so a session that reported no usage degrades gracefully.
        """
        if not usage:
            return cls.zero()
        return cls(
            input_tokens=_as_int(usage.get("input_tokens")),
            output_tokens=_as_int(usage.get("output_tokens")),
            cache_creation_input_tokens=_as_int(
                usage.get("cache_creation_input_tokens")
            ),
            cache_read_input_tokens=_as_int(usage.get("cache_read_input_tokens")),
        )

    @classmethod
    def from_litellm_usage(cls, usage: Any) -> "TokenUsage":
        """Parse a litellm completion ``usage`` (object or dict).

        litellm uses OpenAI-shaped names (``prompt_tokens`` / ``completion_tokens``)
        and reports cached-read tokens under ``prompt_tokens_details.cached_tokens``
        on newer versions (older ones expose ``cache_read_input_tokens``). Cache
        attribution is best-effort — token shape drifts across litellm versions —
        but the input/output totals are always captured.
        """
        if usage is None:
            return cls.zero()

        def _get(obj: Any, key: str) -> Any:
            if isinstance(obj, Mapping):
                return obj.get(key)
            return getattr(obj, key, None)

        input_tokens = _as_int(_get(usage, "prompt_tokens"))
        output_tokens = _as_int(_get(usage, "completion_tokens"))

        cache_read = _get(usage, "cache_read_input_tokens")
        if cache_read is None:
            details = _get(usage, "prompt_tokens_details")
            if details is not None:
                cache_read = _get(details, "cached_tokens")
        cache_creation = _get(usage, "cache_creation_input_tokens")

        return cls(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_creation_input_tokens=_as_int(cache_creation),
            cache_read_input_tokens=_as_int(cache_read),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total": self.total,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any] | None) -> "TokenUsage":
        """Inverse of :meth:`to_dict` (the extra ``total`` key is ignored)."""
        if not d:
            return cls.zero()
        return cls(
            input_tokens=_as_int(d.get("input_tokens")),
            output_tokens=_as_int(d.get("output_tokens")),
            cache_creation_input_tokens=_as_int(d.get("cache_creation_input_tokens")),
            cache_read_input_tokens=_as_int(d.get("cache_read_input_tokens")),
        )


class ModelTokenLedger:
    """Accumulates :class:`TokenUsage` keyed by normalized model id."""

    def __init__(self) -> None:
        self._by_model: dict[str, TokenUsage] = {}

    def add(self, model: str | None, usage: TokenUsage) -> None:
        key = normalize_model_id(model)
        self._by_model[key] = self._by_model.get(key, TokenUsage.zero()) + usage

    def merge(self, other: "ModelTokenLedger | Mapping[str, Any] | None") -> None:
        """Fold another ledger (or a ``{model: usage-dict}`` mapping) into this one."""
        if other is None:
            return
        if isinstance(other, ModelTokenLedger):
            items = other._by_model.items()
        elif isinstance(other, Mapping):
            items = (
                (model, TokenUsage.from_dict(val)) for model, val in other.items()
            )
        else:  # pragma: no cover - defensive
            return
        for model, usage in items:
            self.add(model, usage)

    def grand_total(self) -> TokenUsage:
        total = TokenUsage.zero()
        for usage in self._by_model.values():
            total = total + usage
        return total

    def snapshot(self) -> dict[str, TokenUsage]:
        """A shallow copy of the per-model totals (TokenUsage is immutable).

        Used to compute the per-iteration delta of a run-scoped ledger (e.g. the
        constraint-checker callable's ledger): snapshot before, diff after.
        """
        return dict(self._by_model)

    def delta_since(self, prior: dict[str, TokenUsage]) -> dict[str, TokenUsage]:
        """Per-model usage accrued since a prior :meth:`snapshot` (zero entries dropped)."""
        out: dict[str, TokenUsage] = {}
        for model, usage in self._by_model.items():
            diff = usage - prior.get(model, TokenUsage.zero())
            if diff.total > 0:
                out[model] = diff
        return out

    def is_empty(self) -> bool:
        return not self._by_model

    def to_dict(self) -> dict[str, dict[str, int]]:
        out = {model: usage.to_dict() for model, usage in self._by_model.items()}
        if out:
            out["__total__"] = self.grand_total().to_dict()
        return out

    def log_summary(self, logger: logging.Logger, label: str) -> None:
        """Emit one structured log line per model plus a grand total."""
        if self.is_empty():
            logger.info("%s token usage: (none recorded)", label)
            return
        for model, usage in sorted(self._by_model.items()):
            logger.info(
                "%s token usage [%s]: in=%d out=%d cache_w=%d cache_r=%d total=%d",
                label,
                model,
                usage.input_tokens,
                usage.output_tokens,
                usage.cache_creation_input_tokens,
                usage.cache_read_input_tokens,
                usage.total,
            )
        gt = self.grand_total()
        logger.info(
            "%s token usage [ALL MODELS]: in=%d out=%d cache_w=%d cache_r=%d total=%d",
            label,
            gt.input_tokens,
            gt.output_tokens,
            gt.cache_creation_input_tokens,
            gt.cache_read_input_tokens,
            gt.total,
        )


__all__ = [
    "ANTHROPIC_PREFIX",
    "ModelTokenLedger",
    "TokenUsage",
    "normalize_model_id",
]
