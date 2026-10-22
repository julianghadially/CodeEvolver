"""Copyright © 2026 440 Labs LLC

Framework-agnostic sandbox environment validation for the EE seed step.

Runs the generic evaluator on the first N training rows and fails fast if
the environment is broken (missing deps, import errors, metric blow-ups,
etc). It uses the same `run_eval` path the architect loop uses, so it can
never drift from production eval semantics and is not tied to any AI
framework.

Failure rule (from specs/requirements.md):
    - system error rate > 10%, or
    - aggregate primary metric == 0 across the sampled rows.

A "system error" is any row the mounted evaluator marked `failed=True` —
i.e. the program call raised or a metric raised. Low-but-nonzero primary
scores are NOT treated as failures; that is what optimization is for.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from src.engine.evaluator.config import EvaluatorConfig
from src.engine.evaluator.dataset_config import DatasetConfig
from src.engine.evaluator.sandbox_client import run_eval
from src.engine.telemetry.trace_layout import iter_rows

logger = logging.getLogger(__name__)


DEFAULT_VALIDATION_ROWS = 5
DEFAULT_ERROR_THRESHOLD = 0.10


@dataclass
class SandboxValidationError(RuntimeError):
    """Raised when the seed-time environment check fails.

    Carries structured details so the caller can surface actionable
    diagnostics to the user instead of a generic traceback.
    """

    message: str
    n_rows: int
    system_error_rows: list[int] = field(default_factory=list)
    system_error_rate: float = 0.0
    primary_metric: str = ""
    primary_score: float = 0.0
    examples: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        super().__init__(self.message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": self.message,
            "n_rows": self.n_rows,
            "system_error_rows": list(self.system_error_rows),
            "system_error_rate": self.system_error_rate,
            "primary_metric": self.primary_metric,
            "primary_score": self.primary_score,
            "examples": list(self.examples),
        }


def validate_sandbox_environment(
    *,
    workspace: str,
    evaluator_config: EvaluatorConfig,
    datasets: DatasetConfig,
    n_rows: int = DEFAULT_VALIDATION_ROWS,
    timeout: int | None = None,
) -> dict[str, Any]:
    """Run a smoke-test eval on the first `n_rows` training rows.

    Returns a small report dict on success. Raises `SandboxValidationError`
    on failure so the caller can abort the run before the expensive full
    evaluation.
    """
    if n_rows <= 0:
        raise ValueError(f"n_rows must be > 0, got {n_rows}")
    if not datasets.trainset_ids:
        raise ValueError("trainset is empty — cannot validate environment")

    row_ids = datasets.trainset_ids[:n_rows]
    effective_n = len(row_ids)
    logger.info(
        "seed-validate: running environment check on first %d training row(s)",
        effective_n,
    )
    print(
        f"[VALIDATE] Running environment check on first {effective_n} training row(s)",
        flush=True,
    )

    try:
        result = run_eval(
            workspace=workspace,
            config=evaluator_config,
            row_ids=row_ids,
            iteration_idx=0,
            suffix="validate_env",
            capture_traces=True,
            timeout=timeout,
        )
    except RuntimeError as exc:
        # `run_eval` raises on `success=False` from the mounted script. At
        # this point the evaluator itself could not run at all — no trace
        # meta to inspect, so we report the raw error as a catastrophic
        # environment failure.
        raise SandboxValidationError(
            message=(
                "Environment validation failed: the mounted evaluator could "
                f"not run at all.\n\nUnderlying error: {exc}"
            ),
            n_rows=effective_n,
            system_error_rows=list(row_ids),
            system_error_rate=1.0,
            primary_metric=evaluator_config.primary_metric,
            primary_score=0.0,
            examples=[],
        ) from exc

    primary_score = result.primary_score
    trace_meta = result.trace_meta

    system_error_rows: list[int] = []
    for record in trace_meta:
        if record.get("failed"):
            eid = record.get("example_id")
            if eid is not None:
                system_error_rows.append(int(eid))

    system_error_rate = (
        len(system_error_rows) / effective_n if effective_n else 0.0
    )

    logger.info(
        "seed-validate: primary=%.4f system_errors=%d/%d (%.1f%%)",
        primary_score,
        len(system_error_rows),
        effective_n,
        system_error_rate * 100,
    )
    print(
        f"[VALIDATE] primary={primary_score:.4f} "
        f"system_errors={len(system_error_rows)}/{effective_n} "
        f"({system_error_rate:.1%})",
        flush=True,
    )

    reasons: list[str] = []
    if system_error_rate > DEFAULT_ERROR_THRESHOLD:
        reasons.append(
            f"system error rate {system_error_rate:.1%} exceeds "
            f"{DEFAULT_ERROR_THRESHOLD:.0%} threshold "
            f"({len(system_error_rows)}/{effective_n} row(s) threw)"
        )
    if primary_score == 0.0:
        reasons.append(
            f"primary metric {evaluator_config.primary_metric!r} is 0.0 across "
            f"all {effective_n} validation rows — likely a broken metric, "
            f"program, or dataset shape"
        )

    if not reasons:
        return {
            "success": True,
            "n_rows": effective_n,
            "primary_metric": evaluator_config.primary_metric,
            "primary_score": primary_score,
            "system_error_rate": system_error_rate,
            "trace_dir": result.trace_dir,
        }

    # Collect detailed per-row examples from the trace directory so the error
    # message is actionable. We read it inside the sandbox to avoid
    # depending on shared filesystems.
    examples = _collect_failure_examples(
        trace_dir=result.trace_dir,
        system_error_rows=system_error_rows,
        max_examples=5,
    )

    message = _build_error_message(
        reasons=reasons,
        n_rows=effective_n,
        primary_metric=evaluator_config.primary_metric,
        primary_score=primary_score,
        system_error_rate=system_error_rate,
        examples=examples,
    )

    raise SandboxValidationError(
        message=message,
        n_rows=effective_n,
        system_error_rows=system_error_rows,
        system_error_rate=system_error_rate,
        primary_metric=evaluator_config.primary_metric,
        primary_score=primary_score,
        examples=examples,
    )


def _collect_failure_examples(
    *,
    trace_dir: str,
    system_error_rows: list[int],
    max_examples: int,
) -> list[dict[str, Any]]:
    """Pull up to `max_examples` trace rows for failed rows.

    Falls back to an empty list if the trace directory is unreadable — the
    validation error is still raised, just with less detail.
    """
    if not trace_dir:
        return []

    try:
        rows = list(iter_rows(trace_dir))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "seed-validate: failed to read trace directory %s: %s",
            trace_dir,
            exc,
        )
        return []

    want = set(system_error_rows) if system_error_rows else None
    picked: list[dict[str, Any]] = []
    for entry in rows:
        eid = entry.get("example_id")
        if want is not None and eid not in want:
            continue
        if want is None and not entry.get("failed"):
            continue
        picked.append(
            {
                "row_index": eid,
                "failed": entry.get("failed", False),
                "score": entry.get("score"),
                "input": _truncate(entry.get("input"), 400),
                "output": _truncate(entry.get("output"), 400),
                "error_summary": entry.get("error_summary"),
            }
        )
        if len(picked) >= max_examples:
            break
    return picked


def _truncate(value: Any, limit: int) -> Any:
    """Stringify and truncate large fields so the error message stays readable."""
    if value is None:
        return None
    text = value if isinstance(value, str) else repr(value)
    if len(text) > limit:
        return text[:limit] + "…"
    return text


def _build_error_message(
    *,
    reasons: list[str],
    n_rows: int,
    primary_metric: str,
    primary_score: float,
    system_error_rate: float,
    examples: list[dict[str, Any]],
) -> str:
    lines: list[str] = ["Sandbox environment validation failed:"]
    for reason in reasons:
        lines.append(f"  - {reason}")
    lines.append("")
    lines.append(
        f"Metrics over {n_rows} row(s):  "
        f"{primary_metric}={primary_score:.4f}  "
        f"system_errors={system_error_rate:.1%}"
    )

    if examples:
        lines.append("")
        lines.append("First failing row(s):")
        for ex in examples:
            lines.append(f"  --- row {ex['row_index']} ---")
            if ex.get("error_summary"):
                lines.append(f"    error: {ex['error_summary']}")
            if ex.get("input") is not None:
                lines.append(f"    input: {ex['input']}")
            if ex.get("output") is not None:
                lines.append(f"    output: {ex['output']}")
            lines.append(f"    score: {ex.get('score')}")

    lines.append("")
    lines.append(
        "This usually means the sandbox is missing dependencies, the program "
        "factory path is wrong, the metric signature does not match row "
        "fields, or the dataset shape is broken. Fix the environment and "
        "re-run."
    )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_VALIDATION_ROWS",
    "DEFAULT_ERROR_THRESHOLD",
    "SandboxValidationError",
    "validate_sandbox_environment",
]
