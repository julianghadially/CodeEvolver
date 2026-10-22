"""Copyright © 2026 440 Labs LLC

`run_variance_estimate` — the architect's once-per-run noise estimate.

Architects on every arm eventually discover they need to know how noisy the
training subsample is, and buy that knowledge late and badly with their own
budget (re-running the same tree, arguing over one-row deltas). This tool
gives them one principled reading (specs/archive-public/sampling_variance.md): draw ONE
fresh random `subsample_size` sample from the trainset, evaluate the CURRENT
program on those same rows R times (`variance_estimate_samples`, default 5),
and report

- `within_item_variance` — the pooled per-row variance across the repeated
  runs, `mean_i(s_i²)` with `s_i² = Σ_r (Y_ir − ȳ_i)² / (R − 1)`;
- `run_to_run_std` — the expected run-to-run standard deviation of the
  subsample metric, `sqrt(within_item_variance / N)`;
- `aggregate_scores` — the R aggregate scores themselves (and their
  empirical sample std, for the record).

It is a heuristic: it assumes row-to-row independence, item-level
homoskedasticity and a representative calibration sample; it knowingly does
not track candidate-to-candidate variance differences or variance drift as the
program improves (which is why the prompt tells the architect to run it after
a few improvement cycles, not on the seed program). Judge instability is
bundled in when an LLM judge is used.

Mechanics: the R runs are recorded as `eval_sample="variance"` — never as
train evals — so they can never become an iteration's baseline,
`subsample_best`, a revert-check reading or a close-check-5c satisfier. They
DO count against the eval-row budget. The result lives on `CEState`
(`variance_estimate`) so every later iteration's system prompt renders it,
is stamped on the iteration record that ran it, and sets the revert-check
tolerance (`architect_watchdogs.decide_revert_check`). Once it has run, the
tool is dropped from later iterations' registries and a second call in the
same iteration is refused without spending rows.
"""

from __future__ import annotations

import logging
import math
import statistics
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from src.engine.services.subprocess_service import (
    git_get_current_branch,
    git_get_sha,
    git_tree_signature,
)

from .errors import RunEvalError
from .eval_log import (
    VARIANCE_EVAL_SAMPLE,
    row_scores_from_trace_meta,
    sample_key_for_rows,
    variance_suffix,
)

if TYPE_CHECKING:
    from .run_eval_tool import RunEvalBinding

logger = logging.getLogger(__name__)

# Tool-response note when the run's one estimate has already been taken.
ALREADY_RUN_NOTE = (
    "run_variance_estimate has already been run once this optimization — it "
    "may only be called once per run. The result below is the run's "
    "estimate (it is also rendered in your system prompt); no eval rows were "
    "spent on this call."
)

VARIANCE_AUTO_COMMIT_NOTE = (
    "Your uncommitted changes were auto-committed before the variance runs "
    "so the recorded sha names the content that was measured."
)

IDENTICAL_OUTPUTS_NOTE = (
    "WARNING: every run produced byte-identical row outputs — an LLM response "
    "cache is probably active in the client program, so this estimate "
    "understates the true run-to-run variance."
)

HEURISTIC_NOTE = (
    "Heuristic: assumes row-to-row independence, item-level homoskedasticity "
    "and a representative calibration sample; it does not track variance "
    "differences between candidates or as the program improves. Judge "
    "instability is included when an LLM judge scores the rows. Use "
    "run_to_run_std as the noise floor for train-subsample deltas: a "
    "before/after difference within about one standard deviation is not "
    "evidence of a change."
)


@dataclass(frozen=True)
class VarianceStats:
    """The pure statistics of one estimate (see `estimate_variance`)."""

    within_item_variance: float
    run_to_run_std: float
    aggregate_std_empirical: float
    n_rows: int
    n_runs: int


@dataclass(frozen=True)
class VarianceEstimate:
    """The run's recorded estimate — `CEState.variance_estimate` is its
    `to_dict()`; the same dict is stamped on the iteration record that ran
    the tool and threaded onto every later `IterationContext`."""

    iteration: int
    sha: str | None
    tree: str | None
    branch: str | None
    subsample_size: int
    n_rows: int
    n_runs: int
    row_ids: list[Any]
    sample_key: str
    eval_ids: list[str | None]
    aggregate_scores: list[float]
    within_item_variance: float
    run_to_run_std: float
    aggregate_std_empirical: float
    identical_outputs_warning: bool
    # True when the budget ran out mid-way and fewer than the configured
    # runs completed (at least two did, or there is no estimate at all).
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def estimate_variance(
    per_run_row_scores: Sequence[Mapping[str, float]],
    aggregate_scores: Sequence[float],
) -> VarianceStats:
    """Within-item variance and run-to-run std from R repeated runs.

    `per_run_row_scores[r]` maps `str(example_id) -> score` for run r. Only
    rows scored in EVERY run enter the estimate (a row that failed in one
    run has no variance reading). Raises `ValueError` with fewer than two
    runs or no common rows.
    """
    n_runs = len(per_run_row_scores)
    if n_runs < 2:
        raise ValueError(f"at least 2 runs are needed to estimate variance, got {n_runs}")
    if len(aggregate_scores) != n_runs:
        raise ValueError("aggregate_scores must have one entry per run")
    common = set(per_run_row_scores[0].keys())
    for run in per_run_row_scores[1:]:
        common &= set(run.keys())
    if not common:
        raise ValueError("no row was scored in every run; cannot estimate variance")
    per_item_variances: list[float] = []
    for rid in sorted(common):
        values = [float(run[rid]) for run in per_run_row_scores]
        mean = sum(values) / n_runs
        per_item_variances.append(
            sum((v - mean) ** 2 for v in values) / (n_runs - 1)
        )
    n_rows = len(per_item_variances)
    within_item_variance = sum(per_item_variances) / n_rows
    run_to_run_std = math.sqrt(within_item_variance / n_rows)
    aggregate_std_empirical = statistics.stdev(float(s) for s in aggregate_scores)
    return VarianceStats(
        within_item_variance=within_item_variance,
        run_to_run_std=run_to_run_std,
        aggregate_std_empirical=aggregate_std_empirical,
        n_rows=n_rows,
        n_runs=n_runs,
    )


def run_variance_estimate_tool(
    *,
    ctx: Any,
    state: Any,
    binding: "RunEvalBinding",
    workspace: str,
    n_runs: int,
) -> dict[str, Any]:
    """The LLM-facing `run_variance_estimate` call (no arguments).

    Once per run: a second call returns a structured refusal carrying the
    existing estimate and spends nothing. On pareto_evolution the call is
    refused after the iteration's comparison run (the comparison is the
    close step there; the auto-commit below would otherwise move HEAD past
    the comparison point).
    """
    if not ctx.parent_locked:
        raise RunEvalError(
            "run_variance_estimate is not available yet — the iteration has "
            "not been initialized by the optimizer loop."
        )
    existing = getattr(state, "variance_estimate", None)
    if existing is not None:
        return {
            "ok": False,
            "refused": "already_run",
            "note": ALREADY_RUN_NOTE,
            "variance_estimate": dict(existing),
        }
    if binding.comparison_closes_iteration and any(
        rec.get("eval_sample") == "comparison" and not rec.get("by_loop")
        for rec in ctx.eval_records
    ):
        from .run_eval_tool import COMPARISON_CLOSED_NOTE

        raise RunEvalError(COMPARISON_CLOSED_NOTE)
    if n_runs < 2:
        raise RunEvalError(
            f"variance_estimate_samples must be >= 2 to estimate variance, got {n_runs}"
        )

    auto_committed = binding._auto_commit_dirty_tree("variance")
    # The content being measured, captured BEFORE the runs (they do not
    # change it, but the architect might edit while waiting on a resume).
    sha = git_get_sha(workspace)
    tree = git_tree_signature(workspace)
    try:
        branch = git_get_current_branch(workspace)
    except Exception:  # noqa: BLE001
        branch = None

    row_ids, results = binding.run_variance_samples(n_runs)
    if len(results) < 2:
        raise RunEvalError(
            f"run_variance_estimate stopped after {len(results)} of {n_runs} "
            "runs (eval budget exhausted); at least two are needed, so no "
            "estimate was recorded. The rows already run are counted."
        )

    from src.engine.evaluator.holdout import samples_are_identical

    per_run = [row_scores_from_trace_meta(r.trace_meta) for r in results]
    stats = estimate_variance(per_run, [float(r.primary_score) for r in results])
    identical = samples_are_identical([r.trace_dir for r in results])
    eval_ids = [rec.get("eval_id") for rec in ctx.eval_records[-len(results):]]
    estimate = VarianceEstimate(
        iteration=int(ctx.iteration_idx),
        sha=sha,
        tree=tree,
        branch=branch,
        subsample_size=int(binding.subsample_size),
        n_rows=stats.n_rows,
        n_runs=stats.n_runs,
        row_ids=list(row_ids),
        sample_key=sample_key_for_rows(row_ids),
        eval_ids=eval_ids,
        aggregate_scores=[float(r.primary_score) for r in results],
        within_item_variance=stats.within_item_variance,
        run_to_run_std=stats.run_to_run_std,
        aggregate_std_empirical=stats.aggregate_std_empirical,
        identical_outputs_warning=bool(identical),
        truncated=len(results) < n_runs,
    )
    payload = estimate.to_dict()
    record = getattr(state, "record_variance_estimate", None)
    if callable(record):
        record(payload)
    else:
        setattr(state, "variance_estimate", payload)
    ctx.variance_estimate = payload
    ctx.variance_estimate_ran = True
    logger.info(
        "iteration %d: run_variance_estimate on sha %s — %d runs x %d rows: "
        "scores=%s within_item_variance=%.6f run_to_run_std=%.4f "
        "(empirical std %.4f)%s%s",
        ctx.iteration_idx,
        (sha or "?")[:8],
        stats.n_runs,
        stats.n_rows,
        [round(s, 4) for s in estimate.aggregate_scores],
        stats.within_item_variance,
        stats.run_to_run_std,
        stats.aggregate_std_empirical,
        " TRUNCATED" if estimate.truncated else "",
        " IDENTICAL-OUTPUTS" if identical else "",
    )

    notes = [HEURISTIC_NOTE]
    if identical:
        notes.insert(0, IDENTICAL_OUTPUTS_NOTE)
    if estimate.truncated:
        notes.append(
            f"Only {stats.n_runs} of {n_runs} runs completed before the eval "
            "budget ran out; the estimate uses the completed runs."
        )
    response: dict[str, Any] = {"ok": True, **payload}
    response["trace_dirs"] = [r.trace_dir for r in results]
    response["note"] = " ".join(notes)
    if auto_committed:
        response["auto_committed"] = True
        response["auto_commit_note"] = VARIANCE_AUTO_COMMIT_NOTE
    response.update(binding._budget_fields())
    return response


# ----------------------------------------------------------------------
# System-prompt text (rendered by prompts._subsample_note)
# ----------------------------------------------------------------------
_NOISE_SENTENCE = (
    "Subsamples when small (e.g., < 100 rows) can be quite noisy depending on "
    "the application, even when the seed is fixed - but especially when the "
    "seed differs between iterations."
)

_OVERFIT_SENTENCE = (
    "Given the variance above, use the training data to see examples and run "
    "quick tests, but be careful not to overfit. Ultimately, the application "
    "is judged by its ability to generalize beyond the training set."
)


def variance_note(
    estimate: Mapping[str, Any] | None,
    subsample_size: int,
    n_runs: int,
) -> str:
    """The sampling-variance paragraph of the architect system prompt.

    Before the tool has run: the noise caveat plus the recommendation to
    call `run_variance_estimate` after a few improvement cycles. After: the
    recorded estimate, so every later iteration reads the same numbers.
    """
    if not estimate:
        return (
            f"{_NOISE_SENTENCE} {_OVERFIT_SENTENCE}\n\n"
            "You can and should estimate the run-to-run variance at least once, "
            "using the run_variance_estimate tool. This will run the program "
            f"{n_runs} times with a fixed sample of {subsample_size}, and the "
            "result is available to all future iterations. You will receive the "
            "within-item variance, the run-to-run standard deviation and the "
            "program score for each run. The best time to estimate run-to-run "
            "variance is after you have run a few improvement cycles. Usually at "
            "the end of iterations 1-3 would be a good time to run this tool. "
            "Running it on the seed program immediately is not recommended "
            "because if you then make quick improvements, the variance may "
            "differ on a program that scores 2x higher."
        )
    scores = ", ".join(f"{float(s):.4f}" for s in estimate.get("aggregate_scores") or [])
    sha = str(estimate.get("sha") or "?")[:8]
    lines = [
        _NOISE_SENTENCE,
        "",
        (
            f"On this codebase, the program at iteration {estimate.get('iteration')} "
            f"and sha {sha} was run {estimate.get('n_runs')} times with a fixed "
            f"sample of size {estimate.get('subsample_size', subsample_size)}."
        ),
        f"The within-item variance was {float(estimate.get('within_item_variance') or 0.0):.4f}",
        (
            f"The run-to-run standard deviation for the {estimate.get('n_rows')}-item "
            f"evaluation metric was {float(estimate.get('run_to_run_std') or 0.0):.4f}"
        ),
        f"The program scored [{scores}]",
    ]
    if estimate.get("identical_outputs_warning"):
        lines.append(IDENTICAL_OUTPUTS_NOTE)
    lines += ["", _OVERFIT_SENTENCE]
    return "\n".join(lines)


__all__ = [
    "ALREADY_RUN_NOTE",
    "HEURISTIC_NOTE",
    "IDENTICAL_OUTPUTS_NOTE",
    "VARIANCE_AUTO_COMMIT_NOTE",
    "VARIANCE_EVAL_SAMPLE",
    "VarianceEstimate",
    "VarianceStats",
    "estimate_variance",
    "run_variance_estimate_tool",
    "variance_note",
    "variance_suffix",
]
