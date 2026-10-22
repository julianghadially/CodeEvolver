"""Copyright © 2026 440 Labs LLC

Positive behavioral contribution, eligibility and the `branch_comparison`
table (specs/archive-public/v3_asa.md §Eligibility / §`branch_comparison` contract). Pure —
git access is injected as a `tip_tree` per branch and an `equivalent(a, b)`
predicate.

Vocabulary (spec glossary):
- reference_row_scores / branch_row_scores: `example_id -> score` on the
  fixed comparison set for the reference (usually the primary) and the
  branch under comparison.
- positive_behavioral_contribution = sum over rows of
  max(0, branch - reference) — score DIFFERENCES, not a row count, so
  non-0/1 metrics are handled.
- n_rows_ahead = count of rows where branch > reference (separate metric).
- positive_behavioral_contribution_normalized = contribution /
  comparison_set_size (mean per-row advantage, in score units).
- threshold = min(cap, constant * (max_score - primary_comparison_score)).
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from ..pareto_front import pareto_counts, pareto_front, prune_dominated

logger = logging.getLogger(__name__)

ComparisonRecord = dict[str, Any]  # a comparison_set_history.jsonl line


# ----------------------------------------------------------------------
# Records
# ----------------------------------------------------------------------
def load_comparison_records(path: str | None) -> list[ComparisonRecord]:
    """Read `comparison_set_history.jsonl` (oldest -> newest). Lines lacking
    `row_scores` or `tree` are kept — they just cannot be matched by tree or
    compared per row."""
    if not path or not os.path.exists(path):
        return []
    out: list[ComparisonRecord] = []
    try:
        with open(path) as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    rec = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(rec, dict):
                    out.append(rec)
    except OSError:
        logger.warning("branch_comparison: could not read %s", path, exc_info=True)
    return out


@dataclass(frozen=True)
class LineageRecord:
    record: ComparisonRecord
    up_to_date: bool  # matched the branch tip by tree (exact or program-equivalent)


def lineage_record(
    branch: str,
    records: Sequence[ComparisonRecord],
    *,
    tip_tree: str | None,
    equivalent: Callable[[str, str], bool],
) -> LineageRecord | None:
    """The comparison record that represents `branch`.

    Newest record whose tree equals the tip tree or is program-equivalent to
    it (any branch — the seed's single eval represents both seed lineages);
    failing that, the newest record that ran on `branch`. None when neither
    exists (the lineage is unknown, not different — never eligible).
    """
    if tip_tree:
        for rec in reversed(records):
            tree = rec.get("tree")
            if not tree:
                continue
            if tree == tip_tree:
                return LineageRecord(rec, True)
            try:
                if equivalent(tree, tip_tree):
                    return LineageRecord(rec, True)
            except Exception:  # noqa: BLE001
                continue
    for rec in reversed(records):
        if rec.get("branch") == branch:
            return LineageRecord(rec, False)
    return None


# ----------------------------------------------------------------------
# Contribution / threshold / eligibility
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Contribution:
    positive_behavioral_contribution: float
    n_rows_ahead: int
    n_rows_compared: int
    n_rows_missing: int


def positive_contribution(
    branch_row_scores: Mapping[str, float | None] | None,
    reference_row_scores: Mapping[str, float | None] | None,
) -> Contribution:
    """Sum of max(0, branch - reference) over shared rows, plus n_rows_ahead."""
    b = {k: v for k, v in (branch_row_scores or {}).items() if v is not None}
    r = {k: v for k, v in (reference_row_scores or {}).items() if v is not None}
    shared = set(b) & set(r)
    missing = len((set(b) | set(r)) - shared)
    total = 0.0
    ahead = 0
    for rid in shared:
        diff = float(b[rid]) - float(r[rid])
        if diff > 0:
            total += diff
            ahead += 1
    return Contribution(total, ahead, len(shared), missing)


def contribution_threshold(
    *,
    primary_comparison_score: float | None,
    max_score: float,
    constant: float,
    cap: float,
) -> float:
    """min(cap, constant * (max_score - primary_score)); per-row score units.
    With no primary score the gap is the whole scale."""
    gap = max_score - (primary_comparison_score if primary_comparison_score is not None else 0.0)
    return min(cap, constant * max(0.0, gap))


# ----------------------------------------------------------------------
# Table
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class LineageRow:
    branch: str
    is_primary: bool
    is_reference: bool
    comparison_score: float | None
    positive_behavioral_contribution: float | None
    positive_behavioral_contribution_normalized: float | None
    n_rows_ahead: int | None
    pareto_score: int | None          # None when dominated
    pareto_dominated: bool
    up_to_date: bool
    eligible: bool
    record_iteration: int | None
    note: str | None = None


@dataclass(frozen=True)
class ComparisonTable:
    reference: str
    primary: str
    threshold: float
    rows: list[LineageRow]
    missing: list[str]  # registered branches with no comparison record

    @property
    def eligible(self) -> dict[str, float]:
        return {
            r.branch: r.positive_behavioral_contribution_normalized or 0.0
            for r in self.rows
            if r.eligible
        }


def build_table(
    *,
    lineages: Mapping[str, LineageRecord | None],
    reference: str,
    primary: str,
    comparison_set_size: int,
    threshold: float,
) -> ComparisonTable:
    """Rows for every lineage with a record; `missing` lists the rest.
    Raises ValueError when the reference has no record."""
    ref_rec = lineages.get(reference)
    if ref_rec is None:
        raise ValueError(
            f"reference branch {reference!r} has no comparison-set record — run "
            "`run_eval(eval_sample=\"comparison\")` on it first"
        )
    ref_rows = ref_rec.record.get("row_scores") or {}
    size = comparison_set_size or len(ref_rows) or 1

    present = {b: lr for b, lr in lineages.items() if lr is not None}
    # Lineages that resolve to the same record (same program tree — e.g. the
    # seed twins, or a branch represented by a program-equivalent record on
    # another branch) are ONE candidate for the front: GEPA's pruning would
    # otherwise drop one identical twin as dominated by the other. Prune over
    # one representative per record (primary, then reference, then by name)
    # and give every twin the representative's Pareto score.
    rep_of: dict[str, str] = {}
    for b in sorted(present, key=lambda x: (x != primary, x != reference, x)):
        rec = present[b].record
        ident = rec.get("tree") or rec.get("branch") or b
        rep_of[b] = rep_of.setdefault(f"__rec__{ident}", b)
    reps = {rep_of[b] for b in present}
    front = pareto_front({b: present[b].record.get("row_scores") or {} for b in reps})
    pruned_front, dominated = prune_dominated(
        front,
        keys=reps,
        scores={b: present[b].record.get("score") for b in reps},
    )
    counts = pareto_counts(pruned_front)

    rows: list[LineageRow] = []
    for branch, lr in present.items():
        rec = lr.record
        note = None
        if branch == reference:
            contrib = Contribution(0.0, 0, len(ref_rows), 0)
        else:
            contrib = positive_contribution(rec.get("row_scores"), ref_rows)
            if contrib.n_rows_missing:
                note = (
                    f"{contrib.n_rows_missing} row(s) not shared with the reference "
                    "record counted as 0"
                )
        normalized = contrib.positive_behavioral_contribution / size
        rep = rep_of[branch]
        is_dominated = rep in dominated
        rows.append(
            LineageRow(
                branch=branch,
                is_primary=(branch == primary),
                is_reference=(branch == reference),
                comparison_score=rec.get("score"),
                positive_behavioral_contribution=contrib.positive_behavioral_contribution,
                positive_behavioral_contribution_normalized=normalized,
                n_rows_ahead=contrib.n_rows_ahead,
                pareto_score=None if is_dominated else int(counts.get(rep, 0)),
                pareto_dominated=is_dominated,
                up_to_date=lr.up_to_date,
                # Eligibility is always vs the PRIMARY, whatever the reference.
                eligible=(
                    branch != primary
                    and reference == primary
                    and normalized > threshold
                ),
                record_iteration=rec.get("iteration"),
                note=note,
            )
        )
    # Reference first, then primary, then by contribution desc.
    rows.sort(
        key=lambda r: (
            not r.is_reference,
            not r.is_primary,
            -(r.positive_behavioral_contribution_normalized or 0.0),
            r.branch,
        )
    )
    missing = sorted(b for b, lr in lineages.items() if lr is None)
    return ComparisonTable(
        reference=reference, primary=primary, threshold=threshold, rows=rows, missing=missing
    )


def eligible_set(
    *,
    lineages: Mapping[str, LineageRecord | None],
    primary: str,
    comparison_set_size: int,
    threshold: float,
) -> dict[str, float]:
    """`{branch: normalized contribution}` for eligible non-primary lineages;
    empty when the primary has no record (rule 2 inert)."""
    try:
        table = build_table(
            lineages=lineages,
            reference=primary,
            primary=primary,
            comparison_set_size=comparison_set_size,
            threshold=threshold,
        )
    except ValueError:
        return {}
    return table.eligible


def format_table(table: ComparisonTable) -> str:
    """Markdown rendering for the agent / history file."""
    lines = [
        f"Behavioral contribution is expressed relative to `{table.reference}`. "
        "`positive_behavioral_contribution` sums the row-wise score differences "
        "where the branch beats the reference (rows where it is worse are not "
        "subtracted — use `comparison_score` for the overall picture). "
        f"Eligibility is vs the primary (`{table.primary}`) with threshold "
        f"{table.threshold:.4f} on the normalized (per-row) contribution.",
        "",
        "| branch | comparison_score | positive_behavioral_contribution | normalized | "
        "n_rows_ahead | pareto_score | up_to_date | eligible | is_primary |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in table.rows:
        name = f"`{r.branch}`" + (" (reference)" if r.is_reference else "")
        score = "n/a" if r.comparison_score is None else f"{r.comparison_score:.4f}"
        pareto = (
            "n/a - fully dominated by other programs"
            if r.pareto_dominated
            else str(r.pareto_score)
        )
        lines.append(
            f"| {name} | {score} | {r.positive_behavioral_contribution:.4f} | "
            f"{r.positive_behavioral_contribution_normalized:.4f} | {r.n_rows_ahead} | "
            f"{pareto} | {str(r.up_to_date).lower()} | {str(r.eligible).lower()} | "
            f"{str(r.is_primary).lower()} |"
        )
    stale = [r.branch for r in table.rows if not r.up_to_date]
    if stale:
        lines.append("")
        lines.append(
            "Out-of-date comparison records (branch tip moved since the record; "
            "still used): " + ", ".join(f"`{b}`" for b in stale)
        )
    notes = [f"`{r.branch}`: {r.note}" for r in table.rows if r.note]
    if notes:
        lines.append("")
        lines.extend(notes)
    if table.missing:
        lines.append("")
        lines.append(
            "Not compared (no comparison-set record yet — run "
            "`run_eval(eval_sample=\"comparison\")` on them to include): "
            + ", ".join(f"`{b}`" for b in table.missing)
        )
    return "\n".join(lines)


def rows_as_dicts(table: ComparisonTable) -> list[dict[str, Any]]:
    return [
        {
            "branch": r.branch,
            "is_primary": r.is_primary,
            "is_reference": r.is_reference,
            "comparison_score": r.comparison_score,
            "positive_behavioral_contribution": r.positive_behavioral_contribution,
            "positive_behavioral_contribution_normalized": r.positive_behavioral_contribution_normalized,
            "n_rows_ahead": r.n_rows_ahead,
            "pareto_score": r.pareto_score,
            "pareto_dominated": r.pareto_dominated,
            "up_to_date": r.up_to_date,
            "eligible": r.eligible,
            "record_iteration": r.record_iteration,
        }
        for r in table.rows
    ]


__all__ = [
    "ComparisonTable",
    "Contribution",
    "LineageRecord",
    "LineageRow",
    "build_table",
    "contribution_threshold",
    "eligible_set",
    "format_table",
    "lineage_record",
    "load_comparison_records",
    "positive_contribution",
    "rows_as_dicts",
]
