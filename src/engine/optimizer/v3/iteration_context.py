"""Copyright © 2026 440 Labs LLC

`IterationContext` — engine 3's per-iteration state.

Extends the shared `BaseIterationContext` (`src/engine/schemas/iteration_context.py`,
the eval-tool + harness contract every arm uses) with the closing-protocol
state and the logic that interprets it: parent linkage, the `close_iteration`
payload, deterministic closing outputs, and `finalize()` /
`reconcile_eval_shas()` / `score_for_content()` / `history_fields()`. The
simple-loop baseline constructs the base class and never touches any of this.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Callable

from src.engine.schemas.iteration_context import (
    TRAIN_EVAL_SAMPLES,
    BaseIterationContext,
    EvalRecord,
)

from .records import RegisteredBranch

# `equivalent(parent_tree, close_tree) -> bool`: program-equivalence predicate
# injected into finalize()/score_for_content so they stay pure and testable
# without a git workspace (production wires `program_paths.diff_is_non_program_only`).
ProgramEquivalenceFn = Callable[[str | None, str | None], bool]


@dataclass
class FinalizeResult:
    best_subsample_score: float | None
    comparison_best_score: float | None
    changes_landed: bool


@dataclass(kw_only=True)
class IterationContext(BaseIterationContext):
    """Per-iteration state for the EE architect session.

    Base fields (eval log, subsample lock, budget snapshot, token usage) live
    on `BaseIterationContext`. The fields below are engine-3 closing state,
    written by the tools / loop as the session progresses.
    """

    # EVO-only: the pareto_evolution preselect stamps its exploit-and-merge
    # draw here; True flips both best-candidate selection and the merge note.
    exploit_phase: bool = False

    # Iteration-start snapshot (agent-managed-git mode only).
    start_branch: str | None = None
    start_sha: str | None = None
    primary_branch_at_start: str | None = None
    memory_md_hash_at_start: str | None = None
    # Set by the explore-frequency guard: this iteration MUST explore.
    must_explore: bool = False

    # Parent linkage. Preselect: set by `on_iteration_start`. Agent-managed-git:
    # derived POST-SESSION by `closing.derive_iteration_parent`.
    parent_sha: str | None = None
    parent_branch: str | None = None
    child_branch: str | None = None

    # Coding agent state.
    coding_agent_called: bool = False
    change_requests: list[str] = field(default_factory=list)

    # close_iteration payload — written only by the close_iteration handler.
    close_called: bool = False
    registered_branch: RegisteredBranch | None = None
    claimed_improvement_mode: str | None = None
    memory_updated_claimed: bool = False

    # Deterministic closing outputs (loop-written, post-session).
    close_sha: str | None = None
    close_tree: str | None = None
    parent_iteration: int | None = None
    parent_is_primary: bool | None = None
    new_lineage: bool | None = None
    parent_derivation: str | None = None  # "ancestor" | "merge_base" | "seed"
    parent_tree: str | None = None
    closed_by_fallback: bool = False

    # Every `codeevolver-*` branch in the workspace at close, {branch: sha}.
    branches_seen: dict[str, str] = field(default_factory=dict)

    # The iteration produced nothing (architect raised / no branch registered).
    skipped: bool = False

    # Filled by `finalize()` on iteration exit.
    changes_landed: bool = False
    best_subsample_score: float | None = None
    best_subsample_tree: str | None = None
    best_subsample_sha: str | None = None
    comparison_best_score: float | None = None
    comparison_best_tree: str | None = None
    comparison_best_sha: str | None = None
    comparison_state: str | None = None
    comparison_eval_id_at_close: str | None = None
    comparison_score_at_close: float | None = None

    watchdog_fired: bool = False

    # Run-variance estimate (specs/archive-public/sampling_variance.md). `variance_estimate`
    # is the run-level snapshot stamped at construction (None until the
    # architect has called `run_variance_estimate` in some iteration) — read
    # by the prompt builders and the revert-check tolerance. `..._ran` marks
    # the iteration that took it; `..._samples` is the configured run count,
    # for the prompt text.
    variance_estimate: dict[str, Any] | None = None
    variance_estimate_ran: bool = False
    variance_estimate_samples: int = 5

    # ASA (specs/archive-public/v3_asa.md) — stamped by `asa_on_iteration_start`; every
    # other arm leaves these at their defaults, which render nothing.
    # Fired requirements: subset of {"new_lineage", "eligible_nonprimary"}.
    asa_requirements: list[str] = field(default_factory=list)
    # Eligible non-primary lineages at iteration start (frozen for the
    # iteration) and their normalized positive contribution vs the primary.
    asa_eligible_at_start: list[str] = field(default_factory=list)
    asa_eligible_contributions_at_start: dict[str, float] = field(default_factory=dict)
    asa_allowance_left: int = 0
    asa_free_tail: bool = False
    asa_phase1: bool = False
    # The primary's comparison record is older than the stale threshold
    # (landed iterations on the primary); the close hook refreshes it.
    asa_primary_record_stale: bool = False
    # The run_eval gate let a non-compliant baseline through on the
    # iteration's allowance (holds for the rest of the iteration).
    asa_violation_granted: bool = False

    def comparison_record_for_tree(
        self, equivalent: ProgramEquivalenceFn
    ) -> EvalRecord | None:
        """Newest comparison eval whose content is the closing content.

        Same attribution hardness as `finalize()`: an exact `close_tree`
        match wins, else a program-equivalent tree (documentation-only
        differences). None when no comparison eval matches — the
        pareto_evolution loop then runs the fitness eval itself.
        """
        if not self.close_tree:
            return None
        for rec in reversed(self.eval_records):
            if rec.get("eval_sample") != "comparison":
                continue
            tree = rec.get("tree")
            if not tree:
                continue
            if tree == self.close_tree or equivalent(tree, self.close_tree):
                return rec
        return None

    def reconcile_eval_shas(self, commit_trees: Sequence[tuple[str, str]]) -> int:
        """Re-point eval records at the commit that actually holds their code.

        Returns the number of records rewritten.

        `run_eval` stamps each record with HEAD's sha, but the evaluator runs
        the WORKING TREE — so an eval of uncommitted edits is recorded under
        the parent commit. Once that content is committed, the new commit's
        tree equals the record's `tree`, and this promotes the record's `sha`
        to the real commit.

        `finalize()` no longer depends on this (it matches trees, not shas),
        but the untested-HEAD and revert-check watchdogs, the close gate's
        fast path, and the eval audit trail all still read record shas — an
        un-promoted record makes them reason about the parent commit rather
        than the one holding the code that was scored.

        `commit_trees` is `[(sha, tree), ...]` NEWEST FIRST (see
        `subprocess_service.git_commit_trees`); the newest commit wins per
        tree so an in-iteration revert that restores an earlier tree does not
        drag a record backwards in history.

        Called at every dirty->committed boundary: inside the
        `close_iteration` tool (between the clean-tree check and the
        was-it-evaluated check), post-session after the drift commit, and once
        more in the loop after `close_sha` resolves. Everywhere else, plain
        sha comparison is correct — that is the whole point of correcting the
        sha rather than teaching every reader to match two keys.

        Idempotent — safe to call after every commit-producing step. Records
        with no `tree` (git failure or legacy) are left untouched.
        """
        if not commit_trees or not self.eval_records:
            return 0
        tree_to_sha: dict[str, str] = {}
        for sha, tree in commit_trees:  # newest first
            if tree:
                tree_to_sha.setdefault(tree, sha)

        rewritten = 0
        for rec in self.eval_records:
            tree = rec.get("tree")
            if not tree:
                continue
            target = tree_to_sha.get(tree)
            if target and rec.get("sha") != target:
                # Preserve the pre-reconciliation sha for run-audit forensics
                # (comparison_set_history.jsonl is written synchronously and
                # cannot be retro-edited, so the two can legitimately differ).
                rec.setdefault("sha_at_eval", rec.get("sha"))
                rec["sha"] = target
                rec["committed"] = True
                rewritten += 1
        if rewritten:
            # eval_history is the (sha, score) projection of the train/single
            # records; rebuild it with the same filter `record_eval` uses so
            # the two can never drift apart.
            self.eval_history = [
                (r["sha"], r["score"])
                for r in self.eval_records
                if r.get("eval_sample") in TRAIN_EVAL_SAMPLES
                and r.get("sha")
            ]
        return rewritten

    def record_coding_agent(self, change_request: str) -> None:
        self.coding_agent_called = True
        self.change_requests.append(change_request)

    def _train_records(self) -> list[EvalRecord]:
        """Train-sample eval records in call order (comparison evals excluded)."""
        return [
            rec for rec in self.eval_records
            if rec.get("eval_sample") in TRAIN_EVAL_SAMPLES
        ]

    def finalize(self, equivalent: ProgramEquivalenceFn) -> FinalizeResult:
        """Decide what this iteration measured, and whether the program moved.

        The end state is `close_sha` — the registered branch's HEAD, the
        content the next iteration inherits. It is NOT chosen by searching the
        eval log, and no score is attributed to it here.

        Why no "score of the closing content": the architect may close on a
        commit it never re-measured (a formatter run after the last eval, a
        `codeevolver.md` edit, a small follow-up), and evaluated-but-never-
        committed states cannot be ordered against the closing commit by git
        ancestry (a dirty-tree eval leaves only a tree object) or by time (an
        abandoned experiment evaluated after the closing content was authored
        but before it was committed looks identical). Walking `eval_history`
        for the newest sha reachable from the branch mis-attributes reverted
        evals. The per-arm answer:

        - Explore/exploit (ee): the iteration is described by its BEST train
          eval on the locked baseline subsample (`best_subsample_score`) and
          its best comparison eval on the registered branch
          (`comparison_best_score`). Progress readings, not head certificates.
        - pareto_evolution: admission is `best > baseline`; the fitness of the
          closing content is then measured by content — an exact /
          program-equivalent comparison eval (`comparison_record_for_tree`)
          or a loop re-run — and stamped on the context by the close hook.

        `equivalent(a, b)` is injected so this method stays pure and testable
        without a git workspace; production wires it to
        `program_paths.diff_is_non_program_only`. It is used here only for
        `changes_landed`.
        """
        train = self._train_records()

        # Best train eval on the baseline's subsample. `single` rows and
        # train evals drawn on a different sample (a ceiling baseline that was
        # redrawn) are never comparable to the baseline, so they are excluded
        # rather than allowed to masquerade as progress.
        best = None
        for rec in train:
            score = rec.get("score")
            if score is None or rec.get("eval_sample") == "single":
                continue
            if (
                self.baseline_sample_key is not None
                and rec.get("sample_key") is not None
                and rec.get("sample_key") != self.baseline_sample_key
            ):
                continue
            if best is None or score > best.get("score", float("-inf")):
                best = rec
        self.best_subsample_score = best.get("score") if best else None
        self.best_subsample_tree = best.get("tree") if best else None
        self.best_subsample_sha = _durable_ref(best) if best else None

        # Best comparison eval of the registered branch. Records carry the
        # branch that was checked out at eval time; a comparison run on another
        # branch (promotion evidence for the incumbent) is that branch's
        # reading, not this iteration's.
        cmp_best = None
        for rec in self.eval_records:
            if rec.get("eval_sample") != "comparison":
                continue
            score = rec.get("score")
            if score is None:
                continue
            if self.child_branch and rec.get("branch") not in (None, self.child_branch):
                continue
            if cmp_best is None or score > cmp_best.get("score", float("-inf")):
                cmp_best = rec
        self.comparison_best_score = cmp_best.get("score") if cmp_best else None
        self.comparison_best_tree = cmp_best.get("tree") if cmp_best else None
        self.comparison_best_sha = _durable_ref(cmp_best) if cmp_best else None

        if self.child_branch is None or self.close_sha is None:
            # No branch was ever registered. Nothing landed; the loop marks
            # this iteration `skipped`.
            self.changes_landed = False
            return FinalizeResult(
                self.best_subsample_score, self.comparison_best_score, False
            )

        # Did the PROGRAM change? Content question, not eval bookkeeping.
        # Degrade to sha inequality when either tree could not be read, so a
        # git failure never silently reports "nothing changed".
        if self.close_tree and self.parent_tree:
            self.changes_landed = not equivalent(self.parent_tree, self.close_tree)
        else:
            self.changes_landed = bool(
                self.parent_sha and self.close_sha != self.parent_sha
            )

        return FinalizeResult(
            self.best_subsample_score,
            self.comparison_best_score,
            self.changes_landed,
        )

    def score_for_content(
        self, tree: str | None, equivalent: ProgramEquivalenceFn
    ) -> tuple[float | None, str]:
        """Train score of exactly this content, by tree.

        Newest-first over train evals: an eval on this exact tree wins
        (`"exact_tree"`), else one on a program-equivalent tree
        (`"program_equivalent"`), else `(None, "none")`. A pure content
        lookup — no ancestry, no "nearest before" — for anything that needs
        "was THIS program measured?".
        """
        if not tree:
            return None, "none"
        for rec in reversed(self._train_records()):
            rec_tree = rec.get("tree")
            if not rec_tree:
                continue
            if rec_tree == tree:
                return rec.get("score"), "exact_tree"
            if equivalent(rec_tree, tree):
                return rec.get("score"), "program_equivalent"
        return None, "none"

    def refresh_durable_refs(self) -> None:
        """Re-resolve `best_subsample_sha` / `comparison_best_sha` after the
        loop has anchored never-committed trees (`anchor_ref` stamped on the
        records). Idempotent."""
        for rec in self.eval_records:
            if rec.get("tree") and rec.get("tree") == self.best_subsample_tree:
                self.best_subsample_sha = _durable_ref(rec) or self.best_subsample_sha
            if (
                rec.get("eval_sample") == "comparison"
                and rec.get("tree")
                and rec.get("tree") == self.comparison_best_tree
            ):
                self.comparison_best_sha = (
                    _durable_ref(rec) or self.comparison_best_sha
                )

    def history_fields(self) -> dict[str, Any]:
        """Produce the subset of fields to write into `IterationHistoryRecord`."""
        return {
            "git_branch": self.child_branch,
            "selected_component": (
                "_code" if (self.coding_agent_called or self.changes_landed) else None
            ),
            "change_request": (
                "\n---\n".join(self.change_requests) if self.change_requests else None
            ),
            "close_tree": self.close_tree,
            "changes_landed": self.changes_landed,
            "best_subsample_score": self.best_subsample_score,
            "best_subsample_sha": self.best_subsample_sha,
            "comparison_best_score": self.comparison_best_score,
            "comparison_best_sha": self.comparison_best_sha,
            "comparison_state": self.comparison_state,
            "comparison_eval_id_at_close": self.comparison_eval_id_at_close,
            "comparison_score_at_close": self.comparison_score_at_close,
            "baseline_ran_on_modified_tree": self.baseline_tree_dirty,
            "parent_branch": self.parent_branch,
            "parent_sha": self.parent_sha,
            "parent_iteration": self.parent_iteration,
            "parent_is_primary": self.parent_is_primary,
            "parent_derivation": self.parent_derivation,
            "new_lineage": self.new_lineage,
            "improvement_mode": self.claimed_improvement_mode,
            "memory_updated": self.memory_updated_claimed,
            "closed_by_fallback": self.closed_by_fallback,
            "parent_subsample_score": self.initial_subsample_score,
        }


def _durable_ref(rec: EvalRecord | None) -> str | None:
    """A checkout-able pointer at an eval record's content.

    The record's `sha` when that commit actually holds the evaluated tree
    (`committed`), else the `refs/codeevolver/evals/<N>/<i>` anchor the loop
    stamps as `anchor_ref`, else — before anchoring — the sha at eval time
    (the parent commit; better than nothing for a log line).
    """
    if not rec:
        return None
    if rec.get("committed") and rec.get("sha"):
        return rec["sha"]
    if rec.get("anchor_ref"):
        return rec["anchor_ref"]
    return rec.get("sha")


__all__ = [
    "IterationContext",
    "BaseIterationContext",
    "EvalRecord",
    "FinalizeResult",
    "ProgramEquivalenceFn",
    "TRAIN_EVAL_SAMPLES",
]
