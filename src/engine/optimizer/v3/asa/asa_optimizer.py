"""Copyright © 2026 440 Labs LLC

ASA — Artificial Selection Algorithm (Treatment C in
`specs/research_plan.md`; design in `specs/archive-public/v3_asa.md`).

ASA is the EE arm plus a REQUIREMENT to traverse the population tree.
Everything the agent sees — prompt skeleton, skills, close tool, explore
guard, watchdogs, auditors — is EE's, and the deltas are exactly:

- `on_iteration_start`: compute the eligible non-primary lineages (positive
  behavioral contribution vs the primary above a threshold) and the fired
  requirements (rule 1: new lineages at 40% during phase 1; rule 2: eligible
  non-primary parent at 30% while eligible lineages exist); stamp them on the
  context so EE's prompt builder / user message append the rule sections.
- `run_eval_gate`: at the iteration's baseline, check the current branch
  against the fired requirements — warn while the violation allowance
  (`ceil(closed/10)`) holds, refuse otherwise (no rows spent).
- `extra_tools`: the read-only `branch_comparison` tool.
- `on_seed_complete`: register the seed as its own non-primary lineage
  (`codeevolver-{ts}-seed`) alongside the `-main` primary, and run ONE
  counted comparison eval that represents both (same tree).
- `on_iteration_closed`: derive satisfied / violation from the closing facts;
  run a counted comparison eval when a branch was promoted without a record
  for its tip, or when the primary's record is stale (> k landed iterations).
- `commit_before_comparison`: `run_eval` commits a dirty tree before a
  comparison eval so the JSONL sha names the content.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Mapping

from src.engine.paths import history_dir
from src.engine.services.subprocess_service import (
    git_checkout,
    git_create_branch_simple,
    git_get_current_branch,
    git_get_sha,
    git_is_ancestor,
    git_merge_base,
    git_push_authenticated,
    git_tree_signature,
)

from ..closing import comparison_entries, derive_iteration_parent
from ..errors import RunEvalError
from ..iteration_context import IterationContext
from ..loop_eval import run_loop_comparison_eval
from ..optimizer_loop import run_optimization
from ..program_paths import diff_is_non_program_only
from ..prompts import build_architect_system_prompt
from ..strategy import LoopServices, OptimizerStrategy
from .branch_comparison import (
    ComparisonTable,
    LineageRecord,
    build_table,
    contribution_threshold,
    eligible_set,
    format_table,
    lineage_record,
    load_comparison_records,
    rows_as_dicts,
)
from .prompt_sections import (
    GATE_NOTE_ELIGIBLE_NONPRIMARY,
    GATE_NOTE_NEW_LINEAGE,
    REQ_ELIGIBLE_NONPRIMARY,
    REQ_NEW_LINEAGE,
)
from .rules import plan_iteration, primary_record_is_stale

if TYPE_CHECKING:
    from ..ce_state import CEState
    from ..records import CloseRecord
    from src.engine.schemas.iteration_history import IterationHistory

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
BRANCH_PREFIX = "codeevolver-"

DEFAULT_CONTRIBUTION_CONSTANT = 0.10
DEFAULT_CONTRIBUTION_CAP = 0.05
DEFAULT_STALE_LANDED_ITERATIONS = 4


# ----------------------------------------------------------------------
# Config / git helpers
# ----------------------------------------------------------------------
def _knobs(services: LoopServices) -> dict[str, Any]:
    cfg = services.config
    return {
        "constant": getattr(cfg, "asa_behavioral_contribution_constant", DEFAULT_CONTRIBUTION_CONSTANT),
        "cap": getattr(cfg, "asa_behavioral_contribution_threshold_cap", DEFAULT_CONTRIBUTION_CAP),
        "stale_k": getattr(cfg, "asa_primary_stale_landed_iterations", DEFAULT_STALE_LANDED_ITERATIONS),
        "max_score": float(getattr(cfg, "max_score", 1.0) or 1.0),
        "subsample_size": int(getattr(cfg, "subsample_size", 20) or 20),
    }


def _equivalent(workspace: str) -> Callable[[str, str], bool]:
    return lambda a, b: diff_is_non_program_only(workspace, a, b)


def _tip_tree(workspace: str, branch: str) -> str | None:
    try:
        return git_tree_signature(workspace, branch)
    except Exception:  # noqa: BLE001
        return None


def lineages_for(
    state: "CEState",
    workspace: str,
    *,
    extra_branches: set[str] | None = None,
) -> dict[str, LineageRecord | None]:
    """`branch -> representing comparison record` for every registered
    lineage (plus `extra_branches`)."""
    records = load_comparison_records(state.comparison_history_path)
    eq = _equivalent(workspace)
    names = set(state.branches) | set(extra_branches or ())
    return {
        b: lineage_record(b, records, tip_tree=_tip_tree(workspace, b), equivalent=eq)
        for b in sorted(names)
    }


def _threshold(lineages: Mapping[str, LineageRecord | None], primary: str, knobs: Mapping[str, Any]) -> float:
    primary_rec = lineages.get(primary)
    return contribution_threshold(
        primary_comparison_score=primary_rec.record.get("score") if primary_rec else None,
        max_score=knobs["max_score"],
        constant=knobs["constant"],
        cap=knobs["cap"],
    )


def _comparison_set_size(ctx: IterationContext, services: LoopServices) -> int:
    return int(ctx.comparison_set_size or len(services.comparison_ids) or 0)


# ----------------------------------------------------------------------
# Seed → two lineages, one counted comparison eval
# ----------------------------------------------------------------------
def seed_lineage_name(seed_branch: str) -> str:
    return seed_branch[: -len("-main")] + "-seed" if seed_branch.endswith("-main") else seed_branch + "-seed"


def asa_on_seed_complete(
    state: "CEState",
    history: "IterationHistory",
    services: LoopServices,
) -> None:
    if not services.comparison_ids:
        raise RuntimeError("asa requires a comparison set (comparison_set_size > 0)")
    ws = services.workspace
    seed_lineage = seed_lineage_name(state.seed_branch)

    # The seed program is a legitimate (often the simplest) lineage: keep it
    # as its own non-primary branch so later primaries can be compared to it
    # and it can become an eligible parent.
    git_checkout(ws, state.seed_branch)
    git_create_branch_simple(ws, seed_lineage)
    git_checkout(ws, state.seed_branch)
    state.register_lineage(
        seed_lineage, sha=git_get_sha(ws, seed_lineage), iteration=0, is_primary=False,
    )

    ctx = IterationContext(
        iteration_idx=0,
        comparison_set_size=len(services.comparison_ids),
        max_eval_rows=services.max_eval_rows,
        evals_used=int(state.total_num_evals or 0),
        primary_branch_at_start=state.primary_branch,
        parent_locked=True,
    )
    result = run_loop_comparison_eval(ctx=ctx, state=state, services=services, ref=state.seed_branch)
    rec = ctx.eval_records[-1]
    if 0 in history:
        history[0]["comparison_score"] = float(result.primary_score)
        history[0]["comparison_scores"] = [
            {
                "branch": rec.get("branch"),
                "sha": rec.get("sha"),
                "score": rec.get("score"),
                "n_rows": rec.get("n_rows"),
            }
        ]
    logger.info(
        "asa: seed lineage %s registered beside primary %s; one comparison eval "
        "(score=%.4f, %d rows counted) represents both",
        seed_lineage, state.seed_branch, result.primary_score, len(services.comparison_ids),
    )

    job = getattr(services.config, "job", None)
    if job is not None and getattr(job, "repo_url", None):
        try:
            push = git_push_authenticated(
                workspace=ws, branch=seed_lineage, **job.git_creds(), set_upstream=True,
            )
            if not push.get("success"):
                logger.warning("asa: push of %s failed (continuing): %s", seed_lineage, push.get("stderr"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("asa: push of %s failed (continuing): %s", seed_lineage, exc)


# ----------------------------------------------------------------------
# Iteration start — eligible set + fired requirements
# ----------------------------------------------------------------------
def _write_table_file(iteration_idx: int, text: str) -> None:
    try:
        path = os.path.join(history_dir(), f"branch_comparison_iter{iteration_idx}.md")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            f.write(text)
    except OSError:
        logger.warning("asa: could not write branch_comparison_iter%d.md", iteration_idx, exc_info=True)


def asa_on_iteration_start(
    state: "CEState",
    ctx: IterationContext,
    services: LoopServices,
) -> None:
    ws = services.workspace
    knobs = _knobs(services)
    primary = state.primary_branch
    lineages = lineages_for(state, ws)
    threshold = _threshold(lineages, primary, knobs)
    size = _comparison_set_size(ctx, services)
    eligible = eligible_set(lineages=lineages, primary=primary, comparison_set_size=size, threshold=threshold)

    plan = plan_iteration(
        state.iterations,
        eligible=list(eligible),
        evals_used=int(ctx.evals_used or 0),
        max_eval_rows=services.max_eval_rows,
        subsample_size=knobs["subsample_size"],
    )
    ctx.asa_requirements = list(plan.requirements)
    ctx.asa_eligible_at_start = sorted(eligible)
    ctx.asa_eligible_contributions_at_start = dict(eligible)
    ctx.asa_allowance_left = plan.allowance_left
    ctx.asa_free_tail = plan.free_tail
    ctx.asa_phase1 = plan.phase1

    primary_rec = lineages.get(primary)
    ctx.asa_primary_record_stale = primary_record_is_stale(
        state.iterations,
        primary=primary,
        record_iteration=primary_rec.record.get("iteration") if primary_rec else None,
        stale_after_landed=int(knobs["stale_k"]),
    )

    try:
        table = build_table(
            lineages=lineages, reference=primary, primary=primary,
            comparison_set_size=size, threshold=threshold,
        )
        text = format_table(table)
    except ValueError as exc:
        text = f"(no table: {exc})"
    header = (
        f"# branch_comparison — iteration {ctx.iteration_idx} start\n\n"
        f"phase1={plan.phase1} free_tail={plan.free_tail} closed={plan.n_closed} "
        f"new_lineage_rate={plan.rule1_rate} eligible_window={plan.n_eligible_window} "
        f"eligible_parent_rate={plan.rule2_rate} allowance_left={plan.allowance_left} "
        f"requirements={plan.requirements} primary_record_stale={ctx.asa_primary_record_stale}\n\n"
    )
    _write_table_file(ctx.iteration_idx, header + text + "\n")
    logger.info(
        "asa iteration %d: eligible=%s requirements=%s (phase1=%s free_tail=%s "
        "closed=%d rule1_rate=%s rule2_rate=%s allowance_left=%d)",
        ctx.iteration_idx, ctx.asa_eligible_at_start, plan.requirements, plan.phase1,
        plan.free_tail, plan.n_closed, plan.rule1_rate, plan.rule2_rate, plan.allowance_left,
    )


# ----------------------------------------------------------------------
# run_eval gate (baseline only — the binding restricts the call)
# ----------------------------------------------------------------------
def _head_parent_branch(state: "CEState", workspace: str) -> str | None:
    """Branch whose last close is an ANCESTOR of HEAD (or HEAD itself).

    The merge-base fallback of `derive_iteration_parent` is deliberately not
    honoured here: a fresh fork from the primary shares a merge-base with
    every lineage, so it would attribute the fork to whichever eligible
    branch closed most recently and let rule 2 pass on primary content.
    """
    try:
        head = git_get_sha(workspace, "HEAD")
    except Exception:  # noqa: BLE001
        return None
    derived = derive_iteration_parent(
        close_sha=head,
        past_iterations=state.iterations,
        seed_sha=state.seed_sha,
        seed_branch=state.seed_branch,
        is_ancestor=lambda a, d: git_is_ancestor(workspace, a, d),
        merge_base=lambda a, b: git_merge_base(workspace, a, b),
    )
    return derived.parent_branch if derived.via == "ancestor" else None


def asa_run_eval_gate(
    ctx: IterationContext,
    state: "CEState",
    workspace: str,
    eval_sample: str,
) -> str | None:
    reqs = list(ctx.asa_requirements or [])
    if not reqs:
        return None
    try:
        branch = git_get_current_branch(workspace)
    except Exception:  # noqa: BLE001
        branch = None

    unmet: list[str] = []
    if REQ_NEW_LINEAGE in reqs:
        is_new = (
            branch is not None
            and branch.startswith(BRANCH_PREFIX)
            and branch not in state.known_branches()
        )
        if not is_new:
            unmet.append(GATE_NOTE_NEW_LINEAGE)
    if REQ_ELIGIBLE_NONPRIMARY in reqs:
        eligible = set(ctx.asa_eligible_at_start or [])
        ok = branch in eligible
        if not ok and branch is not None:
            ok = _head_parent_branch(state, workspace) in eligible
        if not ok:
            unmet.append(
                GATE_NOTE_ELIGIBLE_NONPRIMARY
                + " Eligible: " + ", ".join(sorted(eligible))
            )
    if not unmet:
        return None
    msg = " ".join(unmet)
    if ctx.asa_violation_granted or ctx.asa_allowance_left > 0:
        if not ctx.asa_violation_granted:
            ctx.asa_violation_granted = True
            logger.info(
                "asa iteration %d: baseline on %s does not meet %s — proceeding on the "
                "violation allowance (%d left at iteration start)",
                ctx.iteration_idx, branch, reqs, ctx.asa_allowance_left,
            )
        return (
            "WARNING — this iteration's ASA requirement is not met by the current "
            "branch; the eval proceeds on the run's violation allowance, but the "
            "iteration will be recorded as a violation. " + msg
        )
    logger.info(
        "asa iteration %d: baseline on %s refused — requirement %s unmet and no allowance left",
        ctx.iteration_idx, branch, reqs,
    )
    raise RunEvalError("run_eval refused — this iteration's ASA requirement is not met. " + msg)


# ----------------------------------------------------------------------
# Close — derived facts + counted comparison evals
# ----------------------------------------------------------------------
def _run_counted_comparison(
    ctx: IterationContext, state: "CEState", services: LoopServices, rec: "CloseRecord", *, ref: str, reason: str,
) -> None:
    result = run_loop_comparison_eval(ctx=ctx, state=state, services=services, ref=ref)
    state.note_loop_comparison_rerun(len(services.comparison_ids))
    # `comparison_scores` was projected before this hook ran (closing.py).
    rec["comparison_scores"] = comparison_entries(ctx)
    rec["asa_loop_comparison_reason"] = reason
    logger.info(
        "asa iteration %d: loop comparison eval on %s (%s) score=%.4f, %d rows counted",
        ctx.iteration_idx, ref, reason, result.primary_score, len(services.comparison_ids),
    )


def asa_on_iteration_closed(
    ctx: IterationContext,
    state: "CEState",
    rec: "CloseRecord",
    services: LoopServices,
) -> None:
    ws = services.workspace
    reqs = list(ctx.asa_requirements or [])
    eligible = set(ctx.asa_eligible_at_start or [])
    parent_is_eligible = bool(ctx.parent_branch and ctx.parent_branch in eligible)
    met = {
        REQ_NEW_LINEAGE: bool(ctx.new_lineage),
        REQ_ELIGIBLE_NONPRIMARY: parent_is_eligible,
    }
    satisfied = all(met.get(r, False) for r in reqs)
    violation = bool(reqs) and not satisfied

    rec["asa_requirements"] = reqs
    rec["asa_satisfied"] = satisfied
    rec["asa_violation"] = violation
    rec["asa_eligible_at_start"] = sorted(eligible)
    rec["asa_parent_is_eligible"] = parent_is_eligible
    rec["asa_parent_positive_contribution_to_primary_normalized"] = (
        None
        if ctx.parent_is_primary or not ctx.parent_branch
        else ctx.asa_eligible_contributions_at_start.get(ctx.parent_branch)
    )
    rec["asa_allowance_left"] = int(ctx.asa_allowance_left)
    rec["asa_free_tail"] = bool(ctx.asa_free_tail)
    rec["asa_phase1"] = bool(ctx.asa_phase1)
    rec["asa_loop_comparison_reason"] = None
    if violation:
        logger.info("asa iteration %d: requirement %s NOT met (parent=%s new_lineage=%s)",
                    ctx.iteration_idx, reqs, ctx.parent_branch, ctx.new_lineage)

    if ctx.skipped:
        return

    # specs/archive-public/comparison_credits.md: when the agent already ran a comparison
    # eval this iteration, the loop never triggers a second run — the agent's
    # reading stands (even if it is not tip-matching; recorded as skipped).
    agent_ran_comparison = any(
        r.get("eval_sample") == "comparison" and not r.get("by_loop")
        for r in ctx.eval_records
    )

    # Promotion: judged on `primary_branch_after`, which the constraint gate
    # resets on a force-revert (`is_primary` is not).
    primary_after = rec.get("primary_branch_after")
    promoted = bool(primary_after) and primary_after != ctx.primary_branch_at_start
    if promoted:
        records = load_comparison_records(state.comparison_history_path)
        lr = lineage_record(
            primary_after, records, tip_tree=_tip_tree(ws, primary_after), equivalent=_equivalent(ws),
        )
        if lr is None or not lr.up_to_date:
            if agent_ran_comparison:
                rec["asa_loop_comparison_reason"] = "skipped_agent_ran_comparison"
                logger.info(
                    "asa iteration %d: promotion of %s lacks a tip-matching "
                    "comparison record, but the agent already ran a comparison "
                    "this iteration — loop run skipped",
                    ctx.iteration_idx, primary_after,
                )
            else:
                _run_counted_comparison(ctx, state, services, rec, ref=primary_after, reason="promotion")
        return

    if ctx.asa_primary_record_stale and not ctx.asa_free_tail and ctx.primary_branch_at_start:
        if agent_ran_comparison:
            rec["asa_loop_comparison_reason"] = "skipped_agent_ran_comparison"
            logger.info(
                "asa iteration %d: primary record stale, but the agent already "
                "ran a comparison this iteration — loop refresh skipped",
                ctx.iteration_idx,
            )
            return
        _run_counted_comparison(
            ctx, state, services, rec, ref=ctx.primary_branch_at_start, reason="stale_primary",
        )


# ----------------------------------------------------------------------
# branch_comparison tool
# ----------------------------------------------------------------------
def branch_comparison_tool(
    *,
    ctx: IterationContext,
    state: "CEState",
    workspace: str,
    services: LoopServices,
    reference: str | None = None,
) -> dict[str, Any]:
    """Read-only. Recomputes from the JSONL (comparison evals append there
    synchronously, so one run mid-iteration shows up immediately)."""
    primary = state.primary_branch
    ref = (reference or "").strip() or primary
    records = load_comparison_records(state.comparison_history_path)
    seen_in_records = {r.get("branch") for r in records if r.get("branch")}
    current = None
    try:
        current = git_get_current_branch(workspace)
    except Exception:  # noqa: BLE001
        pass
    extra = {b for b in seen_in_records | {ref, current} if b and b.startswith(BRANCH_PREFIX)}
    lineages = lineages_for(state, workspace, extra_branches=extra)
    threshold = _threshold(lineages, primary, _knobs(services))
    size = _comparison_set_size(ctx, services)
    try:
        table: ComparisonTable = build_table(
            lineages=lineages, reference=ref, primary=primary, comparison_set_size=size, threshold=threshold,
        )
    except ValueError as exc:
        raise RunEvalError(str(exc)) from exc
    return {
        "ok": True,
        "reference": ref,
        "primary": primary,
        "threshold": threshold,
        "table": format_table(table),
        "rows": rows_as_dicts(table),
        "not_compared": table.missing,
        "eligible": sorted(table.eligible),
    }


def asa_extra_tools(
    ctx: IterationContext, state: "CEState", workspace: str, services: LoopServices,
) -> Mapping[str, Callable[..., dict]]:
    def branch_comparison(reference: str | None = None) -> dict:
        return branch_comparison_tool(
            ctx=ctx, state=state, workspace=workspace, services=services, reference=reference,
        )

    return {"branch_comparison": branch_comparison}


# Acknowledgement note before a comparison run (specs/archive-public/comparison_credits.md
# §ASA): the standard cost framing plus what the comparison buys in this arm.
ASA_COMPARISON_ACK_NOTE = (
    "The comparison set depletes budget at {comparison_set_size} rows per "
    "run, compared to {subsample_size} rows for a training subsample. "
    "Additionally, there are a fixed number of comparison set credits "
    "replenishing at the rate of one comparison set run per iteration. "
    "Remaining credits: {credits}. Remaining eval-row budget: "
    "{rows_remaining}. Please acknowledge that you would like to continue "
    "running the comparison set.\n\n"
    "In addition, future agents will pick up your work based on the results "
    "of the comparison set, including whether it demonstrates behavioral "
    "advantages over other branches (via the branch_comparison tool). Given "
    "the fixed number of credits, it is best practice to run the comparison "
    "set at the end of your improvement run."
)


ASA_STRATEGY = OptimizerStrategy(
    method="asa",
    prompt_builder=build_architect_system_prompt,   # EE's prompt, unchanged
    skills_to_stage=("close_iteration", "exploration"),
    skill_overrides={
        "close_iteration": SKILLS_DIR / "close_iteration_replacement.md",
        "exploration": SKILLS_DIR / "exploration_replacement.md",
    },
    enable_explore_guard=True,                       # rule 3 = EE's must_explore
    loop_owns_parent=False,
    close_iteration_mode="register_branch",
    on_iteration_start=asa_on_iteration_start,
    on_seed_complete=asa_on_seed_complete,
    on_iteration_closed=asa_on_iteration_closed,
    run_eval_gate=asa_run_eval_gate,
    extra_tools=asa_extra_tools,
    commit_before_comparison=True,
    comparison_ack_note=ASA_COMPARISON_ACK_NOTE,
)


def run_asa_optimization(**kwargs) -> dict:
    """Entry point: `run_entry(run_asa_optimization, method="asa")`."""
    return run_optimization(strategy=ASA_STRATEGY, **kwargs)


__all__ = [
    "ASA_STRATEGY",
    "asa_extra_tools",
    "asa_on_iteration_closed",
    "asa_on_iteration_start",
    "asa_on_seed_complete",
    "asa_run_eval_gate",
    "branch_comparison_tool",
    "lineages_for",
    "run_asa_optimization",
    "seed_lineage_name",
]
