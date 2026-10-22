"""Copyright © 2026 440 Labs LLC

Pareto-Evolution optimizer — the fixed-evolution research arm (Treatment B
in `specs/research_plan.md`), engine 3.

Runs on the SAME `optimizer_loop.run_optimization` as EE and freeform.
The arm's agent-facing surface is FREEFORM's (Baseline 1):
the freeform system prompt (plus `## Git Branch Background` and the
exploit-phase merge note), only the `close_iteration` skill, no explore
guard, a one-field `close_iteration(memory_updated)` tool and the macro-free
memory template. That makes freeform -> pareto_evolution a one-variable
comparison: fixed evolutionary control. The things this arm takes away from
the agent are:

- **Candidate selection** (`evo_select_parent`): a per-iteration draw picks
  either the best candidate (exploit-and-merge phase, tanh curve over the
  eval budget) or a Pareto-proportional sample over the comparison-set
  instance frontier; the loop forks `codeevolver-{ts}-iter{N}` from the
  candidate's recorded close sha before the architect session starts.
- **Fitness measurement timing**: the AGENT decides whether to run the
  comparison set at close (the run is the close step). A GEPA-inspired
  fallback stays: when no comparison ran and the train subsample STRICTLY
  progressed (`best > baseline`), `evo_on_iteration_closed` runs one
  (`loop_eval.run_loop_comparison_eval`, counted, overdraws credits).
- **Retention / promotion**: a closed, landed
  iteration is admitted iff its closing content HAS a comparison-set eval
  (agent-run, reverted-to, or the loop fallback). A train regression is not
  a barrier — a strict-train gate would reject comparison-set improvements
  and let the agent's memory "current best" diverge from the loop primary.
  The primary is always the population's best comparison-set score. The agent
  never sets primary and never registers a branch
  (`close_iteration_mode="loop_branch"`).
- **Population log**: loop-internal `population.json` (never agent-facing).

Everything else — prompt text, skills, memory template, watchdogs,
constraint gate, analysis agent, budget gate, finalization — is shared with
freeform unchanged. The agent is never told to explore or exploit; the
`exploit_phase` draw and `selection_policy` record field are loop-internal
names for the parent-selection policy, not agent-facing guidance.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..close_effects import revert_to_comparison_point
from ..iteration_context import IterationContext
from ..loop_eval import run_loop_comparison_eval
from ..optimizer_loop import run_optimization
from ..program_paths import diff_is_non_program_only, program_diff_paths
from ..prompts import build_evo_architect_system_prompt
from ..strategy import LoopServices, OptimizerStrategy
from .population import Candidate, Population
from .select_candidate import evo_select_parent

if TYPE_CHECKING:
    from ..ce_state import CEState
    from src.engine.schemas.iteration_history import IterationHistory
    from ..records import CloseRecord

logger = logging.getLogger(__name__)

SKILLS_DIR = Path(__file__).resolve().parent / "skills"

# Acknowledgement note before this arm's comparison run
# (specs/archive-public/comparison_credits.md §Evo). The comparison IS the close step here:
# once it runs, run_eval refuses further calls and any program changes made
# after it are reverted to the auto-committed comparison-run point.
EVO_COMPARISON_ACK_NOTE = (
    "Once you run the comparison set, you will no longer be able to run "
    "further evals or land program changes this iteration.\n\n"
    "The comparison set is part of the close iteration process — please see "
    "the close_iteration skill. It is used to select the branch the next "
    "agent will work on. Only one run is permitted per iteration, and it "
    "must be run at the end, after your work is committed. Run it when you "
    "want this branch in the candidate population: the comparison-set score "
    "is what admits a branch, even if your training eval did not improve "
    "within noise. If you skip it, the branch is not a candidate unless your "
    "training eval strictly beat the baseline, in which case the optimizer "
    "runs the comparison for you at close. After the run, close the "
    "iteration immediately: any program changes made after the comparison "
    "run will be reverted back to the auto-committed comparison-run point."
    "\n\n"
    "Please acknowledge that you would like to continue running the "
    "comparison set."
)


# ----------------------------------------------------------------------
# Seed → candidate 0
# ----------------------------------------------------------------------
def evo_on_seed_complete(
    state: "CEState",
    history: "IterationHistory",
    services: LoopServices,
) -> None:
    """Measure the seed on the comparison set and seed the population.

    Counted against the eval budget: this is the fixed-fitness maintenance
    cost the research plan assigns to the evolutionary arm. EE does not pay
    it, and that difference is the variable under test.
    """
    if not services.comparison_ids:
        raise RuntimeError(
            "pareto_evolution requires a comparison set (comparison_set_size > 0)"
        )
    if (
        services.max_eval_rows
        and len(services.comparison_ids) >= services.max_eval_rows
    ):
        logger.warning(
            "pareto_evolution: the seed comparison eval (%d rows) alone meets "
            "max_metric_calls=%d — the loop will stop before iteration 1",
            len(services.comparison_ids), services.max_eval_rows,
        )
    ctx = IterationContext(
        iteration_idx=0,
        comparison_set_size=len(services.comparison_ids),
        max_eval_rows=services.max_eval_rows,
        evals_used=int(state.total_num_evals or 0),
        primary_branch_at_start=state.primary_branch,
        parent_locked=True,
    )
    result = run_loop_comparison_eval(
        ctx=ctx, state=state, services=services, ref=state.seed_branch,
    )
    rec = ctx.eval_records[-1]
    pop = Population()
    pop.add(
        Candidate(
            iteration=0,
            branch=state.seed_branch,
            sha=state.seed_sha,
            tree=rec.get("tree"),
            parent_iteration=None,
            row_scores=dict(rec.get("row_scores") or {}),
            comparison_set_score=float(result.primary_score),
            train_baseline=None,
            train_best=None,
        )
    )
    pop.save()
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
        "pareto_evolution: seeded population with candidate 0 (%s @ %s, "
        "comparison_set_score=%.4f, %d rows counted)",
        state.seed_branch, state.seed_sha[:8], result.primary_score,
        len(services.comparison_ids),
    )


# ----------------------------------------------------------------------
# Admission
# ----------------------------------------------------------------------
# Admitted reasons, keyed by how the closing content's comparison eval came
# to exist. Not-admitted reasons: "skipped" | "constraint_violation" |
# "no_change" (closing content == parent content, already represented) |
# "no_comparison" (the agent ran none and train did not strictly progress).
ADMISSION_REASON_BY_STATE = {
    "EXACT": "agent_comparison",
    "PROGRAM_EQUIVALENT": "agent_comparison",
    "REVERTED_TO_COMPARISON_POINT": "reverted_to_comparison_point",
    "RERUN_BY_LOOP": "loop_comparison",
}


def admission_exclusion(ctx: IterationContext, rec: "CloseRecord") -> str | None:
    """Structural reasons the closing content can never be a candidate.

    Cheap and first: a skipped iteration produced nothing; a constraint
    violation was force-reverted to the parent; `no_change` means the closing
    tree is program-equivalent to the parent's, which the population already
    holds. Everything else is admissible — admission itself is decided by
    whether a comparison eval of the closing content exists
    (`evo_on_iteration_closed`), not by the train subsample.
    """
    if rec.get("skipped") or ctx.skipped:
        return "skipped"
    if rec.get("constraint_violation"):
        return "constraint_violation"
    if not ctx.changes_landed:
        return "no_change"
    return None


def train_progressed(ctx: IterationContext) -> bool:
    """The GEPA-inspired fallback predicate: best train eval on the baseline's
    locked subsample STRICTLY beats the baseline. A tie is not progress. Only
    decides whether the LOOP pays for a comparison eval when the agent ran
    none — never whether a measured candidate is admitted."""
    best = ctx.best_subsample_score
    baseline = ctx.initial_subsample_score
    return best is not None and baseline is not None and best > baseline


def evo_on_iteration_closed(
    ctx: IterationContext,
    state: "CEState",
    rec: "CloseRecord",
    services: LoopServices,
) -> None:
    """Apply the fixed admission / fitness / promotion rules to the close
    record (mutated in place) before `CEState.record_iteration`."""
    pop = Population.load_or_none()
    if pop is None:
        raise RuntimeError("pareto_evolution: population.json missing at close")

    exclusion = admission_exclusion(ctx, rec)
    rec["candidate_admitted"] = False
    rec["admission_reason"] = exclusion or "no_comparison"
    rec["selected_parent_iteration"] = ctx.parent_iteration
    rec["selection_policy"] = "best" if ctx.exploit_phase else "pareto_sample"
    rec["comparison_set_score"] = None
    rec["comparison_state"] = None
    rec["comparison_eval_id_at_close"] = None
    rec["comparison_score_at_close"] = None

    cmp = None
    comparison_state: str | None = None
    if exclusion is None:
        workspace = services.workspace

        def equivalent(a: str | None, b: str | None) -> bool:
            return diff_is_non_program_only(workspace, a, b)

        # Map the CLOSING content to a comparison eval by tree. EXACT: the
        # eval ran on this very tree. PROGRAM_EQUIVALENT: it ran on a tree
        # differing only in non-program files (codeevolver.md, README). Both
        # are the same program, so the fitness vector is the closing
        # content's. Otherwise (specs/archive-public/comparison_credits.md): the comparison
        # run IS the close step on this arm — when the architect DID run one
        # and then changed the program, the loop reverts the branch to the
        # auto-committed comparison-run point rather than paying for a rerun.
        # Only when no comparison ran at all does the fixed rule pay for one
        # (RERUN_BY_LOOP) — and only when the train subsample strictly
        # progressed: the agent is the decision maker, the loop's rerun is the
        # GEPA-inspired backstop for an agent that improved and forgot. An
        # iteration with neither is simply not a candidate (`no_comparison`).
        rerun = False
        cmp = ctx.comparison_record_for_tree(equivalent)
        if cmp is not None and cmp.get("row_scores"):
            comparison_state = (
                "EXACT" if cmp.get("tree") == ctx.close_tree else "PROGRAM_EQUIVALENT"
            )
        else:
            newest_cmp = next(
                (
                    r for r in reversed(ctx.eval_records)
                    if r.get("eval_sample") == "comparison" and r.get("tree")
                ),
                None,
            )
            comparison_state = None
            if newest_cmp is not None:
                differing = program_diff_paths(
                    workspace, newest_cmp["tree"], ctx.close_tree
                )
                logger.warning(
                    "iteration %d: pareto_evolution closing tree %s differs "
                    "from the comparison eval's tree %s in program files "
                    "(%s) — reverting to the comparison-run point %s",
                    ctx.iteration_idx, (ctx.close_tree or "?")[:8],
                    newest_cmp["tree"][:8], ", ".join(differing[:8]) or "(unknown)",
                    (newest_cmp.get("sha") or "?")[:8],
                )
                target_sha = newest_cmp.get("sha")
                if target_sha and revert_to_comparison_point(
                    ctx, workspace=workspace, target_sha=target_sha
                ):
                    # finalize() re-ran inside the revert; keep the record
                    # consistent with the tree now on disk.
                    rec["close_sha"] = ctx.close_sha
                    rec["close_tree"] = ctx.close_tree
                    rec["changes_landed"] = ctx.changes_landed
                    if not ctx.changes_landed:
                        # The comparison point IS the parent content — the
                        # agent measured the unchanged parent and then edited.
                        # Already represented; nothing to admit.
                        exclusion = "no_change"
                        cmp = None
                    else:
                        cmp = newest_cmp
                        comparison_state = "REVERTED_TO_COMPARISON_POINT"
                else:
                    # The agent explicitly asked for a measurement; honour it.
                    logger.warning(
                        "iteration %d: revert to comparison point failed — "
                        "falling back to a loop comparison rerun",
                        ctx.iteration_idx,
                    )
                    rerun = True
            elif train_progressed(ctx):
                logger.warning(
                    "iteration %d: pareto_evolution re-running the comparison "
                    "set — the architect never ran one this iteration and the "
                    "train subsample progressed (%.4f -> %.4f)",
                    ctx.iteration_idx, ctx.initial_subsample_score,
                    ctx.best_subsample_score,
                )
                rerun = True
            else:
                logger.info(
                    "iteration %d: pareto_evolution — no comparison eval and no "
                    "strict train progression; the closing content is not a "
                    "candidate (no_comparison)",
                    ctx.iteration_idx,
                )
                cmp = None
            if rerun:
                run_loop_comparison_eval(
                    ctx=ctx, state=state, services=services,
                    ref=ctx.child_branch or ctx.close_sha or "HEAD",
                )
                cmp = ctx.eval_records[-1]
                comparison_state = "RERUN_BY_LOOP"
                from ..closing import comparison_entries

                rec["comparison_scores"] = comparison_entries(ctx)
                state.note_loop_comparison_rerun(int(cmp.get("n_rows") or 0))
    if exclusion is not None:
        rec["admission_reason"] = exclusion
    if cmp is not None and comparison_state is not None:
        ctx.comparison_state = comparison_state
        ctx.comparison_eval_id_at_close = cmp.get("eval_id")
        ctx.comparison_score_at_close = float(cmp["score"])
        rec["comparison_state"] = comparison_state
        rec["comparison_eval_id_at_close"] = cmp.get("eval_id")
        rec["comparison_score_at_close"] = float(cmp["score"])
        pop.add(
            Candidate(
                iteration=ctx.iteration_idx,
                branch=ctx.child_branch or "",
                sha=ctx.close_sha or "",
                tree=ctx.close_tree,
                parent_iteration=ctx.parent_iteration,
                row_scores=dict(cmp.get("row_scores") or {}),
                comparison_set_score=float(cmp["score"]),
                train_baseline=ctx.initial_subsample_score,
                train_best=ctx.best_subsample_score,
            )
        )
        pop.save()
        rec["candidate_admitted"] = True
        rec["admission_reason"] = ADMISSION_REASON_BY_STATE[comparison_state]
        rec["comparison_set_score"] = float(cmp["score"])

    best = pop.best()
    rec["primary_branch_after"] = best["branch"]
    rec["primary_fallback"] = False
    rec["was_primary"] = ctx.parent_branch == ctx.primary_branch_at_start
    rec["is_primary"] = ctx.child_branch == best["branch"]

    logger.info(
        "iteration %d: pareto_evolution close — admitted=%s (%s), "
        "comparison_state=%s, comparison_set_score=%s, population=%d, "
        "best=candidate %d (%s), frontier=%s",
        ctx.iteration_idx, rec["candidate_admitted"], rec["admission_reason"],
        rec["comparison_state"], rec["comparison_set_score"], len(pop),
        best["iteration"], best["branch"], sorted(pop.get_frontier_members()),
    )


PARETO_STRATEGY = OptimizerStrategy(
    method="pareto_evolution",
    # Freeform-based (specs/v3_freeform.md §Prompt applies): no explore /
    # exploit guidance anywhere — prompt, skills, tool schema, memory template.
    prompt_builder=build_evo_architect_system_prompt,
    skills_to_stage=("close_iteration",),
    skill_overrides={
        "close_iteration": SKILLS_DIR / "close_iteration_replacement.md",
    },
    enable_explore_guard=False,
    loop_owns_parent=True,
    close_iteration_mode="loop_branch",
    on_iteration_start=evo_select_parent,
    on_seed_complete=evo_on_seed_complete,
    on_iteration_closed=evo_on_iteration_closed,
    # The comparison run is this arm's close step (specs/archive-public/comparison_credits.md):
    # commit first so the run point is a revert target, refuse evals after it.
    commit_before_comparison=True,
    comparison_ack_note=EVO_COMPARISON_ACK_NOTE,
    comparison_closes_iteration=True,
)


def run_pareto_optimization(**kwargs) -> dict:
    """Entry point: `run_entry(run_pareto_optimization, method="pareto_evolution")`."""
    return run_optimization(strategy=PARETO_STRATEGY, **kwargs)


__all__ = [
    "ADMISSION_REASON_BY_STATE",
    "PARETO_STRATEGY",
    "admission_exclusion",
    "evo_on_iteration_closed",
    "evo_on_seed_complete",
    "run_pareto_optimization",
    "train_progressed",
]
