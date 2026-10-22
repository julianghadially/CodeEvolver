"""Query iteration history from MongoDB.

Usage:
    python scripts/query_iteration_history.py [job_id]

If no job_id is provided, shows the most recent job's iteration history.
Use --full to print the full subsample_analysis for each iteration.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.mongo import get_mongo_db
from src.config import settings


def truncate(text: str | None, max_len: int = 80) -> str:
    if not text:
        return ""
    text = text.replace("\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def print_iteration_table(history: dict, show_full: bool = False) -> None:
    """Print iteration history as a readable table."""
    # Sort by iteration number
    sorted_keys = sorted(history.keys(), key=lambda k: int(k) if k.isdigit() else -1)

    print(f"\n{'='*120}")
    print(f"{'Iter':>4}  {'Status':<10} {'Component':<18} {'Sub Score':>10} {'Parent':>8} {'Best?':>6} {'Val Score':>10}  {'Change Request'}")
    print(f"{'-'*4}  {'-'*10} {'-'*18} {'-'*10} {'-'*8} {'-'*6} {'-'*10}  {'-'*50}")

    for key in sorted_keys:
        rec = history[key]
        iteration = key
        candidate_idx = rec.get("candidate_idx")
        status = "accepted" if candidate_idx is not None else "rejected"
        component = rec.get("selected_component") or ""
        # Engine 3: the progress reading is the best train eval on the
        # baseline's subsample; no score is attributed to the closing commit.
        subsample_score = rec.get("best_subsample_score", rec.get("subsample_score"))
        parent_sub = rec.get("parent_subsample_score")
        score = rec.get("score")
        change_req = truncate(rec.get("change_request"), 60)

        sub_str = f"{subsample_score:.3f}" if subsample_score is not None else "-"
        parent_str = f"{parent_sub:.3f}" if parent_sub is not None else "-"
        score_str = f"{score:.3f}" if score is not None else "-"
        # selected_was_best: True=picked best, False=explored non-best, None=no real choice
        was_best = rec.get("selected_was_best")
        best_str = "yes" if was_best is True else ("no" if was_best is False else "-")

        print(f"{iteration:>4}  {status:<10} {component:<18} {sub_str:>10} {parent_str:>8} {best_str:>6} {score_str:>10}  {change_req}")

    print(f"{'='*120}")
    print_selection_summary(history)

    if show_full:
        print("\n--- Full Analysis Per Iteration ---")
        for key in sorted_keys:
            rec = history[key]
            analysis = rec.get("subsample_analysis")
            valset_analysis = rec.get("valset_analysis")
            arch_title = rec.get("architecture_title")
            git_branch = rec.get("git_branch")
            error = rec.get("subsample_error")

            print(f"\n## Iteration {key}")
            if arch_title:
                print(f"  Architecture: {arch_title}")
            if git_branch:
                print(f"  Branch: {git_branch}")
            if error:
                print(f"  ERROR: {error}")
            if analysis:
                print(f"  Subsample Analysis:\n    {analysis}")
            if valset_analysis:
                print(f"  Valset Analysis: {valset_analysis}")
            if not any([analysis, valset_analysis, error]):
                print("  (no analysis)")


def print_selection_summary(history: dict) -> None:
    """Summarize the proportion of selections that were NOT the best program.

    Only iterations with a genuine choice (pool >= 2 at selection time) are
    counted, encoded as `selected_was_best is not None`. The seed-only early
    iterations (None) are excluded from the denominator.
    """
    eligible = [
        rec for rec in history.values() if rec.get("selected_was_best") is not None
    ]
    if not eligible:
        print(
            "\nNon-best selections: n/a "
            "(no iterations with a genuine choice, or pre-dates selected_was_best tracking)"
        )
        return
    non_best = sum(1 for rec in eligible if rec.get("selected_was_best") is False)
    pct = 100.0 * non_best / len(eligible)
    print(
        f"\nNon-best selections: {non_best}/{len(eligible)} eligible (pool>=2) "
        f"= {pct:.1f}%  (explored a parent that was not the best program at selection time)"
    )


def print_fallback_history(history: dict) -> None:
    """Print ce_state_history (accepted candidates only, less detail)."""
    sorted_keys = sorted(history.keys(), key=lambda k: int(k) if k.isdigit() else -1)

    print(f"\n  NOTE: Full iteration_history not available. Showing ce_state_history (accepted candidates only).\n")
    print(f"{'='*120}")
    print(f"{'Idx':>4}  {'Type':<8} {'Score':>8} {'Branch':<45}  {'Change'}")
    print(f"{'-'*4}  {'-'*8} {'-'*8} {'-'*45}  {'-'*50}")

    for key in sorted_keys:
        rec = history[key]
        score = rec.get("score")
        change_type = rec.get("change_type", "")
        change_desc = truncate(rec.get("change_description"), 60)
        git_branch = rec.get("git_branch") or ""

        score_str = f"{score:.3f}" if score is not None else "-"
        print(f"{key:>4}  {change_type:<8} {score_str:>8} {git_branch:<45}  {change_desc}")

    print(f"{'='*120}")


def main():
    parser = argparse.ArgumentParser(description="Query iteration history from MongoDB.")
    parser.add_argument(
        "job_id",
        nargs="?",
        default=None,
        help="Job ID to query. If omitted, shows the most recent job.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Show full subsample_analysis and valset_analysis for each iteration.",
    )
    args = parser.parse_args()

    if not settings.mongodb_url:
        print("Error: MongoDB URL not configured. Set MONGO_HOST, MONGO_USER and MONGO_KEY (or LOCAL_MONGO_URI in local mode).")
        return

    db = get_mongo_db(db_name=settings.database_name, force_prod=False)

    # Find the job
    if args.job_id:
        job = db.jobs.find_one({"job_id": args.job_id})
        if not job:
            print(f"No job found with job_id: {args.job_id}")
            return
    else:
        job = db.jobs.find_one({}, sort=[("created_at", -1)])
        if not job:
            print("No jobs found in database.")
            return

    job_id = job.get("job_id")
    status = job.get("status", "unknown")
    created_at = job.get("created_at", "unknown")
    best_score = job.get("best_score")
    current_iter = job.get("current_iteration")
    num_candidates = job.get("num_candidates")

    print(f"Job: {job_id}")
    print(f"Status: {status}  |  Created: {created_at}")
    print(f"Iterations: {current_iter}  |  Candidates: {num_candidates}  |  Best Score: {best_score}")

    # Try full iteration_history first (inside ce_state, available on completion)
    ce_state = job.get("ce_state") or {}
    full_history = ce_state.get("iteration_history")

    if full_history and isinstance(full_history, dict) and len(full_history) > 0:
        print(f"\nSource: ce_state.iteration_history ({len(full_history)} iterations, includes rejected)")
        print_iteration_table(full_history, show_full=args.full)
    else:
        # Fallback to ce_state_history (accepted candidates only)
        fallback = job.get("ce_state_history") or {}
        if fallback and isinstance(fallback, dict) and len(fallback) > 0:
            print(f"\nSource: ce_state_history ({len(fallback)} accepted candidates)")
            print_fallback_history(fallback)
        else:
            print("\nNo iteration history found on this job.")
            print("Possible reasons:")
            print("  - Job hasn't completed yet (full history saved on completion)")
            print("  - Job ran without progress callbacks")
            print("  - Job failed before any iterations completed")


if __name__ == "__main__":
    main()
