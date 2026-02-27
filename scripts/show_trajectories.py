#!/usr/bin/env python3
"""Show the top GEPA optimization trajectories for a job.

Loads gepa_state_history from MongoDB, finds the top N trajectories
(default 3), and prints them. Each trajectory walks from a top-scoring
candidate back through its parent chain to the seed.

Later trajectories exclude candidates on branches already shown,
so each trajectory represents a distinct evolutionary path.

Usage:
    python scripts/show_trajectories.py
    python scripts/show_trajectories.py job_abc123
    python scripts/show_trajectories.py job_abc123 --json
    python scripts/show_trajectories.py -n 5
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.mongo import get_mongo_db
from src.config import settings

CODE_COMPONENT_KEY = "_code"
NUM_TRAJECTORIES = 3


def parse_code_field(candidate: dict) -> dict:
    """Parse the _code JSON field from a candidate."""
    code_str = candidate.get(CODE_COMPONENT_KEY, "")
    if not code_str:
        return {}
    try:
        obj = json.loads(code_str)
        cr = obj.get("change_request")
        if isinstance(cr, str):
            try:
                obj["change_request"] = json.loads(cr)
            except (json.JSONDecodeError, TypeError):
                pass
        return obj
    except (json.JSONDecodeError, TypeError):
        return {}


def get_change_text(entry: dict) -> str:
    """Extract the full change request/description from a history entry.

    For code mutations: returns the full change_request from _code.
    For prompt mutations: returns the change_description.
    """
    change_type = entry.get("change_type", "unknown")

    if change_type == "code":
        candidate = entry.get("candidate", {})
        code_data = parse_code_field(candidate)
        cr = code_data.get("change_request", "")
        if cr:
            if isinstance(cr, dict):
                return json.dumps(cr, indent=2, ensure_ascii=False)
            return str(cr)

    return entry.get("change_description", "")


def walk_parent_chain(history: dict[str, dict], start_idx: int) -> list[int]:
    """Walk from start_idx back through parents to the seed. Returns list root-first."""
    chain = [start_idx]
    visited = {start_idx}
    current = start_idx
    while True:
        entry = history.get(str(current))
        if not entry:
            break
        parents = entry.get("parent_candidates")
        if not parents or parents[0] is None:
            break
        parent = parents[0]
        if parent in visited:
            break
        visited.add(parent)
        chain.append(parent)
        current = parent
    chain.reverse()
    return chain


def build_trajectory_json(history: dict[str, dict], chain: list[int]) -> list[dict]:
    """Build a JSON-serializable list of steps for a trajectory."""
    steps = []
    for idx in chain:
        entry = history.get(str(idx), {})
        parents = entry.get("parent_candidates")
        step = {
            "candidate_index": idx,
            "score": entry.get("score"),
            "branch": entry.get("git_branch", ""),
            "type": entry.get("change_type", "unknown"),
            "change": get_change_text(entry),
            "parent": parents[0] if parents and parents[0] is not None else None,
        }
        steps.append(step)
    return steps


def _branch_excluded(branch: str, exclude_set: set[str]) -> bool:
    """Check if a branch matches any entry in the exclude set (substring match)."""
    for pattern in exclude_set:
        if pattern in branch:
            return True
    return False


def find_top_trajectories(
    history: dict[str, dict],
    n: int = NUM_TRAJECTORIES,
    exclude_branches: set[str] | None = None,
) -> list[dict]:
    """Find the top n trajectories, each starting from a distinct branch lineage."""
    # Sort all candidates by score descending
    scored = []
    for idx_str, entry in history.items():
        score = entry.get("score")
        if score is not None:
            scored.append((score, int(idx_str)))
    scored.sort(reverse=True)

    trajectories = []
    seen_branches = set(exclude_branches or ())

    for score, idx in scored:
        if len(trajectories) >= n:
            break

        # Skip candidates on branches already shown or excluded
        entry = history.get(str(idx), {})
        branch = entry.get("git_branch", "")
        if branch and _branch_excluded(branch, seen_branches):
            continue

        # Walk the full parent chain back to seed
        chain = walk_parent_chain(history, idx)

        # Skip entire trajectory if any step is on an excluded branch
        chain_branches = []
        skip = False
        for step_idx in chain:
            step_entry = history.get(str(step_idx), {})
            b = step_entry.get("git_branch", "")
            if b:
                chain_branches.append(b)
                if _branch_excluded(b, exclude_branches or set()):
                    skip = True
                    break
        if skip:
            continue

        steps = build_trajectory_json(history, chain)

        # Collect all branches from this trajectory so later trajectories skip them
        for b in chain_branches:
            seen_branches.add(b)

        trajectory = {
            "rank": len(trajectories) + 1,
            "best_score": score,
            "best_candidate_index": idx,
            "num_steps": len(steps),
            "steps": steps,
        }
        trajectories.append(trajectory)

    return trajectories


def print_trajectories(trajectories: list[dict]) -> None:
    """Pretty-print trajectories to stdout."""
    for traj in trajectories:
        rank = traj["rank"]
        score = traj["best_score"]
        score_str = f"{score:.4f}" if score is not None else "n/a"

        print()
        print("=" * 80)
        print(
            f"  TRAJECTORY #{rank}  |  Best Score: {score_str}"
            f"  |  Candidate: [{traj['best_candidate_index']}]"
            f"  |  Steps: {traj['num_steps']}"
        )
        print("=" * 80)

        for i, step in enumerate(traj["steps"]):
            step_score = step["score"]
            step_score_str = f"{step_score:.4f}" if step_score is not None else "n/a"
            type_str = step["type"].upper()
            branch = step["branch"] or "n/a"
            change = step["change"] or "-"

            if i > 0:
                print(f"       |")
                print(f"       v")

            print(f"  [{step['candidate_index']:>3}]  Score: {step_score_str}  |  Type: {type_str}")
            print(f"         Branch: {branch}")
            if change and change != "-":
                # Print the full change, indented for readability
                lines = change.split("\n")
                print(f"         Change: {lines[0]}")
                for line in lines[1:]:
                    print(f"                 {line}")

        print()


def main():
    parser = argparse.ArgumentParser(description="Show top GEPA optimization trajectories")
    parser.add_argument(
        "job",
        nargs="?",
        default=None,
        metavar="JOB_ID",
        help="Job ID to analyze. If omitted, uses the most recent job.",
    )
    parser.add_argument(
        "-n", "--num-trajectories",
        type=int,
        default=NUM_TRAJECTORIES,
        help=f"Number of trajectories to show (default: {NUM_TRAJECTORIES})",
    )
    parser.add_argument(
        "--exclude-branch",
        action="append",
        default=[],
        metavar="BRANCH",
        help="Exclude trajectories that include this branch (substring match). Can be repeated.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of pretty-printed format",
    )
    args = parser.parse_args()

    if not settings.mongodb_url:
        print("Error: MongoDB URL not configured. Set MONGO_KEY environment variable.", file=sys.stderr)
        sys.exit(1)

    db = get_mongo_db(db_name=settings.database_name, force_prod=False)

    projection = {
        "job_id": 1,
        "status": 1,
        "best_score": 1,
        "current_iteration": 1,
        "gepa_state_history": 1,
    }

    if args.job:
        job = db.jobs.find_one({"job_id": args.job}, projection)
        if not job:
            print(f"No job found: {args.job}", file=sys.stderr)
            sys.exit(1)
    else:
        job = db.jobs.find_one({}, projection, sort=[("created_at", -1)])
        if not job:
            print("No jobs found in database.", file=sys.stderr)
            sys.exit(1)

    job_id = job.get("job_id")
    best = job.get("best_score")
    best_str = f"{best:.4f}" if best is not None else "n/a"
    print(
        f"Job: {job_id}  |  Status: {job.get('status')}"
        f"  |  Best Score: {best_str}"
        f"  |  Iteration: {job.get('current_iteration')}"
    )

    history = job.get("gepa_state_history")
    if not history or not isinstance(history, dict):
        print("No gepa_state_history found for this job.", file=sys.stderr)
        sys.exit(1)

    print(f"Total candidates in history: {len(history)}")

    exclude = set(args.exclude_branch) if args.exclude_branch else None
    trajectories = find_top_trajectories(history, n=args.num_trajectories, exclude_branches=exclude)

    if not trajectories:
        print("No scored candidates found.", file=sys.stderr)
        sys.exit(1)

    if args.json:
        print(json.dumps(trajectories, indent=2, ensure_ascii=False))
    else:
        print_trajectories(trajectories)


if __name__ == "__main__":
    main()
