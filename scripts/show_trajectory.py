#!/usr/bin/env python3
"""Show the optimization trajectory for a GEPA job.

Loads gepa_state_history from MongoDB, finds the best candidate,
walks the parent chain back to the seed, and prints a formatted table.

Usage:
    python scripts/show_trajectory.py --job-id job_abc123
    python scripts/show_trajectory.py --job-id job_abc123 --mongo-uri mongodb://...
"""

import argparse
import json
import os
import sys

from pymongo import MongoClient


CODE_COMPONENT_KEY = "_code"


def get_job(mongo_uri: str, job_id: str) -> dict | None:
    client = MongoClient(mongo_uri)
    db = client.get_default_database()
    return db.jobs.find_one({"job_id": job_id})


def _parse_code_field(candidate: dict) -> dict:
    """Parse the _code JSON field from a candidate. Returns empty dict on failure."""
    code_str = candidate.get(CODE_COMPONENT_KEY, "")
    if not code_str:
        return {}
    try:
        return json.loads(code_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def _detect_change(
    idx: int,
    candidate: dict,
    candidates: list[dict],
    parents: list,
) -> tuple[str, str, str | None]:
    """Detect what changed between a candidate and its parent.

    Returns (change_type, change_description, git_branch).
    """
    code_data = _parse_code_field(candidate)
    git_branch = code_data.get("git_branch")

    # Seed candidate
    parent_list = parents[idx] if idx < len(parents) else None
    if not parent_list or parent_list[0] is None or idx == 0:
        return "seed", "Initial seed candidate", git_branch

    parent_idx = parent_list[0]
    if parent_idx >= len(candidates):
        return "unknown", "Parent not found", git_branch

    parent = candidates[parent_idx]
    parent_code_data = _parse_code_field(parent)

    # Code change: git_branch differs from parent's
    parent_branch = parent_code_data.get("git_branch")
    if git_branch and parent_branch and git_branch != parent_branch:
        change_request = code_data.get("change_request", "Code change")
        return "code", change_request or "Code change", git_branch

    # Prompt change: find which prompt components differ
    changed_components = []
    for key in candidate:
        if key == CODE_COMPONENT_KEY:
            continue
        if candidate[key] != parent.get(key, ""):
            changed_components.append(key)

    if changed_components:
        desc = f"Prompt change at {', '.join(changed_components)}"
        return "prompt", desc, git_branch

    return "unknown", "No detectable change", git_branch


def build_history_from_state(gepa_state: dict) -> dict[str, dict]:
    """Fallback: build history from flat gepa_state when gepa_state_history is missing."""
    candidates = gepa_state.get("program_candidates", [])
    scores = gepa_state.get("candidate_scores", [])
    parents = gepa_state.get("parent_programs", [])
    history = {}
    for idx, candidate in enumerate(candidates):
        change_type, change_description, git_branch = _detect_change(
            idx, candidate, candidates, parents
        )
        history[str(idx)] = {
            "candidate": candidate,
            "score": scores[idx] if idx < len(scores) else None,
            "parent_candidates": parents[idx] if idx < len(parents) else None,
            "change_type": change_type,
            "change_description": change_description,
            "git_branch": git_branch,
        }
    return history


def walk_parent_chain(history: dict[str, dict], start_idx: int) -> list[int]:
    """Walk from start_idx back through parents to the seed. Returns list of indices."""
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


def truncate(text: str | None, max_len: int = 60) -> str:
    if not text:
        return "-"
    text = text.replace("\n", " ").strip()
    if len(text) > max_len:
        return text[:max_len - 3] + "..."
    return text


def print_trajectory(history: dict[str, dict], chain: list[int]) -> None:
    header = f"{'Idx':>4}  {'Type':<6}  {'Branch':<40}  {'Change':<60}  {'Parent':>7}  {'Score':>8}  {'Delta':>8}"
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)

    prev_score = None
    for idx in chain:
        entry = history.get(str(idx), {})
        score = entry.get("score")
        change_type = entry.get("change_type", "?")
        branch = truncate(entry.get("git_branch"), 38)
        change = truncate(entry.get("change_description"), 58)
        parents = entry.get("parent_candidates")
        parent_str = str(parents[0]) if parents and parents[0] is not None else "seed"

        score_str = f"{score:.4f}" if score is not None else "-"
        if prev_score is not None and score is not None:
            delta = score - prev_score
            delta_str = f"{delta:+.4f}"
        else:
            delta_str = "-"

        print(f"{idx:>4}  {change_type:<6}  {branch:<40}  {change:<60}  {parent_str:>7}  {score_str:>8}  {delta_str:>8}")
        if score is not None:
            prev_score = score

    print(sep)


def main():
    parser = argparse.ArgumentParser(description="Show GEPA optimization trajectory")
    parser.add_argument("--job-id", required=True, help="Job ID to analyze")
    parser.add_argument(
        "--mongo-uri",
        default=os.getenv("MONGODB_URI", "mongodb://localhost:27017/codeevolver"),
        help="MongoDB connection URI (default: MONGODB_URI env or localhost)",
    )
    parser.add_argument("--all", action="store_true", help="Show all candidates, not just best chain")
    args = parser.parse_args()

    job = get_job(args.mongo_uri, args.job_id)
    if not job:
        print(f"Job not found: {args.job_id}", file=sys.stderr)
        sys.exit(1)

    print(f"Job: {job['job_id']}  Status: {job.get('status')}  Best score: {job.get('best_score')}")

    # Load history: prefer gepa_state_history, fall back to gepa_state
    history = job.get("gepa_state_history")
    if not history:
        gepa_state = job.get("gepa_state")
        if not gepa_state:
            print("No GEPA state or history found for this job.", file=sys.stderr)
            sys.exit(1)
        print("(Reconstructed from gepa_state — no per-iteration history available)")
        history = build_history_from_state(gepa_state)

    if not history:
        print("Empty history.", file=sys.stderr)
        sys.exit(1)

    if args.all:
        # Show all candidates sorted by index
        all_indices = sorted(int(k) for k in history.keys())
        print(f"\nAll {len(all_indices)} candidates:")
        print_trajectory(history, all_indices)
    else:
        # Find best candidate and walk parent chain
        best_idx = None
        best_score = -float("inf")
        for idx_str, entry in history.items():
            s = entry.get("score")
            if s is not None and s > best_score:
                best_score = s
                best_idx = int(idx_str)

        if best_idx is None:
            print("No scored candidates found.", file=sys.stderr)
            sys.exit(1)

        chain = walk_parent_chain(history, best_idx)
        print(f"\nBest candidate: index {best_idx} (score {best_score:.4f})")
        print(f"Trajectory length: {len(chain)} steps (seed -> best)\n")
        print_trajectory(history, chain)


if __name__ == "__main__":
    main()
