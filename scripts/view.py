"""View job IDs from MongoDB database."""

import argparse
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.mongo import get_mongo_db
from src.config import settings


def main():
    """Connect to MongoDB and print a job record (by ID or most recent)."""
    parser = argparse.ArgumentParser(description="View job record(s) from MongoDB.")
    parser.add_argument(
        "job",
        nargs="?",
        default=None,
        metavar="JOB_ID",
        help="Job ID to look up (e.g. job_ce82f384e502). If omitted, show the most recent job.",
    )
    args = parser.parse_args()

    # Check if MongoDB URL is configured
    if not settings.mongodb_url:
        print("Error: MongoDB URL not configured.")
        print("Please set MONGO_KEY environment variable.")
        return

    # Get MongoDB database connection
    # Use the same database name as the FastAPI app
    db = get_mongo_db(db_name=settings.database_name, force_prod=False)

    # Projection: include GEPA progress and candidate history (stored on the job)
    job_projection = {
        "job_id": 1,
        "client_id": 1,
        "status": 1,
        "created_at": 1,
        "best_candidate": 1,
        "best_score": 1,
        "current_iteration": 1,
        "num_candidates": 1,
        "total_metric_calls": 1,
        "gepa_state_history": 1,
        "program_candidates": 1,
        "candidate_scores": 1,
    }

    # Query jobs collection: by job_id if given, else most recent
    if args.job:
        job = db.jobs.find_one({"job_id": args.job}, job_projection)
        if not job:
            print(f"No job found with job_id: {args.job}")
            return
    else:
        job = db.jobs.find_one(
            {},
            job_projection,
            sort=[("created_at", -1)],  # Most recent first
        )

    if job:
        job_id = job.get("job_id")
        client_id = job.get("client_id")
        status = job.get("status", "unknown")
        created_at = job.get("created_at", "unknown")
        print("=== Job record ===")
        print(f"Job ID: {job_id}")
        print(f"Client ID: {client_id}")
        print(f"Status: {status}")
        print(f"Created at: {created_at}")
        print(f"Best candidate: {job.get('best_candidate') or 'None (no progress callbacks received yet)'}")
        print(f"Best score: {job.get('best_score')}")
        print(f"Current iteration: {job.get('current_iteration')}")
        print(f"Num candidates: {job.get('num_candidates')}")
        print(f"Total metric calls: {job.get('total_metric_calls')}")

        # GEPA candidates: stored on the job (gepa_state_history from progress callbacks)
        history = job.get("gepa_state_history") or {}
        if isinstance(history, dict) and history:
            # Keys may be "0", "1", ... (string for MongoDB)
            indices = sorted(history.keys(), key=lambda k: int(k) if str(k).isdigit() else -1)
            print(f"\n=== GEPA candidates (from job, {len(indices)} found) ===")
            for idx_str in indices:
                rec = history.get(idx_str) or {}
                score = rec.get("score")
                git_branch = rec.get("git_branch", "")
                change_type = rec.get("change_type", "")
                change_desc = (rec.get("change_description") or "")[:60]
                print(f"  [{idx_str}] score={score} branch={git_branch!r} type={change_type} | {change_desc}...")
        elif job.get("program_candidates"):
            # Completed job may have program_candidates at top level
            n = len(job["program_candidates"])
            scores = job.get("candidate_scores") or []
            print(f"\n=== GEPA candidates (completed job, {n} candidates) ===")
            for i in range(n):
                cand = job["program_candidates"][i]
                branch = ""
                if isinstance(cand, dict):
                    code_str = cand.get("_code", "") or "{}"
                    try:
                        code = json.loads(code_str)
                        branch = code.get("git_branch", "")
                    except Exception:
                        pass
                sc = scores[i] if i < len(scores) else None
                print(f"  [{i}] score={sc} branch={branch!r}")
        else:
            print("\n=== GEPA candidates ===")
            print("No candidate history on this job. Progress is saved when the optimizer sends progress callbacks; if the job ran without callbacks (e.g. local run) or callbacks failed, no candidates will be stored here.")

        # Program records: separate collection (may be empty; GEPA stores candidates on the job above)
        if client_id:
            programs = list(
                db.programs.find({"client_id": client_id}).limit(10)
            )
            if programs:
                print(f"\n=== Program record(s) for client_id={client_id} ({len(programs)} found) ===")
                for i, prog in enumerate(programs):
                    print(f"  [{i+1}] program_id={prog.get('program_id')} "
                          f"parent_program_id={prog.get('parent_program_id')} "
                          f"branch_name={prog.get('branch_name')} "
                          f"status={prog.get('status')} "
                          f"created_at={prog.get('created_at')}")
                    pj = prog.get("program_json") or {}
                    print(f"      program_json keys: {list(pj.keys()) if isinstance(pj, dict) else type(pj)}")
            else:
                print("\n=== Program record(s) (programs collection) ===")
                print("No program records for this client_id. In-progress GEPA candidates are stored on the job (see GEPA candidates above), not in the programs collection.")
                try:
                    any_count = db.programs.count_documents({})
                    print(f"(programs collection has {any_count} total document(s).)")
                except Exception as e:
                    print(f"(programs collection: {e})")
        else:
            print("\n=== Program record(s) ===")
            print("Job has no client_id; cannot look up programs collection.")
    else:
        print("No jobs found in database.")
        all_jobs = list(db.jobs.find({}, {"job_id": 1}).limit(10))
        if all_jobs:
            print(f"\nFound {len(all_jobs)} jobs:")
            for j in all_jobs:
                print(f"  - {j.get('job_id')}")
        else:
            print("Database collection 'jobs' is empty.")


if __name__ == "__main__":
    # Temporarily: print last 5 jobs from the database
    if settings.mongodb_url:
        db = get_mongo_db(db_name=settings.database_name, force_prod=False)
        last_five = list(
            db.jobs.find(
                {},
                {"job_id": 1, "status": 1, "created_at": 1},
                sort=[("created_at", -1)],
            ).limit(5)
        )
        print("=== Last 5 jobs (temporary) ===")
        for j in last_five:
            print(f"  {j.get('job_id')}  status={j.get('status')}  created_at={j.get('created_at')}")
        print()

    main()
