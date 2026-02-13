"""View job IDs from MongoDB database."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.db.mongo import get_mongo_db
from src.config import settings


def main():
    """Connect to MongoDB and print a job ID."""
    # Check if MongoDB URL is configured
    if not settings.mongodb_url:
        print("Error: MongoDB URL not configured.")
        print("Please set MONGO_KEY environment variable.")
        return
    
    # Get MongoDB database connection
    # Use the same database name as the FastAPI app
    db = get_mongo_db(db_name=settings.database_name, force_prod=False)
    
    # Query jobs collection for a job (include client_id to link to programs)
    job = db.jobs.find_one(
        {},
        {"job_id": 1, "client_id": 1, "status": 1, "created_at": 1},
        sort=[("created_at", -1)]  # Most recent first
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
        print(f"Best candidate: {job.get('best_candidate', 'unknown')}")

        # Try to fetch program record(s) for this job's client
        if client_id:
            programs = list(
                db.programs.find({"client_id": client_id}).limit(10)
            )
            if programs:
                print(f"=== Program record(s) for client_id={client_id} ({len(programs)} found) ===")
                for i, prog in enumerate(programs):
                    print(f"  [{i+1}] program_id={prog.get('program_id')} "
                          f"parent_program_id={prog.get('parent_program_id')} "
                          f"branch_name={prog.get('branch_name')} "
                          f"status={prog.get('status')} "
                          f"created_at={prog.get('created_at')}")
                    # Show keys only for program_json to avoid huge output
                    pj = prog.get("program_json") or {}
                    print(f"      program_json keys: {list(pj.keys()) if isinstance(pj, dict) else type(pj)}")
            else:
                print("=== Program record(s) ===")
                print("No program records found for this client_id.")
                # Check if programs collection exists and has any docs
                try:
                    any_count = db.programs.count_documents({})
                    print(f"(programs collection exists with {any_count} total document(s).)")
                except Exception as e:
                    print(f"(programs collection: {e})")
        else:
            print("=== Program record(s) ===")
            print("Job has no client_id; cannot look up programs.")
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
    main()
