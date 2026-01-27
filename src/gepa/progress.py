"""MongoDB progress tracker for GEPA optimization.

Implements StopperProtocol: called each iteration by the GEPA engine to
persist state to MongoDB and check for job cancellation.

Uses pymongo (sync driver) because GEPA's optimize() loop is synchronous.
"""

from datetime import datetime, timezone

from gepa.core.state import GEPAState
from src.db.mongo import get_mongo_db


class MongoDBProgressTracker:
    """Persist GEPA optimization progress to MongoDB each iteration.

    Also serves as a cancellation mechanism: if the job status is set
    to "cancelled" in MongoDB, the optimization loop stops.

    Args:
        mongodb_url: MongoDB connection string.
        database_name: Database name.
        job_id: Unique job identifier.
    """

    def __init__(
        self,
        job_id: str,
    ):
        self.db = get_mongo_db()
        self.job_id = job_id

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Called each iteration. Persists progress and checks for cancellation.

        Returns True to stop the optimization (cancellation), False to continue.
        """
        try:
            # Compute best score from aggregate val scores
            full_scores = gepa_state.program_full_scores_val_set
            best_score = max(full_scores) if full_scores else 0.0
            best_idx = full_scores.index(best_score) if full_scores else 0
            best_candidate = (
                gepa_state.program_candidates[best_idx]
                if gepa_state.program_candidates
                else {}
            )

            update = {
                "current_iteration": gepa_state.i + 1,
                "best_score": best_score,
                "best_candidate": best_candidate,
                "total_metric_calls": gepa_state.total_num_evals,
                "num_candidates": len(gepa_state.program_candidates),
                "updated_at": datetime.now(timezone.utc),
            }

            self.db.jobs.update_one(
                {"job_id": self.job_id},
                {"$set": update},
            )

            # Check for cancellation
            job = self.db.jobs.find_one({"job_id": self.job_id}, {"status": 1})
            if job and job.get("status") == "cancelled":
                return True  # Stop optimization

        except Exception:
            # Don't crash the optimization if MongoDB is temporarily unavailable
            pass

        return False

    def close(self):
        '''Not needed with global client'''
        pass
