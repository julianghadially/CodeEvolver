"""HTTP callback clients for GEPA sandbox → FastAPI communication.

These classes replace direct MongoDB access inside the GEPA sandbox.
All writes go through the FastAPI internal endpoints, authenticated
by a job-scoped JWT.
"""

import httpx

from gepa.core.state import GEPAState


class CallbackJobUpdater:
    """Update job status via HTTP callbacks to the FastAPI internal API.

    Each method issues a synchronous PUT to /internal/job/{job_id}/status.
    If callback_url is empty, all methods are no-ops.
    """

    def __init__(self, callback_url: str, jwt_token: str, job_id: str):
        self.base_url = callback_url.rstrip("/") if callback_url else ""
        self.job_id = job_id
        self.headers = {"Authorization": f"Bearer {jwt_token}"}

    def _status_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/status"

    def set_running(self) -> None:
        with httpx.Client(timeout=60) as client:
            client.put(
                self._status_url(),
                json={"status": "running"},
                headers=self.headers,
            )

    def set_completed(
        self,
        best_candidate: dict,
        best_score: float,
        total_metric_calls: int,
        num_candidates: int,
    ) -> None:
        with httpx.Client(timeout=60) as client:
            client.put(
                self._status_url(),
                json={
                    "status": "completed",
                    "best_candidate": best_candidate,
                    "best_score": best_score,
                    "total_metric_calls": total_metric_calls,
                    "num_candidates": num_candidates,
                },
                headers=self.headers,
            )

    def set_failed(self, error: str) -> None:
        with httpx.Client(timeout=60) as client:
            client.put(
                self._status_url(),
                json={"status": "failed", "error": error},
                headers=self.headers,
            )


class CallbackProgressTracker:
    """Drop-in replacement for MongoDBProgressTracker.

    Implements the StopperProtocol: __call__(gepa_state) -> bool.
    Sends iteration progress and checks for cancellation via HTTP.
    Silently swallows all exceptions to avoid crashing the GEPA loop.
    """

    def __init__(self, callback_url: str, jwt_token: str, job_id: str):
        self.base_url = callback_url.rstrip("/")
        self.job_id = job_id
        self.headers = {"Authorization": f"Bearer {jwt_token}"}

    def _progress_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/progress"

    def _cancel_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/check-cancelled"

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Called each iteration. Returns True to stop (cancellation)."""
        try:
            full_scores = gepa_state.program_full_scores_val_set
            best_score = max(full_scores) if full_scores else 0.0
            best_idx = full_scores.index(best_score) if full_scores else 0
            best_candidate = (
                gepa_state.program_candidates[best_idx]
                if gepa_state.program_candidates
                else {}
            )

            with httpx.Client(timeout=60) as client:
                # Report progress
                client.put(
                    self._progress_url(),
                    json={
                        "current_iteration": gepa_state.i + 1,
                        "best_score": best_score,
                        "best_candidate": best_candidate,
                        "total_metric_calls": gepa_state.total_num_evals,
                        "num_candidates": len(gepa_state.program_candidates),
                    },
                    headers=self.headers,
                )

                # Check cancellation
                resp = client.get(self._cancel_url(), headers=self.headers)
                if resp.status_code == 200 and resp.json().get("cancelled"):
                    return True

        except Exception:
            # Don't crash the optimization if the callback is unavailable
            pass

        return False
