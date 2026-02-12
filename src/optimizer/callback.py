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
        gepa_state: GEPAState | None = None,
    ) -> None:
        payload = {
            "status": "completed",
            "best_candidate": best_candidate,
            "best_score": best_score,
            "total_metric_calls": total_metric_calls,
            "num_candidates": num_candidates,
        }

        # Serialize GEPA state if provided
        if gepa_state is not None:
            payload["gepa_state"] = {
                "program_candidates": gepa_state.program_candidates,
                "candidate_scores": gepa_state.program_full_scores_val_set,
                "parent_programs": gepa_state.parent_program_for_candidates,
                "num_iterations": gepa_state.i + 1,
                "total_evals": gepa_state.total_num_evals,
            }

        with httpx.Client(timeout=60) as client:
            client.put(
                self._status_url(),
                json=payload,
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

    def __init__(
        self,
        callback_url: str,
        jwt_token: str,
        job_id: str,
        debug_max_iterations: int | None = None,
    ):
        self.base_url = callback_url.rstrip("/")
        self.job_id = job_id
        self.headers = {"Authorization": f"Bearer {jwt_token}"}
        self.debug_max_iterations = debug_max_iterations

    def _progress_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/progress"

    def _cancel_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/check-cancelled"

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Called each iteration. Returns True to stop (cancellation)."""
        try:
            if self.debug_max_iterations is not None:
                # DEBUG: Print GEPAState attributes on first iteration
                if gepa_state.i == 0:
                    print("\n" + "="*80, flush=True)
                    print("[DEBUG] GEPAState attributes on first iteration:", flush=True)
                    print(f"  - program_candidates: {len(gepa_state.program_candidates)} candidates", flush=True)
                    print(f"  - parent_program_for_candidate: {len(gepa_state.parent_program_for_candidate)} entries", flush=True)
                    print(f"  - prog_candidate_val_subscores: {len(gepa_state.prog_candidate_val_subscores)} entries", flush=True)
                    print(f"  - i (iteration): {gepa_state.i}", flush=True)
                    print(f"  - total_num_evals: {gepa_state.total_num_evals}", flush=True)

                    # Show first candidate structure
                    if gepa_state.program_candidates:
                        first = gepa_state.program_candidates[0]
                        print(f"\n[DEBUG] First candidate keys: {list(first.keys())}", flush=True)
                        for key, value in list(first.items())[:3]:
                            preview = value[:100] if isinstance(value, str) else str(value)[:100]
                            print(f"    {key}: {preview}...", flush=True)
                    print("="*80 + "\n", flush=True)

            full_scores = gepa_state.program_full_scores_val_set
            best_score = max(full_scores) if full_scores else 0.0
            best_idx = full_scores.index(best_score) if full_scores else 0
            best_candidate = (
                gepa_state.program_candidates[best_idx]
                if gepa_state.program_candidates
                else {}
            )

            # Check for early stop (debug mode)
            if self.debug_max_iterations is not None:
                if gepa_state.i + 1 >= self.debug_max_iterations:
                    print(f"\n[DEBUG] Early stop triggered at iteration {gepa_state.i + 1}/{self.debug_max_iterations}", flush=True)
                    print(f"[DEBUG] Total candidates evaluated: {len(gepa_state.program_candidates)}", flush=True)
                    print(f"[DEBUG] Best score so far: {best_score:.4f}", flush=True)
                    return True  # Stop optimization

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

        except Exception as e:
            # Don't crash the optimization if the callback is unavailable
            print(f"[DEBUG] Callback exception: {e}", flush=True)
            pass

        return False
