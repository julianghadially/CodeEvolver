"""HTTP callback clients for GEPA sandbox → FastAPI communication.

These classes replace direct MongoDB access inside the GEPA sandbox.
All writes go through the FastAPI internal endpoints, authenticated
by a job-scoped JWT.
"""

import time

import httpx

from gepa.core.state import GEPAState
from modal.app import P

from .gepa_state import GEPAStateRecord


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
        gepa_state: GEPAStateRecord | None = None,
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
            payload["gepa_state"] = gepa_state.to_dict()
        else:
            print("GEPA state is None during CallbackJobUpdater.set_completed")

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
        max_runtime_seconds: int | None = None,
        optimization_start_time: float | None = None,
    ):
        self.base_url = callback_url.rstrip("/")
        self.job_id = job_id
        self.headers = {"Authorization": f"Bearer {jwt_token}"}
        self.debug_max_iterations = debug_max_iterations
        self.max_runtime_seconds = max_runtime_seconds
        self.optimization_start_time = optimization_start_time
        self._grace_period = 600  # 10 minutes

    def _progress_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/progress"

    def _cancel_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/check-cancelled"

    def __call__(self, gepa_state: GEPAState) -> bool:
        """Called each iteration. Returns True to stop (cancellation or timeout)."""
        # Check graceful timeout before any other logic
        if self.max_runtime_seconds and self.optimization_start_time:
            elapsed = time.time() - self.optimization_start_time
            remaining = self.max_runtime_seconds - elapsed
            if remaining < self._grace_period:
                print(
                    f"[PROGRESS] Approaching timeout ({remaining:.0f}s remaining, "
                    f"grace={self._grace_period}s). Stopping gracefully.",
                    flush=True,
                )
                return True

        try:
            iteration = getattr(gepa_state, "i", "?")
            num_candidates = len(gepa_state.program_candidates) if gepa_state.program_candidates else 0
            print(f"[PROGRESS] Callback invoked at iteration {iteration}, {num_candidates} candidates", flush=True)

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

            # Convert live GEPA state to serializable record
            state_record = GEPAStateRecord.from_gepa_state(gepa_state)
            progress_payload = state_record.create_progress_payload(best_score, best_candidate)

            # Log payload summary before sending
            payload_keys = list(progress_payload.keys())
            gepa_state_keys = list((progress_payload.get("gepa_state") or {}).keys())
            print(
                f"[PROGRESS] Sending progress: iteration={progress_payload.get('current_iteration')}, "
                f"best_score={progress_payload.get('best_score')}, "
                f"num_candidates={progress_payload.get('num_candidates')}, "
                f"payload_keys={payload_keys}, gepa_state_keys={gepa_state_keys}",
                flush=True,
            )

            with httpx.Client(timeout=60) as client:
                # Report progress with full GEPA state
                progress_resp = client.put(
                    self._progress_url(),
                    json=progress_payload,
                    headers=self.headers,
                )

                if progress_resp.status_code != 200:
                    print(
                        f"[PROGRESS] ERROR: Progress PUT returned {progress_resp.status_code}: "
                        f"{progress_resp.text[:500]}",
                        flush=True,
                    )
                else:
                    print(f"[PROGRESS] Progress saved successfully (iteration {iteration})", flush=True)

                # Check cancellation
                resp = client.get(self._cancel_url(), headers=self.headers)
                if resp.status_code == 200 and resp.json().get("cancelled"):
                    return True

        except Exception as e:
            # Don't crash the optimization if the callback is unavailable
            import traceback
            print(f"[PROGRESS] ERROR: Callback exception at iteration {getattr(gepa_state, 'i', '?')}: {e}", flush=True)
            print(f"[PROGRESS] Traceback: {traceback.format_exc()}", flush=True)

        return False
