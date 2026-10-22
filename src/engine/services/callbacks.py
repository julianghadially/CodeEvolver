"""Copyright © 2026 440 Labs LLC

HTTP callback client for optimizer → API communication.

The optimizer process is environment-agnostic: it must reach the Cloud Run
API over plain HTTP with a job-scoped JWT. This module owns that surface.
No DB imports — just `httpx` and a small bag of endpoint
shapes that match `src/api/main.py`'s `/internal/job/{job_id}/*` routes.

Behavior:
  - All calls are best-effort. A failing callback logs and returns; it must
    not crash a running 1-hour optimization loop.
  - Empty `callback_url` or `jwt_token` makes every method a no-op — useful
    for local tests and the `ee_e2e` harness which has no live API.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 60.0


class JobCallbacks:
    """Thin wrapper around the API's `/internal/job/{id}/*` endpoints."""

    def __init__(
        self,
        *,
        callback_url: str | None,
        jwt_token: str | None,
        job_id: str | None,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        base = (callback_url or "").rstrip("/")
        self.base_url = base
        self.job_id = job_id or ""
        self.jwt_token = jwt_token or ""
        self.timeout = timeout
        self.enabled = bool(base and jwt_token and job_id)
        if not self.enabled:
            logger.info(
                "JobCallbacks disabled (callback_url=%r job_id=%r jwt=%s)",
                callback_url, job_id, "set" if jwt_token else "unset",
            )

    # ---- Low-level HTTP helpers ------------------------------------------

    def _headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.jwt_token}"}

    def _status_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/status"

    def _progress_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/progress"

    def _cancel_url(self) -> str:
        return f"{self.base_url}/internal/job/{self.job_id}/check-cancelled"

    def _put_status(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(
                    self._status_url(),
                    json=payload,
                    headers=self._headers(),
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "callback status PUT returned %d: %s",
                        resp.status_code, resp.text[:500],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("callback status PUT raised (non-fatal): %s", exc)

    # ---- Status transitions ----------------------------------------------

    def set_running(self) -> None:
        self._put_status({"status": "running"})

    def set_completed(
        self,
        *,
        final_candidate: dict[str, Any] | None,
        total_metric_calls: int | None,
        num_branches: int | None,
        ce_state: dict[str, Any] | None,
        seed_train_score: float | None = None,
        final_train_score: float | None = None,
        seed_test_score: float | None = None,
        final_test_score: float | None = None,
        seed_test_scores: list[float] | None = None,
        final_test_scores: list[float] | None = None,
        finalization_error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {"status": "completed"}
        if finalization_error is not None:
            payload["finalization_error"] = finalization_error
        if final_candidate is not None:
            payload["final_candidate"] = final_candidate
        if seed_train_score is not None:
            payload["seed_train_score"] = seed_train_score
        if final_train_score is not None:
            payload["final_train_score"] = final_train_score
        if seed_test_score is not None:
            payload["seed_test_score"] = seed_test_score
        if final_test_score is not None:
            payload["final_test_score"] = final_test_score
        if seed_test_scores:
            payload["seed_test_scores"] = list(seed_test_scores)
        if final_test_scores:
            payload["final_test_scores"] = list(final_test_scores)
        if total_metric_calls is not None:
            payload["total_metric_calls"] = total_metric_calls
        if num_branches is not None:
            payload["num_branches"] = num_branches
        if ce_state is not None:
            payload["ce_state"] = ce_state
        self._put_status(payload)

    def set_failed(self, error: str) -> None:
        self._put_status({"status": "failed", "error": error[:4000]})

    # ---- Per-iteration progress ------------------------------------------

    def update_progress(
        self,
        *,
        current_iteration: int,
        final_candidate: dict[str, Any],
        num_branches: int,
        total_metric_calls: int,
        ce_state: dict[str, Any] | None,
        memory_md: str | None = None,
    ) -> None:
        if not self.enabled:
            return
        payload: dict[str, Any] = {
            "current_iteration": current_iteration,
            "final_candidate": final_candidate,
            "num_branches": num_branches,
            "total_metric_calls": total_metric_calls,
        }
        if ce_state is not None:
            payload["ce_state"] = ce_state
        if memory_md is not None:
            payload["memory_md"] = memory_md
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.put(
                    self._progress_url(),
                    json=payload,
                    headers=self._headers(),
                )
                if resp.status_code >= 400:
                    logger.warning(
                        "callback progress PUT returned %d: %s",
                        resp.status_code, resp.text[:500],
                    )
        except Exception as exc:  # noqa: BLE001
            logger.warning("callback progress PUT raised (non-fatal): %s", exc)

    def check_cancelled(self) -> bool:
        if not self.enabled:
            return False
        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.get(self._cancel_url(), headers=self._headers())
                if resp.status_code == 200:
                    data = resp.json()
                    return bool(data.get("cancelled"))
        except Exception as exc:  # noqa: BLE001
            logger.warning("callback cancel GET raised (non-fatal): %s", exc)
        return False


__all__ = ["JobCallbacks"]
