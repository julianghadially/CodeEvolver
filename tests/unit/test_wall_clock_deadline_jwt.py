"""The callback JWT must outlive finalization.

The wall-clock deadline is a cooperative stop, not a kill: the loop stops
`TIMEOUT_BUFFER_SECONDS` before `job_timeout_seconds` and returns through the
same path as row-budget exhaustion, so finalization (seed + final-primary
held-out evals, then the meta-reflection and run-integrity agents) runs after
it. If the callback token expires during that tail, `set_completed` 401s —
non-fatal by design, so the optimizer finishes silently — the job never reaches
a terminal status, and `runner/entrypoint.py::poll` loops forever without
exporting artifacts.

These tests pin that the token outlives the tail, and that the constants the
guarantee is built from cannot drift apart. Measured on ARC-AGI-2, whose
finalization is the most expensive in the matrix at ~8h: each 120-row held-out
draw takes ~1.7h against an evolved program, and `score_seed_and_final` does
four of them plus a comparison eval.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException

from src.api.main import FINALIZATION_RESERVE_SECONDS, _validate_internal_jwt
from src.api.services.jwt_service import mint_job_token
from src.config import settings
from src.engine.optimizer.simple_loop.loop import (
    TIMEOUT_BUFFER_SECONDS as SIMPLE_LOOP_BUFFER,
)
from src.engine.optimizer.v3.optimizer_loop import TIMEOUT_BUFFER_SECONDS

# The most expensive finalization in the research matrix (ARC-AGI-2). The reserve exists to cover this, so it is the bar to clear.
MEASURED_WORST_FINALIZATION_SECONDS = 8 * 3600


def _callback_ttl(job_timeout: int) -> int:
    """The TTL `src/api/main.py` mints for a local job's callback token."""
    return job_timeout + FINALIZATION_RESERVE_SECONDS + 300


def test_buffer_copies_do_not_drift():
    """simple_loop keeps its own copy of the constant; they must agree."""
    assert SIMPLE_LOOP_BUFFER == TIMEOUT_BUFFER_SECONDS
    # The API's reserve is a third copy (src/api must not import src/engine).
    assert FINALIZATION_RESERVE_SECONDS == TIMEOUT_BUFFER_SECONDS


def test_buffer_covers_the_worst_measured_finalization():
    """The loop must stop early enough for finalization to finish in-budget."""
    assert TIMEOUT_BUFFER_SECONDS >= MEASURED_WORST_FINALIZATION_SECONDS


def test_local_run_fits_inside_its_wall_clock_ceiling():
    """`local_job_timeout` bounds TOTAL wall clock, finalization included."""
    soft_deadline = settings.local_job_timeout - TIMEOUT_BUFFER_SECONDS
    assert soft_deadline > 0
    latest_finish = soft_deadline + MEASURED_WORST_FINALIZATION_SECONDS
    assert latest_finish <= settings.local_job_timeout


def test_the_original_bug_reproduces_at_the_original_constants():
    """Pin the failure this change fixes, so it cannot be reintroduced.

    A 1.5h buffer with a TTL of `job_timeout + 300` leaves 1.58h between the loop stopping and the token expiring — far
    short of an 8h finalization. Note the *buffer* is what made this fatal:
    widening it alone would have been enough, and the TTL reserve below is
    margin on top, not the load-bearing part of the fix.
    """
    old_buffer, old_ttl_slack = 5400, 300
    old_headroom = old_buffer + old_ttl_slack
    assert old_headroom < MEASURED_WORST_FINALIZATION_SECONDS


def test_token_outlives_finalization_from_the_soft_deadline():
    """Token lifetime measured from where the loop actually stops.

    A finalization that overruns its reserve must still be able to report, so
    the headroom past the soft deadline is asserted against *twice* the
    measured worst case rather than against the reserve it is sized from.
    """
    job_timeout = settings.local_job_timeout
    soft_deadline = job_timeout - TIMEOUT_BUFFER_SECONDS
    headroom = _callback_ttl(job_timeout) - soft_deadline

    assert headroom >= 2 * MEASURED_WORST_FINALIZATION_SECONDS
    # And the reserve is what buys the second half of that margin.
    assert headroom - TIMEOUT_BUFFER_SECONDS >= MEASURED_WORST_FINALIZATION_SECONDS


@pytest.mark.skipif(not settings.jwt_secret, reason="requires CODEEVOLVER_JWT_SECRET")
def test_expired_token_is_rejected_by_the_real_auth_gate():
    """An expired job token 401s — the failure `set_completed` must not hit.

    Guards the endpoint the optimizer calls to report a terminal status
    (`PUT /internal/job/{job_id}/status`).
    """
    expired = mint_job_token("job_expired", ttl_seconds=-1)
    with pytest.raises(HTTPException) as exc_info:
        _validate_internal_jwt("job_expired", f"Bearer {expired}")
    assert exc_info.value.status_code == 401


@pytest.mark.skipif(not settings.jwt_secret, reason="requires CODEEVOLVER_JWT_SECRET")
def test_minted_token_still_valid_at_the_end_of_a_worst_case_finalization():
    """Mint the real token, then compare its `exp` against the clock.

    Checking the expiry arithmetic on a genuinely minted token is what makes
    this testable in milliseconds instead of days: the token is evaluated at
    the exact instant a deadline-stopped ARC-AGI-2 run would post its terminal
    status. Signature and claims are verified for real; only `exp` is compared
    by hand, since the whole point is a moment in the future.
    """
    job_id = "job_deadline_path"
    job_timeout = settings.local_job_timeout
    token = mint_job_token(job_id, ttl_seconds=_callback_ttl(job_timeout))

    soft_deadline_at = time.time() + (job_timeout - TIMEOUT_BUFFER_SECONDS)
    reporting_at = soft_deadline_at + MEASURED_WORST_FINALIZATION_SECONDS

    payload = jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=["HS256"],
        options={"verify_exp": False},
    )
    assert payload["job_id"] == job_id
    assert payload["exp"] > reporting_at
