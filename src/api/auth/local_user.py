"""Local-mode auth stub.

In a local deploy the API binds to 127.0.0.1 and serves a single operator
on a trusted local machine; there is no Clerk, no multi-tenancy, no API key.
`get_current_user` short-circuits to this fixed `CustomerRecord` so all
authenticated endpoints work uniformly with no auth header.

The loopback bind (uvicorn `--host 127.0.0.1`) is the security boundary —
anyone with shell access to the box can read the workspace and venv
anyway, so adding a key would not raise the bar.
"""
from __future__ import annotations

from src.db.db_schemas.customer import CustomerRecord


LOCAL_OPERATOR_CLERK_ID = "local-operator"

LOCAL_STUB_USER = CustomerRecord(
    clerk_id=LOCAL_OPERATOR_CLERK_ID,
    clerk_email="local@codeevolver.ai",
    first_name="Local",
    last_name="Operator",
    allowed=True,
)
