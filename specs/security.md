# Security

## Mongo DB <> Sandbox Security Solution: Job-Scoped Callback API with JWT

The GEPA sandbox does not receive MongoDB credentials. Instead it gets a short-lived JWT scoped to a single `job_id` and calls back to three internal FastAPI endpoints over HTTP.

### Architecture

```
FastAPI (has MongoDB)                    GEPA sandbox (no MongoDB)
─────────────────────                    ─────────────────────────
POST /optimize                           run_optimization()
 ├─ creates job in MongoDB                ├─ receives callback_url + jwt_token
 ├─ mints JWT {job_id, scope, exp}        ├─ CallbackJobUpdater  → PUT /status
 └─ spawns run_optimization(...)          └─ CallbackProgressTracker
                                               ├─ PUT /progress
Internal endpoints (JWT-validated):            └─ GET /check-cancelled
 PUT  /internal/job/{job_id}/status
 PUT  /internal/job/{job_id}/progress
 GET  /internal/job/{job_id}/check-cancelled
```

### JWT Token

- **Algorithm**: HS256 (symmetric). Minter and validator are the same FastAPI app.
- **Secret**: `CODEEVOLVER_JWT_SECRET` env var, shared only with the FastAPI layer.
- **TTL**: `gepa_optimization_timeout + 300s` (job timeout plus a 5-minute buffer).
- **Payload**:
  ```json
  {
    "job_id": "job_abc123",
    "scope": "job:update",
    "iat": 1706300000,
    "exp": 1706303600
  }
  ```
- The API validates that the JWT's `job_id` matches the URL path parameter. A token for job A cannot update job B.

### What the sandbox can and cannot do

| Can | Cannot |
|-----|--------|
| Set its own job to running/completed/failed | Read or modify any other job |
| Write iteration progress (score, candidate) | Access any other MongoDB collection |
| Check if its job was cancelled | Operate after the JWT expires |

### Callback error handling

`CallbackProgressTracker` silently swallows all HTTP errors to avoid crashing the GEPA optimization loop — same behaviour as the previous `MongoDBProgressTracker`. If the callback is unreachable, optimization continues; progress just isn't persisted until connectivity resumes.

### Operational requirements

1. Set `CODEEVOLVER_JWT_SECRET` in Modal secrets (generate with `python3 -c "import secrets; print(secrets.token_hex(32))"`)
2. Set `CODEEVOLVER_CALLBACK_URL` to the FastAPI Modal web endpoint URL
3. Pass `codeevolver-sandbox-secrets` that excludes `MONGO_KEY`

### Future work
- Network egress allowlist: restrict sandbox outbound traffic to only the callback URL.
