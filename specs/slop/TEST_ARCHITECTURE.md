# Test Architecture

This document explains the organization of test-related code in CodeEvolver.

## File Structure

```
CodeEvolver/
├── modal_app.py                    # Main Modal app (includes test functions)
└── src/
    ├── test_functions.py           # Test implementations (uses GEPASandbox)
    └── routes/
        └── test_routes.py          # FastAPI /test/ endpoints
└── tests/
    ├── remote_unit_tests.py        # Unit tests (use /test/ endpoints)
    ├── test_gepa.py                # Integration tests (full optimization)
    └── README.md                   # Test documentation
```

## Design Principles

### 1. **Production-like Testing with GEPASandbox**
Test functions use **GEPASandbox** to execute tests in the client's environment, exactly like production. This avoids import conflicts between CodeEvolver and client repositories.

**Key Insight**: The test function runs in `web_image` (without DSPy), creates a GEPASandbox with the client's repo, and calls `exec_prebuilt()` to run tests inside the sandbox.

```python
# BAD - Old approach (import conflicts!)
def test_build_seed_candidate():
    # Clone client repo
    # Add client repo to sys.path  ⚠️ CONFLICTS with CodeEvolver's src/
    # Import CodeEvolver's build_seed_candidate
    # Import client's DSPy modules
    # ❌ Both have src/ directories, imports break!

# GOOD - New approach (production-like!)
def test_build_seed_candidate():
    # Create GEPASandbox with client repo
    sandbox = GEPASandbox(app, repo_url, ...)
    sandbox.start(use_venv=True)

    # Run test inside sandbox (via exec_prebuilt)
    result = sandbox.exec_prebuilt({
        "command": "build_seed_candidate",
        "program": "src.module.Program",
    })

    sandbox.stop()
    # ✅ No import conflicts! CodeEvolver and client code stay separate
```

### 2. **Separation of Concerns**
- **`modal_app.py`**: Core application functions + test functions (Modal decorators)
- **`src/test_functions.py`**: Test implementations (business logic, uses GEPASandbox)
- **`src/routes/test_routes.py`**: FastAPI endpoints for tests

### 3. **Modal Decorator Separation**
Modal functions are thin wrappers that delegate to implementations:

```python
# modal_app.py
@app.function(image=gepa_image, timeout=600, ...)  # gepa_image (has GEPA package)
def test_build_seed_candidate(...) -> dict:
    """Modal wrapper with decorator."""
    from src.test_functions import test_build_seed_candidate_impl
    return test_build_seed_candidate_impl(..., app_for_sandbox=app)
```

This keeps Modal-specific code separate from business logic.

## Adding a New Test Function

Let's say you want to add a test for `evaluate()`:

### Step 1: Add Implementation (`src/test_functions.py`)
```python
def test_evaluate_impl(
    repo_url: str,
    program: str,
    metric: str,
    batch: list[dict],
    installation_id: int | None = None,
    app_for_sandbox: Any | None = None,
) -> dict:
    """Test evaluate using GEPASandbox (production-like environment)."""
    from src.optimizer.gepa_sandbox import GEPASandbox
    from src.services.github_app import GitHubAppService

    logs = ["Using GEPASandbox for evaluate test"]
    sandbox = None

    try:
        # Get GitHub token if needed
        github_token = None
        if installation_id:
            github_token = GitHubAppService.get_installation_token(installation_id)

        # Create and start sandbox
        sandbox = GEPASandbox(app=app_for_sandbox, repo_url=repo_url,
                             github_token=github_token, timeout=600)
        sandbox.start(use_venv=True, branch="main")
        logs.append("✓ Sandbox started")

        # Run evaluate via exec_prebuilt (just like in production!)
        result = sandbox.exec_prebuilt({
            "command": "evaluate",
            "program": program,
            "metric": metric,
            "batch": batch,
            "capture_traces": False,
        })

        return {
            "success": result.get("success", False),
            "scores": result.get("scores", []),
            "logs": logs,
        }

    except Exception as e:
        return {"success": False, "error": str(e), "logs": logs}
    finally:
        if sandbox:
            sandbox.stop()
            logs.append("✓ Sandbox stopped")
```

### Step 2: Add Modal Function (`modal_app.py`)
```python
@app.function(
    image=gepa_image,  # Use gepa_image (has GEPA package for GEPASandbox import)
    timeout=600,
    secrets=[modal.Secret.from_name("codeevolver-secrets")],
)
def test_evaluate(
    repo_url: str,
    program: str,
    metric: str,
    batch: list[dict],
    installation_id: int | None = None,
) -> dict:
    """Test evaluate function with a known batch."""
    import sys
    sys.path.append("/app")

    from src.test_functions import test_evaluate_impl
    return test_evaluate_impl(
        repo_url, program, metric, batch,
        installation_id, app_for_sandbox=app
    )
```

### Step 3: Add API Endpoint (`src/routes/test_routes.py`)
```python
class EvaluateTestRequest(BaseModel):
    repo_url: str
    program: str
    metric: str
    batch: list[dict]
    installation_id: int | None = None

@router.post("/evaluate")
async def test_evaluate_endpoint(request: EvaluateTestRequest):
    from modal_app import test_evaluate
    result = await test_evaluate.remote.aio(
        repo_url=request.repo_url,
        program=request.program,
        metric=request.metric,
        batch=request.batch,
        installation_id=request.installation_id,
    )
    return result
```

### Step 4: Add Unit Test (`tests/remote_unit_tests.py`)
```python
@pytest.mark.asyncio
async def test_evaluate_with_known_batch():
    """Test evaluate with a known batch."""
    modal_url = os.getenv("MODAL_APP_URL", "...")

    async with httpx.AsyncClient() as client:
        response = await client.post(f"{modal_url}/test/evaluate", json={
            "repo_url": "https://github.com/...",
            "program": "src.module.Program",
            "metric": "src.metric.metric",
            "batch": [{"input": "test"}],
        })

        result = response.json()
        assert result["success"]
        assert len(result["scores"]) > 0
```

## Benefits

### ✅ **No Import Conflicts**
- CodeEvolver test runs in gepa_image (has GEPA package), client code runs in GEPASandbox (sandbox_image)
- Both can have `src/` directories without conflicts
- Exactly mirrors production environment

### ✅ **Production Parity**
- Tests use the same GEPASandbox as production
- Same image, same dependencies, same execution pattern
- If it works in test, it works in production

### ✅ **Clean Separation**
- `modal_app.py` has Modal decorators
- `src/test_functions.py` has business logic
- Client code stays isolated in sandbox

### ✅ **Easy to Extend**
- Adding a new test function takes ~5 minutes
- Follow the 4-step pattern above
- All tests use the same GEPASandbox pattern

### ✅ **Fast Iteration**
- Test implementations can be modified without redeploying Modal
- Only Modal wrapper needs redeployment when changing resources/timeout

## Common Patterns

### Pattern 1: Test with Refactoring
```python
def test_with_refactoring_impl(..., app_for_sandbox):
    sandbox = GEPASandbox(app=app_for_sandbox, repo_url=repo_url, ...)
    sandbox.start(use_venv=True)

    # Apply refactoring directly in sandbox
    sandbox.exec_bash("mv old_file.py new_file.py")
    sandbox.exec_bash("sed -i 's/OldClass/NewClass/g' src/**/*.py")

    # Run test
    result = sandbox.exec_prebuilt({"command": "..."})

    sandbox.stop()
    return result
```

### Pattern 2: Test with Known Inputs
```python
def test_with_inputs_impl(..., app_for_sandbox):
    sandbox = GEPASandbox(app=app_for_sandbox, repo_url=repo_url, ...)
    sandbox.start(use_venv=True)

    # Run test with known inputs
    result = sandbox.exec_prebuilt({
        "command": "evaluate",
        "batch": [{"input": "test1"}, {"input": "test2"}],
    })

    sandbox.stop()
    return result
```

### Pattern 3: Test Error Handling
```python
def test_error_handling_impl(..., app_for_sandbox):
    sandbox = None
    try:
        sandbox = GEPASandbox(app=app_for_sandbox, repo_url=repo_url, ...)
        sandbox.start(use_venv=True)

        # Intentionally trigger error
        result = sandbox.exec_prebuilt({"command": "invalid"})

    except Exception as e:
        return {"success": False, "error": str(e)}
    finally:
        if sandbox:
            sandbox.stop()
```

## Why GEPASandbox Instead of Direct Execution?

**The Problem**: Loading client repo + CodeEvolver in the same Python process causes import conflicts:

```python
# CodeEvolver has:    src/optimizer/adapter.py
# Client repo has:    src/models/agent.py

sys.path = ["/tmp/client_repo", "/app"]  # Both have src/!

# When you import src.something, Python picks the FIRST match
# Result: ❌ Unpredictable imports, name collisions, chaos
```

**The Solution**: GEPASandbox runs client code in a separate sandbox:

```python
# CodeEvolver Modal function (gepa_image):
#   - Has src/optimizer/, src/routes/, etc.
#   - Has GEPA package (needed for GEPASandbox imports)
#   - NO client code, NO DSPy

# GEPASandbox (created by Modal function, uses sandbox_image):
#   - Has client repo cloned to /workspace/
#   - Has client's requirements.txt installed (including DSPy)
#   - NO CodeEvolver application code

# Communication via JSON over exec_prebuilt()
result = sandbox.exec_prebuilt({"command": "build_seed_candidate", ...})
# ✅ Clean separation, no import conflicts!
```

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ Modal Function (gepa_image)                                  │
│ - Python 3.11                                                │
│ - CodeEvolver code at /app/                                  │
│ - GEPA package (for GEPASandbox imports)                     │
│ - NO DSPy, NO client dependencies                            │
│                                                               │
│  test_build_seed_candidate():                                │
│    sandbox = GEPASandbox(app, repo_url, ...)  ──────┐       │
│    sandbox.start()                                    │       │
│    result = sandbox.exec_prebuilt({...})  ───────────┼────┐  │
│    sandbox.stop()                                     │    │  │
└───────────────────────────────────────────────────────┼────┼──┘
                                                        │    │
                     Creates & Communicates with        │    │
                                                        ▼    ▼
┌─────────────────────────────────────────────────────────────┐
│ GEPASandbox (sandbox_image with claude-agent-sdk + DSPy)    │
│ - Python 3.11 + venv                                        │
│ - Client repo cloned to /workspace/                         │
│ - Client's requirements.txt installed in venv               │
│ - NO CodeEvolver application code                           │
│                                                              │
│  master_script.py:                                          │
│    handle("build_seed_candidate") → build_program()         │
│    print(f"EVAL_RESULT:{json.dumps(result)}")               │
│                                                              │
│  Returns JSON result via stdout ─────────────────────────►  │
└─────────────────────────────────────────────────────────────┘
```
