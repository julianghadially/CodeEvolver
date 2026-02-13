# CodeEvolver Tests

This directory contains tests for CodeEvolver components and integration tests for the full optimization pipeline.

## Test Organization

### Remote Unit Tests (`remote_unit_test.py`)
Fast tests for individual components without running the full optimization pipeline.

**Test via `/test/` API endpoints:**
- `test_factchecker_with_renamed_module_via_api()` - Tests build_seed_candidate with refactored code
- `test_gepa_state_structure()` - Validates GEPA state tracking
- `test_candidate_git_branch_extraction()` - Tests candidate structure parsing
- `test_git_branch_naming_convention()` - Tests branch naming

**Run unit tests:**
```bash
# Start Modal app first
modal serve modal_app.py

# Run unit tests (in another terminal)
pytest tests/remote_unit_test.py -v -s -m unit
```

### Integration Tests (`test_gepa.py`)
End-to-end tests that run the full GEPA optimization pipeline.

**Tests:**
- `test_optimization_completes()` - Verifies optimization runs for N iterations
- `test_tracking_structure()` - Validates candidate tracking fields
- `test_score_improves()` - Verifies optimization improves scores
- `test_mutated_branch_has_meaningful_diff()` - Validates code mutations

**Run integration tests:**
```bash
# Start Modal app first
modal serve modal_app.py

# Run integration tests (in another terminal)
pytest tests/test_gepa.py -v -s -m integration
```

## Test Endpoints

The `/test/` API provides endpoints for component testing:

### `POST /test/build-seed`
Test `build_seed_candidate` with a repository, optionally applying refactoring first.

**Example:**
```python
import httpx

response = httpx.post("http://localhost:8000/test/build-seed", json={
    "repo_url": "https://github.com/julianghadially/FactChecker",
    "program": "src.factchecker.modules.fact_checker_pipeline.FactCheckerPipeline",
    "initial_branch": "main",
    "refactor_files": {
        "src/factchecker/modules/research_agent_module.py":
        "src/factchecker/modules/research_agent_module_test.py"
    },
    "refactor_classes": {
        "ResearchAgentModule": "ResearchAgentModuleTest"
    }
})

result = response.json()
print(f"Success: {result['success']}")
print(f"Predictors: {result['predictor_names']}")
```

### `POST /test/gepa-state`
Validate GEPA state structure with a known example.

**Example:**
```python
response = httpx.post("http://localhost:8000/test/gepa-state", json={
    "mock_state": {
        "program_candidates": [
            {
                "_code": json.dumps({
                    "git_branch": "codeevolver-20260212-main",
                    "parent_module_path": "src.factchecker.FactCheckerPipeline",
                    "change_request": "",
                    "last_change_summary": "Initial state"
                }),
                "module_1.predict": "Test instruction",
            }
        ],
        "prog_candidate_val_subscores": [{"val_0": 0.8}],
        "prog_candidate_objective_scores": [{"accuracy": 0.8}],
    }
})

result = response.json()
print(f"Valid: {result['valid']}")
if result.get('errors'):
    print(f"Errors: {result['errors']}")
```

### `POST /test/git-ops`
Test git operations in isolation.

**Example:**
```python
response = httpx.post("http://localhost:8000/test/git-ops", json={
    "operation": "parse_branch",
    "branch_name": "codeevolver-20260212143000-abc123"
})

result = response.json()
print(f"Timestamp: {result['result']['timestamp']}")
print(f"Is main branch: {result['result']['is_main_branch']}")
```

### `GET /test/list`
List all available test operations.

## Test Strategy

### When to Use Remote Unit Tests (`remote_unit_test.py`)
- Testing individual functions (build_seed_candidate, evaluate, etc.)
- Testing with known/mock inputs
- Fast iteration during development
- Testing edge cases and error handling
- **Cost:** ~$0.01-0.10 per test (fast, minimal LM calls)

### When to Use Integration Tests (`test_gepa.py`)
- End-to-end validation of the optimization pipeline
- Verifying code mutations work correctly
- Testing full GEPA state tracking
- Validating real optimization improvements
- **Cost:** ~$1-10 per test (slow, many LM calls, full optimization)

## Adding New Tests

### Adding a Remote Unit Test
1. Add test method to `tests/remote_unit_test.py`
2. Use `@pytest.mark.unit` decorator
3. Use `/test/` endpoints or direct function calls
4. Keep tests fast (< 30 seconds)

### Adding an Integration Test
1. Add test method to `tests/test_gepa.py`
2. Use `@pytest.mark.integration` decorator
3. Use class-scoped fixtures to share optimization runs
4. Expect longer run times (5-60 minutes)

### Adding a Test Endpoint
1. Add endpoint to `src/routes/test_routes.py`
2. Add test implementation to `src/test_functions.py` (reusable utilities)
3. Add Modal function wrapper to `modal_test_functions.py` (Modal decorator)
4. Document in this README
5. Add unit test that uses the endpoint

## pytest.ini Configuration

See `pytest.ini` for:
- Custom markers (`unit`, `integration`)
- Test discovery patterns
- Output configuration

## Examples

### Quick smoke test (unit tests only):
```bash
modal serve modal_app.py &
sleep 10  # Wait for Modal to start
pytest tests/remote_unit_test.py -v -s -m unit
```

### Full validation (unit + integration):
```bash
modal serve modal_app.py &
sleep 10
pytest tests/ -v -s  # Runs all tests
```

### Test a specific component:
```bash
# Test GEPA state structure
pytest tests/remote_unit_test.py::TestGEPAState::test_gepa_state_structure -v -s

# Test FactChecker refactoring
pytest tests/remote_unit_test.py::TestBuildSeedCandidate::test_factchecker_with_renamed_module_via_api -v -s
```

## CI/CD Integration

For CI/CD pipelines:
1. Run unit tests on every commit (fast, cheap)
2. Run integration tests nightly or on release branches (slow, expensive)
3. Use `MODAL_APP_URL` env var to point to deployed app
4. Set appropriate timeouts (unit: 2min, integration: 60min)

## Troubleshooting

**"Modal not available" error:**
- Ensure `modal serve modal_app.py` is running
- Check that `MODAL_APP_URL` env var is set correctly

**"build_seed_candidate failed" error:**
- Check repository URL and branch name
- Verify program import path is correct
- Check Modal logs for detailed errors

**Timeout errors:**
- Increase httpx timeout in test
- Check Modal function timeout settings
- Verify sandbox isn't blocked on user input
