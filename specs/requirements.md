# Requirements
Copyright © 2026 440 Labs LLC

See `specs/engine3.md` for the optimizer design, `specs/research_plan.md` for the experiment design that motivates the optimizers, and `specs/experiment-docs/` for how the research matrix runs at scale (`experiment-fleet.md`) or on one host (`local-deployment.md`).

## Overview
CodeEvolver is a self-improvement engine offering autonomous coding agents for building reliable AI systems. It evolves your AI system code against a dataset and an outcome metric until it performs well on that metric.

This combines several mechanisms:
- **Optimizer algorithm:** an `IterationArchitect` (one coding-harness session per iteration) runs evaluations, invokes the coding agent, and records an outcome reflection. The optimizers differ in who chooses the parent candidate and how much population management the agent sees (see Optimizer below).
- **Coding agents**: Autonomous agents execute the code changes the architect requests.
- **Git branching:** candidates are git commits on `codeevolver-*` branches, not prompt dicts.
- **Sandboxing for security:** coding agents are a cyber risk without isolation and network policy; research runs execute in a locked-down Docker runner (see Security Architecture).

### Optimizer
The optimizers live under `src/engine/optimizer/`. Engine 3's state substrate is `src/engine/optimizer/v3/ce_state.py::CEState` (branch registry, primary branch, closing records, eval budget, comparison history).

The optimizers `ee`, `freeform` and `pareto_evolution` (and the `asa` arm, which is implemented but not part of the paper) share one driver (`v3/optimizer_loop.py`) and one `IterationArchitect` session shape; they differ only by an `OptimizerStrategy` (`v3/strategy.py`). The driver's mechanics live in sibling modules: `v3/close_effects.py` (git side-effects of closing), `v3/analysis_agent.py` (per-iteration change summaries), `v3/run_reporting.py` (progress payloads and callbacks) and `v3/finalization.py` (held-out test and comparison scoring, post-run auditors).
- `ee` (Treatment A): the architect owns git. It picks its branch, runs baseline and post-change train evals, decides when to buy comparison-set evidence, and registers its branch and the primary through the `close_iteration` tool.
- `freeform` (Baseline 1, `specs/v3_freeform.md`): EE's harness with no explore/exploit guidance and no population management. The agent owns git but is never told about branches or a primary; `close_iteration` takes only `memory_updated`, and the loop registers whatever HEAD is at close (`v3/head_close.py`), so the final program is the HEAD of the last closed iteration.
- `pareto_evolution` (Treatment B): the freeform prompt, skills and memory template, but the LOOP owns candidate selection (Pareto-proportional sampling over the comparison-set instance frontier, with an exploit-and-merge draw late in the budget), the fitness measurement, admission and the primary (population best). The agent's `close_iteration` takes only `memory_updated` and it stays on the loop-forked branch.
- `simple_loop` (Baseline 0, `src/engine/optimizer/simple_loop/`, `specs/v3_simple_loop.md`): a fresh OpenCode session per iteration on one linear branch with only the `run_eval` tool, the eval-row budget and a post-session constraint gate.

Every engine-3 architect also gets a once-per-run `run_variance_estimate` tool (`v3/variance_estimate.py`, `specs/archive-public/sampling_variance.md`) that measures the training subsample's run-to-run noise. All arms share the post-run auditors in `src/engine/optimizer/meta/` (meta-reflection and run-integrity).

The `/optimize` endpoint launches the optimizer through `src/launchers/` as a local subprocess. The optimizer process is **environment-agnostic**: it has no database or cloud SDK imports. It runs as a single Python process against a workspace (the client repo checkout) with API keys in env vars. The launcher writes an optimization config JSON (serialized from `OptimizationConfig`, the single config schema, `src/engine/optimizer/v3/optimization_config.py`) and execs the method's `v3/<method>/optimization_entry.py` (or `simple_loop/optimization_entry.py`). Each entry calls the shared `src/engine/optimizer/v3/optimizer_loop.py::run_optimization` (simple_loop has its own loop), which owns the full run: dataset resolution, seed step, hook wiring, and the architect-driven loop body.

### Coding Agents
CodeEvolver agents run through a selected coding harness (`opencode`, the default, or `claude`) in a fully autonomous mode, with bash / grep / glob and file edit access to the workspace. OpenCode mode uses generated config, staged CodeEvolver skills, and the architect tool contracts over an MCP bridge; Claude mode uses the Claude Agent SDK. After code changes are made, the optimizer runs the mutated program and returns the output.

Code changes are made over many iterations driven by the `IterationArchitect`. Each mutation is small, but costs add up.

### Repository Connection
In local mode the operator registers a checkout on disk with `POST /local/connect-repo` (`src/api/routes/local_repo.py`); the repo record's `workspace_path` is the in-place workspace the launcher uses. `/optimize` looks up the repo by `(repo_url, clerk_id)`.

Optimization branches can be pushed to GitHub when the repo record carries a GitHub App `installation_id`: the API mints a short-lived installation token (`src/shared/services/github_app.py`), `git_push_authenticated` pushes only `codeevolver-*` branches, and the optimizer refreshes the token through `GET /internal/job/{job_id}/github-token`. In local mode pushing is skipped; candidates live as commits in the local checkout (and in the Docker runner's exported `repo.bundle`).

### Security
Security should be designed for from day one, because autonomous coding agents introduce the trifecta of security risk:
1. Untrusted inputs, including prompt injection embedded into popular sites
2. Network access
3. Access to user data (RAG databases, code, secrets, and possibly PII).

See Security Architecture below.

### Execution environments
CodeEvolver runs in two ways:
- **Local mode** (`APP_MODE=local`, `SANDBOX_MODE=local`): the API, MongoDB and the optimizer run on one machine, in place on the operator's checkout. `src/local/local_environment.py::LocalEnvironment` prepares the workspace; `src/launchers/local_launcher.py` runs the optimizer subprocess, tees its stdout to `~/.codeevolver/logs/{job_id}.log` and parses the `OPTIMIZATION_RESULT:` marker. Cancellation is cooperative (the optimizer polls `/internal/job/{id}/check-cancelled`) and forceful (`cancel(job_id)` SIGTERMs the registered subprocess).
- **Docker runner** (`runner/`, driven by `orchestrator/`): each research run is one container that clones a squashed seed bundle, starts a local API and MongoDB, runs one optimization in local mode behind an egress-filtering gateway, and exports its artifacts. See `specs/experiment-docs/experiment-fleet.md`.

The hosted portal API (`SANDBOX_MODE=cloud`) serves results (Clerk auth, MongoDB Atlas); it does not launch optimizations (`/optimize` returns 501). Research runs are imported into it with `POST /import-job-history`.

#### Sandbox Dependency Conventions
Client repositories declare dependencies via two files:
- **`requirements.txt`** — Python packages, installed into the workspace venv with **uv** (`uv pip install --python <venv>/bin/python -r requirements.txt`).
- **`runtime_requirements.txt`** — Shell commands to install non-Python system runtimes. Each non-comment line is executed as a bash command. This is the system-level equivalent of `requirements.txt`.

Common runtimes like Deno are pre-installed in the runner image. The `runtime_requirements.txt` file is for anything else the client needs that isn't a Python package (e.g., Bun, custom tools).

**Installer convention (uv).** `LocalEnvironment` standardizes on **uv** as the single dependency installer — there is no pip fallback. uv is a host-level binary (required on PATH, provided by the runner image), not a venv package: it installs into the venv via the interpreter path, so the **target venv needs neither pip nor uv inside it**. This makes installs immune to pip-less venvs and keeps resolution reproducible. `LocalEnvironment` reuses an existing `<repo>/env` venv as-is and only creates a fresh one — via `uv venv` — when absent, so a user's venv is never clobbered.

#### Optimizer Architecture
The optimizer is **environment-agnostic**: it runs as a single Python process with no database imports. The environment layer (local or the Docker runner) sets up the workspace (checkout, deps, env vars), then execs `optimization_entry.py`. The optimizer uses subprocess calls for isolation — client code crashes don't kill the optimizer.

```
API (uvicorn, local mode)
 │
 │ spawn_optimization()  (src/launchers/local_launcher.py)
 ▼
Optimizer Process (v3/<method>/optimization_entry.py → run_optimization)
  │
  │  seed step
  │  loop:
  │    IterationArchitect.run()     ← selected coding harness session
  │      ├── run_eval               ← subprocess (client code isolation)
  │      ├── coding_agent           ← selected coding harness
  │      ├── inspect_traces         ← subprocess (inspection agent)
  │      └── close_iteration        ← in process
  │    admission / closing logic    ← in process
  │    state management             ← in process
  │  finalization (held-out test, auditors)
  │
  │  PUT /internal/job/{id}/progress|status → API (progress/results callbacks)
  │  print OPTIMIZATION_RESULT:{json} → stdout (final result for the launcher)
```

The canonical `src/__init__.py` merges the client repo's `src/` into the `src` package path when `CODEEVOLVER_WORKSPACE_SRC` is set, so a client repo that ships `src/<pkg>/` (e.g. `src.factchecker.X`) resolves under the same `src` namespace as the engine's `src.engine.X`. `src/engine/services/subprocess_service.py::_build_env` sets it to `<workspace>/src` on every evaluator / coding-agent / inspection-agent subprocess.

**Engine import boundaries.** Engine modules (`src/engine/`) must not import from:
- `src/db/` — database schemas and MongoDB connection. `IterationHistoryRecord` (the one engine-relevant schema) is imported from `src/db/db_schemas/` by exception.
- `src/api/` — API routes, auth middleware.
- `src/config.py` — application settings with env-var loading. The optimizer receives config as a JSON file, not via `Settings`.
- `src/shared/`.

**Import convention.** The repo uses `from src.engine.X import Y` everywhere. No try/except dual imports, no `sys.path` hacks, no import rewriting.

**Engine env-var convention.** `src/engine/paths.py` is the central declaration site for every engine-side env var. Each env var has an accessor function (`traces_dir()`, `memory_dir()`, `repo_venv_dirname()`, `repo_venv_bin()`) that names the env-var string inline and returns the resolved value. Consumers call the accessor, never `os.environ` directly. For env vars that must be **written** by one module so a different module or subprocess can read them, the name is listed in `paths.CROSS_PROCESS_ENV_VARS` to document the cross-module string contract; writers use the literal string. This includes the namespace-merge var (`CODEEVOLVER_WORKSPACE_SRC`), local venv selection (`REPO_VENV_DIR`), per-job artifact scoping (`CODEEVOLVER_JOB_ID`), and the OpenCode harness's provider-process env vars (`CODEEVOLVER_ARCHITECT_MCP_PROXY_*`, `OPENCODE_CONFIG_*`, etc.). `CODEEVOLVER_JOB_ID` is written once by `src/engine/optimizer/v3/_entry.py::run_entry` from the optimization config's `job_id` field; `paths.{traces_dir,memory_dir,runtime_tests_dir}` append `/{job_id}` so per-job artifacts stay isolated across successive runs on the same host. This is separate from the API-tier secret/dev-prod convention in `src/config.py::env_var()` documented below — the engine does not import `src/config.py`.

**Entry point.** `src/engine/optimizer/v3/<method>/optimization_entry.py` parses `--config <path>`, loads env vars from the workspace `.env`, runs the optimization, and emits `OPTIMIZATION_RESULT:{json}` on stdout. It wraps everything in try/except to emit structured errors.

**Subprocess service.** `src/engine/services/subprocess_service.py` provides `run_evaluation()`, `run_coding_agent()`, `run_inspection_agent()`, and git helper functions — all standalone subprocess-based functions. Engine tools call these directly rather than going through an RPC layer.

**Token refresh.** GitHub installation tokens expire after 1 hour. `subprocess_service.refresh_github_token(callback_url, job_id, jwt_token)` calls `GET /internal/job/{job_id}/github-token` with the job JWT; `git_push_authenticated` invokes it before each push. The callback URL, JWT, job id, repo URL, GitHub token, and initial branch are bundled in a `JobContext` (`src/engine/services/job_context.py`) and reach the optimizer via the optimization config JSON. Both are no-ops in local mode.

Per-iteration tracking (`IterationHistory`) and the cross-iteration memory system (`memory.md`, `old_memory.md`, `codeevolver.md`, iteration change summaries) live inside the engine — see `specs/engine3.md`.

### Testing
Tests live in `tests/` (inventory and commands in `tests/README.md`):
- Offline suites (free): `pytest tests/unit tests/controls tests/coding_harness -m "not live and not integration"`.
- Functional agent tests and end-to-end runs call real models and cost money. They run through `local_test_app.py` and `scripts/run_tests.sh` (`--phase functional`, `--include-<arm>-e2e`).

### Privacy
- Option for users to run the execution environment on their own infrastructure (local mode or the Docker runner).
- No unnecessary third-party services. The hosted portal uses Google Cloud Run (API hosting), Clerk (authentication) and MongoDB Atlas (database).

### Technology Stack and Architecture
- **Language**: Python
- **API Framework**: FastAPI (uvicorn locally; Google Cloud Run for the hosted portal API)
- **Database**: MongoDB — a local Community instance in local mode and the Docker runner; MongoDB Atlas for the hosted portal (database `codeevolver` in prod, `codeevolver_dev` in dev/local)
- **Execution Environment**: local subprocess, or one Docker container per run on cloud hosts (`runner/`, `orchestrator/`)
- **Authentication**: Clerk (identity provider) with dual auth — Clerk JWTs for browser sessions, API keys for programmatic access. Local mode skips Clerk and serves a single operator. See Authentication below.

### Project Structure
Code is organized by deployment function under `src/`:
- **`src/api/`** — FastAPI application, request/response schemas, JWT service
- **`src/api/auth/`** — Clerk webhook handler, dual auth middleware (JWT + API key), auth dependencies
- **`src/api/routes/local_repo.py`** — local-mode repo registration (`POST /local/connect-repo`)
- **`src/engine/`** — The optimizers (`optimizer/v3/` — `ee`, `freeform`, `pareto_evolution`, `asa`; `optimizer/simple_loop/`), the post-run auditors (`optimizer/meta/`), the framework-agnostic evaluator (`evaluator/`), coding and inspection agents (`agent/`), the evaluation master script and environment validation (`sandbox/`), and the subprocess service (`services/`). Never imports GitPython or httpx at module level.
- **`src/launchers/`** — the optimization launcher (`local_launcher.py`)
- **`src/local/`** — local-mode workspace setup (`LocalEnvironment`), process registry, repo lock, local history mirror
- **`src/db/`** — MongoDB connection, database record schemas (shared across all layers)
- **`src/shared/`** — Enums and types used across API/engine/db (JobStatus, ProgramStatus, etc.), GitHubAppService
- **`runner/`** — Container image + in-container run driver for research runs (`entrypoint.py` is PID 1 of a run: clone squashed seed repo → egress canary → mongod + local API → submit → poll → export artifacts). See `specs/experiment-docs/experiment-fleet.md`.
- **`orchestrator/`** — controller that dispatches runs to cloud hosts over `docker --context` (SSH); one matrix per experiment, launched with `orchestrator/launch.sh`. See `specs/experiment-docs/experiment-fleet.md`.
- **`experiments/`** — per-benchmark optimization configs (`experiments/<benchmark>/input.py`) and the job submitter (`experiments/run.py`).
- **`scripts/`** — operator and analysis tools, none part of the engine:
  - `run_meta_reflection.py` — replays the post-run meta-reflection auditor on a collected run directory.
  - `best_of_population/` — offline best-of-population audit (`scripts/best_of_population/README.md`).
  - `macro-counts/` — macro-approach and prompt-lesson counts (`scripts/macro-counts/README.md`).
  - `failure-modes/` — failure-mode counts (`scripts/failure-modes/README.md`).
  - `results_tables.py`, `lineage_audit.py`, `transplant_audit.py`, `prompt_only_audit.py` — paper tables and audits over collected runs (`$CE_RUNS_DIR`, default `~/ce-runs`).
  - `smoke_eval.py` — runs N rows (default 5) of a benchmark's eval on the local client checkout and prints per-row scores plus the captured span chains. A viewer, not a gate: it never raises on a low score, and it skips benchmarks that fail preflight. `--split test` is opt-in and routes through the `holdout_*` suffix so those traces keep the `_holdout` deny-path fragment.
- **`src/config.py`** — Application settings (cross-cutting), environment helpers

### Environments (Dev vs Prod)

The application supports three environments controlled by `APP_MODE` (`local`, `dev` or `prod`).

**Key config helpers in `src/config.py`:**
- `determine_environment()` — returns `'local'`, `'dev'`, or `'prod'` based on `APP_MODE`.
- `determine_sandbox_mode()` — returns `'local'` (launches optimizations) or `'cloud'` (hosted portal API). `SANDBOX_MODE` sets it explicitly; `APP_MODE=local` implies `local`.
- `env_var(name)` — loads the right environment variable by appending `_DEV` in dev/local mode. E.g., `env_var("MONGO_KEY")` reads `MONGO_KEY_DEV` in dev, `MONGO_KEY` in prod. **All new secrets and environment-specific config should use `env_var()` rather than reading `os.environ` directly.**
- `determine_database_name()` — returns `'codeevolver_dev'` in dev/local, `'codeevolver'` in prod.

| | Prod | Dev |
|---|---|---|
| Frontend URL | `codeevolver.ai` (portal at `/portal`) | `dev.codeevolver.ai` |
| API URL | `api.codeevolver.ai` | `dev-api.codeevolver.ai` |
| MongoDB database | `codeevolver` | `codeevolver_dev` |
| Env var suffix | (none) | `_DEV` |
| Clerk app | Production Clerk | Dev Clerk (separate app) |

Frontend and API URLs are centrally defined in `src/config.py` as `FRONTEND_URLS` and `API_URLS` dicts. All code should reference these rather than hardcoding URLs. The hosted portal connects to Atlas with `MONGO_HOST`, `MONGO_USER` and `MONGO_KEY` (each through `env_var()`); local mode uses `LOCAL_MONGO_URI` (default `mongodb://localhost:27017`).

When adding a new secret or environment-specific variable:
1. Add the prod var (e.g., `MY_SECRET`) and dev var (e.g., `MY_SECRET_DEV`) to the environment
2. In `Settings`, set the default using `env_var("MY_SECRET")`
3. The correct value will be selected automatically based on `APP_MODE`

**Importing local/research runs.** `POST /import-job-history` (client API key) upserts a locally-run job into the cloud `jobs` collection under the caller's `clerk_id`. Besides the iteration history it accepts optional `name` (display name — the portal shows it instead of `job_id`; research runs use the run id, e.g. `pareto_evolution-hover-s2-qwen3c`), `seed_test_score` / `final_test_score`, and the `memory_md`, `meta_reflection_md` (`meta_reflection.md`) and `integrity_audit_md` (`run_integrity.md`) reports. Omitted/None fields are skipped, so a re-import never wipes a stored value. `GET /job/{id}` returns all of these; `GET /jobs` returns `name` and the scores but leaves the three markdown reports `None` to keep the list small. `GET /jobs` projects away `ce_state`/`ce_state_history`/the reports and relies on the `jobs {clerk_id: 1, created_at: -1}` index (ensured idempotently in `src/db/mongo.py::lifespan`); without it, large research jobs push Mongo's in-memory sort past its 32 MB limit and the list fails.

### Authentication

CodeEvolver uses **Clerk** as the identity provider with a dual authentication model:
- **Clerk JWTs** — for browser/frontend sessions. The frontend includes the Clerk session token in `Authorization: Bearer <jwt>` with `Authorization-Type: clerk`.
- **API keys** — for programmatic access. Sent as `Authorization: Bearer <api_key>` with `Authorization-Type: api_key`.

Both auth paths resolve to a `clerk_id`, which is the universal user identifier across all MongoDB collections. In local mode `get_current_user` returns a single local stub user and skips Clerk.

**User registration:** When a user signs up via Clerk, a `user.created` webhook fires to `POST /clerk/new-user`. This creates a `customers` record and generates an API key (stored as SHA-256 hash for lookup + Fernet-encrypted for recovery).

**MongoDB collections for auth:**
- `customers` — user profile (clerk_id, email, name, allowed flag)
- `auth` — API key credentials (clerk_id, api_hash, encrypted key, revoked flag)

**Auth code lives in `src/api/auth/`:**
- `middleware.py` — resolves request headers to `clerk_id` (Clerk JWT or API key)
- `dependencies.py` — `get_current_user` FastAPI dependency for route protection
- `clerk_webhook.py` — Svix-verified webhook handler for user registration
- `utils.py` — key generation, hashing, encryption
- `routes.py` — `/get-api-key` endpoint

All public API endpoints require authentication via `Depends(get_current_user)`. Internal endpoints (`/internal/*`) use a separate HS256 job-scoped JWT for optimizer-to-API callbacks.

**Job ownership:** `JobRecord` includes a `clerk_id` field set at creation. Public job endpoints (`GET /job/{job_id}`, `GET /job/{job_id}/state`) verify the authenticated user's `clerk_id` matches the job's `clerk_id`. Non-matching requests return 404 (not 403) to prevent job ID enumeration.

#### Optimizer → API callbacks
The optimizer calls back to the API's internal endpoints (`/internal/job/{job_id}/*`) to report progress, update status, check cancellation, and refresh GitHub tokens. These are authenticated by job-scoped HS256 JWTs minted by the API at job creation and passed in the optimization config. The JWT contains `job_id` and `scope: "job:update"` claims, with a TTL of `local_job_timeout` plus a finalization reserve.

When `CODEEVOLVER_JWT_SECRET` is unset in local mode, `src/config.py::_default_jwt_secret()` auto-generates an ephemeral `secrets.token_urlsafe(32)` at Settings init. The same uvicorn process both mints and validates the token, so the in-memory secret is sufficient. `determine_api_url()` reads from the central `API_URLS` dict so local mode resolves `callback_url` to `http://localhost:8080`.

## Flexible, Reliable Sandboxes Work for Every Client
We want the execution environment to work with every code base. An environment check quickly tells the user whether there is a failure.

The validation is implemented in `src/engine/sandbox/verify_environment.py` as the framework-agnostic function `validate_sandbox_environment(...)`. It is invoked from the seed step (`src/engine/optimizer/v3/seed.py::run_seed_step`) **after** the run-main branch and `codeevolver.md` are in place and **before** the seed evaluation, so a broken environment aborts the run before burning time.

The validator runs the generic mounted evaluator (`src/engine/evaluator/mounted/evaluate.py`) on the **first 5 training rows** via the same `run_eval` path the architect loop uses — no DSPy coupling, no separate smoke script. It captures full traces for diagnostics.

It fails if system errors occur on >10% of rows (any row's program call or metric raised) OR if the primary metric is 0.0 across the entire sample. On failure it raises `SandboxValidationError` carrying:
  - Row indices that threw
  - First 5 failing examples (inputs, outputs, error summary, score)
  - The aggregate primary score and system error rate
  - The absolute path to the trace directory (`iteration_{N}_{suffix}/`, see `specs/engine3.md`)

The exception propagates out of the optimization; the launcher reports the structured failure.

## Security Architecture
- **Client-specific isolation:** each research run executes in its own container (`runner/`), with all capabilities dropped except those the runner re-adds, a non-root agent uid, and the held-out test set sequestered outside the agent-readable workspace.
- **Network egress control and whitelists:** every runner container reaches the network only through an egress-filtering gateway (`runner/squid.conf`) that allows the model-provider APIs and the benchmark services. Outbound traffic is logged and audited after the run (`orchestrator/egress_audit.py`).
- **Secrets management:** provider keys are passed into the container as environment variables; no secret is written into the client repo.
- **Monitoring and detection:** each run ends with a run-integrity audit (`src/engine/optimizer/meta/run_integrity.py`) and an egress audit.
- The optimizer has no database access; it reports through the job-scoped callback endpoints.
- Client environments never see the GitHub App private key, only a short-lived installation token scoped to the repository.
