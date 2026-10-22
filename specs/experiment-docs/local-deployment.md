# Local Deployment

CodeEvolver can run end-to-end on a single host (Mac mini or comparable server)
for clients whose code cannot leave their network (legal, healthcare) and for
academic researchers reproducing published results.

Local mode is selected by the `SANDBOX_MODE=local` env var, and it is the only
mode that launches optimizations. `SANDBOX_MODE=cloud` is the hosted portal API
(Clerk auth, Atlas), which serves imported run results and never launches a
run. The research fleet runs this same local mode inside one Docker container
per run (`runner/` + `orchestrator/`, see `experiment-fleet.md`). The relevant
seams live in `src/launchers/`, `src/local/`, `src/api/auth/local_user.py`,
`src/api/routes/local_repo.py`, and `src/local/local_history_mirror.py`.

> **Note on scope.** This document covers the local-execution machinery only.
> The HIPAA-tier work — Azure OpenAI instead of Claude and network-layer domain
> whitelist — is downstream of this setup and tracked separately.

## One-time host setup (macOS)

```bash
# 1. System runtime
brew install python@3.11 node deno
xcode-select --install                # git

# 2. Claude Code CLI (used by the coding/inspection agents)
npm install -g @anthropic-ai/claude-code

# Optional: OpenCode CLI for the open-source/research coding harness
npm install -g opencode-ai
opencode --version

# 3. MongoDB Community (single-node, no replica set)
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0

# 4. Clone CodeEvolver into the conventional location
git clone https://github.com/<your-fork>/CodeEvolver ~/.codeevolver/app
cd ~/.codeevolver/app
python3.11 -m venv env
source env/bin/activate
pip install -e ".[dev]"
```

## Environment variables

| Var | Required | Notes |
| --- | --- | --- |
| `SANDBOX_MODE` | yes | Set to `local`. Setting `APP_MODE=local` also works (it implies `SANDBOX_MODE=local`). |
| `ANTHROPIC_API_KEY` | yes | Used by the coding / inspection / outcome-reflection agents. |
| `LOCAL_MONGO_URI` | no | Override Mongo URI (default `mongodb://localhost:27017`). |
| `OPENAI_API_KEY` | conditional | Required when the **client program** uses an OpenAI-hosted LM (CodeEvolver has no `program_lm` knob — the client configures its own LM), and for the `ee_e2e` / `eval_trace` integration tests. |

A research-only deploy may want a `~/.codeevolver/.env` file with the keys
above; the optimizer's `_entry.py` will load it on subprocess spawn.

## Coding harness selection

`coding_harness` selects the agentic coding harness for the architect, coding
agent, and inspection agent roles in one run.

- `coding_harness=opencode`: OpenCode, the open-source/research harness.
- `coding_harness=claude`: Claude Code, the commercial/service harness.

A missing `coding_harness` resolves to `opencode`
(`src/engine/optimizer/v3/coding_harness_config.py::DEFAULT_CODING_HARNESS`).
Set `coding_harness=claude` explicitly to run Claude Code.

`coding_harness_model` is the OpenCode coding-model selector. It is separate
from `program_lm` and `reflection_lm`; those configure the optimizer's program
and reflection LMs, not the agentic coding harness. For OpenCode smoke tests,
use `openai/gpt-5.3-codex-spark` unless the installed OpenCode provider config
expects a different provider/model string. `coding_harness_model` is ignored
when `coding_harness=claude`.

Example local OpenCode optimizer config:

```json
{
  "coding_harness": "opencode",
  "coding_harness_model": "openai/gpt-5.3-codex-spark"
}
```

Example Claude service config:

```json
{
  "coding_harness": "claude",
  "coding_harness_model": null
}
```

OpenCode is an environment prerequisite when `coding_harness=opencode`. Verify
installation with `opencode --version`. The optimizer checks that the
`opencode` binary is on `PATH` before an OpenCode run and never installs it during a run.

## Launching the API

```bash
cd ~/.codeevolver/app
source env/bin/activate
export SANDBOX_MODE=local
export ANTHROPIC_API_KEY=...
uvicorn src.api.main:app --host 127.0.0.1 --port 8080
```

The bind to `127.0.0.1` is the security boundary: anyone with shell access on
the host can already read the workspace and venv, so an extra API key would
not raise the bar. Do not expose port 8080 to a network.

## Connecting a repo

A repo is registered with a single call. The repo stays on disk; CodeEvolver
runs in place and creates `codeevolver-*` branches in the operator's existing
checkout.

```bash
curl -X POST http://127.0.0.1:8080/local/connect-repo \
     -H 'Content-Type: application/json' \
     -d '{"local_path": "/path/to/my-project"}'
```

Concurrent runs on the same repo are blocked (which is why the fleet gives
each run its own container and its own clone rather than sharing a checkout): `LocalEnvironment.start()` takes
an exclusive `fcntl.flock` on `<local_path>/.codeevolver.lock`. The second
`/optimize` call against a live repo returns `409 Conflict` until the first
finishes. Operators who need true concurrency should register the same code
under separate clones.

## Where state lives

```
~/.codeevolver/
├── app/                                    # this codebase
├── memory/{job_id}/memory.md               # cross-iteration architect memory
├── memory/{job_id}/iteration_history_N.md  # per-iteration record + reflection
├── traces/{job_id}/iteration_*_*.jsonl     # OTel trace JSONL
├── history/{job_id}.jsonl                  # write-ahead iteration mirror
├── history/{job_id}.final.json             # terminal snapshot
├── logs/{job_id}.log                       # optimizer subprocess stdout
└── test-repos/{slug}                       # cached fixture clones for integration tests
```

Memory and traces are scoped per job: `src/engine/optimizer/v3/_entry.py`
sets `CODEEVOLVER_JOB_ID` from the optimization config, and
`src/engine/paths.py` appends `/{job_id}` to every traces/memory dir it
resolves. Successive runs do not share these directories, so one job's
`memory.md` cannot pollute the next architect's context. The workspace itself
is not swept between runs: `scripts/reset_repo.py` cleans a local checkout, and
the Docker runner avoids the question by starting every run from a fresh
container.

The local Mongo (`mongodb://localhost:27017/codeevolver_dev`) is the primary
store; the JSONL mirror is best-effort backup. If Mongo crashes,
`python scripts/export_history.py <job_id>` reconstructs the iteration
history into the same shape `GET /job/{job_id}/iterations` would return.

## Running integration tests locally

Once the prerequisites above are in place, `local_test_app.py` at the repo root runs the integration and functional tests (`python local_test_app.py --help` lists every subcommand; `tests/README.md` is the inventory):

```bash
# Full optimizer loop per arm (ANTHROPIC_API_KEY + OPENAI_API_KEY required)
python local_test_app.py ee_e2e
python local_test_app.py freeform_e2e
python local_test_app.py pareto_e2e
python local_test_app.py simple_loop_e2e

# Seed-only run; checks OTel pipeline + judge spans land in /traces
python local_test_app.py eval_trace --max-rows 5

# Synthetic-workspace tests (ANTHROPIC_API_KEY only)
python local_test_app.py inspect_traces
python local_test_app.py analysis_agent
```

The first `ee_e2e` / `eval_trace` run clones the fixture repo into `~/.codeevolver/test-repos/{slug}`; subsequent runs reuse the cached clone. The synthetic tests clear `/memory` and `/traces` before seeding so a stale file from a prior run cannot mask a failure.

## Opt-in coding-harness smokes

Normal unit tests stub OpenCode and do not require a live `opencode` binary or
provider credentials. Live harness smokes are opt-in because they launch real
coding agents and require provider auth/API key setup.

Minimal OpenCode coding-agent smoke:

```bash
cd ~/.codeevolver/app
source env/bin/activate
export CODEEVOLVER_LIVE_OPENCODE_SMOKE=1
env/bin/pytest tests/functional/test_live_coding_harness_smoke.py::test_live_opencode_coding_agent_smoke -q
```

The OpenCode smoke uses `coding_harness=opencode` and
`coding_harness_model="openai/gpt-5.3-codex-spark"` by default. Override the
model only if OpenCode requires another exact provider/model string:

```bash
export CODEEVOLVER_LIVE_OPENCODE_MODEL=openai/gpt-5.3-codex-spark
```

For a manual OpenCode smoke against a disposable checkout:

```bash
cd ~/.codeevolver/app
source env/bin/activate
export CODEEVOLVER_OPENCODE_SMOKE_WORKSPACE=/path/to/disposable/repo
python - <<'PY'
import os
from src.engine.services.subprocess_service import run_coding_agent

workspace = os.environ["CODEEVOLVER_OPENCODE_SMOKE_WORKSPACE"]
result = run_coding_agent(
    workspace,
    change_request="Inspect the repository, make one harmless documentation-only change, then stop.",
    commit_changes=False,
    coding_harness="opencode",
    coding_harness_model="openai/gpt-5.3-codex-spark",
)
print(result)
PY
```

Minimal Claude smoke:

```bash
cd ~/.codeevolver/app
source env/bin/activate
export CODEEVOLVER_LIVE_CLAUDE_SMOKE=1
env/bin/pytest tests/functional/test_live_coding_harness_smoke.py::test_live_claude_coding_agent_smoke -q
```

The Claude smoke uses `coding_harness=claude`; any supplied
`coding_harness_model` is ignored when `coding_harness=claude`.

## Coding harness troubleshooting

- missing `opencode` binary: run `opencode --version` in the same shell that
  launches CodeEvolver. If it is missing, install OpenCode in the environment
  before selecting `coding_harness=opencode`; CodeEvolver will not install it.
- provider auth/API key setup: OpenCode provider credentials must match the
  provider/model string, such as `openai/gpt-5.3-codex-spark`. Claude mode
  needs the Claude Code CLI and its normal auth setup. Optimizer
  `program_lm`/`reflection_lm` API keys are separate from harness credentials.
- permissions/autonomous mode: optimizer runs are non-interactive. OpenCode
  mode uses autonomous permission configuration and `opencode run` permission
  bypass behavior; Claude mode uses the existing bypass-permissions/user-proxy
  controls. If a harness asks a human question, treat it as a config bug.
- session/resume behavior: architect sessions retain provider session IDs for
  watchdog follow-ups. Coding and inspection smokes are run-to-completion
  checks; a missing session ID there is not automatically a failure.
- MCP/tool registration issues: architect tool failures usually mean the
  selected harness could not register or reach the CodeEvolver MCP/tool bridge.
  Verify the selected `coding_harness`, inspect harness stderr in the optimizer
  log, and confirm staged `.opencode/skills` or `.claude/skills` directories
  were created in the target workspace.

## Optimizer environment contract

The local launcher (`src/launchers/local_launcher.py`) spawns the method's
entry script (`src/engine/optimizer/v3/<method>/optimization_entry.py`, or
`src/engine/optimizer/simple_loop/optimization_entry.py`) as a subprocess. The
engine root is auto-detected from `src/engine/paths.py`'s own location
(`paths.engine_root()`) and prepended to `PYTHONPATH`; there is no env var for
it. The remaining contract:

| Var | Default | Effect |
| --- | --- | --- |
| `SANDBOX_MODE` | inherited | `local` selects the `~/.codeevolver/...` state roots. |
| `REPO_VENV_DIR` | `env` | Client venv directory under the workspace. |
| `REPO_VENV_BIN` | `bin` | Subdir of the venv with executables. |
| `CODEEVOLVER_JOB_ID` | set by `_entry.py` | Scopes traces/memory/history dirs per job. |
| `ANTHROPIC_API_KEY` | passthrough | Used by the Claude harness and the Claude Agent SDK. |

The optimizer reads these in `src/engine/paths.py` (`repo_venv_dirname`,
`repo_venv_bin`, `engine_root`) and `src/engine/services/subprocess_service.py`
(`_build_env`).

## What's deliberately missing in local mode

- No Docker around the optimizer subprocesses. The host is treated as
  operator-trusted infrastructure with no sensitive data of its own.
  (Research runs on the fleet do run containerized, one container per run —
  see `specs/experiment-docs/experiment-fleet.md`. That path reuses this same
  local mode verbatim inside the container; it does not fork the local
  deployment.)
- No egress proxy / domain whitelist. Coming as a separate HIPAA-tier deliverable.
  The fleet ships an SNI-allowlist gateway (`runner/squid.conf`) for the
  research contamination guarantee, which is a starting point for that work.
- No Clerk, no API key, no multi-tenant auth. Single operator on the host.
- No remote git I/O. Local commits and branches only.
  `git_fetch`, `refresh_github_token`, and `git_push_authenticated` all
  short-circuit when `SANDBOX_MODE=local`, so the seed step's run-main
  branch and every accepted child branch stay in the operator's repo and
  the optimizer never touches `origin`. Operators inspect runs locally
  via `git branch` / `git log` instead of a GitHub UI. The branch-name
  pre-push guard is still installed in case the operator wires up an
  origin and pushes by hand.
- Traces and memory live under the operator's home, not at root-level
  paths — the optimizer's `src/engine/paths.py` resolves them to
  `~/.codeevolver/{traces,memory}/{job_id}` under
  `SANDBOX_MODE=local`. The `{job_id}` suffix
  is appended by `paths._job_subdir()` whenever `CODEEVOLVER_JOB_ID` is
  set (the optimizer entry script sets it from the config). Concurrent
  local runs each get their own subdirectory, and the repo lock still
  prevents two runs from racing on the same workspace.
