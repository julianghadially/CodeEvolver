# Experiment Fleet

Runs an experiment matrix (arms × benchmarks × seeds) on disposable cloud
containers: one container per run, one arm per host, every run starting from an
immutable image in a clean environment. The main matrix
(`orchestrator/matrix.main.yaml`) is 5 arms × 5 benchmarks × seeds 1–3.

**You never configure a server.** Everything below runs from your own machine
(the *controller*); it drives the droplets over plain SSH
(`docker --context <host>`). A worker needs nothing beyond Docker and
`orchestrator/provision.sh`.

The quick start is `specs/experiment-docs/README.md`; a single-machine trial is
under "Local dry run" below.

---

## Matrices and the launcher

One matrix file per experiment, one command per matrix:

| Matrix | What it runs |
|---|---|
| `orchestrator/matrix.main.yaml` | Main experiment: `ee`, `freeform`, `pareto_evolution`, `simple_loop`, `asa` × hover, phantomwiki, ragqa, arcagi2, paperreview × seeds 1–3. One droplet per arm. |
| `orchestrator/matrix.evo.yaml` | `pareto_evolution` with the GEPA union-cover pruning rule, seeds 1–3, one host per seed (`host_seeds`), `run_suffix: -redo`, `image_tag: evo-redo`. |
| `orchestrator/matrix.abl.yaml` | Weaker-coder ablation: `pareto_evolution` vs `freeform` with a different coding/architect model (set by the matrix's `launch.set`), hover + phantomwiki. |
| `orchestrator/matrix.po.yaml` | Prompt-only ablation (see "Prompt-only ablation" below). |
| `orchestrator/matrix.local.yaml` | One run on this machine, for proving the runner + controller path (see "Local dry run"). |

```bash
export REGISTRY=registry.digitalocean.com/<your-registry>
CE_MATRIX=$PWD/orchestrator/matrix.main.yaml \
  nohup ./orchestrator/launch.sh > ~/.codeevolver/logs/launch.main.log 2>&1 &
tail -f ~/.codeevolver/logs/launch.main.log
```

`orchestrator/launch.sh` runs the matrix end to end: `fleet.py up` (create the
droplets), `provision.sh` per host, optionally `ce build`, a digest-lock check,
`ce start` (blocks until every run is collected), and `fleet.py down`.

| Input | Meaning |
|---|---|
| `CE_MATRIX` | Required. The matrix file. |
| `CE_STATE` | Controller state file; default `orchestrator/state.<fleet tag>.json`. One controller per state file — the launcher refuses a state file that already records runs, or one a live controller is using. |
| `CE_IMAGE_LOCK` | Digest pins; default `orchestrator/images.lock.json` for `image_tag: latest`, else `orchestrator/images.<image_tag>.lock.json`. The public release ships no lock files (they pin our private registry), so the first launch needs `CE_BUILD=1` to build, push and pin images in your own `CE_REGISTRY`. |
| `CE_BENCHMARKS` | Benchmarks to provision; default every benchmark named in the matrix's `host_benchmarks`, else all five main benchmarks. |
| `CE_REPO_ROOT` | Parent folder of the client repos (default `~/Documents/code`). |
| `CE_BUILD=1` | Rebuild, push and re-pin every image on the build host between provisioning and dispatch. `CE_BUILD_HOST` picks the host (default: the matrix's first host). |
| `CE_QUARTER=1` | Quarter budgets (`max_metric_calls / 4` per benchmark via `CE_SET_<BENCH>`), for a load test. |

Optional `launch:` block in a matrix:

- `set: "<k=v;k=v>"` — default `CE_SET` for every dispatch (e.g. the coder
  model of a coder ablation). An explicit `CE_SET` wins.
- `prompt_only: true` — sources `orchestrator/prompt_only/launch_hooks.sh`,
  which stages the prompt-only seed bundles, kernel fence and per-benchmark
  instructions before dispatch.

The controller machine must stay awake (`caffeinate` is used when present). If
it sleeps, the droplets keep running and `ce status` re-attaches on wake; only
dispatch and collection pause.

### Matrix keys

| Key | Meaning |
|---|---|
| `registry` | Optional. Images are named `<registry>/codeevolver-runner:<bench>-<image_tag>`. Absent means `$REGISTRY` (shared with `build.py`, `provision.sh` and `compose.host.yml`); `matrix.local.yaml` pins `local`. |
| `image_tag` | Image tag, pinned by digest in the matching lock file. |
| `fleet:` | DigitalOcean droplet spec for `fleet.py`: `tag`, `region`, `size`, `image`, `ssh_key` (key name in your account), `project`, `price_hourly`. |
| `hosts` / `assignment` | Droplet names, and the arm (or list of arms) each runs. The two must name the same machines; drift fails at load. |
| `arms`, `seeds` | The matrix axes. |
| `host_benchmarks` | Optional `{host: [bench, ...]}`; restricts a host to those benchmarks (and decides which hosts fetch the ragqa retriever). |
| `host_seeds` | Optional `{host: [seed, ...]}`; restricts a host to those seeds. |
| `run_suffix` | Optional, `-[a-z0-9]+`; appended to every run id (container, `/data/runs`, `$CE_RUNS_DIR`, `refs/runs/<id>`) so a re-run never collides with an earlier run's directory. |
| `slots_per_host` | Concurrency cap per host, not a benchmark count. |
| `bench_concurrency` | Optional (default 1); how many runs of one benchmark a host may have in flight. |
| `defaults` | `poll_seconds`, `rsync_seconds`, `docker_args` (container caps and network), optional `runtime: runsc`. |
| `env_passthrough` | The only controller environment variables a run receives. |

**Arms are sharded across hosts, one arm per host for the whole matrix.** No
two arms share a machine — which matters most for `ee` and `asa`, since asa is
ee plus enforced population-tree traversal, so the treatment effect is the delta
between them and they must not contend for the same CPU, RAM, disk and ragqa
retriever. Tradeoff: a host-level anomaly (noisy neighbour, disk pressure, a
degraded NIC) can confound with the arm on that host. The mitigation is
operational: identical droplets, plus `ce status` and the stall warning to
surface a sick host.

---

## Your benchmark subset

Which benchmarks a researcher provisions and runs comes from the environment,
resolved against the catalog `orchestrator/benchmarks.yaml`:

```bash
export CE_BENCHMARKS=hover,phantomwiki,ragqa       # your subset
export CE_REPO_ROOT=~/Documents/code               # where your client repos live
export CE_REPO_PHANTOMWIKI=/mnt/big/phantom-wiki   # optional per-benchmark override

python3 orchestrator/fleet_config.py               # show what that resolves to
```

`launch.sh` defaults `CE_BENCHMARKS` from the matrix. Outside the launcher
(`ce`, `build.py`, `provision.sh` run by hand) there is deliberately **no
default** — silently inheriting someone else's set would burn days of provider
spend before anyone noticed. Unknown names fail with the catalog listed.

Adding a benchmark is one catalog entry plus an
`experiments/<experiment_dir>/input.py`. The fleet slug is usually **not** the
directory name (`phantomwiki` → `experiments/phantom-wiki`, `ragqa` →
`experiments/RAGQAArenaTech`); the directory is what `experiments.run --program`
takes.

**`seed_branch`.** `build.py bundles` archives each benchmark's `seed_branch`
(default `main`). `arcagi2` and `paperreview` seed from
`seed-program-deepseek-v4flash`. `python3 orchestrator/fleet_config.py` prints
a `[seed: <branch>]` marker for any benchmark not on `main`. A `--branch` flag
overrides it for every benchmark in that build — use it only for a one-off.
A plain clone is enough: the seed branch resolves as `origin/<branch>` when no
local branch exists. If a checkout has been sitting around, `git fetch --all`
first — the build bundles whatever the checkout knows.

**Several benchmarks may share one `repo_dirname`.** `hover` and `ragqa` both
resolve to `LangProBe-CodeEvolver`. `bundle_name` and `image()` are keyed by
slug, so the two still seed and execute from independent bundles and images,
and `CE_REPO_<SLUG>` redirects each separately. Consequence: each run's
workspace also carries the other benchmark's data. Only the run's own
`testset_path` is relocated to `/opt/holdout`, so a sibling benchmark's test set
stays readable — accepted, since a hover run cannot profit from RAGQA's answers,
and the clean-slate inspector's prompt is told this is normal.

Your subset determines what each host pulls and runs:

- **Images.** `provision.sh` pulls only your benchmarks' images.
- **The RAGQA retriever sidecar** (~8 GB resident) starts **only** on hosts whose
  subset includes `ragqa`, via the compose `ragqa` profile. It is shared by the
  host's runs, not started per run.
- **Sizing.** Without the retriever a host needs far less RAM — 32 GB is ample.
  Only ragqa hosts need the 64 GB box.

---

## Shape

```
controller (your machine)    one droplet per arm
┌──────────────────┐         ┌───────────────────────────────────────────┐
│ ce start         │ docker  │ squid gateway (SNI allowlist)             │
│ ce status        │─context▶│   └ ce-internal (no default route)        │
│ ce cancel        │ over SSH│       ├ ce-<arm>-hover-s1     (container) │
│                  │         │       ├ ce-<arm>-phantomwiki-s1           │
│ ~/.codeevolver/  │◀─rsync──│       ├ ce-<arm>-ragqa-s1                 │
│ $CE_RUNS_DIR/    │         │       └ ragqa-retriever (sidecar)         │
└──────────────────┘         │  /data/runs/<run_id>  (only writable mount)│
                             └───────────────────────────────────────────┘
```

**One container == one run.** The container is the isolation boundary, not the
VM: every run starts from an immutable image, so there is no state to reset
between runs and `scripts/reset_repo.py` is not part of the loop. No
CodeEvolver port is exposed to any network.

---

## Daily use

```bash
cd orchestrator && export PATH="$PWD:$PATH"
export CE_MATRIX=$PWD/matrix.main.yaml CE_STATE=$PWD/state.ce-main.json

ce start                      # schedule the matrix (launch.sh does this for you)
ce start --bench hover        # or just one benchmark
ce start --arm ee --arm freeform
ce start --dry-run --once     # show what would dispatch, change nothing
ce start --drain              # finish in-flight runs, dispatch nothing new

ce status                     # every run: host, state, iteration, evals, elapsed
ce logs ee-hover-s1 -f        # tail a container
ce pull ee-hover-s1           # force an artifact sync now
ce cancel ee-hover-s1         # cooperative cancel (see below)
ce cancel --now ee-hover-s1   # docker kill
ce recycle --host <host>      # wipe a host (containers, images, run dirs)
ce build [--host <host>]      # rebuild, push, pin, pull (see Rebuilding)
```

`ce` is `orchestrator/controller.py`. It reads `CE_MATRIX` (default
`matrix.main.yaml`) and `CE_STATE`; export both when driving a matrix by hand.

`ce start` is safe to kill and restart. It reconciles against `docker ps` on
every pass, so it re-attaches to live containers rather than dispatching them
twice; the state file is only a cache.

### Polling and collection timing

The scheduler wakes every `defaults.poll_seconds` (15 minutes in the experiment
matrices). Each pass, in order:

1. **Collects any run that finished** — rsync the artifact directory, merge the
   job trees, fetch the branch bundle — and only *then* removes the container.
   Collection is per run at completion, never batched to the end of the matrix.
2. Syncs still-running runs incrementally (`rsync_seconds`, 10 minutes), so
   losing a host mid-run costs minutes of traces, not the run.
3. Dispatches the next queued run into the freed slot.

A slow poll only costs idle host time between a run finishing and its successor
starting, and avoids thousands of SSH probes per host per day. `provision.sh`
enables SSH `ControlMaster`, so polls reuse one connection. Failed polls are
non-fatal: an unreachable host is treated as empty for that pass and retried on
the next. `ce status` reconciles on demand, so it is always current.

Only a terminal cached status may stand in for the container's own state;
anything else means the cache holds the dispatch marker, and the container's
state wins. That is what lets an exited container be seen as finished,
collected, and its slot refilled.

### Cancellation is slow on purpose

`ce cancel` drops a `CANCEL` sentinel into the run's artifact directory; the
runner relays it to the API, and the optimizer treats cancellation as budget
exhaustion, so it finishes the current iteration and then runs finalization —
the seed and final test-set evals. **That can take hours, and the result is a
fully scored run.** No `docker stop` grace timer is used, so it cannot race
those evals.

`ce cancel --now` kills immediately and loses the in-flight iteration and the
finalization scores. Use it only when you do not want the result.

### Post-run agents

The optimizer marks a job complete and only then runs meta-reflection and the
closing run-integrity inspector, so an advisory agent can never fail a run that
already succeeded. `runner/entrypoint.py::await_optimizer_exit` therefore blocks
between "the API reports a terminal status" and the exports until the optimizer
process is gone:

- **Identity, not liveness.** It reads `/proc/<pid>/cmdline` and looks for
  `optimization_entry` rather than sending signal 0, since one container
  recycles pids freely.
- **Bounded.** `CE_POST_RUN_TIMEOUT` (default 1800 s); past it the runner
  exports what exists and shuts down, so an advisory agent never holds a host
  slot indefinitely.
- **Not under SIGTERM.** `docker stop` ends in SIGKILL, so there the runner
  exports immediately. A CANCEL-sentinel cancel has no timer and still waits.
- **Fail-open.** No pidfile means no wait.

Waiting also orders the exports correctly: the inspector flushes its verdict
onto the last iteration record, so `export_candidates` picks it up. Tests:
`tests/unit/test_experiment_fleet.py`.

---

## Where results land

Inside the container `HOME=/artifacts`, so the engine's state root resolves to
`/artifacts/.codeevolver/` (e.g. `memory_dir()` is
`/artifacts/.codeevolver/memory/{job_id}`). That directory **is** the host bind
mount, so artifacts stream out live and nothing is lost if a container dies. It
is a different directory from `/workspace/.codeevolver/`, which holds the
repo-local `optimization_config.json` and is not collected. Skills are rendered
at staging time from `paths.memory_dir()` / `history_dir()` / `traces_dir()`, so
agents are handed correct absolute paths automatically.

The controller rsyncs every `rsync_seconds` and once at exit, into two places:

| Path | Contents |
|---|---|
| `~/.codeevolver/{memory,architect_traces,history,logs,runtime_tests}/{job_id}/` | Merged into the controller's tree. Paths are scoped by `job_id`, so merging many runs is collision-free and `scripts/query_iteration_history.py`, `export_history.py`, `view.py` work unchanged. `logs/` also holds `<job_id>_clean_slate_report.md`. |
| `$CE_RUNS_DIR/<run_id>/` (default `~/ce-runs`) | `status.json`, `settings.json`, `egress.json`, `integrity.json`, `candidates.jsonl`, `history.json`, `repo.bundle`, `workspace.tar.gz` + `workspace_manifest.json`, squid access log. |
| `$CE_RUNS_DIR/repos/<bench>.git` | Bare repo holding every run's branches under `refs/runs/<run_id>/*`, plus `refs/jobs/<job_id>/heads/*` and `refs/jobs/<job_id>/codeevolver/*` (per-iteration close anchors and every evaluated tree). |
| `$CE_RUNS_DIR/gateway-<host>.access.log` | The host's squid access log, pulled on every collect sweep (input to the egress audit). |

**`traces/` is deliberately not collected.** It records every evaluated row —
each LM input and output — and is by far the largest artifact class. The
`inspect_traces` tool and meta-reflection consume it inside the container. It is
excluded at both layers in `orchestrator/collect.py` (`JOB_TREES` and
`RSYNC_EXCLUDES`). Per-row traces are therefore gone once a host is recycled or
its droplet destroyed. `architect_traces/` — what the architect did each
session — is kept.

### The final workspace snapshot

Commits go through `git add -A` without `--force`, so anything a run wrote to a
gitignored path is never in `repo.bundle`. `export_workspace()` closes that at
the end of each run: it checks out the final primary, tars `/workspace` to
`workspace.tar.gz`, and writes `workspace_manifest.json` with the intended vs
actual branch, the HEAD sha, the archive size, and every path in the tar that
git does not track. Excluded: `env/`/`.venv/` (rebuildable from
`requirements.txt`), `.git/` (`repo.bundle` has every object), and pure caches.
Non-fatal — it is a backup. This is the final state only; per-candidate analysis
uses `repo.bundle` + `candidates.jsonl` and assumes the program is committed.

### Loading branches for per-candidate analysis

All runs of a benchmark descend from the same squashed seed commit, so the bare
repo dedupes objects across runs. Use worktrees, not clones:

```bash
git -C ~/ce-runs/repos/hover.git for-each-ref --format='%(refname)' refs/runs
git -C ~/ce-runs/repos/hover.git worktree add /tmp/cand-1 <sha>
```

`candidates.jsonl` enumerates every closed iteration (`branch`, `sha`,
`subsample_score`, `improvement_mode`, `is_primary`, `constraint_violation`,
`test_scores`, …). The analysis harness is `scripts/best_of_population/`
(`scripts/best_of_population/README.md`); it keeps its own bare repo per
benchmark and fetches straight from `repo.bundle`.

`scripts/results_tables.py` builds the result tables from `$CE_RUNS_DIR`. A
`run_suffix` is parsed as a label; its explicit replacement table
(`ARM_REPLACEMENTS`, `CELL_REPLACEMENTS`) decides which labelled directory a
cell reads, a replaced cell never falls back to the unlabelled original, and
the table is written to `replacements.csv`.

---

## Research mode and the settings banner

The runner image declares `CE_RESEARCH_MODE=true`, and
`entrypoint.py::research_mode_defaults` pins the research controls for every
run: `enable_clean_slate_inspector`, `enable_run_integrity_agent`,
`enable_meta_reflection`, plus `test_samples=3` / `seed_test_samples=1`
(finalization draws the held-out set three times for the final program and
once for the seed; `status.json` and the job record report `seed_test_scores` /
`final_test_scores` beside the means). They are pinned explicitly rather than
inherited from the API schema defaults. An explicit `CE_SET` value always wins,
so a control can be turned off for a deliberate experiment, and the banner says
so. `CE_RESEARCH_MODE` arms controls only; the engine never reads it.

The eval-row budget is hard-enforced by `run_eval` on every run (see
`specs/engine3.md`): once `total_num_evals >= max_eval_rows`, training evals are
refused and one final comparison-set run remains. Eval rows are the controlled
variable every arm is compared on.

Every run prints its control surface before submitting, to stdout (top of
`ce logs`) and to `/artifacts/settings.json` (collected, so "which runs had
clean-slate on?" is one `jq` across `$CE_RUNS_DIR`): run identity and research
mode; the controls that can be off without erroring (the three above, the
mid-run integrity trigger with its resolved budget or `disabled`, the holdout
testset path and agent user, the egress gateway); the models; the budget; and
every `CE_*` / `CODEEVOLVER_*` variable. **Provider keys are fingerprinted, never
printed** — `present (sha256:xxxxxxxx)` or `MISSING` — which still answers
whether every host got the same key.

### What a run receives from the controller

`dispatch()` forwards only what the matrix's `env_passthrough` names. Provider
keys must be exported in the controller's shell — they are the only
credentials a runner holds: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`,
`DEEPINFRA_API_KEY`, `GMI_API_KEY` (or `GMI_CLOUD_API_KEY`; the arcagi2 and
paperreview solvers read either), `DEEPSEEK_API_KEY`. Optional switches:

- `CE_SET` — per-run config overrides, `key=value;key=value`; `CE_SET_<BENCH>`
  scopes overrides to one benchmark. Beats every other source, including the
  research defaults.
- `CE_RESEARCH_MODE` — set true by the image; exists so an operator can turn it
  off for a one-off.
- `CE_MAX_METRIC_CALLS` — fallback budget for the mid-run integrity trigger
  when the submitted config declares none.

Optional entries (`controller.py::OPTIONAL_PASSTHROUGH`) do not warn when unset;
missing provider keys warn on every dispatch.

All five main solvers pin DeepSeek-V4-Flash and reach it through a ranked
provider chain (gmi, deepseek, deepinfra), diverting a failed call to the next
provider and skipping a provider once its breaker opens (see
`specs/engine3.md`). Consequences:

- **All three task-provider keys matter on every benchmark.** A missing key
  never fails a run — it moves every call one step down the order — so the
  controller warns on every dispatch that lacks one.
- **`DEFAULT_PROVIDER`** is `deepseek` on arcagi2/hover/ragqa/phantomwiki
  (paperreview starts on GMI). Override per run with `LM_PROVIDER` /
  `LM_FALLBACK`.
- **Egress:** `api.gmi-serving.com`, `api.deepinfra.com` and `api.deepseek.com`
  are in `runner/squid.conf` and in `runner/preflight.py`'s global
  `ALLOWED_HOSTS`; `BENCH_ALLOWED_HOSTS` adds per-benchmark hosts (alfworld's
  `openrouter.ai`). An unreachable allowed host aborts at startup, so a host
  whose `/opt/ce/squid.conf` predates an ACL change must be re-provisioned.
- Divert and breaker lines land in each eval's `eval_stderr.log`.

---

## Contamination controls

Independent mechanisms, all recorded per run.

### 0. OS-enforced holdout fence (kernel-level, Docker runner only)

The base image creates an unprivileged `ce-agent` user and sets
`CODEEVOLVER_AGENT_USER=ce-agent`, the single activation switch
(`src/engine/paths.py::agent_user()` is None everywhere else, so local runs are
unaffected). The engine spawns every agent subprocess (architect, coding,
inspection; both harnesses) as `ce-agent`; the optimizer driver and evaluator
stay root.

`entrypoint.py::sequester_holdout` moves the configured `testset_path` out of
the workspace to root-owned `/opt/holdout/<name>` (dir 700, file 600), amends it
out of the single seed commit, re-points the `seed` branch, expires reflogs,
gcs, and *verifies the blob is gone* (abort on failure) — file permissions alone
cannot fence a git-tracked testset (`git show seed:<path>`). The run is
submitted with `--set testset_path=/opt/holdout/<name>`, so the root driver
still runs the finalization test evals.

`main()` sets `umask(0)` and a `chmod -R a+rwX` sweep over `/workspace` and
`/artifacts` so `ce-agent` can work everywhere except the holdout.
`trust_workspace_for_agent()` writes `safe.directory /workspace` into the
root-owned system git config, since git's ownership check compares the repo's
owner (root) with the calling uid. The preflight asserts `setuid(ce-agent)`
works (needs `--cap-add=SETUID/SETGID` after `--cap-drop=ALL`) and that
`git -C /workspace rev-parse HEAD` succeeds as `ce-agent` (`agent_git` in
`egress.json`), and aborts before any spend if not.

Privileged exception: the run-integrity inspector runs as root
(`run_as_agent_user=False`), because reading the testset is how it detects
hard-coding. Deny-path rules remain as defense-in-depth.

Accepted residuals: the evaluator runs as root, so AI-written program code could
read the holdout during a training eval (covered by the constraint gate, the
integrity inspector and the egress audit); `additional_deny_paths` extras are
not relocated and keep the name fence only; `/artifacts` stays agent-writable,
so `integrity.json` is tamperable post hoc by a malicious agent. Tests:
`tests/unit/test_agent_user_fence.py`.

### 1. Squashed seed repos

`make bundles` (`build.py bundles`) rebuilds each client repo as a single orphan
commit: a test set once tracked in history is unreachable as a blob, and prior
runs' `codeevolver-*` branches are dropped. Deny-path hooks match file paths and
command text, not blobs, so this is what closes `git cat-file -p <blob>`. The
re-add into the squash uses `git add -A --force`, so the bundle carries exactly
the tracked tree regardless of the client's `.gitignore`. The entrypoint
refuses to start if history depth ≠ 1.

### 2. Egress allowlist

Runners sit on an internal network (`ce-host_ce-internal`) whose only path off
the host is a squid gateway that filters by TLS SNI (peek + splice, no MITM).
SNI filtering rather than `HTTP_PROXY` is deliberate — Node/undici, which the
OpenCode harness speaks HTTP through, ignores proxy env vars. The pieces, all of
which are required together (`runner/compose.host.yml`,
`runner/Dockerfile.gateway`, `runner/squid.conf`):

- a pinned subnet and gateway address;
- a one-shot `netrules` service inserting `-i ce-int0 -o ce-int0 -j ACCEPT` into
  `DOCKER-USER` (traversed before `DOCKER-INTERNAL`, which drops by destination)
  so same-bridge traffic to the gateway survives;
- `--cap-add=NET_ADMIN` plus `entrypoint.configure_egress_route()` pointing the
  container's default route at the gateway;
- `--dns=172.31.66.2`, a dnsmasq inside the gateway;
- a gateway image built on the host from `Dockerfile.gateway` (ubuntu +
  `squid-openssl` + iptables + dnsmasq; SslBump needs OpenSSL);
- enforcement at `ssl_bump peek step1 / splice allowed_sni / terminate all`, so
  allowed hosts complete TLS against the origin's certificate and denied ones
  are cut off before a session exists.

`models.dev` is allowlisted: OpenCode fetches its model catalogue at session
start and every architect session fails without it (a static catalogue of
model ids; no benchmark data can arrive through it). github.com is **not**
allowlisted, and must not be: the "no agent contacted github.com" guarantee
depends on it. OpenCode's `grep` / `glob` / `skill` tools therefore need a local
`rg`; `Dockerfile.base` installs apt `ripgrep`, and
`runner/preflight.py::_check_ripgrep` aborts the run before any spend when it is
missing. The squid access log ships as a run artifact.

### 3. Egress canary

Before any provider spend, the entrypoint asserts github.com /
raw.githubusercontent.com / huggingface.co are unreachable and the providers
are reachable, from both Python and Node, with a real SNI-bearing TLS handshake
and certificate verification (a bare TCP connect would always succeed under
transparent interception). The result is written to `egress.json`; **a fence
that fails open aborts the run.**

### 4. Clean-slate inspector

Runs at seed, before any eval spend, and verifies the environment is genuinely
fresh: no old solutions in git history, every holdout/test file covered by the
fences, no prior-agent leftovers. `contaminated` aborts the run; `uncertain`
fails open with findings logged.

### 5. Run-integrity inspector (agentic, advisory)

Once a run passes `CE_INTEGRITY_AT` (default 20%) of its eval budget, the
entrypoint launches `scripts/inspect_run_integrity.py` in the background: a
read-only auditor that classifies cheating (test-set leakage, hard-coded
answers: `no_cheating` | `non_impactful_cheating` | `impactful_cheating`) and the
% of functioning iterations. The verdict lands in `/artifacts/integrity.json`
(folded into `status.json`); the collect sweep reads it at most once per run and
logs `INTEGRITY WARNING <run_id>` — on stdout and in
`~/.codeevolver/logs/fleet.log` (override `CE_FLEET_LOG`) — when the category
isn't clean or `integrity_pct < 70`. Warning only; it never cancels a run. The
budget comes from `max_metric_calls` in the submitted config, else
`CE_MAX_METRIC_CALLS`; with neither, the check disables itself.
`CE_INTEGRITY_AT=off` disables it. The same inspector runs again at run close
inside the optimizer (`enable_run_integrity_agent`), stamping its verdict on
the last iteration record — see `specs/engine3.md`.

### 6. Egress audit (mechanical, advisory)

Policy: every task-LM call is budgeted, i.e. issued by the evaluator that
`run_eval` spawns; re-scoring cached outputs offline is fine.
`orchestrator/egress_audit.py` checks this from the gateway log: it brackets
every eval from the job log (`run_eval start (... timeout=T)` / `run_eval done`;
a start with no done closes at `start + T`), plus `seed-validate` and the
finalization window, and counts task-provider connections
(`api.gmi-serving.com`, `api.deepseek.com`, `openrouter.ai`) from the run's
container IP outside those windows. The harness provider, Anthropic, OpenAI and
the retrievers are not task providers. The controller records `container_ip` at
dispatch (addresses are recycled, so the join keys on ip + run window), pulls
each host's `/data/gateway-logs/access.log` on every collect sweep, runs the
audit per running run and once more at finish, stores the summary under the
state entry's `egress_audit`, and logs `EGRESS WARNING <run_id>` when violations
appear. Standalone: `python orchestrator/egress_audit.py --run-id <id> --host
<host> [--ip <ip>]` (the IP is inferred from the log when omitted). Tests:
`tests/unit/test_egress_audit.py`.

### Isolation between runs

The only writable host path is `/data/runs/<run_id>` — the run's own artifact
directory, never re-mounted into another run — so an agent has nowhere to leave
a note a later run could read. Containers run with `--cap-drop=ALL` (plus
`SETUID`/`SETGID`/`FOWNER` for the agent-user fence and `NET_ADMIN` for the
egress route), `--security-opt=no-new-privileges`, no `docker.sock`, and no host
PID/network namespace. The root filesystem is writable (the entrypoint writes a
clone plus a multi-GB staged venv to `/workspace`, and `/opt/holdout`), but it
is per-container and discarded. `provision.sh` installs gVisor; set
`defaults.runtime: runsc` in the matrix for a syscall-level boundary.
`ce recycle --host <host>` clears a host (containers, images, collected run
dirs); for a genuinely fresh machine, destroy the droplet and provision a new
one.

---

## Provisioning and images

### Droplets: `fleet.py`

`fleet.py up` creates whatever the matrix's `hosts` lack (region, size, image,
SSH key, project and grouping tag from the matrix's `fleet:` block), assigns
them to the project, waits for SSH, and writes `orchestrator/fleet.<tag>.json`
with the IPs (`--hosts` limits it to some hosts). `fleet.py down` destroys
them, but refuses while any matrix run is non-terminal, uncollected
(`$CE_RUNS_DIR/<run_id>/status.json` absent), or still has a container running
— `/data/runs` dies with the droplet. `fleet.py status` prints uptime and cost.
It is plain HTTPS against the DigitalOcean API with `DIGITALOCEAN_API_KEY`; no
doctl, no SDK. The Memory-Optimized 8 vCPU / 64 GB size (`m-8vcpu-64gb`) is
offered in nyc1/sfo2/sfo3 but not nyc2/nyc3; `up` checks before creating.
Droplet names double as docker context names and SSH targets, so keep them
lowercase.

Sizing: 64 GB is for the RAGQA retriever's ~8 GB resident index beside several
runs; ARC-AGI-2's streaming evals set the per-container cap (`--memory=14g` in
the matrices). Compute is small next to provider spend, so size up rather than
tune.

### Hosts: `provision.sh`

```bash
REGISTRY=registry.digitalocean.com/<your-registry> \
  CE_BENCHMARKS=hover,phantomwiki,ragqa TAG=latest \
  ./orchestrator/provision.sh <host> <ip>
```

The entire server-configuration burden: it creates the docker context and SSH
multiplexing, logs the host into the registry, ships the gateway files
(`compose.host.yml`, `Dockerfile.gateway`, `squid.conf`) to `/opt/ce`, starts
the gateway (and the retriever profile when `ragqa` is in the subset),
downloads the ragqa index and corpus into `/data/ragqa` on ragqa hosts, and
pulls the subset's images at `TAG`. It is idempotent — re-run it after a push so
a host provisioned before the images existed pulls them and starts the
retriever profile. `launch.sh` runs it once per host, with that host's
`host_benchmarks` when the matrix lists them.

**Registry logins.** Two clients need credentials: the **controller's** docker
CLI (a `docker --context` pull or push sends the client's credentials, not the
droplet's) and the **host's** (`provision.sh` does it; `docker compose` over
SSH uses it for the retriever pull). For DigitalOcean the API token is both the
username and the password:

```bash
docker login registry.digitalocean.com
```

### Rebuilding: `ce build`

`ce build` is the whole activation path: bundles, images, push, record the
digests in the lock file, then a pull on every matrix host. `CE_BUILD=1` makes
`launch.sh` run it on the build host.

```bash
ce build                      # build on the local daemon (emulated on an arm64 Mac)
ce build --host <host>        # build natively on an amd64 droplet
ce build --no-pull            # publish only, leave the hosts on the old image
ce build --no-retriever       # leave the ragqa sidecar image alone
```

`--host` is a docker context: the build runs on the droplet's daemon while
bundles, staged assets and client checkouts stay on the controller, and Docker
ships the build context over the connection. A build droplet needs only Docker
and an authorized key. The push uses the controller's registry login.
`orchestrator/build.py <bundles|base|images|retriever|pin|all>` (and
`cd runner && make bundles|base|images|push|show`) are the same steps
individually; `build.py pin --host <build host>` re-records the digests of
already-pushed images without rebuilding.

Images build for `linux/amd64` (the droplet architecture). On Apple Silicon the
build is emulated and the torch-heavy benchmark layers are slow enough to
justify an amd64 build droplet; the base cannot build natively on arm64
(MongoDB publishes no arm64 Debian packages). `ce build` counts in-flight runs
before touching a tag; host containers (`ce-gateway`, `ce-netrules`,
`ce-ragqa-retriever`, `controller.HOST_CONTAINERS`) are excluded.

**The ragqa retriever image is built, not found.** `compose.host.yml` pulls
`<registry>/ragqa-retriever:latest`; `build.py retriever` (folded into `images`
when the subset includes ragqa) builds it from the `ragqa-retriever-server`
checkout's Dockerfile. Its inputs change rarely and a rebuild re-resolves torch
against that day's base images, so pass `--no-retriever` unless they changed.
`docker history --no-trunc <registry>/ragqa-retriever:latest` shows whether the
published image matches the Dockerfile.

**When a rebuild is required.** `Dockerfile.base` bakes CodeEvolver in
(`COPY src/ scripts/ experiments/ runner/`), so a change under any of those
means base *and* every benchmark image. Changes under `orchestrator/` need no
rebuild; that code runs on the controller and never enters an image.

**Frozen between arms.** Nothing else triggers a build: `ce start` only
reconciles and dispatches, no source is bind-mounted into a run, and the client
repo comes from the bundle baked into the image. Because `image_tag` is a
mutable pointer, `ce build` records the pushed digest in the lock file and
`dispatch()` runs that digest, so every arm executes the same bytes even if the
tag has moved, a droplet was re-provisioned, or an image was pruned. The pin is
ignored if the registry or `image_tag` no longer matches it; `launch.sh` refuses
to dispatch a benchmark without a matching pin. Commit the lock file — it is
the matrix's provenance record.

**Never mid-matrix.** Re-pushing a tag while runs are in flight splits the
fleet: which runs got which code depends on when each host last pulled. `ce
build` refuses while any container is running; `--force` overrides, and the
results of a matrix built that way are not comparable.

**A new image beside a live fleet.** Give the new matrix a distinct
`image_tag` (as `matrix.evo.yaml` does with `evo-redo`): `ce build` then pushes
`<bench>-<image_tag>`, the pin lands in `images.<image_tag>.lock.json`, and the
live fleet's `latest` tag and lock file never move. Build on one of the new
matrix's own hosts — `ce build`'s in-flight guard only inspects the matrix's
hosts — and provision them with `TAG=<image_tag>`. `run_suffix` keeps the new
runs' ids distinct from any earlier runs of the same cells.

---

## Prompt-only ablation

`pareto_evolution` and `freeform` with the main coding model, where the agent
may edit only prompts (`specs/research_plan.md` has the design). It needs no
engine change and no image rebuild: it runs the `evo-redo` images and injects
everything else at dispatch. Files: `orchestrator/prompt_only/`,
`orchestrator/matrix.po.yaml` (`launch: {prompt_only: true}`, suffix `-po`).

**Seed.**
- The client repos have a `prompt-only` branch, cut from `main`: hover's
  prompts live in `langProBe/hover/prompts/hover_signatures.py`
  (LangProBe-CodeEvolver), phantomwiki's in
  `src/program/prompts/phantomwiki_signatures.py`, which also holds
  `SEARCH_WIKI_DESC`.
- Each is the same program as `main`, with the inline string signatures moved
  into Signature classes. Docstrings equal DSPy's default instruction and
  `desc=""` renders like no description. `prompt_only/check_seed_parity.sh`
  proves the rendered prompts are byte-identical to `main`.
- `prompt_only/build_bundles.py` squashes both branches into
  `runner/bundles-po/` (gitignored) and never overwrites `runner/bundles/`. The
  launch hooks copy them to `/opt/ce/po/seed/` on every host, and the matrix
  bind-mounts them over `/opt/seed/<bench>.bundle`, which
  `entrypoint.py::clone_workspace` clones at runtime.
- The image's baked venv still matches, because `requirements.txt` is
  unchanged.

**Instructions.**
- `prompt_only/{hover,phantomwiki}.md` rewrite the benchmark's
  `additional_instructions` for prompt-only work: the task description and
  existing constraints are kept; the signatures file is named; the
  architecture-change "What's Allowed" and "Ideas" bullets are dropped; and two
  constraint bullets are added: *"You are only allowed to edit prompts
  (Including signature docstrings, field descriptions, etc.). Do not under any
  circumstances edit code outside the prompts Or your changes will be rejected
  as cheating."* and *"This constraint applies to the program only. Updating
  your memory (memory.md) and `codeevolver.md` is allowed and expected."*
- The hooks JSON-encode each file into `CE_SET_HOVER` / `CE_SET_PHANTOMWIKI` as
  `additional_instructions=<json>`, refusing text that contains `;` (the
  `CE_SET` separator). `experiments/run.py::apply_overrides` parses the JSON
  back to the exact text, and the LM constraint gate judges every diff against
  it.
- The models stay at the image's `input.py` default; the hooks refuse a
  `CE_SET` model override.

**Kernel write fence** (`prompt_only/po_fence.sh`):
- **Where it runs.** Bind-mounted read-only at `/opt/po/po_fence.sh`.
  `prompt_only/po_fence_watch.sh` runs on each host (started over SSH by the
  hooks, pidfile `/opt/ce/po/watch<suffix>.pid`, log
  `/opt/ce/po/po_fence_watch<suffix>.log`) and injects the fence with
  `docker exec -d -u root` into every running `ce-*<suffix>` container whose
  fence loop is not alive.
- **What it does.** Every 10 s, as root (`CAP_FOWNER`, no `CAP_CHOWN`, so mode
  bits only): strips group/other write from everything under `/workspace`
  except `.git/`, the signatures directory (0777) and file (0666), and
  `codeevolver.md` (0666); makes `.git/hooks`, `.git/info` and `.git/config`
  non-writable to others.
- **Why it repeats.** The entrypoint opens the tree with `chmod -R a+rwX` and
  runs with `umask 0`, so a file rewritten by a root git operation comes back
  0666 until the next tick. Its log is `/artifacts/po_fence.log`.
- **What stays writable.** Memory is in `/artifacts`, which is not fenced. The
  agent keeps a working `.git`: commit, reset, checkout and branches all work.
- **Bytecode.** The matrix sets `PYTHONDONTWRITEBYTECODE=1`; a root-created
  `__pycache__` would be 0777 under `umask 0`, a way to plant `.pyc` files the
  root evaluator would import.

**Residuals.** The fence does not stop these; the audit does:
- the agent can commit any path through git plumbing
  (`hash-object`/`update-index`), because `.git` stays writable;
- `.git/config` can be replaced by renaming a file inside `.git`;
- Python code placed in the signatures file runs as root inside `run_eval`.

**Audit** (`scripts/prompt_only_audit.py <run_dir>...`), run on every collected
run:
- It fetches `repo.bundle` and checks every `refs/heads/*`,
  `refs/codeevolver/iterations/*` and `refs/codeevolver/evals/*/*` against
  `refs/heads/seed`.
- A ref passes only if its changed paths are within {signatures file,
  `codeevolver.md`} and the signatures file's AST, with every string constant
  blanked, is unchanged.
- Any violation returns `VOID` (exit 1). Eval anchors are included, so code that
  was evaluated and then reverted still voids the run.
- Main-matrix runs audit as `VOID`, the expected negative control. Tests:
  `tests/unit/test_prompt_only_audit.py`.

**Launch** (billable):
```bash
python orchestrator/prompt_only/build_bundles.py      # after any change to a prompt-only branch
CE_MATRIX=$PWD/orchestrator/matrix.po.yaml \
  nohup ./orchestrator/launch.sh > ~/.codeevolver/logs/launch.po.log 2>&1 &
python scripts/prompt_only_audit.py ~/ce-runs/*-po     # after collection
```

---

## Arm notes

`pareto_evolution` needs a comparison set (`comparison_set_size > 0`) and pays
one counted comparison eval at seed time (candidate 0). Budget the arm's
`max_metric_calls` identically to the other arms; the extra spend is part of
the treatment, not a config difference.

---

## Local dry run (runner in Docker, controller on the same machine)

Before a droplet is involved, the whole path — image, egress fence, holdout
sequestration, submit, iterate, collect — runs on one machine. The runner is a
container; the controller is the same `ce` on the host, so what passes here is
what the fleet runs.

```bash
source env/bin/activate                       # provider keys
export CE_BENCHMARKS=hover
export CE_MATRIX=$PWD/orchestrator/matrix.local.yaml
export CE_STATE=$PWD/orchestrator/state.local.json
export CE_RUNS_ROOT=$HOME/ce-data/runs        # host-side run dirs (no /data on a Mac)
export CE_SET="max_metric_calls=300"          # capped trial; omit for a full run
export CE_MAX_METRIC_CALLS=300

docker context create ce-local --docker host=unix:///var/run/docker.sock

cd runner
make bundles
PLATFORM=linux/arm64 REGISTRY=local make base images     # native on Apple Silicon
mkdir -p ~/ce-data/gateway-logs ssl
test -f ssl/squid.pem || openssl req -new -newkey rsa:2048 -days 3650 -nodes \
  -x509 -subj '/CN=ce-gateway' -keyout ssl/squid.pem -out ssl/squid.pem
REGISTRY=local CE_GATEWAY_LOG_DIR=$HOME/ce-data/gateway-logs \
  docker compose -f compose.host.yml up -d --build

cd .. && python3 orchestrator/controller.py start --bench hover --arm ee
```

`matrix.local.yaml` is one host, one seed, one arm, with container limits sized
for a Docker Desktop VM. `controller.host_addr()` returns None for a docker
context on a unix socket and switches mkdir / rm / rsync / the CANCEL sentinel
to local equivalents — no sshd, no `/data`. Build arm64 images for a local run
on Apple Silicon; an emulated amd64 image is impractically slow there.

The cheapest first check is the runner's self-test, which brings the whole stack
up and stops before submitting a job, so it costs no provider spend:

```bash
docker run --rm -v ~/ce-data/runs/selftest:/artifacts \
  --network=ce-host_ce-internal --dns=172.31.66.2 \
  --cap-drop=ALL --cap-add=SETUID --cap-add=SETGID --cap-add=FOWNER \
  --cap-add=NET_ADMIN --security-opt=no-new-privileges \
  -e CE_ARM=ee -e ANTHROPIC_API_KEY -e DEEPINFRA_API_KEY -e GMI_API_KEY \
  local/codeevolver-runner:hover-latest --self-test
```

---

## Known limitation

`MAX_CONSECUTIVE_ARCHITECT_FAILURES` (`optimizer_loop.py`) counts only
iterations where the architect call raises. When the harness exits non-zero
without raising, the loop records a skipped iteration and resets the counter,
so a systemically broken harness runs to `DEFAULT_MAX_ITERATIONS` at a few
seconds each with zero eval spend. Bounded, but it wastes the run; watch
`ce status` for iterations advancing with evals stuck at 0.
