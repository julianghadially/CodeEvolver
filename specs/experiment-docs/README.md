# Experiment Fleet — Read Me First

CodeEvolver's research experiments run on a **fleet**: a *controller* on your own
machine and *runners* in Docker containers on disposable DigitalOcean droplets.
Each experiment is one matrix file (arms × benchmarks × seeds); the main matrix,
`orchestrator/matrix.main.yaml`, is 5 arms × 5 benchmarks × 3 seeds, one droplet
per arm, every run in its own clean container.

**You never configure a server.** Everything below runs from your own machine;
the controller drives the droplets over plain SSH (`docker --context <host>`).

```
controller (your machine)    one droplet per arm
┌──────────────────┐         ┌───────────────────────────────────────────┐
│ ce start         │ docker  │ squid gateway (SNI allowlist)             │
│ ce status        │─context▶│   └ ce-internal (no default route)        │
│ ce cancel        │ over SSH│       ├ ce-<arm>-hover-s1     (container) │
│ ~/.codeevolver/  │◀─rsync──│       ├ ce-<arm>-phantomwiki-s1           │
│ $CE_RUNS_DIR/    │         │       └ ce-<arm>-ragqa-s1                 │
└──────────────────┘         │  /data/runs/<run_id> (only writable mount)│
                             └───────────────────────────────────────────┘
```

---

## Features

- **Clean-slate containers.** One container == one run. Every run starts from an
  immutable image and a client repo squashed to a **single orphan commit**, so
  there is no prior-run state — no leftover branches, memory, or eval results — and
  no repo reset step in the loop. The only writable host path is that run's own
  artifact directory, so an agent has nowhere to leave a note for a later run.
- **The agent has no permission to read the test set.** OS-enforced, not
  pattern-matched: agent subprocesses drop to an unprivileged `ce-agent` user while
  the holdout is moved to root-owned `/opt/holdout` (mode 600) and its blob is
  *evicted from git history and verified gone*. Deny-path rules remain as
  defense-in-depth. The optimizer/evaluator stay privileged, so final test scores
  still run normally.
- **Clean-slate inspector agent.** Runs at seed, **before any eval spend**, and
  verifies the environment is genuinely fresh: no old solutions in git history, every
  holdout/test file covered by the fences, no prior-agent leftovers in the container.
  A `contaminated` verdict **aborts the run**; `uncertain` fails open with findings
  logged.
- **Run-integrity inspector.** A read-only auditor that checks for **cheating**
  (test-set leakage, hard-coded answers → `no_cheating` / `non_impactful_cheating` /
  `impactful_cheating`) **and functionality** (% of iterations that actually
  functioned — no major blocker to improving). Runs mid-run at 20% of eval budget and
  again at close. Advisory — it never cancels a run.
- **Egress allowlist that fails closed.** Runners sit on a no-default-route network
  behind a squid SNI filter. Before any provider spend a canary asserts github /
  huggingface are unreachable and the providers are reachable; **if the fence fails
  open, the run aborts.** The squid access log ships as a run artifact, and an
  egress audit flags task-LM calls made outside a budgeted eval.
- **Orchestrator logs surface high-risk errors.** The collect sweep logs
  `INTEGRITY WARNING <run_id>` and `EGRESS WARNING <run_id>` — to stdout and
  `~/.codeevolver/logs/fleet.log` — when a verdict isn't clean.

---

## Before your first launch

- [ ] **Clone the client repos** under `CE_REPO_ROOT` (below).
- [ ] **Export your env vars** (below), including `REGISTRY` and
      `DIGITALOCEAN_API_KEY`.
- [ ] **Set your SSH key and DigitalOcean project** in the matrix's `fleet:` block
      (`ssh_key`, `project`).
- [ ] **Dry-run** — `ce start --dry-run --once` and `cd runner && make show`
      before spending anything. For a full trial on one machine, see "Local dry
      run" in `experiment-fleet.md`.
- [ ] **Adding a benchmark** — one entry in `orchestrator/benchmarks.yaml` (slug,
      `experiment_dir`, `repo_dirname`, `est_hours`, optional `seed_branch`) plus an
      `experiments/<experiment_dir>/input.py`. The fleet slug is usually *not* the
      directory name (`phantomwiki` → `experiments/phantom-wiki`).

---

## Env vars

```bash
export REGISTRY=registry.digitalocean.com/<your-registry>   # images: <registry>/codeevolver-runner:<bench>-<tag>
export DIGITALOCEAN_API_KEY=...                    # fleet.py creates/destroys droplets
export CE_REPO_ROOT=~/Documents/code               # where your client repos live
export CE_REPO_PHANTOMWIKI=/mnt/big/phantom-wiki   # optional per-benchmark override
export CE_RUNS_DIR=~/ce-runs                       # collected runs (this is the default)

# provider keys — exported in the controller shell, passed per run.
# These are the ONLY credentials a runner ever holds.
export ANTHROPIC_API_KEY=... OPENAI_API_KEY=... DEEPINFRA_API_KEY=... GMI_API_KEY=... DEEPSEEK_API_KEY=...
# GMI_CLOUD_API_KEY works instead of GMI_API_KEY — the ARC and paperreview
# solvers read whichever of the two is set.

python3 orchestrator/fleet_config.py               # show what CE_BENCHMARKS resolves to
```

`launch.sh` takes `CE_BENCHMARKS` from the matrix. When you run `ce`,
`build.py` or `provision.sh` by hand, export it yourself
(`export CE_BENCHMARKS=hover,phantomwiki,ragqa`); there is deliberately no
default there, since inheriting someone else's subset would burn days of spend.

### Getting the client repos

`CE_REPO_ROOT` says where they live, not where they come from. `build.py
bundles` reads each benchmark's `seed_branch` out of the local checkout, so the
repo has to be on disk before the first build:

| Slug | `repo_dirname` | Seed branch |
|---|---|---|
| hover, ragqa | `LangProBe-CodeEvolver` | `main` |
| phantomwiki | `phantom-wiki` | `main` |
| arcagi2 | `ARC-AGI-2` | `seed-program-deepseek-v4flash` |
| paperreview | `PaperReview` | `seed-program-deepseek-v4flash` |

A plain clone is enough: the seed branch is resolved as `origin/<branch>` when
no local branch exists, which is the normal state right after cloning. If the
repo has been sitting around, `git -C <repo> fetch --all` first — the build
bundles whatever the checkout knows, and a stale ref silently seeds an old
program.

Your subset controls what each host pulls and runs: images, the ~8 GB RAGQA
retriever sidecar (only started if `ragqa` is in the host's subset), and
therefore host sizing. `slots_per_host` is a **concurrency cap, not a benchmark
count**.

---

## Launch a matrix

```bash
CE_MATRIX=$PWD/orchestrator/matrix.main.yaml \
  nohup ./orchestrator/launch.sh > ~/.codeevolver/logs/launch.main.log 2>&1 &
tail -f ~/.codeevolver/logs/launch.main.log
```

`launch.sh` creates the droplets, provisions them, checks the image digest
lock, runs the matrix to completion and tears the droplets down once every run
is collected. Useful switches: `CE_BUILD=1` (rebuild, push and re-pin the images
first), `CE_QUARTER=1` (quarter budgets for a load test), `CE_STATE` (default
`orchestrator/state.<fleet tag>.json`). Other matrices: `matrix.evo.yaml`,
`matrix.abl.yaml`, `matrix.po.yaml`, and `matrix.local.yaml` for a single-machine
trial. Details: `experiment-fleet.md`.

## Daily use

```bash
cd orchestrator && export PATH="$PWD:$PATH"
export CE_MATRIX=$PWD/matrix.main.yaml CE_STATE=$PWD/state.ce-main.json

ce status                     # every run: host, state, iteration, evals, elapsed
ce logs ee-hover-s1 -f        # tail a container
ce pull ee-hover-s1           # force an artifact sync now
ce cancel ee-hover-s1         # cooperative cancel (see below)
ce cancel --now ee-hover-s1   # docker kill
ce start --dry-run --once     # show what would dispatch, change nothing
ce start --bench hover --arm ee   # drive a subset by hand
ce recycle --host <host>      # wipe a host (containers, images, run dirs)
```

`ce start` is safe to kill and restart — it reconciles against `docker ps` every
pass, so it re-attaches to live containers instead of dispatching them twice. The
scheduler wakes every 15 min; each pass collects finished runs, syncs running ones,
then dispatches into freed slots. `ce status` reconciles on demand and is always
current.

**Cancellation is slow on purpose.** `ce cancel` is treated as budget exhaustion:
the run finishes its current iteration, then runs finalization (seed + final test
evals). That can take hours, and you get a fully scored run. `--now` kills
immediately and loses those scores — use it only when you don't want the result.

---

## Where results land

Artifacts stream out live (the container's `$HOME/.codeevolver` **is** the host bind
mount), rsynced to the controller every 10 min and once at exit:

| Path | Contents |
|---|---|
| `~/.codeevolver/{memory,architect_traces,history,logs,runtime_tests}/{job_id}/` | Merged into your existing tree — `query_iteration_history.py`, `view.py` all work unchanged. |
| `$CE_RUNS_DIR/<run_id>/` | `status.json`, `settings.json`, `egress.json`, `integrity.json`, `candidates.jsonl`, `history.json`, `repo.bundle`, `workspace.tar.gz`, squid log. |
| `$CE_RUNS_DIR/repos/<bench>.git` | Bare repo with every run's branches under `refs/runs/<run_id>/*`. |

Use **worktrees, not clones**, for per-candidate analysis — many can exist at once:

```bash
git -C ~/ce-runs/repos/hover.git for-each-ref --format='%(refname)' refs/runs
git -C ~/ce-runs/repos/hover.git worktree add /tmp/cand-1 <sha>
```

Result tables: `scripts/results_tables.py`. Best-of-population analysis:
`scripts/best_of_population/README.md`.

---

Full mechanism, rationale, and accepted residuals:
**`specs/experiment-docs/experiment-fleet.md`**. Single-host (non-fleet) use:
**`specs/experiment-docs/local-deployment.md`**. Engine internals (integrity
agents, holdout fence): **`specs/engine3.md`**.
