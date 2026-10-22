# CodeEvolver
CodeEvolver is a self-improvement engine for AI applications. Provide an evaluation dataset, and our meta agent will inspect your system for failures and make improvements to the prompts and code. Every time a change is made, it is measured against your eval.

Delivering high-reliability output in an AI system requires that the system learns from its failures. Our meta agent is designed to aggregate experience into any AI harness by experimenting with different prompts, context pipelines, and architectures. 

## Optimizers
Every optimizer is budgeted on the number of evaluation rows it runs. Pick one with the `STRATEGY` environment variable.

| `STRATEGY` | What it does |
|---|---|
| **Agent-Directed Search** (recommended) | Agent-directed improvement: the agent controls the improvement process, with CodeEvolver's full feature set (incl. memory, tools, and skills). Use with Strategy = `freeform`. |
| **Pareto-Evolution** | Fixed evolutionary population search: the loop picks parents by Pareto-proportional sampling over the comparison set; the agent makes the change. Use with Strategy = `pareto_evolution`|
| **Agent-Directed-Explore** | Agent-directed exploration: the agent also maintains several lineages and chooses which candidate to build on. Use with Strategy = `ee` |
| **Simple Loop** | A plain coding-agent loop with the run_eval tool, traces, and the row budget. Use with Strategy = `simple_loop` |

Design details: `specs/engine3.md` (optimizers), `specs/requirements.md` (system overview).

## Quickstart (Docker, recommended)

The supported way to run CodeEvolver is in Docker: the API, MongoDB and the coding agents run in one container, and the agent can only edit the repo you mount into it (see [Security](#security)). You need Docker, `python3` and `git` on your machine — nothing else.

```bash
git clone https://github.com/julianghadially/CodeEvolver
cd CodeEvolver
cp env.example .env   # fill in your API keys; see below
```

Then:

1. Prepare your task repo ([Configure your task repo](#configure-your-task-repo)).
2. Write the job config, e.g. `my-job.json`. 
```json
{
  "program": "src.my_app.pipeline.MyPipeline",
  "metric": "src.my_app.metric.score",
  "trainset_path": "data/train.json",
  "testset_path": "data/test.json",
  "input_keys": ["question"],
  "initial_branch": "main",
  "max_metric_calls": 4000,
  "subsample_size": 50,
  "comparison_set_size": 150,
  "coding_harness": "opencode",
  "coding_harness_model": "<provider/model-id>"
}
```
For a first test run, use the following values:
```json
{
  "debug_max_iterations": 2,
  "subsample_size": 5,
  "comparison_set_size": 15,
}
```
3. Start the job:
```bash
scripts/run_local.sh --docker path/to/your-repo my-job.json
```
   The script checks prerequisites, starts the container (or reuses a running API), registers your repo, submits the job, waits for it to finish and prints where the results are. 
4. View results under `~/.codeevolver-mount/.codeevolver/`: `history/<job_id>.final.json` has the best candidate and score (`<job_id>.jsonl` has every iteration), and `memory/<job_id>/memory.md` has what the agent tried and learned.

Before your first real run, also run the test suites — see [Tests](#tests). The offline suite is free; the functional suite exercises the real agents and makes real LLM calls.

### Environment variables
Copy `env.example` to `.env` and fill in your API keys. Docker reads `.env` automatically; if you run natively, `source .env` in each terminal you use.

- `APP_MODE=local` and `SANDBOX_MODE=local`
- `STRATEGY=freeform` (restart the API/container after changing it)
- `ANTHROPIC_API_KEY`: the constraint gate's judge model always needs it
- The provider keys your coding model and your program's task model need (e.g. `DEEPINFRA_API_KEY`)

See `env.example` for the full list.

### Configure your task repo
The program you optimize needs to be configured.

1. Wrap your program in a function, which can be passed as a module path, e.g., `src.program.phantomwiki_pipeline.PhantomWikiReActPipeline`
2. Create an evaluation dataset with examples for your program. Each example must contain column names that match the input arguments of the program being optimized, plus any gold-label columns your metric reads.
3. Define a metric function that accepts program output in an argument named `output` and evaluation gold labels in an argument named `example`. The system will pass outputs and evaluation examples to the metric function as an output and example object, where sub-variables are stored in the output/example object as `output.variable` or `example.variable`. 
    - For example, if your dataset has gold stored in the 'gold' column, it will pass it to the metric function under `example.gold`. 
    - Your metric function should return both the score and helpful feedback to the agent (use a dict with "score" and "feedback" or a python object with .score and .feedback).
4. Configure traces, using opentelemetry or any opentelemetry-compatible tracing system (e.g., Langfuse, Arize, MlFlow, etc.) so the agent can inspect for failures. (See below)
5. The repo must be a git checkout. Python dependencies go in `requirements.txt` (installed into `<repo>/env`), and if applicable, any non-Python tools go in `runtime_requirements.txt` (one shell command per line).

#### Configuring traces
CodeEvolver uses an OpenTelemetry collector to process traces for a variety of application types, including language model applications and traditional code.

**It is HIGHLY recommended to configure your project to send traces.**

To do so, give a coding agent the tracing skill located at `src/engine/optimizer/v3/skills/tracing`. This will configure an OpenTelemetry provider and apply standard filters. We recommend inspecting the traces to ensure that the agent has visibility into every important function call, with the appropriate parent-child relationship for each span.

Note: applications that are not thread-safe will have traces that are processed out of order when num_threads > 1. Do not run num_threads > 1 if your application is not thread-safe.

## Tests

Run these before trusting a run. There are two suites to know about (plus opt-in end-to-end runs per optimizer arm; see `tests/README.md` for the full inventory):

1. **Offline tests (free, no API keys).** Fast, deterministic unit/control/harness tests. Run them before your first job and after any CodeEvolver change:
   `APP_MODE=local python -m pytest tests/unit tests/controls tests/coding_harness -m "not live and not integration"`
2. **Functional agent tests (billable, ~10 minutes).** These launch the real coding agents (session resume, `close_iteration`, trace inspection, the constraint checker, and more) against your configured models and keys, and catch environment problems — a missing harness binary, a model your account can't call, bad provider auth — that the offline suite cannot. Run them before your first real run:
   `APP_MODE=local scripts/run_tests.sh --phase functional --harness opencode`
   (Use `--harness claude` if your jobs use `coding_harness=claude`. A harness pass is skipped when its credentials are absent.)

The test suites run natively on your machine, not inside the Docker container, so they need the [native setup](#native-setup-advanced) below (in particular the `env/` venv from step 4 — `run_tests.sh` calls `python`). They exercise the same engine code, harness configuration, models and keys your Docker jobs use; the Docker image pins its own toolchain (e.g. OpenCode), so a native pass does not guarantee the container's binaries.

## Native setup (advanced)

Running without Docker gives the coding agents full edit access to your file system — prefer the Docker quickstart above. 

```bash
# 1. System runtime
brew install python@3.11 node deno uv ripgrep   # uv builds your repo's venv; rg backs OpenCode's search tools
# On macOS you may also need: xcode-select --install

# 2. Coding agent CLI: OpenCode (default harness) and/or Claude Code
npm install -g opencode-ai          # needs >= 1.17.11; or: curl -fsSL https://opencode.ai/install | bash
npm install -g @anthropic-ai/claude-code   # only if coding_harness=claude

# 3. MongoDB Community (single-node, no replica set)
brew tap mongodb/brew
brew install mongodb-community@7.0
brew services start mongodb-community@7.0

# 4. Clone CodeEvolver and install it
git clone https://github.com/julianghadially/CodeEvolver
cd CodeEvolver
python3.11 -m venv env
source env/bin/activate
# The [local] extra pulls in claude-agent-sdk, anyio, litellm and the
# OpenTelemetry SDK, which the optimizer subprocess needs.
pip install -e ".[local,dev]"
```

Then run a job the same way as the quickstart, without `--docker`:

```bash
scripts/run_local.sh path/to/your-repo my-job.json
```

Results land in `~/.codeevolver/` instead of `~/.codeevolver-mount/.codeevolver/`.

To run the steps by hand instead (MongoDB running, venv activated, `.env` sourced):
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8080
curl -X POST http://127.0.0.1:8080/local/connect-repo \
  -H 'Content-Type: application/json' -d '{"local_path": "/absolute/path/to/your-repo"}'
curl -X POST http://127.0.0.1:8080/optimize -H 'Content-Type: application/json' \
  -d '<my-job.json, plus "repo_url": "file:///absolute/path/to/your-repo">'
curl http://127.0.0.1:8080/job/<job_id>
```

## Additional Notes
### Inspecting logs for a healthy optimization run
1. Open memory (`~/.codeevolver/memory/<job_id>/memory.md`; in Docker, `~/.codeevolver-mount/.codeevolver/memory/<job_id>/memory.md`). Agentic memory includes reflections on ideas attempted and observations on how well they performed. 
2. Open meta reflection (`~/.codeevolver/memory/<job_id>/meta_reflection.md`) and trust it: this post-run audit is the most reliable summary of whether the run actually worked.
3. If needed, open logs (`~/.codeevolver/logs/<job_id>.log`). Identify any errors or warnings.

### Troubleshooting
1. **System Health:** confirm all unit and functional tests pass (see [Tests](#tests)). Check inspection logs and memories for a healthy optimization run. (See above).
2. **Traces:** Inspect architect traces and system traces for irregularities.
3. **Sample size:** Optimizations generally need a few hundred training examples to perform well; the comparison set is drawn from the training set.
4. **Low data quality:** conflicting or inconsistent gold labels make it difficult to find improvement opportunities.
5. **Train/test mismatch:** if the training set does not cover the domain of the held-out test set, the training signal is weak.
6. **Noise:** a noisy metric or a small comparison set interferes with candidate selection. `run_variance_estimate` measures the run-to-run noise of the training subsample.
7. **Saturation:** As a program approaches a perfect score, it requires increasingly more precise measurement to progress and regression to the mean is exacerbated.

### Storage Note
State lives here (`~/.codeevolver-mount/.codeevolver/` in Docker):
```
~/.codeevolver/
├── history/{job_id}.jsonl        # iteration history
├── history/{job_id}.final.json   # terminal snapshot
├── logs/{job_id}.log             # optimizer subprocess stdout
├── memory/{job_id}/              # agent memory
├── traces/{job_id}/              # evaluation traces
└── architect_traces/{job_id}/    # architect session traces
```
Each location can be overridden with the `CODEEVOLVER_*_DIR` variables in `env.example`.

### Security
This repo is designed to be run in a Docker container or on dedicated hardware (like a server). Without a container, Claude agents and OpenCode agents have full edit access to your file system. We strongly recommend using the Docker container setup (or server setup) so you are not unleashing long-running coding agents on a personal computer unprotected. 

To run in Docker, use `scripts/run_local.sh --docker` (see [Quickstart](#quickstart-docker-recommended)). It needs only Docker, `python3` and `.env`. The API, MongoDB and the agents run in one container; your repo is mounted at `/workspace` and job state is written to `~/.codeevolver-mount/`. The Docker image ships OpenCode; for `coding_harness=claude`, update the Docker script. 

## Gratitude and Acknowledgments
This work was originally inspired by AlphaEvolve, GEPA, and DSPy. In fact, the first version of CodeEvolver was built by combining GEPA with coding agents in January 2026. We would like to extend a special thank you to the community that brought GEPA and DSPy to life.

The fixed-evolution optimizer (`pareto_evolution`) selects parents by GEPA's per-instance Pareto-proportional rule (`src/engine/optimizer/v3/pareto_evolution/select_candidate.py`) and prunes dominated candidates by GEPA's rule. Acknowledgement goes to:

GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning.
Authors: Lakshya A Agrawal and Shangyin Tan and Dilara Soylu and Noah Ziems and Rishi Khare and Krista Opsahl-Ong and Arnav Singhvi and Herumb Shandilya and Michael J Ryan and Meng Jiang and Christopher Potts and Koushik Sen and Alexandros G. Dimakis and Ion Stoica and Dan Klein and Matei Zaharia and Omar Khattab
Url: https://arxiv.org/abs/2507.19457
https://github.com/gepa-ai/gepa
