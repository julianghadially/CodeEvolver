#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Fleet controller. Runs on the operator's machine; drives remote Docker over SSH.

Design constraints that shaped this:

* **Nothing is installed on the workers.** Dispatch goes through `docker
  --context <host>`, i.e. plain SSH. A droplet needs Docker and an authorized
  key, nothing else -- no agent to deploy, no CodeEvolver port exposed to a
  network, no per-server script to keep in sync.
* **State is reconciled, never trusted.** Truth is `docker ps` across the hosts
  plus each run's `status.json`; `state.json` is only a cache. Kill this process
  mid-matrix and restart it -- it re-attaches to the live containers instead of
  double-dispatching.
* **Cancellation is cooperative and slow on purpose.** The optimizer treats
  cancellation as budget exhaustion, so it finishes the current iteration and
  then runs finalization. `ce cancel` therefore drops a sentinel file and lets
  the container exit on its own schedule; only `--now` kills. A `docker stop`
  grace timer would race the finalization test evals and lose them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

import yaml

import egress_audit
from fleet_config import Benchmark, needs_retriever, runs_dir, selected_benchmarks

HERE = Path(__file__).resolve().parent
# CE_MATRIX lets a scratch matrix (one host, one run -- see matrix.local.yaml)
# be driven without editing the committed fleet matrix.
MATRIX_PATH = Path(os.environ.get("CE_MATRIX", HERE / "matrix.main.yaml"))
STATE_PATH = Path(os.environ.get("CE_STATE", HERE / "state.json"))
# Written by `ce build` after a push; read here on every dispatch. See
# build.py::record_digest for why the matrix is pinned by digest, not by tag.
LOCK_PATH = Path(os.environ.get("CE_IMAGE_LOCK", HERE / "images.lock.json"))
# Where a host keeps its per-run artifact directories. /data/runs on a droplet;
# somewhere under $HOME when the "host" is this machine, which has no /data.
RUNS_ROOT = os.environ.get("CE_RUNS_ROOT", "/data/runs")
TERMINAL = ("completed", "failed", "cancelled", "error")


LOG_PATH = Path(
    os.environ.get(
        "CE_FLEET_LOG", Path.home() / ".codeevolver" / "logs" / "fleet.log"
    )
)


def log(msg: str) -> None:
    """Print AND append to fleet.log so `ce run`'s history survives the tty."""
    line = f"[ctl {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a") as f:
            f.write(line + "\n")
    except OSError:
        pass  # a read-only or missing dir must never break the controller


# --------------------------------------------------------------------------
# matrix
# --------------------------------------------------------------------------

@dataclass
class Run:
    run_id: str
    host: str
    bench: str
    arm: str
    seed: int

    @property
    def container(self) -> str:
        return f"ce-{self.run_id}"

    @property
    def remote_dir(self) -> str:
        return f"{RUNS_ROOT}/{self.run_id}"


def _bench(slug: str) -> Benchmark:
    for b in selected_benchmarks():
        if b.slug == slug:
            return b
    raise KeyError(f"{slug} is not in CE_BENCHMARKS")


def load_matrix(path: Path = MATRIX_PATH) -> dict:
    with path.open() as fh:
        matrix = yaml.safe_load(fh)
    # The registry is per deployment, so the experiment matrices leave it to
    # $REGISTRY (shared with build.py and provision.sh); a matrix that pins one
    # (matrix.local.yaml: `local`) keeps it. Checked where an image name is
    # built (`registry_of`), so status/cancel work without it.
    matrix["registry"] = matrix.get("registry") or os.environ.get("REGISTRY")
    # `hosts` drives reconcile/status/pull/recycle (it must cover a host even
    # when --arm filters its runs away) and `assignment` drives build_runs, so
    # the two are separate keys naming the same machines. Drift between them
    # silently drops a whole arm's runs, so fail at load instead.
    hosts, assignment = set(matrix["hosts"]), set(matrix["assignment"])
    if hosts != assignment:
        raise ValueError(
            f"{path}: hosts and assignment disagree — "
            f"only in hosts: {sorted(hosts - assignment) or 'none'}; "
            f"only in assignment: {sorted(assignment - hosts) or 'none'}"
        )
    # `host_benchmarks` (optional) restricts a host's arm to the listed slugs
    # (a fan-out matrix: one benchmark per arm). Keys must name real hosts,
    # same fail-at-load rationale as above.
    stray = set(matrix.get("host_benchmarks", {})) - hosts
    if stray:
        raise ValueError(f"{path}: host_benchmarks names unknown hosts: {sorted(stray)}")
    # `host_seeds` (optional) restricts a host's arm to the listed seeds: a
    # seed-sharded matrix (one host per seed of ONE arm). Needed there because cmd_run by default never
    # runs a benchmark twice on a host (`bench_concurrency`, default 1), so
    # three seeds of one arm cannot share a host. Same fail-at-load rules as
    # host_benchmarks; absent means every seed on every host, as before.
    host_seeds = matrix.get("host_seeds", {})
    stray = set(host_seeds) - hosts
    if stray:
        raise ValueError(f"{path}: host_seeds names unknown hosts: {sorted(stray)}")
    for host, seeds in host_seeds.items():
        unknown = set(seeds) - set(matrix["seeds"])
        if unknown:
            raise ValueError(
                f"{path}: host_seeds[{host}] lists seeds not in `seeds`: {sorted(unknown)}"
            )
    # `run_suffix` (optional) is appended to every run_id -- and therefore to
    # the container name, the /data/runs dir, ~/ce-runs/<run_id> and the
    # refs/runs/<run_id> namespace -- so a re-run of a cell whose original
    # artifacts are kept on disk cannot collide with them. Shape-checked so
    # scripts/results_tables.py::RUN_ID_RE can parse it back off the id.
    suffix = matrix.get("run_suffix", "")
    if suffix and not re.fullmatch(r"-[a-z0-9]+", suffix):
        raise ValueError(f"{path}: run_suffix must match -[a-z0-9]+, got {suffix!r}")
    # An `assignment` value is one arm or a list of arms (a host co-running
    # both arms of an ablation). Every
    # arm must be declared in `arms` (when the matrix lists them), so a typo
    # fails at load instead of dispatching an arm the engine doesn't know.
    declared = matrix.get("arms")
    for host in matrix["assignment"] if declared is not None else ():
        unknown = set(host_arms(matrix, host)) - set(declared)
        if unknown:
            raise ValueError(
                f"{path}: assignment[{host}] lists arms not in `arms`: {sorted(unknown)}"
            )
    # `bench_concurrency` (optional, default 1) caps how many runs of ONE
    # benchmark a host runs at once. 1 is the original one-queue-per-benchmark
    # policy; raise it to co-host seeds/arms of the same benchmark.
    bc = matrix.get("bench_concurrency", 1)
    if not isinstance(bc, int) or bc < 1:
        raise ValueError(f"{path}: bench_concurrency must be an int >= 1, got {bc!r}")
    return matrix



def registry_of(matrix: dict) -> str:
    """The container registry images are named under ($REGISTRY or the matrix)."""
    if not matrix.get("registry"):
        raise SystemExit("no registry: set REGISTRY (e.g. registry.digitalocean.com/<name>)")
    return matrix["registry"]

def host_arms(matrix: dict, host: str) -> list[str]:
    """The arms `assignment` gives a host: a bare string or a list."""
    arms = matrix["assignment"][host]
    return [arms] if isinstance(arms, str) else list(arms)


def build_runs(matrix: dict) -> list[Run]:
    """Expand the matrix into runs against the fixed arm -> host assignment.

    Benchmarks come from CE_BENCHMARKS (see fleet_config), never from the matrix
    file, so teammates running different subsets share this one config.

    Ordered seed-major so that a full all-arm, all-benchmark checkpoint lands
    before any seed-2 work starts.
    """
    benches = selected_benchmarks()
    suffix = matrix.get("run_suffix", "")
    host_seeds = matrix.get("host_seeds", {})
    runs: list[Run] = []
    for seed in matrix["seeds"]:
        for host in matrix["assignment"]:
            # Seed-sharded matrices list each host's seeds; a matrix without
            # the key runs every seed on every host.
            seeds = host_seeds.get(host)
            if seeds is not None and seed not in seeds:
                continue
            # Fan-out matrices restrict a host's arm to its own slugs; a matrix
            # without the key keeps the full CE_BENCHMARKS expansion.
            allowed = matrix.get("host_benchmarks", {}).get(host)
            for arm in host_arms(matrix, host):
                for bench in benches:
                    if allowed is not None and bench.slug not in allowed:
                        continue
                    runs.append(Run(
                        run_id=f"{arm}-{bench.slug}-s{seed}{suffix}",
                        host=host, bench=bench.slug, arm=arm, seed=seed,
                    ))
    return runs


# --------------------------------------------------------------------------
# docker over ssh
# --------------------------------------------------------------------------

def docker(host: str, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["docker", "--context", host, *args],
        capture_output=True, text=True, check=check,
    )


def ssh(host_addr: str, command: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["ssh", host_addr, command], capture_output=True, text=True, check=check
    )


def host_addr(matrix: dict, host: str) -> str | None:
    """SSH target for a host, or None when the host is THIS machine.

    Taken from the docker context, which is the single source of truth for
    where a host lives. A context pointing at a unix socket is the local daemon
    — the local development shape: runner in a container, controller outside
    it, one machine. There is then nothing to ssh into and nothing to copy
    across a network, so every remote operation below has a local equivalent
    rather than a second code path.
    """
    out = docker(host, "context", "inspect", host, check=False).stdout
    try:
        endpoint = json.loads(out)[0]["Endpoints"]["docker"]["Host"]
    except Exception:  # noqa: BLE001
        return matrix.get("ssh", {}).get(host, host)
    return endpoint[len("ssh://"):] if endpoint.startswith("ssh://") else None


def make_run_dir(addr: str | None, path: str) -> None:
    if addr is None:
        Path(path).mkdir(parents=True, exist_ok=True)
    else:
        ssh(addr, f"mkdir -p {path}", check=False)


def remove_run_dir(addr: str | None, path: str) -> None:
    if addr is None:
        subprocess.run(["rm", "-rf", path], check=False)
    else:
        ssh(addr, f"rm -rf {path}", check=False)


# Per-host infrastructure from compose.host.yml. They share the ce- prefix with
# run containers but are never runs: the gateway is up on every provisioned
# host, so counting it as an in-flight run made `ce build` and `ce recycle`
# refuse unconditionally.
HOST_CONTAINERS = frozenset({"ce-gateway", "ce-netrules", "ce-ragqa-retriever"})


def running_runs(host: str) -> list[str]:
    """Names of RUN containers currently running on a host (infra excluded)."""
    return [n for n, st in live_containers(host).items()
            if st == "running" and n not in HOST_CONTAINERS]


def live_containers(host: str) -> dict[str, str]:
    """{container_name: state} for every ce-* container on a host."""
    res = docker(
        host, "ps", "-a", "--filter", "name=^ce-",
        "--format", "{{.Names}}\t{{.State}}", check=False,
    )
    out: dict[str, str] = {}
    for line in res.stdout.strip().splitlines():
        if "\t" in line:
            name, state = line.split("\t", 1)
            out[name] = state
    return out


# `env_passthrough` keys that are switches rather than credentials: absent is
# their normal state, so warning about them on every dispatch would be noise
# that teaches the operator to skim past the warnings that do matter (a missing
# provider key really does fail the run).
# GMI_CLOUD_API_KEY joins them for a different reason: it is an ALTERNATIVE
# spelling of GMI_API_KEY (the arcagi/paperreview solvers read
# `GMI_CLOUD_API_KEY or GMI_API_KEY`), so exactly one of the pair being unset is
# the normal case and warning about it would train the operator to skim.
# GMI_API_KEY itself still warns, so a run with neither set is not silent.
OPTIONAL_PASSTHROUGH = frozenset(
    {"CE_SET", "CE_MAX_METRIC_CALLS", "CE_RESEARCH_MODE", "GMI_CLOUD_API_KEY","CE_CONFIG",
     # Simulator worker pools: unset means the client runtime's defaults.
     "APPWORLD_SERVER_COUNT", "APPWORLD_BASE_PORT",
     "ALFWORLD_SERVER_COUNT", "ALFWORLD_BASE_PORT",
     # Task-LM routing overrides. Unset is the normal case and the intended
     # one: the seed programs' DEFAULT_PROVIDER is already the head of their
     # preference order (gmi), so setting these is the exception -- repointing
     # a provider for one dispatch without rebuilding the seed bundle.
     "LM_PROVIDER", "LM_FALLBACK"}
)


# Credentials only SOME benchmarks need. `env_passthrough` is one global list,
# so without this an OPENROUTER_API_KEY that only alfworld calls would warn on
# every hover/phantomwiki/ragqa dispatch -- exactly the noise OPTIONAL_PASSTHROUGH
# exists to prevent. Putting it in OPTIONAL_PASSTHROUGH instead would go too far
# the other way and silence the warning on the one benchmark where the key is
# load-bearing, so the warning is scoped to the benchmarks that actually use it.
BENCH_CREDENTIALS = {"OPENROUTER_API_KEY": {"alfworld"}}
# DEEPSEEK_API_KEY is deliberately NOT scoped here. Every wired benchmark --
# appworld and alfworld included -- runs DeepSeek-V4-Flash through the
# gmi -> deepseek -> deepinfra provider chain, so the key warns
# on every dispatch exactly like GMI_API_KEY / DEEPINFRA_API_KEY. A missing
# key does not fail the run (every call diverts to the next provider), so the
# warning is the only way an operator finds out before the wall clock does.


def resolve_image(matrix: dict, b: Benchmark) -> str:
    """The exact image to run: the locked digest when there is one, else the tag.

    Falling back to the tag keeps a fresh checkout and the local matrix working
    before anything has been pushed. The lock is only trusted when it was built
    for THIS registry and tag, so switching either in the matrix cannot
    silently keep dispatching the old pin.
    """
    image = b.image(registry_of(matrix), matrix["image_tag"])
    try:
        entry = json.loads(LOCK_PATH.read_text())[b.slug]
    except (OSError, KeyError, ValueError):
        return image
    return entry["digest"] if entry.get("image") == image else image


# The headless Xorg display provision.sh's needs_gpu block stands up on GPU
# hosts. One constant shared with that script; override only for a host whose
# :99 is already taken.
GPU_DISPLAY = os.environ.get("CE_GPU_DISPLAY", ":99")


def _host_has_gpu_display(matrix: dict, host: str) -> bool:
    """Whether provision.sh's needs_gpu block ran AND its display is live.

    Probed rather than configured, and probed functionally: a stale Xauthority
    left by a dead Xorg must not cause dispatch to mount a broken display --
    xdpyinfo against the actual server is the check. A GPU benchmark dispatched
    to a host that fails the probe degrades to in-container Xvfb (correct,
    slower) instead of failing on first render.
    """
    probe = (
        f"test -f /opt/ce/x11/Xauthority && "
        f"DISPLAY={GPU_DISPLAY} XAUTHORITY=/opt/ce/x11/Xauthority "
        f"xdpyinfo >/dev/null 2>&1"
    )
    addr = host_addr(matrix, host)
    if addr is None:
        if not Path("/opt/ce/x11/Xauthority").is_file():
            return False
        return subprocess.run(["sh", "-c", probe], capture_output=True).returncode == 0
    return ssh(addr, probe, check=False).returncode == 0


def dispatch(matrix: dict, run: Run) -> None:
    b = _bench(run.bench)
    image = resolve_image(matrix, b)

    # CE_SET is one value for every dispatch, but a partial run needs a
    # different max_metric_calls per benchmark (hover 9000, ragqa 25000 at full
    # budget). CE_SET_<SLUG> is appended to CE_SET for that benchmark's runs
    # only; the runner folds the merged string like any other CE_SET.
    overrides: dict[str, str] = {}
    bench_set = os.environ.get(f"CE_SET_{run.bench.upper()}")
    if bench_set:
        overrides["CE_SET"] = ";".join(
            s for s in (os.environ.get("CE_SET", ""), bench_set) if s
        )

    env_args: list[str] = []
    for key in matrix["env_passthrough"]:
        val = overrides.get(key) or os.environ.get(key)
        if val:
            env_args += ["-e", f"{key}={val}"]
            continue
        needed_by = BENCH_CREDENTIALS.get(key)
        if needed_by is not None and run.bench not in needed_by:
            continue  # another benchmark's provider key; irrelevant to this run
        if key not in OPTIONAL_PASSTHROUGH:
            log(f"WARNING: {key} not set in controller env — {run.run_id} may fail")

    for key, val in (
        ("CE_RUN_ID", run.run_id), ("CE_BENCH", run.bench),
        # The experiments/ directory is often NOT the slug (phantomwiki ->
        # experiments/phantom-wiki, ragqa -> experiments/RAGQAArenaTech), and it
        # is what `experiments.run --program` takes.
        ("CE_PROGRAM", b.experiment_dir),
        ("CE_ARM", run.arm), ("CE_SEED", str(run.seed)),
    ):
        env_args += ["-e", f"{key}={val}"]

    if b.needs_retriever:
        env_args += ["-e", "RAGQA_RETRIEVER_URL=http://ragqa-retriever:8894/api/search"]

    addr = host_addr(matrix, run.host)
    make_run_dir(addr, run.remote_dir)

    # GPU benchmarks (alfworld: AI2-THOR/Unity needs an X display) render
    # against the host's headless Xorg, stood up by provision.sh's needs_gpu
    # block on :99 with an Xauthority under /opt/ce/x11. The mounts and the
    # DISPLAY are only added when the host actually has that display --
    # otherwise (a laptop running matrix.local.yaml) nothing is passed and the
    # container's runtime falls back to in-container Xvfb software rendering,
    # ~5x slower but correct, which is exactly what a local smoke run wants.
    gpu_args: list[str] = []
    if b.needs_gpu and _host_has_gpu_display(matrix, run.host):
        gpu_args = [
            "--gpus", "all",
            "-e", "NVIDIA_DRIVER_CAPABILITIES=all",
            "-e", f"DISPLAY={GPU_DISPLAY}",
            "-e", "XAUTHORITY=/xauth/Xauthority",
            "-v", "/tmp/.X11-unix:/tmp/.X11-unix:ro",
            "-v", "/opt/ce/x11:/xauth:ro",
        ]

    args = [
        "run", "-d", "--name", run.container,
        "--restart", "no",
        "-v", f"{run.remote_dir}:/artifacts",
        *matrix["defaults"]["docker_args"],
        *gpu_args,
        *env_args,
    ]
    if runtime := matrix["defaults"].get("runtime"):
        args += ["--runtime", runtime]
    args.append(image)

    docker(run.host, *args)
    log(f"dispatched {run.run_id} -> {run.host} ({image})")


def collect(matrix: dict, run: Run, final: bool = False) -> None:
    """rsync artifacts to the controller machine, then fold the bundle into the bare repo."""
    addr = host_addr(matrix, run.host)
    # rsync takes a plain path for a local source; only a remote one is
    # prefixed with a host.
    source = f"{run.remote_dir}/" if addr is None else f"{addr}:{run.remote_dir}/"
    cmd = [
        str(HERE / "collect.py"),
        "--run-id", run.run_id, "--bench", run.bench,
        "--remote", source,
    ]
    if final:
        cmd.append("--final")
    subprocess.run([sys.executable, *cmd], check=False)


# Integrity: below this % of functioning iterations the run gets a warning
# even when no cheating was flagged.
INTEGRITY_PCT_WARN_BELOW = 70.0


def check_integrity(run: Run, state: dict) -> None:
    """Surface the run's SINGLE mid-run integrity verdict, once.

    The safe-mode inspector fires once (fleet runner, ~20% budget) and writes
    /artifacts/integrity.json, which `collect` rsyncs to ~/ce-runs/<run_id>/.
    Sweeps only look for that file until it lands; after it is processed the
    run is never re-checked. End-of-run integrity is NOT this path — the
    optimizer's own closing-loop inspector stamps it on the last iteration
    history record. Warning only — the verdict never cancels a run.
    """
    entry = state["runs"].setdefault(run.run_id, {})
    if entry.get("integrity") is not None:
        return  # the single mid-run verdict was already processed
    verdict_path = runs_dir() / run.run_id / "integrity.json"
    if not verdict_path.exists():
        return  # inspector hasn't fired/landed yet — try next sweep
    try:
        verdict = json.loads(verdict_path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    category = verdict.get("cheating_category")
    pct = verdict.get("integrity_pct")
    problems = []
    if category and category != "no_cheating":
        problems.append(f"cheating={category} ({verdict.get('cheating_rationale', '')[:200]})")
    if isinstance(pct, (int, float)) and pct < INTEGRITY_PCT_WARN_BELOW:
        problems.append(f"integrity_pct={pct} (< {INTEGRITY_PCT_WARN_BELOW})")
    entry["integrity"] = {"cheating_category": category, "integrity_pct": pct}
    if problems:
        entry["integrity_warned"] = True
        log(f"INTEGRITY WARNING {run.run_id}: " + "; ".join(problems))
    save_state(state)


# Egress audit: task-LM connections outside run_eval windows (see
# egress_audit.py). The gateway log lives on the host at this path; it is the
# whole host's log, so it is pulled once per host per sweep, not per run.
GATEWAY_LOG_REMOTE = "/data/gateway-logs/access.log"


def container_ip(host: str, container: str) -> str | None:
    """A run container's address on ce-internal — the `%>a` the gateway logs.

    Recorded at dispatch (and again at finish, before `docker rm`) because
    addresses are recycled: the next run on the host gets the same IP, so the
    audit must key on (ip, run window), not on the ip alone.
    """
    res = docker(
        host, "inspect", "-f",
        "{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}", container,
        check=False,
    )
    ips = res.stdout.split()
    return ips[0] if ips else None


def collect_gateway_log(matrix: dict, host: str) -> bool:
    """Pull the host's squid access.log to ~/ce-runs/gateway-<host>.access.log.

    Same file `fleet.py down` copies at teardown; pulling it every sweep means
    the audit runs while the campaign is live and the log survives a lost
    droplet. rsync appends cheaply: the file only grows.
    """
    dest = egress_audit.gateway_log_path(host)
    dest.parent.mkdir(parents=True, exist_ok=True)
    addr = host_addr(matrix, host)
    if addr is None:
        local_dir = os.environ.get("CE_GATEWAY_LOG_DIR")
        if not local_dir:
            return False
        source = str(Path(local_dir) / "access.log")
    else:
        source = f"{addr}:{GATEWAY_LOG_REMOTE}"
    res = subprocess.run(
        ["rsync", "-az", "--partial", "--timeout=120", source, str(dest)],
        capture_output=True, text=True, check=False,
    )
    if res.returncode != 0:
        log(f"gateway log pull failed for {host}: {res.stderr.strip()[:200]}")
    return res.returncode == 0


def check_egress(run: Run, state: dict, final: bool = False) -> None:
    """Join the run's task-LM traffic against its eval windows; warn on misses.

    Runs on every collect sweep (the staged job log and the pulled gateway log
    are both refreshed by then) and once more at finish. The result is stored
    on the state entry; `EGRESS WARNING <run_id>` is logged when violations
    exist and their count changed since the last sweep, so a live run with a
    single stray connection is reported once, not every ten minutes. Warning
    only — never cancels. Silently skips while inputs are missing (job log not
    yet staged, IP unknown, no gateway log).
    """
    entry = state["runs"].setdefault(run.run_id, {})
    ip = entry.get("container_ip")
    if not ip:
        ip = container_ip(run.host, run.container)
        if ip:
            entry["container_ip"] = ip
    gateway_log = egress_audit.gateway_log_path(run.host)
    job_log = egress_audit.resolve_job_log(runs_dir() / run.run_id)
    started = entry.get("started_at")
    if not (ip and started and job_log and gateway_log.exists()):
        return
    ended = entry.get("finished_at") if final else None
    try:
        result = egress_audit.audit(job_log, gateway_log, ip, started, ended or time.time())
    except OSError as exc:
        log(f"egress audit failed for {run.run_id}: {exc}")
        return
    previous = (entry.get("egress_audit") or {}).get("violation_lines")
    entry["egress_audit"] = {
        "verdict": result["verdict"],
        "violation_lines": result["violation_lines"],
        "violation_connections_est": result["violation_connections_est"],
        "violation_clusters": result["violation_clusters"][:10],
        "task_provider_lines": result["task_provider_lines"],
        "eval_windows": result["eval_windows"],
        "final": final,
        "checked_at": time.time(),
    }
    if result["violation_lines"] and result["violation_lines"] != previous:
        entry["egress_warned"] = True
        log(f"EGRESS WARNING {run.run_id}: {egress_audit.summarize(result)}")
    save_state(state)


# A running run whose iteration count and eval count have not moved for this
# long gets a STALLED warning (once per stall episode). Warning only.
STALL_WARN_SECONDS = 4 * 3600


def check_stalled(run: Run, state: dict, now: float | None = None) -> None:
    """Warn when a run's progress.json heartbeat shows no movement for 4h.

    The runner writes /artifacts/progress.json on every poll tick and `collect`
    rsyncs it to ~/ce-runs/<run_id>/. Movement is a change in
    (current_iteration, total_metric_calls); the first sighting starts the
    clock. Arm-agnostic, never cancels.
    """
    now = time.time() if now is None else now
    path = runs_dir() / run.run_id / "progress.json"
    if not path.exists():
        return
    try:
        progress = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return
    marker = [progress.get("current_iteration"), progress.get("total_metric_calls")]
    entry = state["runs"].setdefault(run.run_id, {})
    seen = entry.get("progress")
    if seen is None or seen.get("marker") != marker:
        entry["progress"] = {"marker": marker, "changed_at": now, "warned": False}
        save_state(state)
        return
    if not seen.get("warned") and now - seen["changed_at"] >= STALL_WARN_SECONDS:
        seen["warned"] = True
        hours = (now - seen["changed_at"]) / 3600
        log(f"STALLED WARNING {run.run_id}: no iteration/eval progress for "
            f"{hours:.1f}h (iter={marker[0]}, evals={marker[1]})")
        save_state(state)


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------

def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"runs": {}}


def save_state(state: dict) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2))


def reconcile(matrix: dict, runs: list[Run], state: dict) -> dict[str, str]:
    """Derive each run's status from the hosts, not from the cache.

    Returns {run_id: status} where status is queued|running|<terminal>.
    """
    by_host: dict[str, dict[str, str]] = {}
    for host in matrix["hosts"]:
        try:
            by_host[host] = live_containers(host)
        except (subprocess.CalledProcessError, OSError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            log(f"host {host} unreachable ({detail.strip()[:80]}) — treating as empty")
            by_host[host] = {}

    status: dict[str, str] = {}
    for run in runs:
        cached = state["runs"].get(run.run_id, {})
        container_state = by_host.get(run.host, {}).get(run.container)
        if container_state == "running":
            status[run.run_id] = "running"
        elif container_state in ("exited", "dead"):
            # Only a TERMINAL cached status may stand in for the container's:
            # that is the outcome `finish()` recorded after collecting. Any
            # other cached value is the stale "running" marker written at
            # dispatch; letting it shadow an exited container would keep
            # `finish()` from firing, leave the run uncollected, and keep it
            # counted against the host's slots, deadlocking the matrix.
            cached_status = cached.get("status")
            status[run.run_id] = (
                cached_status if cached_status in TERMINAL else "exited"
            )
        elif cached.get("status") in TERMINAL:
            status[run.run_id] = cached["status"]
        else:
            status[run.run_id] = "queued"
    return status


def finish(matrix: dict, run: Run, state: dict) -> None:
    """Pull final artifacts, read the real outcome, then remove the container."""
    collect(matrix, run, final=True)
    local = runs_dir() / run.run_id / "status.json"
    outcome = "exited"
    if local.exists():
        try:
            outcome = json.loads(local.read_text()).get("status") or "exited"
        except json.JSONDecodeError:
            pass
    entry = state["runs"].setdefault(run.run_id, {})
    # The address dies with the container; capture it before `rm` so the
    # final egress audit can still key on it.
    if not entry.get("container_ip"):
        entry["container_ip"] = container_ip(run.host, run.container)
    docker(run.host, "rm", "-f", run.container, check=False)
    entry.update({"status": outcome, "finished_at": time.time(), **asdict(run)})
    save_state(state)
    log(f"finished {run.run_id}: {outcome}")
    collect_gateway_log(matrix, run.host)
    check_egress(run, state, final=True)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_run(args) -> int:
    """The scheduler loop: keep every host's slots full until the matrix is done."""
    matrix = load_matrix()
    runs = build_runs(matrix)
    if args.arm:
        runs = [r for r in runs if r.arm in args.arm]
    if args.bench:
        runs = [r for r in runs if r.bench in args.bench]

    state = load_state()
    slots = matrix["slots_per_host"]
    bench_cap = matrix.get("bench_concurrency", 1)
    poll = matrix["defaults"]["poll_seconds"]
    rsync_every = matrix["defaults"]["rsync_seconds"]
    last_rsync = 0.0

    while True:
        status = reconcile(matrix, runs, state)

        for run in runs:
            if status[run.run_id] == "exited":
                finish(matrix, run, state)
                status[run.run_id] = state["runs"][run.run_id]["status"]

        if time.time() - last_rsync > rsync_every:
            live_hosts = {r.host for r in runs if status[r.run_id] == "running"}
            for host in live_hosts:
                collect_gateway_log(matrix, host)
            for run in runs:
                if status[run.run_id] == "running":
                    collect(matrix, run)
                    check_integrity(run, state)
                    check_stalled(run, state)
                    check_egress(run, state)
            last_rsync = time.time()

        if args.drain:
            log("drain mode: dispatching nothing new")
        else:
            for host in matrix["hosts"]:
                host_runs = [r for r in runs if r.host == host]
                busy = sum(1 for r in host_runs if status[r.run_id] == "running")
                # One queue per benchmark, advancing independently: a 24h hover
                # run must not wait on a 48h phantomwiki run in the same seed.
                # `bench_concurrency` sets how many runs each queue may have in
                # flight (default 1).
                bench_busy = Counter(
                    r.bench for r in host_runs if status[r.run_id] == "running"
                )
                for run in host_runs:
                    if busy >= slots:
                        break
                    if status[run.run_id] != "queued" or bench_busy[run.bench] >= bench_cap:
                        continue
                    if args.dry_run:
                        log(f"DRY RUN would dispatch {run.run_id} -> {host}")
                    else:
                        dispatch(matrix, run)
                        state["runs"].setdefault(run.run_id, {}).update(
                            {
                                "status": "running",
                                "started_at": time.time(),
                                "container_ip": container_ip(run.host, run.container),
                                **asdict(run),
                            }
                        )
                    busy += 1
                    bench_busy[run.bench] += 1
            save_state(state)

        remaining = [r for r in runs if status[r.run_id] in ("queued", "running")]
        if not remaining:
            log("matrix complete")
            return 0
        if args.once or args.dry_run:
            return 0
        time.sleep(poll)


def cmd_build(args) -> int:
    """Rebuild the images and refresh them on the run hosts, in one pass.

    The whole activation path -- `make bundles`, `make images push`, and the
    per-host pull -- with the two things a human doing it by hand gets wrong:

    * **In-flight runs.** `dispatch()` names an image by TAG, and a host that
      already holds that tag never re-pulls it. Rebuilding a tag mid-matrix
      therefore does not upgrade the fleet; it splits it, and which runs got
      which code depends on when each host last pulled. That is an invisible
      confound in a 36-run comparison, so this refuses to touch a tag while
      anything is running unless you say --force.
    * **The pull.** Forgetting it is silent: the fleet keeps running the old
      image and every symptom points at the code you just changed.
    """
    matrix = load_matrix()
    registry = args.registry or registry_of(matrix)
    tag = args.tag or matrix["image_tag"]
    benches = selected_benchmarks()

    busy = {host: running_runs(host) for host in matrix["hosts"]}
    running = {h: c for h, c in busy.items() if c}
    if running and not args.force:
        for host, names in running.items():
            log(f"{host}: {', '.join(names)}")
        log(f"refusing to rebuild tag '{tag}' while runs are in flight — "
            f"finish or cancel them, or pass --force to accept a fleet that "
            f"is split across two image versions.")
        return 1

    build = [sys.executable, str(HERE / "build.py")]
    common = ["--registry", registry, "--tag", tag, "--platform", args.platform]
    if args.host:
        common += ["--host", args.host]
    if args.no_retriever:
        common += ["--no-retriever"]

    log(f"building {', '.join(b.slug for b in benches)} @ {tag}"
        + (f" on {args.host}" if args.host else " locally"))
    for target in ("bundles", "images"):
        step = [*build, target, *common] + (["--push"] if target == "images" else [])
        if subprocess.run(step, cwd=HERE).returncode != 0:
            log(f"build step '{target}' failed — fleet not touched")
            return 1

    if args.no_pull:
        log("skipping host refresh (--no-pull); hosts still hold the old image")
        return 0

    failed = False
    for host in matrix["hosts"]:
        # Fan-out matrices: only warm what this host will actually run.
        allowed = matrix.get("host_benchmarks", {}).get(host)
        for b in benches:
            if allowed is not None and b.slug not in allowed:
                continue
            # Pull what dispatch will actually run, so a host warms the exact
            # bytes rather than whatever the tag points at by then.
            image = resolve_image(
                {"registry": registry, "image_tag": tag}, b
            )
            res = docker(host, "pull", image, check=False)
            if res.returncode == 0:
                log(f"{host}: pulled {image}")
            else:
                failed = True
                log(f"{host}: pull FAILED {image} — {res.stderr.strip()[:200]}")
    return 1 if failed else 0


def cmd_status(args) -> int:  # noqa: ARG001
    matrix = load_matrix()
    runs = build_runs(matrix)
    state = load_state()
    status = reconcile(matrix, runs, state)

    print(f"{'RUN':<38}{'HOST':<8}{'STATUS':<12}{'ITER':>6}{'EVALS':>8}  ELAPSED")
    for run in runs:
        st = status[run.run_id]
        iter_s = evals_s = "-"
        elapsed = ""
        local = runs_dir() / run.run_id / "status.json"
        if local.exists():
            try:
                blob = json.loads(local.read_text())
                iter_s = str(blob.get("current_iteration") or "-")
                evals_s = str(blob.get("total_metric_calls") or "-")
                elapsed = f"{(blob.get('elapsed_seconds') or 0) / 3600:.1f}h"
            except json.JSONDecodeError:
                pass
        elif st == "running":
            started = state["runs"].get(run.run_id, {}).get("started_at")
            if started:
                elapsed = f"{(time.time() - started) / 3600:.1f}h"
        print(f"{run.run_id:<38}{run.host:<8}{st:<12}{iter_s:>6}{evals_s:>8}  {elapsed}")
    return 0


def cmd_cancel(args) -> int:
    """Graceful by default: drop the sentinel and let the run finalize itself."""
    matrix = load_matrix()
    runs = {r.run_id: r for r in build_runs(matrix)}
    run = runs.get(args.run_id)
    if not run:
        log(f"unknown run: {args.run_id}")
        return 1

    if args.now:
        docker(run.host, "kill", run.container, check=False)
        log(f"KILLED {run.run_id} — the in-flight iteration and finalization are lost")
        return 0

    addr = host_addr(matrix, run.host)
    if addr is None:
        Path(run.remote_dir).mkdir(parents=True, exist_ok=True)
        (Path(run.remote_dir) / "CANCEL").touch()
    else:
        ssh(addr, f"touch {run.remote_dir}/CANCEL", check=False)
    log(f"cancel requested for {run.run_id}. The optimizer finishes the current "
        "iteration and runs finalization (seed + final test evals) before "
        "exiting, so this can take hours and still yields a scored result. "
        "Use --now only if you are willing to lose that.")
    return 0


def cmd_recycle(args) -> int:
    """Wipe a host back to a clean slate between rounds.

    Refuses while runs are live -- recycling under a running matrix would
    destroy in-flight work. This clears containers, images and collected run
    directories and restarts the gateway; for a genuinely fresh machine,
    destroy the droplet and re-run provision.sh against a new one.
    """
    matrix = load_matrix()
    runs = [r for r in build_runs(matrix) if r.host == args.host]
    if not runs:
        log(f"unknown host: {args.host}")
        return 1

    try:
        live = live_containers(args.host)
    except (subprocess.CalledProcessError, OSError) as exc:
        # Unlike the scheduler, recycling an unreachable host must NOT proceed:
        # "no containers found" would be indistinguishable from "cannot ask".
        detail = getattr(exc, "stderr", None) or str(exc)
        log(f"cannot reach {args.host}: {detail.strip()[:120]}")
        return 1
    running = [n for n, state in live.items()
               if state == "running" and n not in HOST_CONTAINERS]
    if running and not args.force:
        log(f"refusing: {len(running)} run(s) still live on {args.host}: "
            f"{', '.join(running)}. Wait, `ce cancel` them, or pass --force.")
        return 1

    for name in live:
        docker(args.host, "rm", "-f", name, check=False)
    addr = host_addr(matrix, args.host)
    # Only sweep run dirs whose artifacts are already on the controller machine.
    collected = {p.name for p in (runs_dir()).glob("*") if (p / "status.json").exists()}
    for run_id in collected:
        remove_run_dir(addr, f"{RUNS_ROOT}/{run_id}")
    docker(args.host, "system", "prune", "-f", check=False)
    if addr is None:
        subprocess.run(["docker", "compose", "-f",
                        str(HERE.parent / "runner" / "compose.host.yml"), "up", "-d"],
                       capture_output=True, check=False)
    else:
        ssh(addr, "cd /opt/ce && docker compose -f compose.host.yml up -d", check=False)
    log(f"{args.host} recycled ({len(live)} containers removed, "
        f"{len(collected)} collected run dirs swept)")
    return 0


def cmd_pull(args) -> int:
    matrix = load_matrix()
    runs = {r.run_id: r for r in build_runs(matrix)}
    run = runs.get(args.run_id)
    if not run:
        log(f"unknown run: {args.run_id}")
        return 1
    collect(matrix, run, final=True)
    return 0


def cmd_logs(args) -> int:
    matrix = load_matrix()
    runs = {r.run_id: r for r in build_runs(matrix)}
    run = runs.get(args.run_id)
    if not run:
        return 1
    cmd = ["docker", "--context", run.host, "logs"]
    if args.follow:
        cmd.append("-f")
    cmd.append(run.container)
    return subprocess.run(cmd).returncode


def main() -> int:
    p = argparse.ArgumentParser(prog="ce", description="CodeEvolver experiment fleet")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("start", help="run the scheduler loop until the matrix is done")
    r.add_argument("--arm", action="append", help="restrict to these arms")
    r.add_argument("--bench", action="append", help="restrict to these benchmarks")
    r.add_argument("--once", action="store_true", help="one scheduling pass, then exit")
    r.add_argument("--drain", action="store_true", help="finish in-flight, dispatch nothing")
    r.add_argument("--dry-run", action="store_true")
    r.set_defaults(func=cmd_run)

    bl = sub.add_parser("build", help="rebuild images, push, and refresh the hosts")
    bl.add_argument("--host", default=os.environ.get("CE_BUILD_HOST") or None,
                    help="docker context to build on (amd64 build droplet)")
    bl.add_argument("--platform", default="linux/amd64")
    bl.add_argument("--registry", default=None, help="default: matrix registry")
    bl.add_argument("--tag", default=None, help="default: matrix image_tag")
    bl.add_argument("--no-pull", action="store_true",
                    help="build and push only; leave the hosts on the old image")
    bl.add_argument("--force", action="store_true",
                    help="rebuild even while runs are in flight")
    bl.add_argument("--no-retriever", action="store_true",
                    help="passed to build.py: leave the ragqa-retriever sidecar "
                         "alone (compose.host.yml pulls ragqa-retriever:latest, so "
                         "a build under any other tag never feeds it anyway)")
    bl.set_defaults(func=cmd_build)

    s = sub.add_parser("status", help="table of every run in the matrix")
    s.set_defaults(func=cmd_status)

    c = sub.add_parser("cancel", help="cooperative cancel (or --now to kill)")
    c.add_argument("run_id")
    c.add_argument("--now", action="store_true", help="docker kill; loses finalization")
    c.set_defaults(func=cmd_cancel)

    rc = sub.add_parser("recycle", help="wipe a host clean between rounds")
    rc.add_argument("--host", required=True)
    rc.add_argument("--force", action="store_true", help="recycle even with live runs")
    rc.set_defaults(func=cmd_recycle)

    pl = sub.add_parser("pull", help="rsync artifacts + fold the bundle into the bare repo")
    pl.add_argument("run_id")
    pl.set_defaults(func=cmd_pull)

    lg = sub.add_parser("logs", help="tail a run's container logs")
    lg.add_argument("run_id")
    lg.add_argument("-f", "--follow", action="store_true")
    lg.set_defaults(func=cmd_logs)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
