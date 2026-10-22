#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

PID 1 of a runner container: one container == one optimization run.

`experiments/run.py` submits a job and returns immediately, and a run can
otherwise only be stopped by SIGKILL via the pidfile. Here the whole lifecycle is one
foreground process whose exit code IS the run result, so the controller can
treat a run as an ordinary container.

Sequence: clone a squashed seed bundle -> stage the prebuilt client venv ->
prove the egress fence is closed -> mongod -> uvicorn -> submit -> poll ->
await the optimizer's post-run agents -> export artifacts.

That second-to-last step is not bookkeeping: a terminal job status means the
RESULT is final, while the optimizer deliberately runs meta-reflection and the
run-integrity inspector AFTER posting it. See `await_optimizer_exit`.

Cancellation has two speeds, because a cooperative cancel is NOT fast: the
optimizer treats cancellation as budget exhaustion (`optimizer_loop.py`
`_budget_exhausted`), so it finishes the current iteration and then runs
finalization (seed + final test-set evals) before exiting. That can take hours,
which is exactly what you want -- a cancelled run still yields a scored,
finalized result instead of a truncated one.

  - CANCEL sentinel (`/artifacts/CANCEL`, touched by the controller over ssh):
    ask the optimizer to wind down, then wait for it as long as it takes. No
    signal, so no `docker stop` timeout racing the finalization evals.
  - SIGTERM (`docker stop`): same request, but Docker will SIGKILL us when its
    grace period expires, so export whatever exists as soon as the optimizer
    goes quiet.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# This container only ever runs local mode, and it already forces APP_MODE=local
# into every child process it spawns (the API, `experiments.run`, the integrity
# inspector). Setting it once here is strictly more consistent than setting it
# three times -- and it is REQUIRED before importing `experiments.run`, which
# resolves APP_MODE at import time and does `os.environ["PROD_API_KEY"]` when it
# is neither local nor dev. Without this, `effective_config()` below would
# KeyError and kill the run at startup.
os.environ.setdefault("APP_MODE", "local")

APP_ROOT = Path(os.environ.get("CE_APP_ROOT", "/opt/codeevolver"))
ARTIFACTS = Path(os.environ.get("CE_ARTIFACTS", "/artifacts"))
WORKSPACE = Path(os.environ.get("CE_WORKSPACE", "/workspace"))
SEED_DIR = Path(os.environ.get("CE_SEED_DIR", "/opt/seed"))
VENV_SRC_DIR = Path(os.environ.get("CE_VENV_DIR", "/opt/venvs"))
ASSET_SRC_DIR = Path(os.environ.get("CE_ASSET_DIR", "/opt/assets"))
API_URL = "http://127.0.0.1:8080"
CANCEL_SENTINEL = ARTIFACTS / "CANCEL"
POLL_INTERVAL = 30.0
# Where LocalLauncher writes `{job_id}.pid`, read by await_optimizer_exit.
# Mirrors `settings.local_log_root` (src/config.py) and therefore its env name,
# NOT the engine's `CODEEVOLVER_LOGS_DIR` — those name two different
# directories, and keying on the wrong one would just fail to find the pidfile
# and silently restore the teardown race this exists to close.
LOCAL_LOG_ROOT = Path(
    os.environ.get("LOCAL_LOG_ROOT", str(Path.home() / ".codeevolver" / "logs"))
)
# How long to let the optimizer keep working after it has reported a terminal
# status, so its post-run agents (meta-reflection, run-integrity) can finish
# before teardown. Generous because both are full agent sessions over the
# architect traces, but bounded: an advisory agent must never own the
# container, and on the fleet it would be holding a host slot.
POST_RUN_TIMEOUT = float(os.environ.get("CE_POST_RUN_TIMEOUT", "1800"))
POST_RUN_POLL_INTERVAL = 5.0
# Mid-run integrity check: fraction of the eval budget at which the safe-mode
# run-integrity inspector fires (once). "off" disables. The check runs in a
# background subprocess and NEVER blocks polling or stops the run — its
# verdict lands in /artifacts/integrity.json for the orchestrator to read.
INTEGRITY_AT = os.environ.get("CE_INTEGRITY_AT", "0.2")
INTEGRITY_JSON = ARTIFACTS / "integrity.json"
INTEGRITY_LOG = ARTIFACTS / "integrity_check.log"
# Mid-run heartbeat for the orchestrator's stall check (status.json is only
# written at the end of a run).
PROGRESS_JSON = ARTIFACTS / "progress.json"
# Root-owned mode-700 home for the sequestered holdout testset. Outside the
# workspace so neither the filesystem NOR git objects can reach it from an
# agent process (which the engine runs as the unprivileged
# CODEEVOLVER_AGENT_USER — see runner/Dockerfile.base and paths.agent_user()).
HOLDOUT_DIR = Path(os.environ.get("CE_HOLDOUT_DIR", "/opt/holdout"))
# Network alias of the squid gateway on the ce-internal network. The runner
# re-points its default route at it so transparent interception can see the
# traffic at all -- see configure_egress_route().
GATEWAY_HOST = os.environ.get("CE_GATEWAY_HOST", "gateway")

_sigterm_received = False


def log(msg: str) -> None:
    print(f"[entrypoint {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    """Run a command, echoing it, raising on failure unless check=False."""
    log("$ " + " ".join(cmd))
    return subprocess.run(cmd, check=kw.pop("check", True), **kw)


def _on_sigterm(signum, frame) -> None:  # noqa: ARG001
    global _sigterm_received
    _sigterm_received = True
    log("SIGTERM received — requesting cooperative cancel")


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------

def clone_workspace(bench: str) -> None:
    """Clone the squashed seed bundle into a pristine, remote-less workspace.

    The bundle carries exactly one orphan commit, so the original repository's
    history -- which still contains the committed holdout test sets -- simply
    does not exist here. `git cat-file` has nothing to find, which is the point:
    the deny-path hooks match on paths and Bash command text, not on blobs.
    """
    bundle = SEED_DIR / f"{bench}.bundle"
    if not bundle.exists():
        raise SystemExit(f"seed bundle missing: {bundle}")
    if WORKSPACE.exists():
        shutil.rmtree(WORKSPACE)
    sh(["git", "clone", "--branch", "seed", str(bundle), str(WORKSPACE)])
    sh(["git", "-C", str(WORKSPACE), "remote", "remove", "origin"], check=False)
    sh(["git", "-C", str(WORKSPACE), "checkout", "-B", "main"])

    depth = subprocess.run(
        ["git", "-C", str(WORKSPACE), "rev-list", "--count", "HEAD"],
        capture_output=True, text=True,
    ).stdout.strip()
    if depth != "1":
        raise SystemExit(f"seed workspace is not squashed (history depth={depth})")
    log(f"workspace cloned from {bundle.name}, history depth 1, no remotes")


# Workspace-relative paths placed by `stage_data_assets`. They come from the
# image, not from the run, so `export_workspace` leaves them out of the archive:
# re-copying a multi-hundred-MB input into every run's artifacts would dwarf
# everything the run actually produced.
_STAGED_ASSETS: set[str] = set()


def stage_data_assets(bench: str) -> None:
    """Place the image's data assets into the freshly cloned workspace.

    These are files the client repo gitignores because they are far too large to
    commit, so `git archive` cannot put them in the seed bundle. They ride in the
    image instead (see Dockerfile.bench) and land here at the same repo-relative
    path the client tree expects. The seed's own .gitignore still covers them, so
    the run's `git add -A` (no --force) never pulls them into a candidate commit.

    A committed file of the same name always wins: staging never overwrites what
    the seed tree already carries.
    """
    src_root = ASSET_SRC_DIR / bench
    if not src_root.is_dir():
        return
    for src in sorted(src_root.rglob("*")):
        if not src.is_file() or src.name == ".keep":
            continue
        rel = src.relative_to(src_root)
        dst = WORKSPACE / rel
        if dst.exists():
            log(f"data asset {rel} already in the seed tree; leaving it alone")
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        _STAGED_ASSETS.add(rel.as_posix())
        log(f"staged data asset {rel} ({src.stat().st_size // 1024 // 1024} MiB)")


def stage_venv(bench: str) -> None:
    """Copy the image's prebuilt client venv to `<workspace>/env`.

    `LocalEnvironment._ensure_venv` reuses an existing venv and only creates one
    when absent, so pre-staging turns a multi-GB torch install into a local copy.
    """
    src = VENV_SRC_DIR / bench
    if not src.exists():
        log(f"WARNING: no prebuilt venv at {src}; LocalEnvironment will build one")
        return
    dst = WORKSPACE / "env"
    log(f"staging prebuilt venv {src} -> {dst}")
    shutil.copytree(src, dst, symlinks=True)


def open_workspace_permissions() -> None:
    """Make workspace + artifacts writable by the unprivileged agent user.

    `main()` sets umask(0) so everything created from here on is world-writable
    already; this sweep fixes what predates the run — the cloned tree and the
    staged venv keep their pre-umask modes, and the /artifacts bind-mount root
    comes from the host. One-run-per-container, so world-writable is fine; the
    only file that must stay unreadable is the sequestered holdout, which
    `sequester_holdout` locks down explicitly afterwards.
    """
    for root in (WORKSPACE, ARTIFACTS):
        sh(["chmod", "-R", "a+rwX", str(root)])


def trust_workspace_for_agent() -> None:
    """Let the unprivileged agent user run git in the root-owned workspace.

    `open_workspace_permissions` fixes MODE, but git's `safe.directory` check
    compares the repository OWNER (`st_uid`) with the calling uid, so every
    git command the agent runs as `ce-agent` fails with `fatal: detected
    dubious ownership in repository at '/workspace'`, and the agent's own fix
    (`git config --global --add safe.directory`) would land in the
    agent-writable `$HOME/.gitconfig` (`HOME=/artifacts`). The trust belongs
    in the root-owned system config (`/etc/gitconfig`, read for every uid,
    not writable by the agent). Only the workspace is trusted; the sequestered
    holdout under `/opt/holdout` is not a git repository and keeps its
    root-only modes — `sequester_holdout` runs after this.
    """
    sh(["git", "config", "--system", "--add", "safe.directory", str(WORKSPACE)])


def effective_config(program: str, config_name: str, extra_sets: list[str]) -> dict:
    """The config exactly as `submit()` will submit it.

    One resolution instead of three hand-rolled partial ones. Reuses
    `experiments.run.resolve_config`, which applies `--set` overrides BEFORE
    validating required keys, and is handed the overrides in submit()'s own
    order so the result is what the run actually receives -- not an
    approximation of it.

    This matters most for the eval budget: reading the submitted config means
    the mid-run integrity trigger's budget CANNOT disagree with the run's
    budget, including a `max_metric_calls` that only input.py sets.

    Raises SystemExit (from resolve_config) on a malformed or incomplete
    config -- the same failure submit() would hit seconds later, just earlier
    and before any provider spend.
    """
    sys.path.insert(0, str(APP_ROOT))
    from experiments.run import resolve_config  # noqa: PLC0415

    module = importlib.import_module(f"experiments.{program}.input")
    # Mirrors submit()'s argv: repo_url first, extra_sets after, last wins.
    overrides = [f"repo_url={WORKSPACE}", *extra_sets]
    _, config = resolve_config(module, program, config_name, overrides)
    return config


def sequester_holdout(config: dict) -> list[str]:
    """Move the held-out testset outside the workspace and out of git objects.

    File permissions alone cannot fence a git-tracked testset — `git show
    seed:<path>` reads the blob straight from .git/objects, and the deny-path
    hooks match command text, not blobs. So: move the file to root-owned
    /opt/holdout (dir 700, file 600), amend it out of the single seed commit,
    re-point the leftover `seed` branch, expire reflogs, gc the unreachable
    blob, and VERIFY it is gone. Agent subprocesses run as `ce-agent`
    (CODEEVOLVER_AGENT_USER), so the kernel enforces the fence; the root
    optimizer driver still reads the file for the finalization test evals.

    Returns the `--set testset_path=...` override for submit(). A configured
    testset that is missing on disk aborts the run — a research run whose
    holdout silently vanished is worse than a failed one.
    """
    rel = config.get("testset_path") or None
    if not rel:
        log("holdout: no testset_path configured — nothing to sequester")
        return []
    src = Path(rel) if os.path.isabs(rel) else WORKSPACE / rel
    if not src.exists():
        raise SystemExit(f"holdout: configured testset missing: {src}")

    HOLDOUT_DIR.mkdir(exist_ok=True)
    os.chmod(HOLDOUT_DIR, 0o700)
    if os.path.isabs(rel):
        os.chmod(src, 0o600)
        log(f"holdout: testset already outside the workspace ({src}) — chmod 600")
        return []

    blob = subprocess.run(
        ["git", "-C", str(WORKSPACE), "rev-parse", f"HEAD:{rel}"],
        capture_output=True, text=True,
    ).stdout.strip()
    dst = HOLDOUT_DIR / src.name
    shutil.move(str(src), str(dst))
    os.chmod(dst, 0o600)

    if blob:  # tracked in the seed commit — evict the blob from git entirely
        git = ["git", "-C", str(WORKSPACE)]
        sh([*git, "rm", "-q", "--cached", rel])
        sh([*git, "-c", "user.email=runner@codeevolver.local",
            "-c", "user.name=codeevolver-runner",
            "commit", "--amend", "--no-edit", "-q"])
        # The bundle's `seed` branch still points at the pre-amend commit.
        sh([*git, "branch", "-f", "seed", "main"])
        sh([*git, "reflog", "expire", "--expire=now", "--all"])
        sh([*git, "gc", "--prune=now", "--quiet"], check=False)
        gone = subprocess.run(
            [*git, "cat-file", "-e", blob], capture_output=True,
        ).returncode != 0
        if not gone:
            raise SystemExit(
                "holdout: testset blob still reachable in git objects after "
                "amend+gc — aborting before any spend"
            )
    log(f"holdout: {rel} -> {dst} (root-owned 600, evicted from git: {bool(blob)})")
    return [f"testset_path={dst}"]


def configure_egress_route() -> None:
    """Point the default route at the squid gateway container.

    Without this the SNI allowlist is not a filter, it is a wall. Runners sit on
    an `internal: true` docker network, where the default route Docker installs
    points at the docker bridge -- and Docker's isolation rules drop anything
    leaving it. Squid's transparent REDIRECT rules live in the GATEWAY's network
    namespace, so they only ever fire on packets actually delivered to that
    container. Nothing routed runner traffic there, so every provider was
    unreachable and the egress canary aborted the run before it started.

    Resolving the alias rather than hardcoding the address keeps this working if
    the ce-internal subnet is renumbered; compose pins the gateway's IP so the
    answer is stable across restarts.

    Requires CAP_NET_ADMIN, which the orchestrator matrices add back after --cap-drop=ALL.
    Failure is logged and NOT raised: the canary immediately afterwards is the
    thing that decides whether the run may proceed, and it fails closed.
    """
    try:
        gateway_ip = socket.gethostbyname(GATEWAY_HOST)
    except OSError as exc:
        log(f"egress route: cannot resolve {GATEWAY_HOST!r} ({exc}) — "
            "leaving routing untouched")
        return
    res = subprocess.run(
        ["ip", "route", "replace", "default", "via", gateway_ip],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log(f"egress route: `ip route replace default via {gateway_ip}` FAILED "
            f"({res.stderr.strip()}) — is CAP_NET_ADMIN granted?")
        return
    log(f"egress route: default via {gateway_ip} ({GATEWAY_HOST})")


def run_preflight() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import preflight  # noqa: PLC0415 - deliberately late, after sys.path setup

    report = preflight.check()
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "egress.json").write_text(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit(
            "egress preflight FAILED — aborting before any provider spend. "
            f"leaked={report['leaked_hosts']} "
            f"unreachable={report['unreachable_allowed_hosts']} "
            f"agent_user={report.get('agent_user')} "
            f"agent_git={report.get('agent_git')} "
            f"ripgrep={report.get('ripgrep')}"
        )
    log("egress preflight PASS (fence closed, providers reachable, agent user + git OK)")


def start_mongod() -> subprocess.Popen:
    dbpath = Path(os.environ.get("CE_MONGO_DBPATH", "/var/lib/mongodb"))
    dbpath.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(
        ["mongod", "--dbpath", str(dbpath), "--bind_ip", "127.0.0.1",
         "--quiet", "--logpath", str(ARTIFACTS / "mongod.log")],
    )
    _await(lambda: _tcp_open(27017), "mongod", timeout=90)
    return proc


def start_api(arm: str) -> subprocess.Popen:
    """Boot the local API. STRATEGY is process-level, which suits one-run-per-container."""
    env = {
        **os.environ,
        "APP_MODE": "local",
        "SANDBOX_MODE": "local",
        "STRATEGY": arm,
        "PYTHONPATH": str(APP_ROOT),
    }
    logfile = (ARTIFACTS / "api.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app",
         "--host", "127.0.0.1", "--port", "8080"],
        cwd=str(APP_ROOT), env=env, stdout=logfile, stderr=subprocess.STDOUT,
    )
    _await(lambda: _http_ok(f"{API_URL}/health"), "api", timeout=180)
    return proc


def _tcp_open(port: int) -> bool:
    import socket
    try:
        with socket.create_connection(("127.0.0.1", port), 2):
            return True
    except OSError:
        return False


def _http_ok(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _await(predicate, name: str, timeout: float) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            log(f"{name} ready")
            return
        time.sleep(2)
    raise SystemExit(f"{name} did not come up within {timeout:.0f}s")


# --------------------------------------------------------------------------
# run
# --------------------------------------------------------------------------

def submit(program: str, config_name: str, extra_sets: list[str]) -> str:
    """Submit via experiments/run.py. `program` is the experiments/ DIRECTORY."""
    cmd = [sys.executable, "-m", "experiments.run",
           "--program", program, "--config", config_name,
           "--set", f"repo_url={WORKSPACE}", "--json"]
    for item in extra_sets:
        cmd += ["--set", item]
    log("$ " + " ".join(cmd))
    proc = subprocess.run(
        cmd, cwd=str(APP_ROOT), capture_output=True, text=True,
        env={**os.environ, "APP_MODE": "local", "PYTHONPATH": str(APP_ROOT)},
    )
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(f"job submission failed (rc={proc.returncode})")
    payload = json.loads(proc.stdout.strip().splitlines()[-1])
    job_id = payload.get("job_id")
    if not job_id:
        raise SystemExit(f"job submission returned no job_id: {payload}")
    log(f"submitted job {job_id}")
    return job_id


def request_cancel(job_id: str) -> None:
    req = urllib.request.Request(
        f"{API_URL}/job/{job_id}/cancel", method="POST",
        headers={"Authorization": "Bearer local", "Authorization-Type": "api_key"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            log(f"cancel requested: {resp.read().decode()}")
    except Exception as exc:  # noqa: BLE001
        log(f"cancel request failed (non-fatal): {exc}")


def _integrity_threshold() -> float | None:
    """Parse CE_INTEGRITY_AT; None disables the mid-run check."""
    if INTEGRITY_AT.strip().lower() in ("off", "0", "none", ""):
        return None
    try:
        frac = float(INTEGRITY_AT)
    except ValueError:
        log(f"CE_INTEGRITY_AT={INTEGRITY_AT!r} unparseable — integrity check off")
        return None
    return frac if frac > 0 else None


def maybe_launch_integrity_check(
    job_id: str, job: dict, max_metric_calls: int | None
) -> subprocess.Popen | str | None:
    """Fire the safe-mode integrity CLI once past the budget threshold.

    Returns the Popen when launched, the string "disabled" when the check
    cannot run for this job (caller stops asking), or None to try again next
    poll tick. Non-blocking and advisory: exit code and output are ignored
    here; the verdict is read from /artifacts/integrity.json by write_status
    and the orchestrator. Never stops the runner.
    """
    frac = _integrity_threshold()
    if frac is None:
        return "disabled"
    if not max_metric_calls:
        log("integrity check disabled: max_metric_calls unknown")
        return "disabled"
    calls = job.get("total_metric_calls") or 0
    if calls < frac * max_metric_calls:
        return None
    log(
        f"integrity check: {calls}/{max_metric_calls} evals >= {frac:.0%} — "
        "launching safe-mode inspector in the background"
    )
    cmd = [
        sys.executable, str(APP_ROOT / "scripts" / "inspect_run_integrity.py"),
        "--job-id", job_id,
        "--workspace", str(WORKSPACE),
        "--output", str(INTEGRITY_JSON),
    ]
    # Output goes to /artifacts, not DEVNULL, so a missing verdict leaves a
    # record of why. The child inherits the fd; closing our copy is fine.
    try:
        log_fh = open(INTEGRITY_LOG, "ab")
    except OSError as exc:
        log(f"integrity check: cannot open {INTEGRITY_LOG} ({exc}); output discarded")
        log_fh = subprocess.DEVNULL
    try:
        return subprocess.Popen(
            cmd, cwd=str(APP_ROOT),
            env={**os.environ, "APP_MODE": "local", "PYTHONPATH": str(APP_ROOT)},
            stdout=log_fh, stderr=subprocess.STDOUT,
        )
    finally:
        if log_fh is not subprocess.DEVNULL:
            log_fh.close()


def reap_integrity_check(proc: subprocess.Popen) -> bool:
    """Log the mid-run inspector's exit once it finishes. True when it has."""
    if proc.poll() is None:
        return False
    verdict = "present" if INTEGRITY_JSON.exists() else "ABSENT"
    log(
        f"integrity check: inspector exited rc={proc.returncode}, verdict "
        f"{verdict} (output: {INTEGRITY_LOG})"
    )
    return True


def write_progress(job: dict) -> None:
    """Best-effort heartbeat: the fields the orchestrator's stall check reads."""
    try:
        PROGRESS_JSON.write_text(json.dumps({
            "status": job.get("status"),
            "current_iteration": job.get("current_iteration"),
            "total_metric_calls": job.get("total_metric_calls"),
            "ts": time.time(),
        }))
    except OSError as exc:
        log(f"progress.json write failed (ignored): {exc}")


def poll(job_id: str, max_metric_calls: int | None = None) -> dict:
    """Poll to a terminal status, honouring the CANCEL sentinel and SIGTERM."""
    cancel_sent = False
    integrity_proc: subprocess.Popen | str | None = None
    while True:
        try:
            with urllib.request.urlopen(
                urllib.request.Request(
                    f"{API_URL}/job/{job_id}",
                    headers={"Authorization": "Bearer local",
                             "Authorization-Type": "api_key"},
                ), timeout=30,
            ) as resp:
                job = json.loads(resp.read().decode())
            if job.get("status") in ("completed", "failed", "cancelled"):
                log(f"job {job_id} terminal: {job['status']}")
                return job
            log(f"job {job_id}: {job.get('status')} "
                f"iter={job.get('current_iteration')} "
                f"evals={job.get('total_metric_calls')}")
            write_progress(job)
            if integrity_proc is None:
                integrity_proc = maybe_launch_integrity_check(
                    job_id, job, max_metric_calls
                )
            elif isinstance(integrity_proc, subprocess.Popen):
                if reap_integrity_check(integrity_proc):
                    integrity_proc = "done"
        except Exception as exc:  # noqa: BLE001 - a transient API blip is not a run failure
            log(f"poll error (retrying): {exc}")

        if not cancel_sent and (_sigterm_received or CANCEL_SENTINEL.exists()):
            reason = "SIGTERM" if _sigterm_received else "CANCEL sentinel"
            log(f"{reason} -> cooperative cancel; the optimizer will still "
                "finish the current iteration and run finalization")
            request_cancel(job_id)
            cancel_sent = True

        time.sleep(POLL_INTERVAL)


# --------------------------------------------------------------------------
# settings banner
# --------------------------------------------------------------------------

# Names whose VALUE must never reach a log. The banner exists to make silent
# misconfiguration visible, and it is rsynced to the controller -- printing a
# provider key would trade one problem for a worse one.
_SECRET_NAME_PARTS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
PROVIDER_KEYS = (
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "DEEPINFRA_API_KEY", "GMI_API_KEY",
    # DeepSeek's first-party API: a route of EVERY benchmark's DeepSeek-V4-Flash
    # provider chain (appworld and alfworld included).
    "DEEPSEEK_API_KEY",
    # Alternative spelling of the GMI key: the arcagi/paperreview solvers read
    # `GMI_CLOUD_API_KEY or GMI_API_KEY`, so the banner has to fingerprint
    # whichever one arrived or "no GMI key" reads as "GMI_API_KEY MISSING" on a
    # run that was in fact configured correctly.
    "GMI_CLOUD_API_KEY",
)


def _is_secret(name: str) -> bool:
    return any(part in name.upper() for part in _SECRET_NAME_PARTS)


def _render_env_value(name: str, value: str | None) -> str:
    """Verbatim for ordinary vars; a fingerprint for anything secret-shaped.

    The fingerprint answers the question an operator actually has -- "did all
    five hosts get the SAME key?" -- without putting the key in the log.
    """
    if value in (None, ""):
        return "MISSING"
    if _is_secret(name):
        digest = hashlib.sha256(value.encode()).hexdigest()[:8]
        return f"present (sha256:{digest})"
    return value


def _file_sha256(path: str | None) -> str | None:
    """sha256 of a dataset file, or None when unset/unreadable (never raises)."""
    if not path:
        return None
    full = Path(path) if os.path.isabs(path) else WORKSPACE / path
    try:
        h = hashlib.sha256()
        with open(full, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def build_settings(
    bench: str, program: str, arm: str, config_name: str,
    config: dict, extra_sets: list[str],
) -> dict:
    """The run's control surface, in one structure.

    Ordered by how badly a wrong value hurts. The point of the exercise is the
    `controls` block: every one of those can be OFF without erroring, and
    before this there was no way to tell from a finished run which were on.
    """
    budget = _max_metric_calls(config)
    # LM_* joins the CE_/CODEEVOLVER_ prefixes because it decides which
    # provider served every task-LM call: a run whose numbers were produced on
    # a different primary than its neighbours is only distinguishable from the
    # provenance record, and these are never secrets (the keys are separate
    # vars, fingerprinted under "provider_keys").
    env_names = sorted(
        n for n in os.environ
        if n.startswith(("CE_", "CODEEVOLVER_", "LM_")) and not _is_secret(n)
    )
    return {
        "run_id": os.environ.get("CE_RUN_ID", ""),
        "bench": bench,
        "program": f"experiments/{program}",
        "arm": arm,
        "seed": os.environ.get("CE_SEED", ""),
        "config_name": config_name,
        "research_mode": research_mode(),
        "controls": {
            "clean_slate_inspector": config.get("enable_clean_slate_inspector"),
            "run_integrity_agent_closing": config.get("enable_run_integrity_agent"),
            "meta_reflection": config.get("enable_meta_reflection"),
            "integrity_check_midrun": (
                "disabled (no budget)" if budget is None
                else f"at {INTEGRITY_AT} of {budget}"
            ),
            "holdout_testset": config.get("testset_path"),
            "holdout_agent_user": os.environ.get("CODEEVOLVER_AGENT_USER") or "(none)",
            "egress_gateway": GATEWAY_HOST,
            "additional_deny_paths": config.get("additional_deny_paths") or [],
        },
        "models": {
            "coding_harness": config.get("coding_harness"),
            "coding_harness_model": config.get("coding_harness_model"),
            "reflection_lm": config.get("reflection_lm"),
        },
        "budget": {
            "max_metric_calls": config.get("max_metric_calls"),
            "subsample_size": config.get("subsample_size"),
            "comparison_set_size": config.get("comparison_set_size"),
            "variance_estimate_samples": config.get("variance_estimate_samples"),
            "num_threads": config.get("num_threads"),
            "seed": config.get("seed"),
        },
        "overrides": list(extra_sets),
        # Row ids are positional indices into these files (offline
        # best-of-population audit); the hashes say which files a run's rows refer to. The
        # testset path is absolute after sequestration; relative paths
        # resolve against the workspace.
        "dataset_hashes": {
            "trainset": _file_sha256(config.get("trainset_path")),
            "testset": _file_sha256(config.get("testset_path")),
        },
        "env": {n: os.environ[n] for n in env_names},
        "provider_keys": {
            n: _render_env_value(n, os.environ.get(n)) for n in PROVIDER_KEYS
        },
    }


def log_settings(
    bench: str, program: str, arm: str, config_name: str,
    config: dict, extra_sets: list[str],
) -> dict:
    """Print the settings banner and persist it as an artifact.

    Both, deliberately: stdout so `ce logs` shows it at the top of a run, and
    /artifacts/settings.json so "which of the 36 runs had clean-slate on?" is
    one grep across collected runs rather than 36 log scrolls.
    """
    settings = build_settings(bench, program, arm, config_name, config, extra_sets)
    lines = ["===== run settings ====="]
    for section in ("controls", "models", "budget"):
        lines.append(f"  [{section}]")
        for key, value in settings[section].items():
            lines.append(f"    {key} = {value}")
    lines.append("  [provider keys]")
    for key, value in settings["provider_keys"].items():
        lines.append(f"    {key} = {value}")
    lines.append("  [env]")
    for key, value in settings["env"].items():
        lines.append(f"    {key} = {value}")
    if settings["overrides"]:
        lines.append(f"  [overrides] {'; '.join(settings['overrides'])}")
    lines.append("========================")
    log("\n".join(lines))
    try:
        (ARTIFACTS / "settings.json").write_text(json.dumps(settings, indent=2))
    except OSError as exc:
        log(f"settings.json write failed (non-fatal): {exc}")
    return settings


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------

def export_candidates(job_id: str) -> int:
    """Write one line per closed iteration for the later per-candidate analysis.

    Sourced from the JSONL mirror rather than Mongo so it still works when a run
    died before its terminal write. This is what the analysis harness enumerates
    instead of walking git.
    """
    sys.path.insert(0, str(APP_ROOT))
    from scripts.export_history import reconstruct  # noqa: PLC0415

    history_root = Path.home() / ".codeevolver" / "history"
    try:
        data = reconstruct(job_id, history_root)
    except Exception as exc:  # noqa: BLE001
        log(f"candidate export failed (non-fatal): {exc}")
        return 0

    (ARTIFACTS / "history.json").write_text(json.dumps(data, indent=2))

    meta = {
        "run_id": os.environ.get("CE_RUN_ID", ""),
        "bench": os.environ.get("CE_BENCH", ""),
        "arm": os.environ.get("CE_ARM", ""),
        "seed": os.environ.get("CE_SEED", ""),
        "job_id": job_id,
    }
    rows = 0
    with (ARTIFACTS / "candidates.jsonl").open("w") as fh:
        for idx, rec in sorted(
            (data.get("iteration_history") or {}).items(),
            key=lambda kv: int(kv[0]),
        ):
            fh.write(json.dumps({
                **meta,
                "iteration": int(idx),
                "branch": rec.get("git_branch"),
                "sha": rec.get("close_sha"),
                "close_sha": rec.get("close_sha"),
                "parent_branch": rec.get("parent_branch"),
                "parent_sha": rec.get("parent_sha"),
                # Progress readings — best train eval on the baseline's
                # subsample and best comparison eval on the branch. Neither
                # is a score of `close_sha` (see records.CloseRecord).
                "best_subsample_score": rec.get("best_subsample_score"),
                "best_subsample_sha": rec.get("best_subsample_sha"),
                "comparison_best_score": rec.get("comparison_best_score"),
                "comparison_state": rec.get("comparison_state"),
                "parent_subsample_score": rec.get("parent_subsample_score"),
                "test_score": rec.get("test_score"),
                "test_scores": rec.get("test_scores"),
                "improvement_mode": rec.get("improvement_mode"),
                "is_primary": rec.get("primary_branch_after") == rec.get("git_branch"),
                "skipped": rec.get("skipped"),
                "constraint_violation": rec.get("constraint_violation"),
                "closed_by_fallback": rec.get("closed_by_fallback"),
            }) + "\n")
            rows += 1
    log(f"wrote candidates.jsonl ({rows} iterations)")
    return rows


def export_bundle() -> None:
    """One self-contained file carrying every evolved branch.

    Fetched on the Mac mini into a per-benchmark bare repo under
    `refs/runs/<run_id>/*`; every run shares the same squashed seed commit, so
    git dedupes objects across all runs of a benchmark.
    """
    out = ARTIFACTS / "repo.bundle"
    res = subprocess.run(
        ["git", "-C", str(WORKSPACE), "bundle", "create", str(out), "--all"],
        capture_output=True, text=True,
    )
    if res.returncode != 0:
        log(f"bundle export failed (non-fatal): {res.stderr.strip()}")
        return
    branches = subprocess.run(
        ["git", "-C", str(WORKSPACE), "branch", "--list", "codeevolver-*"],
        capture_output=True, text=True,
    ).stdout.split()
    log(f"exported repo.bundle ({out.stat().st_size // 1024} KiB, "
        f"{len(branches)} codeevolver-* branches)")


# Directories left out of the workspace snapshot below.
#   env/, .venv/   the staged client venv -- multi-GB torch/sentence-transformers,
#                  and rebuildable from requirements.txt.
#   .git/          repo.bundle already carries every object, and collect.py folds
#                  bundles into a SHARED per-benchmark bare repo where they dedupe
#                  across runs; 36 undeduped .git copies would be the single
#                  largest waste in the scheme.
#   caches         nothing to reproduce. The hover/ragqa pipelines already disable
#                  DSPy's disk cache, but an evolved program could re-enable it and
#                  25k cached completions would swamp the archive.
WORKSPACE_ARCHIVE_EXCLUDES = (
    "env", ".venv", ".git",
    "__pycache__", ".dspy_cache", ".pytest_cache", "node_modules",
)


def export_workspace(job: dict) -> None:
    """Snapshot the FINAL workspace, gitignored files included.

    git is the per-candidate record and stays that way -- the
    discovery/coverage/retention analysis walks repo.bundle + candidates.jsonl
    and assumes the program pipeline is committed. This covers the one thing
    git structurally cannot: commits go through `git add -A` with no --force,
    so anything the run wrote to a gitignored path is never committed, never in
    the bundle, and dies with the container. A program that builds and then
    depends on such a file (the phantom-wiki `_corpus_index.pkl` shape) could
    not otherwise reproduce its own scores.

    Entirely non-fatal: this is a backup, and it must never fail a run that
    already produced results.
    """
    manifest: dict = {
        "excludes": list(WORKSPACE_ARCHIVE_EXCLUDES),
        "staged_assets": sorted(_STAGED_ASSETS),
    }
    try:
        final = (job or {}).get("final_candidate") or {}
        wanted = final.get("git_branch")
        manifest["intended_branch"] = wanted
        # HEAD is only USUALLY the final primary: finalization scores the seed
        # first and the final primary second, so it lands there -- unless the
        # final primary IS the seed, or that scoring raised. Ask explicitly.
        if wanted:
            res = subprocess.run(
                ["git", "-C", str(WORKSPACE), "checkout", wanted],
                capture_output=True, text=True,
            )
            manifest["checkout_ok"] = res.returncode == 0
            if res.returncode != 0:
                log(f"workspace export: checkout of {wanted} failed "
                    f"(archiving HEAD as-is): {res.stderr.strip()[:200]}")
        manifest["actual_branch"] = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()
        manifest["head_sha"] = subprocess.run(
            ["git", "-C", str(WORKSPACE), "rev-parse", "HEAD"],
            capture_output=True, text=True,
        ).stdout.strip()

        # The direct answer to "did this run leave anything outside git?".
        status = subprocess.run(
            ["git", "-C", str(WORKSPACE), "status", "--porcelain", "--ignored"],
            capture_output=True, text=True,
        ).stdout.splitlines()
        untracked = [
            line[3:] for line in status
            if line[:2] in ("??", "!!")
            and line[3:].split("/", 1)[0] not in WORKSPACE_ARCHIVE_EXCLUDES
            and line[3:] not in _STAGED_ASSETS
        ]
        # Capped: an evolved program can generate thousands of scratch files,
        # and a manifest nobody can read is not a manifest. The count is the
        # part you act on; the sample tells you what kind of thing they are.
        manifest["untracked_count"] = len(untracked)
        manifest["untracked_paths"] = sorted(untracked)[:200]

        out = ARTIFACTS / "workspace.tar.gz"
        excludes = set(WORKSPACE_ARCHIVE_EXCLUDES)

        def _filter(info: tarfile.TarInfo) -> tarfile.TarInfo | None:
            name = Path(info.name)
            if any(p in excludes for p in name.parts):
                return None
            # Staged inputs came from the image and are identical in every run.
            return None if name.as_posix() in _STAGED_ASSETS else info

        with tarfile.open(out, "w:gz") as tar:
            tar.add(str(WORKSPACE), arcname=".", filter=_filter)
        manifest["archive_bytes"] = out.stat().st_size
        log(f"exported workspace.tar.gz ({out.stat().st_size // 1024} KiB, "
            f"branch={manifest.get('actual_branch')}, "
            f"{len(untracked)} path(s) outside git)")
    except Exception as exc:  # noqa: BLE001
        manifest["error"] = str(exc)
        log(f"workspace export failed (non-fatal): {exc}")
    try:
        (ARTIFACTS / "workspace_manifest.json").write_text(
            json.dumps(manifest, indent=2)
        )
    except OSError as exc:
        log(f"workspace manifest write failed (non-fatal): {exc}")


def read_integrity() -> dict | None:
    """The mid-run integrity verdict, if the safe-mode inspector produced one."""
    try:
        return json.loads(INTEGRITY_JSON.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _clean_slate_status() -> str | None:
    """The clean-slate verdict, for status.json.

    Only the status: the REPORT needs no exporting. HOME=/artifacts in this
    image (runner/Dockerfile.base), so `paths.clean_slate_report_path()`
    already resolves inside the host bind mount and collect.py already merges
    `logs/` to the controller. Copying it a second time would just make two
    copies drift.

    The status comes from the seed record (iteration 0) in the history mirror,
    which is written before the seed step can abort.
    """
    try:
        history = json.loads((ARTIFACTS / "history.json").read_text())
        seed_record = (history.get("iteration_history") or {}).get("0") or {}
        return seed_record.get("clean_slate_status")
    except (OSError, json.JSONDecodeError, AttributeError):
        return None


def write_status(job_id: str, job: dict, started: float, exit_reason: str) -> None:
    (ARTIFACTS / "status.json").write_text(json.dumps({
        "run_id": os.environ.get("CE_RUN_ID", ""),
        "bench": os.environ.get("CE_BENCH", ""),
        "arm": os.environ.get("CE_ARM", ""),
        "seed": os.environ.get("CE_SEED", ""),
        "job_id": job_id,
        "status": job.get("status"),
        "exit_reason": exit_reason,
        "elapsed_seconds": round(time.time() - started, 1),
        "current_iteration": job.get("current_iteration"),
        "total_metric_calls": job.get("total_metric_calls"),
        "num_branches": job.get("num_branches"),
        "final_candidate": job.get("final_candidate"),
        "seed_train_score": job.get("seed_train_score"),
        "final_train_score": job.get("final_train_score"),
        "seed_test_score": job.get("seed_test_score"),
        "final_test_score": job.get("final_test_score"),
        "seed_test_scores": job.get("seed_test_scores"),
        "final_test_scores": job.get("final_test_scores"),
        "error": job.get("error"),
        "integrity": read_integrity(),
        "clean_slate_status": _clean_slate_status(),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }, indent=2))


# --------------------------------------------------------------------------

def main() -> int:
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, _on_sigterm)
    # Everything created in this container (driver git commits, traces, memory,
    # mongod state) must be writable by the unprivileged agent user; the only
    # protected file is the sequestered holdout, locked to 600 explicitly.
    os.umask(0)

    self_test = "--self-test" in sys.argv
    # Seed-variance mode: bring the full stack up exactly like a real run,
    # then instead of submitting an optimization, evaluate the SEED program
    # `variance_estimate_samples` times on one fixed subsample and save the
    # run_variance_estimate statistics (scripts/seed_variance.py). Answers
    # "how noisy is a subsample reading?" for budget sizing without spending
    # an optimization run.
    seed_variance = "--seed-variance" in sys.argv
    bench = os.environ.get("CE_BENCH") or _die("CE_BENCH is required")
    # The experiments/ directory name is often NOT the benchmark slug
    # (phantomwiki -> experiments/phantom-wiki, ragqa ->
    # experiments/RAGQAArenaTech). The controller supplies the mapping from
    # orchestrator/benchmarks.yaml; falling back to the slug only works for
    # benchmarks whose directory happens to match.
    program = os.environ.get("CE_PROGRAM") or bench
    arm = os.environ.get("CE_ARM", "ee")
    config_name = os.environ.get("CE_CONFIG") or "OPTIMIZE_CONFIG_LOCAL"
    extra_sets = [s for s in os.environ.get("CE_SET", "").split(";") if s.strip()]
    extra_sets += research_mode_defaults(extra_sets)

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    CANCEL_SENTINEL.unlink(missing_ok=True)
    started = time.time()

    log(f"run_id={os.environ.get('CE_RUN_ID')} bench={bench} "
        f"program=experiments/{program} arm={arm} "
        f"config={config_name} self_test={self_test} seed_variance={seed_variance}")

    clone_workspace(bench)
    stage_data_assets(bench)
    stage_venv(bench)
    open_workspace_permissions()
    trust_workspace_for_agent()
    extra_sets += sequester_holdout(
        effective_config(program, config_name, extra_sets)
    )
    configure_egress_route()
    run_preflight()

    # Re-resolve now that every override is final (research-mode flags and the
    # sequester's testset_path), so the banner and the eval budget describe the
    # config that is actually submitted rather than an earlier draft.
    config = effective_config(program, config_name, extra_sets)
    log_settings(bench, program, arm, config_name, config, extra_sets)

    mongod = start_mongod()
    api = start_api(arm)
    try:
        if self_test:
            log("self-test: stack is up, skipping job submission")
            (ARTIFACTS / "status.json").write_text(json.dumps(
                {"status": "self-test-ok", "bench": bench,
                 "program": program, "arm": arm}, indent=2))
            return 0

        if seed_variance:
            sys.path.insert(0, str(APP_ROOT))
            from scripts.seed_variance import run_seed_variance  # noqa: PLC0415

            # Same knob as the in-run tool (research default 5), overridable
            # per invocation via CE_SET variance_estimate_samples=N.
            n_runs = int(config.get("variance_estimate_samples") or 5)
            payload = run_seed_variance(
                workspace=str(WORKSPACE),
                config=config,
                n_runs=n_runs,
                output_path=str(ARTIFACTS / "seed_variance.json"),
                log=log,
            )
            (ARTIFACTS / "status.json").write_text(json.dumps({
                "status": "seed-variance-ok", "bench": bench,
                "program": program, "arm": arm,
                "aggregate_scores": payload["aggregate_scores"],
                "run_to_run_std": payload["run_to_run_std"],
                "aggregate_std_empirical": payload["aggregate_std_empirical"],
                "elapsed_seconds": round(time.time() - started, 1),
            }, indent=2))
            return 0

        job_id = submit(program, config_name, extra_sets)
        job = poll(job_id, max_metric_calls=_max_metric_calls(config))
        exit_reason = "sigterm" if _sigterm_received else (
            "cancelled" if CANCEL_SENTINEL.exists() else "natural"
        )

        # A terminal job status means the RESULT is final, not that the work
        # is over. Wait for the process itself before exporting or tearing
        # down. Skipped under SIGTERM, where docker's grace timer will SIGKILL
        # us and exporting what exists beats waiting for what might not land.
        if not _sigterm_received:
            await_optimizer_exit(job_id, POST_RUN_TIMEOUT)

        export_candidates(job_id)
        export_bundle()
        export_workspace(job)
        write_status(job_id, job, started, exit_reason)
        return 0 if job.get("status") in ("completed", "cancelled") else 1
    finally:
        for proc, name in ((api, "api"), (mongod, "mongod")):
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
            log(f"{name} stopped")


def _optimizer_pid(job_id: str) -> int | None:
    """PID of the optimizer subprocess, from the launcher's pidfile.

    `LocalLauncher` writes `{local_log_root}/{job_id}.pid`, which resolves
    under HOME — and HOME is /artifacts in this image, so the file is right
    there in the run's own artifact directory.
    """
    try:
        return int((LOCAL_LOG_ROOT / f"{job_id}.pid").read_text().strip())
    except (OSError, ValueError):
        return None


def _optimizer_alive(pid: int) -> bool:
    """True only while `pid` is STILL the optimizer.

    Reading the cmdline rather than sending signal 0 makes PID reuse
    unfalsifiable: if the optimizer exited and the container recycled its pid
    onto an eval subprocess, a signal probe would report "alive" and this
    function reports the truth. Every method's entry shim is
    `.../optimization_entry.py` (`local_launcher._ENTRY_SCRIPT_REL`), so one
    substring covers ee, freeform, pareto_evolution and simple_loop alike.
    """
    try:
        return b"optimization_entry" in Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return False


def await_optimizer_exit(job_id: str, timeout: float) -> None:
    """Block until the optimizer process is really gone.

    `poll()` returns as soon as the API reports a terminal status, but the
    optimizer deliberately keeps working past that point: `optimizer_loop`
    calls `set_completed()` BEFORE running meta-reflection and the
    run-integrity inspector, so that an advisory agent can never fail a run
    which has already succeeded. Treating "result is final" as "process is
    done" would tear the container down inside one poll interval and kill
    both agents; the integrity inspector runs AFTER meta-reflection, so it
    would never be reached at all.

    Waiting here also fixes the ordering: the inspector flushes its verdict
    onto the last iteration record post-completion, so `export_*` below
    picks it up instead of racing it.

    Bounded, because an advisory agent must never own the container: past
    `timeout` we export what exists and shut down, exactly as before.
    """
    pid = _optimizer_pid(job_id)
    if pid is None:
        log("post-run: no optimizer pidfile — not waiting")
        return
    if not _optimizer_alive(pid):
        log("post-run: optimizer already exited")
        return

    log(f"post-run: waiting up to {timeout:.0f}s for optimizer pid {pid} to "
        "finish (meta-reflection, run-integrity)")
    deadline = time.time() + timeout
    while _optimizer_alive(pid):
        if time.time() >= deadline:
            log(f"post-run: optimizer pid {pid} still running after "
                f"{timeout:.0f}s — exporting without it")
            return
        time.sleep(POST_RUN_POLL_INTERVAL)
    log("post-run: optimizer exited; post-run agents finished")


def _max_metric_calls(config: dict) -> int | None:
    """The run's eval budget, for the mid-run integrity trigger.

    Taken from the SUBMITTED config, which already has CE_SET folded in -- so
    the trigger's budget cannot disagree with the run's budget. That was the
    old bug: this function knew only about CE_SET and CE_MAX_METRIC_CALLS, so
    a run configured entirely from input.py (i.e. every normal run) silently
    disabled the check.

    CE_MAX_METRIC_CALLS survives only as a fallback for a config that declares
    no budget at all; unset is its normal state.
    """
    for value in (config.get("max_metric_calls"), os.environ.get("CE_MAX_METRIC_CALLS")):
        if value in (None, ""):
            continue
        try:
            return int(float(value))
        except (TypeError, ValueError):
            continue
    return None


# Control flags a research run must never be missing. Pinned explicitly rather
# than inherited: enable_run_integrity_agent and enable_meta_reflection already
# default True in src/api/schemas/requests.py, but that is a SaaS default, and a
# future change to it would disarm the research matrix with nothing to notice.
RESEARCH_DEFAULTS = {
    "enable_clean_slate_inspector": "true",
    "enable_run_integrity_agent": "true",
    "enable_meta_reflection": "true",
    # Finalization draws the held-out set 3x for the FINAL program so every
    # run reports a within-run sample of its final test score; the seed is
    # drawn once — every arm re-measures the same seed, so its readings
    # already average across the matrix.
    "test_samples": "3",
    "seed_test_samples": "1",
    # run_variance_estimate (specs/archive-public/sampling_variance.md): the architect's
    # once-per-run noise estimate evaluates its fixed sample this many times
    # (5 x 50 = 250 counted rows = 5% of a 5000-row budget). Pinned so every
    # arm pays the same price for the same reading.
    "variance_estimate_samples": "5",
    # Comparison-credit ledger (specs/archive-public/comparison_credits.md): identical pacing
    # across every arm of the matrix. Pinned so a SaaS default change cannot
    # silently alter the research budget mechanics.
    "comparison_credits_initial": "1",
    "comparison_credits_per_iteration": "1",
}


def research_mode() -> bool:
    """Whether this is a research run. True inside the runner image.

    The runner container IS the research path -- one container per run, fresh
    from an immutable image, holdout sequestered, egress fenced. Making that
    explicit as `CE_RESEARCH_MODE` (set in runner/Dockerfile.base) means the
    controls below are a property of the environment, not something each
    researcher has to remember to export.
    """
    return os.environ.get("CE_RESEARCH_MODE", "").strip().lower() in (
        "1", "true", "yes", "on",
    )


def research_mode_defaults(extra_sets: list[str]) -> list[str]:
    """`--set` pairs arming the research controls, minus any CE_SET already set.

    An explicit CE_SET value always wins -- turning a control OFF for a
    deliberate experiment has to remain possible, and a duplicate override
    would just be confusing in the banner.
    """
    if not research_mode():
        return []
    already = {item.partition("=")[0].strip() for item in extra_sets}
    injected = [
        f"{key}={value}"
        for key, value in RESEARCH_DEFAULTS.items()
        if key not in already
    ]
    if injected:
        log(f"research mode: arming {', '.join(sorted(injected))}")
    overridden = sorted(already & set(RESEARCH_DEFAULTS))
    if overridden:
        log(f"research mode: CE_SET overrides in effect for {', '.join(overridden)}")
    return injected


def _die(msg: str):
    raise SystemExit(msg)


if __name__ == "__main__":
    raise SystemExit(main())
