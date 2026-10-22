"""Test-environment abstractions for the local integration runs.

Two infrastructure shapes appear across the four ported integration tests:

* **Synthetic workspace** (`inspect_traces`, `analysis_agent`): a small in-memory
  fixture set (`/workspace/.env`, `/memory/…`, `/traces/…`) plus a Python
  driver script that imports `IterationArchitect` / `subprocess_service` and
  prints a JSON marker.
* **Real client repo** (`ee_e2e`, `eval_trace`): a real checkout of a
  client repo (FactChecker by default), a venv with the client's deps, and
  an exec of `optimization_entry.py`.

Both shapes share a tiny operation set (seed files, exec a process,
stream stdout, parse a marker, tear down). Two protocols below capture
the two shapes; each has a local implementation.

The protocols expose `workspace_dir` / `traces_dir` (synthetic also
exposes `memory_dir` / `engine_root`) so callers don't bake `/workspace`
or `/traces` literals into test code. Path values resolve through
`src/engine/paths.py` — the same module the production engine uses —
so test and production read paths from one source of truth. Local envs
set the production env vars (`SANDBOX_MODE`, `CODEEVOLVER_MEMORY_DIR`,
`CODEEVOLVER_TRACES_DIR`, `CODEEVOLVER_ARCHITECT_TRACES_DIR`,
`CODEEVOLVER_LOGS_DIR`, `CODEEVOLVER_JOB_ID`) on `os.environ` in
`__init__` and restore them in `cleanup`.

`LocalSyntheticEnv` keeps its state in a tmpdir that is deleted on
cleanup; `LocalClientRepoEnv` (the e2e path) keeps its state under a
durable `~/.codeevolver/e2e/` root scoped by `test_id`, because a real
optimizer run's memory, traces and post-run reports are worth keeping.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterable, Iterator, Protocol

from src.engine.paths import (
    SANDBOX_MODE_LOCAL,
    architect_traces_dir as _resolve_architect_traces_dir,
    logs_dir as _resolve_logs_dir,
    memory_dir as _resolve_memory_dir,
    traces_dir as _resolve_traces_dir,
)

logger = logging.getLogger(__name__)

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0A-\x1F\x7F]")


# Env vars the Local envs mutate on this process so `paths.traces_dir()` /
# `paths.memory_dir()` return the per-test tmpdirs in both the test process
# and any subprocess that inherits `os.environ`. The literal strings match
# the accessors in `src/engine/paths.py` — a typo here only breaks tests,
# not production.
_LOCAL_PATH_ENV_KEYS = (
    "SANDBOX_MODE",
    "CODEEVOLVER_MEMORY_DIR",
    "CODEEVOLVER_TRACES_DIR",
    "CODEEVOLVER_ARCHITECT_TRACES_DIR",
    "CODEEVOLVER_LOGS_DIR",
    "CODEEVOLVER_JOB_ID",
)


def _apply_local_path_env(
    *,
    memory: str,
    traces: str,
    architect_traces: str,
    logs: str,
    job_id: str | None = None,
) -> dict[str, str | None]:
    """Point `paths.py` at per-test memory/traces/architect-traces/logs dirs.

    ``job_id`` is set on THIS process as well as inherited by subprocesses:
    `paths._job_subdir()` appends ``/{job_id}`` to the memory/traces/
    architect-traces dirs, so without it here the test process and the
    optimizer subprocess would resolve different paths and the test could
    not read the run's own artifacts back. `v3/_entry.py` re-sets the
    identical value from the config, so the two always agree.

    Returns originals so `cleanup()` can restore them.
    """
    originals = {k: os.environ.get(k) for k in _LOCAL_PATH_ENV_KEYS}
    os.environ["SANDBOX_MODE"] = SANDBOX_MODE_LOCAL
    os.environ["CODEEVOLVER_MEMORY_DIR"] = memory
    os.environ["CODEEVOLVER_TRACES_DIR"] = traces
    os.environ["CODEEVOLVER_ARCHITECT_TRACES_DIR"] = architect_traces
    os.environ["CODEEVOLVER_LOGS_DIR"] = logs
    if job_id:
        os.environ["CODEEVOLVER_JOB_ID"] = job_id
    else:
        os.environ.pop("CODEEVOLVER_JOB_ID", None)
    return originals


def _restore_local_path_env(originals: dict[str, str | None]) -> None:
    for key, value in originals.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


# ---------------------------------------------------------------------------
# Git-fixture helper (shared by all synthetic-workspace drivers)
# ---------------------------------------------------------------------------


def git_fixture_snippet(
    *,
    branch: str,
    message: str,
    stage_all: bool = True,
    ignore: Iterable[str] = (),
    workspace_var: str = "WORKSPACE",
    sha_var: str = "START_SHA",
    indent: str = " " * 8,
) -> str:
    """Return *driver-script text* that makes the synthetic workspace a real repo.

    Returns source, not an env method, deliberately: the driver body is
    just Python executed in-env, so one implementation covers every backend
    without touching `SyntheticTestEnvironment` or `LocalSyntheticEnv`.
    It also preserves the required ordering — the runner's `seed_file` calls
    all land before `exec_python_driver`, so the fixtures exist on disk by
    the time this snippet commits.

    Without a repo, the architect's close-time hooks (`_commit_workstation_drift`,
    `_registered_head_sha`, the no-change watchdog) log `fatal: not a git
    repository` on every run and bury real signal.

    Args:
        branch: `codeevolver-`-prefixed branch to create and check out.
        message: seed-commit message.
        stage_all: `git add -A` before committing (the whole fixture becomes
            tracked) vs. an empty seed commit. Pass False for tests that seed
            secrets into the workspace — see `ignore`.
        ignore: paths written to `.gitignore` and committed. Any test that
            seeds a holdout testset must list it here: a `git add -A` (from
            this snippet OR from the later drift commit) would otherwise put
            the secret into git objects, where `git cat-file -p <blob>` reaches
            it. The inspection agent's deny hook matches on Bash command text,
            so it blocks `git show HEAD:<path>` but not a blob read.
        workspace_var / sha_var: names of the driver-local variables holding
            the workspace path and receiving the seed sha.
        indent: prefix applied to every line but the first, so the result
            drops into an indented `textwrap.dedent` block unchanged.
    """
    ignore = tuple(ignore)
    lines = [
        "def _git(*args):",
        "    r = subprocess.run(",
        f"        ['git', *args], cwd={workspace_var},",
        "        capture_output=True, text=True,",
        "    )",
        "    if r.returncode != 0:",
        "        raise RuntimeError(",
        "            'git ' + ' '.join(args) + ' failed: ' + r.stderr.strip()",
        "        )",
        "    return r.stdout.strip()",
        "",
        "_git('init', '-q')",
        "_git('config', 'user.email', 'test@codeevolver.local')",
        "_git('config', 'user.name', 'CodeEvolver Test')",
    ]
    if ignore:
        body = "\n".join(ignore) + "\n"
        lines += [
            f"with open(os.path.join({workspace_var}, '.gitignore'), 'w') as _f:",
            f"    _f.write({body!r})",
            "_git('add', '.gitignore')",
        ]
    if stage_all:
        lines.append("_git('add', '-A')")
    commit = f"_git('commit', '-q', '-m', {message!r}"
    # An empty seed commit still needs --allow-empty when nothing was staged.
    if not stage_all and not ignore:
        commit += ", '--allow-empty'"
    lines.append(commit + ")")
    lines += [
        f"_git('checkout', '-q', '-b', {branch!r})",
        f"{sha_var} = _git('rev-parse', 'HEAD')",
    ]
    return ("\n" + indent).join(lines)


# ---------------------------------------------------------------------------
# Per-run test id
# ---------------------------------------------------------------------------

_TEST_ID_UNSAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def new_test_id(*, method: str | None = None, harness: str | None = None) -> str:
    """A filesystem-safe id for one e2e run.

    The id becomes the optimizer config's `job_id`, which `paths.py` uses two
    ways: as a directory name (`_job_subdir()` appends `/{job_id}` to the
    memory / traces / architect-traces dirs) and as a filename prefix in
    `logs_dir()` (`{job_id}_detailed_agent_log.log`). Nothing validates it, so
    the only real requirement is that it is a safe single path segment —
    hence the sanitising substitution below. Production ids look like
    `job_<hex12>`; these are named so a directory listing is readable.
    """
    parts = ["e2e"]
    parts += [p for p in (method, harness) if p]
    parts.append(_utc_stamp())
    parts.append(uuid.uuid4().hex[:6])
    return _TEST_ID_UNSAFE_RE.sub("-", "-".join(parts)).strip("-.")


def _utc_stamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")


# ---------------------------------------------------------------------------
# Marker-parsing helper (shared by all envs)
# ---------------------------------------------------------------------------


def parse_marker(lines: Iterable[str], marker: str) -> dict | None:
    """Extract the JSON payload after `<marker>:` from a stream of stdout lines.

    Two-pass parser: line-by-line first, then a buffer scan for the LAST
    occurrence as a fallback. The buffer scan handles two issues:
      1. A chunked stdout iterator can split a long marker line into
         multiple sub-lines.
      2. A driver that printed an earlier error marker before recovering
         should not have its final result shadowed.

    Returns the parsed dict, or None if no parseable marker is present.
    """
    materialized: list[str] = []
    payload: dict | None = None
    prefix = marker + ":"
    for line in lines:
        materialized.append(line)
        if line.startswith(prefix):
            try:
                payload = json.loads(line[len(prefix):])
            except json.JSONDecodeError:
                pass

    if payload is not None:
        return payload

    buf = "\n".join(materialized)
    idx = buf.rfind(prefix)
    if idx < 0:
        return None
    tail = buf[idx + len(prefix):].strip()
    tail = _CONTROL_CHAR_RE.sub("", tail)
    # Find the first { and read a balanced object out of the tail.
    brace = tail.find("{")
    if brace < 0:
        return None
    depth = 0
    end = -1
    in_str = False
    esc = False
    for j, ch in enumerate(tail[brace:], start=brace):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = j + 1
                break
    if end <= brace:
        return None
    try:
        return json.loads(tail[brace:end])
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


class SyntheticTestEnvironment(Protocol):
    """Workspace + Python-driver exec for tests that don't need a client repo.

    Path properties expose absolute paths inside the env so driver
    builders can construct trace / memory / workspace paths without
    hardcoding `/workspace`. Local envs return tmpdir subpaths.
    """

    workspace_dir: str
    memory_dir: str
    traces_dir: str
    architect_traces_dir: str
    engine_root: str

    def seed_file(self, abs_path: str, content: str) -> None: ...

    def exec_python_driver(
        self,
        driver_script: str,
        *,
        env_vars: dict[str, str] | None = None,
    ) -> Iterator[str]:
        """Run the driver under `python -u` and yield decoded stdout lines."""
        ...

    def cleanup(self) -> None: ...


class ClientRepoTestEnvironment(Protocol):
    """Real client repo + optimizer-entry exec for end-to-end tests.

    The repo is set up once per test invocation; the optimizer subprocess
    is then exec'd against `optimization_entry.py` like a production run.
    `read_file` is exposed so tests can inspect the
    `/traces/iteration_0_validate_env.jsonl` output (or, with meta-reflection on,
    `{memory_dir}/meta_reflection.md` and the architect traces) after the run.
    """

    workspace_dir: str
    traces_dir: str
    memory_dir: str
    architect_traces_dir: str
    logs_dir: str
    # Per-run id, passed to the optimizer as the config's `job_id` so every
    # artifact path is scoped to this run. `None` on environments that keep
    # the unscoped production defaults.
    test_id: str | None
    # Where this run's artifacts survive after `cleanup()`, or None when the
    # environment does not keep them.
    artifact_root: str | None

    def prepare_workspace(
        self,
        *,
        repo_url: str,
        branch: str | None = None,
        github_token: str | None = None,
    ) -> None: ...

    def write_file(self, abs_path: str, content: str) -> None: ...

    def exec_optimizer(
        self,
        entry_script: str,
        config_path: str,
        *,
        env_vars: dict[str, str] | None = None,
    ) -> Iterator[str]: ...

    def read_file(self, abs_path: str) -> str: ...

    def cleanup(self) -> None: ...


# ---------------------------------------------------------------------------
# Local implementations
# ---------------------------------------------------------------------------


class LocalSyntheticEnv:
    """Local synthetic env. No client repo.

    Stores all per-run state under per-test tmpdirs. The tmpdirs are
    published to the engine via the production env vars (``SANDBOX_MODE``,
    ``CODEEVOLVER_MEMORY_DIR``, ``CODEEVOLVER_TRACES_DIR``) set on this
    process's ``os.environ`` in ``__init__`` and restored in ``cleanup``.
    Subprocesses inherit them via ``os.environ.copy()``; the test process
    itself sees the same values via ``paths.traces_dir()`` /
    ``paths.memory_dir()``, so test and production resolve the same paths
    from one source of truth.

    No host-level setup is required — no `sudo mkdir /memory /traces`.
    """

    workspace_dir: str

    def __init__(self, *, engine_root: str | None = None) -> None:
        from src.engine.paths import engine_root as _engine_root_helper

        self._tmp = tempfile.TemporaryDirectory(prefix="ce-test-syn-")
        root = Path(self._tmp.name)
        self.workspace_dir = str(root / "workspace")
        memory = str(root / "memory")
        traces = str(root / "traces")
        architect_traces = str(root / "architect_traces")
        logs = str(root / "logs")
        for d in (self.workspace_dir, memory, traces, architect_traces, logs):
            Path(d).mkdir(parents=True, exist_ok=True)
        self.engine_root = engine_root or _engine_root_helper()
        self._env_originals = _apply_local_path_env(
            memory=memory, traces=traces, architect_traces=architect_traces,
            logs=logs,
        )

    @property
    def memory_dir(self) -> str:
        return _resolve_memory_dir()

    @property
    def traces_dir(self) -> str:
        return _resolve_traces_dir()

    @property
    def architect_traces_dir(self) -> str:
        return _resolve_architect_traces_dir()

    def seed_file(self, abs_path: str, content: str) -> None:
        target = Path(abs_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)

    def exec_python_driver(
        self,
        driver_script: str,
        *,
        env_vars: dict[str, str] | None = None,
    ) -> Iterator[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = self.engine_root + os.pathsep + env.get("PYTHONPATH", "")
        if env_vars:
            env.update(env_vars)
        result = subprocess.run(
            [sys.executable, "-u", "-"],
            input=driver_script,
            capture_output=True,
            text=True,
            env=env,
            cwd=self.engine_root,
        )
        # Yield stdout (and stderr, merged) line-by-line.
        for line in result.stdout.splitlines():
            yield line
        for line in result.stderr.splitlines():
            yield line

    def cleanup(self) -> None:
        _restore_local_path_env(self._env_originals)
        try:
            self._tmp.cleanup()
        except Exception as exc:  # pragma: no cover
            logger.warning("LocalSyntheticEnv cleanup failed: %s", exc)


class LocalClientRepoEnv:
    """Client-repo test environment on the host machine.

    Caches fixture clones at `~/.codeevolver/test-repos/{slug}` so the
    first run pays the clone cost and subsequent runs are fast. Wraps
    `LocalEnvironment` for venv + deps + pre-push hook setup, then execs
    `optimization_entry.py` directly as a subprocess.

    Memory, traces, architect traces and logs land under a DURABLE per-run
    root — `~/.codeevolver/e2e/` scoped by `test_id` — and are threaded into
    the optimizer via `CODEEVOLVER_MEMORY_DIR` / `CODEEVOLVER_TRACES_DIR` /
    `CODEEVOLVER_ARCHITECT_TRACES_DIR` / `CODEEVOLVER_LOGS_DIR`. The
    architect's system prompts (parameterized via `src/engine/paths.py`)
    pick up the same paths, so the host filesystem stays clean — no
    `sudo mkdir /memory /traces` required.

    The layout mirrors a production local run (`~/.codeevolver/{memory,
    traces,...}/{job_id}`) but is rooted under `e2e/` for two reasons: test
    runs must not pollute the real state tree, and the clean-slate inspector
    reads `~/.codeevolver/` looking for other jobs' leftovers — an e2e run's
    own artifacts sitting next to real runs' would read as contamination.

    Artifacts deliberately SURVIVE `cleanup()`: a two-iteration agent run is
    expensive, and its memory / meta-reflection / integrity report / traces
    are the whole point of running it. Pruning `~/.codeevolver/e2e/` is the
    operator's call.
    """

    REPO_CACHE_ROOT = Path.home() / ".codeevolver" / "test-repos"
    STATE_ROOT = Path.home() / ".codeevolver" / "e2e"

    def __init__(self, *, timeout: int = 7200, test_id: str | None = None) -> None:
        self._timeout = timeout
        self._local_env = None  # set by prepare_workspace
        self.workspace_dir = ""  # populated below
        self.test_id = test_id or new_test_id()
        memory = str(self.STATE_ROOT / "memory")
        traces = str(self.STATE_ROOT / "traces")
        architect_traces = str(self.STATE_ROOT / "architect_traces")
        logs = str(self.STATE_ROOT / "logs")
        # `paths.logs_dir()` is not job-suffixed (the job id goes in the
        # filenames), so create it directly; the other three get their
        # `/{test_id}` subdir from `paths._job_subdir()`.
        Path(logs).mkdir(parents=True, exist_ok=True)
        self._env_originals = _apply_local_path_env(
            memory=memory, traces=traces, architect_traces=architect_traces,
            logs=logs, job_id=self.test_id,
        )
        for d in (self.memory_dir, self.traces_dir, self.architect_traces_dir):
            Path(d).mkdir(parents=True, exist_ok=True)
        logger.info(
            "e2e test_id=%s — artifacts under %s", self.test_id, self.artifact_root
        )

    @property
    def memory_dir(self) -> str:
        return _resolve_memory_dir()

    @property
    def traces_dir(self) -> str:
        return _resolve_traces_dir()

    @property
    def architect_traces_dir(self) -> str:
        return _resolve_architect_traces_dir()

    @property
    def logs_dir(self) -> str:
        return _resolve_logs_dir()

    @property
    def artifact_root(self) -> str:
        """Root the operator should look under for this run's artifacts."""
        return str(self.STATE_ROOT)

    @property
    def run_log_path(self) -> str:
        """Where `exec_optimizer` tees the optimizer's full stdout."""
        return str(Path(self.logs_dir) / f"{self.test_id}.log")

    def prepare_workspace(
        self,
        *,
        repo_url: str,
        branch: str | None = None,
        github_token: str | None = None,
        coding_harness: str = "claude",
    ) -> None:
        from src.local.local_environment import LocalEnvironment

        # `github_token` is only meaningful when the local clone needs auth.
        # For the public FactChecker fixture default it's unused; we still
        # accept it so the signature matches the protocol. `coding_harness` is
        # accepted for signature symmetry but ignored — the local operator
        # installs the harness binary on PATH; there is no image.
        repo_path = self._ensure_clone(repo_url, branch, github_token=github_token)
        self.workspace_dir = str(repo_path)
        self._local_env = LocalEnvironment(
            repo_path=str(repo_path), github_token=github_token,
        )
        self._local_env.start(python_version="3.11", branch=branch)

    def write_file(self, abs_path: str, content: str) -> None:
        Path(abs_path).parent.mkdir(parents=True, exist_ok=True)
        Path(abs_path).write_text(content)

    def exec_optimizer(
        self,
        entry_script: str,
        config_path: str,
        *,
        env_vars: dict[str, str] | None = None,
    ) -> Iterator[str]:
        assert self._local_env is not None, "call prepare_workspace first"
        from src.engine.paths import engine_root

        env = os.environ.copy()
        env["PYTHONPATH"] = engine_root() + os.pathsep + env.get("PYTHONPATH", "")
        # SANDBOX_MODE / CODEEVOLVER_MEMORY_DIR / CODEEVOLVER_TRACES_DIR are
        # set on this process's `os.environ` in __init__ via
        # `_apply_local_path_env`, so the `os.environ.copy()` above already
        # carries them — no separate per-launch assignment needed.
        if self._local_env._venv_dirname != "env":
            env["REPO_VENV_DIR"] = self._local_env._venv_dirname
        env["LOG_LEVEL"] = os.environ.get("LOG_LEVEL", "INFO")
        if env_vars:
            env.update(env_vars)

        # Resolve entry_script relative to engine_root if the caller passed
        # a container-style absolute path like `/app/src/engine/.../entry.py`.
        script_path = entry_script
        if script_path.startswith("/app/"):
            script_path = os.path.join(engine_root(), script_path[len("/app/"):])

        result = subprocess.Popen(
            [sys.executable, script_path, "--config", config_path],
            cwd=self.workspace_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert result.stdout is not None
        # Tee to a durable run log, the same thing `LocalLauncher` does for a
        # production local run (`{local_log_root}/{job_id}.log`). Without it
        # the only copy of a two-iteration run is terminal scrollback.
        log_path = Path(self.run_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as run_log:
            for line in result.stdout:
                line = line.rstrip("\n")
                run_log.write(line + "\n")
                run_log.flush()
                yield line
        result.wait()

    def read_file(self, abs_path: str) -> str:
        try:
            return Path(abs_path).read_text()
        except FileNotFoundError:
            return ""

    def cleanup(self) -> None:
        # The artifact root is deliberately NOT removed — see the class
        # docstring. Only the process-level env overrides and the repo lock
        # are released.
        _restore_local_path_env(self._env_originals)
        if self._local_env is not None:
            self._local_env.stop()

    # ---- internal ----

    def _ensure_clone(
        self,
        repo_url: str,
        branch: str | None,
        *,
        github_token: str | None = None,
    ) -> Path:
        slug = self._slug_from_url(repo_url)
        self.REPO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        target = self.REPO_CACHE_ROOT / slug
        if target.exists() and not self._cache_tracks_branch(target, branch):
            # The cache is a `--single-branch` clone of some OTHER branch, so
            # its fetch refspec will never produce `origin/{branch}`. Re-clone
            # rather than widening the refspec: a widened one would pull the
            # repo's ~114 `codeevolver-*` branches into `git branch -a`, which
            # is exactly what the clean-slate inspector flags.
            logger.info(
                "test-repo cache at %s does not track %r — re-cloning",
                target, branch,
            )
            shutil.rmtree(target, ignore_errors=True)
        if target.exists():
            # Fetch + reset to the requested branch so repeated runs see
            # any upstream changes. Repo lock from LocalEnvironment.start()
            # prevents concurrent runs on the same cached clone.
            subprocess.run(
                ["git", "fetch", "origin"],
                cwd=target, check=False, capture_output=True,
            )
            if branch:
                subprocess.run(
                    ["git", "checkout", branch],
                    cwd=target, check=False, capture_output=True,
                )
                subprocess.run(
                    ["git", "reset", "--hard", f"origin/{branch}"],
                    cwd=target, check=False, capture_output=True,
                )
            self._sweep_prior_run_state(target, branch)
            return target

        clone_args = ["git", "clone", "--single-branch"]
        if branch:
            clone_args += ["--branch", branch]
        clone_args += [repo_url, str(target)]
        result = subprocess.run(clone_args, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"git clone {repo_url} failed (exit={result.returncode}): "
                f"{result.stderr}"
            )
        return target

    @staticmethod
    def _cache_tracks_branch(target: Path, branch: str | None) -> bool:
        """Whether the cached clone's fetch refspec covers `branch`."""
        if not branch:
            return True
        res = subprocess.run(
            ["git", "config", "--get-all", "remote.origin.fetch"],
            cwd=target, check=False, capture_output=True, text=True,
        )
        specs = res.stdout.split()
        # A full clone's `+refs/heads/*:...` covers everything; a
        # `--single-branch` clone names exactly one branch.
        return any(
            f"refs/heads/{branch}:" in s or "refs/heads/*:" in s for s in specs
        )

    @staticmethod
    def _sweep_prior_run_state(target: Path, branch: str | None) -> None:
        """Return the cached clone to a pristine seed state.

        Every e2e run leaves `codeevolver-*` branches behind, and the
        clean-slate inspector treats them — plus anything still reachable
        through the reflog or `git fsck --unreachable` — as evidence of a
        previous optimization run, which aborts the next run. Deleting the
        branches alone is not enough: the inspector's prompt names the reflog
        and unreachable commits explicitly.

        `git clean -x` is deliberately NOT used: `env/` is the reused venv
        (`LocalEnvironment` adopts an existing `<repo>/env`) and removing it
        would force a full dependency reinstall on every run.
        """
        def _git(*args: str) -> subprocess.CompletedProcess:
            return subprocess.run(
                ["git", *args], cwd=target, check=False,
                capture_output=True, text=True,
            )

        stale = [
            line.strip()
            for line in _git("for-each-ref", "--format=%(refname:short)",
                             "refs/heads/codeevolver-*").stdout.splitlines()
            if line.strip() and line.strip() != branch
        ]
        if stale:
            logger.info("sweeping %d stale codeevolver-* branch(es)", len(stale))
            _git("branch", "-D", *stale)
        _git("reflog", "expire", "--expire=now", "--expire-unreachable=now", "--all")
        _git("gc", "--prune=now", "--quiet")
        # Gitignored leftovers a prior run staged into the workspace; the
        # engine's own reset does not sweep these.
        for leftover in (".claude/skills", ".codeevolver.lock"):
            path = target / leftover
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            elif path.exists():
                path.unlink(missing_ok=True)

    @staticmethod
    def _slug_from_url(repo_url: str) -> str:
        """Derive a filesystem-friendly slug from a repo URL."""
        # https://github.com/foo/bar(.git)? → bar
        tail = repo_url.rstrip("/").split("/")[-1]
        if tail.endswith(".git"):
            tail = tail[:-4]
        return tail or uuid.uuid4().hex


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shell_quote(value: str) -> str:
    """Single-quote a value for safe shell use."""
    return "'" + value.replace("'", "'\\''") + "'"


# --- Trace-directory seeding -------------------------------------------------
# Eval traces (`iteration_{N}_{suffix}/`) and architect traces
# (`iteration_{N}_architect_trace/`) are directories of pretty JSON files. Seed
# them through the production writers so fixtures never drift from the real
# layout; files are rendered locally and shipped via `env.seed_file`, which
# works for any synthetic env.


def _ship_tree(env, local_root: str, target_dir: str) -> None:
    for root, _dirs, files in os.walk(local_root):
        for name in sorted(files):
            src = os.path.join(root, name)
            rel = os.path.relpath(src, local_root)
            env.seed_file(os.path.join(target_dir, rel), Path(src).read_text())


def seed_eval_trace_dir(env, trace_dir: str, rows: list[dict]) -> None:
    """Seed `trace_dir` (an `iteration_{N}_{suffix}` path) from evaluator rows."""
    import tempfile

    from src.engine.telemetry.trace_layout import write_eval_trace

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "trace")
        write_eval_trace(local, rows, max_line_chars=None)
        _ship_tree(env, local, trace_dir)


def seed_architect_trace_dir(env, trace_dir: str, turns: list[dict]) -> None:
    """Seed `trace_dir` (an `iteration_{N}_architect_trace` path) from turn dicts."""
    import tempfile

    from src.engine.optimizer.v3.architect_trace import _write_turn
    from src.engine.optimizer.v3.architect_trace_markdown import render_trace_dir

    with tempfile.TemporaryDirectory() as tmp:
        local = os.path.join(tmp, "trace")
        os.makedirs(local)
        for i, turn in enumerate(turns):
            _write_turn(local, i, turn)
        # The auditors are told to read digest.md / transcript.md first — seed
        # them so fixture-driven meta-reflection tests exercise that path.
        render_trace_dir(local, header=None)
        _ship_tree(env, local, trace_dir)
