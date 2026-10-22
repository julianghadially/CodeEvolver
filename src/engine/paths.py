"""Resolved paths for engine source, cross-iteration memory, and per-iteration traces.

The optimizer writes per-iteration trace JSONL files and architect memory
files to absolute directories that are referenced both by Python code and by
the architect's system prompt. Local deploys (including the Docker runner,
which sets ``SANDBOX_MODE=local``) use ``~/.codeevolver/traces`` and
``~/.codeevolver/memory``; the non-local default is ``/traces`` and ``/memory``.

Trace/memory helpers resolve in this order:

1. Explicit env var (``CODEEVOLVER_TRACES_DIR`` / ``CODEEVOLVER_MEMORY_DIR`` /
   ``CODEEVOLVER_ARCHITECT_TRACES_DIR``) — highest priority. Tests use this to
   point each run at an isolated tmpdir.
2. ``SANDBOX_MODE=local`` → ``~/.codeevolver/{traces,memory,architect_traces}``.
3. Non-local default — ``/traces``, ``/memory``, ``/architect_traces``.

Additionally, if ``CODEEVOLVER_JOB_ID`` is set, the resolved directory gets
``/{job_id}`` appended — so every accessor returns a job-scoped path that
all subprocesses agree on via ``os.environ`` inheritance. This isolates
artifacts across successive runs (important for local research, where one
operator runs many jobs back-to-back against the same ``~/.codeevolver``
state tree). The env var is set once by ``optimizer/v3/_entry.py``.

``engine_root()`` is different: there is no env var, no ``SANDBOX_MODE``
branch. It auto-detects from this module's own ``__file__`` because
``paths.py`` lives wherever the engine code lives (e.g.
``<your-checkout>/src/engine/paths.py``), so the answer is correct by
construction in every deployment — no configuration required.

This module does not import ``src.config`` so the engine tree stays usable
without ``src/config.py``. The ``SANDBOX_MODE`` env var check is duplicated
here intentionally.

Env-var convention
------------------
``paths.py`` is the central declaration site for every engine-side env var.
The default rule: each env var has an accessor function (``traces_dir()``,
``memory_dir()``, ``repo_venv_dirname()``, ``repo_venv_bin()``) that names
the env-var string inline and returns the resolved value. Consumers call the
accessor; ``os.environ`` is otherwise not touched elsewhere in the engine.

A small number of env vars are **written** by another module so a
subprocess or sibling module can read the value back. Those names are
listed in ``CROSS_PROCESS_ENV_VARS`` below — writers use the literal
string, and the list documents the cross-module string contract so a
rename has a single point to grep against.
"""
from __future__ import annotations

import os
from pathlib import Path


_LOCAL_ROOT_RELATIVE = (".codeevolver",)

# Env vars that are written by modules outside paths.py so another module
# or subprocess can read them back. Writers use the literal string; this
# tuple is the human-visible registry of cross-module string contracts.
# Update it whenever you add a new producer/consumer env var pair.
CROSS_PROCESS_ENV_VARS: tuple[str, ...] = (
    # Written by subprocess_service._build_env, read by src/__init__.py.
    "CODEEVOLVER_WORKSPACE_SRC",
    # Written by local_launcher, read by repo_venv_dirname().
    "REPO_VENV_DIR",
    # written by optimizer/v3/_entry.py::run_entry, read by _job_subdir() to scope traces/memory/runtime_tests dirs
    "CODEEVOLVER_JOB_ID",
    # Written by optimizer/v3/_entry.py::run_entry when the coding harness is
    # OpenCode, read by trace_max_line_chars() in the evaluator subprocess and
    # the architect-trace writer (OpenCode's read tool caps lines at 2000 chars).
    "CODEEVOLVER_TRACE_MAX_LINE_CHARS",
    # Operator/test override read by OpenCodeHarness when generating config.
    "CODEEVOLVER_OPENCODE_SMALL_MODEL",
    # Written by the Docker runner image (runner/Dockerfile.base), read by
    # agent_user() at every agent-subprocess spawn site.
    "CODEEVOLVER_AGENT_USER",
    # Written by OpenCodeHarness, read by opencode_mcp_proxy / plugin.
    "CODEEVOLVER_ARCHITECT_MCP_PROXY_URL",
    "CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN",
    # Written by OpenCodeHarness, read by opencode_mcp_proxy for the loopback
    # POST /call (tool-execution) timeout — long, since run_eval can run hours.
    "CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS",
    # Written by OpenCodeHarness, surfaced to OpenCode config/plugin diagnostics.
    "CODEEVOLVER_ARCHITECT_MCP_TOOL_NAMES",
    # Provider-process env vars written by OpenCodeHarness for OpenCode.
    "OPENCODE_CONFIG_CONTENT",
    "OPENCODE_CONFIG_DIR",
    "OPENCODE_PERMISSION",
    "OPENCODE_DISABLE_CLAUDE_CODE_SKILLS",
    "OPENCODE_DISABLE_CLAUDE_CODE",
)

# Value sentinel (not an env-var name) used by the local-vs-default branch.
SANDBOX_MODE_LOCAL = "local"

# Default sandbox-side paths used when no env-var override is set and
# SANDBOX_MODE is not "local". Tests import these directly so they cannot
# drift from the default.
MODAL_TRACES_DIR = "/traces"
MODAL_MEMORY_DIR = "/memory"
# Architect (meta) session traces live in a dedicated directory, a SIBLING of
# /traces — kept physically separate from the client (Task Agent) OTel traces
# so meta-traces can never be confused with or pollute client trace data.
MODAL_ARCHITECT_TRACES_DIR = "/architect_traces"
# Human-readable detailed agent log (per job). Sibling of the trace dirs.
MODAL_LOGS_DIR = "/logs"
# Run-history artifacts the architect may read (e.g. comparison_set_history.jsonl).
MODAL_HISTORY_DIR = "/history"

# Engine source root resolved at import time from this module's own location.
# `paths.py` lives at `<root>/src/engine/paths.py`, so `parents[2]` is `<root>` —
# the directory whose subtree contains `src/engine/...`. Works identically in
# a container (`/app/src/engine/paths.py` → `/app`) and on a local clone wherever it
# happens to live.
_ENGINE_ROOT = str(Path(__file__).resolve().parents[2])


def is_local() -> bool:
    """True when the subprocess is running on a local deploy.

    The launcher (or the operator's shell) sets ``SANDBOX_MODE=local`` on
    the API process; ``os.environ.copy()`` propagates it to every spawned
    subprocess.
    """
    return os.environ.get("SANDBOX_MODE", "").lower() == SANDBOX_MODE_LOCAL


_is_local = is_local  # backwards-compatible alias (kept until callers move)


def _local_root() -> Path:
    """~/.codeevolver/ — the conventional local-deploy state root."""
    return Path.home().joinpath(*_LOCAL_ROOT_RELATIVE)


def _job_subdir() -> str:
    """Return ``/{CODEEVOLVER_JOB_ID}`` if the env var is set, else ``''``.

    Set once by ``optimizer/v3/_entry.py::run_entry`` from the config's
    ``job_id`` field, before any optimizer module imports paths.py.
    Every traces/memory/runtime-test path is suffixed, so per-job
    artifacts stay isolated across successive runs on the same host.
    """
    job_id = os.environ.get("CODEEVOLVER_JOB_ID", "").strip()
    return f"/{job_id}" if job_id else ""


# Leaves headroom under OpenCode's hardcoded 2,000-char read/grep line cap for
# the 2-space JSON indent, quotes and the trailing comma.
OPENCODE_TRACE_MAX_LINE_CHARS = 1900


def trace_max_line_chars() -> int | None:
    """Per-line character budget for agent-facing trace files, or None.

    Set (to ``OPENCODE_TRACE_MAX_LINE_CHARS``) by ``run_entry`` only for
    OpenCode runs; strings longer than this are written as chunk lists (see
    ``telemetry/trace_json.py``). Unset/0 → plain pretty JSON.
    """
    raw = os.environ.get("CODEEVOLVER_TRACE_MAX_LINE_CHARS", "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def traces_dir() -> str:
    """Directory holding per-iteration trace directories (``iteration_{N}_{suffix}/``)."""
    explicit = os.environ.get("CODEEVOLVER_TRACES_DIR")
    if explicit:
        return explicit + _job_subdir()
    if is_local():
        return str(_local_root() / "traces") + _job_subdir()
    return MODAL_TRACES_DIR + _job_subdir()


def memory_dir() -> str:
    """Directory holding cross-iteration architect memory files."""
    explicit = os.environ.get("CODEEVOLVER_MEMORY_DIR")
    if explicit:
        return explicit + _job_subdir()
    if is_local():
        return str(_local_root() / "memory") + _job_subdir()
    return MODAL_MEMORY_DIR + _job_subdir()


def history_dir() -> str:
    """Directory holding run-history artifacts (comparison_set_history.jsonl)."""
    explicit = os.environ.get("CODEEVOLVER_HISTORY_DIR")
    if explicit:
        return explicit + _job_subdir()
    if is_local():
        return str(_local_root() / "history") + _job_subdir()
    return MODAL_HISTORY_DIR + _job_subdir()


def architect_traces_dir() -> str:
    """Directory holding per-iteration IterationArchitect *session* traces.

    These are CodeEvolver meta-traces — the architect's own thinking, tool
    calls, and tool results captured from the Claude Agent SDK message stream
    (``iteration_{N}_architect_trace/``). They are deliberately kept in a
    dedicated directory that is a *sibling* of ``traces_dir()`` (which holds
    the client/Task-Agent OTel traces) so the two never mix: the architect
    meta-traces describe how CodeEvolver behaved, not how the client program
    behaved. The directory is created on demand by the writer
    (``architect_trace.append_architect_turn``); no volume/mount is
    required.
    """
    explicit = os.environ.get("CODEEVOLVER_ARCHITECT_TRACES_DIR")
    if explicit:
        return explicit + _job_subdir()
    if is_local():
        return str(_local_root() / "architect_traces") + _job_subdir()
    return MODAL_ARCHITECT_TRACES_DIR + _job_subdir()


def logs_dir() -> str:
    """Directory holding per-job human-readable log files.

    Resolves like the other dir helpers (env override → local root →
    non-local default) but, unlike them, is **not** suffixed with ``_job_subdir()``:
    the detailed agent log carries the job id in its *filename*
    (``detailed_agent_log_path()``), so a flat per-job directory is enough.
    """
    explicit = os.environ.get("CODEEVOLVER_LOGS_DIR")
    if explicit:
        return explicit
    if is_local():
        return str(_local_root() / "logs")
    return MODAL_LOGS_DIR


def detailed_agent_log_path() -> str:
    """Absolute path of the per-job detailed agent log file.

    ``<logs_dir>/<job_id>_detailed_agent_log.log`` where ``<job_id>`` is
    ``CODEEVOLVER_JOB_ID`` (set by ``optimizer/v3/_entry.py``; its value already
    looks like ``job_<12 hex chars>``). When the job id is unset (ad-hoc local
    runs, tests) the file is simply ``detailed_agent_log.log`` so the path is
    always well-defined. The caller (``logger.get_detailed_agent_logger``)
    creates the parent directory on demand.
    """
    job_id = os.environ.get("CODEEVOLVER_JOB_ID", "").strip()
    name = f"{job_id}_detailed_agent_log.log" if job_id else "detailed_agent_log.log"
    return str(Path(logs_dir()) / name)


def clean_slate_report_path() -> str:
    """Absolute path of the start-of-run clean-slate inspector's report.

    ``<logs_dir>/<job_id>_clean_slate_report.md``, following
    ``detailed_agent_log_path()`` exactly — same directory, same ``<job_id>_``
    filename convention, and ``logs_dir()`` is deliberately not job-suffixed.

    It lives in ``logs/`` rather than ``memory_dir()`` **because the architect
    reads memory_dir**. The inspector walks the repository with a shell, and
    unlike the run-integrity report (written after the loop ends) this one
    exists before the architect's first turn — so a finding that quoted a data
    file would otherwise be one ``Read`` away. ``logs/`` is collected by
    ``orchestrator/collect.py`` just like memory, so the report stays durable
    and reaches the controller either way.
    """
    job_id = os.environ.get("CODEEVOLVER_JOB_ID", "").strip()
    name = f"{job_id}_clean_slate_report.md" if job_id else "clean_slate_report.md"
    return str(Path(logs_dir()) / name)


def runtime_tests_dir() -> str:
    """Directory holding inspection-agent runtime-test JSON state.

    The non-local default ``/app/runtime_tests`` is not writable on macOS
    (read-only root filesystem). Local deploys land it under
    ``~/.codeevolver/`` so behavior trackers persist across iterations.
    No env var override — there's no reason this would ever need one
    beyond the SANDBOX_MODE-driven default.
    """
    if is_local():
        return str(_local_root() / "runtime_tests") + _job_subdir()
    return "/app/runtime_tests" + _job_subdir()


def engine_root() -> str:
    """Directory whose subtree contains ``src/engine/...``.

    Auto-detected from this module's own location at import time — no env
    var, no ``SANDBOX_MODE`` branch. ``paths.py`` always lives at
    ``<root>/src/engine/paths.py``, so ``<root>`` is the answer in both
    a container (``/app``) and a local clone (wherever you cloned it).
    """
    return _ENGINE_ROOT


def repo_venv_dirname() -> str:
    """Client venv directory name *relative to the workspace*.

    Defaults to ``.venv`` when non-local
    and ``env`` on local (the conventional name when an operator runs
    ``python3.11 -m venv env`` in their repo).
    """
    explicit = os.environ.get("REPO_VENV_DIR")
    if explicit:
        return explicit
    return "env" if is_local() else ".venv"


def repo_venv_bin() -> str:
    """Subdirectory of the venv that holds executables (``bin`` everywhere we target)."""
    return os.environ.get("REPO_VENV_BIN", "bin")


def opencode_small_model(default: str = "openai/gpt-4.1-mini") -> str:
    """OpenCode small-model selector used in generated OpenCode config."""
    return os.environ.get("CODEEVOLVER_OPENCODE_SMALL_MODEL", default)


def agent_user() -> str | None:
    """Unprivileged OS user that agent subprocesses must run as, or None.

    Set (to ``ce-agent``) only in the research Docker runner image
    (``runner/Dockerfile.base``), where the held-out testset is root-owned
    mode 600 outside the workspace so the kernel — not tool-level deny
    paths — enforces the holdout. When unset (every local / non-Docker
    deploy) agent subprocesses spawn exactly as before: no user switching,
    no permission handling. Spawners must NOT fall back silently if
    switching fails — a run that quietly drops enforcement is worse than a
    failed run.
    """
    value = os.environ.get("CODEEVOLVER_AGENT_USER", "").strip()
    return value or None


def architect_mcp_proxy_url() -> str:
    """Local OpenCode architect MCP proxy URL for subprocess consumers."""
    return os.environ["CODEEVOLVER_ARCHITECT_MCP_PROXY_URL"].rstrip("/")


def architect_mcp_proxy_token() -> str:
    """Bearer token for the local OpenCode architect MCP proxy."""
    return os.environ["CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN"]


# Default proxy→control-server POST /call timeout (seconds). Long because a
# single architect tool call (run_eval / coding_agent / inspect_traces) can run
# for hours; the loopback connection won't idle-drop. Mirrors the OpenCode
# experimental.mcp_timeout (6h) the harness sets.
_DEFAULT_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS = 6 * 60 * 60


# Centre of the smooth curve that governs how likely the Pareto-Evolution
# optimizer is to enter its exploit-and-merge phase on a given iteration. The
# per-iteration probability is `min(0.5, 0.5*(1 + tanh(k*(budget - threshold))))`
# where `budget` is the eval-row fraction spent; at `budget == threshold` the
# probability is exactly 0.5 (its cap).
_DEFAULT_EVO_EXPLOIT_THRESHOLD = 0.75

# Steepness of that curve. Larger `k` makes the transition around the threshold
# sharper (closer to the old hard step); smaller `k` spreads it out.
_DEFAULT_EVO_EXPLOIT_K = 7.0


def evo_exploit_threshold() -> float:
    """Curve centre for the Pareto-Evolution exploit-and-merge probability.

    Paired with `evo_exploit_k()`, this is the eval-budget fraction at which a
    given iteration has a 50% chance (the cap) of entering the exploit phase —
    the EVO loop preselecting the *best* candidate (instead of Pareto-proportional
    sampling) while the architect prompt surfaces the `### Combine Approaches
    (end of run)` merge note. Both flip together via a single per-iteration draw
    (see `pareto_evolution/select_candidate.py::preselect_pareto_parent`, which
    stamps `IterationContext.exploit_phase` for `prompts._evo_merge_note` to read).
    Read from ``CODEEVOLVER_EVO_EXPLOIT_THRESHOLD``; values that fail to parse or
    fall outside ``(0, 1]`` fall back to the 0.75 default.
    """
    raw = os.environ.get("CODEEVOLVER_EVO_EXPLOIT_THRESHOLD")
    if not raw:
        return _DEFAULT_EVO_EXPLOIT_THRESHOLD
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_EVO_EXPLOIT_THRESHOLD
    return value if 0 < value <= 1 else _DEFAULT_EVO_EXPLOIT_THRESHOLD


def evo_exploit_k() -> float:
    """Steepness of the Pareto-Evolution exploit-and-merge probability curve.

    The tanh argument is `k * (budget - evo_exploit_threshold())`, so larger
    `k` sharpens the transition around the threshold. Read from
    ``CODEEVOLVER_EVO_EXPLOIT_K``; values that fail to parse or are ``<= 0``
    fall back to the 7.0 default.
    """
    raw = os.environ.get("CODEEVOLVER_EVO_EXPLOIT_K")
    if not raw:
        return _DEFAULT_EVO_EXPLOIT_K
    try:
        value = float(raw)
    except ValueError:
        return _DEFAULT_EVO_EXPLOIT_K
    return value if value > 0 else _DEFAULT_EVO_EXPLOIT_K


def architect_mcp_call_timeout_seconds() -> int:
    """Loopback POST /call timeout (s) the OpenCode MCP stdio proxy uses for
    tool execution. Read from the env var the harness writes, falling back to
    the 6h default when unset (e.g. proxy run outside the harness)."""
    raw = os.environ.get("CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS")
    if not raw:
        return _DEFAULT_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS
    try:
        value = int(raw)
    except ValueError:
        return _DEFAULT_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS
    return value if value > 0 else _DEFAULT_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS


__all__ = [
    "traces_dir",
    "memory_dir",
    "history_dir",
    "architect_traces_dir",
    "clean_slate_report_path",
    "runtime_tests_dir",
    "engine_root",
    "repo_venv_dirname",
    "repo_venv_bin",
    "opencode_small_model",
    "agent_user",
    "evo_exploit_threshold",
    "evo_exploit_k",
    "architect_mcp_proxy_url",
    "architect_mcp_proxy_token",
    "architect_mcp_call_timeout_seconds",
]
