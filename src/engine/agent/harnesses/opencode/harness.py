"""OpenCode implementation surface for the CodeEvolver coding harness."""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import textwrap
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

from ..base import AgentRole, HarnessRequest, HarnessResult
from ..skills import stage_codeevolver_skills_to
from .architect_mcp import (
    InertOpenCodeArchitectControlServer,
    OpenCodeArchitectControlServer,
    OpenCodeArchitectToolBridge,
)
from .commands import (
    OpenCodeSessionSpec,
    build_opencode_architect_command,
    build_opencode_run_command,
)
from .output import (
    _merge_result_turns,
    log_opencode_events,
    parse_opencode_output,
    serialize_opencode_export,
)
from .permissions import (
    build_architect_permission_config,
    build_coding_permission_config,
    build_inspection_permission_config,
)
from src.engine.paths import agent_user, memory_dir, opencode_small_model

logger = logging.getLogger(__name__)

# Set once per process by `_verify_opencode_version_once` so the
# `opencode --version` subprocess runs a single time, not per iteration.
_OPENCODE_VERSION_CHECKED = False

OPENCODE_BOOTSTRAP_COMMANDS = ("/s",)
OPENCODE_CODE_CHANGE_FINALIZATION_COMMANDS = ("/update-specs",)
OPENCODE_CODING_TIMEOUT_SECONDS_PER_TURN = 120
OPENCODE_INSPECTION_TIMEOUT_SECONDS_PER_TURN = 120
OPENCODE_ARCHITECT_TIMEOUT_SECONDS_PER_TURN = 120
OPENCODE_EXPORT_TIMEOUT_SECONDS = 60  # `opencode export` is a local DB read
# Proxy spawn + loopback tools/list handshake; generous for a cold interpreter start.
OPENCODE_ARCHITECT_MCP_PREFLIGHT_TIMEOUT_SECONDS = 60
# Absolute path to the proxy script. OpenCode spawns the MCP server with cwd set
# to the CLIENT workspace, and `python -m <pkg.mod>` prepends that cwd to
# sys.path AHEAD of PYTHONPATH. A client repo that ships its own `src/` package
# (the common case — see the namespace-merge note in requirements.md) therefore
# shadows CodeEvolver's `src`, and the proxy dies instantly with
# "No module named 'src.engine'". OpenCode logs only
# `server unavailable key=ce status=failed` and continues, so the architect
# silently runs the whole iteration with ZERO ce tools. Invoking the script by
# absolute path (plus PYTHONSAFEPATH) keeps the client's cwd off sys.path.
OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT = str(
    Path(__file__).resolve().parents[1] / "opencode_mcp_proxy.py"
)

# Long tool calls — `run_eval` can run for HOURS (`subsample_timeout` is a
# generous backstop, not a per-call bound), `coding_agent` ~1-2 min, and
# `inspect_traces` can exceed a minute on large trace files — require lifting
# THREE separate opencode timeouts, none of which `experimental.mcp_timeout`
# alone covers:
#
# 1. `experimental.mcp_timeout` (top-level) — what opencode passes to the MCP
#    SDK's `callTool()`. The per-server `mcp.<name>.timeout` is documented as
#    "MCP server requests" but in practice gates discovery; the binary's
#    tool-exec resolution still reads it first (`config[server].timeout ??
#    experimental.mcp_timeout`), so we set BOTH high (opencode #8701 / #8121).
# 2. The provider streaming timeout — opencode's Vercel-AI-SDK `streamText`
#    step / provider `chunkTimeout`/`headerTimeout`/`requestTimeout` wraps ALL
#    tool execution (bash, MCP, native). On opencode <1.17.11 this is a
#    hardcoded ~300s `requestTimeout` with NO knob — `experimental.mcp_timeout`
#    can't override it (opencode #25509), so a >5min `run_eval` was cut at
#    ~301s (architect session ended; eval finished orphaned; iteration skipped).
#    opencode >=1.17.11 makes it configurable via `provider.<id>.options.*`
#    (see `_provider_stream_timeout_config`); we require that version.
#
# The ce MCP server is a LOCAL stdio server (it spawns `opencode_mcp_proxy.py`).
# stdio removes the remote-HTTP idle-timeout class of cut, but NOT the provider
# `streamText` cap above — that one is transport-independent and is the reason
# the version gate + provider config exist. Set all of these generously; the
# eval-row and wall-clock budgets are the real run bounds.
OPENCODE_ARCHITECT_MCP_TIMEOUT_MS = 6 * 60 * 60 * 1000  # 6 hours
# Provider streaming timeout (ms) injected as `provider.<id>.options.{chunkTimeout,
# headerTimeout, requestTimeout}`. Mirrors the 6h MCP timeout. This is the knob
# that actually lets a >5min tool call survive on opencode >=1.17.11.
OPENCODE_PROVIDER_STREAM_TIMEOUT_MS = OPENCODE_ARCHITECT_MCP_TIMEOUT_MS
# Minimum opencode version. <1.17.11 hardcodes the ~300s provider timeout with no
# override (opencode #25509), so long evals are silently aborted — fail fast.
OPENCODE_MINIMUM_VERSION = (1, 17, 11)
# Loopback POST /call timeout (s) the stdio proxy uses for tool execution.
# Mirrors OPENCODE_ARCHITECT_MCP_TIMEOUT_MS; the proxy reads it from the env var
# below (`paths.architect_mcp_call_timeout_seconds`). Loopback won't idle-drop, so
# this only bounds a genuinely hung control server, well under the 6h mcp_timeout.
OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS = OPENCODE_ARCHITECT_MCP_TIMEOUT_MS // 1000


class OpenCodeHarnessNotImplementedError(NotImplementedError):
    """Raised for execution paths deferred to later OpenCode issues."""


class OpenCodeHarness:
    """OpenCode harness.

    Coding, inspection, and architect roles run through ``opencode run``.
    Architect tools are exposed through a local MCP proxy backed by the
    optimizer-owned Python tool registry.
    """

    def __init__(
        self,
        *,
        model: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
        architect_context: Any | None = None,
        architect_config: Any | None = None,
        architect_tool_registry: dict[str, Callable[..., dict]] | None = None,
    ) -> None:
        self.model = model
        self._runner = runner
        self.architect_context = architect_context
        self.architect_config = architect_config
        self.architect_tool_registry = dict(architect_tool_registry or {})
        # Architect meta-trace capture state (capture_architect_traces only):
        # session ids seen per iteration (normally one; a resume after a failed
        # session-id extraction can start a second), and the iterations whose
        # trace file already holds real exported turns (guards the stub
        # fallback on a later no-op capture).
        self._trace_session_ids: dict[int, list[str]] = {}
        self._trace_written: set[int] = set()

    def build_session_spec(self, request: HarnessRequest) -> OpenCodeSessionSpec:
        self._require_supported_role(request)
        opencode_config_dir = _prepare_opencode_config_dir()
        commit_changes = bool(
            dict(request.raw_provider_metadata or {}).get("commit_changes", False)
        )
        # stage_project_skills=False runs OpenCode with no
        # CodeEvolver skills staged, no `/s` study-specs bootstrap, and no
        # `/update-specs` finalization — a clean coding loop with none of the
        # CodeEvolver machinery. Defaults to True (current behavior) for
        # EE / freeform / pareto and whenever no architect_config is present.
        stage_skills = bool(getattr(self.architect_config, "stage_project_skills", True))
        if stage_skills:
            skill_dir = stage_codeevolver_skills_to(
                opencode_config_dir / "skills",
                workspace=request.workspace,
                skills=getattr(self.architect_config, "skills_to_stage", None),
                overrides=getattr(self.architect_config, "skill_overrides", None),
                extra_values=getattr(
                    self.architect_config, "skill_template_values", None
                ),
            )
            bootstrap_commands = list(OPENCODE_BOOTSTRAP_COMMANDS)
            finalization_commands = list(
                OPENCODE_CODE_CHANGE_FINALIZATION_COMMANDS if commit_changes else ()
            )
        else:
            skill_dir = opencode_config_dir / "skills"
            skill_dir.mkdir(parents=True, exist_ok=True)
            bootstrap_commands = []
            finalization_commands = []
        return OpenCodeSessionSpec(
            cwd=str(request.workspace),
            prompt=request.prompt,
            model=self.model,
            role=request.role,
            skill_dir=skill_dir,
            migration_fallback_skill_dir=Path(request.workspace) / ".claude" / "skills",
            uses_migration_fallback=False,
            bootstrap_commands=bootstrap_commands,
            finalization_commands=finalization_commands,
            autonomous_permissions=request.permissions.autonomous,
        )

    def build_coding_command(self, request: HarnessRequest) -> tuple[list[str], str]:
        self._require_role(request, AgentRole.CODING)
        spec = self.build_session_spec(request)
        return build_opencode_run_command(spec), spec.cwd

    def build_inspection_command(
        self, request: HarnessRequest
    ) -> tuple[list[str], str, dict[str, Any]]:
        self._require_role(request, AgentRole.INSPECTION)
        spec = self.build_session_spec(request)
        metadata = dict(request.raw_provider_metadata or {})
        permission_config = build_inspection_permission_config(
            request.workspace,
            writable=bool(metadata.get("writable", False)),
            deny_paths=metadata.get("deny_paths") or [],
        )
        return build_opencode_run_command(spec), spec.cwd, permission_config

    def run_architect(self, request: HarnessRequest) -> HarnessResult:
        return self._run_architect_session(request)

    def resume_architect(
        self, session_id: str, request: HarnessRequest
    ) -> HarnessResult:
        return self._run_architect_session(request, resume_session_id=session_id)

    def run_coding(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.CODING)
        spec = self.build_session_spec(request)
        metadata = dict(request.raw_provider_metadata or {})
        deny_paths = list(metadata.get("deny_paths") or [])
        permission_config = build_coding_permission_config(deny_paths)
        cmd = build_opencode_run_command(spec)
        self._verify_binary_if_needed()
        result = self._run_command(
            cmd,
            cwd=spec.cwd,
            timeout=(request.max_turns or 50) * OPENCODE_CODING_TIMEOUT_SECONDS_PER_TURN,
            env_overrides=self._build_opencode_env_overrides(
                config_dir=spec.skill_dir.parent,
                extra_config={
                    "permission": permission_config,
                    **_provider_stream_timeout_config(spec.model),
                },
                extra_env={"OPENCODE_PERMISSION": json.dumps(permission_config)},
            ),
        )
        output, session_id, parsed_metadata = parse_opencode_output(result.stdout)
        self._attach_usage_model(parsed_metadata, spec.model)
        raw_provider_metadata: dict[str, Any] = {
            "provider": "opencode",
            "role": "coding",
            "returncode": result.returncode,
            "stderr": result.stderr,
            "commit_handled": False,
            "deny_paths": deny_paths,
            "permission": permission_config,
            **parsed_metadata,
        }
        if result.returncode != 0:
            error_detail = result.stderr.strip() or result.stdout.strip()
            return HarnessResult.failed(
                error=(
                    f"OpenCode exited with code {result.returncode}: {error_detail}"
                ),
                output=output or result.stdout,
                session_id=session_id,
                raw_provider_metadata=raw_provider_metadata,
            )
        return HarnessResult.successful(
            output=output or result.stdout,
            session_id=session_id,
            raw_provider_metadata=raw_provider_metadata,
        )

    def run_inspection(self, request: HarnessRequest) -> HarnessResult:
        self._require_role(request, AgentRole.INSPECTION)
        spec = self.build_session_spec(request)
        metadata = dict(request.raw_provider_metadata or {})
        writable = bool(metadata.get("writable", False))
        as_agent_user = bool(metadata.get("run_as_agent_user", True))
        # Every inspection session gets its own OpenCode data dir. The global
        # store under $HOME is shared by every session in a run and its SQLite
        # files are created 0644 by whichever uid ran first (ce-agent); the
        # run-integrity inspector is the one session that runs as ROOT, and
        # under the Docker runner's --cap-drop=ALL root has no CAP_DAC_OVERRIDE,
        # so it would open that db read-only and die on OpenCode's startup
        # `PRAGMA wal_checkpoint(PASSIVE)`. Inspection sessions are never
        # resumed or exported (`_capture_architect_trace` needs an architect
        # context), so nothing reads their store afterwards; architect/coding
        # sessions keep the
        # global store, which session resume and trace export depend on.
        spec = replace(spec, data_dir=_prepare_opencode_data_dir(request.role))
        logger.info("OpenCode inspection session data dir: %s", spec.data_dir)
        permission_config = build_inspection_permission_config(
            request.workspace,
            writable=writable,
            deny_paths=metadata.get("deny_paths") or [],
            allow_bash=bool(metadata.get("allow_bash", False)),
        )
        if writable:
            # The writable inspection agent (valset reflection) edits
            # memory.md. OpenCode's apply_patch — which gpt-5.x-codex defaults
            # to and will not abandon — can only edit files inside its working
            # dir, and /memory is a sibling of the workspace. Run it from the
            # memory dir so apply_patch's target is in-tree. It needs nothing
            # from the workspace: the valset-reflection prompt references only
            # the memory and trace dirs, both read via absolute paths, and the
            # permission policy is keyed on absolute paths so it is unaffected.
            spec = replace(spec, cwd=str(memory_dir()))
        cmd = build_opencode_run_command(spec)
        self._verify_binary_if_needed()
        turn_based_timeout = (
            (request.max_turns or 20) * OPENCODE_INSPECTION_TIMEOUT_SECONDS_PER_TURN
        )
        # Callers that need a hard wall-clock ceiling (e.g. the analysis
        # agent, see analysis_agent.py) pass timeout_seconds explicitly; every
        # other inspection caller keeps the turn-based timeout unchanged.
        # hard_kill only turns on alongside an explicit override, so nothing
        # about resumed/architect sessions or other inspection calls changes.
        timeout_override = metadata.get("timeout_seconds")
        timeout = (
            min(turn_based_timeout, timeout_override)
            if timeout_override
            else turn_based_timeout
        )
        result = self._run_command(
            cmd,
            cwd=spec.cwd,
            as_agent_user=as_agent_user,
            timeout=timeout,
            hard_kill=bool(timeout_override),
            env_overrides=self._build_opencode_env_overrides(
                config_dir=spec.skill_dir.parent,
                extra_config={
                    "permission": permission_config,
                    **_provider_stream_timeout_config(spec.model),
                },
                extra_env={
                    "OPENCODE_PERMISSION": json.dumps(permission_config),
                    **_data_dir_env(spec.data_dir),
                },
            ),
        )
        output, session_id, parsed_metadata = parse_opencode_output(result.stdout)
        self._attach_usage_model(parsed_metadata, spec.model)
        raw_provider_metadata: dict[str, Any] = {
            "provider": "opencode",
            "role": "inspection",
            "data_dir": str(spec.data_dir) if spec.data_dir else None,
            "returncode": result.returncode,
            "stderr": result.stderr,
            "writable": writable,
            "deny_paths": list(metadata.get("deny_paths") or []),
            "permission": permission_config,
            **parsed_metadata,
        }
        # An inspection's "output" is the agent's text and nothing else. When
        # stdout was an OpenCode event stream that carried no text (the
        # session died in bootstrap and emitted only `{"type":"error",…}`),
        # `output` is "" and must STAY "" -- falling back to raw stdout here
        # would hand the auditors an error event as their "report". Plain
        # non-JSON stdout is already `output` as parsed.
        # (A WRITABLE inspection is judged by its writes, not its text --
        # the valset-reflection agent legitimately says nothing.)
        error_events = parsed_metadata.get("error_events") or []
        if result.returncode != 0 or (
            not writable and not output.strip() and parsed_metadata.get("json_events")
        ):
            error_detail = (
                result.stderr.strip()
                or "; ".join(error_events)
                or result.stdout.strip()[:2000]
            )
            return HarnessResult.failed(
                error=(
                    f"OpenCode exited with code {result.returncode}"
                    + (" and produced no text" if not output.strip() else "")
                    + f": {error_detail}"
                ),
                output=output,
                session_id=session_id,
                raw_provider_metadata=raw_provider_metadata,
            )
        if writable:
            if not self._observed_write(parsed_metadata):
                return HarnessResult.failed(
                    error="Writable inspection produced no writes",
                    output="No change proposed",
                    session_id=session_id,
                    raw_provider_metadata=raw_provider_metadata,
                )
            return HarnessResult.successful(
                output="Memory updated",
                session_id=session_id,
                raw_provider_metadata=raw_provider_metadata,
            )
        return HarnessResult.successful(
            output=output,
            session_id=session_id,
            raw_provider_metadata=raw_provider_metadata,
        )

    @staticmethod
    def _attach_usage_model(
        parsed_metadata: dict[str, Any], fallback_model: str | None
    ) -> None:
        """Resolve the token-usage model and surface it for both consumers.

        Token accounting reads the model from two places: the architect path
        from top-level `raw_provider_metadata["model"]`, the coding/inspection
        path from `raw_provider_metadata["usage"]["model"]`. We prefer the bare
        `modelID` parsed from stdout and fall back to the requested model
        (`spec.model`, the `coding_harness_model` passed to `--model`) so the
        ledger bucket is always a real label, never `"unknown"`.
        """
        resolved = parsed_metadata.get("model") or fallback_model
        usage = parsed_metadata.get("usage")
        if isinstance(usage, dict):
            usage["model"] = usage.get("model") or resolved
        parsed_metadata["model"] = resolved

    @staticmethod
    def _require_supported_role(request: HarnessRequest) -> None:
        if request.role not in {
            AgentRole.ARCHITECT,
            AgentRole.CODING,
            AgentRole.INSPECTION,
        }:
            raise ValueError(f"Unsupported OpenCode harness role: {request.role!r}")

    @staticmethod
    def _require_role(request: HarnessRequest, expected: AgentRole) -> None:
        if request.role != expected:
            raise ValueError(
                f"{expected.value} OpenCode harness call received "
                f"{request.role.value!r} request"
            )

    def _run_command(
        self,
        cmd: list[str],
        *,
        cwd: str,
        timeout: int,
        env_overrides: dict[str, str] | None = None,
        as_agent_user: bool = True,
        hard_kill: bool = False,
    ) -> subprocess.CompletedProcess[str]:
        runner = self._runner or subprocess.run
        env = self._build_env()
        if env_overrides:
            env.update(env_overrides)
        agent_user_kwargs = _agent_user_kwargs() if as_agent_user else {}
        if not hard_kill or self._runner is not None:
            return runner(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                # Docker runner only (env-gated, empty elsewhere): agent
                # sessions run as the unprivileged user so the kernel enforces
                # the holdout fence. `as_agent_user=False` is the privileged
                # exception for the run-integrity inspector, which must read
                # the testset. No fallback on failure — see paths.agent_user().
                **agent_user_kwargs,
            )
        # hard_kill: plain subprocess.run(timeout=...) only signals the one
        # PID it tracks. OpenCode can spawn its own child processes (e.g. a
        # glob search shelling out to `rg`) that survive that kill as
        # orphans, leaving the parent Popen call blocked indefinitely on
        # communicate(), well past its nominal timeout. start_new_session
        # puts the whole tree in one killable process group; on timeout we
        # SIGKILL the group, not just the head. Opt-in only — every other
        # caller of _run_command (architect sessions, session resumes,
        # non-analysis inspection calls) is unaffected.
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            start_new_session=True,
            **agent_user_kwargs,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            stdout, stderr = proc.communicate()
            raise subprocess.TimeoutExpired(cmd, timeout, output=stdout, stderr=stderr)
        return subprocess.CompletedProcess(cmd, proc.returncode, stdout, stderr)

    def _run_architect_session(
        self,
        request: HarnessRequest,
        *,
        resume_session_id: str | None = None,
    ) -> HarnessResult:
        self._require_role(request, AgentRole.ARCHITECT)
        config = self._require_architect_config()
        spec = self.build_session_spec(request)
        bridge = OpenCodeArchitectToolBridge(
            self.architect_tool_registry,
            agent_managed_git=bool(getattr(config, "agent_managed_git", True)),
            close_iteration_mode=getattr(
                config, "close_iteration_mode", "register_branch"
            ),
        )
        opencode_config_dir = spec.skill_dir.parent
        instruction_path = self._write_architect_system_prompt(
            request, opencode_config_dir
        )

        # Holdout (valset/test) fence fragments — threaded into BOTH the opencode
        # permission policy (primary fence) and the pre-edit guard (defense-in-
        # depth, including apply_patch target-path checks).
        holdout_deny_paths = _architect_holdout_deny_paths(config)

        control_server = (
            InertOpenCodeArchitectControlServer(
                bridge,
                pre_edit_guard=self._build_pre_edit_guard(spec.cwd, holdout_deny_paths),
            )
            if self._runner is not None
            else OpenCodeArchitectControlServer(
                bridge,
                pre_edit_guard=self._build_pre_edit_guard(spec.cwd, holdout_deny_paths),
            )
        )
        with control_server as control:
            env_overrides = self._build_architect_env_overrides(
                control=control,
                bridge=bridge,
                config_dir=opencode_config_dir,
                instruction_path=instruction_path,
                deny_paths=holdout_deny_paths,
            )
            self._preflight_architect_mcp(
                env_overrides=env_overrides, bridge=bridge, cwd=spec.cwd
            )
            cmd = build_opencode_architect_command(
                spec,
                resume_session_id=resume_session_id,
            )
            self._verify_binary_if_needed()
            result = self._run_command(
                cmd,
                cwd=spec.cwd,
                timeout=(request.max_turns or config.max_turns)
                * OPENCODE_ARCHITECT_TIMEOUT_SECONDS_PER_TURN,
                env_overrides=env_overrides,
            )

        output, session_id, parsed_metadata = parse_opencode_output(result.stdout)

        self._log_detailed_architect_events(result.stdout)

        # Architect meta-trace capture for post-run meta-reflection: export the
        # session (`opencode export <id>` — a consolidated snapshot with
        # reasoning parts, tool state/output, and step tokens) and rewrite the
        # iteration's trace file. Runs after every invocation (initial + each
        # watchdog resume) — the export is idempotent, so rewrite-on-capture
        # converges to the complete session. Falls back to the `captured:false`
        # stub when no session id / export output exists. Best-effort.
        if getattr(config, "capture_architect_traces", False):
            self._capture_architect_trace(
                session_id=session_id or resume_session_id,
                stdout=result.stdout,
                cwd=spec.cwd,
            )

        self._attach_usage_model(parsed_metadata, spec.model)
        raw_provider_metadata: dict[str, Any] = {
            "provider": "opencode",
            "role": "architect",
            "returncode": result.returncode,
            "stderr": result.stderr,
            "resume_session_id": resume_session_id,
            "mcp_tools": [tool["name"] for tool in bridge.list_tools()],
            **parsed_metadata,
        }
        if result.returncode != 0:
            error_detail = result.stderr.strip() or result.stdout.strip()
            return HarnessResult.failed(
                error=(
                    f"OpenCode exited with code {result.returncode}: {error_detail}"
                ),
                output=output or result.stdout,
                session_id=session_id or resume_session_id,
                raw_provider_metadata=raw_provider_metadata,
            )
        return HarnessResult.successful(
            output=output or result.stdout,
            session_id=session_id or resume_session_id,
            raw_provider_metadata=raw_provider_metadata,
        )

    def _preflight_architect_mcp(
        self,
        *,
        env_overrides: dict[str, str],
        bridge: OpenCodeArchitectToolBridge,
        cwd: str,
    ) -> None:
        """Verify the `ce` MCP server actually starts and lists its tools.

        OpenCode is silent about a dead local MCP server — it logs
        `server unavailable key=ce status=failed` to its own logfile and runs
        the session anyway, so the architect spends a full (billable) iteration
        with ZERO ce tools: no run_eval, no coding_agent, no close_iteration.
        That failure is invisible to the optimizer (the session "succeeds") and
        corrupts the run, so it must be caught before the model is invoked.

        This spawns the proxy exactly as OpenCode will — same argv, same
        `environment` dict, same cwd (the CLIENT workspace) — and completes an
        `initialize` + `tools/list` handshake. Using ONLY the configured
        environment is deliberate: it is the strictest case, so passing here
        holds whether or not OpenCode merges its own env into the child.

        Raises RuntimeError on any failure; deterministic (process spawn plus a
        loopback handshake), so a failure is a real misconfiguration rather than
        a flake worth retrying.
        """
        if self._runner is not None:
            return  # injected-runner test path: no real subprocess to spawn
        server = json.loads(env_overrides["OPENCODE_CONFIG_CONTENT"])["mcp"]["ce"]
        expected = {tool["name"] for tool in bridge.list_tools()}
        handshake = "".join(
            json.dumps(message) + "\n"
            for message in (
                {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            )
        )
        try:
            completed = subprocess.run(
                server["command"],
                input=handshake.encode("utf-8"),
                capture_output=True,
                env=dict(server["environment"]),
                cwd=cwd,
                timeout=OPENCODE_ARCHITECT_MCP_PREFLIGHT_TIMEOUT_SECONDS,
                # Mirror the real spawn conditions: OpenCode (running as the
                # agent user) spawns the proxy, so preflight must too.
                **_agent_user_kwargs(),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                "OpenCode architect MCP preflight timed out after "
                f"{OPENCODE_ARCHITECT_MCP_PREFLIGHT_TIMEOUT_SECONDS}s — the "
                "`ce` tool server did not answer tools/list. The architect "
                "would run with no run_eval / close_iteration tools."
            ) from exc
        listed: set[str] = set()
        for line in completed.stdout.decode("utf-8", errors="replace").splitlines():
            try:
                tools = json.loads(line).get("result", {}).get("tools")
            except json.JSONDecodeError:
                continue
            if isinstance(tools, list):
                listed = {
                    str(tool.get("name")) for tool in tools if isinstance(tool, dict)
                }
        missing = expected - listed
        if missing:
            stderr_tail = completed.stderr.decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "OpenCode architect MCP preflight failed: the `ce` tool server "
                f"did not list {sorted(missing)} (listed: {sorted(listed) or 'none'}). "
                "OpenCode would log only 'server unavailable key=ce' and run the "
                "iteration with no architect tools. "
                f"proxy rc={completed.returncode}; stderr: {stderr_tail[-1000:] or '(empty)'}"
            )
        logger.info(
            "OpenCode architect MCP preflight ok: %s", ", ".join(sorted(listed))
        )

    def _capture_architect_trace(
        self, *, session_id: str | None, stdout: str, cwd: str
    ) -> None:
        """Export the OpenCode session(s) and rewrite the iteration's architect
        meta-trace JSONL (`architect_trace.write_opencode_trace`).

        Best-effort: capture must never sink an architect invocation. Each
        capture re-exports every session id recorded for the iteration (in
        order) so the rewritten file always reflects the full session history,
        including watchdog resumes. Runs on the injected-``_runner`` (test)
        path too — tests dispatch on ``cmd[:2] == ["opencode", "export"]``.
        """
        try:
            iteration_idx = getattr(
                getattr(self, "architect_context", None), "iteration_idx", None
            )
            if iteration_idx is None:
                return
            from src.engine.optimizer.v3.architect_trace import (
                append_opencode_stub,
                write_opencode_trace,
            )

            session_ids = self._trace_session_ids.setdefault(iteration_idx, [])
            if session_id and session_id not in session_ids:
                session_ids.append(session_id)
            turns: list[dict[str, Any]] = []
            for sid in session_ids:
                export_doc = self._export_opencode_session(sid, cwd=cwd)
                if export_doc is not None:
                    turns.extend(
                        serialize_opencode_export(
                            export_doc, model=self.model, session_id=sid
                        )
                    )
            # Each per-session serialization appends its own trailing
            # `role="result"` turn, so a watchdog resume would leave a result
            # line stranded MID-FILE with only that session's usage —
            # meta_reflection expects one, at the end, carrying the whole
            # iteration. Collapse them into a single summed result line.
            turns = _merge_result_turns(turns, session_ids)
            written = write_opencode_trace(iteration_idx, turns)
            if written:
                self._trace_written.add(iteration_idx)
            elif iteration_idx not in self._trace_written:
                append_opencode_stub(iteration_idx, stdout_tail=stdout)
        except Exception:  # noqa: BLE001 - capture is never load-bearing
            # Non-fatal, but never silent: a bare `pass` here meant a
            # serializer crash produced no trace AND no `captured:false`
            # stub (append_opencode_stub is inside this try), while
            # meta_reflection is still told the file should exist. That
            # class of silent drop took a live run to discover.
            logger.warning(
                "architect trace capture failed for session %s (non-fatal); "
                "this iteration's meta-trace may be missing or stale",
                session_id, exc_info=True,
            )

    def _export_opencode_session(
        self, session_id: str, *, cwd: str
    ) -> dict[str, Any] | None:
        """Run ``opencode export <session_id>`` and parse its stdout JSON.

        The export document goes to stdout (progress note goes to stderr) and
        is read from OpenCode's global session store, so no config-dir env is
        needed. Returns None on any failure (caller degrades to the stub).

        stdout must be redirected to a FILE, not a pipe: opencode (Bun) exits
        without draining a >64KiB pipe, deterministically truncating the
        document at 64KiB (observed on 1.18.19; a file redirect always yields
        the full document). The injected-``_runner`` test path keeps the
        canned-CompletedProcess route.
        """
        cmd = ["opencode", "export", session_id]
        try:
            if self._runner is not None:
                result = self._run_command(
                    cmd, cwd=cwd, timeout=OPENCODE_EXPORT_TIMEOUT_SECONDS
                )
                if result.returncode != 0:
                    logger.warning(
                        "opencode export %s failed (rc=%s): %s",
                        session_id,
                        result.returncode,
                        (result.stderr or "").strip()[:500],
                    )
                    return None
                stdout = result.stdout
            else:
                with tempfile.TemporaryDirectory(
                    prefix="codeevolver-opencode-export-"
                ) as tmp_dir:
                    out_path = Path(tmp_dir) / "export.json"
                    with open(out_path, "w", encoding="utf-8") as out_file:
                        completed = subprocess.run(
                            cmd,
                            cwd=cwd,
                            stdout=out_file,
                            stderr=subprocess.PIPE,
                            text=True,
                            timeout=OPENCODE_EXPORT_TIMEOUT_SECONDS,
                            env=self._build_env(),
                            # Session state under HOME is written by the agent
                            # user; export must read it as the same user.
                            **_agent_user_kwargs(),
                        )
                    if completed.returncode != 0:
                        logger.warning(
                            "opencode export %s failed (rc=%s): %s",
                            session_id,
                            completed.returncode,
                            (completed.stderr or "").strip()[:500],
                        )
                        return None
                    stdout = out_path.read_text(encoding="utf-8")
            doc = json.loads(stdout)
            return doc if isinstance(doc, dict) else None
        except Exception as exc:  # noqa: BLE001 - capture is never load-bearing
            logger.warning("opencode export %s failed: %s", session_id, exc)
            return None

    def _log_detailed_architect_events(self, stdout: str) -> None:
        """Route the architect's tool calls + reasoning text to the per-job
        detailed agent log file (verbose, kept out of the main stdout log).

        OpenCode runs as a subprocess and only yields its tool/text stream in
        the final JSON stdout, so unlike the Claude path there is nothing to log
        live — we parse it once here, post-session, into an ordered transcript.
        Best-effort: observability must never sink an already-captured result.
        Skipped on the injected-`_runner` (test) path to keep tests hermetic.
        """
        if self._runner is not None:
            return
        try:
            iteration_idx = getattr(
                getattr(self, "architect_context", None), "iteration_idx", None
            )
            if iteration_idx is None:
                return
            from src.engine.services.logger import get_detailed_agent_logger

            log_opencode_events(iteration_idx, stdout, get_detailed_agent_logger())
        except Exception:  # noqa: BLE001 - logging is never load-bearing
            pass

    def _write_architect_system_prompt(
        self, request: HarnessRequest, config_dir: Path
    ) -> Path:
        system_prompt = str(request.raw_provider_metadata.get("system_prompt") or "")
        dest = config_dir / "codeevolver"
        dest.mkdir(parents=True, exist_ok=True)
        path = dest / "architect_system_prompt.md"
        path.write_text(system_prompt, encoding="utf-8")
        return path

    def _build_architect_env_overrides(
        self,
        *,
        control: OpenCodeArchitectControlServer,
        bridge: OpenCodeArchitectToolBridge,
        config_dir: Path,
        instruction_path: Path,
        deny_paths: list[str],
    ) -> dict[str, str]:
        _write_pre_edit_guard_plugin(config_dir)
        permission_config = build_architect_permission_config(deny_paths)
        opencode_config = _base_opencode_config()
        opencode_config.update(
            {
                "instructions": [str(instruction_path)],
                # Top-level execution timeout — the field opencode passes to the
                # MCP SDK's callTool(). This is what actually governs how long a
                # run_eval / coding_agent call may run (see the constant's note).
                "experimental": {"mcp_timeout": OPENCODE_ARCHITECT_MCP_TIMEOUT_MS},
                # LOCAL stdio MCP server: opencode spawns the proxy, which
                # forwards tool calls over loopback HTTP to the in-process
                # control server (`control.url`). stdio (not remote HTTP) is
                # deliberate — a stdio pipe has no socket idle timeout, so a long,
                # quiet run_eval blocks like Claude's in-process tool instead of
                # being cut at opencode's ~300s HTTP idle timeout. The proxy reads
                # the URL/token/call-timeout from `environment` (it does not rely
                # on opencode forwarding its own env to the child MCP process).
                "mcp": {
                    "ce": {
                        "type": "local",
                        "command": [
                            sys.executable,
                            OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT,
                        ],
                        "environment": {
                            "PYTHONPATH": _codeevolver_pythonpath(),
                            # Stop Python from prepending the script dir / cwd to
                            # sys.path so ONLY PYTHONPATH resolves `src` — see
                            # OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT. Ignored on
                            # Python < 3.11, where the absolute-path invocation
                            # already keeps the client cwd off sys.path.
                            "PYTHONSAFEPATH": "1",
                            "CODEEVOLVER_ARCHITECT_MCP_PROXY_URL": control.url,
                            "CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN": control.token,
                            "CODEEVOLVER_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS": str(
                                OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS
                            ),
                        },
                        "enabled": True,
                        # Per-server timeout set high (not a short "discovery"
                        # value): the binary reads it first in the tool-exec
                        # timeout resolution, and the proven 1.17.11 config set
                        # it high. Discovery is instant, so this is harmless.
                        "timeout": OPENCODE_ARCHITECT_MCP_TIMEOUT_MS,
                    }
                },
                # Provider streaming timeout — the knob that actually lets a
                # >5min run_eval survive on opencode >=1.17.11 (see the constant
                # note and `_provider_stream_timeout_config`).
                **_provider_stream_timeout_config(self.model),
                "permission": permission_config,
            }
        )
        tool_names = ",".join(tool["name"] for tool in bridge.list_tools())
        return {
            "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config),
            "OPENCODE_CONFIG_DIR": str(config_dir),
            # OpenCode only enforces the permission policy when it is also
            # exported as OPENCODE_PERMISSION (the coding/inspection roles do
            # this). Without it the architect's holdout deny patterns sit
            # inert in the config and the valset/test fence is unarmed.
            "OPENCODE_PERMISSION": json.dumps(permission_config),
            "CODEEVOLVER_ARCHITECT_MCP_PROXY_URL": control.url,
            "CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN": control.token,
            "CODEEVOLVER_ARCHITECT_MCP_TOOL_NAMES": tool_names,
        }

    @staticmethod
    def _build_opencode_env_overrides(
        *,
        config_dir: Path,
        extra_config: dict[str, Any] | None = None,
        extra_env: dict[str, str] | None = None,
    ) -> dict[str, str]:
        opencode_config = _base_opencode_config()
        if extra_config:
            opencode_config.update(extra_config)
        env = {
            "OPENCODE_CONFIG_CONTENT": json.dumps(opencode_config),
            "OPENCODE_CONFIG_DIR": str(config_dir),
        }
        if extra_env:
            env.update(extra_env)
        return env

    def _build_pre_edit_guard(
        self, workspace: str, deny_paths: list[str] | None = None
    ) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
        from src.engine.optimizer.v3.architect_hooks import (
            check_pre_edit_guard_decision,
        )

        context = self.architect_context
        holdout_deny_paths = list(deny_paths or [])

        def guard(tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            if context is None:
                return {"allowed": True, "reason": None}
            decision = check_pre_edit_guard_decision(
                context,
                workspace,
                tool_name=_opencode_tool_name_for_pre_edit_guard(tool_name),
                tool_input=arguments,
                deny_paths=holdout_deny_paths,
            )
            return {"allowed": decision.allowed, "reason": decision.reason}

        return guard

    def _require_architect_config(self) -> Any:
        if self.architect_config is None:
            raise ValueError("OpenCode architect execution requires architect_config")
        return self.architect_config

    def _verify_binary_if_needed(self) -> None:
        if self._runner is not None:
            return
        if shutil.which("opencode") is None:
            raise RuntimeError(
                "OpenCode harness selected but the opencode binary was not found "
                "on PATH. Install OpenCode before running with "
                "coding_harness='opencode'."
            )
        _verify_opencode_version_once()

    @staticmethod
    def _build_env() -> dict[str, str]:
        env = os.environ.copy()
        env["OPENCODE_DISABLE_CLAUDE_CODE_SKILLS"] = "true"
        env["OPENCODE_DISABLE_CLAUDE_CODE"] = "true"
        return env

    @staticmethod
    def _observed_write(parsed_metadata: dict[str, Any]) -> bool:
        for call in parsed_metadata.get("tool_calls") or []:
            name = str(call.get("name") or "").lower()
            if name not in {"edit", "write", "patch", "apply_patch"}:
                continue
            # A write tool that errored (e.g. apply_patch failing to resolve the
            # external memory dir) did not actually write — counting it as a
            # write reports a false "Memory updated" while memory.md is unchanged.
            if str(call.get("status") or "").lower() == "error":
                continue
            return True
        return False


def _architect_holdout_deny_paths(config: Any) -> list[str]:
    """Holdout path fragments the architect must never read or edit.

    Mirrors the fence the coding / inspection sub-agents get (held-out test
    set as absolute path + basename, plus its `_holdout` trace directories).
    Derived from the architect_config so the fence is armed regardless of
    what the caller threads through the request.

    Also fences the architect's own meta-traces (`*_architect_trace/`) — the
    same fragment the Claude architect gets via the holdout guard hook.
    Meaningful now that OpenCode traces carry real session content (export
    capture); only the end-of-run meta-reflection agent may read them.
    """
    from src.engine.optimizer.v3.iteration_architect import (
        ARCHITECT_TRACE_DENY_FRAGMENT,
        _build_holdout_deny_paths,
    )

    evaluator = getattr(config, "evaluator", None)
    deny_paths = _build_holdout_deny_paths(
        getattr(evaluator, "testset_path", None),
        getattr(config, "additional_deny_paths", None),
    )
    deny_paths.append(ARCHITECT_TRACE_DENY_FRAGMENT)
    return deny_paths


def _codeevolver_pythonpath() -> str:
    repo_root = str(Path(__file__).resolve().parents[5])
    current = os.environ.get("PYTHONPATH")
    if current:
        return os.pathsep.join([repo_root, current])
    return repo_root


def _opencode_version_meets_minimum(
    version_str: str, minimum: tuple[int, ...] = OPENCODE_MINIMUM_VERSION
) -> bool | None:
    """Whether `version_str` is >= `minimum`. Returns None if unparseable.

    Accepts a bare `1.17.11` or a noisier `opencode 1.17.11` — the first
    dotted triple is used.
    """
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", version_str or "")
    if not match:
        return None
    return tuple(int(g) for g in match.groups()) >= tuple(minimum)


def _verify_opencode_version_once() -> None:
    """Fail fast on opencode < OPENCODE_MINIMUM_VERSION (once per process).

    Versions below the minimum silently abort long tool calls (`run_eval`) at
    ~300s (opencode #25509), wasting an entire billable run on skipped
    iterations. Parse-failure is a warning (don't block on an unrecognized
    build); `CODEEVOLVER_SKIP_OPENCODE_VERSION_CHECK` bypasses entirely.
    """
    global _OPENCODE_VERSION_CHECKED
    if _OPENCODE_VERSION_CHECKED:
        return
    if os.environ.get("CODEEVOLVER_SKIP_OPENCODE_VERSION_CHECK"):
        _OPENCODE_VERSION_CHECKED = True
        return
    minimum = ".".join(str(p) for p in OPENCODE_MINIMUM_VERSION)
    try:
        proc = subprocess.run(
            ["opencode", "--version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        version_str = (proc.stdout or proc.stderr or "").strip()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "could not run `opencode --version` (%s); skipping version check "
            "(need >= %s)",
            exc,
            minimum,
        )
        _OPENCODE_VERSION_CHECKED = True
        return
    meets = _opencode_version_meets_minimum(version_str)
    if meets is None:
        logger.warning(
            "could not parse opencode version from %r; proceeding (need >= %s)",
            version_str,
            minimum,
        )
        _OPENCODE_VERSION_CHECKED = True
        return
    if not meets:
        raise RuntimeError(
            f"OpenCode {version_str!r} is too old: long tool calls (run_eval) are "
            f"silently aborted at ~300s on versions < {minimum} (opencode #25509), "
            "wasting the run on skipped iterations. Upgrade opencode (e.g. "
            "`npm i -g opencode-ai@latest`), or set "
            "CODEEVOLVER_SKIP_OPENCODE_VERSION_CHECK=1 to bypass."
        )
    logger.info("opencode version %s OK (>= %s)", version_str, minimum)
    _OPENCODE_VERSION_CHECKED = True


def _provider_stream_timeout_config(model: str | None) -> dict[str, Any]:
    """Provider streaming-timeout config block keyed off the opencode model.

    opencode's Vercel-AI-SDK `streamText` wraps ALL tool execution in a provider
    streaming timeout (`chunkTimeout`/`headerTimeout`/`requestTimeout`); on
    opencode >=1.17.11 it is configurable per provider via
    `provider.<id>.options.*`. Without lifting it, a tool call (e.g. `run_eval`)
    exceeding ~300s is aborted regardless of `experimental.mcp_timeout` or the
    stdio transport (opencode #25509). We set it to the same 6h bound as the MCP
    timeout.

    The provider id is the part of the model string before the first `/`
    (`openai/gpt-5.4-mini` -> `openai`, `deepinfra/zai-org/GLM-5.2` ->
    `deepinfra`), which is exactly how opencode resolves the provider. Returns an
    empty dict (no-op) when there is no model or no `/` to key on.
    """
    if not model or "/" not in model:
        return {}
    provider_id = model.split("/", 1)[0]
    t = OPENCODE_PROVIDER_STREAM_TIMEOUT_MS
    return {
        "provider": {
            provider_id: {
                "options": {
                    "chunkTimeout": t,
                    "headerTimeout": t,
                    "requestTimeout": t,
                }
            }
        }
    }


def _opencode_tool_name_for_pre_edit_guard(tool_name: str) -> str:
    normalized = tool_name.lower()
    if normalized == "edit":
        return "Edit"
    if normalized == "write":
        return "Write"
    # apply_patch is gpt-5.x-codex's preferred edit tool; the shared guard
    # parses its patch text for target paths. Pass the name through (lowercased)
    # so check_pre_edit_guard_decision routes it to the apply_patch branch.
    if normalized in {"apply_patch", "patch"}:
        return "apply_patch"
    return tool_name


def _agent_user_kwargs() -> dict[str, str]:
    """``{"user": <name>}`` when CODEEVOLVER_AGENT_USER is set, else ``{}``.

    Spread into agent-process spawns so that, inside the Docker runner, the
    session runs as the unprivileged user and cannot read the root-owned
    mode-600 holdout. Empty everywhere else (local runs unchanged).
    """
    user = agent_user()
    return {"user": user} if user else {}


def _prepare_opencode_config_dir() -> Path:
    config_dir = Path(tempfile.mkdtemp(prefix="codeevolver-opencode-"))
    if agent_user():
        # mkdtemp creates mode 700 regardless of umask; the opencode process
        # runs as the unprivileged agent user and must read its config dir.
        os.chmod(config_dir, 0o755)
    return config_dir


# Parent of the per-session data dirs. Under ~/.codeevolver (the optimizer's
# own tree, root-created in the runner) rather than ~/.local/share, so the
# optimizer can always create it -- and under $HOME rather than /tmp so a
# failed session's `log/opencode.log` survives in the collected artifacts
# instead of dying with the container's tmpfs.
OPENCODE_SESSION_DATA_ROOT = Path(".codeevolver") / "opencode-sessions"


def _prepare_opencode_data_dir(role: AgentRole) -> Path:
    """A fresh OpenCode data/state dir for one session (see `run_inspection`).

    Unlike the config dir this must be WRITABLE by whichever uid runs the
    session: 0777 when an agent user is configured (root cannot chown --
    CAP_CHOWN is dropped with everything else). Never cleaned up, like the
    config dirs; inspection sessions are few and small.
    """
    root = Path.home() / OPENCODE_SESSION_DATA_ROOT
    root.mkdir(parents=True, exist_ok=True)
    role_name = getattr(role, "value", role)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    data_dir = Path(
        tempfile.mkdtemp(prefix=f"{role_name}-{stamp}-{os.getpid()}-", dir=root)
    )
    if agent_user():
        os.chmod(root, 0o755)
        os.chmod(data_dir, 0o777)
    return data_dir


def _data_dir_env(data_dir: Path | None) -> dict[str, str]:
    """The env that points OpenCode's DATA and STATE dirs at `data_dir`.

    OpenCode resolves each as `$XDG_*_HOME || ~/.local/{share,state}`
    (+ `/opencode`); the SQLite db, `log/`, `tool-output/` and `snapshot/`
    live under data, `locks/` under state. The CACHE (`$XDG_CACHE_HOME ||
    ~/.cache`) is deliberately NOT isolated: it holds `models.json` (a
    ~4 MB models.dev fetch) and `bin/` (OpenCode's ripgrep download), so a
    fresh cache per session puts a network fetch into every bootstrap --
    intermittently fatal locally (`UnknownError`, 0-byte log) and blocked
    by the Docker runner's egress fence. Nothing in the cache is uid-sensitive the
    way the db is. Empty when no dir is set, so architect/coding sessions
    are untouched.
    """
    if data_dir is None:
        return {}
    d = str(data_dir)
    return {"XDG_DATA_HOME": d, "XDG_STATE_HOME": d}


def _opencode_small_model() -> str:
    return opencode_small_model()


def _base_opencode_config() -> dict[str, Any]:
    return {
        "$schema": "https://opencode.ai/config.json",
        "small_model": _opencode_small_model(),
    }


def _write_pre_edit_guard_plugin(config_dir: Path) -> Path:
    plugin_dir = config_dir / "plugins"
    plugin_dir.mkdir(parents=True, exist_ok=True)
    plugin_path = plugin_dir / "codeevolver-pre-edit-guard.js"
    plugin_path.write_text(
        textwrap.dedent(
            """
            export const CodeEvolverPreEditGuard = async () => {
              const controlUrl = process.env.CODEEVOLVER_ARCHITECT_MCP_PROXY_URL
              const token = process.env.CODEEVOLVER_ARCHITECT_MCP_PROXY_TOKEN

              async function check(tool, args) {
                const response = await fetch(`${controlUrl}/guard/pre-edit`, {
                  method: "POST",
                  headers: {
                    "Authorization": `Bearer ${token}`,
                    "Content-Type": "application/json",
                  },
                  body: JSON.stringify({ tool, arguments: args || {} }),
                })
                return await response.json()
              }

              return {
                "tool.execute.before": async (input, output) => {
                  // apply_patch ONLY. Its target paths live inside `patchText`,
                  // so OpenCode's permission deny patterns (which match a
                  // filePath) cannot fence them — the Python side does it.
                  // edit/write/bash are deliberately NOT routed here: they are
                  // already fenced by the permission config, there is no
                  // ordering control left to apply, and routing them would put
                  // a control-server round trip on every shell call.
                  if (input.tool !== "apply_patch") return

                  const result = await check(input.tool, output.args || {})
                  if (!result.ok || !result.result || !result.result.allowed) {
                    const reason =
                      result?.result?.reason ||
                      result?.error ||
                      "CodeEvolver pre-edit guard denied this workspace edit"
                    throw new Error(reason)
                  }
                },
              }
            }
            """
        ).lstrip(),
        encoding="utf-8",
    )
    return config_dir


__all__ = [
    "OPENCODE_ARCHITECT_MCP_CALL_TIMEOUT_SECONDS",
    "OPENCODE_ARCHITECT_MCP_PROXY_SCRIPT",
    "OPENCODE_ARCHITECT_MCP_TIMEOUT_MS",
    "OPENCODE_ARCHITECT_TIMEOUT_SECONDS_PER_TURN",
    "OPENCODE_MINIMUM_VERSION",
    "OPENCODE_PROVIDER_STREAM_TIMEOUT_MS",
    "OPENCODE_BOOTSTRAP_COMMANDS",
    "OPENCODE_CODING_TIMEOUT_SECONDS_PER_TURN",
    "OPENCODE_CODE_CHANGE_FINALIZATION_COMMANDS",
    "OPENCODE_INSPECTION_TIMEOUT_SECONDS_PER_TURN",
    "OpenCodeArchitectControlServer",
    "OpenCodeArchitectToolBridge",
    "OpenCodeHarness",
    "OpenCodeHarnessNotImplementedError",
    "OpenCodeSessionSpec",
]
