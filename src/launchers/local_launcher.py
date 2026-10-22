"""Local optimization launcher — runs the optimizer as a subprocess on the host.

Validates Python, picks the entry script, prepares the workspace, builds the
optimization config, execs the entry, and streams its stdout. Notably:

- The workspace is the operator's already-cloned repo (in-place; no clone).
  We look it up in the `repos` collection by repo_url.
- The subprocess runs asynchronously via `asyncio.create_subprocess_exec`;
  a background task tees stdout to `~/.codeevolver/logs/{job_id}.log` and
  parses the `OPTIMIZATION_RESULT:` marker.
- Status / progress get reported through the same `JobCallbacks` HTTP
  callbacks the optimizer already uses — the local API serves them from
  its own `/internal/job/*` endpoints, so no special handling is needed.

Cancellation is cooperative + forceful: the optimizer polls
`/internal/job/{id}/check-cancelled` each iteration, and `cancel(job_id)`
SIGTERMs the registered subprocess via `src/local/process_registry.py`.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

from src.config import settings, validate_python_sandbox_version
from src.db.mongo import get_database
from src.engine.paths import engine_root
from src.local import process_registry
from src.local.local_environment import LocalEnvironment

logger = logging.getLogger(__name__)


_ENTRY_SCRIPT_REL: dict[str, list[str]] = {
    # Relative path under the engine root (auto-detected via
    # `src.engine.paths.engine_root()`) to each method's entry shim.
    "ee": ["src", "engine", "optimizer", "v3", "ee", "optimization_entry.py"],
    "freeform": ["src", "engine", "optimizer", "v3", "freeform", "optimization_entry.py"],
    "pareto_evolution": [
        "src", "engine", "optimizer", "v3", "pareto_evolution", "optimization_entry.py",
    ],
    "asa": ["src", "engine", "optimizer", "v3", "asa", "optimization_entry.py"],
    "simple_loop": ["src", "engine", "optimizer", "simple_loop", "optimization_entry.py"],
}

_CONTROL_CHAR_RE = re.compile(r"[\x00-\x08\x0A-\x1F\x7F]")


class LocalLauncher:
    """Fire-and-forget local subprocess launcher."""

    async def spawn(self, **kwargs) -> str:
        """Resolve workspace, set up venv + git guard, exec the optimizer entry script."""
        job_id: str = kwargs["job_id"]
        repo_url: str = kwargs["repo_url"]
        method: str = kwargs.get("method", "freeform")
        python_version: str = kwargs.get("python_version", "3.11")
        initial_branch: str = kwargs.get("initial_branch", "main")

        validate_python_sandbox_version(python_version)
        if method not in _ENTRY_SCRIPT_REL:
            raise ValueError(
                f"unsupported optimization method: {method!r} "
                f"(expected one of {sorted(_ENTRY_SCRIPT_REL)})"
            )

        workspace_path = await self._resolve_workspace_path(repo_url, job_id)

        env_obj = LocalEnvironment(
            repo_path=workspace_path,
            github_token=kwargs.get("github_token"),
            callback_url=kwargs.get("callback_url") or None,
            jwt_token=kwargs.get("jwt_token") or None,
            job_id=job_id,
        )
        env_obj.start(python_version=python_version, branch=initial_branch)

        try:
            config = self._build_optimization_config(
                workspace=env_obj.workspace_path,
                kwargs=kwargs,
            )
            config_path = self._write_config(env_obj.workspace_path, config)
            entry_script = self._engine_script(method)
            subprocess_env = self._build_subprocess_env(env_obj)
            log_path = self._log_path(job_id)

            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(entry_script),
                "--config", str(config_path),
                cwd=env_obj.workspace_path,
                env=subprocess_env,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            process_registry.register(job_id, proc)
            # Write a pidfile so an operator can `kill -TERM $(cat …)` the run
            # from the shell — the in-memory registry is lost on an API restart,
            # which orphans the optimizer (reparented to PID 1).
            self._write_pid(job_id, proc.pid)

            # Stream stdout in the background; the API call returns immediately.
            asyncio.create_task(
                self._stream_output(job_id, proc, log_path, env_obj),
                name=f"local-optimizer-stream-{job_id}",
            )
            call_id = f"local-{uuid.uuid4().hex[:12]}"
            logger.info("Local optimizer spawned (pid=%s, call_id=%s)", proc.pid, call_id)
            return call_id
        except Exception:
            env_obj.stop()
            raise

    async def cancel(self, job_id: str) -> bool:
        return await process_registry.terminate(job_id)

    # ---------------- helpers ----------------

    async def _resolve_workspace_path(self, repo_url: str, job_id: str) -> str:
        db = get_database()
        repo = await db["repos"].find_one({"repo_url": repo_url})
        if not repo or not repo.get("workspace_path"):
            raise RuntimeError(
                f"No local repo registered for repo_url={repo_url!r}. "
                f"Use POST /local/connect-repo to register a workspace_path "
                f"before /optimize (job_id={job_id})."
            )
        return repo["workspace_path"]

    def _build_optimization_config(self, workspace: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        """Build the config-JSON dict from `kwargs`, keyed off the single
        `OptimizationConfig` schema. Local-specific: coding_harness defaults to
        `opencode` and the timeout comes from `settings.local_job_timeout`."""
        from src.engine.optimizer.v3.optimization_config import OptimizationConfig
        from src.engine.services.job_context import JobContext

        return OptimizationConfig(
            workspace=workspace,
            program=kwargs["program"],
            metric=kwargs["metric"],
            trainset_json=kwargs.get("trainset_json"),
            trainset_path=kwargs.get("trainset_path"),
            testset_json=kwargs.get("testset_json"),
            testset_path=kwargs.get("testset_path"),
            reflection_lm=kwargs.get("reflection_lm", "anthropic/claude-opus-4-7"),
            coding_harness=kwargs.get("coding_harness") or "opencode",
            coding_harness_model=kwargs.get("coding_harness_model"),
            num_threads=kwargs.get("num_threads", 1),
            seed=kwargs.get("seed", 0),
            additional_instructions=kwargs.get("additional_instructions"),
            additional_deny_paths=kwargs.get("additional_deny_paths"),
            comparison_set_size=kwargs.get("comparison_set_size", settings.comparison_set_size),
            comparison_credits_initial=kwargs.get("comparison_credits_initial", 1),
            comparison_credits_per_iteration=kwargs.get("comparison_credits_per_iteration", 1),
            debug_max_iterations=kwargs.get("debug_max_iterations"),
            asa_behavioral_contribution_constant=kwargs.get("asa_behavioral_contribution_constant", 0.10),
            asa_behavioral_contribution_threshold_cap=kwargs.get("asa_behavioral_contribution_threshold_cap", 0.05),
            asa_primary_stale_landed_iterations=kwargs.get("asa_primary_stale_landed_iterations", 4),
            max_metric_calls=kwargs.get("max_metric_calls", 6000),
            job_timeout_seconds=settings.local_job_timeout,
            subsample_eval_timeout=kwargs.get("subsample_eval_timeout", 7200),
            comparison_eval_timeout=kwargs.get("comparison_eval_timeout", 14400),
            test_samples=kwargs.get("test_samples", 3),
            seed_test_samples=kwargs.get("seed_test_samples", 1),
            variance_estimate_samples=kwargs.get("variance_estimate_samples", 5),
            subsample_size=kwargs.get("subsample_size", 20),
            metric_names=kwargs.get("metric_names"),
            primary_metric=kwargs.get("primary_metric"),
            max_score=kwargs.get("max_score", 1.0),
            input_keys=kwargs.get("input_keys"),
            enable_meta_reflection=kwargs.get("enable_meta_reflection", False),
            enable_run_integrity_agent=kwargs.get(
                "enable_run_integrity_agent", False
            ),
            enable_clean_slate_inspector=kwargs.get(
                "enable_clean_slate_inspector", False
            ),
            job=JobContext(
                repo_url=kwargs["repo_url"],
                github_token=kwargs.get("github_token"),
                callback_url=kwargs.get("callback_url") or None,
                jwt_token=kwargs.get("jwt_token") or None,
                job_id=kwargs["job_id"],
                initial_branch=kwargs.get("initial_branch", "main"),
            ),
        ).to_dict()

    def _write_config(self, workspace: str, config: dict[str, Any]) -> Path:
        """Write the optimization config inside the workspace.

        Stored at `<workspace>/.codeevolver/optimization_config.json` so it
        does not pollute the repo root and is covered by the `.codeevolver/`
        gitignore (or operators can choose to inspect it).
        """
        dir_path = Path(workspace) / ".codeevolver"
        dir_path.mkdir(parents=True, exist_ok=True)
        config_path = dir_path / "optimization_config.json"
        config_path.write_text(json.dumps(config))
        return config_path

    def _engine_script(self, method: str) -> Path:
        return Path(engine_root()).joinpath(*_ENTRY_SCRIPT_REL[method])

    def _build_subprocess_env(self, env_obj: LocalEnvironment) -> dict[str, str]:
        """Compose the env vars handed to the optimizer subprocess.

        All engine paths (traces, memory, venv, engine source) are resolved
        inside the subprocess by ``src.engine.paths`` based on
        ``SANDBOX_MODE`` — which is inherited via ``os.environ.copy()`` from
        the API process the operator launched with ``SANDBOX_MODE=local``.

        The launcher only sets ``PYTHONPATH`` (required for the FIRST
        subprocess's Python startup, before any module reads paths.py) and
        passes through ``ANTHROPIC_API_KEY``. The pre-created
        ``~/.codeevolver/{memory,traces}`` directories are a courtesy —
        the engine would create them on first write either way.
        """
        env = os.environ.copy()
        # PYTHONPATH is set explicitly because Python initializes sys.path
        # at startup, before any user code can call paths.engine_root().
        env["PYTHONPATH"] = engine_root() + os.pathsep + env.get("PYTHONPATH", "")
        # If the caller customized LocalEnvironment.venv_dirname, propagate
        # it; otherwise paths.repo_venv_dirname() picks the SANDBOX_MODE
        # default and we don't need to set anything.
        if env_obj._venv_dirname != "env":
            # REPO_VENV_DIR: literal-string write of a cross-process env var
            # (registry in src/engine/paths.py::CROSS_PROCESS_ENV_VARS). Read
            # by paths.repo_venv_dirname() inside the optimizer subprocess.
            env["REPO_VENV_DIR"] = env_obj._venv_dirname
        # Mkdir-p the conventional local-deploy state dirs so first writes
        # don't race on parent-dir creation. Defaults resolved by
        # src.engine.paths use these same paths.
        (Path.home() / ".codeevolver" / "memory").mkdir(parents=True, exist_ok=True)
        (Path.home() / ".codeevolver" / "traces").mkdir(parents=True, exist_ok=True)
        # Anthropic API key passthrough (Claude Agent SDK needs this).
        if settings.anthropic_api_key:
            env.setdefault("ANTHROPIC_API_KEY", settings.anthropic_api_key)
        return env

    def _log_path(self, job_id: str) -> Path:
        log_dir = settings.local_log_root
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"{job_id}.log"

    def _pid_path(self, job_id: str) -> Path:
        return settings.local_log_root / f"{job_id}.pid"

    def _write_pid(self, job_id: str, pid: int) -> None:
        path = self._pid_path(job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{pid}\n")

    async def _stream_output(
        self,
        job_id: str,
        proc: asyncio.subprocess.Process,
        log_path: Path,
        env_obj: LocalEnvironment,
    ) -> None:
        """Tee subprocess stdout to a log file and parse the OPTIMIZATION_RESULT marker.

        Reads in fixed-size byte chunks and splits on newlines ourselves
        rather than using asyncio's line iterator (`async for ... in
        proc.stdout`). The line iterator inherits the StreamReader's 64 KiB
        separator limit and raises `LimitOverrunError` ("Separator is not
        found, and chunk exceed the limit") on any single line longer than
        that — which the final `OPTIMIZATION_RESULT:{json}` marker routinely
        is, since it serializes the full `history` dict (every iteration's
        change_request + 200-600 word subsample_analysis) onto one line. The
        chunked reader has no per-line limit, so arbitrarily long markers
        stream fine. `StreamReader.read(n)` is unaffected by the separator
        limit.
        """
        result_payload: dict | None = None
        buffer_parts: list[str] = []

        def _handle_line(line: str, log_file) -> None:
            nonlocal result_payload
            log_file.write(line + "\n")
            log_file.flush()
            buffer_parts.append(line)
            if line.startswith("OPTIMIZATION_RESULT:"):
                try:
                    result_payload = json.loads(line[len("OPTIMIZATION_RESULT:"):])
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Failed to parse OPTIMIZATION_RESULT for %s: %s",
                        job_id, exc,
                    )

        try:
            with log_path.open("w") as log_file:
                assert proc.stdout is not None
                pending = b""
                while True:
                    chunk = await proc.stdout.read(65536)
                    if not chunk:
                        break
                    pending += chunk
                    while True:
                        nl = pending.find(b"\n")
                        if nl < 0:
                            break
                        raw_line, pending = pending[:nl], pending[nl + 1:]
                        _handle_line(
                            raw_line.decode("utf-8", errors="replace").rstrip("\r"),
                            log_file,
                        )
                # Flush any trailing partial line (output not newline-terminated).
                if pending:
                    _handle_line(
                        pending.decode("utf-8", errors="replace").rstrip("\r"),
                        log_file,
                    )
            await proc.wait()

            if result_payload is None:
                # Fallback: scan the accumulated buffer.
                buffer = "\n".join(buffer_parts)
                idx = buffer.rfind("OPTIMIZATION_RESULT:")
                if idx >= 0:
                    payload = buffer[idx + len("OPTIMIZATION_RESULT:"):].strip()
                    payload = _CONTROL_CHAR_RE.sub("", payload)
                    try:
                        result_payload = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        logger.warning(
                            "Buffer fallback parse failed for %s: %s",
                            job_id, exc,
                        )

            if result_payload is None and proc.returncode != 0:
                logger.warning(
                    "Local optimizer %s exited with code %s; no OPTIMIZATION_RESULT marker.",
                    job_id, proc.returncode,
                )

            # Final status comes through JobCallbacks via the API's
            # /internal/job/{id}/status endpoint — the optimizer's own
            # `set_completed` / `set_failed` calls have already written it.
            # We log the result locally for diagnostics.
            if result_payload is not None:
                logger.info(
                    "Local optimizer %s finished: success=%s",
                    job_id, result_payload.get("success"),
                )
        except Exception as exc:  # pragma: no cover — defensive
            logger.exception("Error streaming optimizer output for %s: %s", job_id, exc)
        finally:
            process_registry.unregister(job_id)
            self._pid_path(job_id).unlink(missing_ok=True)
            env_obj.stop()
