#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Shared entry-point body for every v3 optimizer.

Each method ships a thin `optimization_entry.py` shim that calls
`run_entry(<run_*_optimization>)`. This module owns the parts that are
identical across EE / Freeform / Pareto-Evolution: argparse, `.env` loading,
config → kwargs mapping, the `IS_SANDBOX` shim, the
`OPTIMIZATION_RESULT:{json}` stdout marker, and structured error reporting.

Keeping this single-sourced means the production EE entry and the two
research-baseline entries can never drift in how they map the optimization
config — only *which* `run_*_optimization` they invoke differs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from typing import Callable


def _load_dotenv(env_path: str) -> None:
    """Parse a simple .env file and set vars in os.environ."""
    if not os.path.exists(env_path):
        return
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue
            key, _, value = line.partition("=")
            # Accept `export FOO='bar'` / `FOO=bar` / `FOO="bar"` forms —
            # strip the leading `export ` and any surrounding quotes.
            key = key.strip()
            if key.startswith("export "):
                key = key[len("export "):].strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def run_entry(runner: Callable[..., dict], *, method: str) -> None:
    """Run an optimizer entry point.

    Args:
        runner: the method's `run_*_optimization` callable. It must accept
            the same keyword contract as
            `v3.optimizer_loop.run_optimization` (minus `strategy`, which
            the wrapper injects).
        method: method label for argparse help / diagnostics only.
    """
    parser = argparse.ArgumentParser(
        description=f"Environment-agnostic {method} optimizer"
    )
    parser.add_argument(
        "--config", required=True,
        help="Path to JSON config file with optimization parameters",
    )
    args = parser.parse_args()

    from src.engine.services.asyncio_utils import install_pidfd_child_watcher
    from src.engine.services.logger import configure_logging

    configure_logging()
    install_pidfd_child_watcher()

    with open(args.config) as f:
        config = json.load(f)

    workspace = config.get("workspace", "/workspace")

    # Load env vars from .env (won't overwrite existing vars)
    _load_dotenv(os.path.join(workspace, ".env"))

    # The runner may run as root. Claude Code CLI refuses
    # `--dangerously-skip-permissions` as root unless IS_SANDBOX=1 is set.
    # The IterationArchitect spawns the Claude Agent SDK in-process, and the
    # coding/reflection subprocesses spawn theirs too — all inherit this env.
    os.environ.setdefault("IS_SANDBOX", "1")

    # Scope traces/memory/runtime_tests dirs to this job. Read by
    # src.engine.paths._job_subdir(); inherited by every subprocess via
    # subprocess_service._build_env's os.environ.copy(), so the agent's
    # system prompt (built with paths.memory_dir() / paths.traces_dir())
    # and every file read/write resolve to the same {job_id} subdirectory.
    job_id = config.get("job_id")
    if job_id:
        os.environ["CODEEVOLVER_JOB_ID"] = str(job_id)

    from src.engine.optimizer.v3.optimization_config import OptimizationConfig

    # OpenCode's read/grep tools truncate every line at 2000 chars, so trace
    # files written for an OpenCode run chunk long strings (see
    # telemetry/trace_json.py). Read by paths.trace_max_line_chars() in the
    # evaluator subprocess and the architect-trace writer.
    from src.engine.optimizer.v3.coding_harness_config import (
        normalize_coding_harness_config,
    )
    from src.engine.paths import OPENCODE_TRACE_MAX_LINE_CHARS

    harness = normalize_coding_harness_config(
        coding_harness=config.get("coding_harness"),
        coding_harness_model=config.get("coding_harness_model"),
    ).coding_harness
    if harness == "opencode":
        os.environ["CODEEVOLVER_TRACE_MAX_LINE_CHARS"] = str(
            OPENCODE_TRACE_MAX_LINE_CHARS
        )

    try:
        result = runner(config=OptimizationConfig.from_dict(config))
        # Compact separators so the marker stays on a single line even when
        # the payload contains large string fields with embedded newlines
        # (e.g. change_request or subsample_analysis). The stdout-marker
        # parser in the launcher scans line-by-line.
        print(
            "OPTIMIZATION_RESULT:"
            + json.dumps(result, separators=(",", ":")),
            flush=True,
        )

    except Exception as e:
        # Best-effort status callback so a seed-step crash surfaces in the
        # UI instead of hanging as "running" until the job timeout. Guarded
        # so a broken callback never masks the original error.
        try:
            cb = OptimizationConfig.from_dict(config).job.make_callbacks()
            cb.set_failed(f"{type(e).__name__}: {e}")
        except Exception:
            pass

        error = {
            "ok": False,
            "error": str(e),
            "error_type": type(e).__name__,
            "traceback": traceback.format_exc(),
        }
        print(
            "OPTIMIZATION_RESULT:"
            + json.dumps(error, separators=(",", ":")),
            flush=True,
        )
        sys.exit(1)


__all__ = ["run_entry"]
