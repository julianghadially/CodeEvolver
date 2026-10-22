"""Copyright © 2026 440 Labs LLC

Offline checks for `LocalSyntheticEnv`'s path wiring — in particular that it
publishes `architect_traces_dir` (added for the meta-reflection functional
tests) the same way it does `memory_dir`/`traces_dir`: under the per-test
tmpdir, resolving through the production `paths` accessors, and restored on
cleanup. No agent / SDK.
"""

from __future__ import annotations

import os

from src.engine.paths import architect_traces_dir as _resolve_architect_traces_dir
from tests._test_env import LocalSyntheticEnv


def test_local_env_publishes_architect_traces_dir():
    env = LocalSyntheticEnv()
    try:
        # Resolves through the production accessor and lands in the tmpdir,
        # as a sibling of memory/ and traces/ (never the client workspace).
        assert env.architect_traces_dir == _resolve_architect_traces_dir()
        assert env.architect_traces_dir.endswith("architect_traces")
        assert os.path.isdir(env.architect_traces_dir)
        assert env.architect_traces_dir != env.traces_dir
        assert env.architect_traces_dir != env.memory_dir
        # The engine and the test agree on the same dir via the env var.
        assert os.environ["CODEEVOLVER_ARCHITECT_TRACES_DIR"] == env.architect_traces_dir
    finally:
        env.cleanup()
    # cleanup restores the original env (unset in a normal test process).
    assert os.environ.get("CODEEVOLVER_ARCHITECT_TRACES_DIR") is None
