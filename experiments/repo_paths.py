"""Copyright © 2026 440 Labs LLC

Locate benchmark client checkouts for local (non-containerized) runs.

Every benchmark repo lives under one parent folder, `CE_REPO_ROOT` (default
`~/Documents/code`), the same variable the orchestrator uses
(`orchestrator/fleet_config.py`). The Docker runner overrides `repo_url` with
the container workspace, so these paths only matter for local runs.
"""

import os
from pathlib import Path

DEFAULT_REPO_ROOT = "~/Documents/code"


def repo_root() -> Path:
    """The parent folder of the benchmark checkouts."""
    return Path(os.environ.get("CE_REPO_ROOT", DEFAULT_REPO_ROOT)).expanduser()


def repo_path(dirname: str) -> str:
    """Absolute path of the checkout `dirname` under `repo_root()`."""
    return str(repo_root() / dirname)
