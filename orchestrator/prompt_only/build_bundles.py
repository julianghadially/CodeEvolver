#!/usr/bin/env python3
# Copyright © 2026 440 Labs LLC
"""Build the prompt-only ablation's seed bundles WITHOUT touching the main ones.

Reuses `build.build_bundle` (same squash-to-one-orphan-commit contamination
control), pointed at each client repo's `prompt-only` branch and redirected to
`runner/bundles-po/`, so `runner/bundles/` (the main experiment's seeds, baked
into every runner image) is never overwritten. No image is built: orchestrator/launch.sh
copies these to the hosts and matrix.po*.yaml bind-mounts them over
`/opt/seed/<bench>.bundle` in the unchanged evo-redo images.

    python orchestrator/prompt_only/build_bundles.py            # hover + phantomwiki
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import build  # noqa: E402
from fleet_config import load_catalog  # noqa: E402

BRANCH = "prompt-only"
BENCHES = ("hover", "phantomwiki")
OUT_DIR = build.REPO_ROOT / "runner" / "bundles-po"


def main() -> None:
    catalog = load_catalog()
    build.BUNDLE_DIR = OUT_DIR  # build_bundle reads the module global
    for slug in BENCHES:
        dest = build.build_bundle(catalog[slug], BRANCH)
        digest = hashlib.sha256(dest.read_bytes()).hexdigest()
        print(f"  {slug}: sha256 {digest}")


if __name__ == "__main__":
    main()
