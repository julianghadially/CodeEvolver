#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

EE (exploit and explore) optimizer entry point — exec'd by the environment
layer (local subprocess / Docker runner) after the workspace is set up. Thin shim over
the shared `v3._entry.run_entry`; all config/dotenv/result handling is
single-sourced there so the production EE path and the research baselines
never drift.

Usage:
    PYTHONPATH=/app:$PYTHONPATH python \
        /app/src/engine/optimizer/v3/ee/optimization_entry.py \
        --config /tmp/optimization_config.json
"""

from __future__ import annotations

from src.engine.optimizer.v3._entry import run_entry


def main() -> None:
    from src.engine.optimizer.v3.ee.ee_optimizer import run_ee_optimization

    run_entry(run_ee_optimization, method="ee")


if __name__ == "__main__":
    main()
