#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Pareto-Evolution optimizer entry point (research baseline #3). Thin shim over
the shared `v3._entry.run_entry`; config/dotenv/result handling is
single-sourced there.

Usage:
    PYTHONPATH=/app:$PYTHONPATH python \
        /app/src/engine/optimizer/v3/pareto_evolution/optimization_entry.py \
        --config /tmp/optimization_config.json
"""

from __future__ import annotations

from src.engine.optimizer.v3._entry import run_entry


def main() -> None:
    from src.engine.optimizer.v3.pareto_evolution.pareto_optimizer import (
        run_pareto_optimization,
    )

    run_entry(run_pareto_optimization, method="pareto_evolution")


if __name__ == "__main__":
    main()
