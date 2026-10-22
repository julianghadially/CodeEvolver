#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

ASA optimizer entry point (research Treatment C). Thin shim over the shared
`v3._entry.run_entry`; config/dotenv/result handling is single-sourced there.

Usage:
    PYTHONPATH=/app:$PYTHONPATH python \
        /app/src/engine/optimizer/v3/asa/optimization_entry.py \
        --config /tmp/optimization_config.json
"""

from __future__ import annotations

from src.engine.optimizer.v3._entry import run_entry


def main() -> None:
    from src.engine.optimizer.v3.asa.asa_optimizer import run_asa_optimization

    run_entry(run_asa_optimization, method="asa")


if __name__ == "__main__":
    main()
