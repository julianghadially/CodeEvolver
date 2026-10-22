#!/usr/bin/env python3
"""Copyright © 2026 440 Labs LLC

Freeform (Baseline 1) optimizer entry point — exec'd by the environment
layer (local / Docker) after the workspace is set up. Thin shim over the
shared `v3._entry.run_entry`.

Usage:
    PYTHONPATH=/app:$PYTHONPATH python \
        /app/src/engine/optimizer/v3/freeform/optimization_entry.py \
        --config /tmp/optimization_config.json
"""

from __future__ import annotations

from src.engine.optimizer.v3._entry import run_entry


def main() -> None:
    from src.engine.optimizer.v3.freeform.freeform_optimizer import (
        run_freeform_optimization,
    )

    run_entry(run_freeform_optimization, method="freeform")


if __name__ == "__main__":
    main()
